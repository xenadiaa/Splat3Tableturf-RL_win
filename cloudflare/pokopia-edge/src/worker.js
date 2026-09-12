const JSON_HEADERS = { "content-type": "application/json; charset=utf-8" };
const DAY_RE = /^\d{4}-\d{2}-\d{2}$/;
const CODE_RE = /^[0-9A-HJ-NP-Y]{6}$/;
const MAX_LIVE_BYTES = 1_000_000;
const MAX_SCREENSHOT_BYTES = 8_000_000;
const MAX_PLAYERS_PER_DAY = 3000;

function securityHeaders(extra = {}) {
  return {
    "x-content-type-options": "nosniff",
    "x-frame-options": "DENY",
    "referrer-policy": "no-referrer",
    "permissions-policy": "camera=(), microphone=(), geolocation=()",
    "content-security-policy": "default-src 'self'; img-src 'self' data:; style-src 'self'; script-src 'self'; connect-src 'self'; base-uri 'none'; frame-ancestors 'none'",
    ...extra,
  };
}

function json(payload, status = 200, extra = {}) {
  return new Response(JSON.stringify(payload), {
    status,
    headers: securityHeaders({ ...JSON_HEADERS, "cache-control": "no-store", ...extra }),
  });
}

function clampCount(value) {
  const parsed = Number.parseInt(value, 10);
  return Number.isFinite(parsed) ? Math.max(0, parsed) : 0;
}

function validDay(value) {
  if (!DAY_RE.test(value || "")) return false;
  const parsed = new Date(`${value}T00:00:00Z`);
  return !Number.isNaN(parsed.getTime()) && parsed.toISOString().slice(0, 10) === value;
}

function dayAdd(day, count) {
  const value = new Date(`${day}T00:00:00Z`);
  value.setUTCDate(value.getUTCDate() + count);
  return value.toISOString().slice(0, 10);
}

function beijingBusinessDay() {
  const shifted = new Date(Date.now() + 3 * 60 * 60 * 1000);
  return shifted.toISOString().slice(0, 10);
}

function fridayStart(day) {
  const value = new Date(`${day}T00:00:00Z`);
  const mondayBased = (value.getUTCDay() + 6) % 7;
  return dayAdd(day, -((mondayBased - 4 + 7) % 7));
}

function monthNext(day) {
  const [year, month] = day.split("-").map(Number);
  const next = month === 12 ? [year + 1, 1] : [year, month + 1];
  return `${String(next[0]).padStart(4, "0")}-${String(next[1]).padStart(2, "0")}-01`;
}

function queryRange(url) {
  const explicitStart = url.searchParams.get("start") || "";
  const explicitEnd = url.searchParams.get("end") || "";
  const anchor = url.searchParams.get("anchor") || beijingBusinessDay();
  if ((explicitStart && !validDay(explicitStart)) || (explicitEnd && !validDay(explicitEnd))) {
    throw new Error("日期必须是 YYYY-MM-DD。");
  }
  if (explicitStart || explicitEnd) {
    const start = explicitStart || "0000-01-01";
    const end = explicitEnd || "9999-12-31";
    if (end < start) throw new Error("结束日期不能早于开始日期。");
    return { start, end, label: "自定义范围" };
  }
  if (!validDay(anchor)) throw new Error("基准日期必须是 YYYY-MM-DD。");
  const period = (url.searchParams.get("period") || "all").toLowerCase();
  if (period === "day") return { start: anchor, end: anchor, label: "日" };
  if (period === "week") {
    const start = fridayStart(anchor);
    return { start, end: dayAdd(start, 6), label: "周" };
  }
  if (period === "month") {
    const start = `${anchor.slice(0, 7)}-01`;
    return { start, end: dayAdd(monthNext(start), -1), label: "月" };
  }
  if (period === "year") {
    return { start: `${anchor.slice(0, 4)}-01-01`, end: `${anchor.slice(0, 4)}-12-31`, label: "年" };
  }
  if (period !== "all") throw new Error("period只支持 day/week/month/year/all。");
  return { start: "0000-01-01", end: "9999-12-31", label: "全部" };
}

async function authorized(request, env) {
  const expected = String(env.UPLOAD_TOKEN || "");
  const supplied = request.headers.get("authorization") || "";
  if (!expected || !supplied.startsWith("Bearer ")) return false;
  const encoder = new TextEncoder();
  const [a, b] = await Promise.all([
    crypto.subtle.digest("SHA-256", encoder.encode(supplied.slice(7))),
    crypto.subtle.digest("SHA-256", encoder.encode(expected)),
  ]);
  const left = new Uint8Array(a);
  const right = new Uint8Array(b);
  let difference = left.length ^ right.length;
  for (let index = 0; index < Math.min(left.length, right.length); index += 1) {
    difference |= left[index] ^ right[index];
  }
  return difference === 0;
}

async function readJsonLimited(request, limit) {
  const declared = Number(request.headers.get("content-length") || 0);
  if (declared > limit) throw new Error("请求内容过大。");
  const bytes = new Uint8Array(await request.arrayBuffer());
  if (bytes.byteLength > limit) throw new Error("请求内容过大。");
  return JSON.parse(new TextDecoder().decode(bytes));
}

async function rateLimit(request, env, bucket) {
  const client = (request.headers.get("x-pokopia-client") || "anonymous").slice(0, 96);
  const ip = request.headers.get("cf-connecting-ip") || "local";
  const digest = await crypto.subtle.digest(
    "SHA-256",
    new TextEncoder().encode(`${ip}\u0000${client}\u0000${bucket}`),
  );
  const key = [...new Uint8Array(digest)].map((byte) => byte.toString(16).padStart(2, "0")).join("");
  const stub = env.RATE_LIMITER.get(env.RATE_LIMITER.idFromName("global-v1"));
  const response = await stub.fetch("https://rate.internal/check", {
    headers: { "x-rate-key": key },
  });
  return response.status === 204;
}

export class ClientRateLimiter {
  constructor(ctx) {
    this.lastAt = new Map();
    this.requestsSinceSweep = 0;
  }

  async fetch(request) {
    const key = request.headers.get("x-rate-key") || "anonymous";
    const now = Date.now();
    const previous = this.lastAt.get(key) || 0;
    if (now - previous < 1000) {
      return new Response(null, { status: 429, headers: { "retry-after": "1" } });
    }
    this.lastAt.set(key, now);
    this.requestsSinceSweep += 1;
    if (this.requestsSinceSweep >= 500) {
      this.requestsSinceSweep = 0;
      for (const [candidate, lastSeen] of this.lastAt) {
        if (now - lastSeen > 60_000) this.lastAt.delete(candidate);
      }
    }
    return new Response(null, { status: 204 });
  }
}

export class LiveState {
  constructor(ctx) {
    this.ctx = ctx;
  }

  async fetch(request) {
    if ((request.headers.get("upgrade") || "").toLowerCase() === "websocket") {
      const pair = new WebSocketPair();
      const [client, server] = Object.values(pair);
      this.ctx.acceptWebSocket(server);
      const current = await this.ctx.storage.get("state");
      if (current) server.send(JSON.stringify(current));
      return new Response(null, { status: 101, webSocket: client });
    }
    if (request.method === "PUT") {
      const payload = await request.json();
      await this.ctx.storage.put("state", payload);
      const message = JSON.stringify(payload);
      for (const socket of this.ctx.getWebSockets()) {
        try {
          socket.send(message);
        } catch {
          try { socket.close(1011, "send failed"); } catch { /* already closed */ }
        }
      }
      return new Response(null, { status: 204 });
    }
    const payload = await this.ctx.storage.get("state");
    return json(payload || {});
  }
}

async function liveStub(env) {
  return env.LIVE.get(env.LIVE.idFromName("pokopia-live-v1"));
}

async function streamLive(request, env) {
  if ((request.headers.get("upgrade") || "").toLowerCase() !== "websocket") {
    return json({ error: "需要WebSocket升级。" }, 426);
  }
  const stub = await liveStub(env);
  return stub.fetch(request);
}

async function publishLive(request, env) {
  if (!(await authorized(request, env))) return json({ error: "未授权" }, 401);
  let payload;
  try {
    payload = await readJsonLimited(request, MAX_LIVE_BYTES);
  } catch (error) {
    return json({ error: String(error.message || error) }, 400);
  }
  if (!payload || typeof payload !== "object" || Array.isArray(payload)) {
    return json({ error: "实时状态必须是JSON对象。" }, 400);
  }
  const code = String(payload.code || "").toUpperCase();
  if (code && !CODE_RE.test(code)) return json({ error: "CODE格式无效。" }, 400);
  payload.code = code;
  payload.edge_received_at_epoch = Date.now() / 1000;
  payload.offline_after_seconds = Math.max(10, clampCount(env.OFFLINE_AFTER_SECONDS || 30));
  payload.screenshot_url = code && payload.screenshot_url ? "/media/current-code.png" : null;
  const stub = await liveStub(env);
  await stub.fetch("https://live.internal/state", {
    method: "PUT",
    headers: JSON_HEADERS,
    body: JSON.stringify(payload),
  });
  return json({ ok: true, received_at_epoch: payload.edge_received_at_epoch });
}

function normalizePlayer(row) {
  const name = String(row?.name || "").trim().slice(0, 128);
  return {
    name,
    nameSearch: name.toLocaleLowerCase("und"),
    visits: clampCount(row?.visits),
    tasks: clampCount(row?.tasks),
    returned: clampCount(row?.returned_before_arrival),
    closed: clampCount(row?.room_closed_before_arrival),
  };
}

async function publishDay(request, env) {
  if (!(await authorized(request, env))) return json({ error: "未授权" }, 401);
  let payload;
  try {
    payload = await readJsonLimited(request, MAX_LIVE_BYTES);
  } catch (error) {
    return json({ error: String(error.message || error) }, 400);
  }
  const day = String(payload?.date || "");
  const players = Array.isArray(payload?.players) ? payload.players : [];
  if (!validDay(day)) return json({ error: "业务日期无效。" }, 400);
  if (players.length > MAX_PLAYERS_PER_DAY) return json({ error: "单日玩家记录过多。" }, 400);
  const normalized = players.map(normalizePlayer).filter((row) => row.name);
  const now = new Date().toISOString();
  await env.DB.batch([
    env.DB.prepare(
      `INSERT INTO daily_summary(day, rounds, successful_visits, failed_visits, task_participations, updated_at)
       VALUES(?, ?, ?, ?, ?, ?)
       ON CONFLICT(day) DO UPDATE SET rounds=excluded.rounds,
       successful_visits=excluded.successful_visits, failed_visits=excluded.failed_visits,
       task_participations=excluded.task_participations, updated_at=excluded.updated_at`,
    ).bind(day, clampCount(payload.rounds), clampCount(payload.successful_visits), clampCount(payload.failed_visits), clampCount(payload.task_participations), now),
    env.DB.prepare("DELETE FROM player_daily WHERE day = ?").bind(day),
  ]);
  for (let offset = 0; offset < normalized.length; offset += 80) {
    const statements = normalized.slice(offset, offset + 80).map((row) => env.DB.prepare(
      `INSERT INTO player_daily(day, name, name_search, visits, tasks, returned_before_arrival, room_closed_before_arrival)
       VALUES(?, ?, ?, ?, ?, ?, ?)`,
    ).bind(day, row.name, row.nameSearch, row.visits, row.tasks, row.returned, row.closed));
    await env.DB.batch(statements);
  }
  return json({ ok: true, date: day, players: normalized.length });
}

async function publishScreenshot(request, env) {
  if (!(await authorized(request, env))) return json({ error: "未授权" }, 401);
  if (request.method === "DELETE") {
    await env.MEDIA.delete("current-code.png");
    return json({ ok: true, deleted: true });
  }
  const declared = Number(request.headers.get("content-length") || 0);
  if (declared > MAX_SCREENSHOT_BYTES) return json({ error: "截图超过8MB。" }, 413);
  const bytes = await request.arrayBuffer();
  if (bytes.byteLength > MAX_SCREENSHOT_BYTES) return json({ error: "截图超过8MB。" }, 413);
  const signature = new Uint8Array(bytes.slice(0, 8));
  const png = [137, 80, 78, 71, 13, 10, 26, 10];
  if (signature.length !== 8 || signature.some((value, index) => value !== png[index])) {
    return json({ error: "只接受PNG截图。" }, 415);
  }
  await env.MEDIA.put("current-code.png", bytes, {
    httpMetadata: { contentType: "image/png", cacheControl: "no-store" },
    customMetadata: {
      code: String(request.headers.get("x-pokopia-code") || "").slice(0, 6),
      revision: String(request.headers.get("x-pokopia-revision") || "0").slice(0, 24),
      uploadedAt: new Date().toISOString(),
    },
  });
  return json({ ok: true, bytes: bytes.byteLength });
}

async function getLive(request, env, ctx) {
  const cacheURL = new URL(request.url);
  cacheURL.search = "";
  const cacheKey = new Request(cacheURL.toString(), { method: "GET" });
  const cached = await caches.default.match(cacheKey);
  if (cached) return cached;
  const stub = await liveStub(env);
  const storedResponse = await stub.fetch("https://live.internal/state");
  const state = await storedResponse.json();
  const received = Number(state.edge_received_at_epoch || 0);
  const offlineAfter = Math.max(10, clampCount(env.OFFLINE_AFTER_SECONDS || 30));
  const stale = received ? Math.max(0, Date.now() / 1000 - received) : null;
  if (stale === null || stale > offlineAfter) {
    state.status = { key: "offline", label: "离线", tone: "gray" };
    state.phase = { key: "offline", label: "Windows状态心跳已中断，当前内容可能过期" };
  }
  state.stale_seconds = stale;
  state.offline_after_seconds = offlineAfter;
  const publicResponse = json(state, 200, { "cache-control": "public, max-age=0, s-maxage=1" });
  ctx.waitUntil(caches.default.put(cacheKey, publicResponse.clone()));
  return publicResponse;
}

async function getCurrentScreenshot(env) {
  const object = await env.MEDIA.get("current-code.png");
  if (!object) return new Response("Not found", { status: 404, headers: securityHeaders() });
  const headers = new Headers(securityHeaders({
    "content-type": object.httpMetadata?.contentType || "image/png",
    "cache-control": "no-store, max-age=0",
    etag: object.httpEtag,
  }));
  return new Response(object.body, { headers });
}

async function queryHistory(request, env, topOnly) {
  if (!(await rateLimit(request, env, topOnly ? "rankings" : "query"))) {
    return json({ error: "请求过快，每秒最多一次。" }, 429, { "retry-after": "1" });
  }
  const url = new URL(request.url);
  let range;
  try {
    range = queryRange(url);
  } catch (error) {
    return json({ error: String(error.message || error) }, 400);
  }
  const needle = String(url.searchParams.get("player") || "").trim().toLocaleLowerCase("und").slice(0, 128);
  const requestedLimit = topOnly ? 10 : Math.min(1000, Math.max(1, clampCount(url.searchParams.get("limit") || 10)));
  const condition = "day BETWEEN ? AND ? AND (? = '' OR instr(name_search, ?) > 0)";
  const args = [range.start, range.end, needle, needle];
  const [summariesResult, filteredDailyResult, visitsResult, tasksResult, failuresResult, namesResult] = await Promise.all([
    env.DB.prepare(
      "SELECT day, rounds, successful_visits, failed_visits, task_participations FROM daily_summary WHERE day BETWEEN ? AND ? ORDER BY day",
    ).bind(range.start, range.end).all(),
    env.DB.prepare(
      `SELECT day, SUM(visits) successful_visits, SUM(tasks) task_participations,
       SUM(returned_before_arrival + room_closed_before_arrival) failed_visits
       FROM player_daily WHERE ${condition} GROUP BY day`,
    ).bind(...args).all(),
    env.DB.prepare(
      `SELECT name, SUM(visits) count FROM player_daily WHERE ${condition}
       GROUP BY name HAVING count > 0 ORDER BY count DESC, name LIMIT ?`,
    ).bind(...args, requestedLimit).all(),
    env.DB.prepare(
      `SELECT name, SUM(tasks) count FROM player_daily WHERE ${condition}
       GROUP BY name HAVING count > 0 ORDER BY count DESC, name LIMIT ?`,
    ).bind(...args, requestedLimit).all(),
    env.DB.prepare(
      `SELECT name, SUM(returned_before_arrival + room_closed_before_arrival) count,
       SUM(returned_before_arrival) returned_before_arrival,
       SUM(room_closed_before_arrival) room_closed_before_arrival
       FROM player_daily WHERE ${condition}
       GROUP BY name HAVING count > 0 ORDER BY count DESC, name LIMIT ?`,
    ).bind(...args, requestedLimit).all(),
    env.DB.prepare(
      `SELECT DISTINCT name FROM player_daily WHERE ${condition} ORDER BY name LIMIT 200`,
    ).bind(...args).all(),
  ]);
  const filteredByDay = new Map((filteredDailyResult.results || []).map((row) => [row.day, row]));
  const daily = (summariesResult.results || []).map((row) => {
    const filtered = filteredByDay.get(row.day) || {};
    return {
      date: row.day,
      rounds: clampCount(row.rounds),
      successful_visits: clampCount(needle ? filtered.successful_visits : row.successful_visits),
      failed_visits: clampCount(needle ? filtered.failed_visits : row.failed_visits),
      task_participations: clampCount(needle ? filtered.task_participations : row.task_participations),
    };
  });
  const ranked = (rows) => (rows || []).map((row, index) => ({ rank: index + 1, ...row, count: clampCount(row.count) }));
  return json({
    period: range.label,
    start: range.start === "0000-01-01" ? null : range.start,
    end_inclusive: range.end === "9999-12-31" ? null : range.end,
    player_query: url.searchParams.get("player") || "",
    matched_players: (namesResult.results || []).map((row) => row.name),
    summary: {
      successful_visits: daily.reduce((sum, row) => sum + row.successful_visits, 0),
      failed_visits: daily.reduce((sum, row) => sum + row.failed_visits, 0),
      task_participations: daily.reduce((sum, row) => sum + row.task_participations, 0),
    },
    rankings: {
      visits: ranked(visitsResult.results),
      tasks: ranked(tasksResult.results),
      failures: ranked(failuresResult.results).map((row) => ({
        ...row,
        returned_before_arrival: clampCount(row.returned_before_arrival),
        room_closed_before_arrival: clampCount(row.room_closed_before_arrival),
      })),
    },
    daily,
    generated_at: new Date().toISOString(),
  });
}

export default {
  async fetch(request, env, ctx) {
    const url = new URL(request.url);
    if (url.hostname === "rabi.date") {
      return Response.redirect(`https://${env.CANONICAL_HOST || "stamp.rabi.date"}${url.pathname}${url.search}`, 308);
    }
    if (request.method === "POST" && url.pathname === "/api/publish/live") return publishLive(request, env);
    if (request.method === "POST" && url.pathname === "/api/publish/day") return publishDay(request, env);
    if (["POST", "DELETE"].includes(request.method) && url.pathname === "/api/publish/screenshot") return publishScreenshot(request, env);
    if (request.method !== "GET" && request.method !== "HEAD") return json({ error: "Method not allowed" }, 405);
    if (url.pathname === "/api/stream") return streamLive(request, env);
    if (url.pathname === "/api/live") return getLive(request, env, ctx);
    if (url.pathname === "/api/query") return queryHistory(request, env, false);
    if (url.pathname === "/api/rankings") return queryHistory(request, env, true);
    if (url.pathname === "/media/current-code.png") return getCurrentScreenshot(env);
    if (url.pathname === "/health") return json({ ok: true, service: "pokopia-stamp-edge" });
    const response = await env.ASSETS.fetch(request);
    const headers = new Headers(response.headers);
    for (const [name, value] of Object.entries(securityHeaders())) headers.set(name, value);
    if (url.pathname === "/" || url.pathname.endsWith(".html")) headers.set("cache-control", "no-cache");
    return new Response(response.body, { status: response.status, headers });
  },
};
