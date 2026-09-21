const JSON_HEADERS = { "content-type": "application/json; charset=utf-8" };
const DAY_RE = /^\d{4}-\d{2}-\d{2}$/;
const CODE_RE = /^[0-9A-HJ-NP-Y]{6}$/;
const COMMUNITY_CODE_RE = /^[0-9A-Z]{6}$/;
const MAX_LIVE_BYTES = 1_000_000;
const MAX_SCREENSHOT_BYTES = 8_000_000;
const MAX_PLAYERS_PER_DAY = 3000;
const MAX_COMMUNITY_BYTES = 32_000;
const MAX_BACKUP_BYTES = 8_000_000;
const COMMUNITY_VISIBLE_SECONDS = 12 * 60 * 60;
const COMMUNITY_INVALID_VISIBLE_SECONDS = 30 * 60;
const COMMUNITY_FRESH_SECONDS = 5 * 60;
const COMMUNITY_UPLOADS_PER_DAY = 30;
const COMMUNITY_FEEDBACKS_PER_DAY = 100;
const COMMUNITY_ACTIVE_PER_PUBLISHER = 5;
const COMMUNITY_ROOM_TYPES = new Set([
  "stamp",
  "task",
  "flower",
  "material_solo",
  "other",
]);
const COMMUNITY_ROOM_TYPE_MAX_LENGTH = Math.max(
  ...Array.from(COMMUNITY_ROOM_TYPES, (value) => value.length),
);

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

function unixNow() {
  return Math.floor(Date.now() / 1000);
}

function cleanText(value, maxLength) {
  return String(value || "").replace(/[\u0000-\u001f\u007f]/g, "").trim().slice(0, maxLength);
}

async function saltedHash(env, value, purpose) {
  const salt = String(env.UPLOAD_TOKEN || "pokopia-public-community");
  const digest = await crypto.subtle.digest(
    "SHA-256",
    new TextEncoder().encode(`${purpose}\u0000${salt}\u0000${value}`),
  );
  return [...new Uint8Array(digest)].map((byte) => byte.toString(16).padStart(2, "0")).join("");
}

async function communityIdentity(request, env) {
  const ip = request.headers.get("cf-connecting-ip") || "local";
  const actorHash = await saltedHash(env, ip, "community-ip");
  return {
    actorHash,
    // Destructive feedback corroboration is separated by hashed source IP,
    // not by the browser-generated client id, so refreshing cannot create a
    // second vote. The raw address is never stored.
    reporterHash: actorHash,
  };
}

async function ownerHash(env, token) {
  return saltedHash(env, token, "community-owner");
}

async function communityWritesEnabled(env) {
  const row = await env.DB.prepare(
    "SELECT value FROM community_settings WHERE key = 'writes_enabled'",
  ).first();
  return !row || String(row.value).toLowerCase() !== "false";
}

function publicRoom(row) {
  const invalidAt = clampCount(row.invalid_reported_at);
  const fullAt = clampCount(row.full_reported_at);
  const pendingVotes = clampCount(row.invalid_votes);
  return {
    id: String(row.id),
    player_name: String(row.player_name),
    code: String(row.code),
    room_type: String(row.room_type),
    description: String(row.description || ""),
    created_at: clampCount(row.created_at),
    full: fullAt > 0,
    invalid: invalidAt > 0,
    invalid_pending: !invalidAt && pendingVotes > 0,
    expires_at: clampCount(row.created_at) + COMMUNITY_VISIBLE_SECONDS,
    invalid_hides_at: invalidAt ? invalidAt + COMMUNITY_INVALID_VISIBLE_SECONDS : null,
  };
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
      const [current, community] = await Promise.all([
        this.ctx.storage.get("state"),
        this.ctx.storage.get("community"),
      ]);
      if (current) server.send(JSON.stringify(current));
      if (community) server.send(JSON.stringify(community));
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
    if (request.method === "POST") {
      const payload = await request.json();
      if (payload?.event === "community_rooms") {
        await this.ctx.storage.put("community", payload);
      }
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
  payload.offline_after_seconds = Math.max(10, clampCount(env.OFFLINE_AFTER_SECONDS || 420));
  payload.code_unknown = !code && payload.code_unknown === true;
  payload.screenshot_url = (code || payload.code_unknown) && payload.screenshot_url
    ? "/media/current-code.png"
    : null;
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
    network: clampCount(row?.network_error_before_arrival),
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
      `INSERT INTO player_daily(day, name, name_search, visits, tasks, returned_before_arrival, room_closed_before_arrival, network_error_before_arrival)
       VALUES(?, ?, ?, ?, ?, ?, ?, ?)`,
    ).bind(day, row.name, row.nameSearch, row.visits, row.tasks, row.returned, row.closed, row.network));
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
  const offlineAfter = Math.max(10, clampCount(env.OFFLINE_AFTER_SECONDS || 420));
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
       SUM(returned_before_arrival + room_closed_before_arrival + network_error_before_arrival) failed_visits
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
      `SELECT name, SUM(returned_before_arrival + room_closed_before_arrival + network_error_before_arrival) count,
       SUM(returned_before_arrival) returned_before_arrival,
       SUM(room_closed_before_arrival) room_closed_before_arrival,
       SUM(network_error_before_arrival) network_error_before_arrival
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
        network_error_before_arrival: clampCount(row.network_error_before_arrival),
      })),
    },
    daily,
    generated_at: new Date().toISOString(),
  });
}

async function communityStatus(request, env) {
  if (!(await rateLimit(request, env, "community-status"))) {
    return json({ error: "请求过快，每秒最多一次。" }, 429, { "retry-after": "1" });
  }
  return json({
    writes_enabled: await communityWritesEnabled(env),
    room_lifetime_seconds: COMMUNITY_VISIBLE_SECONDS,
    invalid_visible_seconds: COMMUNITY_INVALID_VISIBLE_SECONDS,
  });
}

async function communityRoomsPayload(env) {
  const now = unixNow();
  const [result, openRow, writesEnabled] = await Promise.all([
    env.DB.prepare(
    `SELECT id, player_name, code, room_type, description, created_at,
            full_reported_at, invalid_reported_at, invalid_votes
     FROM community_rooms
     WHERE deleted_at IS NULL
       AND created_at > ?
       AND (invalid_reported_at IS NULL OR invalid_reported_at > ?)
     ORDER BY created_at DESC
     LIMIT 200`,
    ).bind(now - COMMUNITY_VISIBLE_SECONDS, now - COMMUNITY_INVALID_VISIBLE_SECONDS).all(),
    env.DB.prepare(
      `SELECT COUNT(*) open_count
       FROM community_rooms
       WHERE deleted_at IS NULL
         AND created_at > ?
         AND full_reported_at IS NULL
         AND invalid_reported_at IS NULL`,
    ).bind(now - COMMUNITY_VISIBLE_SECONDS).first(),
    communityWritesEnabled(env),
  ]);
  return {
    writes_enabled: writesEnabled,
    open_count: clampCount(openRow?.open_count),
    rooms: (result.results || []).map(publicRoom),
    generated_at: new Date().toISOString(),
  };
}

async function broadcastCommunityRooms(env) {
  const data = await communityRoomsPayload(env);
  await broadcastCommunityPayload(env, data);
  return data;
}

async function broadcastCommunityPayload(env, data) {
  const stub = await liveStub(env);
  await stub.fetch("https://live.internal/event", {
    method: "POST",
    headers: JSON_HEADERS,
    body: JSON.stringify({ event: "community_rooms", data }),
  });
}

async function listCommunityRooms(request, env) {
  if (!(await rateLimit(request, env, "community-rooms"))) {
    return json({ error: "请求过快，每秒最多一次。" }, 429, { "retry-after": "1" });
  }
  const data = await communityRoomsPayload(env);
  await broadcastCommunityPayload(env, data);
  return json(data);
}

async function incrementCommunityUsage(env, day, actorHash, field) {
  const upload = field === "uploads" ? 1 : 0;
  const feedback = field === "feedbacks" ? 1 : 0;
  const early = field === "early_invalid_feedbacks" ? 1 : 0;
  return env.DB.prepare(
    `INSERT INTO community_usage_daily(day, actor_hash, uploads, feedbacks, early_invalid_feedbacks)
     VALUES(?, ?, ?, ?, ?)
     ON CONFLICT(day, actor_hash) DO UPDATE SET
       uploads = uploads + excluded.uploads,
       feedbacks = feedbacks + excluded.feedbacks,
       early_invalid_feedbacks = early_invalid_feedbacks + excluded.early_invalid_feedbacks`,
  ).bind(day, actorHash, upload, feedback, early);
}

async function createCommunityRoom(request, env) {
  if (!(await rateLimit(request, env, "community-create"))) {
    return json({ error: "操作过快，请一秒后再试。" }, 429, { "retry-after": "1" });
  }
  if (!(await communityWritesEnabled(env))) return json({ error: "管理员已暂时关闭玩家开门信息提交。" }, 423);
  let payload;
  try {
    payload = await readJsonLimited(request, MAX_COMMUNITY_BYTES);
  } catch (error) {
    return json({ error: String(error.message || error) }, 400);
  }
  const playerName = cleanText(payload?.player_name, 40);
  const code = cleanText(payload?.code, 6).toUpperCase();
  const roomType = cleanText(
    payload?.room_type,
    COMMUNITY_ROOM_TYPE_MAX_LENGTH,
  ) || "stamp";
  const description = cleanText(payload?.description, 240);
  const ownerToken = cleanText(payload?.owner_token, 160);
  if (!playerName || Array.from(playerName).length > 20) return json({ error: "玩家名称应为1至20个字符。" }, 400);
  if (!COMMUNITY_CODE_RE.test(code)) return json({ error: "开门码必须正好是6位数字或英文字母。" }, 400);
  if (!COMMUNITY_ROOM_TYPES.has(roomType)) return json({ error: "开门类型无效。" }, 400);
  if (ownerToken.length < 24) return json({ error: "本地删除凭证无效，请刷新页面后重试。" }, 400);
  const now = unixNow();
  const day = beijingBusinessDay();
  const identity = await communityIdentity(request, env);
  const usage = await env.DB.prepare(
    "SELECT uploads FROM community_usage_daily WHERE day = ? AND actor_hash = ?",
  ).bind(day, identity.actorHash).first();
  if (clampCount(usage?.uploads) >= COMMUNITY_UPLOADS_PER_DAY) {
    return json({ error: `同一网络每天最多提交${COMMUNITY_UPLOADS_PER_DAY}辆车。` }, 429);
  }
  const recent = await env.DB.prepare(
    "SELECT MAX(created_at) last_at, SUM(CASE WHEN deleted_at IS NULL AND created_at > ? THEN 1 ELSE 0 END) active_count FROM community_rooms WHERE publisher_hash = ?",
  ).bind(now - COMMUNITY_VISIBLE_SECONDS, identity.actorHash).first();
  if (clampCount(recent?.active_count) >= COMMUNITY_ACTIVE_PER_PUBLISHER) {
    return json({ error: `同一网络最多同时保留${COMMUNITY_ACTIVE_PER_PUBLISHER}辆有效车。` }, 429);
  }
  if (now - clampCount(recent?.last_at) < 15) return json({ error: "两次提交至少间隔15秒。" }, 429);
  const duplicate = await env.DB.prepare(
    "SELECT id FROM community_rooms WHERE code = ? AND deleted_at IS NULL AND created_at > ? LIMIT 1",
  ).bind(code, now - COMMUNITY_VISIBLE_SECONDS).first();
  if (duplicate) return json({ error: "这个开门码仍在列表中，请勿重复提交。" }, 409);
  const id = crypto.randomUUID();
  const hashedOwner = await ownerHash(env, ownerToken);
  await env.DB.batch([
    env.DB.prepare(
      `INSERT INTO community_rooms(
        id, owner_hash, publisher_hash, player_name, player_search, code,
        room_type, description, business_day, created_at
      ) VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, ?)`,
    ).bind(
      id, hashedOwner, identity.actorHash, playerName,
      playerName.toLocaleLowerCase("und"), code, roomType, description, day, now,
    ),
    await incrementCommunityUsage(env, day, identity.actorHash, "uploads"),
  ]);
  const room = publicRoom({
    id, player_name: playerName, code, room_type: roomType, description,
    created_at: now, full_reported_at: null, invalid_reported_at: null, invalid_votes: 0,
  });
  const community = await broadcastCommunityRooms(env);
  return json({ ok: true, room, community }, 201);
}

function communityRoomId(pathname, suffix = "") {
  const escaped = suffix.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
  const match = pathname.match(new RegExp(`^/api/community/rooms/([0-9a-f-]{36})${escaped}$`, "i"));
  return match ? match[1] : "";
}

async function feedbackCommunityRoom(request, env, roomId) {
  if (!(await rateLimit(request, env, "community-feedback"))) {
    return json({ error: "操作过快，请一秒后再试。" }, 429, { "retry-after": "1" });
  }
  if (!(await communityWritesEnabled(env))) return json({ error: "管理员已暂时关闭反馈功能。" }, 423);
  let payload;
  try {
    payload = await readJsonLimited(request, MAX_COMMUNITY_BYTES);
  } catch (error) {
    return json({ error: String(error.message || error) }, 400);
  }
  const action = cleanText(payload?.action, 20);
  if (!["full", "full_wrong", "invalid", "invalid_wrong"].includes(action)) {
    return json({ error: "反馈类型无效。" }, 400);
  }
  const now = unixNow();
  const room = await env.DB.prepare(
    "SELECT * FROM community_rooms WHERE id = ? AND deleted_at IS NULL AND created_at > ?",
  ).bind(roomId, now - COMMUNITY_VISIBLE_SECONDS).first();
  if (!room) return json({ error: "该开门信息不存在或已经过期。" }, 404);
  if (["full", "full_wrong"].includes(action) && room.room_type !== "stamp") {
    return json({ error: "只有梦幻章车可以反馈已满。" }, 400);
  }
  const identity = await communityIdentity(request, env);
  const day = beijingBusinessDay();
  const usage = await env.DB.prepare(
    "SELECT feedbacks FROM community_usage_daily WHERE day = ? AND actor_hash = ?",
  ).bind(day, identity.actorHash).first();
  if (clampCount(usage?.feedbacks) >= COMMUNITY_FEEDBACKS_PER_DAY) {
    return json({ error: `同一网络每天最多反馈${COMMUNITY_FEEDBACKS_PER_DAY}次。` }, 429);
  }
  const recentDuplicate = await env.DB.prepare(
    "SELECT id FROM community_feedback WHERE room_id = ? AND reporter_hash = ? AND action = ? AND created_at > ? LIMIT 1",
  ).bind(roomId, identity.reporterHash, action, now - 10 * 60).first();
  if (recentDuplicate) return json({ error: "相同反馈已经记录，请勿重复点击。" }, 409);
  const roomAge = Math.max(0, now - clampCount(room.created_at));
  let update;
  let message = "反馈已记录。";
  if (action === "full") {
    update = env.DB.prepare(
      "UPDATE community_rooms SET full_reported_at = COALESCE(full_reported_at, ?) WHERE id = ?",
    ).bind(now, roomId);
    message = "已标记为已满。";
  } else if (action === "full_wrong") {
    update = env.DB.prepare("UPDATE community_rooms SET full_reported_at = NULL WHERE id = ?").bind(roomId);
    message = "已恢复已满状态。";
  } else if (action === "invalid_wrong") {
    update = env.DB.prepare(
      "UPDATE community_rooms SET invalid_reported_at = NULL, invalid_votes = 0 WHERE id = ?",
    ).bind(roomId);
    message = "已恢复为有效信息。";
  } else if (room.invalid_reported_at) {
    return json({ error: "该信息已经被标记失效。" }, 409);
  } else if (roomAge < COMMUNITY_FRESH_SECONDS) {
    update = env.DB.prepare(
      `UPDATE community_rooms SET
        invalid_reported_at = CASE WHEN invalid_votes >= 1 THEN ? ELSE NULL END,
        invalid_votes = CASE WHEN invalid_votes >= 1 THEN 0 ELSE invalid_votes + 1 END
       WHERE id = ?`,
    ).bind(now, roomId);
    message = clampCount(room.invalid_votes) >= 1
      ? "已由两名访问者确认失效，30分钟后自动隐藏。"
      : "新开车辆需要另一名访问者确认失效，当前已进入待核验。";
  } else {
    update = env.DB.prepare(
      "UPDATE community_rooms SET invalid_reported_at = ?, invalid_votes = 0 WHERE id = ?",
    ).bind(now, roomId);
    message = "已标记失效，30分钟后自动隐藏。";
  }
  const statements = [
    update,
    env.DB.prepare(
      "INSERT INTO community_feedback(room_id, reporter_hash, action, created_at, room_age_seconds) VALUES(?, ?, ?, ?, ?)",
    ).bind(roomId, identity.reporterHash, action, now, roomAge),
    await incrementCommunityUsage(env, day, identity.actorHash, "feedbacks"),
  ];
  if (action === "invalid" && roomAge < COMMUNITY_FRESH_SECONDS) {
    statements.push(await incrementCommunityUsage(env, day, identity.actorHash, "early_invalid_feedbacks"));
  }
  await env.DB.batch(statements);
  const community = await broadcastCommunityRooms(env);
  return json({ ok: true, message, community });
}

async function deleteCommunityRoom(request, env, roomId) {
  if (!(await rateLimit(request, env, "community-delete"))) {
    return json({ error: "操作过快，请一秒后再试。" }, 429, { "retry-after": "1" });
  }
  let payload;
  try {
    payload = await readJsonLimited(request, MAX_COMMUNITY_BYTES);
  } catch (error) {
    return json({ error: String(error.message || error) }, 400);
  }
  const token = cleanText(payload?.owner_token, 160);
  if (token.length < 24) return json({ error: "没有这条信息的本地删除凭证。" }, 403);
  const hashedOwner = await ownerHash(env, token);
  const result = await env.DB.prepare(
    "UPDATE community_rooms SET deleted_at = ?, delete_reason = 'owner_correction' WHERE id = ? AND owner_hash = ? AND deleted_at IS NULL",
  ).bind(unixNow(), roomId, hashedOwner).run();
  if (!result.meta?.changes) return json({ error: "删除凭证不匹配，只有原发布浏览器可以删除。" }, 403);
  const community = await broadcastCommunityRooms(env);
  return json({ ok: true, deleted: true, community });
}

async function communityLeaderboard(request, env) {
  if (!(await rateLimit(request, env, "community-leaderboard"))) {
    return json({ error: "请求过快，每秒最多一次。" }, 429, { "retry-after": "1" });
  }
  const url = new URL(request.url);
  let range;
  try {
    range = queryRange(url);
  } catch (error) {
    return json({ error: String(error.message || error) }, 400);
  }
  const playerQuery = cleanText(url.searchParams.get("player"), 40);
  const playerFilter = playerQuery ? `%${playerQuery.toLocaleLowerCase("und")}%` : "%";
  const result = await env.DB.prepare(
    `SELECT player_name name, COUNT(*) count,
            SUM(CASE WHEN room_type = 'stamp' THEN 1 ELSE 0 END) stamp_count,
            SUM(CASE WHEN room_type = 'task' THEN 1 ELSE 0 END) task_count,
            SUM(CASE WHEN room_type = 'flower' THEN 1 ELSE 0 END) flower_count,
            SUM(CASE WHEN room_type = 'material_solo' THEN 1 ELSE 0 END) material_solo_count,
            SUM(CASE WHEN room_type = 'other' THEN 1 ELSE 0 END) other_count
     FROM community_rooms
     WHERE deleted_at IS NULL AND business_day BETWEEN ? AND ?
       AND player_search LIKE ?
     GROUP BY player_name
     ORDER BY count DESC, player_name
     LIMIT 100`,
  ).bind(range.start, range.end, playerFilter).all();
  return json({
    period: range.label,
    start: range.start === "0000-01-01" ? null : range.start,
    end_inclusive: range.end === "9999-12-31" ? null : range.end,
    player_query: playerQuery,
    rankings: (result.results || []).map((row, index) => ({
      rank: index + 1,
      name: String(row.name),
      count: clampCount(row.count),
      by_type: {
        stamp: clampCount(row.stamp_count), task: clampCount(row.task_count),
        flower: clampCount(row.flower_count), material_solo: clampCount(row.material_solo_count),
        other: clampCount(row.other_count),
      },
    })),
  });
}

async function exportCommunityData(request, env) {
  if (!(await authorized(request, env))) return json({ error: "未授权" }, 401);
  const [rooms, feedback, usage, settings] = await Promise.all([
    env.DB.prepare("SELECT * FROM community_rooms ORDER BY created_at, id").all(),
    env.DB.prepare("SELECT * FROM community_feedback ORDER BY id").all(),
    env.DB.prepare("SELECT * FROM community_usage_daily ORDER BY day, actor_hash").all(),
    env.DB.prepare("SELECT * FROM community_settings ORDER BY key").all(),
  ]);
  return json({
    schema: "pokopia-community-backup-v1",
    exported_at: new Date().toISOString(),
    rooms: rooms.results || [],
    feedback: feedback.results || [],
    usage: usage.results || [],
    settings: settings.results || [],
  });
}

async function importCommunityData(request, env) {
  if (!(await authorized(request, env))) return json({ error: "未授权" }, 401);
  let payload;
  try {
    payload = await readJsonLimited(request, MAX_BACKUP_BYTES);
  } catch (error) {
    return json({ error: String(error.message || error) }, 400);
  }
  if (payload?.schema !== "pokopia-community-backup-v1") return json({ error: "备份格式不匹配。" }, 400);
  const rooms = Array.isArray(payload.rooms) ? payload.rooms.slice(0, 20_000) : [];
  const feedback = Array.isArray(payload.feedback) ? payload.feedback.slice(0, 100_000) : [];
  const usage = Array.isArray(payload.usage) ? payload.usage.slice(0, 50_000) : [];
  const statements = [];
  for (const row of rooms) {
    if (!row?.id || !row?.owner_hash || !COMMUNITY_CODE_RE.test(String(row.code || ""))) continue;
    statements.push(env.DB.prepare(
      `INSERT OR REPLACE INTO community_rooms(
        id, owner_hash, publisher_hash, player_name, player_search, code, room_type,
        description, business_day, created_at, full_reported_at, invalid_reported_at,
        invalid_votes, deleted_at, delete_reason
      ) VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)`,
    ).bind(
      row.id, row.owner_hash, row.publisher_hash, row.player_name, row.player_search,
      row.code, row.room_type, row.description || "", row.business_day, clampCount(row.created_at),
      row.full_reported_at || null, row.invalid_reported_at || null, clampCount(row.invalid_votes),
      row.deleted_at || null, row.delete_reason || null,
    ));
  }
  for (const row of feedback) {
    statements.push(env.DB.prepare(
      "INSERT OR REPLACE INTO community_feedback(id, room_id, reporter_hash, action, created_at, room_age_seconds) VALUES(?, ?, ?, ?, ?, ?)",
    ).bind(row.id, row.room_id, row.reporter_hash, row.action, clampCount(row.created_at), clampCount(row.room_age_seconds)));
  }
  for (const row of usage) {
    statements.push(env.DB.prepare(
      "INSERT OR REPLACE INTO community_usage_daily(day, actor_hash, uploads, feedbacks, early_invalid_feedbacks) VALUES(?, ?, ?, ?, ?)",
    ).bind(row.day, row.actor_hash, clampCount(row.uploads), clampCount(row.feedbacks), clampCount(row.early_invalid_feedbacks)));
  }
  for (let offset = 0; offset < statements.length; offset += 80) {
    await env.DB.batch(statements.slice(offset, offset + 80));
  }
  const community = await broadcastCommunityRooms(env);
  return json({ ok: true, rooms: rooms.length, feedback: feedback.length, usage: usage.length, community });
}

async function toggleCommunityWrites(request, env) {
  if (!(await authorized(request, env))) return json({ error: "未授权" }, 401);
  let payload = {};
  try {
    payload = await readJsonLimited(request, MAX_COMMUNITY_BYTES);
  } catch (error) {
    return json({ error: String(error.message || error) }, 400);
  }
  const current = await communityWritesEnabled(env);
  const enabled = typeof payload.enabled === "boolean" ? payload.enabled : !current;
  await env.DB.prepare(
    `INSERT INTO community_settings(key, value, updated_at) VALUES('writes_enabled', ?, ?)
     ON CONFLICT(key) DO UPDATE SET value = excluded.value, updated_at = excluded.updated_at`,
  ).bind(enabled ? "true" : "false", unixNow()).run();
  const community = await broadcastCommunityRooms(env);
  return json({ ok: true, writes_enabled: enabled, community });
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
    if (request.method === "POST" && url.pathname === "/api/community/rooms") return createCommunityRoom(request, env);
    if (request.method === "POST") {
      const feedbackRoomId = communityRoomId(url.pathname, "/feedback");
      if (feedbackRoomId) return feedbackCommunityRoom(request, env, feedbackRoomId);
    }
    if (request.method === "DELETE") {
      const deleteRoomId = communityRoomId(url.pathname);
      if (deleteRoomId) return deleteCommunityRoom(request, env, deleteRoomId);
    }
    if (request.method === "POST" && url.pathname === "/api/admin/community/toggle") return toggleCommunityWrites(request, env);
    if (request.method === "POST" && url.pathname === "/api/admin/community/import") return importCommunityData(request, env);
    if (request.method !== "GET" && request.method !== "HEAD") return json({ error: "Method not allowed" }, 405);
    if (url.pathname === "/api/stream") return streamLive(request, env);
    if (url.pathname === "/api/live") return getLive(request, env, ctx);
    if (url.pathname === "/api/query") return queryHistory(request, env, false);
    if (url.pathname === "/api/rankings") return queryHistory(request, env, true);
    if (url.pathname === "/api/community/status") return communityStatus(request, env);
    if (url.pathname === "/api/community/rooms") return listCommunityRooms(request, env);
    if (url.pathname === "/api/community/leaderboard") return communityLeaderboard(request, env);
    if (url.pathname === "/api/admin/community/export") return exportCommunityData(request, env);
    if (url.pathname === "/media/current-code.png") return getCurrentScreenshot(env);
    if (url.pathname === "/health") return json({ ok: true, service: "pokopia-stamp-edge" });
    const response = await env.ASSETS.fetch(request);
    const headers = new Headers(response.headers);
    for (const [name, value] of Object.entries(securityHeaders())) headers.set(name, value);
    if (url.pathname === "/" || url.pathname.endsWith(".html")) headers.set("cache-control", "no-cache");
    return new Response(response.body, { status: response.status, headers });
  },
};
