"use strict";

const $ = (id) => document.getElementById(id);
const localPreview = ["127.0.0.1", "localhost", "::1"].includes(location.hostname);
const clientId = (() => {
  const key = "pokopia-client-id";
  try {
    let value = sessionStorage.getItem(key);
    if (!value) {
      value = globalThis.crypto?.randomUUID?.() || `${Date.now()}-${Math.random()}`;
      sessionStorage.setItem(key, value);
    }
    return value;
  } catch {
    return globalThis.crypto?.randomUUID?.() || `${Date.now()}-${Math.random()}`;
  }
})();
const state = {
  live: null,
  rankingPeriod: "all",
  rankingView: "all",
  countdownRemaining: null,
  countdownSampledAt: 0,
  toastTimer: null,
  lastRequestAt: new Map(),
  liveSocket: null,
  reconnectTimer: null,
  reconnectDelay: 2000,
  lastLiveReceivedAt: 0,
  staleReconnectFor: 0,
  lastForegroundReconnectAt: 0,
  communityData: null,
  communityExpiryTimer: null,
  queryGeneration: 0,
  rankingGeneration: 0,
  goodPeriod: "all",
  goodView: "all",
  goodGeneration: 0,
  activeDrawer: "",
};

function number(value) {
  const parsed = Number(value);
  return Number.isFinite(parsed) ? parsed : 0;
}

function shortTime(value) {
  if (!value) return "--:--:--";
  const match = String(value).match(/T?(\d{2}:\d{2}:\d{2})/);
  return match ? match[1] : String(value);
}

function showToast(message) {
  const toast = $("toast");
  toast.textContent = message;
  toast.classList.add("show");
  clearTimeout(state.toastTimer);
  state.toastTimer = setTimeout(() => toast.classList.remove("show"), 1800);
}

function setStatus(status = {}) {
  const key = status.key || "offline";
  const pill = $("status-pill");
  pill.className = `status-pill is-${key}`;
  pill.innerHTML = "";
  pill.append(document.createElement("i"), document.createTextNode(status.label || "未知状态"));
}

function renderPlayers(players = []) {
  const list = $("player-list");
  list.innerHTML = "";
  $("room-count").textContent = `${players.length} 人`;
  if (!players.length) {
    const empty = document.createElement("div");
    empty.className = "empty-state";
    empty.textContent = "当前没有已识别的房间玩家";
    list.append(empty);
    return;
  }
  players.forEach((player) => {
    const row = document.createElement("div");
    row.className = "player-row";
    const name = document.createElement("b");
    name.textContent = player.name || "未知玩家";
    const badge = document.createElement("span");
    badge.className = `player-state ${player.status === "到达" ? "arrived" : ""}`;
    badge.textContent = player.status || "路上";
    row.append(name, badge);
    list.append(row);
  });
}

function renderTimeline(targetId, events, builder, emptyText) {
  const list = $(targetId);
  list.innerHTML = "";
  if (!events.length) {
    const empty = document.createElement("li");
    empty.className = "empty-state";
    empty.textContent = emptyText;
    list.append(empty);
    return;
  }
  [...events].reverse().forEach((event) => {
    const item = document.createElement("li");
    const time = document.createElement("time");
    time.textContent = shortTime(event.time);
    const content = builder(event);
    item.append(time, content);
    list.append(item);
  });
}

function renderRecent(rounds = []) {
  const root = $("recent-rounds");
  root.innerHTML = "";
  if (!rounds.length) {
    const empty = document.createElement("div");
    empty.className = "empty-state";
    empty.textContent = "暂无已结算轮次";
    root.append(empty);
    return;
  }
  rounds.forEach((round) => {
    const card = document.createElement("article");
    card.className = "round-chip";
    const header = document.createElement("header");
    const title = document.createElement("span");
    title.textContent = `第 ${round.reopen_index || "?"} 次重开`;
    const time = document.createElement("time");
    time.textContent = shortTime(round.time);
    header.append(title, time);
    const summary = document.createElement("p");
    summary.textContent = `成功 ${number(round.successful)} 人 · 失败 ${number(round.failed)} 人 · 共 ${Array.isArray(round.players) ? round.players.length : 0} 次记录`;
    card.append(header, summary);
    const networkError = round.network_connection_error || {};
    const pendingNames = Array.isArray(networkError.players_not_arrived)
      ? networkError.players_not_arrived.filter(Boolean)
      : [];
    if (round.reopen_reason === "network_connection_error" || networkError.detected) {
      const detail = document.createElement("p");
      detail.textContent = pendingNames.length
        ? `网络连接错误时未抵达：${pendingNames.join("、")}`
        : "网络连接错误：没有尚未抵达的玩家";
      card.append(detail);
    }
    root.append(card);
  });
}

function renderLive(data) {
  const incomingReceivedAt = Number(data.edge_received_at_epoch || data.updated_at_epoch || 0);
  if (incomingReceivedAt && incomingReceivedAt !== state.lastLiveReceivedAt) {
    state.lastLiveReceivedAt = incomingReceivedAt;
    state.staleReconnectFor = 0;
  }
  state.live = data;
  setStatus(data.status);
  $("phase-label").textContent = data.phase?.label || "等待状态更新";
  const code = String(data.code || "");
  const codeUnknown = Boolean(data.code_unknown);
  $("code-value").textContent = codeUnknown ? "？？？？？？" : (code || "------");
  $("copy-code").disabled = !code;
  const counts = data.counts || {};
  $("round-label").textContent = (code || codeUnknown)
    ? `今日第 ${number(counts.daily_runs)} 轮 · 总第 ${number(counts.total_runs)} 轮`
    : "等待新轮次";

  const screenshot = $("code-screenshot");
  if (data.screenshot_url) {
    const separator = data.screenshot_url.includes("?") ? "&" : "?";
    const screenshotURL = `${data.screenshot_url}${separator}revision=${number(data.code_revision)}`;
    if (screenshot.getAttribute("src") !== screenshotURL) screenshot.src = screenshotURL;
    screenshot.hidden = false;
    $("screenshot-empty").hidden = true;
  } else {
    screenshot.hidden = true;
    $("screenshot-empty").hidden = false;
  }

  const completed = Math.min(3, number(data.tasks?.completed));
  $("task-count").textContent = completed;
  $("task-progress").style.width = `${completed / 3 * 100}%`;
  $("round-success").textContent = number(counts.round_player_entries);
  $("round-failure").textContent = number(counts.round_player_entry_failures);
  $("day-success").textContent = number(counts.daily_player_entries);
  $("day-failure").textContent = number(counts.daily_player_entry_failures);
  $("total-success").textContent = number(counts.total_player_entries);
  $("total-failure").textContent = number(counts.total_player_entry_failures);

  const announcement = data.announcement || {};
  $("previous-summary").textContent = announcement.previous_summary || "暂无上轮完整记录";
  $("restart-summary").textContent = announcement.restart_summary || "重开耗时尚未记录";
  $("announcement-text").textContent = announcement.text || "尚未生成开放播报";
  $("deadline-label").textContent = announcement.deadline ? `预计至 ${announcement.deadline}` : "等待轮次开放";
  const timerDeadline = Number(data.timer?.deadline_epoch);
  const timerRemaining = Number(data.timer?.remaining_seconds);
  state.countdownRemaining = data.timer?.active && Number.isFinite(timerDeadline) && timerDeadline > 0
    ? Math.max(0, timerDeadline - Date.now() / 1000)
    : data.timer?.active && Number.isFinite(timerRemaining)
      ? Math.max(0, timerRemaining)
      : null;
  state.countdownSampledAt = performance.now();

  renderPlayers(Array.isArray(data.players) ? data.players : []);
  const logs = data.round_log || {};
  renderTimeline("task-events", logs.tasks || [], (event) => {
    const block = document.createElement("div");
    const title = document.createElement("b");
    title.textContent = `任务进度更新为 ${number(event.completed)}/3`;
    const note = document.createElement("p");
    note.textContent = `本次识别新增 ${number(event.added)} 条`;
    block.append(title, note);
    return block;
  }, "本轮尚无任务完成记录");
  renderTimeline("player-events", logs.players || [], (event) => {
    const block = document.createElement("div");
    const title = document.createElement("b");
    title.textContent = `${event.name} ${event.notification}`;
    const note = document.createElement("p");
    note.textContent = event.room_status ? `当前记录：${event.room_status}` : "已从房间列表移除";
    block.append(title, note);
    return block;
  }, "本轮尚无玩家进出记录");
  renderRecent(data.recent_rounds || []);

  const stale = data.stale_seconds == null ? null : Math.round(data.stale_seconds);
  $("last-updated").textContent = stale == null ? "尚未收到心跳" : stale <= 2 ? "刚刚更新" : `${stale}秒前更新`;
}

function updateCountdown() {
  const receivedAt = Number(state.live?.edge_received_at_epoch || state.live?.updated_at_epoch || 0);
  const offlineAfter = number(state.live?.offline_after_seconds) || 420;
  const stale = receivedAt ? Math.max(0, Math.round(Date.now() / 1000 - receivedAt)) : null;
  $("last-updated").textContent = stale == null ? "尚未收到心跳" : stale <= 2 ? "刚刚更新" : `${stale}秒前更新`;
  const heartbeatStale = Boolean(receivedAt && Date.now() / 1000 - receivedAt > offlineAfter);
  if (heartbeatStale) {
    setStatus({ key: "offline", label: "离线" });
    $("phase-label").textContent = "Windows状态心跳已中断，正在重新连接";
    if (!localPreview && state.staleReconnectFor !== receivedAt) {
      state.staleReconnectFor = receivedAt;
      reconnectLiveStreamNow("stale-heartbeat");
    }
  }
  if (heartbeatStale || state.countdownRemaining == null || state.live?.status?.key !== "open") {
    $("countdown").textContent = state.live?.status?.key === "reopening" ? "重开中" : "--:--";
    return;
  }
  const sinceSnapshot = Math.max(0, (performance.now() - state.countdownSampledAt) / 1000);
  const remaining = Math.max(0, Math.ceil(state.countdownRemaining - sinceSnapshot));
  const minutes = Math.floor(remaining / 60);
  const seconds = remaining % 60;
  $("countdown").textContent = `${String(minutes).padStart(2, "0")}:${String(seconds).padStart(2, "0")}`;
}

async function fetchJSON(url, options = {}) {
  const bucket = new URL(url, location.href).pathname;
  const now = performance.now();
  const reservedAt = Math.max(now, (state.lastRequestAt.get(bucket) || -Infinity) + 1000);
  state.lastRequestAt.set(bucket, reservedAt);
  const wait = Math.max(0, reservedAt - now);
  if (wait) await new Promise((resolve) => setTimeout(resolve, wait));
  const response = await fetch(url, {
    cache: "no-store",
    method: options.method || "GET",
    headers: {
      "X-Pokopia-Client": clientId,
      ...(options.body ? { "Content-Type": "application/json" } : {}),
    },
    body: options.body ? JSON.stringify(options.body) : undefined,
  });
  const body = await response.json();
  if (!response.ok) throw new Error(body.error || `请求失败 ${response.status}`);
  return body;
}

async function refreshLive() {
  try {
    renderLive(await fetchJSON(`/api/live?t=${Date.now()}`));
  } catch (error) {
    setStatus({ key: "offline", label: "页面服务离线" });
    $("phase-label").textContent = error.message;
  }
  updateCountdown();
}

function connectLiveStream() {
  clearTimeout(state.reconnectTimer);
  const protocol = location.protocol === "https:" ? "wss:" : "ws:";
  const socket = new WebSocket(`${protocol}//${location.host}/api/stream`);
  state.liveSocket = socket;
  socket.addEventListener("open", () => { state.reconnectDelay = 2000; });
  socket.addEventListener("message", (event) => {
    try {
      const payload = JSON.parse(event.data);
      if (payload?.event === "community_rooms") {
        renderCommunityRooms(payload.data || {});
        return;
      }
      renderLive(payload);
      updateCountdown();
    } catch {
      // A malformed update is ignored; the next server snapshot will replace it.
    }
  });
  socket.addEventListener("close", () => {
    if (state.liveSocket !== socket) return;
    state.liveSocket = null;
    state.reconnectTimer = setTimeout(connectLiveStream, state.reconnectDelay);
    state.reconnectDelay = Math.min(60_000, state.reconnectDelay * 2);
  });
  socket.addEventListener("error", () => socket.close());
}

function reconnectLiveStreamNow(reason = "foreground-resume") {
  if (localPreview) {
    refreshLive();
    return;
  }
  const now = Date.now();
  if (reason === "foreground-resume") {
    if (now - state.lastForegroundReconnectAt < 1000) return;
    state.lastForegroundReconnectAt = now;
  }
  clearTimeout(state.reconnectTimer);
  const previous = state.liveSocket;
  state.liveSocket = null;
  if (previous && previous.readyState < WebSocket.CLOSING) {
    try { previous.close(1000, reason); } catch { /* reconnect below */ }
  }
  state.reconnectDelay = 2000;
  connectLiveStream();
}

document.addEventListener("visibilitychange", () => {
  if (document.visibilityState === "visible") reconnectLiveStreamNow();
});
window.addEventListener("pageshow", (event) => {
  if (event.persisted) reconnectLiveStreamNow();
});

function rankingItem(row, kind) {
  const item = document.createElement("li");
  const rank = document.createElement("span");
  rank.className = "rank-number";
  rank.textContent = row.rank;
  const name = document.createElement("b");
  name.textContent = row.name;
  const count = document.createElement("strong");
  count.textContent = `${row.count} 次`;
  item.append(rank, name, count);
  if (kind === "failures") {
    const note = document.createElement("small");
    note.className = "ranking-note";
    note.textContent = `到达前返回 ${number(row.returned_before_arrival)} · 正常重开冻结 ${number(row.room_closed_before_arrival)} · 网络错误 ${number(row.network_error_before_arrival)}`;
    item.append(note);
  }
  return item;
}

function fillRanking(target, rows, kind) {
  const list = $(target);
  list.innerHTML = "";
  if (!rows.length) {
    const empty = document.createElement("li");
    empty.className = "empty-state";
    empty.textContent = "暂无记录";
    list.append(empty);
    return;
  }
  rows.forEach((row) => list.append(rankingItem(row, kind)));
}

async function loadRankings(period = state.rankingPeriod) {
  const generation = ++state.rankingGeneration;
  state.rankingPeriod = period;
  try {
    const data = await fetchJSON(`/api/rankings?period=${encodeURIComponent(period)}`);
    if (generation !== state.rankingGeneration) return;
    const range = data.start ? `${data.start} ～ ${data.end_inclusive}` : "全部业务日期";
    $("ranking-range").textContent = `${range} · 前10名`;
    fillRanking("visit-ranking", data.rankings.visits, "visits");
    fillRanking("task-ranking", data.rankings.tasks, "tasks");
    fillRanking("failure-ranking", data.rankings.failures, "failures");
  } catch (error) {
    if (generation !== state.rankingGeneration) return;
    $("ranking-range").textContent = error.message;
  }
}

function renderQuery(data) {
  const names = data.matched_players || [];
  const range = data.start ? `${data.start} ～ ${data.end_inclusive}` : "全部业务日期";
  $("query-meta").textContent = `${range}${data.player_query ? ` · 模糊匹配“${data.player_query}”：${names.length ? names.join("、") : "无结果"}` : " · 所有玩家"}`;
  const summary = data.summary || {};
  $("query-summary").innerHTML = "";
  [["成功人次", summary.successful_visits], ["失败人次", summary.failed_visits], ["任务参与", summary.task_participations]].forEach(([label, value]) => {
    const card = document.createElement("div");
    const text = document.createTextNode(label);
    const count = document.createElement("b");
    count.textContent = number(value);
    card.append(text, count);
    $("query-summary").append(card);
  });
  const tbody = $("daily-table");
  tbody.innerHTML = "";
  let rounds = 0, success = 0, failed = 0, tasks = 0;
  (data.daily || []).forEach((row) => {
    const tr = document.createElement("tr");
    [row.date, row.rounds, row.successful_visits, row.failed_visits, row.task_participations].forEach((value) => {
      const td = document.createElement("td"); td.textContent = value; tr.append(td);
    });
    tbody.append(tr);
    rounds += number(row.rounds); success += number(row.successful_visits); failed += number(row.failed_visits); tasks += number(row.task_participations);
  });
  if (!data.daily?.length) {
    const tr = document.createElement("tr"); const td = document.createElement("td");
    td.colSpan = 5; td.className = "empty-state"; td.textContent = "所选范围暂无记录"; tr.append(td); tbody.append(tr);
  }
  $("daily-total").innerHTML = `<tr><td>总计</td><td>${rounds}</td><td>${success}</td><td>${failed}</td><td>${tasks}</td></tr>`;
  fillRanking("query-visits", data.rankings.visits, "visits");
  fillRanking("query-tasks", data.rankings.tasks, "tasks");
  fillRanking("query-failures", data.rankings.failures, "failures");
}

async function submitQuery(event) {
  event?.preventDefault();
  const generation = ++state.queryGeneration;
  const params = new URLSearchParams();
  const period = $("query-period").value;
  const start = $("query-start").value;
  const end = $("query-end").value;
  const player = $("query-player").value.trim();
  if (start || end) { if (start) params.set("start", start); if (end) params.set("end", end); }
  else params.set("period", period);
  if (player) params.set("player", player);
  params.set("limit", "1000");
  try {
    const result = await fetchJSON(`/api/query?${params}`);
    if (generation === state.queryGeneration) renderQuery(result);
  } catch (error) {
    if (generation === state.queryGeneration) $("query-meta").textContent = error.message;
  }
}

function communityOwners() {
  try { return JSON.parse(localStorage.getItem("pokopia-community-owners") || "{}"); }
  catch { return {}; }
}

function saveCommunityOwner(roomId, token) {
  const owners = communityOwners();
  owners[roomId] = token;
  localStorage.setItem("pokopia-community-owners", JSON.stringify(owners));
}

function newOwnerToken() {
  const bytes = new Uint8Array(32);
  crypto.getRandomValues(bytes);
  return [...bytes].map((value) => value.toString(16).padStart(2, "0")).join("");
}

function roomTypeLabel(type) {
  return ({ stamp: "梦幻章车", task: "任务车", flower: "花车", material_solo: "材料单车", other: "其它车" })[type] || "其它车";
}

function beijingTime(epoch) {
  if (!epoch) return "时间未知";
  return new Intl.DateTimeFormat("zh-CN", {
    timeZone: "Asia/Shanghai", year: "numeric", month: "2-digit", day: "2-digit",
    hour: "2-digit", minute: "2-digit", hour12: false,
  }).format(new Date(number(epoch) * 1000)).replaceAll("/", "-");
}

function communityBadge(text, kind = "") {
  const badge = document.createElement("span");
  badge.className = `community-badge ${kind}`;
  badge.textContent = text;
  return badge;
}

function communityAction(label, callback, danger = false) {
  const button = document.createElement("button");
  button.type = "button";
  button.textContent = label;
  if (danger) button.classList.add("danger");
  button.addEventListener("click", callback);
  return button;
}

async function feedbackRoom(room, action) {
  try {
    const result = await fetchJSON(`/api/community/rooms/${room.id}/feedback`, {
      method: "POST", body: { action },
    });
    showToast(result.message || "反馈已记录");
    if (result.community) renderCommunityRooms(result.community);
  } catch (error) { showToast(error.message); }
}

async function deleteRoom(room) {
  const owners = communityOwners();
  const ownerToken = owners[room.id];
  if (!ownerToken) return;
  if (!confirm("真的要删除吗？删除后不计入好人榜统计哦；如果确实是输错了再删。")) return;
  try {
    const result = await fetchJSON(`/api/community/rooms/${room.id}`, {
      method: "DELETE", body: { owner_token: ownerToken },
    });
    delete owners[room.id];
    localStorage.setItem("pokopia-community-owners", JSON.stringify(owners));
    showToast("已删除，不计入好人榜");
    if (result.community) renderCommunityRooms(result.community);
  } catch (error) { showToast(error.message); }
}

function renderCommunityRooms(data) {
  state.communityData = data;
  clearTimeout(state.communityExpiryTimer);
  const now = Date.now() / 1000;
  const sourceRooms = Array.isArray(data.rooms) ? data.rooms : [];
  const rooms = sourceRooms.filter((room) => {
    if (number(room.expires_at) && number(room.expires_at) <= now) return false;
    if (number(room.invalid_hides_at) && number(room.invalid_hides_at) <= now) return false;
    return true;
  });
  const openCount = rooms.filter((room) => !room.full && !room.invalid).length;
  const countBadge = $("community-open-count");
  countBadge.textContent = openCount > 999 ? "999+" : String(openCount);
  countBadge.setAttribute("aria-label", `当前有${openCount}辆车正在开放`);
  countBadge.hidden = false;
  const stateBox = $("community-write-state");
  stateBox.textContent = data.writes_enabled
    ? "玩家提交与反馈：开放中"
    : "玩家提交与反馈：管理员已暂停";
  stateBox.classList.toggle("closed", !data.writes_enabled);
  $("community-submit").disabled = !data.writes_enabled;
  const root = $("community-rooms");
  root.innerHTML = "";
  if (!rooms.length) {
    const empty = document.createElement("div");
    empty.className = "empty-state";
    empty.textContent = "当前没有其它玩家发布的开门信息";
    root.append(empty);
  } else {
    const owners = communityOwners();
    rooms.forEach((room) => {
      const card = document.createElement("article"); card.className = "community-room";
      const header = document.createElement("header");
      const code = document.createElement("div"); code.className = "community-code"; code.textContent = `CODE：${room.code}`;
      const badges = document.createElement("div"); badges.className = "community-badges";
      badges.append(communityBadge(roomTypeLabel(room.room_type)));
      if (room.full) badges.append(communityBadge("已满", "full"));
      if (room.invalid_pending) badges.append(communityBadge("失效待核验", "pending"));
      if (room.invalid) badges.append(communityBadge("失效", "invalid"));
      if (!room.full && !room.invalid && !room.invalid_pending) badges.append(communityBadge("开放中"));
      header.append(code, badges);
      const meta = document.createElement("p"); meta.className = "community-room-meta";
      meta.textContent = `开门人：${room.player_name}（${beijingTime(room.created_at)}开）`;
      const description = document.createElement("p"); description.className = "community-room-description";
      description.textContent = room.description ? `描述：${room.description}` : "描述：未填写";
      const actions = document.createElement("div"); actions.className = "community-actions";
      if (data.writes_enabled) {
        if (room.room_type === "stamp") {
          if (!room.full) actions.append(communityAction("梦幻章满", () => feedbackRoom(room, "full")));
          else actions.append(communityAction("已满有误", () => feedbackRoom(room, "full_wrong")));
        }
        if (!room.invalid) actions.append(communityAction("上报失效", () => feedbackRoom(room, "invalid"), true));
        else actions.append(communityAction("失效有误", () => feedbackRoom(room, "invalid_wrong")));
        if (room.invalid_pending) actions.append(communityAction("失效有误", () => feedbackRoom(room, "invalid_wrong")));
      }
      if (owners[room.id]) actions.append(communityAction("哎呀输错了！我删！", () => deleteRoom(room), true));
      card.append(header, meta, description, actions);
      root.append(card);
    });
  }
  const expiryCandidates = rooms.flatMap((room) => [
    number(room.expires_at), number(room.invalid_hides_at),
  ]).filter((epoch) => epoch > now);
  if (expiryCandidates.length) {
    const nextExpiry = Math.min(...expiryCandidates);
    state.communityExpiryTimer = setTimeout(
      () => renderCommunityRooms(state.communityData || {}),
      Math.max(50, (nextExpiry - Date.now() / 1000) * 1000 + 50),
    );
  }
}

async function loadCommunityRooms() {
  try { renderCommunityRooms(await fetchJSON("/api/community/rooms")); }
  catch (error) {
    $("community-rooms").innerHTML = "";
    const empty = document.createElement("div"); empty.className = "empty-state"; empty.textContent = error.message;
    $("community-rooms").append(empty);
  }
}

async function submitCommunityRoom(event) {
  event.preventDefault();
  const playerName = $("community-player").value.trim();
  const code = $("community-code").value.trim().toUpperCase();
  if (!/^[A-Z0-9]{6}$/.test(code)) {
    showToast("开门码必须正好是6位数字或英文字母");
    $("community-code").focus();
    return;
  }
  const ownerToken = newOwnerToken();
  try {
    const result = await fetchJSON("/api/community/rooms", {
      method: "POST",
      body: {
        player_name: playerName,
        code,
        room_type: $("community-type").value,
        description: $("community-description").value.trim(),
        owner_token: ownerToken,
      },
    });
    localStorage.setItem("pokopia-community-player-name", playerName);
    saveCommunityOwner(result.room.id, ownerToken);
    $("community-code").value = "";
    $("community-type").value = "stamp";
    $("community-description").value = "";
    showToast("开门信息已发布");
    if (result.community) renderCommunityRooms(result.community);
  } catch (error) { showToast(error.message); }
}

function fillGoodRanking(target, rows = []) {
  const list = $(target);
  list.innerHTML = "";
  if (!rows.length) {
    const empty = document.createElement("li");
    empty.className = "empty-state";
    empty.textContent = "暂无记录";
    list.append(empty);
    return;
  }
  rows.forEach((row) => {
    const item = rankingItem(row, "good");
    const breakdown = document.createElement("div");
    breakdown.className = "room-type-counts";
    const byType = row.by_type || {};
    ["stamp", "task", "flower", "material_solo", "other"].forEach((type) => {
      const span = document.createElement("span");
      span.textContent = `${roomTypeLabel(type)} ${number(byType[type])}次`;
      breakdown.append(span);
    });
    item.append(breakdown);
    list.append(item);
  });
}

async function loadGoodLeaderboard(period = state.goodPeriod, query = null) {
  const generation = ++state.goodGeneration;
  state.goodPeriod = period;
  try {
    const params = new URLSearchParams();
    if (query?.start || query?.end) {
      if (query.start) params.set("start", query.start);
      if (query.end) params.set("end", query.end);
    } else params.set("period", query?.period || period);
    if (query?.player) params.set("player", query.player);
    const data = await fetchJSON(`/api/community/leaderboard?${params}`);
    if (generation !== state.goodGeneration) return;
    const range = data.start ? `${data.start} ～ ${data.end_inclusive}` : "全部业务日期";
    const queryMode = Boolean(query);
    const playerText = data.player_query ? ` · 模糊匹配“${data.player_query}”` : "";
    $(queryMode ? "good-query-range" : "good-range").textContent = `${range}${playerText} · 只统计本网站提交且未自行删除的开门信息`;
    fillGoodRanking(queryMode ? "good-query-ranking" : "good-ranking", data.rankings || []);
  } catch (error) {
    if (generation === state.goodGeneration) $(query ? "good-query-range" : "good-range").textContent = error.message;
  }
}

function selectGoodView(view) {
  state.goodView = view;
  document.querySelectorAll("[data-good-view]").forEach((button) => button.classList.toggle("active", button.dataset.goodView === view));
  const precise = view === "query";
  $("good-ranking-pane").hidden = precise;
  $("good-query-pane").hidden = !precise;
  if (precise) submitGoodQuery();
  else loadGoodLeaderboard(view);
}

function submitGoodQuery(event) {
  event?.preventDefault();
  loadGoodLeaderboard("all", {
    period: $("good-query-period").value,
    start: $("good-query-start").value,
    end: $("good-query-end").value,
    player: $("good-query-player").value.trim(),
  });
}

function selectRankingView(view) {
  state.rankingView = view;
  document.querySelectorAll("[data-ranking-view]").forEach((button) => {
    button.classList.toggle("active", button.dataset.rankingView === view);
  });
  const preciseQuery = view === "query";
  $("ranking-pane").hidden = preciseQuery;
  $("query-pane").hidden = !preciseQuery;
  if (preciseQuery) submitQuery();
  else loadRankings(view);
}

function openDrawer(kind) {
  const drawer = $("side-drawer");
  const metadata = {
    ranking: ["玩家排行榜", "RANKINGS & SEARCH"],
    welfare: ["福利与任务指南", "GUIDE"], community: ["其它玩家开门信息", "COMMUNITY ROOMS"],
    good: ["好人榜", "COMMUNITY THANKS"],
  };
  document.querySelectorAll(".drawer-pane").forEach((pane) => { pane.hidden = pane.id !== `${kind}-pane`; });
  $("ranking-navigation").hidden = kind !== "ranking";
  $("drawer-title").textContent = metadata[kind][0];
  $("drawer-eyebrow").textContent = metadata[kind][1];
  $("drawer-backdrop").hidden = false;
  drawer.classList.add("open");
  drawer.setAttribute("aria-hidden", "false");
  state.activeDrawer = kind;
  ["ranking", "welfare", "community", "good"].forEach((name) => {
    const button = $(`${name === "ranking" ? "rank" : name}-button`);
    if (button) button.classList.toggle("is-active", name === kind);
  });
  if (kind === "ranking") selectRankingView(state.rankingView);
  if (kind === "community" && !state.communityData) loadCommunityRooms();
  if (kind === "good") selectGoodView(state.goodView);
}

function closeDrawer() {
  $("side-drawer").classList.remove("open");
  $("side-drawer").setAttribute("aria-hidden", "true");
  $("drawer-backdrop").hidden = true;
  state.activeDrawer = "";
  document.querySelectorAll(".top-actions .soft-button").forEach((button) => button.classList.remove("is-active"));
}

$("rank-button").addEventListener("click", () => openDrawer("ranking"));
$("welfare-button").addEventListener("click", () => openDrawer("welfare"));
$("community-button").addEventListener("click", () => openDrawer("community"));
$("good-button").addEventListener("click", () => openDrawer("good"));
$("drawer-close").addEventListener("click", closeDrawer);
$("drawer-backdrop").addEventListener("click", closeDrawer);
document.addEventListener("keydown", (event) => { if (event.key === "Escape") closeDrawer(); });
document.querySelectorAll("[data-ranking-view]").forEach((button) => button.addEventListener("click", () => selectRankingView(button.dataset.rankingView)));
document.querySelectorAll("[data-good-view]").forEach((button) => button.addEventListener("click", () => selectGoodView(button.dataset.goodView)));
$("query-form").addEventListener("submit", submitQuery);
$("good-query-form").addEventListener("submit", submitGoodQuery);
$("community-form").addEventListener("submit", submitCommunityRoom);
$("community-player").value = localStorage.getItem("pokopia-community-player-name") || "";
$("community-player").addEventListener("change", () => localStorage.setItem("pokopia-community-player-name", $("community-player").value.trim()));
$("query-period").addEventListener("change", () => { $("query-start").value = ""; $("query-end").value = ""; });
$("good-query-period").addEventListener("change", () => { $("good-query-start").value = ""; $("good-query-end").value = ""; });
$("copy-code").addEventListener("click", async () => {
  const code = String(state.live?.code || "");
  if (!code) return;
  try { await navigator.clipboard.writeText(code); showToast(`已复制 ${code}`); }
  catch { showToast("复制失败，请长按CODE复制"); }
});

if (localPreview) refreshLive();
else connectLiveStream();
setTimeout(() => {
  if (!state.communityData) loadCommunityRooms();
}, localPreview ? 0 : 3000);
setInterval(() => {
  if (!state.liveSocket || state.liveSocket.readyState !== WebSocket.OPEN) refreshLive();
}, localPreview ? 2000 : 60_000);
setInterval(updateCountdown, 1000);
