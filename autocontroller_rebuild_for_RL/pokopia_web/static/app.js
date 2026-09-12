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
  countdownRemaining: null,
  countdownSampledAt: 0,
  toastTimer: null,
  lastRequestAt: new Map(),
  liveSocket: null,
  reconnectTimer: null,
  queryGeneration: 0,
  rankingGeneration: 0,
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
    root.append(card);
  });
}

function renderLive(data) {
  state.live = data;
  setStatus(data.status);
  $("phase-label").textContent = data.phase?.label || "等待状态更新";
  const code = String(data.code || "");
  $("code-value").textContent = code || "------";
  $("copy-code").disabled = !code;
  const counts = data.counts || {};
  $("round-label").textContent = code ? `今日第 ${number(counts.daily_runs)} 轮 · 总第 ${number(counts.total_runs)} 轮` : "等待新轮次";

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
  const timerRemaining = Number(data.timer?.remaining_seconds);
  state.countdownRemaining = data.timer?.active && Number.isFinite(timerRemaining)
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
  const offlineAfter = number(state.live?.offline_after_seconds) || 30;
  if (receivedAt && Date.now() / 1000 - receivedAt > offlineAfter && state.live?.status?.key !== "offline") {
    state.live.status = { key: "offline", label: "离线" };
    state.live.phase = { key: "offline", label: "Windows状态心跳已中断，当前内容可能过期" };
    setStatus(state.live.status);
    $("phase-label").textContent = state.live.phase.label;
  }
  if (state.countdownRemaining == null || state.live?.status?.key !== "open") {
    $("countdown").textContent = state.live?.status?.key === "reopening" ? "重开中" : "--:--";
    return;
  }
  const sinceSnapshot = Math.max(0, (performance.now() - state.countdownSampledAt) / 1000);
  const remaining = Math.max(0, Math.ceil(state.countdownRemaining - sinceSnapshot));
  const minutes = Math.floor(remaining / 60);
  const seconds = remaining % 60;
  $("countdown").textContent = `${String(minutes).padStart(2, "0")}:${String(seconds).padStart(2, "0")}`;
}

async function fetchJSON(url) {
  const bucket = new URL(url, location.href).pathname;
  const now = performance.now();
  const reservedAt = Math.max(now, (state.lastRequestAt.get(bucket) || -Infinity) + 1000);
  state.lastRequestAt.set(bucket, reservedAt);
  const wait = Math.max(0, reservedAt - now);
  if (wait) await new Promise((resolve) => setTimeout(resolve, wait));
  const response = await fetch(url, {
    cache: "no-store",
    headers: { "X-Pokopia-Client": clientId },
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
  socket.addEventListener("message", (event) => {
    try {
      renderLive(JSON.parse(event.data));
      updateCountdown();
    } catch {
      // A malformed update is ignored; the next server snapshot will replace it.
    }
  });
  socket.addEventListener("close", () => {
    if (state.liveSocket === socket) state.liveSocket = null;
    state.reconnectTimer = setTimeout(connectLiveStream, 2000);
  });
  socket.addEventListener("error", () => socket.close());
}

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
    note.textContent = `到达前返回 ${number(row.returned_before_arrival)} · 重开冻结 ${number(row.room_closed_before_arrival)}`;
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
  document.querySelectorAll("[data-period]").forEach((button) => button.classList.toggle("active", button.dataset.period === period));
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

function openDrawer(kind) {
  const drawer = $("side-drawer");
  const ranking = kind === "ranking";
  $("ranking-pane").hidden = !ranking;
  $("query-pane").hidden = ranking;
  $("drawer-title").textContent = ranking ? "玩家排行榜" : "历史数据查询";
  $("drawer-eyebrow").textContent = ranking ? "TOP 10" : "HISTORY SEARCH";
  $("drawer-backdrop").hidden = false;
  drawer.classList.add("open");
  drawer.setAttribute("aria-hidden", "false");
  $("rank-button").classList.toggle("is-active", ranking);
  $("query-button").classList.toggle("is-active", !ranking);
  if (ranking) loadRankings(); else submitQuery();
}

function closeDrawer() {
  $("side-drawer").classList.remove("open");
  $("side-drawer").setAttribute("aria-hidden", "true");
  $("drawer-backdrop").hidden = true;
  $("rank-button").classList.remove("is-active");
  $("query-button").classList.remove("is-active");
}

$("rank-button").addEventListener("click", () => openDrawer("ranking"));
$("query-button").addEventListener("click", () => openDrawer("query"));
$("drawer-close").addEventListener("click", closeDrawer);
$("drawer-backdrop").addEventListener("click", closeDrawer);
document.addEventListener("keydown", (event) => { if (event.key === "Escape") closeDrawer(); });
document.querySelectorAll("[data-period]").forEach((button) => button.addEventListener("click", () => loadRankings(button.dataset.period)));
$("query-form").addEventListener("submit", submitQuery);
$("query-period").addEventListener("change", () => { $("query-start").value = ""; $("query-end").value = ""; });
$("copy-code").addEventListener("click", async () => {
  const code = String(state.live?.code || "");
  if (!code) return;
  try { await navigator.clipboard.writeText(code); showToast(`已复制 ${code}`); }
  catch { showToast("复制失败，请长按CODE复制"); }
});

refreshLive();
if (!localPreview) connectLiveStream();
setInterval(() => {
  if (!state.liveSocket || state.liveSocket.readyState !== WebSocket.OPEN) refreshLive();
}, localPreview ? 2000 : 10000);
setInterval(updateCountdown, 1000);
