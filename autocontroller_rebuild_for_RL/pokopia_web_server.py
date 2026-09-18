#!/usr/bin/env python3
"""Read-only public dashboard server for Macro6's live and historical data."""

from __future__ import annotations

import argparse
from datetime import date, datetime, timedelta, timezone
from http import HTTPStatus
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
import json
import mimetypes
from pathlib import Path
import re
import threading
import time
from urllib.parse import parse_qs, unquote, urlparse
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

if __package__:
    from .pokopia_archive_cleanup import cleanup_archives
else:
    from pokopia_archive_cleanup import cleanup_archives


BASE_DIR = Path(__file__).resolve().parent
STATIC_DIR = BASE_DIR / "pokopia_web" / "static"
STATE_PATH = BASE_DIR / "macro6_web_state.json"
STATS_PATH = BASE_DIR / "macro6_watchdog_stats.json"
ARCHIVE_ROOT = BASE_DIR / "pokopia_stamp_records"
CONFIG_PATH = BASE_DIR / "pokopia_web_config.json"
CONFIG_EXAMPLE_PATH = BASE_DIR / "pokopia_web_config.example.json"
BEIJING_TIMEZONE = timezone(timedelta(hours=8))
DEFAULT_CONFIG = {
    "bind": "127.0.0.1",
    "port": 8787,
    "offline_after_seconds": 20,
    "image_retention_days": 30,
    "site_title": "满当当集章的城镇 · 梦幻章车",
    "public_base_url": "https://stamp.rabi.date",
}


def _load_json(path: Path, default):
    try:
        value = json.loads(path.read_text(encoding="utf-8-sig"))
    except (OSError, ValueError, json.JSONDecodeError):
        return default
    return value


def _load_config(path: Path) -> dict[str, object]:
    config = dict(DEFAULT_CONFIG)
    raw = _load_json(path, {})
    if isinstance(raw, dict):
        config.update(raw)
    return config


def _business_day(now: datetime | None = None) -> date:
    current = now or datetime.now(BEIJING_TIMEZONE)
    return (current.astimezone(BEIJING_TIMEZONE) - timedelta(hours=5)).date()


def _friday_week_start(day: date) -> date:
    return day - timedelta(days=(day.weekday() - 4) % 7)


def _next_month(day: date) -> date:
    if day.month == 12:
        return date(day.year + 1, 1, 1)
    return date(day.year, day.month + 1, 1)


def _safe_int(value, default: int = 0) -> int:
    try:
        return max(0, int(value))
    except (TypeError, ValueError):
        return default


def _counter(payload: dict, key: str) -> dict[str, int]:
    raw = payload.get(key, {})
    if not isinstance(raw, dict):
        return {}
    return {str(day): _safe_int(value) for day, value in raw.items()}


def _nested(payload: dict, key: str) -> dict[str, dict[str, int]]:
    raw = payload.get(key, {})
    if not isinstance(raw, dict):
        return {}
    result: dict[str, dict[str, int]] = {}
    for day, values in raw.items():
        if not isinstance(values, dict):
            continue
        result[str(day)] = {
            str(name): _safe_int(count)
            for name, count in values.items()
            if str(name).strip()
        }
    return result


def _parse_date(raw: str, field: str) -> date:
    value = raw.strip()
    if not re.fullmatch(r"\d{4}-?\d{2}-?\d{2}", value):
        raise ValueError(f"{field}必须是 YYYY-MM-DD。")
    return datetime.strptime(value.replace("-", ""), "%Y%m%d").date()


def _range_from_query(query: dict[str, list[str]]) -> tuple[date, date, str]:
    current = _business_day()
    raw_start = query.get("start", [""])[0].strip()
    raw_end = query.get("end", [""])[0].strip()
    period = query.get("period", ["all"])[0].strip().lower()
    raw_anchor = query.get("anchor", [""])[0].strip()
    anchor = _parse_date(raw_anchor, "基准日期") if raw_anchor else current

    if raw_start or raw_end:
        start = _parse_date(raw_start, "开始日期") if raw_start else date.min
        inclusive_end = _parse_date(raw_end, "结束日期") if raw_end else date.max - timedelta(days=1)
        if inclusive_end < start:
            raise ValueError("结束日期不能早于开始日期。")
        return start, inclusive_end + timedelta(days=1), "自定义范围"
    if period == "day":
        return anchor, anchor + timedelta(days=1), "日"
    if period == "week":
        start = _friday_week_start(anchor)
        return start, start + timedelta(days=7), "周"
    if period == "month":
        start = anchor.replace(day=1)
        return start, _next_month(start), "月"
    if period == "year":
        start = date(anchor.year, 1, 1)
        return start, date(anchor.year + 1, 1, 1), "年"
    if period != "all":
        raise ValueError("period只支持 day/week/month/year/all。")
    return date.min, date.max, "全部"


def _day_in_range(raw_day: str, start: date, end: date) -> bool:
    try:
        day = datetime.strptime(raw_day, "%Y-%m-%d").date()
    except ValueError:
        return False
    return start <= day < end


def _name_matches(name: str, needle: str) -> bool:
    return not needle or needle.casefold() in name.casefold()


def _sum_nested(
    daily: dict[str, dict[str, int]],
    start: date,
    end: date,
    needle: str,
) -> tuple[dict[str, int], dict[str, int]]:
    total: dict[str, int] = {}
    by_day: dict[str, int] = {}
    for raw_day, players in daily.items():
        if not _day_in_range(raw_day, start, end):
            continue
        for name, count in players.items():
            if not _name_matches(name, needle):
                continue
            total[name] = total.get(name, 0) + count
            by_day[raw_day] = by_day.get(raw_day, 0) + count
    return total, by_day


def _ranking(counts: dict[str, int], limit: int) -> list[dict[str, object]]:
    rows = sorted(
        ((name, count) for name, count in counts.items() if count > 0),
        key=lambda item: (-item[1], item[0].casefold(), item[0]),
    )
    return [
        {"rank": index, "name": name, "count": count}
        for index, (name, count) in enumerate(rows[:limit], 1)
    ]


class DashboardData:
    def __init__(self, config: dict[str, object]) -> None:
        self.config = config
        self._log_lock = threading.Lock()
        self._live_cache_lock = threading.Lock()
        self._live_cache_at = 0.0
        self._live_cache: dict[str, object] | None = None
        self._screenshot_cache_code = ""
        self._screenshot_cache_path: Path | None = None

    def query(self, query: dict[str, list[str]]) -> dict[str, object]:
        stats = _load_json(STATS_PATH, {})
        if not isinstance(stats, dict):
            stats = {}
        start, end, period_label = _range_from_query(query)
        needle = query.get("player", [""])[0].strip()
        limit = min(1000, max(1, _safe_int(query.get("limit", [10])[0], 10)))

        visits_daily = _nested(stats, "daily_player_visits")
        tasks_daily = _nested(stats, "daily_player_completed_tasks")
        returned_daily = _nested(stats, "daily_player_returned_before_arrival")
        closed_daily = _nested(stats, "daily_player_room_closed_before_arrival")
        network_daily = _nested(
            stats,
            "daily_player_network_error_before_arrival",
        )
        visits, visits_by_day = _sum_nested(visits_daily, start, end, needle)
        tasks, tasks_by_day = _sum_nested(tasks_daily, start, end, needle)
        returned, returned_by_day = _sum_nested(returned_daily, start, end, needle)
        closed, closed_by_day = _sum_nested(closed_daily, start, end, needle)
        network, network_by_day = _sum_nested(
            network_daily,
            start,
            end,
            needle,
        )
        failures = {
            name: (
                returned.get(name, 0)
                + closed.get(name, 0)
                + network.get(name, 0)
            )
            for name in set(returned) | set(closed) | set(network)
        }

        runs = _counter(stats, "daily_runs")
        aggregate_success = _counter(stats, "daily_player_entries")
        aggregate_failure = _counter(stats, "daily_player_entry_failures")
        known_days = sorted(
            day
            for day in (
                set(runs)
                | set(aggregate_success)
                | set(aggregate_failure)
                | set(visits_by_day)
                | set(tasks_by_day)
                | set(returned_by_day)
                | set(closed_by_day)
                | set(network_by_day)
            )
            if _day_in_range(day, start, end)
        )
        daily = []
        for day in known_days:
            success = visits_by_day.get(day, 0) if needle else aggregate_success.get(day, 0)
            failed = (
                returned_by_day.get(day, 0)
                + closed_by_day.get(day, 0)
                + network_by_day.get(day, 0)
                if needle
                else aggregate_failure.get(day, 0)
            )
            daily.append(
                {
                    "date": day,
                    "rounds": runs.get(day, 0),
                    "successful_visits": success,
                    "failed_visits": failed,
                    "task_participations": tasks_by_day.get(day, 0),
                }
            )
        matched_names = sorted(
            {
                name
                for source in (visits, tasks, returned, closed, network)
                for name in source
            },
            key=lambda item: (item.casefold(), item),
        )
        return {
            "period": period_label,
            "start": None if start == date.min else start.isoformat(),
            "end_inclusive": None if end == date.max else (end - timedelta(days=1)).isoformat(),
            "player_query": needle,
            "matched_players": matched_names,
            "summary": {
                "successful_visits": sum(row["successful_visits"] for row in daily),
                "failed_visits": sum(row["failed_visits"] for row in daily),
                "task_participations": sum(row["task_participations"] for row in daily),
            },
            "rankings": {
                "visits": _ranking(visits, limit),
                "tasks": _ranking(tasks, limit),
                "failures": [
                    {
                        **row,
                        "returned_before_arrival": returned.get(str(row["name"]), 0),
                        "room_closed_before_arrival": closed.get(str(row["name"]), 0),
                        "network_error_before_arrival": network.get(
                            str(row["name"]),
                            0,
                        ),
                    }
                    for row in _ranking(failures, limit)
                ],
            },
            "daily": daily,
            "generated_at": datetime.now(BEIJING_TIMEZONE).isoformat(timespec="seconds"),
        }

    @staticmethod
    def _read_recent_jsonl(path: Path, max_bytes: int = 4_000_000) -> list[dict]:
        try:
            with path.open("rb") as source:
                source.seek(0, 2)
                size = source.tell()
                source.seek(max(0, size - max_bytes))
                if size > max_bytes:
                    source.readline()
                raw_lines = source.read().decode("utf-8", errors="replace").splitlines()
        except OSError:
            return []
        records = []
        for line in raw_lines:
            try:
                record = json.loads(line)
            except (ValueError, json.JSONDecodeError):
                continue
            if isinstance(record, dict):
                records.append(record)
        return records

    @staticmethod
    def _record_time(record: dict) -> str:
        raw = (
            record.get("record_time")
            or record.get("recorded_at_beijing")
            or record.get("changed_at_utc")
            or record.get("recorded_at_utc")
            or ""
        )
        value = str(raw)
        if value.endswith("Z"):
            try:
                return datetime.fromisoformat(value[:-1] + "+00:00").astimezone(
                    BEIJING_TIMEZONE
                ).replace(tzinfo=None).isoformat(timespec="milliseconds")
            except ValueError:
                return value
        return value[:23]

    def _latest_code_screenshot_path(
        self,
        code: str,
        *,
        code_unknown: bool = False,
        revision: int = 0,
    ) -> Path | None:
        if (not code and not code_unknown) or not ARCHIVE_ROOT.is_dir():
            return None
        safe_code = (
            "__UNKNOWN__"
            if code_unknown
            else re.sub(r"[^0-9A-Za-z_-]+", "_", code.strip())
        )
        cache_key = f"{safe_code}:{max(0, int(revision))}"
        cached = self._screenshot_cache_path
        if (
            cache_key == self._screenshot_cache_code
            and cached is not None
            and cached.is_file()
        ):
            return cached

        current_folder = ARCHIVE_ROOT / f"STAMP_{_business_day().isoformat()}"
        if code_unknown:
            pattern = "ERROR_*_code_ocr_ten_failures_unknown.png"
            candidates = list(current_folder.glob(pattern))
        else:
            candidates = list(current_folder.glob(f"STAMP_*_{safe_code}.png"))
            if not candidates:
                candidates = list(
                    ARCHIVE_ROOT.glob(f"STAMP_*/STAMP_*_{safe_code}.png")
                )
        if not candidates:
            self._screenshot_cache_code = cache_key
            self._screenshot_cache_path = None
            return None
        latest = max(candidates, key=lambda item: item.stat().st_mtime)
        self._screenshot_cache_code = cache_key
        self._screenshot_cache_path = latest
        return latest

    def current_code_screenshot(self) -> Path | None:
        state = _load_json(STATE_PATH, {})
        code = str(state.get("code", "")) if isinstance(state, dict) else ""
        code_unknown = (
            bool(state.get("code_unknown"))
            if isinstance(state, dict)
            else False
        )
        return self._latest_code_screenshot_path(
            code,
            code_unknown=code_unknown,
            revision=_safe_int(state.get("code_revision")),
        )

    def live(self) -> dict[str, object]:
        now_monotonic = time.monotonic()
        with self._live_cache_lock:
            if self._live_cache is not None and now_monotonic - self._live_cache_at < 0.75:
                return self._live_cache

        state = _load_json(STATE_PATH, {})
        if not isinstance(state, dict):
            state = {}
        updated_epoch = float(state.get("updated_at_epoch") or 0.0)
        stale_seconds = max(0.0, time.time() - updated_epoch) if updated_epoch else None
        offline_after = max(5, _safe_int(self.config.get("offline_after_seconds"), 20))
        if stale_seconds is None or stale_seconds > offline_after:
            state["status"] = {"key": "offline", "label": "离线", "tone": "gray"}
            state["phase"] = {
                "key": "offline",
                "label": "watchdog心跳已中断，页面数据可能过期",
            }
        state["stale_seconds"] = stale_seconds
        state["offline_after_seconds"] = offline_after
        screenshot = self._latest_code_screenshot_path(
            str(state.get("code", "")),
            code_unknown=bool(state.get("code_unknown")),
            revision=_safe_int(state.get("code_revision")),
        )
        state["screenshot_url"] = (
            f"/media/current-code.png?v={screenshot.stat().st_mtime_ns}"
            if screenshot
            else None
        )

        counts = state.get("counts", {}) if isinstance(state.get("counts"), dict) else {}
        business_day = str(counts.get("operational_date") or _business_day().isoformat())
        folder = ARCHIVE_ROOT / f"STAMP_{business_day}"
        records = self._read_recent_jsonl(folder / f"STAMP_{business_day}.jsonl")
        announcement = state.get("announcement", {}) if isinstance(state.get("announcement"), dict) else {}
        round_started = str(announcement.get("round_started_at") or "")[:19]
        round_records = [
            record
            for record in records
            if not round_started or self._record_time(record)[:19] >= round_started
        ]
        task_events = []
        player_events = []
        for record in round_records:
            status = record.get("status")
            when = self._record_time(record)
            if status == "reward_tasks_detected":
                task_events.append(
                    {
                        "time": when,
                        "added": _safe_int(record.get("reward_lines_added")),
                        "completed": _safe_int(record.get("round_tasks_completed")),
                        "total": 3,
                    }
                )
            elif status == "player_room_status":
                player_events.append(
                    {
                        "time": str(record.get("capture_time") or when),
                        "name": str(record.get("player_name") or "未知玩家"),
                        "notification": str(record.get("notification_status") or ""),
                        "room_status": str(record.get("room_status") or "已离开"),
                    }
                )
        state["round_log"] = {
            "tasks": task_events[-20:],
            "players": player_events[-100:],
        }
        summaries = self._read_recent_jsonl(folder / f"PLAYERS_{business_day}.jsonl")
        state["recent_rounds"] = [
            {
                "time": self._record_time(record),
                "reopen_index": record.get("reopen_index_today"),
                "successful": _safe_int(record.get("round_player_entries")),
                "failed": _safe_int(record.get("round_player_entry_failures")),
                "players": record.get("players", []),
                "reopen_reason": str(record.get("reopen_reason") or "normal_reopen"),
                "network_connection_error": (
                    record.get("network_connection_error", {})
                    if isinstance(record.get("network_connection_error"), dict)
                    else {}
                ),
            }
            for record in summaries
            if record.get("status") == "room_reopen_summary"
        ][-8:][::-1]
        with self._live_cache_lock:
            self._live_cache = state
            self._live_cache_at = time.monotonic()
        return state


class PokopiaHandler(SimpleHTTPRequestHandler):
    server_version = "PokopiaStatus/1.0"

    @property
    def dashboard(self) -> DashboardData:
        return self.server.dashboard  # type: ignore[attr-defined]

    def log_message(self, fmt: str, *args) -> None:
        if self.path.startswith("/api/live"):
            return
        print(f"[{datetime.now():%Y-%m-%d %H:%M:%S}] {self.address_string()} {fmt % args}")

    def _security_headers(self) -> None:
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("X-Frame-Options", "DENY")
        self.send_header("Referrer-Policy", "no-referrer")
        self.send_header("Permissions-Policy", "camera=(), microphone=(), geolocation=()")
        self.send_header(
            "Content-Security-Policy",
            "default-src 'self'; img-src 'self' data:; style-src 'self'; "
            "script-src 'self'; connect-src 'self'; base-uri 'none'; frame-ancestors 'none'",
        )

    def _send_json(self, payload: object, status: int = 200) -> None:
        body = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store, max-age=0")
        self._security_headers()
        self.end_headers()
        self.wfile.write(body)

    def _send_file(self, path: Path, *, cache: str) -> None:
        try:
            body = path.read_bytes()
        except OSError:
            self.send_error(HTTPStatus.NOT_FOUND)
            return
        content_type = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", cache)
        self._security_headers()
        self.end_headers()
        self.wfile.write(body)

    def _proxy_community_request(self, method: str) -> None:
        """Let localhost preview use the always-on Cloudflare community API."""
        parsed = urlparse(self.path)
        if not parsed.path.startswith("/api/community/"):
            self.send_error(HTTPStatus.NOT_FOUND)
            return
        base_url = str(
            self.dashboard.config.get("public_base_url")
            or "https://stamp.rabi.date"
        ).rstrip("/")
        body = None
        if method in {"POST", "DELETE"}:
            declared = min(
                32_000,
                max(0, _safe_int(self.headers.get("Content-Length"))),
            )
            body = self.rfile.read(declared)
        headers = {
            "Accept": "application/json",
            "Content-Type": "application/json",
            "X-Pokopia-Client": self.headers.get(
                "X-Pokopia-Client",
                "localhost-preview",
            ),
        }
        request = Request(
            base_url + parsed.path + (f"?{parsed.query}" if parsed.query else ""),
            data=body,
            headers=headers,
            method=method,
        )
        try:
            with urlopen(request, timeout=15) as response:
                payload = response.read()
                status = response.status
        except HTTPError as exc:
            payload = exc.read()
            status = exc.code
        except URLError as exc:
            self._send_json(
                {"error": f"Cloudflare社区接口连接失败：{exc.reason}"},
                HTTPStatus.BAD_GATEWAY,
            )
            return
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(payload)))
        self.send_header("Cache-Control", "no-store, max-age=0")
        self._security_headers()
        self.end_headers()
        self.wfile.write(payload)

    @staticmethod
    def _safe_child(root: Path, raw_relative: str) -> Path | None:
        relative = Path(unquote(raw_relative.replace("\\", "/")))
        if relative.is_absolute() or ".." in relative.parts:
            return None
        candidate = (root / relative).resolve()
        try:
            candidate.relative_to(root.resolve())
        except ValueError:
            return None
        return candidate

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        try:
            if parsed.path.startswith("/api/community/"):
                self._proxy_community_request("GET")
                return
            if parsed.path == "/api/live":
                self._send_json(self.dashboard.live())
                return
            if parsed.path in {"/api/query", "/api/rankings"}:
                query = parse_qs(parsed.query, keep_blank_values=True)
                if parsed.path == "/api/rankings":
                    query["limit"] = ["10"]
                self._send_json(self.dashboard.query(query))
                return
            if parsed.path == "/media/current-code.png":
                media = self.dashboard.current_code_screenshot()
                if media is None:
                    self.send_error(HTTPStatus.NOT_FOUND)
                    return
                self._send_file(media, cache="no-store, max-age=0")
                return
            if parsed.path.startswith("/media/"):
                self.send_error(HTTPStatus.NOT_FOUND)
                return
            relative = "index.html" if parsed.path in {"", "/"} else parsed.path.lstrip("/")
            static = self._safe_child(STATIC_DIR, relative)
            if static is None or not static.is_file():
                static = STATIC_DIR / "index.html"
            self._send_file(static, cache="no-cache" if static.name == "index.html" else "public, max-age=3600")
        except ValueError as exc:
            self._send_json({"error": str(exc)}, HTTPStatus.BAD_REQUEST)
        except Exception as exc:
            self._send_json({"error": f"服务器读取数据失败：{exc}"}, HTTPStatus.INTERNAL_SERVER_ERROR)

    def do_POST(self) -> None:
        self._proxy_community_request("POST")

    def do_DELETE(self) -> None:
        self._proxy_community_request("DELETE")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Pokopia 梦幻章实时状态网页")
    parser.add_argument("--config", type=Path, default=CONFIG_PATH)
    parser.add_argument("--bind", default="")
    parser.add_argument("--port", type=int, default=0)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = _load_config(args.config)
    bind = args.bind or str(config.get("bind") or "127.0.0.1")
    port = args.port or _safe_int(config.get("port"), 8787)
    if not STATIC_DIR.is_dir():
        print(f"网页静态文件不存在：{STATIC_DIR}")
        return 1
    try:
        server = ThreadingHTTPServer((bind, port), PokopiaHandler)
    except OSError as exc:
        print(f"网页服务无法监听 {bind}:{port}：{exc}")
        print("如果8787端口已被旧进程占用，请先关闭旧网页服务后重试。")
        return 1
    server.daemon_threads = True
    server.dashboard = DashboardData(config)  # type: ignore[attr-defined]
    retention_days = max(1, _safe_int(config.get("image_retention_days"), 30))

    def archive_maintenance() -> None:
        while True:
            result = cleanup_archives(ARCHIVE_ROOT, retention_days=retention_days)
            if result.deleted or result.failures:
                print(
                    f"图片保留维护：已删除{result.deleted}张超过{retention_days}天的"
                    f"非ERROR图片，释放{result.freed_bytes / 1024 / 1024:.1f}MB，"
                    f"失败{result.failures}张。"
                )
            time.sleep(86400)

    threading.Thread(
        target=archive_maintenance,
        name="pokopia-image-retention",
        daemon=True,
    ).start()
    shown_host = "127.0.0.1" if bind in {"0.0.0.0", "::"} else bind
    print(
        "Pokopia 实时网页服务已启动（Macro6状态只读；"
        "玩家开门功能由Cloudflare边缘独立处理）。"
    )
    print(f"本机访问：http://{shown_host}:{port}/")
    print(f"监听地址：{bind}:{port}")
    print("按 Ctrl+C 停止网页服务。")
    try:
        server.serve_forever(poll_interval=0.5)
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
