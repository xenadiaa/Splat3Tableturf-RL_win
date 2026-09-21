#!/usr/bin/env python3
"""Push Macro6's public projection to Cloudflare without exposing Windows."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
import threading
import time
from urllib.error import HTTPError, URLError
from urllib.parse import urljoin
from urllib.request import Request, urlopen

if __package__:
    from .pokopia_edge_config import CONFIG_PATH, EdgeConfig, load_edge_config
else:
    from pokopia_edge_config import CONFIG_PATH, EdgeConfig, load_edge_config


BASE_DIR = Path(__file__).resolve().parent
STATS_PATH = BASE_DIR / "macro6_watchdog_stats.json"
SYNC_PATH = BASE_DIR / "pokopia_edge_sync_state.json"
MAX_SCREENSHOT_BYTES = 8_000_000


def _load_json(path: Path, default):
    try:
        return json.loads(path.read_text(encoding="utf-8-sig"))
    except (OSError, ValueError, json.JSONDecodeError):
        return default


def _stable_bytes(payload) -> bytes:
    return json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")


def _count_map(stats: dict, key: str) -> dict[str, int]:
    raw = stats.get(key, {})
    if not isinstance(raw, dict):
        return {}
    result = {}
    for day, value in raw.items():
        try:
            result[str(day)] = max(0, int(value))
        except (TypeError, ValueError):
            continue
    return result


def _nested_map(stats: dict, key: str) -> dict[str, dict[str, int]]:
    raw = stats.get(key, {})
    if not isinstance(raw, dict):
        return {}
    result = {}
    for day, rows in raw.items():
        if not isinstance(rows, dict):
            continue
        normalized = {}
        for name, value in rows.items():
            if not str(name).strip():
                continue
            try:
                normalized[str(name)] = max(0, int(value))
            except (TypeError, ValueError):
                continue
        result[str(day)] = normalized
    return result


def public_days(stats: dict) -> list[dict]:
    runs = _count_map(stats, "daily_runs")
    entries = _count_map(stats, "daily_player_entries")
    failures = _count_map(stats, "daily_player_entry_failures")
    visits = _nested_map(stats, "daily_player_visits")
    tasks = _nested_map(stats, "daily_player_completed_tasks")
    returned = _nested_map(stats, "daily_player_returned_before_arrival")
    closed = _nested_map(stats, "daily_player_room_closed_before_arrival")
    network = _nested_map(stats, "daily_player_network_error_before_arrival")
    days = sorted(set(runs) | set(entries) | set(failures) | set(visits) | set(tasks) | set(returned) | set(closed) | set(network))
    payloads = []
    for day in days:
        names = sorted(set(visits.get(day, {})) | set(tasks.get(day, {})) | set(returned.get(day, {})) | set(closed.get(day, {})) | set(network.get(day, {})))
        players = [
            {
                "name": name,
                "visits": visits.get(day, {}).get(name, 0),
                "tasks": tasks.get(day, {}).get(name, 0),
                "returned_before_arrival": returned.get(day, {}).get(name, 0),
                "room_closed_before_arrival": closed.get(day, {}).get(name, 0),
                "network_error_before_arrival": network.get(day, {}).get(name, 0),
            }
            for name in names
        ]
        payloads.append({
            "date": day,
            "rounds": runs.get(day, 0),
            "successful_visits": entries.get(day, 0),
            "failed_visits": failures.get(day, 0),
            "task_participations": sum(tasks.get(day, {}).values()),
            "players": players,
        })
    return payloads


class EdgeUploader:
    def __init__(self, config: EdgeConfig) -> None:
        self.config = config
        self.stop_event = threading.Event()
        self._last_live_hash = ""
        self._last_live_upload = 0.0
        self._last_screenshot_revision = -1
        self._last_screenshot_relative = ""
        self._timer_deadline_epoch: int | None = None
        checkpoint = _load_json(SYNC_PATH, {})
        self._day_hashes = checkpoint.get("day_hashes", {}) if isinstance(checkpoint, dict) else {}
        if not isinstance(self._day_hashes, dict):
            self._day_hashes = {}

    def _request(self, path: str, *, body: bytes | None = None, content_type: str = "application/json", headers: dict | None = None, method: str | None = None):
        request_headers = {
            "Authorization": f"Bearer {self.config.upload_token}",
            "User-Agent": "Pokopia-Windows-Uploader/1.0",
        }
        if body is not None:
            request_headers["Content-Type"] = content_type
            request_headers["Content-Length"] = str(len(body))
        if headers:
            request_headers.update(headers)
        request = Request(
            self.config.base_url + path,
            data=body,
            headers=request_headers,
            method=method or ("POST" if body is not None else "GET"),
        )
        with urlopen(request, timeout=15) as response:
            payload = response.read()
            return json.loads(payload) if payload else {}

    def _local_json(self, path: str):
        with urlopen(self.config.local_url + path, timeout=5) as response:
            return json.loads(response.read())

    def _publish_screenshot(self, live: dict) -> bool:
        revision = int(live.get("code_revision") or 0)
        relative = str(live.get("screenshot_url") or "")
        code = str(live.get("code") or "")
        if (
            revision == self._last_screenshot_revision
            and relative == self._last_screenshot_relative
        ):
            return bool(relative)
        if not relative:
            self._request("/api/publish/screenshot", method="DELETE")
            self._last_screenshot_revision = revision
            self._last_screenshot_relative = relative
            return False
        with urlopen(urljoin(self.config.local_url + "/", relative.lstrip("/")), timeout=10) as response:
            image = response.read(MAX_SCREENSHOT_BYTES + 1)
        if len(image) > MAX_SCREENSHOT_BYTES:
            raise RuntimeError("当前CODE截图超过8MB，未上传。")
        self._request(
            "/api/publish/screenshot",
            body=image,
            content_type="image/png",
            headers={"X-Pokopia-Code": code, "X-Pokopia-Revision": str(revision)},
        )
        self._last_screenshot_revision = revision
        self._last_screenshot_relative = relative
        print(f"边缘截图已更新：CODE={code or '<空>'}，revision={revision}", flush=True)
        return True

    def _public_live_projection(self, live: dict, screenshot_ready: bool) -> dict:
        """Return the event-driven public state used for hashing and upload.

        Local heartbeat timestamps, recognition-frame details and ticking timer
        counters are intentionally omitted.  They used to make an unchanged
        room look different every two seconds and therefore forced a cloud
        write on every local poll.
        """
        public_live = copy.deepcopy(live)
        for key in ("updated_at", "updated_at_epoch", "stale_seconds"):
            public_live.pop(key, None)
        public_live.pop("detection", None)

        timer = public_live.get("timer")
        if isinstance(timer, dict):
            active = bool(timer.get("active"))
            limit_seconds = max(0, int(timer.get("limit_seconds") or 0))
            remaining_raw = timer.get("remaining_seconds")
            try:
                remaining = max(0.0, float(remaining_raw))
            except (TypeError, ValueError):
                remaining = None
            if active and remaining is not None:
                candidate = round(time.time() + remaining)
                if (
                    self._timer_deadline_epoch is None
                    or candidate < self._timer_deadline_epoch - 3
                    or (
                        candidate > self._timer_deadline_epoch + 3
                        and remaining >= max(0, limit_seconds - 5)
                    )
                ):
                    self._timer_deadline_epoch = candidate
            else:
                self._timer_deadline_epoch = None
            public_live["timer"] = {
                "active": active,
                "limit_seconds": limit_seconds,
                "deadline_epoch": self._timer_deadline_epoch,
            }

        public_live["screenshot_url"] = (
            "/media/current-code.png" if screenshot_ready else None
        )
        return public_live

    def publish_live_once(self, *, publish_screenshot: bool = True) -> None:
        live = self._local_json("/api/live")
        screenshot_ready = (
            self._publish_screenshot(live)
            if publish_screenshot
            else bool(live.get("screenshot_url"))
        )
        public_live = self._public_live_projection(live, screenshot_ready)
        encoded = _stable_bytes(public_live)
        digest = hashlib.sha256(encoded).hexdigest()
        now = time.monotonic()
        if digest == self._last_live_hash and now - self._last_live_upload < self.config.heartbeat_seconds:
            return
        self._request("/api/publish/live", body=encoded)
        self._last_live_hash = digest
        self._last_live_upload = now

    def publish_stats_once(self) -> None:
        stats = _load_json(STATS_PATH, {})
        if not isinstance(stats, dict):
            return
        changed = False
        for day in public_days(stats):
            encoded = _stable_bytes(day)
            digest = hashlib.sha256(encoded).hexdigest()
            key = str(day["date"])
            if self._day_hashes.get(key) == digest:
                continue
            self._request("/api/publish/day", body=encoded)
            self._day_hashes[key] = digest
            changed = True
            print(f"边缘历史统计已同步：{key}，玩家{len(day['players'])}名。", flush=True)
        if changed:
            temporary = SYNC_PATH.with_suffix(SYNC_PATH.suffix + ".tmp")
            temporary.write_text(
                json.dumps({"day_hashes": self._day_hashes}, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
            temporary.replace(SYNC_PATH)

    def run(self) -> int:
        print(f"Pokopia边缘上传已启动：{self.config.base_url}", flush=True)
        next_stats_at = 0.0
        backoff = 1.0
        while not self.stop_event.is_set():
            try:
                self.publish_live_once()
                now = time.monotonic()
                if now >= next_stats_at:
                    self.publish_stats_once()
                    next_stats_at = now + self.config.stats_interval_seconds
                backoff = 1.0
                self.stop_event.wait(self.config.live_interval_seconds)
            except (HTTPError, URLError, OSError, ValueError, RuntimeError) as exc:
                print(f"边缘上传暂时失败：{exc}；{backoff:.0f}秒后重试。", flush=True)
                self.stop_event.wait(backoff)
                backoff = min(30.0, backoff * 2.0)
        return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--once-no-media",
        action="store_true",
        help="立即上传一次实时状态，但不重复上传CODE截图。",
    )
    args = parser.parse_args()
    if not CONFIG_PATH.is_file():
        print("尚未配置Cloudflare边缘上传；本地网页和Macro6不受影响。")
        return 2
    try:
        config = load_edge_config()
    except Exception as exc:
        print(f"边缘上传配置无效：{exc}")
        return 2
    uploader = EdgeUploader(config)
    if args.once_no_media:
        uploader.publish_live_once(publish_screenshot=False)
        print("边缘最终状态已同步。", flush=True)
        return 0
    return uploader.run()


if __name__ == "__main__":
    raise SystemExit(main())
