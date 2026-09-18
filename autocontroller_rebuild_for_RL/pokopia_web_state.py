"""Non-blocking, atomic runtime-state export for the Pokopia public dashboard.

This module deliberately has no web-framework or OpenCV dependency.  Macro6
only updates an in-memory dictionary and occasionally replaces one small JSON
file; the public web server remains a separate process and is strictly read
only.
"""

from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timedelta, timezone
import json
from pathlib import Path
import threading
import time
from typing import Any


BEIJING_TIMEZONE = timezone(timedelta(hours=8))
DEFAULT_STATE_PATH = Path(__file__).resolve().with_name("macro6_web_state.json")


class PokopiaWebStatePublisher:
    """Publish a throttled atomic JSON snapshot without delaying Macro6."""

    def __init__(
        self,
        path: Path = DEFAULT_STATE_PATH,
        *,
        heartbeat_seconds: float = 2.0,
    ) -> None:
        self.path = Path(path)
        self.heartbeat_seconds = max(0.5, float(heartbeat_seconds))
        self.minimum_write_interval = 0.5
        self._lock = threading.Lock()
        self._wake = threading.Event()
        self._stop = threading.Event()
        self._dirty = True
        self._last_write_monotonic = 0.0
        now = datetime.now(BEIJING_TIMEZONE)
        self._state: dict[str, Any] = {
            "schema_version": 1,
            "service": "pokopia_stamp",
            "process_started_at": now.isoformat(timespec="seconds"),
            "updated_at": now.isoformat(timespec="milliseconds"),
            "updated_at_epoch": time.time(),
            "status": {
                "key": "starting",
                "label": "正在启动",
                "tone": "amber",
            },
            "phase": {
                "key": "starting",
                "label": "正在初始化采集、识别与手柄连接",
            },
            "code": "",
            "code_unknown": False,
            "code_revision": 0,
            "macro_key": 0,
            "operation_locked": False,
            "room_disconnected": False,
            "players": [],
            "counts": {},
            "tasks": {"completed": 0, "total": 3},
            "timer": {
                "elapsed_seconds": None,
                "limit_seconds": 900,
                "remaining_seconds": None,
                "deadline": None,
                "active": False,
            },
            "detection": {},
            "announcement": {
                "text": "",
                "round_started_at": None,
                "deadline": None,
                "previous_summary": "暂无上轮完整记录",
                "restart_summary": "正在等待首轮数据",
            },
        }
        self._thread = threading.Thread(
            target=self._writer_loop,
            name="pokopia-web-state-writer",
            daemon=True,
        )
        self._thread.start()
        self.flush(force=True)

    @staticmethod
    def _now_fields() -> tuple[str, float]:
        now = datetime.now(BEIJING_TIMEZONE)
        return now.isoformat(timespec="milliseconds"), time.time()

    def _writer_loop(self) -> None:
        while not self._stop.is_set():
            self._wake.wait(self.heartbeat_seconds)
            self._wake.clear()
            self.flush()

    def update(self, **fields: Any) -> None:
        """Merge top-level fields; nested dictionaries are replaced."""
        with self._lock:
            changed = False
            for key, value in fields.items():
                if self._state.get(key) != value:
                    self._state[key] = deepcopy(value)
                    changed = True
            if changed:
                self._dirty = True
        if changed:
            self._wake.set()

    def set_phase(
        self,
        key: str,
        label: str,
        *,
        status_key: str | None = None,
        status_label: str | None = None,
        tone: str | None = None,
    ) -> None:
        with self._lock:
            self._state["phase"] = {"key": str(key), "label": str(label)}
            if status_key is not None:
                self._state["status"] = {
                    "key": str(status_key),
                    "label": str(status_label or status_key),
                    "tone": str(tone or "amber"),
                }
            self._dirty = True
        self._wake.set()

    def set_announcement(self, text: str) -> None:
        lines = [line.strip() for line in str(text).splitlines() if line.strip()]
        deadline = None
        previous_summary = "暂无上轮完整记录"
        restart_summary = "重开耗时尚未记录"
        for line in lines:
            if line.startswith("预计开到盖满或[") and line.endswith("]"):
                deadline = line.removeprefix("预计开到盖满或[")[:-1]
            elif line.startswith("（上轮") or line.startswith("（日期变更") or line.startswith("（严重超时"):
                previous_summary = line.strip("（）")
            elif line.startswith("（此次"):
                restart_summary = line.strip("（）")
        now = datetime.now(BEIJING_TIMEZONE)
        with self._lock:
            self._state["announcement"] = {
                "text": str(text),
                "round_started_at": now.isoformat(timespec="seconds"),
                "deadline": deadline,
                "previous_summary": previous_summary,
                "restart_summary": restart_summary,
            }
            self._dirty = True
        self._wake.set()

    def flush(self, *, force: bool = False) -> None:
        now_monotonic = time.monotonic()
        with self._lock:
            if (
                not force
                and now_monotonic - self._last_write_monotonic
                < self.minimum_write_interval
            ):
                return
            heartbeat_due = (
                now_monotonic - self._last_write_monotonic
                >= self.heartbeat_seconds
            )
            if not force and not self._dirty and not heartbeat_due:
                return
            updated_at, updated_epoch = self._now_fields()
            self._state["updated_at"] = updated_at
            self._state["updated_at_epoch"] = updated_epoch
            snapshot = deepcopy(self._state)
            self._dirty = False
            self._last_write_monotonic = now_monotonic
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            temporary = self.path.with_suffix(self.path.suffix + ".tmp")
            temporary.write_text(
                json.dumps(snapshot, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
            temporary.replace(self.path)
        except OSError:
            # Dashboard export must never interrupt controller execution.
            with self._lock:
                self._dirty = True

    def close(self, *, clean_exit: bool = True) -> None:
        if clean_exit:
            self.set_phase(
                "offline",
                "watchdog 已停止",
                status_key="offline",
                status_label="离线",
                tone="gray",
            )
            self.flush(force=True)
        self._stop.set()
        self._wake.set()
        self._thread.join(timeout=1.0)
