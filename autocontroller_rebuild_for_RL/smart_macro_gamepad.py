"""Video-aware macro gamepad built on top of ``macro_gamepad``.

Unlike the Tableturf auto-battle runtime, this module keeps the macro-gamepad
execution model and only adds a video-observation thread plus an explicit
state machine.  SmartMacro1's game-specific state rules are intentionally
empty until their visual conditions and actions are defined.
"""

from __future__ import annotations

import argparse
import contextlib
import sys
import threading
import time
import urllib.request
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import List

import cv2
import numpy as np


REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from autocontroller_rebuild_for_RL.macro_gamepad import (
    DEFAULT_CONFIG,
    SerialRemoteController,
    _MacroContext,
    _listen_for_quit,
    _load_config,
    _resolve_config_path,
    _run_controller_detection,
    wait_for_serial_selection,
)
from autocontroller_rebuild_for_RL.runtime import FrameApiAutoLauncher, load_config


class SmartMacroState(str, Enum):
    WAITING_FOR_VIDEO = "waiting_for_video"
    VIDEO_READY = "video_ready"


@dataclass(frozen=True)
class FrameSnapshot:
    sequence: int
    captured_at: float
    frame: np.ndarray | None
    error: str = ""


class LatestFrameObserver:
    """Continuously decode the newest frame without blocking controller input."""

    def __init__(
        self,
        frame_url: str,
        stop_event: threading.Event,
        poll_seconds: float,
    ) -> None:
        self.frame_url = frame_url
        self.stop_event = stop_event
        self.poll_seconds = max(0.03, float(poll_seconds))
        self._opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
        self._lock = threading.Lock()
        self._snapshot = FrameSnapshot(0, 0.0, None)

    def snapshot(self) -> FrameSnapshot:
        with self._lock:
            return self._snapshot

    def _store(self, frame: np.ndarray | None, error: str = "") -> None:
        with self._lock:
            self._snapshot = FrameSnapshot(
                sequence=self._snapshot.sequence + 1,
                captured_at=time.monotonic(),
                frame=frame,
                error=error,
            )

    def run(self, worker_errors: List[BaseException]) -> None:
        try:
            while not self.stop_event.is_set():
                try:
                    request = urllib.request.Request(
                        self.frame_url,
                        headers={"Cache-Control": "no-cache"},
                    )
                    with self._opener.open(request, timeout=2.0) as response:
                        encoded = np.frombuffer(response.read(), dtype=np.uint8)
                    frame = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
                    if frame is None or frame.size == 0:
                        raise RuntimeError("视频接口返回的图像无法解码")
                    self._store(frame)
                except Exception as exc:
                    self._store(None, str(exc))
                self.stop_event.wait(self.poll_seconds)
        except BaseException as exc:
            worker_errors.append(exc)
            self.stop_event.set()


class SmartMacro1:
    """State machine for SmartMacro1.

    Add game-specific image judgments in ``detect_state`` and controller
    actions in ``run_state``.  The shared ``_MacroContext`` already supports
    held buttons, simultaneous buttons, taps, waits, and four stick directions.
    """

    def __init__(self) -> None:
        self.state = SmartMacroState.WAITING_FOR_VIDEO

    def detect_state(self, snapshot: FrameSnapshot) -> SmartMacroState:
        if snapshot.frame is None:
            return SmartMacroState.WAITING_FOR_VIDEO
        return SmartMacroState.VIDEO_READY

    def run_state(
        self,
        state: SmartMacroState,
        context: _MacroContext,
        snapshot: FrameSnapshot,
    ) -> bool:
        del snapshot
        if state is SmartMacroState.WAITING_FOR_VIDEO:
            return context.wait_ms(100)

        # SmartMacro1 的具体画面状态与按键动作将在这里逐步加入。
        return context.wait_ms(100)

    def run(self, context: _MacroContext, observer: LatestFrameObserver) -> None:
        while not context.stop_event.is_set():
            snapshot = observer.snapshot()
            next_state = self.detect_state(snapshot)
            if next_state is not self.state:
                self.state = next_state
                print(f"[SmartMacro1] 状态切换：{self.state.value}")
            if not self.run_state(self.state, context, snapshot):
                return


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Video-aware state-machine macro gamepad (SmartMacro1)"
    )
    parser.add_argument("--config", default=DEFAULT_CONFIG, help="config JSON path")
    parser.add_argument("--baudrate", type=int, default=9600)
    parser.add_argument("--scan-interval", type=float, default=1.0)
    parser.add_argument("--probe-timeout", type=float, default=1.2)
    parser.add_argument(
        "--vision-interval",
        type=float,
        default=0.10,
        help="seconds between video-state observations",
    )
    return parser.parse_args()


def run_smart_macro_forever(
    controller: SerialRemoteController,
    frame_url: str,
    vision_interval: float,
) -> None:
    stop_event = threading.Event()
    worker_errors: List[BaseException] = []
    observer = LatestFrameObserver(frame_url, stop_event, vision_interval)
    context = _MacroContext(controller, stop_event)
    smart_macro = SmartMacro1()

    def controller_worker() -> None:
        try:
            if _run_controller_detection(context):
                smart_macro.run(context, observer)
        except BaseException as exc:
            worker_errors.append(exc)
            stop_event.set()

    workers = [
        threading.Thread(
            target=observer.run,
            args=(worker_errors,),
            name="smart-macro-video-observer",
            daemon=True,
        ),
        threading.Thread(
            target=controller_worker,
            name="smart-macro-controller",
            daemon=True,
        ),
        threading.Thread(
            target=_listen_for_quit,
            args=(stop_event, worker_errors),
            name="smart-macro-quit-listener",
            daemon=True,
        ),
    ]
    for worker in workers:
        worker.start()

    try:
        while not stop_event.wait(0.2):
            pass
    finally:
        stop_event.set()
        for worker in workers:
            worker.join(timeout=2.0)

    if worker_errors:
        raise worker_errors[0]


def main() -> int:
    args = _parse_args()
    config_path = _resolve_config_path(args.config)
    raw_config = _load_config(config_path)
    runtime_config = load_config(config_path)
    frame_url = str(
        raw_config.get("frame_api_url")
        or runtime_config.frame_api_url
        or ""
    ).strip()
    if not frame_url:
        print("启动失败：config 中缺少 frame_api_url。", file=sys.stderr)
        return 1

    launcher = FrameApiAutoLauncher(runtime_config)
    controller = None
    try:
        launcher.ensure_started()
        selected_port = wait_for_serial_selection(
            config_path=config_path,
            baudrate=args.baudrate,
            probe_timeout=max(0.1, args.probe_timeout),
            scan_interval=max(0.1, args.scan_interval),
        )
        controller = SerialRemoteController(
            port=selected_port,
            baudrate=args.baudrate,
            timeout=0.1,
        )
        print(
            f"智能宏手柄已启动：{selected_port}；视频：{frame_url}。"
            "手柄检测只执行一次；按 Q 或 Ctrl+C 退出。"
        )
        run_smart_macro_forever(
            controller=controller,
            frame_url=frame_url,
            vision_interval=args.vision_interval,
        )
        return 0
    except KeyboardInterrupt:
        return 0
    except Exception as exc:
        print(f"启动失败：{exc}", file=sys.stderr)
        return 1
    finally:
        if controller is not None:
            with contextlib.suppress(Exception):
                controller.release()
            with contextlib.suppress(Exception):
                controller.close()
        launcher.stop()


if __name__ == "__main__":
    raise SystemExit(main())
