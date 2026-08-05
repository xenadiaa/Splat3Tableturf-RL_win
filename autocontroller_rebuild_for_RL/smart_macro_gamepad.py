"""Video-aware macro gamepad built on top of ``macro_gamepad``.

SmartMacro1 reuses the Tableturf frame API and macro5's round/sell-equipment
workflow, but waits for four stable white HUD icons before finishing a round.
"""

from __future__ import annotations

import argparse
import contextlib
import os
import sys
import threading
import time
import urllib.request
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import List

if os.name == "nt":
    import msvcrt
else:
    import select
    import termios
    import tty

import cv2
import numpy as np


REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from autocontroller_rebuild_for_RL.macro_gamepad import (
    DEFAULT_CONFIG,
    SerialRemoteController,
    _MacroContext,
    _PauseState,
    _commit_status_line,
    _load_config,
    _resolve_config_path,
    _run_controller_detection,
    _timestamped_log,
    wait_for_serial_selection,
)
from autocontroller_rebuild_for_RL.runtime import FrameApiAutoLauncher, load_config
from switch_connect.virtual_gamepad.input_mapper import (
    BIT_A,
    BIT_B,
    BIT_L,
    BIT_LSTICK_DOWN,
    BIT_LSTICK_LEFT,
    BIT_LSTICK_RIGHT,
    BIT_LSTICK_UP,
    BIT_MINUS,
    BIT_PLUS,
    BIT_R,
    BIT_X,
    BIT_Y,
    BIT_ZL,
    BIT_ZR,
)


_MANUAL_KEY_BITS = {
    "z": BIT_A,
    "x": BIT_B,
    "a": BIT_Y,
    "s": BIT_X,
    "c": BIT_L,
    "v": BIT_R,
    "f": BIT_ZL,
    "g": BIT_ZR,
    "d": BIT_PLUS,
    "e": BIT_MINUS,
    "+": BIT_PLUS,
    "=": BIT_PLUS,
    "-": BIT_MINUS,
}
_MANUAL_ARROW_BITS = {
    "UP": BIT_LSTICK_UP,
    "DOWN": BIT_LSTICK_DOWN,
    "LEFT": BIT_LSTICK_LEFT,
    "RIGHT": BIT_LSTICK_RIGHT,
}
class SmartMacroState(str, Enum):
    WAITING_FOR_VIDEO = "waiting_for_video"
    ENTERING_MAP = "entering_map"
    WAITING_FOR_FOUR_WHITE_ICONS = "waiting_for_four_white_icons"
    FINISHING_ROUND = "finishing_round"


@dataclass(frozen=True)
class FrameSnapshot:
    sequence: int
    captured_at: float
    frame: np.ndarray | None
    error: str = ""


@dataclass(frozen=True)
class WhiteIconDetection:
    detected: bool
    aligned_icon_count: int
    candidate_count: int


def detect_four_white_top_left_icons(frame: np.ndarray) -> WhiteIconDetection:
    """Detect four similarly sized, horizontally aligned pure-white HUD glyphs."""
    if frame is None or frame.size == 0 or frame.ndim != 3:
        return WhiteIconDetection(False, 0, 0)

    height, width = frame.shape[:2]
    # 参考 1920x1080 截图：四个图标位于约 x=25..220、y=28..75。
    roi_x0 = max(0, int(width * 0.005))
    roi_x1 = max(roi_x0 + 1, int(width * 0.13))
    roi_y0 = max(0, int(height * 0.01))
    roi_y1 = max(roi_y0 + 1, int(height * 0.09))
    roi = frame[roi_y0:roi_y1, roi_x0:roi_x1]

    # 采集卡截图中的白色图标边缘最低约为 205，三个通道同时过阈值可排除彩色高光。
    white_mask = np.all(roi >= 205, axis=2).astype(np.uint8) * 255
    close_size = max(3, int(round(min(width, height) / 360.0)))
    if close_size % 2 == 0:
        close_size += 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_size, close_size))
    white_mask = cv2.morphologyEx(white_mask, cv2.MORPH_CLOSE, kernel)

    count, _, stats, centroids = cv2.connectedComponentsWithStats(
        white_mask,
        connectivity=8,
    )
    frame_area = float(width * height)
    candidates: list[tuple[float, float, int, int, int]] = []
    for label in range(1, count):
        x, y, component_width, component_height, area = (
            int(value) for value in stats[label]
        )
        del x, y
        if not (frame_area * 0.00015 <= area <= frame_area * 0.0008):
            continue
        if not (width * 0.014 <= component_width <= width * 0.045):
            continue
        if not (height * 0.022 <= component_height <= height * 0.075):
            continue
        aspect_ratio = component_width / max(1.0, float(component_height))
        fill_ratio = area / max(1.0, float(component_width * component_height))
        if not (0.70 <= aspect_ratio <= 1.35 and 0.15 <= fill_ratio <= 0.65):
            continue
        center_x, center_y = (float(value) for value in centroids[label])
        candidates.append(
            (
                center_x + roi_x0,
                center_y + roi_y0,
                component_width,
                component_height,
                area,
            )
        )

    # 四个 HUD 图案应在近似同一横排，且尺寸不能相差悬殊。
    best_aligned_count = 0
    y_tolerance = max(4.0, height * 0.012)
    for anchor in candidates:
        row = [item for item in candidates if abs(item[1] - anchor[1]) <= y_tolerance]
        if len(row) < 4:
            best_aligned_count = max(best_aligned_count, len(row))
            continue
        row.sort(key=lambda item: item[0])
        for start in range(0, len(row) - 3):
            group = row[start : start + 4]
            areas = [item[4] for item in group]
            heights = [item[3] for item in group]
            if max(areas) > min(areas) * 4.0:
                continue
            if max(heights) > min(heights) * 2.5:
                continue
            gaps = [group[index + 1][0] - group[index][0] for index in range(3)]
            if not all(width * 0.015 <= gap <= width * 0.045 for gap in gaps):
                continue
            best_aligned_count = max(best_aligned_count, 4)
    return WhiteIconDetection(
        detected=best_aligned_count >= 4,
        aligned_icon_count=best_aligned_count,
        candidate_count=len(candidates),
    )


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


class StableWhiteIconWatcher:
    """Continuously detect two consecutive matching frames during controller input."""

    def __init__(self, observer: LatestFrameObserver, context: _MacroContext) -> None:
        self.observer = observer
        self.context = context
        self.detected = threading.Event()
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        minimum_sequence = self.observer.snapshot().sequence

        def worker() -> None:
            last_sequence = minimum_sequence
            stable_matches = 0
            while not self._stop_event.is_set() and not self.context.stop_event.is_set():
                snapshot = self.observer.snapshot()
                if self.context.is_paused():
                    stable_matches = 0
                elif snapshot.sequence > minimum_sequence and snapshot.sequence != last_sequence:
                    last_sequence = snapshot.sequence
                    if snapshot.frame is None:
                        stable_matches = 0
                    else:
                        result = detect_four_white_top_left_icons(snapshot.frame)
                        stable_matches = stable_matches + 1 if result.detected else 0
                        if stable_matches >= 2:
                            self.detected.set()
                            return
                self._stop_event.wait(0.03)

        self._thread = threading.Thread(
            target=worker,
            name="smart-macro-white-icon-watcher",
            daemon=True,
        )
        self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=0.5)


class SmartMacro1:
    """Macro5 workflow gated by four stable white top-left HUD icons."""

    def __init__(self) -> None:
        self.state = SmartMacroState.WAITING_FOR_VIDEO

    def _set_state(self, state: SmartMacroState) -> None:
        if state is self.state:
            return
        self.state = state
        _timestamped_log(f"smart macro1 状态切换：{state.value}")

    def _wait_for_video(self, context: _MacroContext, observer: LatestFrameObserver) -> bool:
        self._set_state(SmartMacroState.WAITING_FOR_VIDEO)
        last_error = ""
        while not context.stop_event.is_set():
            snapshot = observer.snapshot()
            if snapshot.frame is not None:
                return True
            if snapshot.error and snapshot.error != last_error:
                last_error = snapshot.error
                _timestamped_log(f"smart macro1 等待视频：{last_error}")
            if not context.wait_ms(100):
                return False
        return False

    @staticmethod
    def _run_first_round_setup(context: _MacroContext) -> bool:
        # 手柄检测后、第一轮前：Y、A、等待 10 秒，再向后推动摇杆 5 秒。
        if not context.tap(BIT_Y, hold_ms=50, gap_ms=500):
            return False
        if not context.tap(BIT_A, hold_ms=50, gap_ms=500):
            return False
        if not context.wait_ms(10000):
            return False
        if not context.move_stick(BIT_LSTICK_DOWN, duration_ms=5000):
            return False
        context.center_stick()
        return True

    @staticmethod
    def _run_enter_map(context: _MacroContext) -> bool:
        for bit_index, gap_ms in (
            (BIT_X, 500),
            (BIT_A, 500),
            (BIT_A, 500),
            (BIT_A, 500),
            (BIT_A, 500),
            (BIT_A, 500),
            (BIT_A, 500),
        ):
            if not context.tap(bit_index, hold_ms=50, gap_ms=gap_ms):
                return False
        return True

    @staticmethod
    def _wait_or_detect(
        context: _MacroContext,
        watcher: StableWhiteIconWatcher,
        duration_ms: int,
    ) -> tuple[bool, bool]:
        remaining_ms = max(0, int(duration_ms))
        while remaining_ms > 0:
            if watcher.detected.is_set():
                return True, True
            chunk_ms = min(50, remaining_ms)
            if not context.wait_ms(chunk_ms):
                return False, False
            remaining_ms -= chunk_ms
        return True, watcher.detected.is_set()

    @classmethod
    def _tap_or_detect(
        cls,
        context: _MacroContext,
        watcher: StableWhiteIconWatcher,
        bit_index: int,
        hold_ms: int,
        gap_ms: int,
    ) -> tuple[bool, bool]:
        if watcher.detected.is_set():
            return True, True
        context.set_held(bit_index, True)
        ok, detected = cls._wait_or_detect(context, watcher, hold_ms)
        context.set_held(bit_index, False)
        if not ok or detected:
            return ok, detected
        return cls._wait_or_detect(context, watcher, gap_ms)

    @classmethod
    def _run_first_round_attempt(
        cls,
        context: _MacroContext,
        watcher: StableWhiteIconWatcher,
    ) -> tuple[bool, bool]:
        # 每次尝试：额外 A4秒 → 进入地图 → A4秒 + A×3。
        ok, detected = cls._tap_or_detect(context, watcher, BIT_A, 4000, 500)
        if not ok or detected:
            return ok, detected
        for bit_index, hold_ms, gap_ms in (
            (BIT_X, 50, 500),
            (BIT_A, 50, 500),
            (BIT_A, 50, 500),
            (BIT_A, 50, 500),
            (BIT_A, 50, 500),
            (BIT_A, 50, 500),
            (BIT_A, 50, 500),
            (BIT_A, 4000, 500),
            (BIT_A, 50, 500),
            (BIT_A, 50, 500),
            (BIT_A, 50, 500),
        ):
            ok, detected = cls._tap_or_detect(
                context,
                watcher,
                bit_index,
                hold_ms,
                gap_ms,
            )
            if not ok or detected:
                return ok, detected
        return True, watcher.detected.is_set()

    @classmethod
    def _run_confirmation_attempt(
        cls,
        context: _MacroContext,
        watcher: StableWhiteIconWatcher,
    ) -> tuple[bool, bool]:
        # 第二轮起的小循环：每个按键前、按住期间和按键间隔均可立即检测退出。
        for bit_index, hold_ms in (
            (BIT_A, 4000),
            (BIT_A, 50),
            (BIT_A, 50),
            (BIT_A, 50),
        ):
            ok, detected = cls._tap_or_detect(
                context,
                watcher,
                bit_index,
                hold_ms,
                500,
            )
            if not ok or detected:
                return ok, detected
        return True, watcher.detected.is_set()

    def _wait_for_first_round_four_white_icons(
        self,
        context: _MacroContext,
        observer: LatestFrameObserver,
    ) -> bool:
        self._set_state(SmartMacroState.WAITING_FOR_FOUR_WHITE_ICONS)
        watcher = StableWhiteIconWatcher(observer, context)
        watcher.start()
        try:
            while not context.stop_event.is_set():
                ok, detected = self._run_first_round_attempt(context, watcher)
                if not ok:
                    return False
                if detected:
                    _timestamped_log(
                        "smart macro1：第一轮持续检测到四个白色图案，直接进入 ZR 阶段。"
                    )
                    return True
            return False
        finally:
            watcher.stop()

    def _wait_for_four_white_icons(
        self,
        context: _MacroContext,
        observer: LatestFrameObserver,
    ) -> bool:
        self._set_state(SmartMacroState.WAITING_FOR_FOUR_WHITE_ICONS)
        watcher = StableWhiteIconWatcher(observer, context)
        watcher.start()
        try:
            while not context.stop_event.is_set():
                ok, detected = self._run_confirmation_attempt(context, watcher)
                if not ok:
                    return False
                if detected:
                    _timestamped_log(
                        "smart macro1：第二轮起检测到四个白色图案，"
                        "跳过当前及剩余按键并进入 ZR 阶段。"
                    )
                    return True
            return False
        finally:
            watcher.stop()

    @staticmethod
    def _run_zr_skill_rotation(
        context: _MacroContext,
        duration_ms: int,
    ) -> bool:
        """Keep existing held inputs and tap L/R/A once per second in rotation."""
        started_active = context.active_monotonic()
        deadline_active = started_active + max(0, duration_ms) / 1000.0
        next_action_active = started_active + 1.0
        action_index = 0
        action_bits = (BIT_L, BIT_R, BIT_A)

        while not context.stop_event.is_set():
            now_active = context.active_monotonic()
            if now_active >= deadline_active:
                return True
            target_active = min(deadline_active, next_action_active)
            wait_ms = max(0, int(round((target_active - now_active) * 1000)))
            if wait_ms > 0 and not context.wait_ms(wait_ms):
                return False
            now_active = context.active_monotonic()
            if now_active >= deadline_active:
                return True
            if now_active + 0.001 < next_action_active:
                continue
            if not context.tap(
                action_bits[action_index % len(action_bits)],
                hold_ms=50,
                gap_ms=0,
            ):
                return False
            action_index += 1
            next_action_active += 1.0
        return False

    @staticmethod
    def _run_finish_round(context: _MacroContext) -> bool:
        # 开局位移：保持 ZR 32 秒，再在保持期间前进 3 秒。
        context.set_held(BIT_ZR, True)
        if not SmartMacro1._run_zr_skill_rotation(context, duration_ms=32000):
            return False
        if not context.move_stick(BIT_LSTICK_UP, duration_ms=3000):
            return False
        context.center_stick()
        context.set_held(BIT_ZR, False)

        if not context.move_stick(BIT_LSTICK_UP, duration_ms=3800):
            return False
        context.center_stick()

        # 结束等待 + 结束结算。
        if not context.wait_ms(11000):
            return False
        for gap_ms in (1500, 1500, 1500, 1500, 1500, 500):
            if not context.tap(BIT_A, hold_ms=50, gap_ms=gap_ms):
                return False
        return True

    def run(self, context: _MacroContext, observer: LatestFrameObserver) -> None:
        if not self._wait_for_video(context, observer):
            return

        # 第一轮专用流程会重复：额外 A4秒 → 进入地图 → A4秒+A×3。
        if not self._run_first_round_setup(context):
            return
        if not context.run_auto_sell_if_due():
            return
        if not self._wait_for_first_round_four_white_icons(context, observer):
            return
        self._set_state(SmartMacroState.FINISHING_ROUND)
        if not self._run_finish_round(context):
            return
        context.complete_macro_loop()

        # 第二轮起每轮先进入地图一次，再重复 A4秒 + A×3 直到检测成功。
        while not context.stop_event.is_set():
            if not context.run_auto_sell_if_due():
                return
            self._set_state(SmartMacroState.ENTERING_MAP)
            if not self._run_enter_map(context):
                return
            if not self._wait_for_four_white_icons(context, observer):
                return
            self._set_state(SmartMacroState.FINISHING_ROUND)
            if not self._run_finish_round(context):
                return
            context.complete_macro_loop()


class SmartMacro2:
    """Independent copy of SmartMacro1 for separate future editing."""

    def __init__(self) -> None:
        self.state = SmartMacroState.WAITING_FOR_VIDEO

    def _set_state(self, state: SmartMacroState) -> None:
        if state is self.state:
            return
        self.state = state
        _timestamped_log(f"smart macro2 状态切换：{state.value}")

    def _wait_for_video(self, context: _MacroContext, observer: LatestFrameObserver) -> bool:
        self._set_state(SmartMacroState.WAITING_FOR_VIDEO)
        last_error = ""
        while not context.stop_event.is_set():
            snapshot = observer.snapshot()
            if snapshot.frame is not None:
                return True
            if snapshot.error and snapshot.error != last_error:
                last_error = snapshot.error
                _timestamped_log(f"smart macro2 等待视频：{last_error}")
            if not context.wait_ms(100):
                return False
        return False

    @staticmethod
    def _run_first_round_setup(context: _MacroContext) -> bool:
        # 手柄检测后、第一轮前：短按 A 四次，等待 10 秒，再向后推动摇杆 5 秒。
        for _ in range(4):
            if not context.tap(BIT_A, hold_ms=50, gap_ms=500):
                return False
        if not context.wait_ms(10000):
            return False
        if not context.move_stick(BIT_LSTICK_DOWN, duration_ms=5000):
            return False
        context.center_stick()
        return True

    @staticmethod
    def _run_enter_map(context: _MacroContext) -> bool:
        for bit_index, gap_ms in (
            (BIT_X, 500),
            (BIT_A, 500),
            (BIT_A, 500),
            (BIT_A, 500),
            (BIT_A, 500),
            (BIT_A, 500),
            (BIT_A, 500),
        ):
            if not context.tap(bit_index, hold_ms=50, gap_ms=gap_ms):
                return False
        return True

    def _wait_for_four_white_icons(
        self,
        context: _MacroContext,
        observer: LatestFrameObserver,
    ) -> bool:
        self._set_state(SmartMacroState.WAITING_FOR_FOUR_WHITE_ICONS)
        gate_started_sequence = observer.snapshot().sequence
        last_checked_sequence = gate_started_sequence
        stable_matches = 0

        while not context.stop_event.is_set():
            # 每个按键发送前都重新检查画面；命中后不再发送当前及剩余按键。
            for bit_index, hold_ms in (
                (BIT_A, 4000),
                (BIT_A, 50),
                (BIT_A, 50),
                (BIT_A, 50),
            ):
                while not context.stop_event.is_set():
                    snapshot = observer.snapshot()
                    if snapshot.frame is None:
                        stable_matches = 0
                        if not self._wait_for_video(context, observer):
                            return False
                        self._set_state(SmartMacroState.WAITING_FOR_FOUR_WHITE_ICONS)
                        continue

                    if (
                        snapshot.sequence > gate_started_sequence
                        and snapshot.sequence != last_checked_sequence
                    ):
                        last_checked_sequence = snapshot.sequence
                        detection = detect_four_white_top_left_icons(snapshot.frame)
                        stable_matches = stable_matches + 1 if detection.detected else 0
                        if stable_matches >= 2:
                            _timestamped_log(
                                "smart macro2：连续两帧检测到左上角四个纯白图案，"
                                "跳过剩余按键并进入下一阶段。"
                            )
                            return True
                        if detection.detected:
                            if not context.wait_ms(120):
                                return False
                            continue
                    elif stable_matches > 0:
                        if not context.wait_ms(50):
                            return False
                        continue

                    break

                if context.stop_event.is_set():
                    return False
                if not context.tap(bit_index, hold_ms=hold_ms, gap_ms=500):
                    return False
        return False

    @staticmethod
    def _run_zr_skill_rotation(
        context: _MacroContext,
        duration_ms: int,
    ) -> bool:
        """Keep existing held inputs and tap L/R/A once per second in rotation."""
        started_active = context.active_monotonic()
        deadline_active = started_active + max(0, duration_ms) / 1000.0
        next_action_active = started_active + 1.0
        action_index = 0
        action_bits = (BIT_L, BIT_R, BIT_A)

        while not context.stop_event.is_set():
            now_active = context.active_monotonic()
            if now_active >= deadline_active:
                return True
            target_active = min(deadline_active, next_action_active)
            wait_ms = max(0, int(round((target_active - now_active) * 1000)))
            if wait_ms > 0 and not context.wait_ms(wait_ms):
                return False
            now_active = context.active_monotonic()
            if now_active >= deadline_active:
                return True
            if now_active + 0.001 < next_action_active:
                continue
            if not context.tap(
                action_bits[action_index % len(action_bits)],
                hold_ms=50,
                gap_ms=0,
            ):
                return False
            action_index += 1
            next_action_active += 1.0
        return False

    @staticmethod
    def _run_finish_round(context: _MacroContext) -> bool:
        context.set_held(BIT_ZR, True)
        if not SmartMacro2._run_zr_skill_rotation(context, duration_ms=32000):
            return False
        if not context.move_stick(BIT_LSTICK_UP, duration_ms=3000):
            return False
        context.center_stick()
        context.set_held(BIT_ZR, False)

        if not context.move_stick(BIT_LSTICK_UP, duration_ms=3800):
            return False
        context.center_stick()

        if not context.wait_ms(11000):
            return False
        for gap_ms in (1500, 1500, 1500, 1500, 1500, 500):
            if not context.tap(BIT_A, hold_ms=50, gap_ms=gap_ms):
                return False
        return True

    def run(self, context: _MacroContext, observer: LatestFrameObserver) -> None:
        if not self._wait_for_video(context, observer):
            return

        # 第一轮：初始化一次，直接循环 A4秒+A×3，直到检测成功。
        if not self._run_first_round_setup(context):
            return
        if not context.run_auto_sell_if_due():
            return
        if not self._wait_for_four_white_icons(context, observer):
            return
        self._set_state(SmartMacroState.FINISHING_ROUND)
        if not self._run_finish_round(context):
            return
        context.complete_macro_loop()

        # 第二轮起不执行选图，直接循环检测序列并进入刷取阶段。
        while not context.stop_event.is_set():
            if not context.run_auto_sell_if_due():
                return
            if not self._wait_for_four_white_icons(context, observer):
                return
            self._set_state(SmartMacroState.FINISHING_ROUND)
            if not self._run_finish_round(context):
                return
            context.complete_macro_loop()


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Video-aware state-machine macro gamepad (SmartMacro1/2)"
    )
    parser.add_argument("--config", default=DEFAULT_CONFIG, help="config JSON path")
    parser.add_argument("--baudrate", type=int, default=9600)
    parser.add_argument("--scan-interval", type=float, default=1.0)
    parser.add_argument("--probe-timeout", type=float, default=1.2)
    parser.add_argument(
        "--macro",
        choices=("macro1", "macro2"),
        default="macro1",
        help="smart macro profile",
    )
    parser.add_argument(
        "--vision-interval",
        type=float,
        default=0.10,
        help="seconds between video-state observations",
    )
    return parser.parse_args()


def _send_manual_controller_pulse(context: _MacroContext, bit_index: int) -> None:
    """Overlay one short manual input without changing the macro's saved state."""
    paused = context.is_paused()
    base_bits = 0 if paused else context.active_bits
    if bit_index in _MANUAL_ARROW_BITS.values():
        output_bits = (
            base_bits & ~context.stick_direction_mask
        ) | (1 << bit_index)
        hold_ms = 150
    else:
        output_bits = base_bits | (1 << bit_index)
        hold_ms = 80 if bit_index in {BIT_L, BIT_R, BIT_ZL, BIT_ZR} else 50

    context.controller.send_bits(output_bits)
    context.stop_event.wait(hold_ms / 1000.0)
    restore_bits = 0 if context.is_paused() else context.active_bits
    context.controller.send_bits(restore_bits)


def _handle_terminal_control_key(
    key: str,
    context: _MacroContext,
    pause_state: _PauseState,
) -> bool:
    normalized = key.lower()
    if normalized == "q":
        context.stop_event.set()
        return False
    if normalized == "p":
        paused = pause_state.toggle()
        if paused:
            _timestamped_log("宏已暂停；可使用手动键盘控制。再次按 P 恢复。")
        else:
            _timestamped_log("收到恢复指令，准备重新执行手柄检测。")
        return True

    bit_index = _MANUAL_ARROW_BITS.get(key.upper())
    if bit_index is None:
        bit_index = _MANUAL_KEY_BITS.get(normalized)
    if bit_index is not None:
        _send_manual_controller_pulse(context, bit_index)
    return True


def _read_windows_terminal_key() -> str:
    key = msvcrt.getwch()
    if key not in {"\x00", "\xe0"}:
        return key
    extended = msvcrt.getwch()
    return {
        "H": "UP",
        "P": "DOWN",
        "K": "LEFT",
        "M": "RIGHT",
    }.get(extended, "")


def _read_posix_terminal_key(input_fd: int) -> str:
    first = os.read(input_fd, 1)
    if not first:
        return ""
    if first != b"\x1b":
        return first.decode("utf-8", errors="ignore")

    tail = b""
    deadline = time.monotonic() + 0.05
    while len(tail) < 2 and time.monotonic() < deadline:
        readable, _, _ = select.select([input_fd], [], [], 0.005)
        if not readable:
            continue
        tail += os.read(input_fd, 1)
    if len(tail) >= 2 and tail[0:1] in {b"[", b"O"}:
        return {
            b"A": "UP",
            b"B": "DOWN",
            b"C": "RIGHT",
            b"D": "LEFT",
        }.get(tail[-1:], "")
    return ""


def _listen_for_keyboard_control(
    context: _MacroContext,
    pause_state: _PauseState,
    worker_errors: List[BaseException],
) -> None:
    if not sys.stdin.isatty():
        context.stop_event.wait()
        return
    try:
        if os.name == "nt":
            while not context.stop_event.is_set():
                if not msvcrt.kbhit():
                    context.stop_event.wait(0.05)
                    continue
                key = _read_windows_terminal_key()
                if key and not _handle_terminal_control_key(key, context, pause_state):
                    return
            return

        input_fd = sys.stdin.fileno()
        previous_terminal_mode = termios.tcgetattr(input_fd)
        try:
            tty.setcbreak(input_fd)
            while not context.stop_event.is_set():
                readable, _, _ = select.select([input_fd], [], [], 0.05)
                if not readable:
                    continue
                key = _read_posix_terminal_key(input_fd)
                if key and not _handle_terminal_control_key(key, context, pause_state):
                    return
        finally:
            termios.tcsetattr(input_fd, termios.TCSADRAIN, previous_terminal_mode)
    except BaseException as exc:
        worker_errors.append(exc)
        context.stop_event.set()


def run_smart_macro_forever(
    controller: SerialRemoteController,
    frame_url: str,
    vision_interval: float,
    macro_profile: str,
) -> None:
    stop_event = threading.Event()
    pause_state = _PauseState()
    worker_errors: List[BaseException] = []
    observer = LatestFrameObserver(frame_url, stop_event, vision_interval)
    context = _MacroContext(controller, stop_event, pause_state)
    smart_macro = SmartMacro1() if macro_profile == "macro1" else SmartMacro2()

    def controller_worker() -> None:
        try:
            if _run_controller_detection(context):
                context.enable_auto_sell(f"smart_{macro_profile}")
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
            target=_listen_for_keyboard_control,
            args=(context, pause_state, worker_errors),
            name="smart-macro-keyboard-listener",
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
        if context.status_callback is not None:
            context.status_callback(True)
            context.status_callback = None
            _commit_status_line()

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
            f"智能宏手柄已启动：{selected_port}；配置：{args.macro}；视频：{frame_url}。"
            "手柄检测只执行一次；按 P 暂停/恢复，按 Q 或 Ctrl+C 退出。\n"
            "键盘：Z=A X=B A=Y S=X；方向键=左摇杆；"
            "C=L V=R F=ZL G=ZR；+/D=Plus，-/E=Minus。"
        )
        run_smart_macro_forever(
            controller=controller,
            frame_url=frame_url,
            vision_interval=args.vision_interval,
            macro_profile=args.macro,
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
