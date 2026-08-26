from __future__ import annotations

import argparse
import contextlib
import json
import msvcrt
import shutil
import sys
import threading
import time
import unicodedata
from pathlib import Path
from typing import Callable, Dict, List


REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from switch_connect.ui.terminal_select import choose_with_arrows
from switch_connect.virtual_gamepad.device_discovery import (
    list_serial_port_labels,
    parse_device_from_label,
)
from switch_connect.virtual_gamepad.input_mapper import (
    BIT_A,
    BIT_B,
    BIT_CAPTURE,
    BIT_DPAD_DOWN,
    BIT_DPAD_LEFT,
    BIT_DPAD_RIGHT,
    BIT_DPAD_UP,
    BIT_HOME,
    BIT_L,
    BIT_LSTICK_DOWN,
    BIT_LSTICK_LEFT,
    BIT_LSTICK_RIGHT,
    BIT_LSTICK_UP,
    BIT_MINUS,
    BIT_PLUS,
    BIT_R,
    BIT_RSTICK_DOWN,
    BIT_RSTICK_LEFT,
    BIT_RSTICK_RIGHT,
    BIT_RSTICK_UP,
    BIT_X,
    BIT_Y,
    BIT_ZL,
    BIT_ZR,
)
from switch_connect.virtual_gamepad.serial_controller import SerialRemoteController


DEFAULT_CONFIG = "autocontroller_rebuild_for_RL/runtime_config.local.json"
MACRO_PROFILE_NAMES = tuple(f"macro{index}" for index in range(1, 1000))
AUTO_SELL_MACRO_PROFILES = frozenset(f"macro{index}" for index in range(1, 6))
SELL_EQUIPMENT_INTERVAL_SECONDS = 90 * 60
_TERMINAL_LOCK = threading.Lock()
_ACTIVE_STATUS_LINE: str | None = None

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
    "h": BIT_HOME,
    "\r": BIT_HOME,
    "\n": BIT_HOME,
    "i": BIT_RSTICK_UP,
    "j": BIT_RSTICK_LEFT,
    "k": BIT_RSTICK_DOWN,
    "l": BIT_RSTICK_RIGHT,
    ";": BIT_CAPTURE,
    ":": BIT_CAPTURE,
}
_MANUAL_ARROW_BITS = {
    "UP": BIT_LSTICK_UP,
    "DOWN": BIT_LSTICK_DOWN,
    "LEFT": BIT_LSTICK_LEFT,
    "RIGHT": BIT_LSTICK_RIGHT,
}


class _PauseState:
    """Thread-safe pause clock shared by the macro and keyboard listener."""

    def __init__(self) -> None:
        self._paused = threading.Event()
        self._resume_detection_pending = threading.Event()
        self._lock = threading.Lock()
        self._paused_since: float | None = None
        self._total_paused_seconds = 0.0

    def toggle(self) -> bool:
        """Toggle pause and return True when the new state is paused."""
        now = time.monotonic()
        with self._lock:
            if self._paused.is_set():
                if self._paused_since is not None:
                    self._total_paused_seconds += now - self._paused_since
                self._paused_since = None
                self._paused.clear()
                self._resume_detection_pending.set()
                return False
            self._paused_since = now
            self._paused.set()
            return True

    def is_paused(self) -> bool:
        return self._paused.is_set()

    def consume_resume_detection(self) -> bool:
        if not self._resume_detection_pending.is_set():
            return False
        self._resume_detection_pending.clear()
        return True

    def has_resume_detection(self) -> bool:
        return self._resume_detection_pending.is_set()

    def exclude_elapsed(self, seconds: float) -> None:
        """Exclude non-macro work, such as resume detection, from active time."""
        with self._lock:
            self._total_paused_seconds += max(0.0, seconds)

    def active_monotonic(self) -> float:
        now = time.monotonic()
        with self._lock:
            paused_seconds = self._total_paused_seconds
            if self._paused_since is not None:
                paused_seconds += now - self._paused_since
        return now - paused_seconds


def _timestamped_log(message: str, *, file=None) -> None:
    """Clear status, print a timestamped log, then redraw status below it."""
    stream = file if file is not None else sys.stdout
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    with _TERMINAL_LOCK:
        if sys.stdout.isatty() and _ACTIVE_STATUS_LINE is not None:
            sys.stdout.write("\r\033[2K")
        stream.write(f"[{timestamp}] {message}\n")
        stream.flush()
        if sys.stdout.isatty() and _ACTIVE_STATUS_LINE is not None:
            sys.stdout.write(f"\r\033[2K{_ACTIVE_STATUS_LINE}")
            sys.stdout.flush()


def _set_status_line(status: str) -> None:
    global _ACTIVE_STATUS_LINE
    if not sys.stdout.isatty():
        return
    with _TERMINAL_LOCK:
        _ACTIVE_STATUS_LINE = status
        sys.stdout.write(f"\r\033[2K{status}")
        sys.stdout.flush()


def _clear_status_line() -> None:
    global _ACTIVE_STATUS_LINE
    with _TERMINAL_LOCK:
        had_status = _ACTIVE_STATUS_LINE is not None
        _ACTIVE_STATUS_LINE = None
        if had_status and sys.stdout.isatty():
            sys.stdout.write("\r\033[2K")
            sys.stdout.flush()


def _commit_status_line() -> None:
    """Convert the active status row into a permanent timestamped log line."""
    global _ACTIVE_STATUS_LINE
    with _TERMINAL_LOCK:
        status = _ACTIVE_STATUS_LINE
        _ACTIVE_STATUS_LINE = None
        if status is None:
            return
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        if sys.stdout.isatty():
            sys.stdout.write("\r\033[2K")
        sys.stdout.write(f"[{timestamp}] 最终状态：{status}\n")
        sys.stdout.flush()


def _write_transient_line(message: str) -> None:
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    with _TERMINAL_LOCK:
        sys.stdout.write(f"\r\033[2K[{timestamp}] {message}")
        sys.stdout.flush()


def _display_width(text: str) -> int:
    width = 0
    for char in text:
        if unicodedata.combining(char):
            continue
        width += 2 if unicodedata.east_asian_width(char) in {"W", "F"} else 1
    return width


def _fit_terminal_line(text: str, columns: int) -> str:
    max_width = max(1, columns - 1)
    if _display_width(text) <= max_width:
        return text
    result: List[str] = []
    width = 0
    for char in text:
        char_width = 0 if unicodedata.combining(char) else (
            2 if unicodedata.east_asian_width(char) in {"W", "F"} else 1
        )
        if width + char_width > max_width:
            break
        result.append(char)
        width += char_width
    return "".join(result)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Select an AutoController serial port, save it, and start the macro gamepad loop."
    )
    parser.add_argument(
        "--config",
        default=DEFAULT_CONFIG,
        help=f"config JSON path (default: {DEFAULT_CONFIG})",
    )
    parser.add_argument(
        "--scan-interval",
        type=float,
        default=1.0,
        help="seconds between serial scans when no compatible controller is found",
    )
    parser.add_argument("--baudrate", type=int, default=9600)
    parser.add_argument(
        "--macro",
        default="macro1",
        metavar="macro1..macro999",
        help="macro profile to run from macro1 through macro999 (default: macro1)",
    )
    parser.add_argument(
        "--probe-timeout",
        type=float,
        default=1.2,
        help="seconds to wait for the AutoController firmware acknowledgement",
    )
    args = parser.parse_args()
    if args.macro not in MACRO_PROFILE_NAMES:
        parser.error("--macro must be between macro1 and macro999")
    return args


def _resolve_config_path(config_path: str | Path) -> Path:
    resolved = Path(config_path).expanduser()
    if not resolved.is_absolute():
        resolved = REPO_ROOT / resolved
    return resolved


def _load_config(config_path: Path) -> Dict[str, object]:
    if not config_path.exists():
        return {}
    try:
        payload = json.loads(config_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"配置文件不是有效 JSON：{config_path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise RuntimeError(f"配置文件顶层必须是 JSON object：{config_path}")
    return payload


def _save_serial_selection(config_path: Path, serial_port: str) -> None:
    payload = _load_config(config_path)
    payload["serial_port"] = serial_port
    payload["pick_serial"] = False
    config_path.parent.mkdir(parents=True, exist_ok=True)

    temporary_path = config_path.with_name(f".{config_path.name}.tmp")
    temporary_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary_path.replace(config_path)


def _firmware_probe(port: str, baudrate: int, timeout_seconds: float) -> bool:
    controller = None
    try:
        controller = SerialRemoteController(
            port=port,
            baudrate=baudrate,
            timeout=0.1,
        )
        return controller.probe_firmware(timeout_seconds=timeout_seconds)
    except Exception:
        return False
    finally:
        if controller is not None:
            with contextlib.suppress(Exception):
                controller.close()


def _probe_priority(label: str, configured_port: str = "") -> tuple[int, str]:
    text = str(label).lower()
    port = parse_device_from_label(label)
    if configured_port and port == configured_port:
        return (0, text)
    if "cp210" in text or "usbserial" in text:
        return (1, text)
    if "wch" in text or "ch340" in text:
        return (2, text)
    if port.upper().startswith("COM"):
        return (3, text)
    return (4, text)


def _usable_port_labels(
    labels: List[str],
    configured_port: str,
    baudrate: int,
    probe_timeout: float,
) -> List[str]:
    usable: List[str] = []
    for label in sorted(labels, key=lambda item: _probe_priority(item, configured_port)):
        port = parse_device_from_label(label)
        if _firmware_probe(port, baudrate, probe_timeout):
            usable.append(label)
    return usable


def wait_for_serial_selection(
    config_path: Path,
    baudrate: int = 9600,
    probe_timeout: float = 1.2,
    scan_interval: float = 1.0,
) -> str:
    config = _load_config(config_path)
    configured_port = str(config.get("serial_port", "") or "").strip()

    while True:
        labels = list_serial_port_labels()
        available_ports = {parse_device_from_label(label) for label in labels}

        if configured_port and configured_port in available_ports:
            if _firmware_probe(configured_port, baudrate, probe_timeout):
                _timestamped_log(f"已使用 config 中可用的 switch_link 串口：{configured_port}")
                return configured_port

        usable_labels = _usable_port_labels(
            labels=labels,
            configured_port=configured_port,
            baudrate=baudrate,
            probe_timeout=probe_timeout,
        )
        if usable_labels:
            picked = choose_with_arrows(
                usable_labels,
                "选择可用的 switch_link 串口",
                "使用 ↑/↓ 选择，Enter 确认，Ctrl+C 取消。",
            )
            if not picked:
                raise KeyboardInterrupt

            selected_port = parse_device_from_label(picked)
            _save_serial_selection(config_path, selected_port)
            _timestamped_log(f"已选择并记录 switch_link 串口：{selected_port}")
            return selected_port

        _write_transient_line(
            f"未检测到可用的 switch_link 串口，{max(0.1, scan_interval):.1f}s 后重新扫描……"
        )
        time.sleep(max(0.1, scan_interval))


class _MacroContext:
    def __init__(
        self,
        controller: SerialRemoteController,
        stop_event: threading.Event,
        pause_state: _PauseState,
    ) -> None:
        self.controller = controller
        self.stop_event = stop_event
        self.pause_state = pause_state
        self.status_callback: Callable[[bool], None] | None = None
        self.auto_sell_profile_name: str | None = None
        self.auto_sell_started_active = 0.0
        self.next_sell_active = 0.0
        self.last_sell_wall: float | None = None
        self.sell_count = 0
        self.macro_loop_count = 0
        self.last_status_render = 0.0
        self.active_bits = 0
        self.macro_profile_name = ""
        self.macro6_part_triggers = {
            1: threading.Event(),
            2: threading.Event(),
        }
        self.stick_direction_mask = (
            (1 << BIT_LSTICK_UP)
            | (1 << BIT_LSTICK_DOWN)
            | (1 << BIT_LSTICK_LEFT)
            | (1 << BIT_LSTICK_RIGHT)
        )
        self.right_stick_direction_mask = (
            (1 << BIT_RSTICK_UP)
            | (1 << BIT_RSTICK_DOWN)
            | (1 << BIT_RSTICK_LEFT)
            | (1 << BIT_RSTICK_RIGHT)
        )

    def send_active_bits(self) -> None:
        self.controller.send_bits(self.active_bits)

    def active_monotonic(self) -> float:
        return self.pause_state.active_monotonic()

    def is_paused(self) -> bool:
        return self.pause_state.is_paused()

    def _raw_wait_ms(self, duration_ms: int) -> bool:
        """Wait during resume detection without recursively handling pause."""
        return not self.stop_event.wait(max(0, duration_ms) / 1000.0)

    def _run_resume_detection(self) -> bool:
        detection_bit = 1 << BIT_A
        for gap_ms in (1000, 1000, 1000, 1000, 1000, 1000, 3000):
            self.controller.send_bits(detection_bit)
            if not self._raw_wait_ms(50):
                return False
            self.controller.release()
            if not self._raw_wait_ms(gap_ms):
                return False
        return True

    def _service_pause(self) -> bool:
        if not (
            self.pause_state.is_paused()
            or self.pause_state.has_resume_detection()
        ):
            return True

        saved_bits = self.active_bits
        self.controller.release()

        while True:
            while self.pause_state.is_paused():
                if self.status_callback is not None:
                    self.status_callback(False)
                if self.stop_event.wait(0.1):
                    return False

            if self.pause_state.consume_resume_detection():
                _timestamped_log("宏已恢复，重新执行手柄检测后继续原序列。")
                self.active_bits = 0
                detection_started = time.monotonic()
                detection_completed = self._run_resume_detection()
                self.pause_state.exclude_elapsed(time.monotonic() - detection_started)
                if not detection_completed:
                    return False
                if self.pause_state.is_paused():
                    self.controller.release()
                    continue
            break

        self.active_bits = saved_bits
        self.send_active_bits()
        return True

    def wait_ms(self, duration_ms: int) -> bool:
        remaining = max(0, duration_ms) / 1000.0
        while remaining > 0:
            if not self._service_pause():
                return False
            if self.status_callback is not None:
                self.status_callback(False)
            started_wait = self.active_monotonic()
            poll_seconds = min(remaining, 0.1)
            if self.stop_event.wait(poll_seconds):
                return False
            if not self.pause_state.is_paused():
                remaining -= max(0.0, self.active_monotonic() - started_wait)
        return self._service_pause()

    def set_held(self, bit_index: int, pressed: bool) -> None:
        bit = 1 << bit_index
        if pressed:
            self.active_bits |= bit
        else:
            self.active_bits &= ~bit
        self.send_active_bits()

    def set_held_many(self, bit_indices: tuple[int, ...], pressed: bool) -> None:
        bits = 0
        for bit_index in bit_indices:
            bits |= 1 << bit_index
        if pressed:
            self.active_bits |= bits
        else:
            self.active_bits &= ~bits
        self.send_active_bits()

    def tap(self, bit_index: int, hold_ms: int, gap_ms: int) -> bool:
        self.set_held(bit_index, True)
        if not self.wait_ms(hold_ms):
            return False
        self.set_held(bit_index, False)
        return self.wait_ms(gap_ms)

    def move_stick(self, bit_index: int, duration_ms: int) -> bool:
        self.active_bits = (
            self.active_bits & ~self.stick_direction_mask
        ) | (1 << bit_index)
        self.send_active_bits()
        return self.wait_ms(duration_ms)

    def center_stick(self) -> None:
        self.active_bits &= ~self.stick_direction_mask
        self.send_active_bits()

    @staticmethod
    def _format_duration(total_seconds: float) -> str:
        seconds = max(0, int(total_seconds))
        hours, remainder = divmod(seconds, 3600)
        minutes, seconds = divmod(remainder, 60)
        return f"{hours:02d}:{minutes:02d}:{seconds:02d}"

    def enable_auto_sell(self, profile_name: str) -> None:
        self.auto_sell_profile_name = profile_name
        self.auto_sell_started_active = self.active_monotonic()
        self.next_sell_active = (
            self.auto_sell_started_active + SELL_EQUIPMENT_INTERVAL_SECONDS
        )
        self.status_callback = self.render_auto_sell_status
        self.render_auto_sell_status(force=True)

    def render_auto_sell_status(self, force: bool = False) -> None:
        if self.auto_sell_profile_name is None or not sys.stdout.isatty():
            return
        now_wall = time.monotonic()
        if not force and now_wall - self.last_status_render < 1.0:
            return
        self.last_status_render = now_wall
        now_active = self.active_monotonic()
        last_text = (
            time.strftime("%m-%d %H:%M:%S", time.localtime(self.last_sell_wall))
            if self.last_sell_wall is not None
            else "尚未执行"
        )
        status = (
            f"宏 {self.auto_sell_profile_name}"
            f"｜状态 {'暂停' if self.is_paused() else '运行'}"
            f"｜已运行 {self._format_duration(now_active - self.auto_sell_started_active)}"
            f"｜宏循环次数 {self.macro_loop_count}"
            f"｜上次卖装 {last_text}"
            f"｜卖装次数 {self.sell_count}"
            f"｜下次卖装 {self._format_duration(self.next_sell_active - now_active)}"
        )
        columns = shutil.get_terminal_size(fallback=(120, 24)).columns
        _set_status_line(_fit_terminal_line(status, columns))

    def run_auto_sell_if_due(self) -> bool:
        if self.auto_sell_profile_name is None:
            return True
        now_active = self.active_monotonic()
        if now_active < self.next_sell_active:
            return True
        _timestamped_log(
            f"{self.auto_sell_profile_name}：已到 90 分钟定时点，"
            f"开始第 {self.sell_count + 1} 次卖装序列。"
        )
        if not _run_sell_equipment_sequence(self):
            return False
        now_active = self.active_monotonic()
        while self.next_sell_active <= now_active:
            self.next_sell_active += SELL_EQUIPMENT_INTERVAL_SECONDS
        self.sell_count += 1
        self.last_sell_wall = time.time()
        self.render_auto_sell_status(force=True)
        return True

    def complete_macro_loop(self) -> None:
        if self.auto_sell_profile_name is None:
            return
        self.macro_loop_count += 1
        self.render_auto_sell_status(force=True)


def _run_controller_detection(context: _MacroContext) -> bool:
    """Run the common controller detection sequence exactly once."""
    for gap_ms in (1000, 1000, 1000, 1000, 1000, 1000, 3000):
        if not context.tap(BIT_A, hold_ms=50, gap_ms=gap_ms):
            return False
    return True


def _run_macro_1_loop(context: _MacroContext) -> None:
    """宏1：三配件为 L 跃升、R 冲刺、A 黏索。"""
    while not context.stop_event.is_set():
        if not context.run_auto_sell_if_due():
            return
        # 进入地图。
        for bit_index, gap_ms in (
            (BIT_X, 500),
            (BIT_A, 500),
            (BIT_A, 500),
            (BIT_A, 500),
            (BIT_A, 500),
            (BIT_A, 500),
            (BIT_A, 7000),
        ):
            if not context.tap(bit_index, hold_ms=50, gap_ms=gap_ms):
                return

        # 起跳。
        if not context.tap(BIT_B, hold_ms=300, gap_ms=250):
            return
        if not context.tap(BIT_B, hold_ms=50, gap_ms=500):
            return

        # 持续推动左摇杆向前，同时交替点按 R/L。
        context.set_held(BIT_LSTICK_UP, True)
        if not context.wait_ms(1000):
            return
        for bit_index in (BIT_R, BIT_L, BIT_R, BIT_L):
            if not context.tap(bit_index, hold_ms=50, gap_ms=1000):
                return
        context.set_held(BIT_LSTICK_UP, False)

        # 上平台。
        if not context.wait_ms(1700):
            return
        if not context.tap(BIT_A, hold_ms=50, gap_ms=3000):
            return

        # 清平台：持续按住 ZR，同时按顺序推动左摇杆。
        context.set_held(BIT_ZR, True)
        if not context.wait_ms(50):
            return
        if not context.move_stick(BIT_LSTICK_LEFT, duration_ms=1800):
            return
        for direction, duration_ms in (
            (BIT_LSTICK_UP, 2000),
            (BIT_LSTICK_RIGHT, 4000),
            (BIT_LSTICK_DOWN, 1000),
            (BIT_LSTICK_LEFT, 2700),
            (BIT_LSTICK_DOWN, 1500),
            (BIT_LSTICK_RIGHT, 300),
        ):
            if not context.move_stick(direction, duration_ms=duration_ms):
                return
        context.center_stick()
        context.set_held(BIT_ZR, False)

        if not context.wait_ms(12000):
            return

        # 结束结算。
        for _ in range(7):
            if not context.tap(BIT_A, hold_ms=50, gap_ms=1500):
                return
        context.complete_macro_loop()


def _run_macro_2_loop(context: _MacroContext) -> None:
    """宏2（速刷）：L 跃升、R 黏索、A 风扇；风扇收集蛋。"""
    while not context.stop_event.is_set():
        if not context.run_auto_sell_if_due():
            return
        # 进入地图。
        for bit_index, gap_ms in (
            (BIT_X, 500),
            (BIT_A, 500),
            (BIT_A, 500),
            (BIT_A, 500),
            (BIT_A, 500),
            (BIT_A, 500),
            (BIT_A, 6500),
        ):
            if not context.tap(bit_index, hold_ms=50, gap_ms=gap_ms):
                return

        # 开局跃升。
        if not context.tap(BIT_B, hold_ms=300, gap_ms=250):
            return
        if not context.tap(BIT_B, hold_ms=50, gap_ms=1000):
            return
        if not context.tap(BIT_X, hold_ms=50, gap_ms=500):
            return
        if not context.tap(BIT_B, hold_ms=50, gap_ms=0):
            return

        # 同时保持 Y 与左摇杆前，依次执行 L、L、B、右移、A。
        context.set_held_many((BIT_Y, BIT_LSTICK_UP), True)
        if not context.wait_ms(500):
            return
        if not context.tap(BIT_L, hold_ms=50, gap_ms=1500):
            return
        if not context.tap(BIT_L, hold_ms=50, gap_ms=1800):
            return
        if not context.tap(BIT_B, hold_ms=50, gap_ms=0):
            return
        context.set_held(BIT_LSTICK_RIGHT, True)
        if not context.wait_ms(300):
            return
        context.set_held(BIT_LSTICK_RIGHT, False)
        if not context.wait_ms(700):
            return
        if not context.tap(BIT_A, hold_ms=50, gap_ms=0):
            return
        context.set_held_many((BIT_Y, BIT_LSTICK_UP), False)
        if not context.wait_ms(1000):
            return

        # 清平台：间隔 500 ms 点按三次 R，然后等待并调整位置。
        for _ in range(3):
            if not context.tap(BIT_R, hold_ms=50, gap_ms=500):
                return
        if not context.wait_ms(5000):
            return
        if not context.move_stick(BIT_LSTICK_DOWN, duration_ms=150):
            return
        context.center_stick()

        # 结束等待。
        if not context.wait_ms(11000):
            return

        # 结束结算。
        for gap_ms in (1500, 1500, 1500, 1500, 1500, 500):
            if not context.tap(BIT_A, hold_ms=50, gap_ms=gap_ms):
                return
        context.complete_macro_loop()


def _run_macro_3_loop(context: _MacroContext) -> None:
    """宏3（力量）：L 任意、R 砸地、A 卫星；卫星收集蛋。"""
    while not context.stop_event.is_set():
        if not context.run_auto_sell_if_due():
            return
        # 进入地图。
        for bit_index, gap_ms in (
            (BIT_X, 500),
            (BIT_A, 500),
            (BIT_A, 500),
            (BIT_A, 500),
            (BIT_A, 500),
            (BIT_A, 500),
            (BIT_A, 6000),
        ):
            if not context.tap(bit_index, hold_ms=50, gap_ms=gap_ms):
                return

        # 持续推动左摇杆向前，并叠加 ZR、ZL 与左右斜向动作。
        context.set_held(BIT_LSTICK_UP, True)
        context.set_held(BIT_ZR, True)
        if not context.wait_ms(1000):
            return
        context.set_held(BIT_ZR, False)

        context.set_held_many((BIT_ZL, BIT_Y), True)
        if not context.wait_ms(1200):
            return
        context.set_held(BIT_LSTICK_RIGHT, True)
        if not context.wait_ms(500):
            return
        context.set_held(BIT_LSTICK_RIGHT, False)
        if not context.wait_ms(1200):
            return
        context.set_held(BIT_LSTICK_LEFT, True)
        if not context.wait_ms(1000):
            return
        context.set_held(BIT_LSTICK_LEFT, False)
        if not context.wait_ms(1200):
            return
        context.set_held(BIT_LSTICK_RIGHT, True)
        if not context.wait_ms(500):
            return
        context.set_held(BIT_LSTICK_RIGHT, False)
        if not context.wait_ms(500):
            return
        context.set_held_many((BIT_ZL, BIT_Y), False)

        context.set_held(BIT_ZR, True)
        if not context.wait_ms(800):
            return
        context.set_held(BIT_ZR, False)
        context.set_held(BIT_ZL, True)
        if not context.wait_ms(900):
            return
        context.set_held(BIT_ZL, False)
        context.set_held(BIT_LSTICK_UP, False)

        # 连续点按三次 X，再按 B、B、X、B；随后前进并点按 B、R。
        for gap_ms in (50, 50, 1300):
            if not context.tap(BIT_X, hold_ms=50, gap_ms=gap_ms):
                return
        if not context.tap(BIT_B, hold_ms=300, gap_ms=250):
            return
        if not context.tap(BIT_B, hold_ms=50, gap_ms=500):
            return
        if not context.tap(BIT_X, hold_ms=50, gap_ms=500):
            return
        if not context.tap(BIT_B, hold_ms=50, gap_ms=0):
            return
        context.set_held(BIT_LSTICK_UP, True)
        if not context.wait_ms(1800):
            return
        if not context.tap(BIT_B, hold_ms=50, gap_ms=500):
            return
        if not context.tap(BIT_R, hold_ms=50, gap_ms=1800):
            return
        context.set_held(BIT_LSTICK_UP, False)

        # 清平台：连续点按三次 A，然后等待。
        for _ in range(3):
            if not context.tap(BIT_A, hold_ms=50, gap_ms=0):
                return
        if not context.wait_ms(5000):
            return
        if not context.move_stick(BIT_LSTICK_DOWN, duration_ms=300):
            return
        if not context.move_stick(BIT_LSTICK_RIGHT, duration_ms=150):
            return
        context.center_stick()

        # 结束等待。
        if not context.wait_ms(12000):
            return

        # 结束结算。
        for _ in range(6):
            if not context.tap(BIT_A, hold_ms=50, gap_ms=1500):
                return
        context.complete_macro_loop()


def _run_macro_4_loop(context: _MacroContext) -> None:
    """宏4（技术）：L 砸地、R 风扇、A 任意；手动收集蛋。"""
    while not context.stop_event.is_set():
        if not context.run_auto_sell_if_due():
            return
        # 进入地图。
        for bit_index, gap_ms in (
            (BIT_X, 500),
            (BIT_A, 500),
            (BIT_A, 500),
            (BIT_A, 500),
            (BIT_A, 500),
            (BIT_A, 500),
            (BIT_A, 6000),
        ):
            if not context.tap(bit_index, hold_ms=50, gap_ms=gap_ms):
                return

        # 持续推动左摇杆向前，并叠加 ZR、ZL 与左右斜向动作。
        context.set_held(BIT_LSTICK_UP, True)
        context.set_held(BIT_ZR, True)
        if not context.wait_ms(1000):
            return
        context.set_held(BIT_ZR, False)

        context.set_held_many((BIT_ZL, BIT_Y), True)
        if not context.wait_ms(1200):
            return
        context.set_held(BIT_LSTICK_RIGHT, True)
        if not context.wait_ms(500):
            return
        context.set_held(BIT_LSTICK_RIGHT, False)
        if not context.wait_ms(1200):
            return
        context.set_held(BIT_LSTICK_LEFT, True)
        if not context.wait_ms(1000):
            return
        context.set_held(BIT_LSTICK_LEFT, False)
        if not context.wait_ms(1200):
            return
        context.set_held(BIT_LSTICK_RIGHT, True)
        if not context.wait_ms(500):
            return
        context.set_held(BIT_LSTICK_RIGHT, False)
        if not context.wait_ms(500):
            return
        context.set_held_many((BIT_ZL, BIT_Y), False)

        context.set_held(BIT_ZR, True)
        if not context.wait_ms(800):
            return
        context.set_held(BIT_ZR, False)
        context.set_held(BIT_ZL, True)
        if not context.wait_ms(900):
            return
        context.set_held(BIT_ZL, False)
        context.set_held(BIT_LSTICK_UP, False)

        # 呼叫升空：连续点按三次 X，再按 B、B。
        for gap_ms in (50, 50, 1300):
            if not context.tap(BIT_X, hold_ms=50, gap_ms=gap_ms):
                return
        if not context.tap(BIT_B, hold_ms=300, gap_ms=250):
            return
        if not context.tap(BIT_B, hold_ms=50, gap_ms=0):
            return

        # 先保持 Y，短暂等待后再保持左摇杆前。
        context.set_held(BIT_Y, True)
        if not context.wait_ms(50):
            return
        context.set_held(BIT_LSTICK_UP, True)
        if not context.wait_ms(1500):
            return
        if not context.tap(BIT_B, hold_ms=50, gap_ms=500):
            return
        if not context.tap(BIT_R, hold_ms=50, gap_ms=500):
            return
        for direction, duration_ms in (
            (BIT_LSTICK_LEFT, 1000),
            (BIT_LSTICK_RIGHT, 2000),
            (BIT_LSTICK_LEFT, 1000),
        ):
            context.set_held(direction, True)
            if not context.wait_ms(duration_ms):
                return
            context.set_held(direction, False)
        if not context.tap(BIT_L, hold_ms=50, gap_ms=2000):
            return
        context.set_held(BIT_Y, False)
        if not context.wait_ms(50):
            return
        context.set_held(BIT_LSTICK_UP, False)

        # 清平台：全程保持 ZR，依次推动摇杆。
        context.set_held(BIT_ZR, True)
        if not context.tap(BIT_X, hold_ms=50, gap_ms=0):
            return
        for direction, duration_ms in (
            (BIT_LSTICK_LEFT, 2800),
            (BIT_LSTICK_UP, 3800),
        ):
            if not context.move_stick(direction, duration_ms=duration_ms):
                return
        if not context.tap(BIT_X, hold_ms=50, gap_ms=0):
            return
        for direction, duration_ms in (
            (BIT_LSTICK_RIGHT, 4000),
            (BIT_LSTICK_DOWN, 1000),
            (BIT_LSTICK_LEFT, 2600),
            (BIT_LSTICK_DOWN, 1600),
            (BIT_LSTICK_RIGHT, 300),
        ):
            if not context.move_stick(direction, duration_ms=duration_ms):
                return
        context.center_stick()
        context.set_held(BIT_ZR, False)

        # 结束等待。
        if not context.wait_ms(12000):
            return

        # 结束结算。
        for _ in range(6):
            if not context.tap(BIT_A, hold_ms=50, gap_ms=1500):
                return
        context.complete_macro_loop()


def _run_sell_equipment_sequence(context: _MacroContext) -> bool:
    """Run the common sell-equipment sequence before a new map round."""
    for bit_index, gap_ms in (
        (BIT_X, 500),
        (BIT_DPAD_UP, 500),
        (BIT_DPAD_UP, 500),
        (BIT_A, 1500),
        (BIT_DPAD_RIGHT, 500),
        (BIT_A, 500),
        (BIT_X, 500),
        (BIT_A, 500),
        (BIT_DPAD_RIGHT, 500),
        (BIT_A, 500),
        (BIT_DPAD_RIGHT, 500),
        (BIT_A, 500),
        (BIT_DPAD_RIGHT, 500),
        (BIT_A, 500),
        (BIT_DPAD_DOWN, 500),
        (BIT_A, 500),
        (BIT_PLUS, 500),
        (BIT_DPAD_RIGHT, 500),
        (BIT_A, 5000),
        (BIT_B, 500),
        (BIT_B, 2000),
    ):
        if not context.tap(bit_index, hold_ms=50, gap_ms=gap_ms):
            return False
    return True


def _run_macro_5_loop(context: _MacroContext) -> None:
    """宏5（打怪）：天妇罗巢穴长蓝自动。"""
    while not context.stop_event.is_set():
        if not context.run_auto_sell_if_due():
            return

        # 进入地图。
        for bit_index, gap_ms in (
            (BIT_X, 500),
            (BIT_A, 500),
            (BIT_A, 500),
            (BIT_A, 500),
            (BIT_A, 500),
            (BIT_A, 500),
            (BIT_A, 6500),
        ):
            if not context.tap(bit_index, hold_ms=50, gap_ms=gap_ms):
                return

        # 持续保持 ZR：先等待 27 秒，然后在保持期间向前推动 2 秒。
        context.set_held(BIT_ZR, True)
        if not context.wait_ms(27000):
            return
        if not context.move_stick(BIT_LSTICK_UP, duration_ms=2000):
            return
        context.center_stick()
        context.set_held(BIT_ZR, False)

        # 释放 ZR 后继续向前推动左摇杆 2.8 秒。
        if not context.move_stick(BIT_LSTICK_UP, duration_ms=2800):
            return
        context.center_stick()

        # 结束等待。
        if not context.wait_ms(11000):
            return

        # 结束结算：最后一次 A 后只等待 500 ms。
        for gap_ms in (1500, 1500, 1500, 1500, 1500, 500):
            if not context.tap(BIT_A, hold_ms=50, gap_ms=gap_ms):
                return
        context.complete_macro_loop()


def _run_macro_6_part_1(context: _MacroContext) -> bool:
    """Run the first user-triggered Pokopia sequence."""
    for direction in (
        BIT_LSTICK_UP,
        BIT_LSTICK_DOWN,
        BIT_LSTICK_LEFT,
        BIT_LSTICK_RIGHT,
    ):
        if not context.move_stick(direction, duration_ms=100):
            return False
    context.center_stick()
    if not context.wait_ms(3000):
        return False
    for _ in range(3):
        if not context.tap(BIT_B, hold_ms=50, gap_ms=500):
            return False

    first_steps = (
        (BIT_PLUS, 500),
        (BIT_MINUS, 2000),
        (BIT_DPAD_UP, 500),
        (BIT_A, 5000),
        (BIT_HOME, 500),
        (BIT_DPAD_DOWN, 500),
        (BIT_DPAD_LEFT, 500),
        (BIT_DPAD_LEFT, 500),
        (BIT_A, 1500),
        *((BIT_DPAD_DOWN, 500) for _ in range(6)),
        (BIT_A, 500),
        *((BIT_DPAD_DOWN, 500) for _ in range(3)),
        (BIT_A, 500),
        (BIT_A, 3000),
        (BIT_A, 500),
        (BIT_A, 3500),
        (BIT_A, 1000),
        (BIT_A, 4000),
        (BIT_DPAD_UP, 500),
        (BIT_A, 0),
    )
    for bit_index, gap_ms in first_steps:
        if not context.tap(bit_index, hold_ms=50, gap_ms=gap_ms):
            return False
    return True


def _run_macro_6_part_2(context: _MacroContext) -> bool:
    """Run the second user-triggered Pokopia sequence."""
    for bit_index in (BIT_HOME, BIT_A, BIT_A, BIT_A, BIT_A):
        if not context.tap(bit_index, hold_ms=50, gap_ms=500):
            return False
    if not context.wait_ms(45000):
        return False

    # 原序列中的“A;空,20000”按 A 按压 50 ms 处理。
    if not context.tap(BIT_A, hold_ms=50, gap_ms=20000):
        return False
    for _ in range(5):
        if not context.tap(BIT_B, hold_ms=50, gap_ms=500):
            return False

    # 左摇杆向上 100 ms。
    if not context.move_stick(BIT_LSTICK_UP, duration_ms=100):
        return False
    context.center_stick()
    if not context.wait_ms(500):
        return False

    if not context.tap(BIT_A, hold_ms=50, gap_ms=3000):
        return False
    if not context.tap(BIT_DPAD_DOWN, hold_ms=50, gap_ms=500):
        return False
    if not context.tap(BIT_A, hold_ms=50, gap_ms=500):
        return False
    if not context.tap(BIT_A, hold_ms=50, gap_ms=500):
        return False
    if not context.wait_ms(1500):
        return False
    if not context.tap(BIT_A, hold_ms=50, gap_ms=1500):
        return False
    if not context.tap(BIT_A, hold_ms=50, gap_ms=1500):
        return False
    if not context.tap(BIT_PLUS, hold_ms=50, gap_ms=20000):
        return False
    if not context.tap(BIT_A, hold_ms=50, gap_ms=500):
        return False
    return True


def _run_macro_6_triggered(context: _MacroContext) -> None:
    """Wait idle and run either Pokopia part when keyboard 1 or 2 is pressed."""
    _timestamped_log("macro6 已待机：按 1 执行第一段，按 2 执行第二段。")
    part_runners = {
        1: _run_macro_6_part_1,
        2: _run_macro_6_part_2,
    }
    while not context.stop_event.is_set():
        selected_part = 0
        for part_number, trigger in context.macro6_part_triggers.items():
            if trigger.is_set():
                trigger.clear()
                selected_part = part_number
                break
        if not selected_part:
            context.stop_event.wait(0.05)
            continue
        _timestamped_log(f"macro6 开始执行第 {selected_part} 段。")
        if not part_runners[selected_part](context):
            return
        _timestamped_log(f"macro6 第 {selected_part} 段执行完成，返回待机。")


def _run_empty_macro_loop(context: _MacroContext) -> None:
    """Empty profile: wait for quit without sending controller input."""
    while not context.stop_event.wait(0.25):
        pass


MacroLoop = Callable[[_MacroContext], None]
MACRO_PROFILES: Dict[str, MacroLoop] = {
    profile_name: _run_empty_macro_loop
    for profile_name in MACRO_PROFILE_NAMES
}
MACRO_PROFILES["macro1"] = _run_macro_1_loop
MACRO_PROFILES["macro2"] = _run_macro_2_loop
MACRO_PROFILES["macro3"] = _run_macro_3_loop
MACRO_PROFILES["macro4"] = _run_macro_4_loop
MACRO_PROFILES["macro5"] = _run_macro_5_loop
MACRO_PROFILES["macro6"] = _run_macro_6_triggered


def _run_macro_profile(
    profile_name: str,
    context: _MacroContext,
    worker_errors: List[BaseException],
) -> None:
    try:
        if not _run_controller_detection(context):
            return
        if profile_name in AUTO_SELL_MACRO_PROFILES:
            context.enable_auto_sell(profile_name)
        MACRO_PROFILES[profile_name](context)
    except BaseException as exc:
        worker_errors.append(exc)
        context.stop_event.set()
    finally:
        if context.status_callback is not None:
            context.status_callback(True)
            context.status_callback = None
            _commit_status_line()


def _send_manual_controller_pulse(context: _MacroContext, bit_index: int) -> None:
    """Overlay one short manual input without changing the macro's saved state."""
    paused = context.is_paused()
    base_bits = 0 if paused else context.active_bits
    if bit_index in _MANUAL_ARROW_BITS.values():
        output_bits = (base_bits & ~context.stick_direction_mask) | (1 << bit_index)
        hold_ms = 150
    elif bit_index in {BIT_RSTICK_UP, BIT_RSTICK_DOWN, BIT_RSTICK_LEFT, BIT_RSTICK_RIGHT}:
        output_bits = (
            base_bits & ~context.right_stick_direction_mask
        ) | (1 << bit_index)
        hold_ms = 150
    else:
        output_bits = base_bits | (1 << bit_index)
        hold_ms = 80 if bit_index in {BIT_L, BIT_R, BIT_ZL, BIT_ZR} else 50

    context.controller.send_bits(output_bits)
    context.stop_event.wait(hold_ms / 1000.0)
    restore_bits = 0 if context.is_paused() else context.active_bits
    context.controller.send_bits(restore_bits)


def _send_manual_controller_hold(
    context: _MacroContext,
    bit_index: int,
    hold_ms: int,
) -> None:
    """Keep one manual button asserted even while the macro sends other states."""
    bit = 1 << bit_index
    if context.is_paused():
        context.controller.send_bits(bit)
        context.stop_event.wait(max(0, hold_ms) / 1000.0)
        context.controller.send_bits(0)
        return

    context.active_bits |= bit
    context.send_active_bits()
    context.stop_event.wait(max(0, hold_ms) / 1000.0)
    context.active_bits &= ~bit
    context.send_active_bits()


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
    if context.macro_profile_name == "macro6" and normalized in {"1", "2"}:
        part_number = int(normalized)
        context.macro6_part_triggers[part_number].set()
        _timestamped_log(f"已触发 macro6 第 {part_number} 段。")
        return True
    if key in {"'", '"'}:
        _send_manual_controller_hold(context, BIT_CAPTURE, hold_ms=2000)
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


def _listen_for_keyboard_control(
    context: _MacroContext,
    pause_state: _PauseState,
    worker_errors: List[BaseException],
) -> None:
    if not sys.stdin.isatty():
        context.stop_event.wait()
        return

    try:
        while not context.stop_event.is_set():
            if not msvcrt.kbhit():
                context.stop_event.wait(0.05)
                continue
            key = _read_windows_terminal_key()
            if key and not _handle_terminal_control_key(key, context, pause_state):
                return
    except BaseException as exc:
        worker_errors.append(exc)
        context.stop_event.set()


def run_macro_forever(controller: SerialRemoteController, macro_profile: str = "macro1") -> None:
    """Run independent macro and keyboard-listener threads until stopped."""
    if macro_profile not in MACRO_PROFILES:
        raise ValueError(f"未知宏配置：{macro_profile}")

    stop_event = threading.Event()
    pause_state = _PauseState()
    worker_errors: List[BaseException] = []
    context = _MacroContext(controller, stop_event, pause_state)
    context.macro_profile_name = macro_profile
    workers = [
        threading.Thread(
            target=_run_macro_profile,
            args=(macro_profile, context, worker_errors),
            name=f"{macro_profile}-sequence-worker",
            daemon=True,
        ),
        threading.Thread(
            target=_listen_for_keyboard_control,
            args=(context, pause_state, worker_errors),
            name="macro-keyboard-listener",
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

    try:
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
        try:
            if args.macro == "macro1":
                loop_description = "随后循环执行宏1动作"
            elif args.macro == "macro6":
                loop_description = "随后待机；按 1 执行第一段，按 2 执行第二段"
            elif MACRO_PROFILES.get(args.macro) is not _run_empty_macro_loop:
                loop_description = f"随后循环执行{args.macro}动作"
            else:
                loop_description = "随后进入不发送任何输入的空循环"
            _timestamped_log(
                f"宏手柄已启动：{selected_port}，当前配置：{args.macro}。"
                f"先执行一次手柄检测，{loop_description}；"
                "按 P 暂停/恢复，按 Q 或 Ctrl+C 退出。\n"
                "键盘：Z=A X=B A=Y S=X；方向键=左摇杆；"
                "I/J/K/L=右摇杆上/左/下/右；C=L V=R F=ZL G=ZR；"
                "H/Enter=Home；;/:=截图；'/\"=录屏；+/D=Plus，-/E=Minus。"
            )
            run_macro_forever(controller, macro_profile=args.macro)
        finally:
            with contextlib.suppress(Exception):
                controller.release()
            controller.close()
        _timestamped_log("宏手柄已停止。")
    except KeyboardInterrupt:
        _timestamped_log("宏手柄已停止。")
        return 0
    except Exception as exc:
        _timestamped_log(f"启动失败：{exc}", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
