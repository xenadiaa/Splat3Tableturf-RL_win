"""Video-aware macro5 with an unattended stuck-round watchdog.

This script replaces the separate ffplay window.  It captures the configured
DirectShow device through FFmpeg, displays the video, runs macro5, and watches
the upper-center battle progress bar after the combat sequence.  It advances
only when a Clear!! or Result settlement screen is visible.  If the bar is
still present at the round deadline, it takes one short step backward, checks
the 1..6 top-left in-game life icons, and leaves only when life is valid and
the BAR is detected in the same frame.  It then confirms with A and starts the
next X+A entry directly; visual misses retry instead of sending Home.
"""

from __future__ import annotations

import argparse
import base64
import contextlib
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import unicodedata
import zlib
from dataclasses import dataclass
from pathlib import Path
from typing import List

import cv2
import numpy as np


REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from autocontroller_rebuild_for_RL.macro_gamepad import (
    DEFAULT_CONFIG,
    SELL_EQUIPMENT_INTERVAL_SECONDS,
    SerialRemoteController,
    _MacroContext,
    _PauseState,
    _handle_terminal_control_key,
    _listen_for_keyboard_control,
    _load_config,
    _read_windows_terminal_key,
    _resolve_config_path,
    _run_controller_detection,
    _timestamped_log,
    wait_for_serial_selection,
)
from vision_capture.adapter import (
    FFmpegCaptureSource,
    capture_device_busy_message,
    capture_error_may_indicate_device_busy,
    is_usb_capture_device_name,
    list_avfoundation_video_device_rows,
    rank_capture_device_name,
)
from switch_connect.virtual_gamepad.input_mapper import (
    BIT_A,
    BIT_B,
    BIT_DPAD_DOWN,
    BIT_DPAD_RIGHT,
    BIT_DPAD_UP,
    BIT_L,
    BIT_LSTICK_DOWN,
    BIT_LSTICK_RIGHT,
    BIT_LSTICK_UP,
    BIT_PLUS,
    BIT_R,
    BIT_X,
    BIT_ZL,
    BIT_ZR,
)


CAPTURE_WIDTH = 960
CAPTURE_HEIGHT = 540
DEFAULT_CAPTURE_SPEC = "1920x1080 / mjpeg"
DEFAULT_CAPTURE_FPS = 30
DEFAULT_CAPTURE_READ_TIMEOUT_SECONDS = 5.0
CAPTURE_FALLBACK_SPECS = (
    (1920, 1080, "mjpeg"),
    (1280, 720, "mjpeg"),
    (1920, 1080, "uyvy422"),
    (1280, 720, "uyvy422"),
    (1920, 1080, "nv12"),
    (1280, 720, "nv12"),
    (1920, 1080, "yuyv422"),
    (1280, 720, "yuyv422"),
)
ROUND_WATCHDOG_MS = 45_000
CONTINUOUS_BAR_ESCAPE_SECONDS = 55.0
SETTLEMENT_RESULT_WAIT_SECONDS = 30.0
DEADLINE_STATE_SAMPLE_SECONDS = 1.2
STUCK_BACKSTEP_MS = 450
BASE_HUD_YELLOW_RESTART_SECONDS = 0.1
BASE_HUD_WHITE_RESTART_SECONDS = 4.0
RESULT_FIRST_A_READY_DELAY_MS = 800
RESULT_FIRST_A_HOLD_MS = 180
RESULT_A_RETRY_GAP_MS = 650
RESULT_A_MAX_ATTEMPTS = 4
ENTRY_BATTLE_READY_TIMEOUT_MS = 6_500
ENTRY_BATTLE_READY_POLL_MS = 50
LOOT_PAGE_SCAN_SECONDS = 1.2
LOOT_SECOND_SLOT_MOVE_MS = 150
LOOT_SECOND_SLOT_SETTLE_MS = 900
SUPREME_PREFIX_MIN_ORANGE_FRACTION = 0.025
SUPREME_PREFIX_MIN_ORANGE_PIXELS = 60
AUTO_RECOVERY_MACRO_TIMEOUT_SECONDS = 45.0
AUTO_RECOVERY_CAPTURE_TIMEOUT_SECONDS = 15.0
AUTO_RECOVERY_RESTART_DELAY_SECONDS = 3.0
AUTO_RECOVERY_CHILD_ENV = "MACRO5_SUPERVISED_CHILD"
AUTO_RECOVERY_MACRO_HEARTBEAT_ENV = "MACRO5_MACRO_HEARTBEAT"
AUTO_RECOVERY_CAPTURE_HEARTBEAT_ENV = "MACRO5_CAPTURE_HEARTBEAT"
LRA_REPEAT_PRESS_MS = 50
LRA_REPEAT_RELEASE_MS = 50
LRA_REPEAT_TOTAL_MS = 28_000
LRA_FORWARD_START_MS = 25_200
LRA_FORWARD_DURATION_MS = 2_800
ZL_ZR_COMBO_HOLD_MS = 100
PRESS_PROMPT_POST_WAIT_SECONDS = 5.0
PRESS_PROMPT_CONFIRM_FRAMES = 2
PRESS_PROMPT_CLEAR_FRAMES = 3
PRESS_PROMPT_MIN_SCORE = 0.82
PRESS_PROMPT_ROI = (0.835, 0.960, 0.800, 0.945)
PRESS_PROMPT_TEMPLATE_SHAPE = (80, 160)
# Consensus neutral-white glyph mask built from all seven 2026-09-01 title
# screen samples. Runtime recognition is self-contained and never opens Pic.
PRESS_PROMPT_TEMPLATE_ZLIB_B64 = (
    "eNrt1LFuwyAQBmAQVdwpvEBkxr6EFR4Nogx5LSwPfY1UfQGkLhks/t5B4lzSSu3Q"
    "oVKLB58+4MwBslL/7cv2KGJ/GjJUoUfY7F8wYg9h297vx9fp+RClbc2ObJ+utukH"
    "M06pE8OqgSzd21j0LG0zmKRCDB/M3CRkm9Kt9QPZNB3k+txp0NjByDpcHhTXK7/b"
    "cy13e7T5ZN9+2n5107heiSec6rsDtVyvCAWtn6PWHTiqZ8wB5ppkmVEjPgmDZUYzO"
    "jG7mC4BKXAa/0bZQjVKFT2OZFmtlOXMZlY6hsXMxdLVdBFmhXVFWrerhnIxZDJeqc"
    "VsLnZcP+hm2ZK9IHukLRVDq3c4Oh7HFgsNqZZ4bqK5hS6Y5YocIt1lpDWbpU9ylWe"
    "r4+zKSGvjukOz2WTVIjiLlg/nLfdL9C2L0tyd1V3jjoD4536N79CDL6g="
)
# Start a fresh statistics generation.  The former macro5_history.json is no
# longer read or modified, so it remains available as a manual backup.
LOOT_HISTORY_FILENAME = "macro5_history_v2.json"
LOOT_SCREENSHOT_DIRNAME = "macro5_loot_records"
TARGET_LOOT_PREFIX = "绝品"
TARGET_LOOT_NAME = "高辛烷值汽油4K"
LIFE_ICON_TEMPLATE_SHAPE = (32, 25)
LIFE_ICON_TEMPLATE_B64 = (
    "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA"
    "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA"
    "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAABAQEBAQAAAAAA"
    "AAAAAAAAAAAAAAAAAAEBAQEBAQEBAQEAAAAAAAAAAAAAAAAAAQEBAQEBAQEBAQEB"
    "AQAAAAAAAAAAAAAAAQEBAQEBAQEBAQEBAQEBAQAAAAAAAAAAAQEBAQEBAQEBAQEB"
    "AQEBAQEAAAAAAAAAAAEBAAAAAAAAAQEAAAAAAAABAQAAAAAAAAEBAAAAAAAAAAAA"
    "AAAAAAAAAAEBAAAAAAABAAAAAAAAAAAAAAAAAAAAAAAAAQAAAAAAAQAAAQAAAAAA"
    "AQEAAAAAAAEAAAEAAAAAAQEAAQEAAAAAAAEBAAAAAAABAAABAQAAAAEBAAEBAQAA"
    "AAEBAQEAAAABAQAAAQEAAAABAQAAAQEBAQEBAQEBAQEBAQEAAAEBAAAAAQEAAAAB"
    "AQEBAQAAAQEBAQEBAAABAQAAAAABAAAAAAEBAQAAAAABAAAAAAAAAQEAAAAAAQEA"
    "AAAAAAAAAAAAAAAAAAAAAQEAAAAAAAEBAQEAAAAAAAEBAAAAAAAAAQEBAAAAAAAA"
    "AQEBAQEBAQEBAQEBAQEBAQEBAAAAAAAAAAEBAQEBAQEBAQEBAQEBAQEBAAAAAAAA"
    "AAAAAQEBAQEBAQEBAQEBAQEBAQAAAAAAAAAAAAABAQEBAQEBAQEBAQEBAQAAAAAA"
    "AAAAAAAAAAAAAQEBAQEBAQEBAAAAAAAAAAAAAAAAAAAAAAAAAAEBAQEAAAAAAAAA"
    "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA"
    "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA"
    "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA="
)
LIFE_ICON_TEMPLATE_MASK = np.frombuffer(
    base64.b64decode(LIFE_ICON_TEMPLATE_B64),
    dtype=np.uint8,
)[: LIFE_ICON_TEMPLATE_SHAPE[0] * LIFE_ICON_TEMPLATE_SHAPE[1]].reshape(
    LIFE_ICON_TEMPLATE_SHAPE
).astype(bool)
BASE_HUD_TEMPLATE_SHAPE = (76, 96)
BASE_HUD_TEMPLATE_B64 = (
    "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA"
    "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA"
    "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA"
    "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA"
    "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA"
    "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAB4AAAAAAAAAAAAAAP/AAAAAAAA"
    "AAAAAAf/wAAAAAAAAAAAAAf/4D/8MGAAAAAAAA//8D/8//AAAAAAAB4c8BgY//AA"
    "AAAAAB4Y+Bg8xjAAAAAAAD4I+D/8xjAAAAAAAD8A+AAA//AAAAAAAD8B+DMI5zAA"
    "AAAAAD8B+DmYxjAAAAAAAD+B+AGA//AAAAAAAD8B+AGA//AAAAAAAB8R+D/8BgAA"
    "AAAAAB8Q8AGABwAAAAAAAB848A2w//gAAAAAAA//4A2wxwAAAAAAAAf/wDmcBgAA"
    "AAAAAAP/gAEAAAAAAAAAAAD+AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA"
    "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA"
    "AAAAAAA4AAAAAAAAAAAAAAH/AAAAAAAAAAAAAAP/wAAAAAAAAAAAAAf/4DYwMAAA"
    "AAAAAAf/8Bb8P/gAAAAAAA/H8B7+f/gAAAAAAB/D+AYwHPAAAAAAAB/D/B4wD8AA"
    "AAAAAD7D/B54B4AAAAAAAD4AfDb8f/gAAAAAAD4AfADAePgAAAAAAD4AfD/8AAAA"
    "AAAAAB4AeD/+f/gAAAAAAB/D+AYEYxgAAAAAAB/D+AZkYxgAAAAAAA/38A88f/gA"
    "AAAAAA//4D84YxgAAAAAAAf/wDv8f/gAAAAAAAP/wAPcf/AAAAAAAAD/AAAAAAAA"
    "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA"
    "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA"
)
BASE_HUD_TEMPLATE_MASK = np.unpackbits(
    np.frombuffer(base64.b64decode(BASE_HUD_TEMPLATE_B64), dtype=np.uint8)
)[: BASE_HUD_TEMPLATE_SHAPE[0] * BASE_HUD_TEMPLATE_SHAPE[1]].reshape(
    BASE_HUD_TEMPLATE_SHAPE
).astype(bool)
BASE_HUD_ALT_TEMPLATE_B64 = (
    "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA"
    "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA"
    "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA"
    "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA"
    "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA"
    "AAAAAAAAAAAAD//wCAAAAAAAAAAAH//wCAD4AAAAAAAAP//wAAf+AAAAAAAAP7/wAA//"
    "gAAAAAAAf4/wAA//wH/4YMAAx8eAAB//4H/5/+AA58YAADw54DAx/+AAsf4AADwx8DA5"
    "jGAA5fIAAHwR8H/5jGAAwfAAAH4B+AAB/+AAAXAAAP4D8GYZzmAA4AAAAH4D8Hc5jGAA"
    "4IAAAH8D8AMB/+AAYEAAAH4D8AMB/+AAcAAAAD4j8H/4DAAAOAAAAD4h8AMADgAAOAAA"
    "AD5x4Bth//AAPAAAAB//wBth/gAABgAAAA//gHM4DAAAAAAAAAf/AGMYTAAAAAAAAAAA"
    "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA"
    "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAeAAAEAAAAAAABwAB+AAAAAAAAAAAP/AAAAAAAA"
    "AAAAAA//wAAAAAAAAAAAAA//wGxgYAAAAAAAAB//4G38f/AAAAAAAB/n8j38//AAAAAA"
    "AD8H8wxg+eAAAAAAAD+H/zxgH4AAAAAAAD+G/z34DwAAAAAAAHwA+234//AAAAAAAHwA"
    "+2WA8fAAAAAAAHwA+3/8AAAAAAAAAHwH+H/8//AAAAAAAD/H+AwMxjAAAAAAAD/H8A7I"
    "xjAAAAAAAN/P4B54//AAAAAAAd//4H5wxjAAAAAAA+//wHf8//AAAAAAB///gAe8/+AA"
    "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA"
    "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA"
)
BASE_HUD_ALT_TEMPLATE_MASK = np.unpackbits(
    np.frombuffer(base64.b64decode(BASE_HUD_ALT_TEMPLATE_B64), dtype=np.uint8)
)[: BASE_HUD_TEMPLATE_SHAPE[0] * BASE_HUD_TEMPLATE_SHAPE[1]].reshape(
    BASE_HUD_TEMPLATE_SHAPE
).astype(bool)

# The preview draws a thin rectangle around this HUD after detection.  Both
# reference screenshots therefore contain its two vertical borders.  Exclude
# those columns so the detector learns only the two real Switch prompts.
BASE_HUD_VALID_MASK = np.ones(BASE_HUD_TEMPLATE_SHAPE, dtype=bool)
BASE_HUD_VALID_MASK[:, 20:28] = False
BASE_HUD_VALID_MASK[:, 88:96] = False
BASE_HUD_ROW_MASK = np.zeros(BASE_HUD_TEMPLATE_SHAPE, dtype=bool)
BASE_HUD_ROW_MASK[22:42] = True
BASE_HUD_ROW_MASK[48:68] = True
BASE_HUD_TEMPLATE_MASKS = tuple(
    template & BASE_HUD_VALID_MASK & BASE_HUD_ROW_MASK
    for template in (BASE_HUD_TEMPLATE_MASK, BASE_HUD_ALT_TEMPLATE_MASK)
)
CLEAR_TEMPLATE_SHAPE = (77, 157)
CLEAR_TEMPLATE_B64 = (
    "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA"
    "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA"
    "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA"
    "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA"
    "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA"
    "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA"
    "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA"
    "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA"
    "AAAAAAYAAAAAAAAAAAAAAAAAAAAAAA/A/AAAAAAAAAAAHwAAAAAAAAAAfgfgAAAA"
    "AAAAP+D+AAAAAAAAAAPgHwAAAAAAAAf/h/AAAAAAAAAAH4D4AAAAAAAB//g/gAAA"
    "AAAAAAD8B8AAAAAAAA//wfwH+AH/wH8fB+A+AAAAAAAA//4P4H/gD/+D//geAfAA"
    "AAAAAAf/8H8H/8B//x//wPAPgAAAAAAAP+GD+H//A//4f/4HgDwAAAAAAAH8AA/H"
    "//wf/8P/4DwB4AAAAAAAD4AAfj4H4AB+H/8B4A8AAAAAAAD8AAPx8D8D8/D/AA8A"
    "eAAAAAAAB+AAH4+B+D//h/AAeAPAAAAAAAA/AAD8f//H//w/AAAAAAAAAAAAAfgA"
    "B+P//j//4fAAAABgAAAAAAAHwBA/H//B8H8PgAD4B4AAAAAAAB/D4fj+AA+D+HwA"
    "B8B+AAAAAAAA//8Pw/vwfB/D4AB+A/AAAAAAAAP/+H4P/4P//B8AAfAPAAAAAAAA"
    "H//D8H/8D//g+AAPgHgAAAAAAAB/+B+B/+A//wfAADAAAAAAAAAAAD8A/ADwAAAA"
    "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA"
    "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA"
    "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA"
    "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA"
    "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA"
    "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA"
    "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA"
    "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA"
    "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA"
    "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA"
    "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA"
    "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA"
    "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA"
    "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA"
    "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA"
)
CLEAR_TEMPLATE_MASK = np.unpackbits(
    np.frombuffer(base64.b64decode(CLEAR_TEMPLATE_B64), dtype=np.uint8)
)[: CLEAR_TEMPLATE_SHAPE[0] * CLEAR_TEMPLATE_SHAPE[1]].reshape(
    CLEAR_TEMPLATE_SHAPE
).astype(bool)
RESULT_TEMPLATE_SHAPE = (35, 116)
RESULT_TEMPLATE_B64 = (
    "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAgAAAAAD4AAAAAA"
    "AAAAB+AAAAAgBwAAAAAAAAAAfgPwAAP//AAAAAAAAAAH4H8AAD//wAAAAAAAAAB+P/4A"
    "A//8Af4Af/j8Dwfj/+AAPwPgP/AP/4fA8H4f/gAD8D4P/4H/+HwPB+H/4AA/A+H//h//"
    "h8Dwfh/iAAHwPh8H4/wAfA8H4HwAAB8D4fA+P4AHwPB+B8AAAfB+HwPj/AB8DwfgfAAA"
    "H//B+H4//wfA8H4HwAAB//Af/+H/+HwfB+B8AAAf/gH//gP/h8P4fgfgAAH/8B/gAAD4"
    "fn+H4H/gAB//gfwAAB+H//h+A/4AAfP8D/+Af/g//4fgH/AAHw/Af/gH/4H/+H4A/gAB"
    "8HwD/4B/8A/nh8AAAAAfAYAP4ADAAAAAAAAAAAFgAAAAAAAAAAAAAAAAAAAAAAAAAAAA"
    "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA"
    "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA"
    "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA=="
)
RESULT_TEMPLATE_MASK = np.unpackbits(
    np.frombuffer(base64.b64decode(RESULT_TEMPLATE_B64), dtype=np.uint8)
)[: RESULT_TEMPLATE_SHAPE[0] * RESULT_TEMPLATE_SHAPE[1]].reshape(
    RESULT_TEMPLATE_SHAPE
).astype(bool)
PRESS_PROMPT_TEMPLATE_MASK = np.unpackbits(
    np.frombuffer(
        zlib.decompress(
            base64.b64decode(PRESS_PROMPT_TEMPLATE_ZLIB_B64)
        ),
        dtype=np.uint8,
    )
)[: PRESS_PROMPT_TEMPLATE_SHAPE[0] * PRESS_PROMPT_TEMPLATE_SHAPE[1]].reshape(
    PRESS_PROMPT_TEMPLATE_SHAPE
).astype(bool)

# These morphology inputs depend only on the embedded templates.  Building
# them once avoids repeating the same allocations and dilations on every
# preview-analysis frame.
MATCH_KERNEL_3 = np.ones((3, 3), dtype=np.uint8)
CLEAR_KERNEL_5 = np.ones((5, 5), dtype=np.uint8)
RESULT_TEMPLATE_DILATED = cv2.dilate(
    RESULT_TEMPLATE_MASK.astype(np.uint8),
    MATCH_KERNEL_3,
    iterations=1,
).astype(bool)
RESULT_TEMPLATE_PIXEL_COUNT = max(
    1,
    int(np.count_nonzero(RESULT_TEMPLATE_MASK)),
)
PRESS_PROMPT_TEMPLATE_DILATED = cv2.dilate(
    PRESS_PROMPT_TEMPLATE_MASK.astype(np.uint8),
    MATCH_KERNEL_3,
    iterations=1,
).astype(bool)
PRESS_PROMPT_TEMPLATE_PIXEL_COUNT = max(
    1,
    int(np.count_nonzero(PRESS_PROMPT_TEMPLATE_MASK)),
)
BASE_HUD_TEMPLATE_DILATED_MASKS = tuple(
    cv2.dilate(
        template.astype(np.uint8),
        MATCH_KERNEL_3,
        iterations=1,
    ).astype(bool)
    for template in BASE_HUD_TEMPLATE_MASKS
)
LIFE_ICON_TEMPLATE_DILATED = cv2.dilate(
    LIFE_ICON_TEMPLATE_MASK.astype(np.uint8),
    MATCH_KERNEL_3,
    iterations=1,
).astype(bool)
LIFE_ICON_TEMPLATE_PIXEL_COUNT = max(
    1,
    int(np.count_nonzero(LIFE_ICON_TEMPLATE_MASK)),
)


@dataclass(frozen=True)
class FrameSnapshot:
    sequence: int
    frame: np.ndarray | None
    error: str = ""


@dataclass(frozen=True)
class CaptureSettings:
    device_name: str
    device_label: str
    width: int
    height: int
    fps: int
    pixel_format: str
    read_timeout_seconds: float
    fallback_specs: tuple[tuple[int, int, str], ...]


def _parse_capture_spec(spec: str) -> tuple[int, int, str]:
    value = str(spec or "").strip()
    try:
        size, pixel_format = [part.strip() for part in value.split("/", 1)]
        width_text, height_text = [part.strip() for part in size.lower().split("x", 1)]
        width = int(width_text)
        height = int(height_text)
    except (TypeError, ValueError) as exc:
        raise RuntimeError(
            f"无效的视频规格“{value}”；应类似 1920x1080 / mjpeg。"
        ) from exc
    if width <= 0 or height <= 0:
        raise RuntimeError(f"视频规格尺寸必须大于0：{value}")
    return width, height, pixel_format.lower()


def _capture_config_path(runtime_config: dict[str, object]) -> Path:
    configured = str(
        runtime_config.get("frame_api_launch_config")
        or "vision_capture/capture_config.json"
    ).strip()
    path = Path(configured).expanduser()
    return path if path.is_absolute() else REPO_ROOT / path


def _match_capture_device(
    requested: str,
    rows: list[dict[str, str]],
) -> tuple[str, str] | None:
    value = str(requested or "").strip()
    if not value or value.lower() == "invalid":
        return None
    for row in rows:
        if value in {
            str(row.get("index", "")),
            str(row.get("name", "")),
            str(row.get("device_id", "")),
        }:
            return (
                str(row.get("device_id") or row.get("name") or value),
                str(row.get("name") or value),
            )
    return None


def resolve_capture_settings(
    args: argparse.Namespace,
    config_path: Path,
) -> CaptureSettings:
    """Resolve the same device/profile inputs used by the Smart frame server."""
    runtime_config = _load_config(config_path)
    capture_config_path = _capture_config_path(runtime_config)
    capture_config = _load_config(capture_config_path)

    requested_spec = str(
        args.capture_spec or capture_config.get("preview_spec") or ""
    ).strip()
    if requested_spec:
        width, height, pixel_format = _parse_capture_spec(requested_spec)
    else:
        default_width, default_height, default_format = _parse_capture_spec(
            DEFAULT_CAPTURE_SPEC
        )
        width = int(
            capture_config.get("width")
            or runtime_config.get("capture_width")
            or default_width
        )
        height = int(
            capture_config.get("height")
            or runtime_config.get("capture_height")
            or default_height
        )
        pixel_format = str(
            capture_config.get("pixel_format")
            or runtime_config.get("capture_pixel_format")
            or default_format
        ).strip().lower()
        if width <= 0 or height <= 0:
            raise RuntimeError(
                f"视频规格尺寸必须大于0：{width}x{height}"
            )
    fps = int(
        args.capture_fps
        or capture_config.get("fps")
        or runtime_config.get("capture_fps")
        or DEFAULT_CAPTURE_FPS
    )
    if fps <= 0:
        raise RuntimeError(f"采集帧率必须大于0：{fps}")
    read_timeout_seconds = float(
        args.capture_timeout
        or runtime_config.get("capture_read_timeout_seconds")
        or capture_config.get("read_timeout_seconds")
        or DEFAULT_CAPTURE_READ_TIMEOUT_SECONDS
    )
    if read_timeout_seconds <= 0:
        raise RuntimeError(
            f"采集读取超时必须大于0：{read_timeout_seconds}"
        )

    explicit_device = str(args.device_name or "").strip()
    configured_device = str(
        capture_config.get("device_name")
        or runtime_config.get("capture_device_name")
        or ""
    ).strip()
    rows: list[dict[str, str]] = []
    enumeration_error = ""
    try:
        rows = list_avfoundation_video_device_rows()
    except Exception as exc:
        enumeration_error = str(exc)

    selected: tuple[str, str] | None = None
    if explicit_device:
        selected = _match_capture_device(explicit_device, rows)
        if selected is None:
            selected = (explicit_device, explicit_device)
    elif configured_device:
        selected = _match_capture_device(configured_device, rows)
        if selected is None and not rows:
            selected = (configured_device, configured_device)

    if selected is None and rows:
        candidates = [
            row for row in rows
            if is_usb_capture_device_name(str(row.get("name", "")))
        ] or list(rows)
        candidates.sort(
            key=lambda row: rank_capture_device_name(str(row.get("name", ""))),
            reverse=True,
        )
        picked = candidates[0]
        selected = (
            str(picked.get("device_id") or picked.get("name") or ""),
            str(picked.get("name") or picked.get("device_id") or ""),
        )

    if selected is None or not selected[0]:
        suffix = f" FFmpeg设备枚举错误：{enumeration_error}" if enumeration_error else ""
        raise RuntimeError(f"没有找到可用的视频采集设备。{suffix}".strip())

    primary_spec = (width, height, pixel_format)
    unique_specs: list[tuple[int, int, str]] = []
    for candidate in (primary_spec, *CAPTURE_FALLBACK_SPECS):
        normalized = (int(candidate[0]), int(candidate[1]), str(candidate[2]))
        if normalized not in unique_specs:
            unique_specs.append(normalized)

    return CaptureSettings(
        device_name=selected[0],
        device_label=selected[1],
        width=width,
        height=height,
        fps=fps,
        pixel_format=pixel_format,
        read_timeout_seconds=read_timeout_seconds,
        fallback_specs=tuple(unique_specs[1:]),
    )


class _RestartRequested(Exception):
    """Internal control-flow signal used to restart macro5 from X."""


class _HeartbeatFile:
    """Throttled cross-process heartbeat used by the hard-restart supervisor."""

    def __init__(self, environment_name: str) -> None:
        raw_path = os.environ.get(environment_name, "").strip()
        self.path = Path(raw_path) if raw_path else None
        self._lock = threading.Lock()
        self._last_write = 0.0

    def beat(self, *, force: bool = False) -> None:
        if self.path is None:
            return
        now = time.monotonic()
        with self._lock:
            if not force and now - self._last_write < 0.5:
                return
            try:
                self.path.parent.mkdir(parents=True, exist_ok=True)
                self.path.touch(exist_ok=True)
                self._last_write = now
            except OSError:
                # Losing telemetry must not interrupt controller input.  The
                # outer supervisor will recover the process if needed.
                pass


class _RestartableMacroContext(_MacroContext):
    """Macro context whose waits can be interrupted by the R hotkey."""

    def __init__(
        self,
        controller: SerialRemoteController,
        stop_event: threading.Event,
        pause_state: _PauseState,
        restart_event: threading.Event,
        heartbeat: _HeartbeatFile,
        external_hold_event: threading.Event,
        external_hold_ack_event: threading.Event,
    ) -> None:
        super().__init__(controller, stop_event, pause_state)
        self.restart_event = restart_event
        self.heartbeat = heartbeat
        self.external_hold_event = external_hold_event
        self.external_hold_ack_event = external_hold_ack_event

    def send_active_bits(self) -> None:
        self.heartbeat.beat()
        super().send_active_bits()

    def _raise_if_restart_requested(self) -> None:
        if self.restart_event.is_set():
            raise _RestartRequested

    def _service_external_hold(self) -> bool:
        """Freeze the current macro position during title-screen recovery."""
        if not self.external_hold_event.is_set():
            return True
        active_started = self.pause_state.active_monotonic()
        self.controller.release()
        self.external_hold_ack_event.set()
        try:
            while self.external_hold_event.is_set():
                self.heartbeat.beat()
                self._raise_if_restart_requested()
                if self.stop_event.wait(0.05):
                    return False
                if self.status_callback is not None:
                    self.status_callback(False)
        finally:
            active_elapsed = max(
                0.0,
                self.pause_state.active_monotonic() - active_started,
            )
            self.pause_state.exclude_elapsed(active_elapsed)
            self.external_hold_ack_event.clear()
        self._raise_if_restart_requested()
        if not self.pause_state.is_paused():
            self.send_active_bits()
        return not self.stop_event.is_set()

    def _raw_wait_ms(self, duration_ms: int) -> bool:
        remaining = max(0, duration_ms) / 1000.0
        while remaining > 0:
            self.heartbeat.beat()
            self._raise_if_restart_requested()
            if not self._service_external_hold():
                return False
            if self.stop_event.is_set():
                return False
            wait_seconds = min(remaining, 0.05)
            started = time.monotonic()
            if self.restart_event.wait(wait_seconds):
                raise _RestartRequested
            remaining -= max(0.0, time.monotonic() - started)
        self._raise_if_restart_requested()
        return not self.stop_event.is_set()

    def _service_pause(self) -> bool:
        self.heartbeat.beat()
        self._raise_if_restart_requested()
        if not (
            self.pause_state.is_paused()
            or self.pause_state.has_resume_detection()
        ):
            return True

        saved_bits = self.active_bits
        self.controller.release()
        while True:
            while self.pause_state.is_paused():
                self.heartbeat.beat()
                if self.status_callback is not None:
                    self.status_callback(False)
                if self.stop_event.is_set():
                    return False
                if self.restart_event.wait(0.05):
                    raise _RestartRequested

            self._raise_if_restart_requested()
            if self.pause_state.consume_resume_detection():
                _timestamped_log("宏已恢复，重新执行手柄检测后继续原序列。")
                self.active_bits = 0
                detection_started = time.monotonic()
                detection_completed = self._run_resume_detection()
                self.pause_state.exclude_elapsed(
                    time.monotonic() - detection_started
                )
                if not detection_completed:
                    return False
                if self.pause_state.is_paused():
                    self.controller.release()
                    continue
            break

        self._raise_if_restart_requested()
        self.active_bits = saved_bits
        self.send_active_bits()
        return True

    def wait_ms(self, duration_ms: int) -> bool:
        remaining = max(0, duration_ms) / 1000.0
        while remaining > 0:
            self.heartbeat.beat()
            self._raise_if_restart_requested()
            if not self._service_external_hold():
                return False
            if not self._service_pause():
                return False
            if self.status_callback is not None:
                self.status_callback(False)
            started_wait = self.active_monotonic()
            poll_seconds = min(remaining, 0.05)
            if self.stop_event.is_set():
                return False
            if self.restart_event.wait(poll_seconds):
                raise _RestartRequested
            if not self.pause_state.is_paused():
                remaining -= max(0.0, self.active_monotonic() - started_wait)
        self._raise_if_restart_requested()
        if not self._service_external_hold():
            return False
        return self._service_pause()


@dataclass
class WatchdogLoopState:
    next_sell_active: float
    round_count: int = 0
    failure_count: int = 0


class ContinuousBarMonitor:
    """Thread-safe duration of uninterrupted positive BAR observations."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._detected_since: float | None = None

    def reset(self) -> None:
        with self._lock:
            self._detected_since = None

    def update(self, detected: bool, now_active: float) -> None:
        with self._lock:
            if detected:
                if self._detected_since is None:
                    self._detected_since = now_active
            else:
                self._detected_since = None

    def elapsed(self, now_active: float) -> float:
        with self._lock:
            if self._detected_since is None:
                return 0.0
            return max(0.0, now_active - self._detected_since)


@dataclass(frozen=True)
class LootPageDetection:
    detected: bool
    panel_fraction: float
    detail_fraction: float
    header_fraction: float
    slot1_rainbow: bool
    slot2_rainbow: bool
    slot1_score: float
    slot2_score: float


@dataclass(frozen=True)
class InventoryFullDetection:
    detected: bool
    panel_fraction: float
    first_line_fraction: float
    second_line_fraction: float
    third_line_fraction: float
    icon_fraction: float


@dataclass(frozen=True)
class WhiteIconDetection:
    detected: bool
    life_count: int = 0
    slot_scores: tuple[float, ...] = ()
    slot_white_fractions: tuple[float, ...] = ()
    slot_present: tuple[bool, ...] = ()


@dataclass(frozen=True)
class BarDetection:
    detected: bool
    aligned_tick_count: int
    candidate_count: int
    dark_fraction: float


@dataclass(frozen=True)
class ShiverPageDetection:
    detected: bool
    purple_fraction: float
    teal_left_fraction: float
    teal_right_fraction: float


@dataclass(frozen=True)
class BaseHudDetection:
    detected: bool
    menu_fraction: float
    equipment_fraction: float
    menu_match: float
    equipment_match: float
    variant: str | None


@dataclass(frozen=True)
class ClearScreenDetection:
    detected: bool
    yellow_fraction: float
    template_recall: float


@dataclass(frozen=True)
class ResultScreenDetection:
    detected: bool
    white_fraction: float
    template_match: float


@dataclass(frozen=True)
class PressPromptDetection:
    detected: bool
    white_fraction: float
    template_match: float


class Macro5History:
    """Persist all-time round counters and rare-loot records beside the script."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self._lock = threading.Lock()
        self._session_started_at = time.strftime("%Y-%m-%d %H:%M:%S")
        self._session_rounds = 0
        self._session_failures = 0
        self._session_drops: list[dict[str, object]] = []
        self._session_round_records: list[dict[str, object]] = []
        self._pending_round_drops: list[dict[str, object]] = []
        self._save_error_reported = False
        self._data = self._load()

    @staticmethod
    def _new_data() -> dict[str, object]:
        now_text = time.strftime("%Y-%m-%d %H:%M:%S")
        return {
            "schema_version": 2,
            "tracking_started_at": now_text,
            "total_rounds": 0,
            "total_failures": 0,
            "rainbow_drops": [],
            "rainbow_round_tracking_started_at": now_text,
            "rainbow_round_tracked_rounds": 0,
            "rainbow_round_tracked_successes": 0,
            "rainbow_rounds": 0,
        }

    def _load(self) -> dict[str, object]:
        if not self.path.exists():
            return self._new_data()
        try:
            loaded = json.loads(self.path.read_text(encoding="utf-8"))
            if not isinstance(loaded, dict):
                raise ValueError("history root is not an object")
            loaded.setdefault("schema_version", 1)
            loaded.setdefault("tracking_started_at", "未知")
            loaded.setdefault("total_rounds", 0)
            loaded.setdefault("total_failures", 0)
            loaded.setdefault("rainbow_drops", [])
            loaded.setdefault(
                "rainbow_round_tracking_started_at",
                time.strftime("%Y-%m-%d %H:%M:%S"),
            )
            loaded.setdefault("rainbow_round_tracked_rounds", 0)
            loaded.setdefault("rainbow_round_tracked_successes", 0)
            loaded.setdefault("rainbow_rounds", 0)
            loaded["schema_version"] = 2
            return loaded
        except Exception as exc:
            backup = self.path.with_name(
                f"{self.path.stem}.broken-{time.strftime('%Y%m%d-%H%M%S')}"
                f"{self.path.suffix}"
            )
            with contextlib.suppress(Exception):
                shutil.copy2(self.path, backup)
            _timestamped_log(
                f"历史记录读取失败，已从0重新记录；旧文件备份为 {backup.name}：{exc}"
            )
            return self._new_data()

    def _save_locked(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.path.with_suffix(self.path.suffix + ".tmp")
        temporary.write_text(
            json.dumps(self._data, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        os.replace(temporary, self.path)

    def _try_save_locked(self) -> None:
        try:
            self._save_locked()
            self._save_error_reported = False
        except Exception as exc:
            if not self._save_error_reported:
                self._save_error_reported = True
                _timestamped_log(
                    f"永久记录暂时无法写入磁盘，宏会继续运行：{exc}"
                )

    def record_round(self, failed: bool) -> None:
        with self._lock:
            self._session_rounds += 1
            round_drops = [dict(item) for item in self._pending_round_drops]
            self._pending_round_drops.clear()
            self._session_round_records.append(
                {
                    "number": self._session_rounds,
                    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                    "failed": bool(failed),
                    "drops": round_drops,
                }
            )
            self._data["total_rounds"] = int(
                self._data.get("total_rounds", 0)
            ) + 1
            if failed:
                self._session_failures += 1
                self._data["total_failures"] = int(
                    self._data.get("total_failures", 0)
                ) + 1
            self._data["rainbow_round_tracked_rounds"] = int(
                self._data.get("rainbow_round_tracked_rounds", 0)
            ) + 1
            if not failed:
                self._data["rainbow_round_tracked_successes"] = int(
                    self._data.get("rainbow_round_tracked_successes", 0)
                ) + 1
            if round_drops:
                self._data["rainbow_rounds"] = int(
                    self._data.get("rainbow_rounds", 0)
                ) + 1
            self._try_save_locked()

    def record_drop(self, record: dict[str, object]) -> None:
        with self._lock:
            drops = self._data.setdefault("rainbow_drops", [])
            if not isinstance(drops, list):
                drops = []
                self._data["rainbow_drops"] = drops
            drops.append(record)
            copied_record = dict(record)
            self._session_drops.append(copied_record)
            self._pending_round_drops.append(copied_record)
            self._try_save_locked()

    def totals(self) -> tuple[int, int, int]:
        with self._lock:
            rounds = int(self._data.get("total_rounds", 0))
            failures = int(self._data.get("total_failures", 0))
            drops = self._data.get("rainbow_drops", [])
            drop_count = len(drops) if isinstance(drops, list) else 0
            return rounds, failures, drop_count

    def session_totals(self) -> tuple[int, int, int, int, int]:
        """Return launch rounds, failures, rainbow rounds/items, supreme items."""
        with self._lock:
            rainbow_rounds = sum(
                bool(item.get("drops")) for item in self._session_round_records
            )
            supreme_items = sum(
                item.get("supreme") is True for item in self._session_drops
            )
            return (
                self._session_rounds,
                self._session_failures,
                rainbow_rounds,
                len(self._session_drops),
                supreme_items,
            )

    @staticmethod
    def _rate(numerator: int, denominator: int) -> float:
        return 100.0 * numerator / denominator if denominator else 0.0

    def print_session_summary(self) -> None:
        """Print this launch's totals and only rounds that produced rare loot."""
        with self._lock:
            session_started_at = self._session_started_at
            session_rounds = self._session_rounds
            session_failures = self._session_failures
            session_drops = [dict(item) for item in self._session_drops]
            round_records = [
                {
                    **dict(item),
                    "drops": [
                        dict(drop)
                        for drop in item.get("drops", [])
                        if isinstance(drop, dict)
                    ],
                }
                for item in self._session_round_records
            ]

        session_successes = max(0, session_rounds - session_failures)
        rainbow_rounds = sum(bool(item.get("drops")) for item in round_records)
        failure_rate = self._rate(session_failures, session_rounds)
        success_rainbow_rate = self._rate(rainbow_rounds, session_successes)
        all_round_rainbow_rate = self._rate(rainbow_rounds, session_rounds)
        supreme_items = sum(item.get("supreme") is True for item in session_drops)
        known_supreme_items = sum("supreme" in item for item in session_drops)
        supreme_rate = self._rate(supreme_items, known_supreme_items)
        _timestamped_log("========== 本次启动记录（L） ==========")
        _timestamped_log(f"启动时间：{session_started_at}")
        _timestamped_log(
            f"本次共{session_rounds}轮｜成功{session_successes}｜"
            f"失败{session_failures}｜失败率{failure_rate:.2f}%"
        )
        _timestamped_log(
            f"彩装轮数{rainbow_rounds}｜彩装{len(session_drops)}件｜"
            f"成功轮彩装率{success_rainbow_rate:.3f}%｜"
            f"全部轮彩装率{all_round_rainbow_rate:.3f}%"
        )
        _timestamped_log(
            f"绝品{supreme_items}件｜绝品率{supreme_rate:.3f}%"
            f"（绝品件数/已判定彩装{known_supreme_items}件）"
        )
        if not round_records:
            _timestamped_log("本次启动尚未完成任何一轮。")
        loot_round_records = [item for item in round_records if item.get("drops")]
        if round_records and not loot_round_records:
            _timestamped_log("本次启动暂未出彩装（未出货轮次已省略）。")
        for item in loot_round_records:
            status = "失败" if item.get("failed") else "成功"
            drops = item.get("drops", [])
            drop_text = "；".join(
                (
                    f"第{drop.get('slot', '?')}格 "
                    f"{drop.get('name', '名称未识别')}"
                    f"{'【绝品】' if drop.get('supreme') is True else ''}"
                    f"{'【目标出货】' if drop.get('target') else ''}"
                )
                for drop in drops
            )
            _timestamped_log(
                f"本次#{item.get('number', '?')} "
                f"{item.get('timestamp', '未知时间')}｜{status}｜{drop_text}"
            )
        _timestamped_log("=====================================")

    def print_all_time_summary(self) -> None:
        """Print persistent totals and the complete historical loot list."""
        with self._lock:
            tracking_started_at = str(
                self._data.get("tracking_started_at", "未知")
            )
            rainbow_round_tracking_started_at = str(
                self._data.get("rainbow_round_tracking_started_at", "未知")
            )
            total_rounds = int(self._data.get("total_rounds", 0))
            total_failures = int(self._data.get("total_failures", 0))
            tracked_rounds = int(
                self._data.get("rainbow_round_tracked_rounds", 0)
            )
            tracked_successes = int(
                self._data.get("rainbow_round_tracked_successes", 0)
            )
            rainbow_rounds = int(self._data.get("rainbow_rounds", 0))
            all_drops_raw = self._data.get("rainbow_drops", [])
            all_drops = (
                [dict(item) for item in all_drops_raw if isinstance(item, dict)]
                if isinstance(all_drops_raw, list)
                else []
            )

        total_successes = max(0, total_rounds - total_failures)
        failure_rate = self._rate(total_failures, total_rounds)
        success_drop_rate = self._rate(len(all_drops), total_successes)
        all_round_drop_rate = self._rate(len(all_drops), total_rounds)
        tracked_success_rainbow_rate = self._rate(
            rainbow_rounds,
            tracked_successes,
        )
        tracked_all_rainbow_rate = self._rate(rainbow_rounds, tracked_rounds)
        known_supreme_drops = [item for item in all_drops if "supreme" in item]
        supreme_drops = [
            item for item in known_supreme_drops if item.get("supreme") is True
        ]
        unknown_supreme_count = len(all_drops) - len(known_supreme_drops)
        supreme_rate = self._rate(len(supreme_drops), len(known_supreme_drops))
        estimated_hours = total_rounds * 71.0 / 3600.0
        _timestamped_log("========== 永久累计记录（H） ==========")
        _timestamped_log(f"开始统计时间：{tracking_started_at}")
        _timestamped_log(
            f"永久累计：{total_rounds}轮，成功{total_successes}轮，"
            f"失败{total_failures}轮，失败率{failure_rate:.1f}%，"
            f"约{estimated_hours:.1f}小时（按每轮71秒估算）。"
        )
        _timestamped_log(
            f"累计彩装{len(all_drops)}件｜"
            f"彩装件数/成功轮{success_drop_rate:.3f}%｜"
            f"彩装件数/全部轮{all_round_drop_rate:.3f}%"
        )
        _timestamped_log(
            f"准确彩装轮率（从{rainbow_round_tracking_started_at}起）："
            f"统计{tracked_rounds}轮，彩装轮{rainbow_rounds}｜"
            f"成功轮彩装率{tracked_success_rainbow_rate:.3f}%｜"
            f"全部轮彩装率{tracked_all_rainbow_rate:.3f}%"
        )
        _timestamped_log(
            f"累计绝品{len(supreme_drops)}件｜绝品率{supreme_rate:.3f}%"
            f"（绝品件数/已判定彩装{len(known_supreme_drops)}件）｜"
            f"旧记录未判定{unknown_supreme_count}件"
        )
        if not all_drops:
            _timestamped_log("累计彩装：尚未记录到彩装。")
        else:
            _timestamped_log("累计出货清单：")
            for index, item in enumerate(all_drops, 1):
                target_mark = "【目标出货】" if item.get("target") else ""
                supreme_mark = (
                    "【绝品】" if item.get("supreme") is True
                    else "【非绝品】" if item.get("supreme") is False
                    else "【绝品未判定】"
                )
                _timestamped_log(
                    f"#{index} {item.get('timestamp', '未知时间')} "
                    f"第{item.get('slot', '?')}格 "
                    f"{item.get('name', '名称未识别')}"
                    f"{supreme_mark}{target_mark} "
                    f"截图：{item.get('screenshot', '未保存')}"
                )
        target_records = [item for item in all_drops if item.get("target")]
        if target_records:
            _timestamped_log(
                f"目标“{TARGET_LOOT_PREFIX}{TARGET_LOOT_NAME}”已出货 "
                f"{len(target_records)}次！"
            )
        else:
            _timestamped_log(
                f"目标“{TARGET_LOOT_PREFIX}{TARGET_LOOT_NAME}”：尚未出货。"
            )
        _timestamped_log("=====================================")

    def print_summary(self) -> None:
        """Backward-compatible alias for the new session-only L report."""
        self.print_session_summary()

    @staticmethod
    def _prompt_nonnegative_int(label: str, current: int) -> int | None:
        raw = input(f"{label}（当前{current}，直接回车取消）：").strip()
        if not raw:
            return None
        try:
            value = int(raw)
        except ValueError:
            print("请输入0或正整数。")
            return None
        if value < 0:
            print("不能输入负数。")
            return None
        return value

    def _editor_list_drops(self) -> list[dict[str, object]]:
        with self._lock:
            drops_raw = self._data.get("rainbow_drops", [])
            drops = (
                [dict(item) for item in drops_raw if isinstance(item, dict)]
                if isinstance(drops_raw, list)
                else []
            )
        if not drops:
            print("永久出货清单为空。")
            return []
        print("\n永久出货清单：")
        for index, item in enumerate(drops, 1):
            target_mark = "【目标】" if item.get("target") else ""
            supreme_mark = (
                "【绝品】" if item.get("supreme") is True
                else "【非绝品】" if item.get("supreme") is False
                else "【绝品未判定】"
            )
            print(
                f"  {index}. {item.get('timestamp', '未知时间')}｜"
                f"第{item.get('slot', '?')}格｜"
                f"{item.get('name', '名称未识别')}{supreme_mark}{target_mark}"
            )
        return drops

    def edit_history_interactively(self) -> None:
        """Edit persistent counters and loot records without controller input."""
        if not sys.stdin.isatty():
            _timestamped_log("统计编辑器只能在可输入的CMD窗口中打开。")
            return

        while True:
            print(
                "\n========== 永久统计编辑器（E） ==========\n"
                "1. 修改总轮数/失败数/彩装轮数\n"
                "2. 手动添加一条彩装记录\n"
                "3. 修改彩装名称/绝品标记/目标标记\n"
                "4. 删除一条误识别彩装\n"
                "5. 查看当前永久统计\n"
                "6. 清空全部统计（包括本次启动）\n"
                "0. 保存并退出编辑器\n"
                "========================================="
            )
            choice = input("请选择：").strip().lower()
            if choice in {"0", "q", "quit", "exit"}:
                _timestamped_log("统计编辑器已关闭。")
                return

            if choice == "1":
                with self._lock:
                    rounds = int(self._data.get("total_rounds", 0))
                    failures = int(self._data.get("total_failures", 0))
                    rainbow_rounds = int(self._data.get("rainbow_rounds", 0))
                new_rounds = self._prompt_nonnegative_int("永久总轮数", rounds)
                if new_rounds is None:
                    continue
                new_failures = self._prompt_nonnegative_int(
                    "永久失败数",
                    failures,
                )
                if new_failures is None:
                    continue
                if new_failures > new_rounds:
                    print("失败数不能超过总轮数，本次修改取消。")
                    continue
                new_rainbow_rounds = self._prompt_nonnegative_int(
                    "彩装轮数",
                    rainbow_rounds,
                )
                if new_rainbow_rounds is None:
                    continue
                successes = new_rounds - new_failures
                if new_rainbow_rounds > successes:
                    print("彩装轮数不能超过成功轮数，本次修改取消。")
                    continue
                with self._lock:
                    self._data["total_rounds"] = new_rounds
                    self._data["total_failures"] = new_failures
                    self._data["rainbow_round_tracked_rounds"] = new_rounds
                    self._data["rainbow_round_tracked_successes"] = successes
                    self._data["rainbow_rounds"] = new_rainbow_rounds
                    self._try_save_locked()
                print("永久轮次统计已保存。")
                continue

            if choice == "2":
                name = input("彩装名称（可写“名称未识别”）：").strip()
                if not name:
                    print("名称为空，已取消。")
                    continue
                slot_raw = input("格子编号（默认1）：").strip()
                try:
                    slot = int(slot_raw) if slot_raw else 1
                except ValueError:
                    print("格子编号无效，已取消。")
                    continue
                target = input("这是目标出货吗？(y/N)：").strip().lower() == "y"
                supreme = input("这是绝品吗？(y/N)：").strip().lower() == "y"
                record = {
                    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                    "slot": max(1, slot),
                    "name": name,
                    "ocr_text": "手动录入",
                    "target": target,
                    "supreme": supreme,
                    "supreme_score": 1.0 if supreme else 0.0,
                    "rainbow_score": 1.0,
                    "screenshot": "手动录入",
                    "name_crop": "手动录入",
                }
                with self._lock:
                    drops = self._data.setdefault("rainbow_drops", [])
                    if not isinstance(drops, list):
                        drops = []
                        self._data["rainbow_drops"] = drops
                    drops.append(record)
                    self._try_save_locked()
                print("彩装记录已添加；如需调整彩装轮数，请用选项1。")
                continue

            if choice == "3":
                drops = self._editor_list_drops()
                if not drops:
                    continue
                index_value = self._prompt_nonnegative_int("要修改的编号", 0)
                if index_value is None or not 1 <= index_value <= len(drops):
                    print("编号无效。")
                    continue
                selected = drops[index_value - 1]
                old_name = str(selected.get("name", "名称未识别"))
                new_name = input(
                    f"新名称（当前“{old_name}”，直接回车保持）："
                ).strip()
                target_raw = input(
                    "目标标记：y=是，n=否，直接回车保持："
                ).strip().lower()
                supreme_raw = input(
                    "绝品标记：y=是，n=否，直接回车保持："
                ).strip().lower()
                with self._lock:
                    stored = self._data.get("rainbow_drops", [])
                    if not isinstance(stored, list) or not (
                        1 <= index_value <= len(stored)
                    ):
                        print("记录已变化，请重新打开编辑器。")
                        continue
                    record = stored[index_value - 1]
                    if not isinstance(record, dict):
                        print("该记录格式异常，无法修改。")
                        continue
                    if new_name:
                        record["name"] = new_name
                        record["ocr_text"] = "手动修正"
                    if target_raw in {"y", "n"}:
                        record["target"] = target_raw == "y"
                    if supreme_raw in {"y", "n"}:
                        record["supreme"] = supreme_raw == "y"
                        record["supreme_score"] = (
                            1.0 if record["supreme"] else 0.0
                        )
                    self._try_save_locked()
                print("彩装记录已修改。")
                continue

            if choice == "4":
                drops = self._editor_list_drops()
                if not drops:
                    continue
                index_value = self._prompt_nonnegative_int("要删除的编号", 0)
                if index_value is None or not 1 <= index_value <= len(drops):
                    print("编号无效。")
                    continue
                confirm = input(
                    f"确认删除第{index_value}条？输入YES："
                ).strip()
                if confirm != "YES":
                    print("已取消删除。")
                    continue
                with self._lock:
                    stored = self._data.get("rainbow_drops", [])
                    if not isinstance(stored, list) or not (
                        1 <= index_value <= len(stored)
                    ):
                        print("记录已变化，请重新操作。")
                        continue
                    del stored[index_value - 1]
                    self._try_save_locked()
                print("该条彩装记录已删除；必要时请用选项1调整彩装轮数。")
                continue

            if choice == "5":
                self.print_all_time_summary()
                continue

            if choice == "6":
                confirm = input(
                    "这会把永久统计和本次统计全部归零。输入YES确认："
                ).strip()
                if confirm != "YES":
                    print("已取消清空。")
                    continue
                with self._lock:
                    self._data = self._new_data()
                    self._session_started_at = time.strftime(
                        "%Y-%m-%d %H:%M:%S"
                    )
                    self._session_rounds = 0
                    self._session_failures = 0
                    self._session_drops.clear()
                    self._session_round_records.clear()
                    self._pending_round_drops.clear()
                    self._try_save_locked()
                print("全部统计已清空并从0重新开始。")
                continue

            print("没有这个选项。")


def _fractional_crop(
    frame: np.ndarray,
    x0: float,
    x1: float,
    y0: float,
    y1: float,
) -> np.ndarray:
    height, width = frame.shape[:2]
    return frame[
        max(0, int(height * y0)):min(height, int(height * y1)),
        max(0, int(width * x0)):min(width, int(width * x1)),
    ]


def detect_press_prompt(frame: np.ndarray) -> PressPromptDetection:
    """Detect the fixed lower-right title-screen ``press ZL + ZR`` glyphs."""
    empty = PressPromptDetection(False, 0.0, 0.0)
    if frame is None or frame.size == 0 or frame.ndim != 3:
        return empty
    roi = _fractional_crop(frame, *PRESS_PROMPT_ROI)
    if roi.size == 0:
        return empty
    normalized = cv2.resize(
        roi,
        (
            PRESS_PROMPT_TEMPLATE_SHAPE[1],
            PRESS_PROMPT_TEMPLATE_SHAPE[0],
        ),
        interpolation=cv2.INTER_AREA,
    )
    channel_min = normalized.min(axis=2)
    channel_max = normalized.max(axis=2)
    candidate = (
        (channel_min >= 185)
        & (
            channel_max.astype(np.int16)
            - channel_min.astype(np.int16)
            <= 55
        )
    )
    white_fraction = float(np.mean(candidate))
    candidate_dilated = cv2.dilate(
        candidate.astype(np.uint8),
        MATCH_KERNEL_3,
        iterations=1,
    ).astype(bool)
    recall = float(
        np.count_nonzero(
            PRESS_PROMPT_TEMPLATE_MASK & candidate_dilated
        )
        / PRESS_PROMPT_TEMPLATE_PIXEL_COUNT
    )
    precision = float(
        np.count_nonzero(candidate & PRESS_PROMPT_TEMPLATE_DILATED)
        / max(1, int(np.count_nonzero(candidate)))
    )
    template_match = min(recall, precision)
    return PressPromptDetection(
        template_match >= PRESS_PROMPT_MIN_SCORE,
        white_fraction,
        template_match,
    )


def detect_inventory_full_popup(frame: np.ndarray) -> InventoryFullDetection:
    """Detect the fixed weapon-inventory-full warning dialog.

    The dialog is identified by its large central teal panel, three fixed
    bands of bright neutral text and the red/orange weapon-box illustration.
    Requiring all of these features prevents an ordinary teal game screen
    from being mistaken for the warning.
    """
    empty = InventoryFullDetection(False, 0.0, 0.0, 0.0, 0.0, 0.0)
    if frame is None or frame.size == 0 or frame.ndim != 3:
        return empty

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    panel_roi = _fractional_crop(hsv, 0.20, 0.80, 0.12, 0.89)
    if panel_roi.size == 0:
        return empty
    panel_hue, panel_saturation, panel_value = cv2.split(panel_roi)
    panel_fraction = float(
        np.mean(
            (panel_hue >= 52)
            & (panel_hue <= 102)
            & (panel_saturation >= 55)
            & (panel_value >= 30)
        )
    )

    def neutral_white_fraction(
        x0: float,
        x1: float,
        y0: float,
        y1: float,
    ) -> float:
        roi = _fractional_crop(frame, x0, x1, y0, y1)
        if roi.size == 0:
            return 0.0
        channel_min = np.min(roi, axis=2)
        channel_max = np.max(roi, axis=2)
        return float(
            np.mean((channel_min >= 150) & ((channel_max - channel_min) <= 70))
        )

    first_line_fraction = neutral_white_fraction(0.40, 0.60, 0.52, 0.59)
    second_line_fraction = neutral_white_fraction(0.35, 0.65, 0.58, 0.67)
    third_line_fraction = neutral_white_fraction(0.32, 0.68, 0.69, 0.78)

    icon_roi = _fractional_crop(hsv, 0.40, 0.60, 0.25, 0.49)
    if icon_roi.size:
        icon_hue, icon_saturation, icon_value = cv2.split(icon_roi)
        icon_fraction = float(
            np.mean(
                ((icon_hue <= 15) | (icon_hue >= 170))
                & (icon_saturation >= 120)
                & (icon_value >= 120)
            )
        )
    else:
        icon_fraction = 0.0

    detected = (
        panel_fraction >= 0.75
        and first_line_fraction >= 0.05
        and second_line_fraction >= 0.05
        and third_line_fraction >= 0.05
        and icon_fraction >= 0.025
    )
    return InventoryFullDetection(
        detected=detected,
        panel_fraction=panel_fraction,
        first_line_fraction=first_line_fraction,
        second_line_fraction=second_line_fraction,
        third_line_fraction=third_line_fraction,
        icon_fraction=icon_fraction,
    )


def detect_loot_page(frame: np.ndarray) -> LootPageDetection:
    """Detect the obtained-items page and its first two rainbow gear cards."""
    if frame is None or frame.size == 0 or frame.ndim != 3:
        return LootPageDetection(False, 0.0, 0.0, 0.0, False, False, 0.0, 0.0)

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    def teal_fraction(x0: float, x1: float, y0: float, y1: float) -> float:
        roi = _fractional_crop(hsv, x0, x1, y0, y1)
        if roi.size == 0:
            return 0.0
        hue, saturation, value = cv2.split(roi)
        mask = (
            (hue >= 52)
            & (hue <= 102)
            & (saturation >= 55)
            & (value >= 30)
        )
        return float(np.mean(mask))

    header = _fractional_crop(frame, 0.02, 0.24, 0.02, 0.15)
    if header.size:
        header_min = np.min(header, axis=2)
        header_max = np.max(header, axis=2)
        header_fraction = float(
            np.mean((header_min >= 160) & ((header_max - header_min) <= 60))
        )
    else:
        header_fraction = 0.0

    panel_fraction = teal_fraction(0.02, 0.58, 0.08, 0.90)
    detail_fraction = teal_fraction(0.60, 0.98, 0.08, 0.85)
    page_detected = (
        panel_fraction >= 0.42
        and detail_fraction >= 0.48
        and header_fraction >= 0.14
    )

    def rainbow_score(x0: float, x1: float) -> tuple[bool, float]:
        roi = _fractional_crop(hsv, x0, x1, 0.23, 0.41)
        if roi.size == 0:
            return False, 0.0
        hue, saturation, value = cv2.split(roi)
        bright = value >= 120
        colorful = bright & (saturation >= 35)
        cyan = (
            colorful
            & (hue >= 74)
            & (hue <= 102)
        )
        magenta = (
            colorful
            & (hue >= 130)
            & (hue <= 172)
        )
        cyan_fraction = float(np.mean(cyan))
        magenta_fraction = float(np.mean(magenta))

        # A genuine rainbow card has saturated colour spread across almost its
        # entire background.  Purple-rarity cards can contain enough cyan and
        # magenta locally (from the weapon and pale purple sparkle) to fool a
        # simple two-colour count, but they do not fill the card spatially.
        colorful_fraction = float(np.mean(colorful))
        tile_hits = 0
        roi_height, roi_width = colorful.shape
        for tile_y in range(3):
            y_start = tile_y * roi_height // 3
            y_end = (tile_y + 1) * roi_height // 3
            for tile_x in range(3):
                x_start = tile_x * roi_width // 3
                x_end = (tile_x + 1) * roi_width // 3
                tile = colorful[y_start:y_end, x_start:x_end]
                if tile.size and float(np.mean(tile)) >= 0.40:
                    tile_hits += 1

        score = min(
            cyan_fraction / 0.25,
            magenta_fraction / 0.05,
            colorful_fraction / 0.45,
            tile_hits / 6.0,
        )
        return score >= 1.0, score

    slot1_rainbow, slot1_score = rainbow_score(0.06, 0.17)
    slot2_rainbow, slot2_score = rainbow_score(0.18, 0.29)
    if not page_detected:
        slot1_rainbow = False
        slot2_rainbow = False
    return LootPageDetection(
        detected=page_detected,
        panel_fraction=panel_fraction,
        detail_fraction=detail_fraction,
        header_fraction=header_fraction,
        slot1_rainbow=slot1_rainbow,
        slot2_rainbow=slot2_rainbow,
        slot1_score=slot1_score,
        slot2_score=slot2_score,
    )


def _save_png(path: Path, image: np.ndarray) -> bool:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        success, encoded = cv2.imencode(".png", image)
        if not success:
            return False
        encoded.tofile(str(path))
        return True
    except Exception as exc:
        _timestamped_log(f"彩装截图保存失败：{exc}")
        return False


def _windows_ocr(
    image_path: Path,
    language_tag: str = "zh-Hans",
) -> str:
    if sys.platform != "win32" or shutil.which("powershell.exe") is None:
        return ""
    script = r'''
$ErrorActionPreference = "Stop"
[Console]::OutputEncoding = New-Object System.Text.UTF8Encoding($false)
Add-Type -AssemblyName System.Runtime.WindowsRuntime
$null = [Windows.Storage.StorageFile, Windows.Storage, ContentType=WindowsRuntime]
$null = [Windows.Storage.Streams.IRandomAccessStream, Windows.Storage.Streams, ContentType=WindowsRuntime]
$null = [Windows.Graphics.Imaging.BitmapDecoder, Windows.Graphics.Imaging, ContentType=WindowsRuntime]
$null = [Windows.Graphics.Imaging.SoftwareBitmap, Windows.Graphics.Imaging, ContentType=WindowsRuntime]
$null = [Windows.Media.Ocr.OcrEngine, Windows.Foundation, ContentType=WindowsRuntime]
$null = [Windows.Media.Ocr.OcrResult, Windows.Foundation, ContentType=WindowsRuntime]
$null = [Windows.Globalization.Language, Windows.Globalization, ContentType=WindowsRuntime]
function Await-Result($operation, $resultType) {
    $method = [System.WindowsRuntimeSystemExtensions].GetMethods() |
        Where-Object { $_.Name -eq "AsTask" -and $_.IsGenericMethod -and $_.GetParameters().Count -eq 1 } |
        Select-Object -First 1
    $task = $method.MakeGenericMethod($resultType).Invoke($null, @($operation))
    $task.Wait()
    return $task.Result
}
$file = Await-Result ([Windows.Storage.StorageFile]::GetFileFromPathAsync($env:MACRO5_OCR_IMAGE)) ([Windows.Storage.StorageFile])
$stream = Await-Result ($file.OpenAsync([Windows.Storage.FileAccessMode]::Read)) ([Windows.Storage.Streams.IRandomAccessStream])
$decoder = Await-Result ([Windows.Graphics.Imaging.BitmapDecoder]::CreateAsync($stream)) ([Windows.Graphics.Imaging.BitmapDecoder])
$bitmap = Await-Result ($decoder.GetSoftwareBitmapAsync()) ([Windows.Graphics.Imaging.SoftwareBitmap])
$language = New-Object Windows.Globalization.Language($env:MACRO5_OCR_LANGUAGE)
$engine = [Windows.Media.Ocr.OcrEngine]::TryCreateFromLanguage($language)
if ($null -eq $engine) { $engine = [Windows.Media.Ocr.OcrEngine]::TryCreateFromUserProfileLanguages() }
if ($null -eq $engine) { exit 2 }
$result = Await-Result ($engine.RecognizeAsync($bitmap)) ([Windows.Media.Ocr.OcrResult])
Write-Output $result.Text
'''
    environment = dict(os.environ)
    environment["MACRO5_OCR_IMAGE"] = str(image_path.resolve())
    environment["MACRO5_OCR_LANGUAGE"] = language_tag or "zh-Hans"
    try:
        completed = subprocess.run(
            ["powershell.exe", "-NoProfile", "-STA", "-Command", script],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="ignore",
            timeout=15,
            env=environment,
            check=False,
        )
        return completed.stdout.strip() if completed.returncode == 0 else ""
    except Exception:
        return ""


def _ocr_loot_name(image_path: Path) -> str:
    executable = shutil.which("tesseract")
    if executable is not None:
        try:
            completed = subprocess.run(
                [
                    executable,
                    str(image_path),
                    "stdout",
                    "-l",
                    "chi_sim+eng",
                    "--psm",
                    "7",
                ],
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="ignore",
                timeout=15,
                check=False,
            )
            if completed.returncode == 0 and completed.stdout.strip():
                return " ".join(completed.stdout.split())
        except Exception:
            pass
    text = _windows_ocr(image_path)
    return " ".join(text.split())


def _is_target_loot(ocr_text: str, supreme: bool = False) -> bool:
    normalized = unicodedata.normalize("NFKC", ocr_text)
    normalized = re.sub(r"\s+", "", normalized).casefold()
    return (
        supreme
        and TARGET_LOOT_NAME.casefold() in normalized
    )


def _detect_supreme_prefix(title_crop: np.ndarray) -> tuple[bool, float, int]:
    """Detect the fixed orange 绝品 prefix at the left of the item title."""
    if title_crop is None or title_crop.size == 0 or title_crop.ndim != 3:
        return False, 0.0, 0
    hsv = cv2.cvtColor(title_crop, cv2.COLOR_BGR2HSV)
    prefix_roi = _fractional_crop(hsv, 0.0, 0.18, 0.30, 0.82)
    if prefix_roi.size == 0:
        return False, 0.0, 0
    hue, saturation, value = cv2.split(prefix_roi)
    orange = (
        (hue >= 4)
        & (hue <= 28)
        & (saturation >= 100)
        & (value >= 100)
    )
    orange_pixels = int(np.count_nonzero(orange))
    orange_fraction = float(np.mean(orange))
    detected = (
        orange_pixels >= SUPREME_PREFIX_MIN_ORANGE_PIXELS
        and orange_fraction >= SUPREME_PREFIX_MIN_ORANGE_FRACTION
    )
    score = min(
        orange_pixels / SUPREME_PREFIX_MIN_ORANGE_PIXELS,
        orange_fraction / SUPREME_PREFIX_MIN_ORANGE_FRACTION,
    )
    return detected, float(score), orange_pixels


def _wait_for_stable_loot_page(
    context: _MacroContext,
    capture: FFmpegCapture,
) -> tuple[LootPageDetection, np.ndarray | None]:
    deadline = context.active_monotonic() + LOOT_PAGE_SCAN_SECONDS
    last_sequence = -1
    samples: list[tuple[LootPageDetection, np.ndarray]] = []
    while context.active_monotonic() < deadline:
        snapshot = capture.snapshot()
        if snapshot.frame is not None and snapshot.sequence != last_sequence:
            last_sequence = snapshot.sequence
            detection = detect_loot_page(snapshot.frame)
            if detection.detected:
                samples.append((detection, snapshot.frame.copy()))
        if not context.wait_ms(80):
            break
    if len(samples) < 3:
        return (
            LootPageDetection(
                False, 0.0, 0.0, 0.0, False, False, 0.0, 0.0
            ),
            None,
        )

    required_hits = max(2, int(len(samples) * 0.60 + 0.999))
    slot1 = sum(item[0].slot1_rainbow for item in samples) >= required_hits
    slot2 = sum(item[0].slot2_rainbow for item in samples) >= required_hits
    best_detection, best_frame = max(
        samples,
        key=lambda item: item[0].slot1_score + item[0].slot2_score,
    )
    stable = LootPageDetection(
        detected=True,
        panel_fraction=best_detection.panel_fraction,
        detail_fraction=best_detection.detail_fraction,
        header_fraction=best_detection.header_fraction,
        slot1_rainbow=slot1,
        slot2_rainbow=slot2,
        slot1_score=max(item[0].slot1_score for item in samples),
        slot2_score=max(item[0].slot2_score for item in samples),
    )
    return stable, best_frame


def _capture_loot_record_images(
    frame: np.ndarray,
    slot: int,
    score: float,
) -> tuple[dict[str, object], Path]:
    stamp = (
        time.strftime("%Y%m%d-%H%M%S")
        + f"-{time.time_ns() % 1_000_000_000:09d}"
    )
    record_dir = Path(__file__).resolve().with_name(LOOT_SCREENSHOT_DIRNAME)
    full_path = record_dir / f"{stamp}-slot{slot}-full.png"
    title_path = record_dir / f"{stamp}-slot{slot}-name.png"
    title_crop = _fractional_crop(frame, 0.62, 0.98, 0.065, 0.19)
    supreme, supreme_score, supreme_orange_pixels = _detect_supreme_prefix(
        title_crop
    )
    full_saved = _save_png(full_path, frame)
    title_saved = title_crop.size > 0 and _save_png(title_path, title_crop)
    record = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "slot": slot,
        "name": "名称未识别",
        "ocr_text": "",
        "target": False,
        "supreme": supreme,
        "supreme_score": round(float(supreme_score), 3),
        "supreme_orange_pixels": supreme_orange_pixels,
        "rainbow_score": round(float(score), 3),
        "screenshot": (
            f"{LOOT_SCREENSHOT_DIRNAME}/{full_path.name}"
            if full_saved
            else "未保存"
        ),
        "name_crop": (
            f"{LOOT_SCREENSHOT_DIRNAME}/{title_path.name}"
            if title_saved
            else "未保存"
        ),
    }
    return record, title_path


def scan_and_record_loot_page(
    context: _MacroContext,
    capture: FFmpegCapture,
    history: Macro5History,
    *,
    close_gap_ms: int = 500,
) -> bool | None:
    """Inspect rainbow cards and use the current settlement A to close the page.

    Return True when the page was scanned and closed, None when the page could
    not be confirmed, and False only when the macro itself must stop.
    """
    detection, first_frame = _wait_for_stable_loot_page(context, capture)
    if not detection.detected or first_frame is None:
        _timestamped_log(
            "结算：画面一度像获得物品页面，但未能稳定确认；"
            "本次继续原来的A结算流程。"
        )
        return None

    _timestamped_log(
        "获得物品页面已确认："
        f"第一格彩装={'是' if detection.slot1_rainbow else '否'}"
        f"({detection.slot1_score:.2f})，"
        f"第二格彩装={'是' if detection.slot2_rainbow else '否'}"
        f"({detection.slot2_score:.2f})。"
    )
    pending: list[tuple[dict[str, object], Path]] = []
    if detection.slot1_rainbow:
        pending.append(
            _capture_loot_record_images(
                first_frame,
                slot=1,
                score=detection.slot1_score,
            )
        )
        if detection.slot2_rainbow:
            _timestamped_log(
                "第二格也确认是彩装：左摇杆向右轻推一次后读取名称。"
            )
            if not context.move_stick(
                BIT_LSTICK_RIGHT,
                duration_ms=LOOT_SECOND_SLOT_MOVE_MS,
            ):
                return False
            context.center_stick()
            if not context.wait_ms(LOOT_SECOND_SLOT_SETTLE_MS):
                return False
            second_snapshot = capture.snapshot()
            if second_snapshot.frame is not None:
                pending.append(
                    _capture_loot_record_images(
                        second_snapshot.frame,
                        slot=2,
                        score=detection.slot2_score,
                    )
                )
            else:
                _timestamped_log("第二格彩装画面读取失败，本次只记录第一格。")
    elif detection.slot2_rainbow:
        _timestamped_log(
            "检测结果异常：第二格像彩装但第一格不是；按稀有度排序规则不移动光标。"
        )

    # A closes the obtained-items page from either selected slot.  The user
    # explicitly does not want the cursor moved back to slot 1.
    if not context.tap(BIT_A, hold_ms=50, gap_ms=close_gap_ms):
        return False

    for record, title_path in pending:
        text = _ocr_loot_name(title_path) if title_path.exists() else ""
        if text:
            record["name"] = text
            record["ocr_text"] = text
            record["target"] = _is_target_loot(
                text,
                supreme=record.get("supreme") is True,
            )
        history.record_drop(record)
        supreme_mark = "【绝品】" if record.get("supreme") is True else ""
        target_mark = "【目标出货】" if record["target"] else ""
        _timestamped_log(
            f"已记录第{record['slot']}格彩装：{record['name']}"
            f"{supreme_mark}{target_mark}；"
            f"绝品橙色像素{record['supreme_orange_pixels']}；"
            f"截图 {record['screenshot']}"
        )
    return True


def detect_clear_screen(frame: np.ndarray) -> ClearScreenDetection:
    """Detect the fixed yellow 'Clear!!' title on the settlement screen."""
    if frame is None or frame.size == 0 or frame.ndim != 3:
        return ClearScreenDetection(False, 0.0, 0.0)

    height, width = frame.shape[:2]
    title_roi = frame[
        int(height * 0.1257):int(height * 0.2692),
        int(width * 0.4906):int(width * 0.6540),
    ]
    if title_roi.size == 0:
        return ClearScreenDetection(False, 0.0, 0.0)
    title_roi = cv2.resize(
        title_roi,
        (CLEAR_TEMPLATE_SHAPE[1], CLEAR_TEMPLATE_SHAPE[0]),
        interpolation=cv2.INTER_AREA,
    )

    blue = title_roi[:, :, 0].astype(np.int16)
    green = title_roi[:, :, 1].astype(np.int16)
    red = title_roi[:, :, 2].astype(np.int16)
    yellow_mask = (
        (red >= 150)
        & (green >= 170)
        & (blue <= 110)
        & ((red + green - 2 * blue) >= 180)
    )
    yellow_fraction = float(np.mean(yellow_mask))
    yellow_dilated = cv2.dilate(
        yellow_mask.astype(np.uint8),
        CLEAR_KERNEL_5,
        iterations=1,
    ).astype(bool)
    template_pixel_count = int(np.count_nonzero(CLEAR_TEMPLATE_MASK))
    template_recall = float(
        np.count_nonzero(CLEAR_TEMPLATE_MASK & yellow_dilated)
        / max(1, template_pixel_count)
    )
    detected = yellow_fraction >= 0.04 and template_recall >= 0.65
    return ClearScreenDetection(
        detected=detected,
        yellow_fraction=yellow_fraction,
        template_recall=template_recall,
    )


def detect_result_screen(frame: np.ndarray) -> ResultScreenDetection:
    """Detect the fixed white 'Result' title on a failed settlement page."""
    if frame is None or frame.size == 0 or frame.ndim != 3:
        return ResultScreenDetection(False, 0.0, 0.0)

    height, width = frame.shape[:2]
    title_roi = frame[
        int(height * 0.140):int(height * 0.205),
        int(width * 0.505):int(width * 0.625),
    ]
    if title_roi.size == 0:
        return ResultScreenDetection(False, 0.0, 0.0)
    title_roi = cv2.resize(
        title_roi,
        (RESULT_TEMPLATE_SHAPE[1], RESULT_TEMPLATE_SHAPE[0]),
        interpolation=cv2.INTER_AREA,
    )

    channel_min = np.min(title_roi, axis=2)
    channel_max = np.max(title_roi, axis=2)
    white_mask = (
        (channel_min >= 150)
        & ((channel_max - channel_min) <= 65)
    )
    white_fraction = float(np.mean(white_mask))
    white_dilated = cv2.dilate(
        white_mask.astype(np.uint8),
        MATCH_KERNEL_3,
        iterations=1,
    ).astype(bool)
    recall = float(
        np.count_nonzero(RESULT_TEMPLATE_MASK & white_dilated)
        / RESULT_TEMPLATE_PIXEL_COUNT
    )
    precision = float(
        np.count_nonzero(white_mask & RESULT_TEMPLATE_DILATED)
        / max(1, int(np.count_nonzero(white_mask)))
    )
    template_match = min(recall, precision)
    detected = white_fraction >= 0.18 and template_match >= 0.72
    return ResultScreenDetection(
        detected=detected,
        white_fraction=white_fraction,
        template_match=template_match,
    )


def settlement_screen_name(frame: np.ndarray) -> str | None:
    """Return the visible usable settlement title, if either one is ready."""
    if detect_clear_screen(frame).detected:
        return "Clear!!"
    if detect_result_screen(frame).detected:
        return "Result"
    return None


def detect_base_hud(frame: np.ndarray) -> BaseHudDetection:
    """Detect the stacked bottom-right 'X menu' and '+ equipment' hints."""
    if frame is None or frame.size == 0 or frame.ndim != 3:
        return BaseHudDetection(False, 0.0, 0.0, 0.0, 0.0, None)

    height, width = frame.shape[:2]

    def neutral_white_fraction(
        x0: float,
        x1: float,
        y0: float,
        y1: float,
    ) -> float:
        roi = frame[
            int(height * y0):int(height * y1),
            int(width * x0):int(width * x1),
        ]
        if roi.size == 0:
            return 0.0
        channel_min = np.min(roi, axis=2)
        channel_max = np.max(roi, axis=2)
        neutral_white = (
            (channel_min >= 170)
            & ((channel_max - channel_min) <= 60)
        )
        return float(np.mean(neutral_white))

    menu_fraction = neutral_white_fraction(
        0.925, 0.995, 0.885, 0.940
    )
    equipment_fraction = neutral_white_fraction(
        0.925, 0.995, 0.945, 0.995
    )
    template_roi = frame[
        int(height * 0.86):height,
        int(width * 0.90):width,
    ]
    template_roi = cv2.resize(
        template_roi,
        (BASE_HUD_TEMPLATE_SHAPE[1], BASE_HUD_TEMPLATE_SHAPE[0]),
        interpolation=cv2.INTER_AREA,
    )
    template_min = np.min(template_roi, axis=2)
    template_max = np.max(template_roi, axis=2)
    candidate_mask = (
        (template_min >= 170)
        & ((template_max - template_min) <= 60)
        & BASE_HUD_VALID_MASK
        & BASE_HUD_ROW_MASK
    ).astype(np.uint8)
    candidate_dilated = cv2.dilate(
        candidate_mask,
        MATCH_KERNEL_3,
        iterations=1,
    ).astype(bool)
    def row_match(
        template: np.ndarray,
        template_dilated: np.ndarray,
        y0: int,
        y1: int,
    ) -> float:
        template_band = template[y0:y1]
        candidate_band = candidate_mask[y0:y1].astype(bool)
        recall = float(
            np.count_nonzero(template_band & candidate_dilated[y0:y1])
            / max(1, int(np.count_nonzero(template_band)))
        )
        precision = float(
            np.count_nonzero(candidate_band & template_dilated[y0:y1])
            / max(1, int(np.count_nonzero(candidate_band)))
        )
        return min(recall, precision)

    # The two prompts occupy separate fixed rows in the captured 960x540
    # frame.  Score them independently so the in-stage single "- Pause" row
    # can never borrow pixels from nearby HUD text to pass as the base menu.
    template_scores: list[tuple[float, float]] = []
    for template, template_dilated in zip(
        BASE_HUD_TEMPLATE_MASKS,
        BASE_HUD_TEMPLATE_DILATED_MASKS,
    ):
        template_scores.append(
            (
                row_match(template, template_dilated, 22, 42),
                row_match(template, template_dilated, 48, 68),
            )
        )

    # Keep each X/+ pair tied to one visual reference.  Taking the best X row
    # and best + row independently could combine two unrelated partial matches.
    best_template_index, (menu_match, equipment_match) = max(
        enumerate(template_scores),
        key=lambda item: min(item[1]),
    )
    detected = (
        menu_fraction >= 0.11
        and equipment_fraction >= 0.14
        and menu_match >= 0.75
        and equipment_match >= 0.75
    )
    return BaseHudDetection(
        detected=detected,
        menu_fraction=menu_fraction,
        equipment_fraction=equipment_fraction,
        menu_match=menu_match,
        equipment_match=equipment_match,
        # The first reference was captured over the yellow-ink base view;
        # the alternate reference is the clean/white base view.
        variant=("yellow", "white")[best_template_index] if detected else None,
    )


def detect_shiver_page(frame: np.ndarray) -> ShiverPageDetection:
    """Detect Shiver's accessory-development page from its fixed color blocks."""
    if frame is None or frame.size == 0 or frame.ndim != 3:
        return ShiverPageDetection(False, 0.0, 0.0, 0.0)

    height, width = frame.shape[:2]
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    def color_fraction(
        x0: float,
        x1: float,
        y0: float,
        y1: float,
        lower: tuple[int, int, int],
        upper: tuple[int, int, int],
    ) -> float:
        roi = hsv[
            int(height * y0):int(height * y1),
            int(width * x0):int(width * x1),
        ]
        if roi.size == 0:
            return 0.0
        return float(np.mean(cv2.inRange(roi, lower, upper) > 0))

    # The page consistently has a large purple equipment panel in the middle,
    # a teal "accessory development" plaque at upper-left, and a teal preview
    # panel at right.  Requiring all three avoids confusing purple battle ink
    # with this menu.
    purple_fraction = color_fraction(
        0.27, 0.64, 0.27, 0.98,
        (120, 50, 40),
        (165, 255, 255),
    )
    teal_left_fraction = color_fraction(
        0.00, 0.13, 0.00, 0.15,
        (45, 60, 70),
        (105, 255, 255),
    )
    teal_right_fraction = color_fraction(
        0.65, 0.99, 0.30, 0.66,
        (45, 60, 70),
        (105, 255, 255),
    )
    detected = (
        purple_fraction >= 0.35
        and teal_left_fraction >= 0.20
        and teal_right_fraction >= 0.15
    )
    return ShiverPageDetection(
        detected=detected,
        purple_fraction=purple_fraction,
        teal_left_fraction=teal_left_fraction,
        teal_right_fraction=teal_right_fraction,
    )


def detect_in_game_life(frame: np.ndarray) -> WhiteIconDetection:
    """Count a contiguous left-to-right prefix of 1..6 life icons."""
    if frame is None or frame.size == 0 or frame.ndim != 3:
        return WhiteIconDetection(False)

    height, width = frame.shape[:2]
    slot_scores: list[float] = []
    slot_white_fractions: list[float] = []
    slot_present: list[bool] = []
    for reference_center_x in (25, 50, 75, 100, 125, 150):
        x0 = int(round((reference_center_x - 12) * width / 960.0))
        x1 = int(round((reference_center_x + 13) * width / 960.0))
        y0 = int(round(10 * height / 540.0))
        y1 = int(round(42 * height / 540.0))
        slot_roi = frame[max(0, y0):min(height, y1), max(0, x0):min(width, x1)]
        if slot_roi.size == 0:
            slot_scores.append(0.0)
            slot_white_fractions.append(0.0)
            slot_present.append(False)
            continue
        slot_roi = cv2.resize(
            slot_roi,
            (LIFE_ICON_TEMPLATE_SHAPE[1], LIFE_ICON_TEMPLATE_SHAPE[0]),
            interpolation=cv2.INTER_AREA,
        ).astype(np.int16)
        blue, green, red = cv2.split(slot_roi)
        channel_spread = np.max(slot_roi, axis=2) - np.min(slot_roi, axis=2)
        candidate_mask = (
            (red >= 190)
            & (green >= 190)
            & (blue >= 160)
            & (channel_spread <= 75)
        )
        white_fraction = float(np.mean(candidate_mask))
        candidate_dilated = cv2.dilate(
            candidate_mask.astype(np.uint8),
            MATCH_KERNEL_3,
            iterations=1,
        ).astype(bool)
        candidate_pixels = max(1, int(np.count_nonzero(candidate_mask)))
        recall = float(
            np.count_nonzero(LIFE_ICON_TEMPLATE_MASK & candidate_dilated)
            / LIFE_ICON_TEMPLATE_PIXEL_COUNT
        )
        precision = float(
            np.count_nonzero(candidate_mask & LIFE_ICON_TEMPLATE_DILATED)
            / candidate_pixels
        )
        score = min(recall, precision)
        present = 0.16 <= white_fraction <= 0.42 and score >= 0.78
        slot_scores.append(score)
        slot_white_fractions.append(white_fraction)
        slot_present.append(present)

    life_count = 0
    while life_count < 6 and slot_present[life_count]:
        life_count += 1
    has_hole = any(slot_present[life_count:])
    if life_count == 0 or has_hole:
        life_count = 0
    return WhiteIconDetection(
        detected=life_count > 0,
        life_count=life_count,
        slot_scores=tuple(slot_scores),
        slot_white_fractions=tuple(slot_white_fractions),
        slot_present=tuple(slot_present),
    )


def detect_battle_bar(frame: np.ndarray) -> BarDetection:
    """Detect the fixed upper-center battle bar by its aligned tick marks.

    The FFmpeg input is always scaled to 960x540, but all coordinates are kept
    proportional so minor capture-size changes remain harmless.  The detector
    only examines the narrow upper-center region occupied by the black bar and
    requires both a long row of thin neutral-colored ticks and a dark backing.
    """
    if frame is None or frame.size == 0 or frame.ndim != 3:
        return BarDetection(False, 0, 0, 0.0)

    height, width = frame.shape[:2]
    roi_x0 = max(0, int(width * 0.36))
    roi_x1 = max(roi_x0 + 1, int(width * 0.70))
    roi_y0 = max(0, int(height * 0.03))
    roi_y1 = max(roi_y0 + 1, int(height * 0.10))
    roi = frame[roi_y0:roi_y1, roi_x0:roi_x1]

    channel_min = np.min(roi, axis=2)
    channel_max = np.max(roi, axis=2)
    neutral_bright_mask = (
        (channel_min >= 90) & ((channel_max - channel_min) <= 75)
    ).astype(np.uint8) * 255
    count, _, stats, centroids = cv2.connectedComponentsWithStats(
        neutral_bright_mask,
        connectivity=8,
    )
    candidates: list[tuple[float, float, int, int]] = []
    for label in range(1, count):
        _, _, component_width, component_height, area = (
            int(value) for value in stats[label]
        )
        if not (1 <= component_width <= max(4, int(width * 0.005))):
            continue
        if not (max(4, int(height * 0.007)) <= component_height <= int(height * 0.027)):
            continue
        if component_height < component_width * 1.5 or area < 4:
            continue
        center_x, center_y = (float(value) for value in centroids[label])
        candidates.append((center_x, center_y, component_width, component_height))

    best_group: list[tuple[float, float, int, int]] = []
    y_tolerance = max(3.0, height * 0.008)
    for anchor in candidates:
        row = [item for item in candidates if abs(item[1] - anchor[1]) <= y_tolerance]
        row.sort(key=lambda item: item[0])
        if len(row) > len(best_group):
            best_group = row

    aligned_tick_count = len(best_group)
    tick_span = 0.0
    regular_gap_count = 0
    dark_fraction = 0.0
    if best_group:
        tick_span = best_group[-1][0] - best_group[0][0]
        gaps = [
            best_group[index + 1][0] - best_group[index][0]
            for index in range(len(best_group) - 1)
        ]
        regular_gap_count = sum(
            width * 0.004 <= gap <= width * 0.023
            for gap in gaps
        )
        center_y = int(round(sum(item[1] for item in best_group) / len(best_group)))
        patch_x0 = max(0, int(best_group[0][0]) - 5)
        patch_x1 = min(roi.shape[1], int(best_group[-1][0]) + 6)
        patch_y0 = max(0, center_y - 11)
        patch_y1 = min(roi.shape[0], center_y + 12)
        backing = roi[patch_y0:patch_y1, patch_x0:patch_x1]
        if backing.size:
            dark_fraction = float(np.mean(np.max(backing, axis=2) < 90))

    detected = (
        aligned_tick_count >= 10
        and tick_span >= width * 0.18
        and regular_gap_count >= 7
        and dark_fraction >= 0.40
    )
    return BarDetection(
        detected=detected,
        aligned_tick_count=aligned_tick_count,
        candidate_count=len(candidates),
        dark_fraction=dark_fraction,
    )


class FFmpegCapture:
    """Read the configured DirectShow source and retain only its latest frame."""

    def __init__(
        self,
        settings: CaptureSettings,
        stop_event: threading.Event,
        heartbeat: _HeartbeatFile,
    ) -> None:
        self.settings = settings
        self.device_name = settings.device_name
        self.device_label = settings.device_label
        self.stop_event = stop_event
        self.heartbeat = heartbeat
        self._lock = threading.Lock()
        self._snapshot = FrameSnapshot(0, None)
        self._source: FFmpegCaptureSource | None = None
        self._thread: threading.Thread | None = None

    def snapshot(self, after_sequence: int | None = None) -> FrameSnapshot:
        with self._lock:
            if (
                after_sequence is not None
                and self._snapshot.sequence == after_sequence
            ):
                return FrameSnapshot(
                    self._snapshot.sequence,
                    None,
                    self._snapshot.error,
                )
            frame = None if self._snapshot.frame is None else self._snapshot.frame.copy()
            return FrameSnapshot(self._snapshot.sequence, frame, self._snapshot.error)

    def _store(self, frame: np.ndarray | None, error: str = "") -> None:
        with self._lock:
            self._snapshot = FrameSnapshot(
                self._snapshot.sequence + 1,
                frame,
                error,
            )

    @property
    def active_capture_spec(self) -> dict[str, object]:
        source = self._source
        return source.active_capture_spec if source is not None else {}

    @property
    def startup_timeout_seconds(self) -> float:
        attempts = 1 + len(self.settings.fallback_specs)
        return attempts * self.settings.read_timeout_seconds + 5.0

    @staticmethod
    def _preview_frame(frame: np.ndarray) -> np.ndarray:
        if frame.shape[:2] == (CAPTURE_HEIGHT, CAPTURE_WIDTH):
            return frame.copy()
        return cv2.resize(
            frame,
            (CAPTURE_WIDTH, CAPTURE_HEIGHT),
            interpolation=cv2.INTER_AREA,
        )

    def start(self, worker_errors: List[BaseException]) -> None:
        def reader() -> None:
            try:
                settings = self.settings
                source = FFmpegCaptureSource(
                    device_name=settings.device_name,
                    width=settings.width,
                    height=settings.height,
                    fps=settings.fps,
                    pixel_format=settings.pixel_format,
                    strict_usb_only=False,
                )
                self._source = source
                fallback_specs = [
                    {
                        "width": width,
                        "height": height,
                        "pixel_format": pixel_format,
                    }
                    for width, height, pixel_format in settings.fallback_specs
                ]
                frame = source.read_with_fallbacks(
                    timeout_seconds=settings.read_timeout_seconds,
                    fallback_specs=fallback_specs,
                )
                if frame is None:
                    self._store(
                        None,
                        str(source.last_error or "FFmpeg没有输出首帧"),
                    )
                    self.stop_event.set()
                    return
                self._store(self._preview_frame(frame))
                self.heartbeat.beat(force=True)
                latest_timestamp = source.latest_frame_ts
                while not self.stop_event.is_set():
                    frame = source.read_next(
                        after_ts=latest_timestamp,
                        timeout_seconds=max(2.0, settings.read_timeout_seconds),
                    )
                    if frame is None:
                        if not self.stop_event.is_set():
                            self._store(
                                None,
                                str(source.last_error or "FFmpeg视频流停止刷新"),
                            )
                            self.stop_event.set()
                        return
                    latest_timestamp = source.latest_frame_ts
                    self._store(self._preview_frame(frame))
                    self.heartbeat.beat()
            except BaseException as exc:
                self._store(None, str(exc))
                worker_errors.append(exc)
                self.stop_event.set()

        self._thread = threading.Thread(
            target=reader,
            name="macro5-ffmpeg-capture",
            daemon=True,
        )
        self._thread.start()

    def wait_until_ready(self, timeout_seconds: float | None = None) -> bool:
        if timeout_seconds is None:
            timeout_seconds = self.startup_timeout_seconds
        deadline = time.monotonic() + timeout_seconds
        while time.monotonic() < deadline and not self.stop_event.is_set():
            if self.snapshot().frame is not None:
                return True
            time.sleep(0.05)
        return False

    def close(self) -> None:
        if self._source is not None:
            with contextlib.suppress(Exception):
                self._source.stop()
        if self._thread is not None:
            self._thread.join(timeout=2.0)


def sample_round_visual_state(
    context: _MacroContext,
    capture: FFmpegCapture,
    duration_seconds: float,
) -> str | None:
    """Return 'settlement', 'battle', 'transition', or None from fresh frames."""
    deadline = context.active_monotonic() + max(0.1, duration_seconds)
    last_sequence = -1
    sampled = 0
    battle_hits = 0
    while context.active_monotonic() < deadline:
        snapshot = capture.snapshot()
        if snapshot.frame is not None and snapshot.sequence != last_sequence:
            last_sequence = snapshot.sequence
            sampled += 1
            if settlement_screen_name(snapshot.frame) is not None:
                return "settlement"
            if detect_battle_bar(snapshot.frame).detected:
                battle_hits += 1
        if not context.wait_ms(80):
            return None
    if sampled < 2:
        return None
    if battle_hits >= 2:
        return "battle"
    if battle_hits == 0:
        return "transition"
    return None


def wait_for_battle_start_after_entry(
    context: _MacroContext,
    capture: FFmpegCapture,
) -> str | None:
    """Wait until the entered stage shows its battle bar, with a 6.5s fallback."""
    started = context.active_monotonic()
    deadline = started + ENTRY_BATTLE_READY_TIMEOUT_MS / 1000.0
    last_sequence = -1
    while context.active_monotonic() < deadline:
        snapshot = capture.snapshot()
        if snapshot.frame is not None and snapshot.sequence != last_sequence:
            last_sequence = snapshot.sequence
            if detect_shiver_page(snapshot.frame).detected:
                return "shiver"
            if detect_battle_bar(snapshot.frame).detected:
                elapsed = max(0.0, context.active_monotonic() - started)
                _timestamped_log(
                    "macro5 watchdog：检测到关卡鲑鱼卵条，"
                    f"等待{elapsed:.2f}秒后立即开始ZR。"
                )
                return "battle"
        if not context.wait_ms(ENTRY_BATTLE_READY_POLL_MS):
            return None
    _timestamped_log(
        "macro5 watchdog：6.5秒内未检测到关卡鲑鱼卵条，"
        "按原等待上限兜底开始ZR。"
    )
    return "timeout"


def _settlement_outcome_from_frame(
    context: _MacroContext,
    capture: FFmpegCapture,
    frame: np.ndarray,
) -> str | None:
    """Return clear/result_pressed and verify Result actually accepts A."""
    if detect_result_screen(frame).detected:
        _timestamped_log(
            "macro5 watchdog：检测到Result结算页面，等待页面稳定后按A并确认离开。"
        )
        if not context.wait_ms(RESULT_FIRST_A_READY_DELAY_MS):
            return None
        for attempt in range(1, RESULT_A_MAX_ATTEMPTS + 1):
            _timestamped_log(
                f"macro5 watchdog：Result页面发送第{attempt}次A。"
            )
            if not context.tap(
                BIT_A,
                hold_ms=RESULT_FIRST_A_HOLD_MS,
                gap_ms=RESULT_A_RETRY_GAP_MS,
            ):
                return None
            snapshot = capture.snapshot()
            if (
                snapshot.frame is not None
                and not detect_result_screen(snapshot.frame).detected
            ):
                _timestamped_log(
                    "macro5 watchdog：已确认离开Result页面。"
                )
                remaining_gap_ms = max(0, 1_500 - RESULT_A_RETRY_GAP_MS)
                if not context.wait_ms(remaining_gap_ms):
                    return None
                return "result_pressed"
        _timestamped_log(
            "macro5 watchdog：连续补按A后Result页面仍未离开，"
            "不中断执行，继续画面检测与恢复。"
        )
        return None
    if detect_clear_screen(frame).detected:
        _timestamped_log(
            "macro5 watchdog：检测到Clear!!结算页面，立即开始按A。"
        )
        return "clear"
    return None


def wait_for_settlement_result_after_transition(
    context: _MacroContext,
    capture: FFmpegCapture,
) -> str | None:
    """Wait through victory/loading animation until a result page is usable."""
    deadline = context.active_monotonic() + SETTLEMENT_RESULT_WAIT_SECONDS
    last_sequence = -1
    sampled = 0
    battle_streak = 0
    while context.active_monotonic() < deadline:
        snapshot = capture.snapshot()
        if snapshot.frame is not None and snapshot.sequence != last_sequence:
            last_sequence = snapshot.sequence
            sampled += 1
            outcome = _settlement_outcome_from_frame(
                context,
                capture,
                snapshot.frame,
            )
            if outcome is not None:
                return outcome
            if detect_battle_bar(snapshot.frame).detected:
                battle_streak += 1
                if battle_streak >= 3:
                    return "battle"
            else:
                battle_streak = 0
        if not context.wait_ms(80):
            return None
    if sampled >= 2:
        _timestamped_log(
            "macro5 watchdog：战斗条已消失但30秒内未出现Clear!!/Result，"
            "为避免盲按停止。"
        )
    return None


def wait_for_settlement_screen_or_battle_timeout(
    context: _MacroContext,
    capture: FFmpegCapture,
    deadline: float,
) -> str | None:
    """Return clear/result_pressed, or battle when the bar survives 60 seconds."""
    last_sequence = -1
    while context.active_monotonic() < deadline:
        snapshot = capture.snapshot()
        if snapshot.frame is not None and snapshot.sequence != last_sequence:
            last_sequence = snapshot.sequence
            outcome = _settlement_outcome_from_frame(
                context,
                capture,
                snapshot.frame,
            )
            if outcome is not None:
                return outcome
        if not context.wait_ms(80):
            return None

    state = sample_round_visual_state(
        context,
        capture,
        DEADLINE_STATE_SAMPLE_SECONDS,
    )
    if state == "settlement":
        snapshot = capture.snapshot()
        if snapshot.frame is not None:
            return _settlement_outcome_from_frame(
                context,
                capture,
                snapshot.frame,
            )
        return None
    if state == "battle":
        return "battle"
    if state == "transition":
        _timestamped_log(
            "macro5 watchdog：45秒时战斗条已消失；"
            "等待胜利动画结束并出现Clear!!/Result。"
        )
        return wait_for_settlement_result_after_transition(context, capture)
    return None


def wait_for_continuous_bar_escape_trigger(
    context: _MacroContext,
    capture: FFmpegCapture,
    bar_monitor: ContinuousBarMonitor,
) -> str | None:
    """Return bar_continuous after 55 uninterrupted seconds, or battle on a break."""
    initial_elapsed = bar_monitor.elapsed(context.active_monotonic())
    if initial_elapsed < 40.0:
        _timestamped_log(
            "macro5 watchdog：BAR在本轮中途曾中断，"
            f"当前连续时间仅{initial_elapsed:.1f}秒；"
            "不等待55秒脱离，继续局内恢复。"
        )
        return "battle"
    _timestamped_log(
        "macro5 watchdog：45秒截止点仍在局内；"
        "BAR若连续命中满55秒将直接主动脱离。"
    )
    last_sequence = -1
    while not context.stop_event.is_set():
        snapshot = capture.snapshot()
        if snapshot.frame is not None and snapshot.sequence != last_sequence:
            last_sequence = snapshot.sequence
            outcome = _settlement_outcome_from_frame(
                context,
                capture,
                snapshot.frame,
            )
            if outcome is not None:
                return outcome
            bar_detected = detect_battle_bar(snapshot.frame).detected
            now_active = context.active_monotonic()
            bar_monitor.update(bar_detected, now_active)
            if not bar_detected:
                _timestamped_log(
                    "macro5 watchdog：BAR连续计时已中断，"
                    "改用后退与局内命数恢复逻辑。"
                )
                return "battle"
            elapsed = bar_monitor.elapsed(now_active)
            if elapsed >= CONTINUOUS_BAR_ESCAPE_SECONDS:
                _timestamped_log(
                    f"macro5 watchdog：BAR已无中断连续检测"
                    f"{elapsed:.1f}秒，触发+号主动脱离。"
                )
                return "bar_continuous"
        if not context.wait_ms(80):
            return None
    return None


def run_failed_round_leave_sequence(
    context: _MacroContext,
    *,
    reason: str,
) -> str | None:
    """Leave a failed round and hand off to the existing six-A restart path."""
    _timestamped_log(
        f"macro5 watchdog：{reason}，执行主动脱离序列并等待5秒。"
    )
    for bit_index, hold_ms, gap_ms in (
        (BIT_PLUS, 50, 1000),
        (BIT_DPAD_DOWN, 50, 1000),
        (BIT_A, 50, 1000),
        (BIT_DPAD_RIGHT, 50, 1000),
        (BIT_A, 50, 5000),
    ):
        if not context.tap(bit_index, hold_ms=hold_ms, gap_ms=gap_ms):
            return None
    _timestamped_log(
        "macro5 watchdog：失败轮已主动脱离，"
        "不等待Clear!!/Result，直接进入多次A确认。"
    )
    return "failed_round_left"


def run_stuck_backstep_recovery(
    context: _MacroContext,
    capture: FFmpegCapture,
    bar_monitor: ContinuousBarMonitor,
) -> str | None:
    """Take one backward step and leave a positively identified failed round."""
    while not context.stop_event.is_set():
        _timestamped_log(
            f"macro5 watchdog：超时后仅向后补走 {STUCK_BACKSTEP_MS}ms，"
            "随后同帧检查局内命数与BAR。"
        )
        if not context.move_stick(BIT_LSTICK_DOWN, duration_ms=STUCK_BACKSTEP_MS):
            return None
        context.center_stick()

        snapshot = capture.snapshot()
        life_detection = (
            detect_in_game_life(snapshot.frame)
            if snapshot.frame is not None
            else WhiteIconDetection(False)
        )
        bar_detected = bool(
            snapshot.frame is not None
            and detect_battle_bar(snapshot.frame).detected
        )
        if life_detection.life_count > 0 and bar_detected:
            return run_failed_round_leave_sequence(
                context,
                reason=(
                    f"检测到局内命数{life_detection.life_count}"
                    "且BAR同时命中"
                ),
            )

        _timestamped_log(
            f"macro5 watchdog：45秒脱离条件未同时满足："
            f"IN GAME LIFE:{life_detection.life_count}，"
            f"BAR:{'DETECTED' if bar_detected else 'NOT DETECTED'}；"
            "不发送Home、不中断，保留连续BAR计时。"
        )
        bar_outcome = wait_for_continuous_bar_escape_trigger(
            context,
            capture,
            bar_monitor,
        )
        if bar_outcome == "bar_continuous":
            return run_failed_round_leave_sequence(
                context,
                reason=(
                    f"BAR无中断连续检测"
                    f"{CONTINUOUS_BAR_ESCAPE_SECONDS:g}秒"
                ),
            )
        if bar_outcome in {"clear", "result_pressed"}:
            return bar_outcome

        _timestamped_log(
            "macro5 watchdog：连续BAR已中断，"
            "继续当前局内攻击后再检查。"
        )
        if not run_l_r_a_repeat_window(context, LRA_FORWARD_START_MS):
            return None
        context.active_bits = (
            context.active_bits & ~context.stick_direction_mask
        ) | (1 << BIT_LSTICK_UP)
        context.send_active_bits()
        if not run_l_r_a_repeat_window(context, LRA_FORWARD_DURATION_MS):
            return None
        context.center_stick()

        retry_deadline = (
            context.active_monotonic() + ROUND_WATCHDOG_MS / 1000.0
        )
        retry_outcome = wait_for_settlement_screen_or_battle_timeout(
            context,
            capture,
            retry_deadline,
        )
        if retry_outcome in {"clear", "result_pressed"}:
            return retry_outcome
        # A missing/ambiguous frame or a still-active battle both retry this
        # recovery loop.  Only an explicit user stop leaves the loop.

    return None


def run_normal_settlement(
    context: _MacroContext,
    capture: FFmpegCapture,
    history: Macro5History,
    loop_state: WatchdogLoopState,
    *,
    first_a_already_pressed: bool = False,
    failed_round_left: bool = False,
) -> bool:
    """Run the original six settlement presses and inspect obtained items.

    The obtained-items page appears between these A presses, once per round.
    When it is found, the A used to close it consumes the current scheduled
    settlement press, so the normal button count does not increase.
    """
    if failed_round_left:
        _timestamped_log(
            "macro5 watchdog：失败轮脱离后执行6次A确认，"
            "完成后将从X+连续A重新进图。"
        )
    else:
        _timestamped_log(
            "macro5 watchdog：结算页面已确认，执行正常结算并检查本轮出货。"
        )
    loot_scanned = False
    settlement_gaps = (1500, 1500, 1500, 1500, 1500, 500)
    if first_a_already_pressed:
        # Result detection has already sent the first A and waited its 1500ms
        # gap, so only the remaining five original presses are needed.
        settlement_gaps = settlement_gaps[1:]
    for gap_ms in settlement_gaps:
        if not failed_round_left and not loot_scanned:
            snapshot = capture.snapshot()
            if (
                snapshot.frame is not None
                and detect_loot_page(snapshot.frame).detected
            ):
                scan_result = scan_and_record_loot_page(
                    context,
                    capture,
                    history,
                    close_gap_ms=gap_ms,
                )
                if scan_result is False:
                    return False
                if scan_result is True:
                    loot_scanned = True
                    continue
        if not context.tap(BIT_A, hold_ms=50, gap_ms=gap_ms):
            return False

    # Normally the page is caught before one of the six original presses.  If
    # it appeared only after the final press, inspect and close it here rather
    # than leaving the next round stuck on the obtained-items screen.
    if not failed_round_left and not loot_scanned:
        snapshot = capture.snapshot()
        if (
            snapshot.frame is not None
            and detect_loot_page(snapshot.frame).detected
        ):
            scan_result = scan_and_record_loot_page(
                context,
                capture,
                history,
                close_gap_ms=500,
            )
            if scan_result is False:
                return False

    # The inventory warning can appear only after the remaining settlement A
    # presses have completed.  Close it with three confirmations, then make
    # the 90-minute sale timer immediately due.  The actual sale remains at
    # the next loop boundary, where it is safe to leave the result screens.
    snapshot = capture.snapshot()
    if (
        snapshot.frame is not None
        and detect_inventory_full_popup(snapshot.frame).detected
    ):
        loop_state.next_sell_active = context.active_monotonic()
        _timestamped_log(
            "macro5 watchdog：检测到武器仓库已满，卖装倒计时已归零；"
            "执行三次A确认，下一轮开始前立即卖装。"
        )
        for _ in range(3):
            if not context.tap(BIT_A, hold_ms=50, gap_ms=500):
                return False
    return True


def run_l_r_a_repeat_window(
    context: _MacroContext,
    duration_ms: int,
) -> bool:
    """Rapidly press and release L, R, and A together for a fixed duration."""
    repeat_bits = (1 << BIT_L) | (1 << BIT_R) | (1 << BIT_A)
    elapsed_ms = 0
    while elapsed_ms < duration_ms:
        press_ms = min(
            LRA_REPEAT_PRESS_MS,
            duration_ms - elapsed_ms,
        )
        context.active_bits |= repeat_bits
        context.send_active_bits()
        if not context.wait_ms(press_ms):
            return False
        elapsed_ms += press_ms

        context.active_bits &= ~repeat_bits
        context.send_active_bits()
        if elapsed_ms >= duration_ms:
            break

        release_ms = min(
            LRA_REPEAT_RELEASE_MS,
            duration_ms - elapsed_ms,
        )
        if not context.wait_ms(release_ms):
            return False
        elapsed_ms += release_ms
    return True


def run_sell_equipment_sequence(context: _MacroContext) -> bool:
    """Run the original 21-step periodic equipment-sale sequence unchanged."""
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


def dismiss_shiver_page_if_present(
    context: _MacroContext,
    capture: FFmpegCapture,
) -> bool:
    """Press B exactly once when Shiver's page is currently visible."""
    snapshot = capture.snapshot()
    if snapshot.frame is None:
        return True
    detection = detect_shiver_page(snapshot.frame)
    if not detection.detected:
        return True
    _timestamped_log(
        "macro5 watchdog：检测到莎莎（Shiver）配件页面，按一次B返回；随后从X重新开始。"
    )
    return context.tap(BIT_B, hold_ms=50, gap_ms=1_500)


def run_macro5_watchdog_loop(
    context: _MacroContext,
    capture: FFmpegCapture,
    bar_monitor: ContinuousBarMonitor,
    loop_state: WatchdogLoopState,
    history: Macro5History,
    sale_in_progress_event: threading.Event,
    restart_in_progress_event: threading.Event,
) -> None:
    while not context.stop_event.is_set():
        # A failed entry can leave the accessory-development page open.  Close
        # it only when the capture image positively matches Shiver's page.
        if not dismiss_shiver_page_if_present(context, capture):
            return

        now_active = context.active_monotonic()
        if now_active >= loop_state.next_sell_active:
            _timestamped_log("macro5 watchdog：到达90分钟定时点，开始卖装。")
            sale_in_progress_event.set()
            try:
                if not run_sell_equipment_sequence(context):
                    return
                for _ in range(5):
                    if not context.tap(BIT_B, hold_ms=50, gap_ms=500):
                        return
                if not context.wait_ms(1000):
                    return
            finally:
                sale_in_progress_event.clear()
            now_active = context.active_monotonic()
            while loop_state.next_sell_active <= now_active:
                loop_state.next_sell_active += SELL_EQUIPMENT_INTERVAL_SECONDS

        # Enter map: always begin a new round explicitly with X.  Keep the
        # base-HUD auto-restart watchdog suspended through every confirmation
        # press and the loading transition.  Clearing this flag immediately
        # after X allowed the still-visible base HUD to request another restart
        # while the map was loading; that delayed X could then reach gameplay
        # and recall the drone/boat.
        restart_in_progress_event.set()
        _timestamped_log("macro5 watchdog：开始新一轮，正在发送X。")
        if not context.tap(BIT_X, hold_ms=50, gap_ms=1_000):
            return
        _timestamped_log("macro5 watchdog：X已发送，继续确认进入地图。")
        restart_from_x = False
        for bit_index, gap_ms in (
            (BIT_A, 500),
            (BIT_A, 500),
            (BIT_A, 500),
            (BIT_A, 500),
            (BIT_A, 500),
            (BIT_A, 0),
        ):
            if not context.tap(bit_index, hold_ms=50, gap_ms=gap_ms):
                return
            # If X was swallowed, the first A may open Shiver's page.  Detect it
            # immediately, close it with one B, then restart at the single X.
            snapshot = capture.snapshot()
            if (
                snapshot.frame is not None
                and detect_shiver_page(snapshot.frame).detected
            ):
                if not dismiss_shiver_page_if_present(context, capture):
                    return
                restart_from_x = True
                break
        if restart_from_x:
            continue

        entry_state = wait_for_battle_start_after_entry(context, capture)
        if entry_state is None:
            return
        if entry_state == "shiver":
            if not dismiss_shiver_page_if_present(context, capture):
                return
            continue

        # Only now is the round visibly active.  The display thread also keeps
        # the base-HUD watchdog suspended while the battle bar is present, so
        # neither X nor the sale-menu D-pad sequence can leak into combat.
        restart_in_progress_event.clear()

        round_started_active = context.active_monotonic()
        bar_monitor.reset()

        # Keep L+R+A repeating for 25.2 seconds in place, then continue the
        # same repeat while walking forward for the final 2.8 seconds.
        if not run_l_r_a_repeat_window(
            context,
            LRA_FORWARD_START_MS,
        ):
            return

        context.active_bits = (
            context.active_bits & ~context.stick_direction_mask
        ) | (1 << BIT_LSTICK_UP)
        context.send_active_bits()
        if not run_l_r_a_repeat_window(
            context,
            LRA_FORWARD_DURATION_MS,
        ):
            return
        context.center_stick()

        # From here until the 45-second deadline, wait for an actual Clear!! or
        # Result page rather than treating a missing battle bar as usable.
        # This avoids sending the first A into the transition animation.
        check_deadline = round_started_active + ROUND_WATCHDOG_MS / 1000.0
        settlement_ready = wait_for_settlement_screen_or_battle_timeout(
            context,
            capture,
            check_deadline,
        )
        if settlement_ready in {None, "battle"}:
            if settlement_ready is None:
                _timestamped_log(
                    "macro5 watchdog：画面状态暂时无法确认；"
                    "不中断执行，转入后退及局内恢复检查。"
                )
            settlement_ready = run_stuck_backstep_recovery(
                context,
                capture,
                bar_monitor,
            )
            if settlement_ready is None:
                # This path now returns None only for an explicit stop or a
                # failed serial operation; visual misses retry internally.
                return
        failed_round_left = settlement_ready == "failed_round_left"
        round_failed = settlement_ready in {
            "result_pressed",
            "failed_round_left",
        }
        if not run_normal_settlement(
            context,
            capture,
            history,
            loop_state,
            first_a_already_pressed=(settlement_ready == "result_pressed"),
            failed_round_left=failed_round_left,
        ):
            return

        loop_state.round_count += 1
        if round_failed:
            loop_state.failure_count += 1
        history.record_round(round_failed)
        failure_rate = (
            100.0 * loop_state.failure_count / loop_state.round_count
        )
        (
            session_rounds,
            _session_failures,
            session_rainbow_rounds,
            session_rainbow_items,
            session_supreme_items,
        ) = history.session_totals()
        rainbow_rate = (
            100.0 * session_rainbow_rounds / session_rounds
            if session_rounds
            else 0.0
        )
        supreme_rate = (
            100.0 * session_supreme_items / session_rainbow_items
            if session_rainbow_items
            else 0.0
        )
        _timestamped_log(
            "macro5 watchdog："
            f"总轮数 {loop_state.round_count}｜"
            f"失败次数 {loop_state.failure_count}｜"
            f"失败率 {failure_rate:.1f}%｜"
            f"彩装轮数 {session_rainbow_rounds}｜"
            f"彩装 {session_rainbow_items}件｜"
            f"彩装率 {rainbow_rate:.3f}%｜"
            f"绝品 {session_supreme_items}件｜"
            f"绝品率 {supreme_rate:.3f}%。"
        )


def request_macro_restart(
    context: _MacroContext,
    pause_state: _PauseState,
    restart_event: threading.Event,
    restart_needs_detection_event: threading.Event,
    restart_in_progress_event: threading.Event,
    *,
    reconnect_controller: bool,
    reason: str,
) -> None:
    """Discard the current sequence and request a fresh start from X."""
    if restart_event.is_set():
        return
    if pause_state.is_paused():
        pause_state.toggle()
    # R is a clean restart, so never run P's "resume old sequence" detection.
    pause_state.consume_resume_detection()
    if reconnect_controller:
        restart_needs_detection_event.set()
    else:
        restart_needs_detection_event.clear()
    restart_in_progress_event.set()
    restart_event.set()
    context.controller.release()
    _timestamped_log(reason)


def send_zl_zr_combo(
    context: _MacroContext,
    *,
    isolated: bool = False,
    source: str = "U指令",
) -> None:
    """Overlay one simultaneous ZL+ZR pulse and restore the current state."""
    combo_bits = (1 << BIT_ZL) | (1 << BIT_ZR)
    paused = context.is_paused()
    base_bits = 0 if isolated or paused else context.active_bits
    context.controller.send_bits(base_bits | combo_bits)
    context.stop_event.wait(ZL_ZR_COMBO_HOLD_MS / 1000.0)
    restore_bits = (
        0
        if isolated or context.is_paused()
        else context.active_bits
    )
    context.controller.send_bits(restore_bits)
    _timestamped_log(
        f"{source}：ZL+ZR已同时按下{ZL_ZR_COMBO_HOLD_MS}ms并释放。"
    )


def run_press_prompt_recovery(
    context: _RestartableMacroContext,
    hold_event: threading.Event,
    hold_ack_event: threading.Event,
    action_in_progress_event: threading.Event,
) -> None:
    """Press the title prompt once, wait five seconds, then resume the macro."""
    if action_in_progress_event.is_set():
        return
    action_in_progress_event.set()
    hold_event.set()
    try:
        # Let the macro worker release its current state before sending the
        # isolated title-screen combination. If it is already manually paused
        # or between actions, the timeout still allows recovery to proceed.
        hold_ack_event.wait(0.5)
        send_zl_zr_combo(
            context,
            isolated=True,
            source="检测到PRESS提示",
        )
        _timestamped_log(
            f"PRESS提示已处理：暂停原宏"
            f"{PRESS_PROMPT_POST_WAIT_SECONDS:g}秒后继续。"
        )
        context.stop_event.wait(PRESS_PROMPT_POST_WAIT_SECONDS)
    finally:
        hold_event.clear()
        action_in_progress_event.clear()


def listen_for_watchdog_keyboard(
    context: _MacroContext,
    combo_context: _MacroContext,
    pause_state: _PauseState,
    restart_event: threading.Event,
    restart_needs_detection_event: threading.Event,
    restart_in_progress_event: threading.Event,
    history: Macro5History,
    worker_errors: List[BaseException],
) -> None:
    """P pauses, R restarts, U sends ZL+ZR, L/H report, E edits."""
    if sys.platform != "win32":
        _listen_for_keyboard_control(context, pause_state, worker_errors)
        return
    if not sys.stdin.isatty():
        context.stop_event.wait()
        return

    try:
        import msvcrt

        while not context.stop_event.is_set():
            if not msvcrt.kbhit():
                context.stop_event.wait(0.05)
                continue
            key = _read_windows_terminal_key()
            if key.lower() == "l":
                history.print_session_summary()
                continue
            if key.lower() == "h":
                history.print_all_time_summary()
                continue
            if key.lower() == "e":
                history.edit_history_interactively()
                continue
            if key.lower() == "r":
                request_macro_restart(
                    context,
                    pause_state,
                    restart_event,
                    restart_needs_detection_event,
                    restart_in_progress_event,
                    reconnect_controller=True,
                    reason=(
                        "收到R重开指令：已释放手柄；重新检测连接后从X开始。"
                    ),
                )
                continue
            if key.lower() == "u":
                send_zl_zr_combo(combo_context)
                continue
            if key and not _handle_terminal_control_key(
                key,
                context,
                pause_state,
            ):
                return
    except BaseException as exc:
        worker_errors.append(exc)
        context.stop_event.set()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Video-aware macro5 with a 45-second battle-bar watchdog."
    )
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument(
        "--device-name",
        default="",
        help="DirectShow设备名/编号；留空时从配置读取并自动枚举",
    )
    parser.add_argument(
        "--capture-spec",
        default="",
        help="输入规格，例如 '1920x1080 / mjpeg'；留空时读取采集配置",
    )
    parser.add_argument(
        "--capture-fps",
        type=int,
        default=0,
        help="采集帧率；0表示读取配置",
    )
    parser.add_argument(
        "--capture-timeout",
        type=float,
        default=0.0,
        help="每种输入规格等待首帧的秒数；0表示读取配置",
    )
    parser.add_argument("--baudrate", type=int, default=9600)
    parser.add_argument("--probe-timeout", type=float, default=1.2)
    parser.add_argument("--scan-interval", type=float, default=1.0)
    return parser.parse_args()


def _heartbeat_age(path: Path, now: float) -> float | None:
    try:
        return max(0.0, now - path.stat().st_mtime)
    except OSError:
        return None


def _terminate_supervised_child(child: subprocess.Popen[bytes]) -> None:
    """Terminate a wedged child and its FFmpeg subprocess."""
    if child.poll() is not None:
        return
    if sys.platform == "win32":
        with contextlib.suppress(Exception):
            subprocess.run(
                ["taskkill", "/PID", str(child.pid), "/T", "/F"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                timeout=10,
                check=False,
            )
    else:
        with contextlib.suppress(Exception):
            child.terminate()
    with contextlib.suppress(Exception):
        child.wait(timeout=5)
    if child.poll() is None:
        with contextlib.suppress(Exception):
            child.kill()


def run_with_hard_recovery_supervisor() -> int:
    """Run macro5 in a child process that can be killed if Python wedges."""
    with tempfile.TemporaryDirectory(prefix="macro5-watchdog-") as raw_dir:
        heartbeat_dir = Path(raw_dir)
        macro_path = heartbeat_dir / "macro.heartbeat"
        capture_path = heartbeat_dir / "capture.heartbeat"
        restart_count = 0

        while True:
            with contextlib.suppress(OSError):
                macro_path.unlink()
            with contextlib.suppress(OSError):
                capture_path.unlink()
            environment = dict(os.environ)
            environment[AUTO_RECOVERY_CHILD_ENV] = "1"
            environment[AUTO_RECOVERY_MACRO_HEARTBEAT_ENV] = str(macro_path)
            environment[AUTO_RECOVERY_CAPTURE_HEARTBEAT_ENV] = str(capture_path)
            command = [sys.executable, str(Path(__file__).resolve()), *sys.argv[1:]]
            child = subprocess.Popen(command, env=environment)
            monitoring_ready = False
            reason = ""
            try:
                while True:
                    return_code = child.poll()
                    if return_code is not None:
                        if return_code == 0:
                            return 0
                        if not monitoring_ready:
                            # A bad command line, missing dependency, occupied
                            # capture card, etc. needs a visible user fix and
                            # must not become an endless restart loop.
                            return return_code
                        reason = f"子程序异常退出（代码{return_code}）"
                        break

                    now = time.time()
                    macro_age = _heartbeat_age(macro_path, now)
                    capture_age = _heartbeat_age(capture_path, now)
                    if not monitoring_ready:
                        # Do not punish the legitimate indefinite serial-port
                        # discovery screen. Monitoring starts only after both
                        # the controller and capture threads have reported in.
                        if macro_age is not None and capture_age is not None:
                            monitoring_ready = True
                            _timestamped_log(
                                "外层防卡死监控已就绪：宏和采集画面均有心跳。"
                            )
                    else:
                        if (
                            macro_age is None
                            or macro_age > AUTO_RECOVERY_MACRO_TIMEOUT_SECONDS
                        ):
                            reason = (
                                "宏线程超过"
                                f"{AUTO_RECOVERY_MACRO_TIMEOUT_SECONDS:g}秒无心跳"
                            )
                            break
                        if (
                            capture_age is None
                            or capture_age > AUTO_RECOVERY_CAPTURE_TIMEOUT_SECONDS
                        ):
                            reason = (
                                "采集画面超过"
                                f"{AUTO_RECOVERY_CAPTURE_TIMEOUT_SECONDS:g}秒未刷新"
                            )
                            break
                    time.sleep(0.5)
            except KeyboardInterrupt:
                _terminate_supervised_child(child)
                return 130

            restart_count += 1
            _timestamped_log(
                f"检测到程序卡死：{reason}。正在强制完整重启"
                f"（第{restart_count}次），随后从X开始。"
            )
            _terminate_supervised_child(child)
            time.sleep(AUTO_RECOVERY_RESTART_DELAY_SECONDS)


def main() -> int:
    args = parse_args()
    config_path = _resolve_config_path(args.config)
    macro_heartbeat = _HeartbeatFile(AUTO_RECOVERY_MACRO_HEARTBEAT_ENV)
    capture_heartbeat = _HeartbeatFile(AUTO_RECOVERY_CAPTURE_HEARTBEAT_ENV)
    stop_event = threading.Event()
    restart_event = threading.Event()
    restart_needs_detection_event = threading.Event()
    restart_in_progress_event = threading.Event()
    restart_in_progress_event.set()
    sale_in_progress_event = threading.Event()
    press_prompt_hold_event = threading.Event()
    press_prompt_hold_ack_event = threading.Event()
    press_prompt_action_event = threading.Event()
    pause_state = _PauseState()
    bar_monitor = ContinuousBarMonitor()
    worker_errors: List[BaseException] = []
    history = Macro5History(
        Path(__file__).resolve().with_name(LOOT_HISTORY_FILENAME)
    )
    controller = None
    capture: FFmpegCapture | None = None
    capture_settings: CaptureSettings | None = None

    try:
        capture_settings = resolve_capture_settings(args, config_path)
        capture = FFmpegCapture(capture_settings, stop_event, capture_heartbeat)
        _timestamped_log(
            "视频采集配置："
            f"{capture_settings.device_label}；"
            f"{capture_settings.width}x{capture_settings.height} / "
            f"{capture_settings.pixel_format or '自动'} @ "
            f"{capture_settings.fps}fps。"
        )
        capture.start(worker_errors)
        if not capture.wait_until_ready():
            detail = capture.snapshot().error or "等待首帧超时"
            if capture_error_may_indicate_device_busy(detail):
                raise RuntimeError(capture_device_busy_message(detail))
            raise RuntimeError(
                f"无法读取视频设备“{capture_settings.device_label}”。"
                f"FFmpeg原始错误：{detail}"
            )
        active_spec = capture.active_capture_spec
        _timestamped_log(
            f"采集卡已连接：{capture_settings.device_label}；"
            f"实际输入：{active_spec.get('width', capture_settings.width)}x"
            f"{active_spec.get('height', capture_settings.height)} / "
            f"{active_spec.get('pixel_format', capture_settings.pixel_format) or '自动'} "
            f"@ {capture_settings.fps}fps。"
        )

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
        macro_context = _RestartableMacroContext(
            controller,
            stop_event,
            pause_state,
            restart_event,
            macro_heartbeat,
            press_prompt_hold_event,
            press_prompt_hold_ack_event,
        )
        macro_heartbeat.beat(force=True)
        keyboard_context = _MacroContext(controller, stop_event, pause_state)

        def macro_worker() -> None:
            try:
                controller_ready = False
                loop_state: WatchdogLoopState | None = None
                while not stop_event.is_set():
                    try:
                        if not controller_ready:
                            if not _run_controller_detection(macro_context):
                                return
                            controller_ready = True
                            if loop_state is None:
                                loop_state = WatchdogLoopState(
                                    next_sell_active=(
                                        macro_context.active_monotonic()
                                        + SELL_EQUIPMENT_INTERVAL_SECONDS
                                    )
                                )
                        assert loop_state is not None
                        run_macro5_watchdog_loop(
                            macro_context,
                            capture,
                            bar_monitor,
                            loop_state,
                            history,
                            sale_in_progress_event,
                            restart_in_progress_event,
                        )
                        return
                    except _RestartRequested:
                        controller.release()
                        macro_context.active_bits = 0
                        needs_detection = (
                            restart_needs_detection_event.is_set()
                        )
                        restart_event.clear()
                        restart_needs_detection_event.clear()
                        controller_ready = not needs_detection
                        if needs_detection:
                            _timestamped_log(
                                "R重开已生效：旧动作已丢弃；重新检测手柄后从X开始。"
                            )
                        else:
                            _timestamped_log(
                                "基地船防卡死已生效：旧动作已丢弃，现在从X开始。"
                            )
            except BaseException as exc:
                worker_errors.append(exc)
                stop_event.set()

        workers = [
            threading.Thread(
                target=macro_worker,
                name="macro5-watchdog-controller",
                daemon=True,
            ),
            threading.Thread(
                target=listen_for_watchdog_keyboard,
                args=(
                    keyboard_context,
                    macro_context,
                    pause_state,
                    restart_event,
                    restart_needs_detection_event,
                    restart_in_progress_event,
                    history,
                    worker_errors,
                ),
                name="macro5-watchdog-keyboard",
                daemon=True,
            ),
        ]
        for worker in workers:
            worker.start()

        print(
            "\n智能 macro5 已启动：P=暂停/恢复，R=丢弃当前进度并从X重开，"
            "U=同时按下ZL+ZR，"
            "L=查看本次启动逐轮记录，H=查看过去永久总和，"
            "E=编辑永久统计，Q=退出。"
            "\n外层防卡死监控已启用：程序彻底无响应时会自动完整重启并从X开始。"
            "\n首次请观察一整轮，确认窗口左上角 BAR 状态提示正确。\n"
        )
        total_rounds, total_failures, total_drops = history.totals()
        _timestamped_log(
            f"永久记录已载入：总轮数{total_rounds}，失败{total_failures}，"
            f"彩装{total_drops}；按L看本次，按H看永久累计。"
        )
        window_title = (
            f"{capture_settings.device_label} - Smart Macro5 Watchdog"
        )
        cv2.namedWindow(window_title, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window_title, CAPTURE_WIDTH, CAPTURE_HEIGHT)
        base_hud_since: float | None = None
        base_hud_elapsed = 0.0
        base_hud_variant: str | None = None

        # Keep video input at its configured frame rate, but avoid spending a
        # full detector pass on every frame.  Critical watchdog state runs at
        # about 15 Hz; detectors used only by the preview/debug overlay run at
        # about 5 Hz.  Frame-sequence strides scale automatically with the
        # configured capture FPS and also guarantee that the same frame is
        # never analysed twice.
        input_fps = max(1, int(capture_settings.fps))
        critical_target_fps = min(input_fps, 15)
        debug_target_fps = min(input_fps, 5)
        critical_frame_stride = max(
            1,
            int(round(input_fps / critical_target_fps)),
        )
        debug_frame_stride = max(
            1,
            int(round(input_fps / debug_target_fps)),
        )
        actual_critical_fps = input_fps / critical_frame_stride
        actual_debug_fps = input_fps / debug_frame_stride
        _timestamped_log(
            "识别调度已启用："
            f"输入{input_fps}fps，关键检测约{actual_critical_fps:g}fps，"
            f"调试检测约{actual_debug_fps:g}fps。"
        )
        last_display_sequence = -1
        last_critical_sequence = -1
        last_debug_sequence = -1
        detection = BarDetection(False, 0, 0, 0.0)
        shiver_detection = ShiverPageDetection(False, 0.0, 0.0, 0.0)
        clear_detection = ClearScreenDetection(False, 0.0, 0.0)
        result_detection = ResultScreenDetection(False, 0.0, 0.0)
        base_hud_detection = BaseHudDetection(
            False,
            0.0,
            0.0,
            0.0,
            0.0,
            None,
        )
        loot_detection = LootPageDetection(
            False,
            0.0,
            0.0,
            0.0,
            False,
            False,
            0.0,
            0.0,
        )
        white_icon_detection = WhiteIconDetection(False)
        press_prompt_detection = PressPromptDetection(False, 0.0, 0.0)
        press_prompt_confirm_count = 0
        press_prompt_clear_count = 0
        press_prompt_latched = False

        while not stop_event.is_set():
            snapshot = capture.snapshot(after_sequence=last_display_sequence)
            if snapshot.frame is None:
                time.sleep(0.03)
                continue
            last_display_sequence = snapshot.sequence
            # snapshot() already returned a private copy; draw directly onto
            # it instead of copying another 960x540 BGR image every frame.
            display = snapshot.frame
            critical_updated = (
                last_critical_sequence < 0
                or snapshot.sequence - last_critical_sequence
                >= critical_frame_stride
            )
            if critical_updated:
                last_critical_sequence = snapshot.sequence
                detection = detect_battle_bar(display)
                base_hud_detection = detect_base_hud(display)
                press_prompt_detection = detect_press_prompt(display)
                bar_monitor.update(
                    detection.detected,
                    macro_context.active_monotonic(),
                )
                if press_prompt_detection.detected:
                    press_prompt_confirm_count += 1
                    press_prompt_clear_count = 0
                    if (
                        press_prompt_confirm_count
                        >= PRESS_PROMPT_CONFIRM_FRAMES
                        and not press_prompt_latched
                        and not press_prompt_action_event.is_set()
                    ):
                        press_prompt_latched = True
                        threading.Thread(
                            target=run_press_prompt_recovery,
                            args=(
                                macro_context,
                                press_prompt_hold_event,
                                press_prompt_hold_ack_event,
                                press_prompt_action_event,
                            ),
                            name="macro5-press-prompt-recovery",
                            daemon=True,
                        ).start()
                else:
                    press_prompt_confirm_count = 0
                    press_prompt_clear_count += 1
                    if (
                        press_prompt_clear_count
                        >= PRESS_PROMPT_CLEAR_FRAMES
                        and not press_prompt_action_event.is_set()
                    ):
                        press_prompt_latched = False
            if (
                last_debug_sequence < 0
                or snapshot.sequence - last_debug_sequence
                >= debug_frame_stride
            ):
                last_debug_sequence = snapshot.sequence
                shiver_detection = detect_shiver_page(display)
                clear_detection = detect_clear_screen(display)
                result_detection = detect_result_screen(display)
                loot_detection = detect_loot_page(display)
                white_icon_detection = detect_in_game_life(display)
            base_watchdog_suspended = (
                pause_state.is_paused()
                or sale_in_progress_event.is_set()
                or restart_in_progress_event.is_set()
                or detection.detected
            )
            if critical_updated:
                if base_hud_detection.detected and not base_watchdog_suspended:
                    now = time.monotonic()
                    if (
                        base_hud_since is None
                        or base_hud_variant != base_hud_detection.variant
                    ):
                        base_hud_since = now
                        base_hud_variant = base_hud_detection.variant
                    base_hud_elapsed = max(0.0, now - base_hud_since)
                    restart_seconds = (
                        BASE_HUD_YELLOW_RESTART_SECONDS
                        if base_hud_variant == "yellow"
                        else BASE_HUD_WHITE_RESTART_SECONDS
                    )
                    if base_hud_elapsed >= restart_seconds:
                        variant_label = (
                            "黄色版"
                            if base_hud_variant == "yellow"
                            else "白色版"
                        )
                        request_macro_restart(
                            keyboard_context,
                            pause_state,
                            restart_event,
                            restart_needs_detection_event,
                            restart_in_progress_event,
                            reconnect_controller=False,
                            reason=(
                                f"基地船右下角菜单提示（{variant_label}）连续存在"
                                f"{restart_seconds:g}秒，自动从X重开。"
                            ),
                        )
                        base_hud_since = None
                        base_hud_elapsed = 0.0
                        base_hud_variant = None
                else:
                    base_hud_since = None
                    base_hud_elapsed = 0.0
                    base_hud_variant = None
            label = (
                "BAR: BATTLE" if detection.detected else "BAR: NOT DETECTED"
            ) + (
                f"  ticks={detection.aligned_tick_count}"
                f" dark={detection.dark_fraction:.2f}"
                f" held={bar_monitor.elapsed(macro_context.active_monotonic()):.1f}s"
            )
            color = (0, 220, 0) if detection.detected else (0, 0, 255)
            display_height, display_width = display.shape[:2]
            cv2.rectangle(
                display,
                (int(display_width * 0.36), int(display_height * 0.03)),
                (int(display_width * 0.70), int(display_height * 0.10)),
                color,
                2,
            )
            cv2.putText(
                display,
                label,
                (16, 218),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.85,
                color,
                2,
                cv2.LINE_AA,
            )
            white_icon_label = (
                f"IN GAME LIFE:{white_icon_detection.life_count}"
            )
            white_icon_color = (
                (0, 255, 0)
                if white_icon_detection.detected
                else (0, 0, 255)
            )
            cv2.rectangle(
                display,
                (
                    int(display_width * 0.005),
                    int(display_height * 0.01),
                ),
                (
                    int(display_width * 0.192),
                    int(display_height * 0.09),
                ),
                white_icon_color,
                2,
            )
            cv2.putText(
                display,
                white_icon_label,
                (16, 248),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                white_icon_color,
                2,
                cv2.LINE_AA,
            )
            shiver_label = (
                "SHIVER: DETECTED"
                if shiver_detection.detected
                else "SHIVER: NO"
            ) + (
                f"  p={shiver_detection.purple_fraction:.2f}"
                f" tl={shiver_detection.teal_left_fraction:.2f}"
                f" tr={shiver_detection.teal_right_fraction:.2f}"
            )
            shiver_color = (
                (0, 255, 255) if shiver_detection.detected else (170, 170, 170)
            )
            cv2.putText(
                display,
                shiver_label,
                (16, 68),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                shiver_color,
                2,
                cv2.LINE_AA,
            )
            clear_label = (
                "CLEAR: DETECTED"
                if clear_detection.detected
                else "CLEAR: NO"
            ) + (
                f"  yellow={clear_detection.yellow_fraction:.2f}"
                f" match={clear_detection.template_recall:.2f}"
            )
            clear_color = (
                (0, 255, 255) if clear_detection.detected else (170, 170, 170)
            )
            cv2.rectangle(
                display,
                (
                    int(display_width * 0.4906),
                    int(display_height * 0.1257),
                ),
                (
                    int(display_width * 0.6540),
                    int(display_height * 0.2692),
                ),
                clear_color,
                2,
            )
            cv2.putText(
                display,
                clear_label,
                (16, 98),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                clear_color,
                2,
                cv2.LINE_AA,
            )
            result_label = (
                "RESULT: DETECTED"
                if result_detection.detected
                else "RESULT: NO"
            ) + (
                f"  white={result_detection.white_fraction:.2f}"
                f" match={result_detection.template_match:.2f}"
            )
            result_color = (
                (0, 255, 0)
                if result_detection.detected
                else (170, 170, 170)
            )
            cv2.rectangle(
                display,
                (
                    int(display_width * 0.505),
                    int(display_height * 0.140),
                ),
                (
                    int(display_width * 0.625),
                    int(display_height * 0.205),
                ),
                result_color,
                2,
            )
            cv2.putText(
                display,
                result_label,
                (16, 128),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                result_color,
                2,
                cv2.LINE_AA,
            )
            base_status = (
                "BASE HUD: YES"
                if base_hud_detection.detected
                else "BASE HUD: NO"
            ) + (
                f"  X={base_hud_detection.menu_match:.2f}"
                f" +={base_hud_detection.equipment_match:.2f}"
                f" type={base_hud_detection.variant or '-'}"
                f" held={base_hud_elapsed:.1f}s"
            )
            if base_watchdog_suspended:
                base_status += "  SUSPENDED"
            base_color = (
                (0, 200, 255)
                if base_hud_detection.detected
                else (170, 170, 170)
            )
            cv2.rectangle(
                display,
                (int(display_width * 0.925), int(display_height * 0.885)),
                (int(display_width * 0.995), int(display_height * 0.995)),
                base_color,
                2,
            )
            cv2.putText(
                display,
                base_status,
                (16, 158),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                base_color,
                2,
                cv2.LINE_AA,
            )
            press_status = (
                "PRESS: YES"
                if press_prompt_detection.detected
                else "PRESS: NO"
            ) + (
                f"  match={press_prompt_detection.template_match:.2f}"
                f" white={press_prompt_detection.white_fraction:.2f}"
            )
            if press_prompt_action_event.is_set():
                press_status += "  HANDLING"
            press_color = (
                (0, 255, 0)
                if press_prompt_detection.detected
                else (170, 170, 170)
            )
            cv2.rectangle(
                display,
                (
                    int(display_width * PRESS_PROMPT_ROI[0]),
                    int(display_height * PRESS_PROMPT_ROI[2]),
                ),
                (
                    int(display_width * PRESS_PROMPT_ROI[1]),
                    int(display_height * PRESS_PROMPT_ROI[3]),
                ),
                press_color,
                2,
            )
            cv2.putText(
                display,
                press_status,
                (16, 188),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                press_color,
                2,
                cv2.LINE_AA,
            )
            loot_status = (
                "LOOT: YES" if loot_detection.detected else "LOOT: NO"
            ) + (
                f"  slot1={loot_detection.slot1_score:.2f}"
                f" slot2={loot_detection.slot2_score:.2f}"
            )
            loot_color = (
                (255, 120, 255)
                if loot_detection.detected
                else (170, 170, 170)
            )
            cv2.putText(
                display,
                loot_status,
                (16, 188),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                loot_color,
                2,
                cv2.LINE_AA,
            )
            if loot_detection.detected:
                cv2.rectangle(
                    display,
                    (
                        int(display_width * 0.06),
                        int(display_height * 0.23),
                    ),
                    (
                        int(display_width * 0.17),
                        int(display_height * 0.41),
                    ),
                    (255, 120, 255)
                    if loot_detection.slot1_rainbow
                    else (170, 170, 170),
                    2,
                )
                cv2.rectangle(
                    display,
                    (
                        int(display_width * 0.18),
                        int(display_height * 0.23),
                    ),
                    (
                        int(display_width * 0.29),
                        int(display_height * 0.41),
                    ),
                    (255, 120, 255)
                    if loot_detection.slot2_rainbow
                    else (170, 170, 170),
                    2,
                )
            cv2.imshow(window_title, display)
            key = cv2.waitKey(1) & 0xFF
            if key in {ord("q"), 27}:
                stop_event.set()
                break
            if key in {ord("r"), ord("R")}:
                request_macro_restart(
                    keyboard_context,
                    pause_state,
                    restart_event,
                    restart_needs_detection_event,
                    restart_in_progress_event,
                    reconnect_controller=True,
                    reason=(
                        "收到R重开指令：已释放手柄；重新检测连接后从X开始。"
                    ),
                )
            if key in {ord("u"), ord("U")}:
                send_zl_zr_combo(macro_context)
            if key in {ord("l"), ord("L")}:
                history.print_session_summary()
            if key in {ord("h"), ord("H")}:
                history.print_all_time_summary()
            time.sleep(0.01)

        for worker in workers:
            worker.join(timeout=2.0)
        if worker_errors:
            raise worker_errors[0]
        return 0
    except KeyboardInterrupt:
        return 0
    except Exception as exc:
        print(f"启动失败：{exc}", file=sys.stderr)
        return 1
    finally:
        stop_event.set()
        if controller is not None:
            with contextlib.suppress(Exception):
                controller.release()
            with contextlib.suppress(Exception):
                controller.close()
        if capture is not None:
            capture.close()
        with contextlib.suppress(Exception):
            cv2.destroyAllWindows()


if __name__ == "__main__":
    if os.environ.get(AUTO_RECOVERY_CHILD_ENV) == "1":
        raise SystemExit(main())
    raise SystemExit(run_with_hard_recovery_supervisor())
