"""Video frontend and hard-recovery watchdog for Smart Macro6.

The child process monitors both the controller worker and FFmpeg capture with
separate heartbeats.  Its preview recognizes the requested Pokopia screen
states while keyboard 1/2/3/4 starts a macro, 0 interrupts the active macro,
F10 saves the untouched capture frame, and F12 controls the operation lock.
A parent process replaces the child if either heartbeat stops after startup.

After recovery Macro6 deliberately returns to its safe idle state.  It never
replays part 1 or part 2 automatically because either part can cross the Home
menu and repeating an interrupted sequence may operate on the wrong screen.
"""

from __future__ import annotations

import argparse
import base64
from collections import deque
import contextlib
import difflib
import json
import os
import re
import signal
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import unicodedata
import zlib
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

if os.name == "nt":
    import msvcrt
    import winsound
else:
    import select
    import termios
    import tty


REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from autocontroller_rebuild_for_RL.macro_gamepad import (
    DEFAULT_CONFIG,
    SerialRemoteController,
    _MANUAL_ARROW_BITS,
    _MANUAL_KEY_BITS,
    _MacroContext,
    _PauseState,
    _commit_status_line,
    _handle_terminal_control_key,
    _load_config,
    _resolve_config_path,
    _timestamped_log,
    wait_for_serial_selection,
)
from autocontroller_rebuild_for_RL.smart_macro_gamepad import (
    _read_posix_terminal_key,
)
from autocontroller_rebuild_for_RL.chrome_code_input import (
    input_code_to_chrome,
)
from autocontroller_rebuild_for_RL.player_name_recognizer import (
    PLAYER_NAME_MODEL,
    PlayerNameRecognizerUnavailable,
    recognize_player_name,
    repair_player_name_right_edge,
    warm_up_player_name_recognizer,
)
from autocontroller_rebuild_for_RL.pokopia_web_state import (
    PokopiaWebStatePublisher,
)
from switch_connect.virtual_gamepad.input_mapper import (
    BIT_A,
    BIT_B,
    BIT_DPAD_DOWN,
    BIT_DPAD_LEFT,
    BIT_DPAD_RIGHT,
    BIT_DPAD_UP,
    BIT_HOME,
    BIT_LSTICK_DOWN,
    BIT_LSTICK_LEFT,
    BIT_LSTICK_RIGHT,
    BIT_LSTICK_UP,
    BIT_MINUS,
    BIT_PLUS,
)
from vision_capture.adapter import (
    FFmpegCaptureSource,
    capture_device_busy_message,
    capture_error_may_indicate_device_busy,
    is_usb_capture_device_name,
    list_avfoundation_video_device_rows,
    rank_capture_device_name,
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
MACRO_HEARTBEAT_TIMEOUT_SECONDS = 45.0
CAPTURE_HEARTBEAT_TIMEOUT_SECONDS = 15.0
RESTART_DELAY_SECONDS = 3.0
MACRO6_BUILD_ID = "v1.62"
SUPERVISED_CHILD_ENV = "MACRO6_SUPERVISED_CHILD"
MACRO_HEARTBEAT_ENV = "MACRO6_MACRO_HEARTBEAT"
CAPTURE_HEARTBEAT_ENV = "MACRO6_CAPTURE_HEARTBEAT"
RUN_COUNTER_FILENAME = "macro6_watchdog_stats.json"
CODE_ARCHIVE_DIRNAME = "pokopia_stamp_records"
CODE_TIMER_RESTART_SECONDS = 15 * 60
SEVERE_CODE_TIMER_RESTART_SECONDS = 30 * 60
CODE_OCR_TIMEOUT_SECONDS = 3 * 60
CODE_OCR_UNKNOWN_AFTER_FAILURES = 10
CONNECT_OK_CODE_RETRY_SECONDS = 30.0
CONTROLLER_IDLE_KEEPALIVE_SECONDS = 5 * 60
CONTROLLER_IDLE_KEEPALIVE_HOLD_MS = 50
CONTROLLER_IDLE_KEEPALIVE_GAP_MS = 100
BEIJING_TIMEZONE = timezone(timedelta(hours=8))
DAILY_BOUNDARY_HOUR_BEIJING = 5
BUSINESS_DATE_HOLD_MINUTES = 10


def _beijing_operational_date(now: datetime | None = None) -> str:
    """Return the date for a Beijing 05:00-to-next-05:00 operating day."""
    current = now or datetime.now(timezone.utc)
    if current.tzinfo is None:
        current = current.replace(tzinfo=timezone.utc)
    beijing_now = current.astimezone(BEIJING_TIMEZONE)
    shifted = beijing_now - timedelta(hours=DAILY_BOUNDARY_HOUR_BEIJING)
    return shifted.date().isoformat()


def _in_business_date_hold_window(now: datetime | None = None) -> bool:
    """Return true from Beijing 04:50:00 through 04:59:59."""
    current = now or datetime.now(timezone.utc)
    if current.tzinfo is None:
        current = current.replace(tzinfo=timezone.utc)
    beijing_now = current.astimezone(BEIJING_TIMEZONE)
    boundary = beijing_now.replace(
        hour=DAILY_BOUNDARY_HOUR_BEIJING,
        minute=0,
        second=0,
        microsecond=0,
    )
    hold_start = boundary - timedelta(minutes=BUSINESS_DATE_HOLD_MINUTES)
    return hold_start <= beijing_now < boundary


@dataclass(frozen=True)
class FrameSnapshot:
    sequence: int
    frame: np.ndarray | None
    error: str = ""
    raw_frame: np.ndarray | None = None


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
        width_text, height_text = [
            part.strip() for part in size.lower().split("x", 1)
        ]
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
    """Resolve Macro6 capture settings without depending on Macro5."""
    runtime_config = _load_config(config_path)
    capture_config = _load_config(_capture_config_path(runtime_config))

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
            raise RuntimeError(f"视频规格尺寸必须大于0：{width}x{height}")

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
        raise RuntimeError(f"采集读取超时必须大于0：{read_timeout_seconds}")

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
            row
            for row in rows
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
        suffix = (
            f" FFmpeg设备枚举错误：{enumeration_error}"
            if enumeration_error
            else ""
        )
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


SAVE_CHECK_ROI = (0.795, 0.395, 0.835, 0.455)
ABLE_ACCESS_ROI = (0.915, 0.925, 0.999, 0.995)
BLACK_ROI = (0.450, 0.411, 0.550, 0.589)
CONNECT_ROI = (0.200, 0.800, 0.400, 0.910)
CODE_PANEL_ROI = (0.715, 0.215, 0.860, 0.290)
# Keep enough room on the left for six wide glyphs such as M/W.  The right
# edge deliberately stops before the eye icon next to the CODE pill.
CODE_TEXT_ROI = (0.720, 0.220, 0.855, 0.285)
CURSOR_AREA_ROI = (0.050, 0.200, 0.930, 0.940)
STAMP_CONTEXT_ROI = (0.680, 0.070, 0.995, 0.920)
STAMP_BUTTON_ROI = (0.710, 0.770, 0.980, 0.880)
REWARD_ROI = (0.400, 0.150, 0.590, 0.260)
# The completed-task list is horizontally centered as a whole, so its left
# and right edges move with proportional text width.  Count horizontal text
# bands across a wide central region instead of sampling fixed x positions.
REWARD_LIST_ROI = (0.250, 0.340, 0.800, 0.620)
NETWORK_ERROR_CLOSE_ROI = (0.325, 0.650, 0.400, 0.735)
NETWORK_ERROR_CLOSE_TEMPLATE_SHAPE = (48, 96)
NETWORK_ERROR_CLOSE_MIN_DICE = 0.78
NETWORK_ERROR_CONFIRM_FRAMES = 2
REOPEN_STAGE_FEATURE_SHAPE = (16, 32)
REOPEN_STAGE_MIN_CORRELATION = 0.72
REOPEN_STAGE_CONFIRM_FRAMES = 2
# “覆盖主机的保存数据” and its disabled “无法覆盖数据” state have almost
# identical grayscale structure.  The enabled button is bright orange-red,
# while the disabled button is dark red, so OVERWRITE additionally requires
# a substantial proportion of bright orange-red pixels in its button ROI.
REOPEN_OVERWRITE_MIN_BRIGHT_RED_RATIO = 0.50
# Compact 32x16 grayscale features generated from the four supplied Switch
# system screens.  Runtime recognition is self-contained and never opens the
# source PNG files.
REOPEN_STAGE_FEATURES = {
    "GO_GAME": (
        (0.455, 0.270, 0.535, 0.320),
        "eNpTUiIIlFWAUFlZSRnMBNJAjCwtIyElISMrpyIHlJCUlVeWl5eVRVKhYB/gEWBvbybhJC6jGGGhJ2tlZW2rAFegLOeX3pXlZ5WRNiXeorm7uUQwJs83x0AOYYC0ZJykhLOBT629Q3Rdp6Nqe42oc540woDMypn1GVFRMe1hwT3lFZOzJxXU1oXIqMDkVRQsJrnLyHs3FwSalZeWVRqk2eRG+ErC5ZUkc7LNtAJTA23C5SvKy8s0oz18rX0kleHeK4r2qCkuSu3KmWlRXVVVr5nX7FptJ4Ow31VOK9rDwMDV1NbYzcrSQ9HH1iJGESmApJUVxKXk5aVkZeVlZGSllaRl5CRQAhDoRmUEAIWvihK9AAACC2PN",
    ),
    "DOWNLOAD": (
        (0.130, 0.195, 0.305, 0.275),
        "eNpzcqIQOLu4OLs4AylnMAAxUKRtLSxtrGwdbKxtraxtbe2d7KztkBXYZE3uyquvCGtsT6hITqjK9shvTrJFVuDREVexb1XZ5IaylXO7jiyduHJ2kbULQtrVtKKjeGHLxImLMoo7S6fOye5bl2mDpN/ZLqOoZWZHUdea+slzW3pbGxYtT0eRtyl0S9u6tLhtQUpBWdqyyTlTJ2cj2+9sm+rjV9FRkp+WlFsYNqGqoacv2wbFh45O9nbW9nYOdsAQsLZ2dLB3xAwjYLA4O4MDC4idBhMAAC3rneE=",
    ),
    "CLOSE_GAME": (
        (0.495, 0.640, 0.780, 0.760),
        "eNqLKi0oLIKAwsLCgoL8vNycrIy05ITo8JCgwADfttLi4uISMCguBqsCKsnJBqqIj44ID40oNLOygANLKLCysra2sbG1s7N1qXJydcEAzjDg6FNl7+yEBIBiyFwH32oUeUd7BzsUeb8aZHlHj+DACGT1Dv61SPLO9iEl6WUujkgC6PLl2XlI+p3tA5Dtd3YIiE9IckXWH1DhgGyfs6OTM6r+YhR5NACUz3Vwwiuf4uCIGzjZ+ke5uLq6ukGAuxsCuIOAh1O9f3pyWlZ+SUVNfWNza1s7ELS1Njc11Ld09U+ePntZKwA5t5pt",
    ),
    "OVERWRITE": (
        (0.245, 0.680, 0.755, 0.800),
        "eNpjY2HFB9gFxMVwAXFxCQlBh4ToGJwgOsHBNT4qGieIindzwS/viiQfEQkUiYqMiorCKh+VnhoTmxCXGhuXiE0+NmTexskTZ82YvXx1f3gMpnxM6PQt02ZPX7Zw4ZqpoVjlZ2ydOnnWtAWzVk8OwyYfXtXfWVVZVVTWWR2BRR6oICQ0PCw8IiI0HLv/YmJjYeGGJB8H8jAOEBnn6pYcF48TxCW7aTnbO+ACjg4uOgBGp6aq",
    ),
}
# Neutral-white “关闭” glyph mask from the fixed lower-left button in
# SCREENSHOT_20260831T103601_694441Z.  Runtime recognition is self-contained;
# the source screenshot is not opened by watchdog.
NETWORK_ERROR_CLOSE_MASK_ZLIB_BASE64 = (
    "eNpjYBhWgJGRoQGJnf4fxmEH8qFM+f8fGRph4gKMDA8R7PqfCDZjI0I9nM1QwcgG"
    "Z2cwSsDZbYwFMDY7YyM7jC3H8BFkNRQYMvAxDDMAACVUEAI="
)
PLAYER_NOTIFICATION_ROI = (0.580, 0.035, 0.998, 0.100)
PLAYER_CONNECTION_ICON_ROI = (0.535, 0.018, 0.590, 0.115)
# The globe is also used for fixed system notices.  This tight region covers
# the recycle-bin icon and “回收箱中存有物品。” text but excludes the globe.
PLAYER_RECYCLE_NOTICE_ROI = (0.590, 0.042, 0.770, 0.094)
PLAYER_RECYCLE_NOTICE_SHAPE = (28, 192)
PLAYER_RECYCLE_NOTICE_MAX_ERROR = 0.080
PLAYER_RECYCLE_NOTICE_ZLIB_BASE64 = (
    "eNrtlntQVFUYwD+Ut/LQBEEUw4BdQWJWkxhYWW0V0RhHsF0EkhwVLRI3dFKccdXBR5MuTlIoClgoLioqvrdBhCXdEkYdWcSVCmnQVUlHknTGUvz6zt3dy0Zmm9PUOPH74373nHvuOb977nkB/MtEK9+BXnrppReeoenDCV8bStoXuwMMKvYwJ5OSuLAhyLqMW7qzdVK8uEcdC6XWqeVjAVzT+OTg4TzDbNX3vGjMydEs2ack1pbMDiQGULZriTUf09qXDU7oDZDX4WV6MbhpB5X1TLmfxN6xtPcuWgTdWfapM4Fm+nKZLpWZ3U2XDNTJARLr+Yyax9+ZedBhq79C25icnNwfEpTKpAHzq3NycjY0zKXPwjnJPOuaAWbVcP7ezUVCwhvELYb2ToNhWdf3rY8MBsN+AIFQGOS1Gmez50IP2NuO7Yj3W7p+REbKejVxEs+pTRSNBBzG/A/quRfYX6y5bbGqtNl/7Ymk7oSTjCjDbczfDXz9whiDQdrtH/VEr9ffwyUhzR9CVjkMbC5yHXVzGns3pQvx4WU0o4WAYFnw3uoYt4SRrFKZ0yS6pN+4KDPzBvjisTvf7PMy6Dtu6n+50f6c/psSxb/PeHmGxb+8lVNZY+Xfv+YRFTlQBBAGIIsG5/dHi0Tz5aIRAB+dSCAKcCULF7n2Ay/nTD1oIpWreykuFplwoJaPJFxRJWw8AOqlcDWL8+8wPxWdtdl/d1UP/7wsi/+XefGExtp/J3L+Cq4k0qBX3aojfqgg/3yr8c/1n8shtYOEVRFf3/Uepe2yHl6q46hHP3BUbwKdfM5NaVDrrPhTscx/ax3PKVv9u7CHf3wt7z+PpbdZ+xevMs4Fd60CQso1GtRqSlWfsDKZT/P3/OJBIFejd0ltNgXndYhjTG30IX/nXCH5CxaB/GvIrRjT/pzLZ9fZXOXnJZtnecMQCav9JcGFZ/jT71E7im9Rx4+V7seZUrGqrYq48gf/8xBdiZ1cC6lN5YMoJOoMhVhXZYL8Gbrcqj7iNumIx5PJf5rpkcjNFMNs9M9UZo1YFXj6M+XWLW8qldWNZ57tL8CYspNcr17DHR6gKpcQmyvAscyoJb7FBhbulsHCbWmdcJzuW/EcXWWJ+4ZkY7qEYzzrf6lWe69xmb+u1HFPmy/5Z6BCIln8QDoQJ0skU7HQRv8zM7IgTrBACLEA08WRUccuYKjJf3+BrlhXucvs3xCXx/zt8m+3RbN9YyfiZo2PqiKO2FIBDrsORBEbcREL9bRjQEgnvE732zGNrjRhIRsXxnFMIX8vzeoovVzQUuA+HoMglfxPs52xmfz7sfaXAIxan//aX/mPLd2jVBYWZBzalB8fM313sv/R21ecTf7+fSN8IsL6DOX8z2P9ceZPa0iTD4V51REYXjtOdfUo0UjjZ+hwq/EjGG3yZ6zAcHNb2fjVUY5jlvEjn7PC6S2jwg5ql0PGYbD2j4SA4xERu6Js+guhtGaVH9ytrL1+qOEe0/PElASeleQfJvJg46ffjOsn7l54BWLbomj98YPu+dtz/32qv9X8NfnTBvpzJd3VpkLGT9TZAUaLfyGE0PidNOVvzebBtF+FcMcKtKbZtLuh98TDTz6wn3DdKPOhT+DWzzu0n+mNf+4/Tranp79UJu/23/hrqY9MtuLaOBr/V6kmJP9Len0TBkNIS2jAEdHzLUuu6nAhz5Rcs//b2w8L6cZPkWkPEHue/FMVrEAaf0ybqA7l6/ArZsONzgr+5oxp6gAW1lDWAC5jdSSkRbp4UnotdbU6jqr6NLS/+lWKC1zAKTwvd8w/eFK1C+zr0s/66Gffe3rv5X9M0Ez+NuZF9LfnZ/MEpxf6Ryz3eqH1Jzr8J83+BqcMxhI="
)
PLAYER_NOTIFICATION_SCAN_SECONDS = 0.05
PLAYER_NOTIFICATION_FRAMES_PER_SEGMENT = 8
PLAYER_NOTIFICATION_FINGERPRINT_SHAPE = (24, 128)
PLAYER_NOTIFICATION_FINGERPRINT_MAX_CHANGE_RATIO = 0.04
PLAYER_NOTIFICATION_FINGERPRINT_MAX_BIT_CHANGES = 12
PLAYER_OCR_INTRA_OP_THREADS = 1
PLAYER_OCR_INTER_OP_THREADS = 1
# Compact four-bit glyph atlas learned per character, never per player name.
# It covers repeated full-width glyph shapes present in the verification set;
# unknown glyphs fall through to the general OCR ensemble.
PLAYER_NAME_GLYPH_LABELS = "百变小樱蕾梅黛丝波寶貝晃哈是我啦怪十一号吱佳"
PLAYER_NAME_GLYPH_SHAPE = (32, 24)
PLAYER_NAME_GLYPH_ATLAS_ZLIB_BASE64 = (
    "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"
)
PLAYER_CONNECTION_ICON_SHAPE = (48, 48)
PLAYER_CONNECTION_ICON_MAP_ZLIB_BASE64 = (
    "eNpjYBipgFFJSQCLsFjP3bsnEjGFz705c+bcG3QJxnXXDRkYhGt/oRnl81KBASjEfH4hqvL1jQxsDeoCghovYRoYQYTkC5EA2wm2"
    "jRaC8wuQ1dsdkg6YP8Husc5E28vI4vMc5yqdn2D3nf0H209k42+xvmJ9P8HuJ/Nbw7NILmJ5xvmL4/8Eu7/K7xv6HRDinI91v3MC"
    "xf8p3d/oF4AQ53qsfVjqTIP2aYWeTXYTkMUZBRgFgYhBUMBuAao4AxZxzsfaR9lA5iiuWWSH5DHWZ9Iw9yxAdg/zS7bfrO8m2P1i"
    "eRe4TwHJX2cNzyndm2D3g+Mn8yvkcIibGBvQP8Huks5j6SfI4jLfWTfpTLBtyA3sQ4kA1ruOjswFGoLB4n8NUCIm9rkAgwAjA2Pd"
    "VdR4ZH2/GWR/zp8AtIi3/rcrxHXNu0UYqSrr3P//b6ZgSVjKFe1GA5reBbEldwbm7lXLEGEhBGMwdZUqZSwXQGgGyygriC8GUpYT"
    "GVB1mCnOACll7EC3kwUSQNoKaBJsDZAEEgAAHpuGlw=="
)
PLAYER_CONNECTION_ICON_COLOR_SHAPE = (32, 32, 3)
PLAYER_CONNECTION_ICON_COLOR_ZLIB_BASE64 = (
    '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'
    'SP9Dzci/ACykw5s='
)
PLAYER_STATUS_LABELS = ("incoming", "arrived", "arrived", "left")
PLAYER_STATUS_TEMPLATE_SHAPE = (4, 24, 80)
PLAYER_STATUS_TEMPLATE_MAP_ZLIB_BASE64 = (
    'eNrll11oU1ccwP83bXObNKWXomLXdU0vss1pTRAfhOkaXFCE2QbfxpiNQ4NOtH0dzqXCXgZq220IAfUqyJgbcqO4zRJsan2ZbJj6'
    'QddRQvx6cGwaTb1p03yc/c+5597kpq3Pg/0pyek5v3vP//ucyLLcIciyLIHjEggjgKIEAJpl9T7I+I94HZgQQubq8SMIDg3EvN/v'
    'Rc7mHyTkPjxzI5fl3M3IVH2ec+/hA2eRwweLkQFQL5W5kheu1t8JxSknJk6GFMZN7svj2tpcmSsCclEY3POzQxvVJFC9H6V+eztq'
    'o1xNKbCA231rdZYc89lnhTQhV3UOjktl/dAORxROBNSbGrgyPWeFZPqpwUEjofKCcrHYk84oHHO3IVeTLkrUL/UG54xdK8Rilyr2'
    'DYpfaWBPnGf+QzuKoZBESXHeql+wpkUTzxRWBUH3C4qXcegxC4d+GS283xY1ufkVscsAbdOPdDuSySfrUD+MW7Yog4NyH2+8LJ+S'
    'OxxkBsBzvWBwf+jvc2YxzD3jHaCiX+AwtQK5Ufcuie8LjLOFC0wrbStBP8PnbsbZfoXuczrn17nXiBaJRJQ/vyUE/QLhIOOcj6Fh'
    'VudOe3H6uDs8rYnD0BoVvo9XcH0TUIcKfhYsguLD6RO++Iea6wWgvR3Uz7COcTUp/DoTaEg/LsIbE2zfEYdWmwOP7pcoNN6n3JEM'
    'GtHW377zWdG27oFjhGBmYP6FAw7ONRx73gKt2pEANRMugKO0g8x1RskUUM7pM97XcH6mxxsu+vSgXYBWUopMr4xGfIwDvm/r5LZs'
    'IbXh7zOzISrBA1JPaS/EN0bpQ5Tbr0RheyLgIV+/TORbxu1Ez6v2Cz/4ATZsNjmVBCBMAp659RllzuETtjFuqDY1TgnbWfopnKPl'
    'CBDOS7YOcLc/EHBOllnNym6bbAh6oJ0mebMsKxNgY1F13QYufcSQIOOW+cM4noCmKxR0zRucWsX14ugvrF0nGbJwK0JcBjk3Gzo4'
    'RrVV5yycKV3Bmli8MNKbARfl4J1SfyVXd9fk6nDLoskJnVIlJ740udpkqjRtcgArueLP2XKZ0/Urc+/GYqOlWCzGorCmmkM7JkMh'
    'rz7pKvAB2Ocjuqhlv6AMcI7c4NzaRfxHSOEtvVscmipI3A4r92PvjNw5L8tCinUfpUPhypp2UFd06X7JsYAiZyeS8+ECTviA6H42'
    'uZYCiP+4qzn0M/OLaHKD2GBSl6s5WyjMODC4euqVPsy1+oi3zB0Y4n42uLpROhALbhgk42UOk5NzpSDjugk9Y2qJV3wkX6zgojon'
    'zqT6PQMQntlxh83vhV4fuEzOFpd0rm6ya7h3wF58tsV4A3Kb1+T0vgqfEp2b3aOsyaUH+u715XkGBzcNg2LHLGdCiLRdyQpxModz'
    'RW/SZ0bqeX0uMmU34zaLZmVtRNui2NQ83ADBn+ABtXWTo4JZl9T8YaxUaIbXH4Jbr1AqqLss5Yyg0UEqqycTyUPTEPOBZNhibss5'
    'Pa8kaCK0VykL6s14H/UVS6W+ooUjL03OQ20zuXJqeqs46/uW4FSLfnkjlkRbyg7q335sL7ody7Uq/ZikKvTrQ66LLMdrRzXnYflc'
    'wTVibel9q4Kzz5h24L6UEwl9oJ148ZgNLuSYfshtIRlg3PpZatBCbqarP6014f3CyEzGmcK4Jn2cR7VsxsJ6kukh9rJfPDT8zB2D'
    'eWtQG6Pw/5CF+fyfk/ahRaed+rXLqHFW/1T8pli58nmbtdYHWYrTFwrG+ZvAW1QkcnJHaQ82ryHYyqf38TS8ax7sANifSiny1Dgi'
    'mTSQo+x7zMpVHqXsspLKSZxL6AuqVT8m0ptmxb+S+44UoiZ3C/U+vTinFr8Bk/udri7BPYZFOBHP+F92lX46HIuNcM63GHcI9Syl'
    '6R/hzUKFxbiDySTeOJJMXsWx7nOvMiks3K3Q7mWbOFdnXCAW4RJk/mKCc+wesr+/mnNEvFY/f0HDrY5VcwoZHwOejHTOnqZJ2Zar'
    '4hwZ+drtyjl2ZmCKDFs5pxdceSOd4/QakoFVODSbvMG5+Q8qnra9ZC4UZz1Oz6WaBOcGoLuS4/l3cj+hF2B75JRRC7lI6l7Fvj1X'
    '2EgSVJqnTrNmanrIl6YPmoHfhVFaafwFefUnRoeWpX8BkYKMxA=='
)

FIXED_TEXT_EDGE_THRESHOLD = 0.88
STAMP_CONTEXT_EDGE_THRESHOLD = 0.85
CODE_OCR_RETRY_SECONDS = 1.0
CODE_PANEL_STABILIZATION_SECONDS = 1.0
CODE_OCR_FAILURE_LOG_INTERVAL_SECONDS = 5.0
CODE_ALLOWED_CHARACTERS = "ABCDEFGHJKLMNPQRSTUVWXY0123456789"
CODE_OCR_WHITELIST = CODE_ALLOWED_CHARACTERS
_CODE_FORBIDDEN_GLYPH_TRANSLATION = str.maketrans(
    {"I": "1", "O": "0", "Z": "2"}
)
_CODE_GLYPH_TEMPLATE_LABELS = "0123456789BCDFGHJKLMNPQRSTVWXY"
_CODE_GLYPH_TEMPLATE_SHAPE = (30, 32, 24)
_CODE_GLYPH_TEMPLATE_MAP_ZLIB_BASE64 = (
    "eNrlmnt0U1W6wPdJ+lKcWTl9QBmctdqAyEPXpQTwybXqEQQFQYg8BC0qAYtFEGcqFAqoSKHlIRdYxtLWUUDpK3dQ"
    "sDRto46UR9PkLh+ltDnNrCUjbZOc/KFCm6Rn3/04r5R0Rtd476x17/6j/fV0n29/+9vf/va39z4AgJu6IDyTA0hR"
    "edqWLrElKAa+yQRgZgUU7UEIf8oCYIv3KmfiuK3hYwBUiFdw1ScGzgHOcfUVzBNqeoz57q8yMA8vDHFW91cGzClz"
    "wpzd+zlt5dYw53S+R/kWnwUen0d5mMeu8M2ID2ZTTmqE/xR/KvVuGG+HdeskDlQ6WyUdhvksim7DejmrV9J5epgr"
    "cn+dhTktL8yZHd2vYx5f1mMCFfBHzHMD5wBY477OoVLYewiA5CcgKT8iEclzupDVhMDXRFxCNYR1Uu+0DPRFfBCK"
    "gqBlk2mWvYGnoszmPCgXq7WW77RaG6V/VR5n2dxKVJ/n+aIdgDKVihgOweL/If6H/QIvWlVeYg3wfBSPLSSGFPDw"
    "a5gr6TyOhnRV1+WDwOIIvYH+O8blb0bPmx9CnFzWdRUYF68ngtDws+yiFYSXuvtYST4YXtFvisV32zTs6DPKPBvJ"
    "kRnJlJEt8ZyVeYz3+wdjcIK7821wI6ceCW7IkHisC8r+Njo3cGKSxDPd4Yfkdla7LksC9bNhRxa4gVOXdX0kP05d"
    "Cw9Lso25TRD5ZNGuordMpgpqwqDYa8l3EBah6CuCQxQ0szajKi1UN5WNe7qCfEujGPhLJjAimfzFGgivZAJ281HO"
    "ZJrC7e1E8Sd3G3krj29D1qA9uru2W7Fh2t5+xba3V6jjMs6m2nZ2Y5vMo/dgmQAwo02mR8ouv0mM2YomsI/aU8vx"
    "XVj5758czIBF5Z5aX5HaVkRh3f28XWYwincqPLJRUDipBir8O83zUa0Kpy7j7XJnJjgiRZxU1rh6LYphUXzmcfRB"
    "JdBxEICp2LdFgUQ/LZPC2LBtb+RVbnhFbuldL/yOMmNqCMIfDLRPq6RYhz3lNUEISnw7/ONrDqm+hpOc9ewi209E"
    "/s2BejDThtcU1KPwQjCj4kf8fMTekAFMrSD1x9e0Y8Y6ME+EXkWakzpa3tuOJ+GMCixHwxbH19g7p2EucpPYPhWz"
    "3XvJgsouR/9Oy1A+KQzBRlpWO64vlP1qpu2aPH0kHWihukm870fF9UZv/Q5EsXFpE22i540o5soDTlLObwCcLWIn"
    "pW4DMNv6XjDhMjkT8VUlGGiZ+zlcoX3evYH0zIA5/CE20MpszChwouI7YEA80IJV8B97COtzFFVdHBSvAlNx/UHM"
    "dlcPh5yfDH5qXqBSlp2yjFeYmajx+Sh2CRpW/f8OgVf4zkClbMXb1oYsRupm4Gl3n9Gy0kxKGd/GKoPV8yZw8l4y"
    "hB5kQ/QS9vmL2VSAyrgkQ+g/OZiZUTwPPR+QsMLcil7wvE/i3cSmsJkz0ZF+AIbkYZ5W2P2CpBPIbbqiBm/nxzKb"
    "bD0HVJu3rVC49j+VblqaPk8h45tlAFA4u5z0BfkYFJrz8FgEry8AyEmu4Wane0NGLUP4uYFqG7BD+Jmc5zihX8pz"
    "4nMF6NkvhfVZkG9R2WlXuahR0iexWOAckpybWwWTre05Mucneuxy/pO6PGKR859FW3rNaG5S+0dOI0Wm89ihBzoO"
    "SZKh8GcDiOKh5qAc08xrRcUHmOmiEj91cwasSjxcHrYoMT/Pp/AzvmOKXZ8JKzym4m/qPLVdkQdNV+A8KXOKo3e7"
    "UsVxSRmpTd4/y1XA7qDMusXwQ3k9xSwvuWkN/ufk6sO9PtndmEeDShV9MVTiR3ypIOuluy/wobpqDCicXt47WVk0"
    "antk3xuT5zksN/RoRfgFucqs2u6HZX7a/gGIxSjnb0VLxHuDGef/vDMYuLQARbl9UOTRBqDveQA2u75HCzCXF7iE"
    "4q1Aotl0bz/iXuLbE2oRV1wliVH6XpXHEe7ervCejg+QyuxjDSh/znXgFWHK1n605i6uuL7QZFpUgdsCd5AJLH5H"
    "Bl9i9NYSG8S6wdCOKM53tz2H199xe8KWInfbfKpD2OLk/0TnxQSnU/DIPW0QhDPSCph4BGXjObLPQ0GeF6gO3yXN"
    "hWGCUNn1JYmZqY8HKvPdve+TvUB52Gyu6TyFffXpxsscsj91YNyXmXtIahBof3VoOwMmHVXy5AzmWTVd4sXGYKAd"
    "xa6lDhStqoNYJlfuO4btn9t4CZgdPXQ/UtvHVjo/Ic6Tstxn4Tsl/W/1W4UL26RcwlkJz0t9TKpyqlzNK3xTE5Tr"
    "M79zQauL7olS5qB9h+sLuidCEclYIX5rwHMHRjNIfoYsGM2ri3mQPJeYoXkRYkBDWuZIe6XqdXZl/oJbe5U5y9wZ"
    "4mQeW6jmb/e5umWcsoXGE1x7UxDnKsljkPUfKe38GGkybR9p932s7VjMAX47ZqyPuIDOEg0rAecvWdHsb8YyVwX9"
    "pwD0vU+yHi+8BuBlbHLGuNHWx8GzdEozcyNFsRhM98VmbR3MzTTArK5FMs9RbhK/R20dIjHQ7T+r6nbplWj2k+WF"
    "bQr2maJYkjNlTYtVaWtCtV3hcSW8wmkvwZg8fHNsnh5Q5cwdqIS+Eyj5t1qttpBFzZn6TEru5Dtt5GkqCsU/AaBl"
    "UmrxH+wgZlv5oOhshThnZr14X4DmDA/YKvgHDs8AM8o3gpAmMUYW7BJ20qCK3m1VAikwCieURIyDCxWXyxeUyAic"
    "HqPKTqU64FtATIZD8FD1ea1MTVv5gqoPB/+osDFYrySMoCug9AW8K+yUwj8HTBDuoGwFRnvwI7pcVqH+u9W9FdtI"
    "bU3zKKYG25ZOL5nluv4MEM2dWNxMl2jPAsJFslQ1QrRWwTM067d2vQPgR0CWZVD4CMyUGdwW3BmTmVJeYbBbUHkj"
    "VHn1ELzoF/JQcrTtNqr6sK0ehWfCnbF4Whev9PEu6DFAj5WUamGlOjc9BpnFFqsBSHMhgPMELQ+V/0DRb5YLysAV"
    "FxD8F1TuORyTYf+vxOE6Ey1ZOK/mafnGpCp5jUOKFuGS/8JDKP8cQNpaVnKTM1n0LlqJjZl4R/Cr6fOL+ZBqn+4d"
    "MZls7MtQVz7NGMTxrV1BvJrx76Cn+BhPQONWD/QNA8fIVNv9MIiHEbIAg8UGZnfkD3SijjbohH4lS0wJHFI5rCBY"
    "pGFhKO4FmvFVjhdYeB7jCHy64tGyVGfsykfQOk7fTVvIYabpD8sWy4xKAVwUeVLhlIENKgunDTKrfSmATKlf6mMJ"
    "BGlBqe/lEKRUSzZpgtRWdOtMbEhdnez1klz4KV03ZFZOQeM2OYdgXsNozqosDMFR9X8OX5TywIRSdX80olqAgaM0"
    "qi5pFPB5CS3B6OOBuvlkTNj0BlVmYrXK8bv5f9gvbZ1odv4ijrYVL3wordFxL/LOHdIORfeoU8vK5sN6wRqLE+zK"
    "2SniIPz/xQFeZZ+89xx9j/NSvrSQjX2Jb7NIZWOZcNasusZJo7OVuorf+TYL4qx4/xImx4cahh30QHG8I2yOwTMa"
    "e95kJV4SvP6kVH9qqe9olsQlwbOTqBzGeET4ROI4ByTBbxAn8MLzrMSJUKTZEwxcOBJ5i+YoJF2T3ANzh8R4AtRl"
    "3Miw8/hL4vEY3HHgtlLPDoPUlm5TkAZYxMyMxo4Fsm66XHjcIDGzRPgoE/0hXMT9Kgi2ZUczvIy3j8mbxLYHJQYv"
    "QmITwmM2d35q0tpQuEA38SP2+Iv4OnpWz4xsglqGsEg5Bv15bN2m4e2/Em/7ZTpE6bP9X8/8BZoB7cxX/d/vVDnA"
    "C5qoqWX9Pkj6EhcMZWs5rgIS+QkQsY1yImGBbHLGC6F5cTb+J8z3e8Lz4msbCM+x+3MSLk8OI7dltj68b5uWk65n"
    "VqOdrt42eSvijCOoMb3DSLgc7cUTW9hntyPe9xkAN3czT7ydfs1Q+Dnhfz8w/pph4ncA3HIV/Paolgv+CpK6DeDl"
    "98Bvz2u55K8g3jlJV5YDfoM5zpWtq86mrC/PiWvJBsPOIGae3R7fMwkknTjyX5iHd2eApMoGpMwDJyf+gCJxrZY7"
    "kPI392JV4x0dKMzfFLbtx9yLODHgJRxGHO+FynO9Q0TdjLORA6pCnFwwhVpG4yif9cNo5s0qy5uTVXjw6hXmeWhQ"
    "2A4ny5zPQk9MBq/BBQqvotsxwia+HkvKxwwaYIbCu8j5GGUdfCcmIxE7Y7Kx2qMweFdU+SnRpDCDXlgls/AOKzN6"
    "aIlmgTJbhRxYYvAuto7EZg0nYz+XrMaimeA0A6WSlpVybB6IYrEX5dKLygTPWQ4OtONZsUyAfWYYPo0X7rusrt4i"
    "2H+Qyt/sEhQuaFIYLK0dgm0wFrNrHEKUzNBxch3GHWl1QhEnl2lreKEnX+I8iPURyT2FCx6bz5KpjRIkfPcOBzry"
    "SXkoA+0jPqU3J2gzAEPHNPn5wX85o1lE9ssD7w3ixCb/G0bj7S4fmj5J3sh6fLE6sA3ojlx9HnknM+WZjkl61090"
    "Ffw3MXux/306FUbY3ioNSQfGuq31Vd1SDsZsvehsk3Pvl1ucp2R+zOl/TjmadmtZSc8H8VD1Vb7bq/I93gbSrv1N"
    "AJ5xVl3H3HUM67CxH+tW3f5gmu3wYqJzcWh+umPDVOcpFCFm2E+UtM7X1/SjKKLbDGHnJF1u5+EFBt0eOHA6AyQ4"
    "4PlJ+iYYRkEprtjV8cIMFwyvIwe19Px/Hb3xwcWzXb1pfKBzkpYVn/84A2hZELpR4r24FPYcQPwt9rdNMPIx2v9+"
    "gndSM6r4NhOkx59ghiNkicm1vbE5dn12laOPkzi5yH3dCH34EwDuqcrWb42Kbu2vgxs5clw9S5QYvbvRHSaXyFjm"
    "TEf4VZlvK/F8yErMLGnyKfxojWeygTJIXuo+kx2DQWqucP51iXVLsH0oM2Oq+HOy/mC1G/WLfq4CHinrtML2V6SL"
    "8D1Qy/jXXqz2euqPCo8owMdvLhj4CoB09A5mfBA/vcu+kDNN4bgHAfsAtMpebN4a5mQuqu1RxjuKm1TOrwgr30CY"
    "noWV8rvGuUH7w5Olv1C7ol/SNb1EEAIbpBtGALbgXebR7BuYYZPv4sVuKcIyt/OwXz5bZriCRmW/wy6tVe8WZ5Wr"
    "949371PvH+/RPNewsaCRlw/ozO6IXb6ryu/qVb9zQHbju0iiI/ItJwEYnos5RE7/NPz37r9y3aEd+Df9jqBvvvot"
    "wzXlTDBt34+G/1HeqnLqkPwD+N/iofRE+sSsn/Kyyslzb2T4KVAPoTVcJZ+9jG6q3xWRQ6zrnacGJsuHgCtTxFel"
    "L4mCWSmCdKa0RMhiGkKUy88Yonh3mIzbbe5XAbMxQjZfY71IxrQBEq1nBdEvvfgGuaKDaLj1PFkslwVQVaa4D/tw"
    "YZ0hmsHtERS5h7vJkqPl1NYDaHHxZNOzW+SBy2nk1ZV9A3RbLpCVS1dyHcSX09MepqDPkFArnfwkRtaP9K4HN3Bc"
    "zaW8y1IA15d3F7ZL20BmWcR1Vh6MCRB+KVtyfHBAuWuMs/VnD+Z+nJzv+z6DPNFwPfqvvgKlxiChsxKzjXBPKc5v"
    "W1sIl6B+JvaWZWPejBRM7NmDeET7HWiJSP94DjLg7M+0POwUAPe+d+tRtJ/chvm+/aOQfYvXJaFxf3rbKPTkyLqE"
    "OsCsyRl1Fq2L6+JPZDAFOSObAVM1T1+aw5Tm6M8B3RmDlpmSnPjzAFStG3YaaBks3RaPfpauuwWJnb0/ATW0e91v"
    "EN/9QTr6uWnd7/8DqdqM1Vuy//crME9EPGv/nZjbJ6IMZcLJtfOiOa6hEXUk8XwvNkgjj34mXgwhU+jK/YjjG4kB"
    "t+In8dSYL2OOq7iKM5Zn8VgxL38Botgp0Etv3b28Xctm9yUSb4bvCXNGRw+9z60Ns2yxEyc6zH2t51kw03YFydGv"
    "jRzGX7/gW/XUwl48Dx7nmzOYlzxkp6bl1Lzwc2MdbVn0Bj/yyl3ub4gDMeOq66s9UgYVh5KNy5K/6Z9Gs1IOfCO8"
    "A+vBjRz3EoSnJt3II1vraqRP4ZiJgeMlwhfSBZdvw6Muejt8b1N7hn4vuTZjHms9l6HbTPScUtG/AvflemY0l3R+"
    "kEGukZHDlOGvYnAj7cDS1EzmctoWn6XSK9nwcbHS7jyg2BMelzXcBKO4U76HFaLY3nWUWEa/LGKpJCc8qPqWMGtx"
    "tZMDvhG1vSzSZzv9LAJ5XXEnbkw3ZwDpNrW2ex4+AsE6T7X1YSafFibPdp6exOSJ35K+t57LZtaKX9JR7lhxV1MP"
    "WWp1yyPb84J/o+Nyf+BQeeBLOi7paHcgj0u6A2URUpbLjPOKyteH6RpOc3Qq4xVvu7BC5bp/gi8OwWeGqK+4D+Jj"
    "GYP5vwGkf25u"
)
_CODE_GLYPH_TEMPLATE_ASPECT_ZLIB_BASE64 = (
    "eNrjlmKQ5Jb6zKvJ/ZlXVlJW8rk0t1SBOLfUc+mTQDxFEsSGwOfSEFkIBIlD1DvIL5VxkK9WrFZ0U9ETDxKTlbyt"
    "CILPpZ9Lb5R7KHRC8KoIRG+H8D6BDuGlOkt18vWrFd8qVCtOlgCpL1UpVfFQk5WcJTZfukD8qkiB+Ea5jXJxCm4q"
    "IDPPmTObnjOHmMYjByIB94s0zQ=="
)
CURSOR_CORRECTION_INTERVAL_SECONDS = 3.0
CURSOR_INVALID_GRACE_SECONDS = 0.75
STAMP_CHECK_INTERVAL_SECONDS = 7.0
STAMP_PAGE_OPEN_DELAY_SECONDS = 2.0
CODE_CHIME_REPEAT_COUNT = 3
_CODE_CHIME_LOCK = threading.Lock()
_CODE_CHIME_WAV_BYTES: bytes | None = None

# SoundReality, ‘Notification Ding Dong’, Pixabay Content License:
# https://pixabay.com/sound-effects/notification-ding-dong-432437/
# Converted to PCM 16-bit, 22050 Hz, mono WAV and zlib-compressed.
_EMBEDDED_CODE_CHIME_WAV_ZLIB_BASE64 = (
    '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'
    '2961f3z1bWYjpeMSnSzKf7AETNuSVZfUFOZkieEQzZJf63/am5qa53fetz5x/GJvFiSIKr1O++R7Y1zMM9xOtzO6F8xJsbq0jJ6bC5W9FhfV5r5MTY4vUPSK'
    '9058lvg5QS+vjjkU5uW3nbOQvKi3NK2qDvo6+8uIyuTGYJONVhDEky1IuZh+VLU39q7/J4YVnNkR+qP4u1ets/JwRWr51VfFb/B3ro+2T5avo2u9m5n62/gA'
    'zvmghbLkHvX5q3JvJ7wJdjImwvH6s50jOvLVT3X/Wnn4NpeZLmXmM+ex0tgb6eeoE/AVsBoro3v5nYhtSV6WJc67kDco65XKV2aL2RNnkF1UPFQUyndJRovv'
    'hOSKermrbeEdZxu210yrHlpzvZ6vfkl+8JPJZ6XvyGL3yEpaGlPmv5ERD4/VYR1X233bdjRtrXn85cPnpq8/Kq9UPau++L208Wf7cOtIJ8p/EpwtPZyyK62H'
    'nPSb5XxpPPfrTUNBHVgX2fj9l1xbYEvF7pNL7E+d+Z5WVyLRAEaAeXgS/Zv3+mhX0pAsad6/eXuzj6cOVPaPexH7S1qjqI7fnjA3YUD8FNn5qPX+vuxGPNts'
    '79rYcb/9Q+dJ8x33ocAB8og0RtbNzIjUd7KeoQt4/+E99GPaVrV0NMsbS6pjKi6+/VzeVD6t/J83s8qDPggqHc23zdkOHt8WPEqqSvSXjw6+w7ADIZ0JTY/q'
    'Pn+315/9qez811SEfCXnOBqdgZ75tHGuq6QRH2yfwaz2XRc7Ob2sMLQvrwTuG150IAtKTkhanFyeOizNktozNTKlIrFU6gwtFjY4zlovds1qDWzO/RnQlmyp'
    '9gj8fcRD5PsUnyTPI2J8Uxh10Hf1780FTXuaYn+cryr9wHrjeV1drnwb+Eb/KrV8YEX590S1HmqhzfNdEzMk6ZDqYtxL3/2u36wNXUh7UxvVfrRrhekWkuGs'
    'pSezv3EHCs95z/aycncyjtCOsDZ7f4ncnPQ4By4c1/t7r489v2RYE/XyqfLLilPKqfEnlddk+2PehR73VjGX4KuNee0jmmbVc+v2NxzrmI7s49aGALEHZFck'
    'WVGR/lXMNmSetrDV2Ew1N/4MbhhX1fczWiH7vPbLiC+/PsNfU2pfNFv0m7E/mUUBgBTL2JF7JF0YZ/L6z6EB+pqkOkFXm5qmH2YbhX2g6h3hrjBPKWMF84Rn'
    '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'
)

# Generated from the requested Pokopia reference images.  Each value is a
# PNG-encoded 240x96 ROI feature, not a path to a runtime image file.
_EMBEDDED_TEMPLATE_PNG_BASE64 = {
    'save_check': (
        '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'
    ),
    'able_access': (
        'iVBORw0KGgoAAAANSUhEUgAAAPAAAABgCAIAAACsUWiGAAAgAElEQVR42u29WZdkx3UeeiLOlJk1Tz1PaEwNAiAIAoJpL/rS1oss+9Ve1uv9J77r/gf9Bt9H'
        '6S4t+UWWtWiTvkskABpokmgBPVSPNU9ZOZwh4n577zgnT2blVN1JCEtGotjMyjp5hogvdnx7Vr/s/N+epzx6qeKNvPeUcr/ym/JPXvWYsy9l6fjhf6u8tNbe'
        '+IOssnbSJe3YWykOUXrYYf0nt2cv1ndd/qsx4x/K0iFm7Hncow0fuMqd9d/P4L1p/quyEwaZJkKr8lTWuDf40J3nzNBZa4obMJUbc59YOhf/U7klRefRHv3F'
        '+0d/BXzf8lBa8QBUblRVR9iroHvoKPfmbDg6+mCO66rxECJEq6nW0FiEYT7MWcwM+aT/ne3DuaKbKWA9Fvfjj1FWCTRG3Ks3ZPBHnsp60yDIuEGkha2q3y0v'
        'Wx1l67nT2mIOrQOz+5PclnqVSZk0hq8G6PJhZRAnjiM9lXq5i1r+qpoaimoWI2V7c2anvU1VwbNnZeeyY9bwueZDeS8hyNTwpTrFs5SjqEaOjx3xia1M3KBI'
        'HnK+74B4LgGtvrUV1BthO3zj9UZL01da9HbwxtVYtKlCWqlZS6BZPpWd0eD8E3oF53/+VxwBJ6etN4lynUOcj7tb69lybUxYJCyIymOsdStO2X9y0/4dXmAz'
        'AfQ5Zmy88uRk8FDWfJ6TFOTNvDKUeOEoxxzsePlU3Hnv9goJLR/MQOtR01KXyTvdjNAzs9X6XaIcM36VYyTIeGlBO6uxHiWY1dA/WVv9sKQcs8PPTE723YDP'
        'd55y2GlEw4TDRBZWUfzSeqSd4czZiadXZw5Ug9vKRCvHNKosGfdmsUzV9xx6AqCnk4gT+II6A2Ix+5xPTtsZCqIh5q3hRg87YoLtOa6kzn8zL886ZoHGmVGO'
        '7xKHVjNS+AaB3y+mB3+dONJKzV58qBEkRNmee+nsg1g127myM3iO760cwwGtbGHnV9PJD+tNMLflhSsMe6LmrVEcUv1G7sKAXwpGO0yYqSqV7bO9qhEQmWTR'
        'sCMn1Q5xm9nKXjEDRUz1PdSrncdOh1Y1UWcerhj3fXP47IzXUgop2fe49oxybPu3ilc1kwbi1eYteHrlZzznMMozPAI+H6uLk1dv1zoJo8wZbqrcRKjS66b6'
        'H/jMAKuB1Wj7rC1jXRDKThSDvYmegWXBFktjIg61mkgUet7pl90JaNcUf9oA7SqG1Nqp9ZiBlcEUXxVhEKJdGJO7ebEFmtmvLte3xS29IqBVZcKnIVR24l+L'
        'NaL6JauqyqaKA52AWwnKKLDet3zdqay1/adlWmJtnwGhuvpHjI4dkBcjvNF9rNraafQ9O1bhc/5qOzkIw5usqNhprJ+TD1H9cBZTvB387pDzFMKA/PmyMGx1'
        'yHrCuDwQkm4gZKVfYM2AAc3cbMc0QwJCiG/oIgCmupXY6ursUwBFHgiilbGFkC6CCCDPy4UhMSOer/z+NVaR047VqPELUxWxC4MLuirimW9MNdjjEWSndFmf'
        'XyK+9ITZAUui6T1CEc1kz8aNqAoZs8wrrHZfpzOU08mxTLzI2UxQRW0FFQ7zM3igwBvgSjN4+QXg9FmmUUDZ0I/Xi+QS6NqCttIQqNzTRql+GqKK5cHDp8QV'
        'OFHZGR4q1SdKKkFZ7pQD8nhKjdl4eqw0tGoWQSrsvJyxziyoLTdsY3OlSho61iVFs+IrYInuKrc2Y1iLddIYL6Op5P2EQiydPJI3vog8AoSRzRnH2FcF9Gw9'
        'u7anzdmCLttSDip5TuVWrdeTyOCMWkiXNUK7rS4kseqJec3vtUO2HSqnSkXTurGz/qglq6p+TTte+E1BFQgHEyjZZH48hQ1I2UGy8GpAtgPRupZQZka4gc7E'
        '7LKEJtXJY0B7OUsrw+8hrRnQ9GdLH/TxZFUSahEoaiYSerZGyYoab3sxPvIxWAT90fD+Y2xPadAF7eb1qnwaIKDQ+CLRrFhIrPaKYwqNc5TNoGTSjGYbjCEe'
        'tupSGbeP24nx0BO1TCtzOQmIk7VPO6sdtY/T275/BgBtR+gWvWBpHj5Gs+JNWOdsHoB4zt2eLEAnrBu3ipQLsC5mwc4W0DNi0b2QRVtMD8tm9zz0Kyu8OMCX'
        'mFVFmA4YsgBxADSrPFR5hE9I4XBhxPSeRYnjM1a4ihozW4x+E4xastarBGz0McNhnMWYmWxf3kwox2z2VTtI8HvM2JzRs70R+PYElkyTDW+0VngEkWYF6iJo'
        'zpXOSWyrzLMpf5hZIiR8WGU1zRDQswqd6IX0KLcz2oKXFruMM9YBcCGB2AuViVia8g/QnNVV1vBAP6ySsQWOxeyDX0FLZBi19qeYLTNoFBuRC6PsKBKuZuWW'
        '85SdhVtl8m5wnjMV24KtgrtqBrXeBDu09vJyhiWyXngiODEEkQA61UHmafx0Pd2xKvV0YlWbBDkpS5m4k619ZQ49U9VCyfIUy5Pq4ViXmPZ68ZyKyIAJFWE6'
        'MnkEmNoM8A1UHvvJXJCsKOOLVFbyb+6MRCYvPTV67P3gC5ABXW+YiWgUHx0CFydD1ER4sN4z3swxC+/3zCJcdMXU0MvSsf32IqXGsyRI28Dm2mksSngj6zng'
        'HBDVmvm06tqIQaxPvaDp+R3ltzwf45FAYNMxs8Bi4Ki823JHhmuqguHYkep7Ty/S/VOv+rYmHhlL1JZWr4nxo2yo89imsZf7NgtMFno5/ZvkYnm2itUoF0hN'
        'Ql752kmDPDfjvTxMgfSA6bkKVve2AsRhqhB9w+QT7ccTFD52z9jJDpFJNFuSamaCalXatNUZ7beXszXuWoQNLT4yoWymyCz1e5qyYmhhZjE+eI/fdOQFsfJr'
        'kM3KT5VOWPQA2R3HS9k2UqGpbp745H7FWlLQWgZWYMv5VV4fwbPjgK368nH6knygBChnfSy8IV6f8Q4Ppy2IREhE2dR0FhNXBsFIanhglYY6q1nDCjEl0tJK'
        '9/2AEh5BL/yAtUhd5O2y84kP0qovXY95OqueKgcr6U8RtI4N9RYzkfrSz8CplbrwZ5UCXOXZZCBqpadQ+KZ0W38LUlpVbPEDlnQ77QKzVUdKkR7EEKMJyCW3'
        'Fq/A2NxmPhFGUGgd+UHNj+a0bxRZQiC5wUMgvC0hm419ziYGluIVFm2Csl+Yhp39yhYuZ1w2sEp5Z1TWM558VUVAaRbuH5HCRO6EUMkter49y5ocLWcTagAa'
        'uhqwCxwD0CkBmhTBLAbEaX2DWgUMZ63CINSEZh0EgdgydWmjF7rmeHqxWzprqiJAs/3PVlOp5TClXD6L7BnFqNP2oX1cpZS1LLz44Pz7KORx9psK7kntES0x'
        'ywxjmkY4TbM8z7wcAPAJ6YRjQ1METJNsblua7q7nzNWGEcc2E1t62ZQtAoUE0xVd0njnySlUoyWH6rdxhMx68Q/ba5ypnLch4jd4klCbhs4bJJiT2HQAaCA7'
        'JPphfcjgwCcBTG/jQNN7QNkX0JYqoCAPryiqsXg1SZLk/JLjAf0ojpSPX/TRyaEaZMmFC7KIJGnUG1mG4c5xHmvyHGppGIqw9lzmvlXfZ2JNMGeTki6bJ8sd'
        '2qwwdcY4/2MYGp6iJElDzFhuummSkAVb6yCMFEnXzBpQDtCPhPQum1pi+fh+TmUCtPiTdHUbrshNphwzN9oBYxDAbDHrKEXEiElIzrgPLFHnSGcLtrNo0th2'
        'a6YLQOOhVRREQGIUqSgk1Qo8Q0exdgygL+6UhGkhLNvtlhwQx7FwBUt4pIE7OTkxgLrNF5cWxzguRDAfHR5hH8AymJ+bl08I2fzy+YVbyrP8e+COlney0cn2'
        'qQnR2nnBSNdnJhOyToS5SbMky9M0bSdpyxh6SyI4UyqEajnv+QlTagivFAo9/s5WwCwgxZF1zTK2ydmvemEU0+cU2jFRlv3WH7o1Unad9ZEEmybSoyGbbRap'
        'HKQiyhLfpNqmWuVkeAZgwiiKAh1BVYCmqtllSHAkcqEGw8FYO2R3SxRGTBYMhqc0VghcGeIuJMbIQcbw/2yZf+A77VItzC0afnVaXbE6xWHsY0PAoky6STfF'
        'xokTfo/ccTu4MD22PZMXyjiViaSsG3KfFR7sqz7EcRh6QQrNJCV8Z908g4qvAq+mmFz6AI5OWJnJCNm0qQLyvuqp+JYdNF4llO0cElpNqaxYnYv7V7NF3bkA'
        'yUvoY40pAjQ0v8h0AWhf5YHvRb4fhpFfi6Mg8MLI8upk2Zg7D3lJZ1VPZbWFFYPxaQqvowO0of+YJOAFFkc0DqIgS3NGbaH4Yf2ErABCLcmE1QQ6EBpiUsM2'
        'KHoIX01U9r5HtKp6YXgP5TemKNFD5CGXuQQRDDwV+Jh6ADr006BjbYqNNQvTriU0GzYDkEzH2KeKzHtJ6bcrzHEDlvLzAdqeraTVrxcW6iRFyZmCpLOIJGuG'
        'BlEGmr00Yuoc2AT4xuP4YRDhp1YLIRB9H/t7blgfMC7SUkKVykvpAQWcVAsmEAFrcjgF1jukdbfbbbVaZGjD0ALFDGR8zoyFAC1EAruCz18Ega7Tq7aw4EQ1'
        'ziC4Bw/BgiPzjcm+x+1IOKu++DMyTZLloYyqYWsd0OkZsV2SuNZ+pOPcD/COrFUpMeUsDbDdekT0OOaJqoxhz9RsvTCVema28N2VRhZaSy+hFI7TDdmdl8v9'
        'kzPPhuz2iz02Ldu0ppKazSOb+pqNy9D6ajG2HgW27LMORrLZhIbZBh5YeWddR31xwKImkk0oCOQNqPPR0eH+/v6TJ0+6XeILrdOWKHZl4TldvMIoZK4HUR1t'
        'bGysrq7evn0b6qCQZpkoeu/r7wE9DhtEKtg0oVycnRL8KVuJt6ctETjPc7dJ4t+ADLIa2ouJa6TIh7Wj5onNuyCdWUpOYzJ26dDj0AirElagTJXhDmSiTcuh'
        'x+r4/RwaO7xOOaSwAewaG6ls3kvqCpjuRiYJyWqT6kZYD/0I0IkBZd/zA5Mrkqc51FwbkD1Gy6J31604Zvt0xOXlpU63e3x8vPl4s9lsnhyf7Oxsn562Tk9P'
        '0zQB3EEhIJhFDAO1zh7HFA8Qx2FCT2q1GlbCo0ePvrr31fLS8uLi4htvvLG8vNxoNMCgj0+O8O/C0sL34B3xKiPADJu2hF04L6PzczjzB2/iuZPiubgRoA1C'
        'iGCvjvxOJ8zJ0HEKykcTY0BSAzaAUPypM0b3on4HrfGB7St9oobYoEvI2r5Ah6J2YZlM1VsxPhlcPG20Nr4GSzYAcQghbfOA5DQ5t1UY1KIoxl4OzoGVi69l'
        'eZbLTkXEnxSDHOi2/UGbLMN1aXz21PbWTvO0eXR49OTpkxbj+OT4OM3oZPV6oxYDqLWVlRVIAgjaeqPuCAY0kQQ4T49PjrOU/tvd3RWU41TNkybGF/jeuLCB'
        '717YuFCr4WSN3L66lcN+R0Xsq56AYusYuTmTAvyXsyHZVqJNJc5OiaOsTNFgFsFfYV85QJ0B5ibN8hqzySwP6pqENfYAnx0Lnu6FUKlKkJ5waNUfXmarh/VJ'
        '4EJUlk5SW2QKlUK0VD1ZQ8sZ0DYEoG0emiwwqSYLnUfoAm1m06OWKCOOZLNlqrdjL2yZwBtHJ4zFZsVoJlCaLE+S7Ouvvz5tAsbNg4NDsN6k28V2twRZurBw'
        '7eq1OI6jOF5aWhQKEcWR2JNyrBV+gWfL+729PYh54HvrxRZ+TdLk+YvnR8dHc3Nz7Vvt9QvrC4sLtDMSGbe8Y7K+rp0JyZQLb1JoaE4m8+5cYw7rEitJKt5i'
        'mWF9sRNVlS4e3LM4qdgNSocZIxLKFsZMZYukEEV34l70hCQOcjHWk2pSlMF13zIujQITIR4QHCMaM7RkSAM8YfEt8n1KbGiluLKqGDxtQfnYkwWJq8UxIMkZ'
        'NESyGUrskfO+KRfd6LR/ST/kk0KNwUh0kjzrJsQ7sTAyy6ohbiVg1dIURg9Hz3v+bUuA7mOnfcU0CupzJpCyV+7rTEYDh1oYMWuQZUMwnWcBVMA8UT7QDFzF'
        '9SCMocxif8h6UBCjuRYt2eRYEKRgQtYC0LRUTcYOacIELthqtQ8PDr75+huAMkkShrlfi+sgDBcuXrx69cqdt+5o0izUNGFpGPRHm4+ePnmKyzFiTafT2dnZ'
        '2dra6rQ7N1o3Lly8cPXaFUt2UFOOjq+0rRjHi6SMcTEYltXNpaUlPCze0Eojnd+33S6BWPtydeKXvIxFerFd3oda66bPGXSUi9qkL4pzjlGrSZEFoCUWSvsM'
        'KT6vXM4oI/eriaLS44RhgDXMmjHwRB4m8FnyVJM7w+d1ZNzWWHFRFfWiDf8Q9i3Uf9Z8jMupIWmM2y4CXbRy6TDmjIHEuZPr9RgLGepKtx07AmMYS/SVSFIJ'
        'LNREe8acbIdYOYYamAdSr/uVwGGBAWziYi1BIuOgBxhaumwVD6D9YbAwQUWxAzsstICuFddiIBVbP6YH9AAU+datW3hzsH9w2mxBlD64/6Db6bbb7aOjo9WV'
        '1StXrrz55ptvvfVWFEQk3s7j5cJJINFfu/XaT3/6U9DoBw8efPbpZywW9YsXLyCqFx4tfKI+mZ+fqxH/qItb0SFSaxCabIqkVTyYUCBxbUJC59A4NIRjJmgG'
        'A8PNO82VZaRh5yVAB70UD4uhAI/C5xiHTrejSh8GwcgjNZpYqsrTPEsycpeGEVEpCAO644zsYNAq/CAn3SzHCRkimKZQs+KFBReQEPDjOGSfKQsY+gO5cAXK'
        'hW6tSkWczUodPH2hT9ORvEWQf0q0cPxJSOXYoBAsDChgWMw+6GhOPNrHXbMYIZccE19aWoXruwJQegrl/5//6V/1m3jVqIikIoRwSMxKtWBwEawEXlzzTESk'
        'OQnyrm8zCrSPw7koBB4aWoZeDQa8VwMVMUzk8lhcPG0Sqb106RKIAfCddJNf/epXjx8/3t/fxyfzc/Nvv/X2H//xH79z553r129gvj0eTfeUerLHWjlZQmsA'
        'KIGMv3HjxuUrl0WI4kMiIbhqSvsAcLa4tCg7PShO+V3x2ExKi4I+ZNqtNn4AqPnGPDABIEE41mt1DllRHbpOAqiRxsR0kjefAH+l9aOhSkd4A6GwOL8Qx1BF'
        'IvyJJa44M9hXzzYi/JskqSgbrAvTJiYmHYoOgOj16BpYT/g1z8l5hN8ssQULqUMkhNQY48JHyaZGt4bp6HY6Gc1OEfOmNbu96cyOFAs5ZPSL0R+bgBkfGinC'
        '20i4hOZVg28kWPuUBIC9O0h5O4K6mZRY60Ui81IJ+teHGp02p4YUcxmqWxDpwCT5pHEa3gcxRJw3ydsmWZ3DMDammu893AYIXc23PqCMBYBlj4GHHIV4hmx+'
        '/uwZJhuTApl9/dr1N996E3Bn2mfIuKF8t1+zIIzicLKWTjNK9iSAQ571+vXrINA4/6effnpwcHB62trc3MTwYEPG55geTFJA0X+ag/6sN132Ky4EAg0mg0lu'
        '1BrAceZenNkBIeeHZPRhygE4Qj8AYo1P4PbYZkW3mdlWt3VydExmcoxmFDmHsHXEGi/cYcbuDRLhBGRmyTmJatt2LicvV4Efh0QEoV1gzRgKrpC4bkhyoox4'
        'RujQpDmw9TIkZ25YLyScEv8XNgDcBi6Di4IB4njcalSjyAUR57hv2YUmaMxKTH44j89+WSgzIegPzXXuU0gFRd4FZdRoP31Q3pCcQjWC+g3PlFTDPhGTJGHa'
        'WG3B0TLRd2nbCbCp+hHGhTYz0YIH628M7gw5k2kMKab26Oh4e3v7wYOHzeYJUL68vPzBBx9cvHjx8uXLEmsB+MZhLNuwN2XdXtk0s1TuokQ2YNeoNzbWN7An'
        '3L9//8XWi/2DgyiK8df5+fmV5eV6o4FBF6WGzU/TmQssPdHmo00svI2NDWwpZLsMYyIf/JprNMhGi5Ez+eHB4TfffAM+A5HHiCfjD5ZxSjyXXgA9CAxZ0NdW'
        'RXck2ZaLa5QYiI59t9H7LmrS4Y/MPkBgJPGKeWoDooIq67I+l+MTj2L2KXqCIteZ93LgLkHcWXsJ5uQXNrTJpNBwSKdkJZsGs6ZqPc2V53HS9iURHzgP7hxL'
        'I8oh0XScpR26EqmqIUd3cKqeMkMF65SeQutIR5mCZ4cS6BLTnEbF/0LxMIxmngssYvyIVDMjSj+VCX6kgwsuscOm/Nre3oKidnh4gDMsLixCiL7//vvgGBgp'
        'UAXaoCFiaddzg4gJjTBnUxjLIFfAbuMotrJhsiqG0+LDD3/0YUA2/+Du7+6CrIvmhKnB7c3Nz+USo4fpVHpiGjawgyW3v7v/6a8/PT46XlldAWUHHBcWFkg/'
        'I3XXlHUy2s02cP/zn/8c9CPpdtudNj2aFQqrMOFYTritW7duvn3nbaxtHbCrVGlR4XCfWCrYDZ/uPJWlgmeR7AIMf0hsF3uRZqONxc3MzTVAGLARZWwtjZtt'
        'Q2AlfZSlL+RFZuygoqA5ckCH6qh5BORH0RWX+8wyRdDsmHfhepkgW9iMTSyelhxkX+xRuBtWTmbzgBJPfd8Z7lzsfx87CGy/HB6lHVZqA9pKNGp1n+1VYpEY'
        'fM5pJWsc9p+A0QxJRBJaB9aoXi0lNbQEmUvJwAw1GnPEmxOyG5ORjrbgaG1t7fZrr//4wx9DWsiQEXWOXbqQWOJ8zYPqTw7CwNcXFxc7nQ6Uv5XlFTGqlHcB'
        'KHz00Y/ffe8HJ81jABqbA1RGCS5dWloujSTa15NT4qzXaXUePXz04vkLbDX37t0DkqDL3r59+86dOxD8eDRoBdglcD/doJt20xfPXmA943OMAxaSaKKa/fa4'
        'FBRWLIbWaRsDiwfHUwN94q4HpponoLvNv/iLv9jZ3cFpl5aWZNW12q2cTfWdTpfpU3j/m/sXL13EqsBfMc64Cq4oVBgLSUgwfcILossGGWHMtJngk6xz5907'
        'r712a2lxUVaRAJriZ9IsiiMOAFayQsZKaMOuQewmhg07QRjVgqROZm0DPRdziVVbUA5JXvLK6jVk8wlsv4Ct1NAYVg1RVeOCzpo9bFE+R1M0ieS0GhGUeCB6'
        'Uk2hP5osnaUwdrekhibKY/M6PobwmMO/X3zxBXEp1nU+/uhjkOalpUXO/qHd2e/lpbBg1qSdQL3qtjoLi5M9fBh90FmsipzSFd1GKRMAnNGHSoPegHtsbW+D'
        'UoME45nW1tZptth77rTPSZsBkPrRhx8tLiw9f/4c6xOIPNi/e++re0eHR+Drd96+EwWEm1YTmAPX0lAccfDrr7/+zz75pN6oi6Me0u7g4HB7Z+uXv/yfCwuL'
        'Mb8AxKSbao8Te3hNzs8v4IfUNeWDLL3//g9XV1aB2l7vOWMwsEdHhxjC119/49q1q+vrG2I2Jgqe5aDU4Fp4akhubIaXLl7CplRpXWefPn22tb119/d3D/b3'
        'FxcXAG7ZVyVmhoKfeehc0NhkImDYKpexMgpaDxqEXahhoB3mHXBPVj85ctMzrjhcD3xseOkv/mhtv7WuEm+iVP+RZ+hIiWmjKa2Qt07DnhQbaC/GdkSUg56M'
        'WZr4UyRvrG9N9c4c6ECW+MHx8faLbUg1DOvC/MLlS5euXrkKyeSQR0QuL3Z8y3Gkqgw/mjLmUxUWel2UPKFPtMu9wTCCjWBPaDZPW632ztY2hB8g2Do9pStB'
        'UEmurq2EhY++EO70+o1ri8uLcwtzjXt1iOqnT5/+5ovfHDePwduXV5ZBb+jqiWS4G1COMAqu3bhG9sEggFwE1OJazSsYKlAjYe9iIpB9CuMgzSDbkMc5MSKM'
        '28WLl9bX19kExMFsQfDs6VOyfvghFsb6+gWsHFX4ohNIg3bb0D6wnef22tXrN2/eXFxa6nY6Us4EoMecYqb1vd9LZBhGWyweuHpIuc+ei4tmS9EkvqHYOBlx'
        'H0actQt8Yx8KAmilKiMDQ8CGZr8SA2c4aQXHu1yswA5KYOsNyuyKD7GU4cUvzDtstaOhEmO4zSj8KKcgO9+CBmEfDFmVlkjm3NWpO1OzspKrK9mtIGbBsyfP'
        'sPO2Tlp4zOWl5Xd/8B5GPwojyZAY6D8kRnslAVpait9NqSg4+KqKHVIQw6qQXl/duHLpStJJnj55im33+PBob3cfko9nUbmcDetJMDdr9sYV6ytqfFIiERP0'
        'sBauBMtxLcLPkydP2t327t4u8N1J2lYt8eogNc4o7OZdgqm2cZ30Uc7BwT+AdQydtdsFE8uKTUZxiBolnnH8TyY+3W7SwRnwV+gJoFUgb+12R2zXqfabzVaz'
        '2QYJTJO8005ara649Op1qC4pPjk97ZAl2PppajqdNAw7bUI5xzj4frvdxRfJBsgvDIZl4x+uIjRD7PRiqw6ZDo3ZJqlWrQ2Kuegwd9JUe4n2ccrcsyaR8FHr'
        'lcWZRLqGnHpsBgA9yNlVfz6CyHBXNMoW/EQp2xcrQqYzSzHEnKqSAQk1m0BPB32OOBuYlqHx8kLuqwFdUGQl1grZNxSJxhfPtna399invXj9yvUP3v8AxA7g'
        'Blnruc2KkcKW55DtScJPXvNr02RPOeZfwbRsvo5SW9oxrl8lO/fDBw9B5bHpP3n8BGw+XA4yupQzS2klqPISVlIJmWRfJh8ojsKZIReBCaztlbXl+cW5tY21'
        'IPJ/e/e3a+trN27eoNCSPGWHHxWY6ySd9dV1SG5LcdlebrNWN4vDuuLtFwA15K8G1jMgp9GoU+pjliUZ0BOU/jNmtOrzzz8HU4IOenR0AvpHZsFOcnhweHx8'
        'Aoq4t3uAJ9je2u0mCc65srIsUbhPnjw9OTnGSPz+9/eePXtR7njALaCNQcCKYMNLxkFgOmETJFQe6KBVIz0ZXmwwFtAMDePTk6rUqC7phRpUCusnA4tiQPtc'
        'OMBVrqHiYtZ39TC4MsjLp2CpUTkLY4wf543gUmR7bp408a9UCLhw4QI0G1qPxVrPWRhBwWE1iJNQKqVn6H3ovMcTOHQRwC8psUzfyatRBqbKq16rgfNgqiS6'
        'ent7CzoiqDwALU4yctFnjGzsney/wOfGipOEFjdwGJDTjc3YQQTAgwZALzBXDagAACAASURBVPzZz/6VJU9NIrk2oNq/+MUvnzx5hlt44803b9y8BTZ8eHgk'
        'foqcKrNQjs9cY47dqBC67YAIgG632mAC9XqdMhvIZ5LiVCcnzTRJHz169OzZcwwdmJsAmtJwOCKcDO2tk/hFvLq2BsUXVAqIBKmLohC/yrMfHx+UcMR3JXGT'
        'LmEySPb5JbI2QgfFBOGL4NPsjYLIbzEjD3H/kzyFI13QZ13JVVjZ6c12Y+JKR/zJDv/lpYqrQAJhS4VQgW4nnmFMPCSiWIt7kc2UCOsL8RAlvQgDyiW6CbrU'
        '9Pmtuog3UByBTgwVao11mTeYQmDl0uVLxBM6bczZ/v4+PllcXnJG4iDodm1pVBYx3+l2MjJ6ZSHxrlDHgCBlEAN+EouVsZMIohmcdXVlGZRgc/Pu3//93+/s'
        '7kCHe/PNt65cueqTCZ9eHEPsQ3bi0rgH/Pv48eP5ebKBxLUYOh6+sra2Kt5mrL2f/exnAcca8w+AGBSbkMrI8dfFUwBtUmn7pHly//43W1vbuN8777xz+fIV'
        'rC6fjKe2Q9SZHQ1abNrgAz5nytlu3gliH0p1o9HA0sJuhVmgL/r+8vIKjwNRjgmzYIfBZuTfhn/w8hkrw/9kR0jxl4tPZM8zARpyi0PDLl4k0xLGqwQ0eYbZ'
        'm4AdT3O+dy88SEx/2nqTu3yyosphjCqsmCFJbmu3UDwXPIb9HbrR7s7OXpZB2mHLxsytrq+5IIc8L2LHnDQjix7Z3rGDgFnEwCLIXiH1oVcR6ex0uxJqB9kG'
        'AgCA/vrTTx88eAhKeuvWa5cvX56fm++0O2zHJHsKiDNubQ6vxhwQubm5CRxTZGEt7nY7r732GpC7tLx0cnJCJvwktaEjdZC77IMMeblZCYvzKOKiS6HHWba9'
        'tbW/t4+zQUUBo2idNpnMEH/EeTj2loL0gU73XFCRPC81STcjxOO1yJY74WkcDhWwqkoyqMw2eXVCMOpEM8n6nqLts32pkGBLkRXbO9tJN5Eg0tXVVUwi7d2F'
        '0b7cwrqdruzvJavWnqutYaaoVORSaK2EhhJxwvxx1LlfeAEdvcZkXbt69e6XdykTIQiOT05CtrMSSaRscw6O48yhsv0XzS5rH+x5zqFFaY5jyfPEuSE8D/yG'
        '8hzi8O7d337zzf27X/42CMO11bV3331vfW2d4NXpKKmyZT3IctwbNqtLly4dnxwDePv7e8AWKxUWq+LG9evLauXw8PDZc+jTW7IPGKqMQU5G8A1JG4k40x03'
        'z+GKtPaeP3++t78PnrC8vPTN/fvb29shG5tFrZUEH9LXOh3RcAI2OSd5V4fkE19eWb5z5w7kNOgWVqAMXUZORDJa53Yy5Tg/5O05AK3Ok4LllRx6qBtRnVta'
        'A10YaHA4iGFgAhIII4UxJYEdSsimKQku2BsfE4swIHrHtiqt9TR1lIEPbYowS5ZbR8dHEDbkCc8k21f5kiyraRuFlhbFZOrG1GL6K/HBRktUGpQzDpAScg/h'
        'd3R0uLdzkCV5oKOS41KI0vzctWvXcB2cChz3L/7y/8WzrK6svn3n7ddv3/7ggw8AUJwKWzlukiPH6anpwa167fZrOK1knZFOS14kAHoeuz/ghRt7/uz506fP'
        'DAd4AsoimGu1urgk5xsNDmg12zs74tYBgk9bLb6xBeyN+BWDIPKYjHQ+hYbyoiC1NSIXCoW+EKAjTdSOLS14OnxO3IMLP3C+8hTVrNVA14RxNWAqHFrZWUvo'
        'P9SLVJoklRBNvBqNOZEUfpFHiElqd9sSMPY3f/M35ADLc8ogZCFNJITDcqZRRKQKhyBbXpjgjz/++M0336QIMq7ag8toNt6IfwTk58WLFysReRaBRVYKKUBC'
        'IiaERsssdjptaF2/++r3X/7m7t7uQfPoNI5r4lmElvnOO+/8x//4Z/PzdTwjzvMf/v1/ILqp1Y0bN5YWF6FjtTsdw4EQWeY04MX5RbK+5/m/+Of/AtcEdjkH'
        'hIbktNkEmsGnIQhu376N+4cKSIkOUUz1XUISysAjxCf0E6IEXCcKgrZRr+OE//k//z+Q6zjVn/zJn2AhYT/c2NhoMcTZ/p3JPhbXahw0kuMsJIZ9owKye4DT'
        'Q+hgQZIZR2siY6ur+DqeCP+yKvyHLdYzvVI4Juu73xU/inLYc28qolSJVJB1LxFeZJhTrmwBxCTkN1CLEZRxByY4gMenYEXe0CUwd7zBjlR+DsGkVZSlqq1E'
        'SRdrhpbiKa4zDtnFZF3Z/hZebG3NyoBVqWBGaTK+xqmh0SbttH2tG6iQBfbRgwcPKA8NCMvSLlgotmZfX7hwQe4WaAAEISOBYp+emlx3sgu0NJkOOHYlz7sZ'
        'eHPhUvFW11Yk8taQpqHFJeRzONHjzU1cjiKklYRg0L6ZpxLF393bBXU+/t3vfo8v1hsNQDdJMlzq8PAbYBjHYp2A+9FoB77sMNgZsjQNovDy9YuGwoasOFZE'
        'Lgh2Jf1CjEXTKIV2Ou5aQdMg5bDDUgn7HIbVbiRlPEcRatrXDc71pFKDwFaq4kVRVZDbai7XwM2KWwKwEC0Qg0WeVGtcNrgUTGePN7FAThIUzkDNLMR3XdgZ'
        'JhJox7wpR6ZMs8p75hTJ99TiwbISvQ5JKQEPFDfi+2VJEKLQrtoe7bZ0Vxb7dQikzsXzJrehHwEZjzYf3bv3Fdc986B+kZT1NSTr559+dnp66pGvjsImAVbm'
        'uWSiAL49yfxi7xRF5FKhrVwCEiGFgbl/9pNPGFLWp6QoRdk9ivYf3OSnn31e8nVhC3nGAamSgpVmkK/NkxOPwzPu3v1SMuqxvUj8Ke6t3e5wiB/eN3CJvb19'
        'fLi2trq6saiowAkru6yCl4DGBitxdmS+NMabooCr9c5mBU6rwgUTdUl15nNbocpn/OaFu0X1t2SrVgy3Z0JTXWDTqCDVHjjkJSCzRT0DqaEh2a+yy4shL6e0'
        'zcEOtiN3A9BxDlcrMydrtZpciFQ64/Le2BJI6JT4J4rOM8R43C25pD03L7K6KA6Y61SCPKyvbJCqaiS2u5WS6DLcziD3tUQPKyAJejCtJzL1Qfp2l5cX2VgX'
        'OAs3kR+3+YRRwLZmKs0DhWxtfe39D96jiH+ALhY7A7Yn002pSgnOLOZEIF/SAqh0otRoY4sebv3ylcsyiU+fPZFAwoxDSrDYIFAE0FL+Ac/1aHNzeWnp2o1r'
        '73/4bm2+FtCSdDq0y/WyRu55oJ7bOcwMRZcsVU0gcTFtlUZyBVSCoqGgp840wB7ePk5VyhdVIp2kfkgZxDDYYLBSgbTMUikaVdiynZXqb1FMFUiFltUbmNzd'
        '3V1VJF2KYKbgIU3ZU3t7e5B/lMCyv1emCUl0mFiXJvN1ds0EZEV2SShra+tAw+Hhwfr6htwTNYbS0sHIE3bOQSm+LXIKpZykdf0JiiB631V4t5zxQckHSixZ'
        'VAENsMb7jY21dquNS4Mv3b59a2lp4ejw6NmzZxgeUARwFQhUqd+HY8BQl5eWyUKY5sA5vgTl4ev7/1Cfq+PXMA6jWowlRw+i2ZYO6WjDmqGQa7BrqCL/7k//'
        'najLi4sLkko4zQ4GgSF7ZsBcf3dv72//9m+xmXS7bZcHRvEkVlxURPE91old8LQ3uTKgcg37VF8vp8Ea+UUUhgyx9coeLfzXvrocr94y6Ky5105FnYdaRjzI'
        'EOgyHCWTiOlN3Ap9zmrPg+4CBW59Y130FSm+IeLBcNXGaWRD6TCHtiTxk0BPg0P4nVGPku1cmAHei3EDp4+DiMpJ+C7deLypVWwUmUll4mVflqQs0X3X1tY+'
        '/uhjnAra2J//+Z+DQOPZ/+2f/tv1dQrrw2Hi2TlttiRiBU927x/uffb5ZxcvXXrnB3c+/PGHrESSDkp3ztNdi2pZQomvB/v7amVtYWGRIpB4Dz06Plpu1MXW'
        'Oc4ExIlVx0dHIh1wCfH/Ub4j2bmTiJJEZ6btjXMa2IFQ5UGqGky2xE1yqwzbLOzI38d+d6C5JYRfHFNKKcSVZBZLpH9pbC4y+ej0YAg9V59ypWQM5aJRqYdp'
        'HIRu8oKwVwygMPnRtZQvNcQMh1w2T5sdsFsf8GrMzc+fx1lEt23Z/0L6AK9RLbYGLVJzEaMGOH7yR588fPTw63/4GqIan0BON+YauD0gO+lwmlkYPX36ZGtr'
        '68mTx8uryxcuXLx27ZqEr7C3OTIcvUkmNiIVPu6zmyTPnz//q7/6K0nRpQQTSR0a++I2CdgrYnbGeOw4JG3v8PCQc8AiyhK336rFTI1wHgYvhdqxCQdnUwX6'
        'dcExXz5D4UlrvnjxYuu0JVQMIwgBLBG94iAsBbC4Knplllw9aZeVNI1jRUwW8pWSt4gxWz6RFFRsncCQ3BLghV17eXmJuYc6B6gLPbkof+auSyoXZSvSPb/1'
        '1lv4FYD+zf/6zelpExsR5Df5RDk+ltlt+7e/++3m483T09aHH3146dIlqiLSabHNOOyFpkieA9sUOTaabelM+jG8HLIyscQZsYUojjlA3BVsENOQFCexf/AC'
        'OnbU739oO7R6FRfQwIqiuidxdOXyladPn0pgxtb2FtSREtCDju5KgjrTiHPscYVlg6utiRuS8wbwkk+EToCyA0+4H8gnocjYf1dWVsnOYAsH5aSLleVPHdck'
        'YmSEeoqTn4Vivrq6it2pVq/d++oebgQCGMKbzIs5LZ5m8/Rg/+CzTz9NOPzo3XffXVtb5SSUDoYI+xX4uoSzilLLESMZCPTC/OLHH3/EbT1kGZuJc1NG6Ocs'
        'VnyOFoRK8/TZ09PC6v+Hkb6TcWUHI1ArR6uXwekgrR95tJq4DM6OCpkaLl2+xHFh5DUEkqD/ldFeMvHychaQ/t2zTBAc/yNW0oi0qVrR9cITcwoVrQ4cxOU2'
        'MJePHj4iazcLvUUC9Iown2nqchh2f4pIlrIyFMqTpVIPUiyA+OT46Bhv3njjjT/7sz8D5QV8//qv//reV181T07Ap7H77+zs/PKXv3z67Fm9Uf/4448hzpeW'
        'l6EdEicJKUZbViOrEUbekNEic2Nlpd6wMdP7BWxRUcYW5iYJCpXSrvbbEYuVaLuhBpEZKIW2r0HLoHF8yrTroctSwuoxPY16A1s8tHvoOuDT0KyhromtrTQA'
        'D9kmi2g7SQAZL6EBMmGxzm5tyY1cNLr1BPTAH7So7Z3tvf090j7jaHVlFffGAaK55juZqBTiJHEjFnDhovML81i083PzHJ95TJUSPEpiABsG21mOl1fXVu9+'
        'effhw4ePHj365pv7zdPT/YODn//df3/y+Mmjh5u3bt965847P/rRjzrtNhXsS5O19bUca4bTVWSFSbMOUjPqtVa7fXLS/Mu//Mtuh4KWKMgzT10Nk7FuVNw6'
        'p6IxhDn9FpfY2trmWDx/Nn3H7RRKoXfWsfJSwUnnVApfckUOXqUoSw4aLcWNMLIA09dff/32229zYEPRJ405Q/klR50pZ0UK+1o7qRpYILXZ+PpSsa60T3KO'
        'Uy58Y3tr+/HmY1anaJldvnwZb8iyW4szOsZMHAHpgcROPrLCKI7dAZShHkDo4tFAqHCtS5cuJkmKHQCIefvO2yDHeHP37t37979ZWlo+OjrGqTY2Nn7yk39+'
        '8+ZN6IudbotzwWKOQMwpeo4L3pFbWweUN24pZroW1+eW5/7lv/w/JEmWSsxwOYPJnN/zuAQmFePDnpCm2QG/xJlVOghnTjVGeghHHDOtlUNNt1xUUXdauuRa'
        'rk1nXXNvayuGckluog5VbIrmgGPtGh15WvWMM4TpK1evYPZ39nYALAD67u/uXrt+DVKtLz27ai0pjOGqqEUxjZWjPKxaB5EjSqXztTk5Pnny7Mnm40f4gx9S'
        'LVPcWBBRFNR8OJ9mZMUr2hrZocMkrvPSMqM4g+Ho6Oj+N/cfPnq4u7v37OnTD3/8IVusM2HYQMzVK1fBMbAM/st/+evd3QxyMQrjpcUlaBfv3HkbTCMMg06X'
        'q3D4OqUAUQrg5iQVsdRwRwgyZkBDIK/1xQsXZKUmaSLGlYk2Tdx5iwOdISqgnraJcek4jjIGtGdnwjjIqsm9plg22YCLYZS1Dww3MuS/iiNDcbMLN/euxksw'
        'BrhThcepvoKl1pnAGccqwyZtucaU1hkj23IXZHH9aIE+9xPyihKpFDGmOXXccvKr1Ke5cPlCkiWbzzabx83Do8MXWy8++vgjsXA57RAzGRSzYlwzcecjtN40'
        'lQxK1yNjuSiCaFztQyrmabInzx4/frL5fOs5pDJE8tzC3MbFDUpBtRnl/3EmHxW7HhqFrlyyGikDlCHFdbD9oNPuQup/+utPv7n/DSQorrq+tr6yvIIvxCz7'
        'KZFb6XpcB1Mnh3Obcs+gkC3Mz1+6fBGQwvAeHx1xmgNVZe22O8BfLYgLi6lKurgcRW5GQehRE5nW11/fk4ULocD2u3Qaw4zmwENxqgPZWIdlLkWn262HmAF/'
        'QKjb87UHta77t3jVvJgLB+jCUkY9lbldiTSg4sRYxRX1OWBH1kOvgv9Act8YV3v1GFMG/ClVEgbpN29M5sdYZCY5aUe6YVTe6rTr4YIUByavm+sQmhdc23Kj'
        'b+6zyyo11U7nUo94rqXVxR9+8P6vfvWr03bz5PT4F7/4H6/ffv39994HtCT3qTT9SWAaCybtjW50WVaYddVctfN2SZSfRI/owrzVTbtgzz//xc9PTk6w9Opz'
        '9Vu3b1K5plBFHiCV7+7ugNbXG7UszYYPWlFgFY9fD+tfbX714tkLCqUw2e727pPHT8E03nj9jU8++eRHH/yIrGl51mq1qTJBknz+2Wf379//8ssvNWUNNsCw'
        'cWf7+3t/93f/bW9vB3zj+vXr+JeLtFiKccVYp6a4fT81SaCgioTkSQcilPr9738neqHnKtyqiW5UnGt9fQMco03uJANRgqeGqkoac9HW11NOk5ZkGdGTRQ91'
        'e+AE1ZHaP+SmQ6TMn1O2RqU6O6dFi6gUvF0HeIKce2/7Pbc0zXAmJdSHK4UvbZCWssOUWs5JuYq7H1udQ+kyXpKZbm5ibg/L7TcpmUQX1jXllQSBa7Fa6o0V'
        'cFB8Ll5fiChgSMJoMLJQkjBkP/nkJ1RLkwtpsqWCGwIoV6NI5ENZ264M1XciR0i2HXTpy3oQdRNX39reevL0MQQz5CJk88LiwtVrV6GrQZ8TuLsqKtxtbuKe'
        '2u12D/cOW+Qb7IA6LyxS8Yybr93643/9r6Eq4LRbO9siNb/48ou93b2d3d2EnYjXb16/c+fO+vo6DgOFBcS/+OKLzz77X7/5zZdBGH744Y/wOZj3wsIip1uF'
        'i0uLIOoUMhrFSlN/VtDpufmFldWVf/Nv/nRuDspovLu3T8dOqtQqKXBYrxJ+COWSag1v7+zt70MZDVx+p7PcuxCxwjMlAQUynpSoMYGcUIATRwtA5GXGE4Gd'
        'Uqd7/PgZ1bZznMT1dLWuWaH2zhRrnJkJsQhlMFSJWBuyTZEsSjC8OZU3oF5Bkh/tcfPyarkE2+sb7iLmqLhMQKkokApXr14FfEETW80WhaKftrAXQze6eOki'
        'yWmqqUORa2RPk5RVFzVdFDaoGoklPKOo5eOKKBdhNIJR6Tz08MHDh5sPtnYoEwyLam1tDRKRCUC98O9wSjKbACY2rycpO9fIDGXg1hoxuBNW6YULF/BoWJOf'
        'fv7pva/ukclF62fPnhFbyLILFy+ura1eunQJInxunhKvAHjQjwsXL/36V7/eJ91s/94/fP1o83GNi4Otrq4C93/0Rx9L34KtrV0JhfXB90McEBdxtVR9hp90'
        'AiWLZCKAUU6dqBOyU+Ny83VmzMHhIfYBLugQ12p1KuwtRlQOaCm6bSZ64m5AeqxU3uA+WpYBTQ0MM+mXrLysYBf94UJF0Ofs3ZWV3sgFmn2qf4r/y/IOMO2T'
        'YFaeGCCchLb91jv3q9iVyTZnfdb3a5h4gBIyYPPhJqb/5KR5N/jy6tVrOBCwUKEqAoa05wi0s3LIBXURqlykGdrCfmzLYj8uM0UpiXna399/8ODBi50XJ81j'
        'ihhZX7/EL+E5KRsipHetyc00hJE0t8BfXlm+cvXyu++9CxUQD4W1ihs4aZ5sPt68+9svse9C8WpwNUpqRXD9GjB6kWC95ha5sRsXL1yhPJdgZ2dnGzR8e+fk'
        '5Pj5ixedVhtrg0x1ECFpJ8vSL774kk1tAB7V/202m1/evUuF/MC/pYSVnawxi41crDS4MZwEIwMCxg1Om19//bUfUXzp5cuXNjYuSMaKM3gbCSWi9U40f5I7'
        'QikuskPKRkYF3m3isXjGwsHMOwndq9SonNbouWqJwUvb3kbTEumZYDyI54BaXqogtxrbEPSqdgaGZFRo/Upre7/SZ65k4/QXMoRZU0Z7Yd1j4sVp3Gl39vcP'
        'urv7QBzGDTs4joEutbJCSYfiokmSrngBSjt0Wf7edXKoVMOXOWNjC8epZubFixdgNc+fP3/8+DEEahRGN2/euHbtGjC9tLQk9T8hQUPrkrGnqQPGLDPHDv7m'
        'W2/evHUTP4065JmP+1/fWJe6HF/9/ischvX53nvvAcdU94OTTqmQnFbUpDJNxNOOSf/Bu3ey7A0Q7a/v33/+DErr44ODQ6DqwqULi0vzT54+ffx485f/3y/E'
        'jplk3fwkO201d/5uu0jPiabxB1G0iaJijTk7YsT1SN4fqkXvZyb97PPPj473IaHfe/fdd37wjgyR2xU5kZY7WSnc+mTKoXyJesWDcs4ldZVVfqp87FoJ9Swk'
        'xbHY1bmofrXqp/qv9j/JOz2dUuj1H3M2YFuzAwoUOTTzKq+pNPZajawZ2STy8kYjWovCuUZ9QSspn2XLDnNlm9syypXLovXKxJMXWqo+5/bp4yeQnlsvtqBR'
        'QeoIGbhy6crNmzd/+N4PKUGw4IW2aAUp2p5rX8IzNNQbgFOBngIZd397VwgJlWe2Jq7HP/3ZT8UejF015ahLHCw+S3HvSVb2JGuBEmsANwLNeIOmr3Q6bSY8'
        'lEPAeTfdK1eviODnfEGpv6jkW6AKOAA6mchCKY4oDaFzjlfWnA9xyq+HDx6wwtoAZZdAlTo3NqcWM6ctPwgndjsok5GLJAYrliWclqoKZenxySEYJoQ+Nges'
        'QFwa9yn9a8rArzKydPz4BPzISZpCy8jydmqa1oci3lJB06/tKZ/ktFUtBo02EpvLVZxII6au8cOk7blaJZ+xvIi9gqwTpG9rSjgjIuCzizfvKi9IKAsu1FpV'
        'LH+214u8sOMKly2j6cVjTPXPa8Q9KMytMXd8dCyEAZvgw0cPIVafP3sOOb2+vgbFHyMbckCZxBf1Zk65NO+8SGqC2Hv29NnW1hbOIJeDEJJkOEjl115/7dLl'
        'i1gzki2ScjkYlxXm4tHy6bY2WpZdMuLSQqjV6xybr6QYDfsRPSxPajUyV2+enhjW6qgIXQj6G7Q7ba5mlMx5XlyLlpYXT5pNqI+dTktKJHLFDCr/bHMsOGpE'
        'Um/Urt24ygQjnluoKwmDXlgQjz9X4VdqclM/T5LTyuAZEQ1FmS/T6a5QQwDfj6RmZOJirMXc4RVft97YZAuWtdQgOSMzPB4096DWM9nw6YdpdMocOi9rntsq'
        'h1Yzj7Zz27o4zKjCHduprA0oBodCxLCDmCDNatQwqCfobdl21hYSuioeSuFmi+48nNNGJYh++P4Pj46PwOeARQzZaXYK4Qp9cXOzDqUKrHd+YT6mPjRcbTiK'
        'XUlw43qOtLgcMoYPTPT45Lh50tzd2xXBDARQQYz5eaiAq+ur84vzWBui+VHOEjvKy6ZV5ylko6CbsZGQOAMWkgyaM36xux04qbHlLie2nFOpXCoLxmEZ0gaX'
        'siqxCLnsnbJ+KDYZj62l1NFZS8kBlwdpJbmwMVfnmCTmqNQryFJ9aw7zmCa2ljJ3uUQx3UDKrZuItFBnGGi3Lmu5tHJ4lkokUFyXX8Z+TOEQwAZPKOHMzsSq'
        'xPMzH7xZ54o0sZQM1VY49FkyoYZ6Cu10nRUmTqC1rqJeTqphQIwBS80aKCsB5LTn7NDVmuccD13NjTHSoDsThBddJlwNHsh4f2FheXH58OhQOjyAWGMjhhij'
        '5M0s297ZJovA/BxVI2CFn3Kti5ZnGXHgtHl6ysIghWAWMi2Xw1EbGxvYQPEvlev08qrR2tkYXbGKMiB0qqBCKuEVOK8ktQ1gPg+hJpkEUqJA9i63pUTcWI0b'
        'dJQBhlCuAPekTcYQFo1hjxhwP2FuhOW2oygORb2WtCjJh+ckBvLYU0OBSeGj3PtLORenMnI/zkjL3l0eUlMkfFlpmR4GEWvniol3NQ5x1Bhxc4+cAE2TA5Do'
        'rvY75E/xyaViVWrJ0JE7V7QUC7Vlfh/99KJ2VCEa1Vj0DoTTDQmRUOXKENe3keVF+iephl1q/UWl2LskR1z/ikI/o/PpXnMsbt+EHxEMcRhzAAQ1wuGsZS1p'
        'pEtLQOzCysry7u4ekA3KQZOn/A7nJr3YfiFtU8pQ/aHiXyq5YJqxBMSOduvWLQm1g1wEzwQSulm3cNn0Wpk4cqmLMKYJfXFY0Ka55O3iK1hwUrdFbN7dbrfR'
        'mGOfBVUjkAIjZP3lnaHcvnG8CM6ydKI0N5IQZ7mrk+MTWYQLCwvlpk91SynjlvPVLTvJCy1lbIC/5OHqkp+UQdui27Q7p0EgBeax5CIWGqE1mhtosm1E+XFI'
        'iWe20JSGu+ysyo2iAO08MaqldMvqlheeEKCDtlUdTvUr6rLxUpLyt0XPQ63+m/2/SnlYYHpUm4ieTU3Z0SYOlfOpwNJjlYc6D/28oTNqjWy7se3EXhrYFFrM'
        'PPX5jhu1eB4CBsOQcka3YVMk7cs+lX3wCrgU8XfKliiUjki504F8JhI4x+HhgcT6UPk5iOtWm5aQNVXGzI5cLYCQEAjoN1IMHAQDjLOalcQhw7TbGGW8719D'
        'ya/KsCg4ydcnj7TRXMs57MXVUJZr7geGl3zGnQttqel6VLwmwxaeZcHRYWK8tvKbQXzgBU0VHqtwmnvUiAAAC/RJREFUC0IaaLaqyQ4OUM2YIz0Cm8flTiH/'
        'H0yt8k2pOCpuEqCLeJHEA80gNCTUNdHUqZe3Cq2t5TYmCzTtuRnXOJSWx/gmFjFZTDOr+wwRrqGXLWLSXaakVO/ipjjkC8WplpaWCJ2r5ImQCGBq6M0HSSMm'
        'UU1EhwqYA1K/jIiqGVEk5/ycBN/nleIH4io6V+/D/70AjcnO2Moj3T898QMYtyNKz3ZysGQlaeOkAUI/SQvQyMTjQqYdz2+DPHq6afWp0m1PdTnAo4hYqrZy'
        'Y3cyT6dX+tT/EHlgWrnKHFRBi6MzMrotHVL/Z6rdjeXlQ3cALzBk7SdAh0bqpgFnRClFv7DssHVUqW9rKrYLpUIxk1kJEbUyVCEZUSIySPOyBa+WdcCtMKRH'
        'r2siQYElPLYh98URuxIFUrv+xy7UgdektcZ+D97R4o/LkLmayl4ZNFc6m0pJJFDm7dYH5ZF+QAmVbIU6eWr9pkeAZrKh20p3iyC7stZFyQ/c2rClZcyeA9Bq'
        '6uMkYc5ydBE/lZ9bnXgmUIYhR4w+yk2X7i+Hghj5OYfJ+L5gykrnRs/rhRcr7VX6AXuq5Nxe0duGrf7W+Vt1xX1N6lE8b4vG4S5xUBIQi3TDIhvFcl5MN3ea'
        'DfVyFW+tKwBiv6ccIyc9oOZmbn6d0OSS+iWOC6lMOOauihTDlnOl+DTJki52z05qT3R86PkJ4dgHoBOIQtsL3lBekXJhe7YHz6tU4Aj623Tb6eE76gguCSKc'
        'F9qoL+vJMA3RoFBWjC9JnrdoH7EUZJrbyMtjPzNsVneaOAWNnYmzLXW48g1FMNpe6SNeEKU70JOI5rNZzVI+0KtUHXAll8p+3VwEqMz6Lpn399gdbrnBlptx'
        'aRZtnBFWIpg97crusPsjDGPpU8pKCXeuSLtUJydN0rybe6dGHejggPyCOtMBebwpLKlAs/UcnKxVFa9Z4WTm6R5lh55skh55hCU4sBc5lL7MbLcwZKQLMiLT'
        'ZMkJOAXCeOSoV1Quy6RYn0HiJ12uccn8NuaeBl6xzDk03ml1pqh967NWoYuuFNxx2T2eO5gTuXtWjjI7pojfd4p86rLHReV3OaEu/a6vgNP3ryGANiqjebTU'
        'eENzG2PdF5ErjdFyIhg0L2QzZX9okrYMNfRmBPunOoSE3uOADeN+dGEYoZUQFqUEVdmRzYUNOdOYevmC5yMltLNysKmQQrA5sI73itzjwCQv82ybDgs6XtBV'
        'acemgUmiJI3TLEi6URTM+dQNPiiz7UunMTm0GV9BsSIV22y5W557OlMkxxT6oy3DI6vFUOSgntWyZ0jRha3KtbLg/t60ZPJJDoj/fSk0BXj4TI+NS5l3neg9'
        'Z6ni/yVJxjKCasrmFEt8mtuWUqkOgOZUBW0VtUzQJaWrjFmgndHvVUH0es3a5BK9/D07LtruVURRXpRm9DnomXYKSpVxsaFQ9TJFD9DWlpo5QSR6fmi9ujV1'
        'a2Ob11USYiVrP5e2mWXwkJjnSjLtST9aJg0Fjp1+KNRZFzntJDlcuUXr9TdGosQZXiEUAOoawBeaOcVr68Ik4uMOjPe9UjiUZLpygMr9YgqvIRmKDBkCmP7Z'
        'nEIAyWbXzfK28cgM5/lNpRMdAtNtos5hW/nd0qBhXbizPgNJ05NK3KvWFjQ6OGvaHg/x8SFlBZ8xrBdy/D5xf58WGYEq86SnG8VMkXtUq5B7RlEPZ1Ij0pxy'
        'XbIotyEVuuhkypV9KbIvAjZKKDFHOMdetcapcF7W9aQEKGkhnU63UkJ7kDiU8f0mF3+HKqmz0BpuqWmpc3n+vVI4PHxHWfafS7ti66LhOFbK5gLownFPIUS2'
        'Y7y2pzs6aPnRqQ66Okg8/8RCs/Ionq6AknY2Dav7Jk3MggWvtFUBVbVyFPUchwbbKVVBrOr3EaozLexp8mlf0K43KJstxHfjiIKRlScN0bUKjF/vKqDadGzW'
        'MsmhzQMviztdCGxp0erTj/JdApkt3lCR5nAgiobTQcXyIdVHoYBHZ55nCNMoi2iVRT+kvjfHS/C1An+Gm/Q/NTHtdHFQDo5f0MalllB4Sc4h3Bk1F6do5q7S'
        'Xa1TP0zCOJFA59QkmQUjSYJCKFmvSMxzuffFrKkilIMu6ounUFldAtoFb/AGPTxY1dl3ZQ8ooxacg/os+rVxnMlZ2OgZKp5FTr8KPOO7nBTDbhe/C0ktK87W'
        'cYqany6r00XuV+6bnEiLMfhWSO4YG1InT3ojNg11Zovg1e3L4tTuaYciq5C5mZVAR8mVAJRzB3gt8WUsMMwMlEIl8SqVuHrr9Q28sW6rGb8g7KwU1ClUXTs5'
        'C8Aq1Us1Yg9YStMq8XHgEiDKKglqRvk5hRzpjuShZGSYI+ue0VKOtQSPrTADUxEDqogXFVroWxdO4smboKokqSmFi5rIuSccVPQh7u/r6Uxo3OhWdY13YhuZ'
        'x65UyGn2pvqc1cvi2YaWkZpZXQKjQHPRblmVHWmnqm03hZVSv7ocE8J0VqmyfWVWKAJoIsRmIemVms5Ea6c9jxGAkrlNZ2y65XRAxW9CjmGmPyWSBM1eQCOm'
        '6wrTePlX4PXVPfL+EexSqihYUe3toDITNMm0bkU0snQkN6mvXEiTzw3blFRmKyx1jGmVVyZ72on/dgDtjNnDZKIot6W4Gy83lTWq6Oj7yhuGfhW3Q/FHnw0M'
        'RWdrrk5hJVdKwpfJFZHzIa7EhTQ2FvQ7/mrlPOpVYNhXI0v9IUE7YcerzA1bfXKj2zpwJLiIm4JIFkArr7fpuO/2Cqj3hRBJkRAzDdYmHaBefYTUCPdQ5X0R'
        'dTgW0D6HTHiv7rm0nEL5qs8FcQMtJfSs7kkQZQqftC1SAC0nTJV9522J/uLBtVsbryihz4dIdc7VOwbHhbG6b99i8wcLqMBn/4tjxEUErDO+Fe1dJMq29Bap'
        'IXLUt1PJTjNxAF59wSvr2f4kvl58iOqD+AQJLYB4dQntwpRf+cl8LLDUGzAP9BV4qzQpcZrP0J1zdBznyH1VVd+cP2PFju7Ecl7i3WcvdKUVbFmICb9mqujA'
        'Yoqo0dzRbOf284o6VNbr9SWqprywLXOqRT+5OuqMjBNK24G4xZ7IVkXxFDVpkxdH5yw0Qu2pV27coDg7qW+0zvxrexWJVP8nbqZcXmk2nfTRpftMLBCCnQGl'
        'UL0UIl+JgfWxZ+VaYsnW42q0SshLeQvuF5d9qGylCYyjHqpfrKppZ+VbENF28Fq2v9im7RlEJ9seZkQR7UxWqlUDK7+viZoY4Vg+2VHFlp2H5NUsn8F5Ze1L'
        'yN+J6vFAAbRytzWm5JyqSM6yfduZZddJX9Fg1TOlk0GeJbuapjXyNGtyBhOvBgFUtffYvk5434Y9246wvZ5PZLl+T/11KAbMDWxss32+C9UbVzWsTeEU69AO'
        'aetmR2ZevbrKN+UsqEqjQ+XorC2q9JWFS/sWdxFe5yhgr4FSQaOLPnGKq1ZOepkJ2/eMXCEuGc6MUhO9aQMelTdVUdVp2MK0YmfszSjnf7aDNj5VhCqzydJT'
        'Q8ez11diOtNxpaZdBbxKrByFhkWqmB51wooUPJtqq6bfEJUdadcsAzW9iu7gbFRFFk+B+WJboHzPrO82rBowbouVxPtuvIoA9cEwLxdc5ZUhgRPdHUqSQ2Zg'
        'h55iK5h4kJX4yvEMT40mN45nDhTPHz1+RbUkeSN1OeQ0gSrJ6Bg0F2k0qhcuUb0PNZSE9KXTqrKAQpVpDSlXQOJZqaKdoav4pNWZlSUVQDg2tK/TuaqWxrZF'
        'esM0jTf/AHvRGbFS+rjGn9xOpkDWm8LHN4UMVy4owavIK9Fcp8J5+T2dTyLA1hv99wr/VtLwcjCXuZwjsWdICV3NBePZz254iyiVQjVhU6nUVBqQzvZsVxU7'
        'qu3LGZPM8NaxqrITj4aTOgvxkfTefjt8dOJicPaM725Ydd/NTWHxspWomFcewX5VwpZ3YivRDdZ5k3ox/k5qWeE8U7i+Xoog21cTaOofxWX5LaLmn9LT2G9l'
        'gCphSiXxsF5/323leYO1xtRZqaomG7/UMFk+K6n2j3yCmctp9S3ek5pJyaCJtoZZY1oNv4qraTtQhrHnKeY3/z8oIWCNy28jIgAAAABJRU5ErkJggg=='
    ),
    'save': (
        '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'
        'BBEB1ClTPM9Lp9PAwkULPeO98MILO97cUVxUvGDBAuDzn79xxowZR/v6enp6UEABESMiiiaA8rJyhNpTaisqK7LZ7J62PYDxDMco'
        'ijrngJ8/8vM9e/cIgpDwjAc4depU8gBVBdasWXP9DdeHYbh582Zg586dR3uPOueiKAJEhIQgCCAiQBzHQFNT04oVK0RIpdJAaWlp'
        'KpUqLy+fN28esG3btjAMYxc751BUFahvqM9kMjU1NbNmzopd/PS/PA0UFRf5vo/i1AGPPPJIdiKrqs45ThpireUEExHyVBVobm4W'
        'kdbW1paWlra2tq997Wscv8WLF99x5x3APffc09vbOzw0PDAw4Pv+mWeeCUxMTERRVFdXd8aZZwDqFBAjCUBEAFUlT0RIKO8REj2H'
        'erZt2wbU1dWJyDvvvnPkyJGioqIF8xcAN910U11d3csvv/zUvzxFQklIHlPmzZ9njDnnnHOam5t7enq+853vAJ7niYjmAcYY4MiR'
        'IxPjE7wPEWHK6tWr169fH0XRc88/B7z6yqtdXV1hGI6NjQHGGECMJABBAOcccNppp5133rlAUVExMH/+/KqqquHh4fZ97UAYhigI'
        'x7jYATWn1KRSqdra2jlz5jjntv5yK1NU1TkHPPaLx7LZbBzHLnaAiHASEGstJ56IqCp5zc3NItLa2trS0rK3fe/f/d3fcfwWNi38'
        '73/334G77767t7d3eGh4YGDA87wFCxYAQRi42NXV15155plMERHenyAcIxw6dGj7tu1ARUUFcPTo0cGhwaKionlz5wE333xzXV3d'
        'K6+88tTTT5FQEpJANA+ob6g3xlxwwQVnnnlmd3f33XffDaT8lIhUVlXW1taqamdnJ9B7pHdiYoL3ISJMWb169XXXXRe7ePv27cDb'
        'b7/dd7RvfGL8cM9hYHBwUFVJCILwexYtWnT22S0glZWVQFNTU01NzcDAwM63dgJxHKMkFAXiKAaKS4o9z2toaFi0aFEcxw/+5EGg'
        'qqqquLg4DMOxsTHgueefm5ycdLFLACLCSUCstZxgIkKeqgLNzc0i0tra2tLS0tHR8f1//D7Hb86cOTfffDNw15139fb2Dg0NDQwM'
        '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'
    ),
    'connect': (
        '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'
        'n+wr9QyG9uzgDX9wzoQp7bdu3PuFu7Zt3JG3lgfee/Hxp88Zf+BA96e+fuf66oSqlRQCoZAPnTO77b2vPWHJjJa9e/d/4Vv3HHfy'
        'iYvnTJ3QXG4wlBJOacn2VdGfBeWhnCMDBlNEA4FQQML+3njHmkfu2/BIQyuaBpJfXbHwtCVTieIr16/beLD/xIVzZk2fML69sVIK'
        'k9oauw/19GcYVKkarKRqG+L4ctKYWFff0E60rtkycNfDux7cuj8PZbcMLye0laswiiQcJhw1Uiy52vzgWfPazj9x2tzpEz73/UfW'
        '7ejbM6CMaVJprMaI50OAUkd1319ddfY5C6c8tbPrf17zw7UDHQwGzxBzWoluFQ3+6uKO3zx72mtOnvjxr9133SZs6U4pL/uhPJaR'
        '2LLplTefMv69Fx/32Zvv+af7ejcfKhVsSDQU0srx6aHfvmDWWy5Y0J6WTFIoHObOhDlzfup7m76+oXtrF0+Z3PDV9y5/eH/f53/4'
        '1Nfufbo/L7ckAyvPnn3FKdMnj2v68WOH3v/Zu/rKTVVLIgOt3FztO23hlGXHTSmZf+cHa3bGtrRUaWff4nHFr58794LlC79051Nr'
        'H93Zd6h/ekNl2/7+pzztDWUwpHk1tyILDRPLetWchj+88riGmBaVxrQUExVZX3IoSf/qW5vuemhbUP6GU2e841dP+duv3fXdR7sf'
        '668MJY1pvvuK4zuuPGPOlHHN16166AubejNrFWleTePQkDXi5YRcuQqjSRgh/GeQYJC1xK6Lj2+/Yvm0xbMm/J9vPnT31oG9A4gM'
        'tJBDGAOF9uLAJ9++4tyFk7fsOvRH1/5wTe84hgDPEXMLCYVyUb1k8fjfPGf6a5ZO/MQ31n15vW/rCYaiHPcWalRIlkxreOMp4993'
        'yXH/cvOaf1rb89jBJCpNMeBJZV4y8JpTpl5y6owpZS6a1tGdV/f2VvurRUdT0tnW9Hff2fCtdbsPDSWvP23Ohy6bfevGHdet2Xnb'
        'I115wXJSvWROxxvPnLF84eTd3dUPfubuJwfR7RYRpHLFh2ZNbps5sTWhb3x0S5c3lFObUh46fUb5Ha9ePKGj45+//fC2vV1TWtMz'
        '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'
        'XCn+D7B6F1jHh5GAAAAAAElFTkSuQmCC'
    ),
    'reward': (
        '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'
    ),
    'cursor_connect': (
        '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'
        '8XbwmftLQFbSq934K4md0EH3/j5G43/PxaeU7knN3wGDuVHK+o6JfriJ1ddPndsabFvtnqXV5fmF4WB08q1TVa+zvLB4YPVgp9xa'
        'KD6f8n8ttsDcglnBI2kMBsz8KP0fwT3AxPz4i96a692/eWJt500jINejZmvb4h0SKvBSp9rb76eyGNbNheF40rbzZbm20Du4MD/f'
        'rSbZ54fDSZt3Js3maDy2sSxzawKT+nv2UhRAVvXCwSN16oDKKvbtq8BGCg83cwS3lwy4l5hP9AqXSYUkCDuQbRAgKBNVYqGgX2Kl'
        'szXjJg72+fwFixuI63XmU9lP7NrazKORQeB9Kyo6iSlrMsjt2Lwrgof6HFkpqqSd7PMTE4RY7rC3TIaXNpvnthUytyAjpmQoK/WW'
        'JCVBNDHazIT0A0cbIJwipdaZiMfz8D96oAPGkp7rtt8jsinRfnf/hr2X+m+q+vOqf8Huk550elB+TfHHYgjk4t85P/73v/Ds59uI'
        'px7/dLipc/vwoWNF8wtOHzm78/QLx49+9MOfeO6rf/zWuZOP3v/4h+4fp+ZHEmdNJ/SU0uP2huILyedIraMf/b+r5r9zHrv4ZJH/'
        'F+S6+OXMk0KT4IdfaLfmug9svbq2c9LMNIPtdjziHajQ3n5nT68rFRuTemM0riP3U9q/0N+/2F/q9trw+eHk1GC0MRy1jgLtn+s9'
        'tG/P186tnxxOwNyWUHdxWZ0OkFU9f+hIrY5JVdXuW62YEm4YbLXiVlSaA5043E9zVaphkDXJngRhShyAppC5qDF1eNQybmmDQtER'
        'n17Tb59xiNsrOqm3WBgkNjfzcGiQ5H2rqSjFrtFGdjbvhh7oxydWOy3+6mZsjr23m3pFOlfnEyP29/yJvcVioRe32ue2bW5OgLnM'
        'C6ullJhqYrCRZemHjrUZB8nGdnZ8KA8/80CHXeLZbvt9om07/ym+S/m3KJbJL1vfbB3JzaHtc4cmA3Xn856Dx4r4YVxs+e/908//'
        'dtPWS3PL3/bx73zjrdeWFpcPzv83Kf8Ds9h0/sEfv/DGRx796NfeeOmrX/uS4QMPPvGR+55X8z9E9Z9vn//O7QtFWbG4tt3pvZGK'
        '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'
    ),
    'cursor_challenge': (
        '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'
    ),
    'cursor_shop': (
        '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'
        'dj49vPP61vUvN6ON0fpFfoZeP+7+qz9fjC+uC4HBgEXPwjxSkocU2Clhp4tISwpkO6IE6JWbbQ1w2Mbq7AQ7ExtskIxlCRmMbB4y'
        'picJnKYneuacTM+AEF7bLOsXFEV8bmTctif/4p+Pf+FvtD/60bJt17/+8+1bb5XLV7puERnDl75pic/MPIPI5PioHh4kFpgnCDw7'
        'Wvx7F/nPXpw0AsTnL7P+6E//97t/9Qc7X33p7mt/+pVv//333/izKz//63d/8K/Wt59f27lx7WsvC/H5Szhou//6u/O7o9FgVDgl'
        'TpkVgYTNisQjIQHCpkqyLSEQhIpAr9xsa4DDdprOWGStnTCGgpVOG0tAtXnA9Gw+ghEfIgSKYDLReKLxiFIk8dNlUNb2//kjrU8Y'
        'Drvd+14sxztXlseHzJcMm+FL39T6Fp+DrmOxyPncsxPXTo7kg4RFL5nuTX99O/7Tr42vDkNmRXyOnN//P/6nrm2vvPD149vvxHjU'
        'LaZXXvj6a7//v114/m/sfOmlnRe+DuJzZVL+i4Puv//h/K4G440BCkhAmI8RiAfUQ+BiK0IIdeACjYqQXrnZ1gCH7TTVGFdnRcYQ'
        'WHYaGWwQGPcwAmNAGMRKAmLFPJ2gFI2GMRgyGGgwUNOI4HECc058IjLObn77drO+2VzYnL/7noaDZjTK+TK2NrqD4+HVHRR8euIJ'
        'Nl3rtnW79LLL5VJddY8zAvPxZHt+uGBef+1q853Lzbe2B5cHEshYiJ82+9bv/S/HB/uD8agZbjSDspjuNYPRCy//3Vye3Hnz1S/9'
        '8t8Ngp8yYwGt/PZJ/sn99vfu1h8cdOML4+GkAfOQ6MkIzIoAI4HpSSGBEUhYFgoJKjhQoxDSKzfbGuCwjVXtCna6JwyyMCsSBmQM'
        'CAxGgFhJ0TMIZEB8YgaMRAlKQ2lUCtE4QmqkQgQhBBZPZxCPWHwc81Sm58RJprM6O2pVpmula8lqg1gxH2Qw5qmEuq7OT5aLE9eu'
        'uz4pL27GVzfjxY3ywnpzYaixwPQkfkJ23n/7r7a/8HXk+298T1Ga0dbej7/fjNZd6+a15y899yKIn4BBYEh7Ye4u/OZhvj5tXzus'
        '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'
    ),
    'cursor_arche': (
        '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'
    ),
    'cursor_receive': (
        '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'
    ),
    'cursor_stamp': (
        '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'
    ),
    'stamp_yes_context': (
        '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'
    ),
    'stamp_yes_button': (
        'iVBORw0KGgoAAAANSUhEUgAAAPAAAABgCAIAAACsUWiGAAAgAElEQVR4Ae3BWbSd5Z0m9uf5v+/37b3PrHlEEwghWYAYJMDCNh6p'
        'dlUbQVcVVFWvkNVZSae7eq3u5DpZWSsXnUpiXL1WVm4qF33RK2MZl2uIy4YuYxuXBzCjACEGAUKzdKbvnLP3/ob3/+RImA52Gx2h'
        'LvfFYf9+LIoCv06SMDAAkMSvH4uiwK+TJAx87JHEfxQsigIDA8sFi6LArxNJDHzsScJ/FCyKAr9OJDEwAEjCrx+LosCvD0ESAwOC'
        'JPz6sSgKXB1iEUEQBPEBgnAJQRADAxA+SBCERYKwSJCE95HE1WJRFPiIJHGREQDfAy7C+1yOgYFfRBCXCIKgS0BA0CX4AJK4KiyK'
        'Ah8RSRAkARgNBMFFeB9JDAz8Ikn4AF3icgiC5MLfBRZFgY+IxotAEEYjCQLCe3gJBgY+QBLeJwmXCHJ3CC6HIAn/wVgUBT4KkhaM'
        'oJmRBEASBIR/RxAGBj6Al+ASd5dEEIRci1wOwJPjfZJwVVgUBa4IAQEgGWLgIvA9WETIhYtcGBj4FQjiIi0CQBpACIIkl5RSAggY'
        'IMlxVVgUBa4IcZHMLIRAgjSSIBaRTCnhIgckYWDgl4m4yAHwokAYoIsgl6eUAKNMcMlxVVgUBZZG0kiYEbzIaCRASC5P8kUJrMz6'
        'QE0RAwMfIAA0XCQBUEa1DR0YwCAzIXlqkIxNdCS3BleFRVHg8giApBlhZqARXARISEADVfBG6hNd2hxUEo6BgQ8QFhEgQEcA2tQw'
        'MQJrCbksB+DeMAnJHO50XBUWRYHLIEgC4CLQLJgF0iAkT44+0A/oBnThM4Yp8jwwL9QYGPgFTgHIHJmzJYxQE+TKxDXSsDAqRLKh'
        'anpKYhJxVVgUBT4cSRBchGgGM5IQIE8pVUFd43niOHlcOmE8T00DXSBhYOCXiEAQopgBQ9AEuUZ2vWOT0lbHSlkGOFS7w4Wrw6Io'
        '8OFIgiDNEGkwAynB3euUernPGk+Cr5Gvu94xTlJzQI9wDAz8EplgQgAD0KHGwdWyPY7tSLuSNspGRAiNu+S4OpwvCgFOCjI4QYl4'
        'H42LAiOR0xLNIU/eE2fhp3N/zvQywlHZ29K0OagWCJljYOBXEJiEBBnVgjriBHEtdCD5rW7ba44mBrjRCREgfk4gICyJCzMzbtbQ'
        'oBSYCBMMwntIWrDASEayAaHG5bOwt4VXMn/adAx2BjYl9OgCDCQoDAz8ag5IIBWgTByiVgG75bc5b6p5TWMj8kgnZIDhIgECBRFL'
        'YW96OlmoLEBNRG0MQtQlXGSMIRqD0WA13FLl8HOMz8B+GNKPDafBEkwAAQcSKMgwMLA0ggFOYiP8gONTtd1Yc60rQwrGCJmwyMEE'
        'CDIshXMLhZKbm8FAOSFCEhYRNEaLNPMQ0SRTaeks/Xnak7RnqBPgAlgDggIQAAHCwMAVoRClkiSwQbg16dPC/qRNdZNDRgYagAQm'
        'LJJhKZydn7fUBE9GEyyRTryHi4zBAs08RK9T1LzpGP0n5JNmL0NTYAnWWKQIRECAAGFgYGkUolSRNTjm2i19Rrg76foqDcvNGGgE'
        'HEyAIMNSOD3Tb7Nr1RQDk7US2m45AJIgjBZCoJksprqOfsr4Y/jjpqeN50SAAmssUgZlgIAEJgwMLI1uwVIg+rB+wrj7AerexIOV'
        'NnkyIpARECBQEJbE7tR0duFNnXuN/QV0xpr11/vKzR4yMfAihBCDWaMGTRlwFHwc+l7Um6auSJiDNSAoQhEQ4KAwMLA0uSVrRijA'
        'uk5LvsNwsOEXa+xtmrYUzaLBAAMEOJbCaupMfPNn6dizNnsBY6vSDZ9MW/am2HbLCNAYQwzGqumHpmd83vmX5JNRZ8wlGiyBDeBQ'
        'AAIgQPhYIH5OGLg69MQqNBP0HDYvK5OvIW9r+FsN7qzrkeRZDJkhEAGQkLAU+jN/wb/9C/UvIBokdlbhEwerXXdVwxvFzMxjCDSb'
        'r2eHmwuZ/7XCNxDeNCSiFhYZPh7c3cwkBwQYZGSAQV6TAggQAx8N8f8jFOSRnHB82nF/qZsXmpzst0POlJnBISyF9Xf/dXzuMU8L'
        'Kc+YUogdbNmbdn+qWrfLs9Foi5hMZTPTSccy/yuEb8FOEKAkCiA+BiQBIikJcCAABhgNUAM4YAAx8JERECCAUJACMeY4kHBfrTsW'
        '6mGhbIeWKSMlYkns/pv/tnPiuZRZt9UJTdlOfWXjtuuz6aZ7yrFNbQ4JTTc2lqZy/W3QN8m/ZZimZ/BcdNCxnAmXuLsZJdHonowR'
        'MIkkgAZ0KADEwNVw0LFIQQrEkGt3wt9r8Lluva5umra1s5CLDWhYCvt/8s/z4s1mw3peu7s+exrH32jV4ugW7TmonQdsdGNi7KYq'
        '6ni0bwT/luFNWg8weEtMYMKyRbxPALFIiwABRkaAkpMOOEAMXA0DEuiAS4QikQtrHZ9NOlT5Db0+g/I8yxBqMGApLP/X/zIrTzQ7'
        'dsR9d5TH325efaEz3zWM4/rbeeM9WHNtHYb6VRXxegj/e/THTGdgFRapJSYwYdkiQFxEwSCQLk8kXCRzwICGTIAA4VcgQFzkGPgV'
        'CBiQwAS4RCgQQRh33u3+u432LXQzetbKI2INBiyF1R//fghFuvlGO/BpLMz5K0fql4+05udMbezYh0/9Zjm8NVbWZD8K9m8Cfmze'
        'yBpYSTfBQGF5ImAUIBPyfrmirvKR8Un6PBBA6/dWNNVIe3gmZjOQQAecoNhAAQhAA3WUxkUgTBkSIQhOim5wegRdykAHBAgQPl4M'
        'CqADDiRJpGkRgvxm1+/RvrDQX1nXynLGzAXDUlj+8UOZdX3fHt5xEOWMTl+ojrxt757ISseqzbj1M2nTLcyGUvy+8f8OeIqoRBAN'
        'PYgAheWJUqQTyl1DLx8Z6S2suPH2M50wLQ/J7OQ7m0+8Ha7bO7tmzQW6xJps6EOiS233nFalcvzsmU7ZpI1bF9qhpBrAnEQozeat'
        'DrJGGk9OiTHrEhU+XggRFCBAgggKkIK0C36f+OWFanOvbFotxGC0gKWw/Fe/m6PGvutx153wC6iz/mvn0vOH2zNzsMjr9nHfvT6x'
        'sgmPB/6l6WlY5QwmWBNFhzmWJ3PlTGQarjX+9W+0ipmtv/vw8RWd097Eyuy5n97wo++XX3pgdvfus5bc2TPW1qxJpDBaV50Q++Xc'
        '2mefDXPd/l2fTWPtefNKim5kXsRw1vpUqN039usMss7QBWIeA4CQyTfBP+94YKG+oSjLThsRWYwRS2H/X/1OCzX2bdddN8CPkuOp'
        '2tB79QxffKs9M2+hrev2ct8ujT/j9m3Go4KolrnRg5hgCcuRFIvu2NTZVXV33DO98uroc89Un7/X162SwLluffjZfH5ed3ya4yML'
        'pnr12vPj437s5ZE3X2/NV8MJE3DIceZcd342bdlhnaGkJiPIMLNlU7nv9nOdVj+hnCs2/eT7Y/LWZ+4932mfxQAgmLQaOuh6oPTb'
        'Z3opM2/FVog5lsL+H/9Oiw1u3qZP7lB6zmzUbWf/ndKfe6d98nwoa23eYjdfr/VHvfUE8jcEmrcpo1OWQMdyJMTJ6fETb6/qzU14'
        'Xp06PXT4hYXb9udjQ0NJqrz31mtwNXtuslasg6ot26bXrvPXXsrfONqZq0Zmi5FoeXu4VxRz/TlfvTZDCAtzrU6H7fa5bZurW++4'
        'MNSpGlWzM+v/5q9WSq3f/O0zw52zGLiI0grpDtcDle6a6SIwdbK2xRxLYfnHD+YCbtqig2uUHicb5vvku/tvlvzJi9nUWRjD+Cju'
        'JDa+6K1jsJwpI1wUIEBYjoTsxLudn/20fX5yxMNcsAbOzFacOLZ5tqttu861Ao0zQo8pN9Z3HOjs3K3JqemZYrjbrHziO/XqVav2'
        '7j8ZUWTJpNa7pzYcfrGz64a0ZeuJ8Yly48b5Vh6byqfnVz7+F2vdw32//9ZIPosBgJBr2HWz9EDln5tdaBvT6NCQM2Ap7H/twZYM'
        'N27V3auV/pK6gHgd4u1pek1z9FRz9Ghnvh9Cpp01d53QqpMKDRHA5JYIUsSypDBftM6cGVooh8QYYcA8Of7Ssxveeluf//uT7ayi'
        '9ykJALvrN2h8rL0w1+lW7comvvtXyGJ28EtFRDerW0I4+ubQkaN2063Ysnmmk8+NjfbyfA6opooV3/7zCWLovt87PpxPYeASqe2+'
        'W3ig1r2z3TF4Gm7njBFLYfm1B3NF7N2mu1chPUp/28Nmtj7JtLd/ar771DPjZ4tQuq/v2S0FNp31rMtAoEmhocxkWJ7IppV8pLYW'
        'sSJ4RrvgGPvpD9a9+Jz+4R9Ot7M5Nk7lYi2bY+yqnKh6myq0PB/9m29mZVnf+w+aqH6shwG8cMRfe8tvv4vXbKyjT+Whn7dOh9if'
        'nB3/f785Ejl+6PdOdLILGLjElbvvBB9o9FtFd4U3aagdLYtYCsuvPZSzwo3bcecG1Y8Cb4EgVyLeBRxoTnj3Z8+2zpxtkVi5oNuK'
        'tPWU8m70ALggglheJEACLaWRY6+PvnZkRW9hBaxELPtN+8yZ8Quns5v3nw1eU8PujcWFHddXu3aVh5+Z/96/XXVhLvMQmhQMczEM'
        'i5U5zFvuqjEfciJ12pq/+eD8fQ80bRQXZrZ++1udPGWHHjzZak9i4CJCIaWtwn0175teWCdPw3m0GM0Ml8Xyaw/lLHHjdt25EdXX'
        'iTcEEqMe9zC7Q5Nr+i8f17FjnYXShhd084zvOqOhhaAAOOAAsbxIAAQw+dA7b3XePDre647KSliq0sipkyPnTme37J+M7EK5JIZq'
        '+45y587eO2/0X3huzdRCrpABgZonhoTa0ITUgaJbD0GeYovF9r3zB+/u52lhcm7Lt789EuvOod892W6fw8BFhCylrcJ9Ne+bXljn'
        '7iN5sJiZGS6L5dceylnixu26YwOqP6VegSLUrsOwdW4JvEdnx6aeenr8+KmIPrafxf4ZrJwDJQh0LEOUXBIYz54Np062yzITaikk'
        'rXnrzdbJd8Ndd3db+aSrAnMi27C+t3njbLRQp43duuPsGCK8Z8wlNyyENBJ8QnDEKqEM+aSyYqg1jX4909/07cdWamHlAw+earff'
        'xcBFhMx9i+u+2g5NL6z1lEbyzGJmZrgslo88lFuJvdt0xwYv/zSkVyCDMg/G/BrYnfS95cmST78UzhW2/gwPnMXmOVgCJArLjgQQ'
        'kMkn+r2xbj9LqaH1peC+8qUX+Orhzv0PhSxOkbVCDwpDLXQ6cwqztW/43ncmFmbXMZwLFNCq6WSPilDLkcRONP/cPaeGxuezOMuk'
        '2YXVj3131Ltjh377bDs/g4Gfo6etrvsaOzTVXet1M9zKQmyZGS6L5dceylli7zYd2ODV/2P1EYqAyYS4CuEWtu5SsaJ68nl7azKs'
        'Om37T2NHAasBisKyI4mkkDXlqqpa5Yiu0qwPBfeJF5/T4Wc7Dz3cycIULSkWSoyMIc6jdbbR+v/zT1YsTG+NnWMRAWyVdGNfhGQJ'
        'qJvxqPSf/MGp8dXzwQoTZuZXPv7EsPdH7nvgXDs7h4GLBDClrdKhxg5Nddd4XQ+18xjaZobLYvm138+ti0/s0IHV6n2dzVGihhLU'
        'ERx5B3Efs3v8VVU/et2GTob9p+z6KYQ+PRMJCMuLLkLTDD3z7Pjzh4dqb1Gt4AzM6sZm5rLJ6e6O7SlAEEU3tYOq7ZvSbZ8+PbQa'
        'f/TfZHt2H7zr3peyFGJqRc65MlEe1PjY4Rc6P/3h6f/sv2jWbJ7M43nVnJrd/sR3Vwbxy4fezcMUBi5yAZ62CfcnOzS1sLppmuE8'
        'i7FjZrgslo/8wzwU2LNTd6xA9+usXwUTmKCWIFiEX0vdrZdb9asnNHom23+e151XKK1piwAdy4wgIDWt114bffPtdqNM3jIPuVnj'
        'Wuhm3X65anVDBSWj1QRMWL867br5XBzu/NF/31m75qb9B19nFU2dyK4jyEqBSWPHjnVeev7cP/3Dcv3mc8HmXfVMse17/3Y10fy9'
        '+062wgwGLnKBnrZJ9yc7NNNdW9b94XaehY6Z4bJYPvJwHibxiT04MIz5R1EfgTmCQzkgpAa9VejfgRdWNtMz9YrJ1u3T2H5GoQzV'
        'qChYg2WGRiA14cK5ztR0XsHc2yFlrSBHXde5RQf7UsurzGIX1qNaI8PasGky2er/4Y/GmvqafXtPNL0WMWRWO4Qw5x6I8XNT+Yl3'
        'pv/r/6q7adNJyD1bmCm2PvnYOnHhNw6dbnEOAwDhApPvkB9KPDRXrl/oz48MtTLrmBkui/1HHm7ZJD6xGweGMP8NNC8BBhjkSI4y'
        'YmqLv73NLoynOJ02TWe39rXmvEJtqYVFdCwvkkiTmOq40OsUM5vLco0ZmzQrDFmeAKQ6BbSDdY1G+dDYzPDE2+1YNc2ef/k/VqvW'
        'bL395ndjgoWcMgVLqmDWoHrjWH7kufY/+2cLG685YspTKGfmVjz57Y3w/DceONUKFzAAEJKs8e3Q/Un3zZbremV3ZDjPrG1muCz2'
        'H3m4ZZPYcwMODGP+UTQvAQEKUIMaWBjCuS3+1noujKSRC9oxm+2tfGJaobYUAYLC8uIuM4MAZDOzQyeP7+h1VzJ41czJJ8JwAY+p'
        'ZEA75jNBGWVrNl9YveH1DHVT3fQv/6eF9Zu2H7jlVK5Ey6ioEJIa0mr2XnstvvhU55/8YXfTllfM82T1zNzIDx9bgzT0G/efz+N5'
        'DACEJGvSDuD+pK/M9Nf1q97IUJaFtpnhslg+8nBuk9hzg/YPYe5RNi8jBbghNei1cW4zjq/F/KhnrNe9nd9aYX1PsQZlSgIBYnmR'
        'RBJgg9GXDq947M+3Ts6WzIqqHE7llrjyR6FZ3XQnvGl1xt8KaTSg/ekvz37y7lNtztfVtX/01XLjNbvvvnMqpuQ0MLhygvToqI++'
        'UT37VO8//8flpmuOm7uTs0X86Q/7Xo18/jeZ59MYAAhKIfl26VDtX5ntrSmbangoy0NuZrgs9h95uGWT2HODbu9o7lGrX4ETyVBm'
        'mFqtU1t4YViOetzDDWe5c8pHSjCYYKhFAsTyIgkgiaTOK4dX/u13r1m5qRhfM/v2G6258zt33/UCqpHXX2xnNrzrthO9+c6J10f2'
        '3NK94+7TbetWae2f/Ov5wOu2rk0mb5jmu6NVFcbGexldaKbnqvPnun/wB75u9RzRE0Ov7Jw43k1Ve+ceD9bFAEDAFZt0nXB/8t+a'
        '6a5sUt3pZHnMzQyXxfKRh3ObxJ4bdHtHxaNWv4IEJKI3itPX6N1tnAueL/TXNUP7Z7TxZB0rUzu4EZUIgFh2JJF05UcOr3zuJ9fc'
        '9KnzG7dPP/3D/MI7e+598NXUb3//rzHUmvjMV05NT+Y//e74pi31gbvPtNirMPx/fWNm5sK2dt0hVLE8P7W228WGTVOdUAEJeWXW'
        'feB+Wz0GWAHE2kfKqutVGJ1IUI0BgIArNn699EDjX55ZmHA1nXaexczMcFmsHnk4C5PYfYNubWHmG6yOIAG9Dk5vw+mt6rYrZM3E'
        'WOfu2tY8o/bbYsvVJlJQIwDEskPJSQqdwy+teO6nqw/eM71lR/eJJ8K5d7Y8+I+OdWfx+HfGOhz50v0np6ZHf/D4yMYN5f6Dp4KP'
        'CWvrZh6gmnEAdcIzT694463qK//Ax1qzzgCrgyvvgOrTzilUrhz1GrpifgZWYQCiUtJIpQOy30nN3TOzZsDQ0LBFmhkui9VXH87i'
        'JPbcgFvamn6U/SNooubGeGobzm1MTVYNr8Gmra19pzH6Q7beFDJXy5BMSQCIZYeAAAgjh19a/dxTE3ffM7tlW/+JJ7Kzx9c+9I+O'
        'd2fx+GMr22x/6b5T07Mj3398aMP6ev/B8wsz7ZmplZ5qhCQfAqxJduSl8bdP6bOfL0fywo0CY6JiNTbSX7v+gtitm87k6fVkWLvx'
        'RAxdDECUJ41XOij+dpPumJ2VUUPtEctoZrgsVl99OMumsHs39uWYehQLr6I/7Bc28MQ1nB/tx7y+9s6hWz/DsceT/ZXlRyhKmSFR'
        'LmL5ovvEyy9tfv6ZzsFPz23equ89kZ85kT/08Ply3h77zvqW8UtfOTtdtH/wN9na1eH2T8+9/ELvmR92+gsdZX3QXS16NjszOtPr'
        'bFg32QnzdZBrOGuyhhf27a2/+OWCmp5ZGP/edzZm7bF77n1rODuPAYBASmtr3SM8UPu+2fmSVCcbjrktwmWx/urDMU5i9x7sy3D+'
        '6yiOY2qjzl2DolP6kK+7Odz0uXrTmhC+R/tmFp8z1UAAEiCAWHYkgfSUTV3Y+PSzG48etX03Y9368LPnmqlJfukLsSn9h0+FtuGu'
        'u5uZeXv2mXLtis5t+2fJc5OT86lqi5GsBHo1euz1iRPn6/0HmtH2bBP6zpg1WbJm7Yrm2uumyDA5M/GXj2ZZu/2V35kbbZ3HAEBF'
        '92sav7fRoUo7i+6c0dtxNMttES6L9VcfjmESu/dgX4azj2LmFM7twNlrvOJCNmTbPt+65QsL4wz+ZLBvZOHpoB5goAMCiGVGkAQi'
        'pfaJ49f8+OkNb72j3Tvz1auyw0f7RZEO7p9o6uZnL5dtwy37stmuXj4yv2Zs+LZbZtdtPOE8r6YlDRlLMaX+ipdeWHHsZPfzX7SJ'
        '4akmdJ2MKXOEobxZMTEJG5mcnvjzP63zdnbowTTavoCBRZ5LO5J/ufL7Sm2d680E83YczTJbhMti72v/aVvT2H0t9nZw8jGcWMDp'
        'zeqOdVsryt235/u+mNqrovUCXiQeNX03i9MQYXJLFCFiuaGgumr97MdDP3t+KKHFUu5VyhGzFuaGEFLdLtCkPLXcrIm1JVy31T/9'
        'ORtbdZKy4G3aHJDKcuInP1775hs69GA5PnSBbMMWTJVqY8zBMkWv3dQMsVGeJVrCxwxBQACERQQEQmnYfZ/r/tq/0GsmFsqpGNnJ'
        'R4JdhMvi/P/yj4f757BzI24YxfGf4d15L1Y1WlGv21nvvgtb9yHmubrkMeBbxu9k4V3DIsoqCIBh2RGQmnj8rdY77+aOyEZS8giz'
        'iDInPWV9uAePMrolOtau8mt3sT08Q5kpgiXgZTX6/LPj776NL/59Hxk6b+qQfaKvFBAi0CgoOeU5HSE46fiYoQgIlECBoAB4Wim/'
        'y/2BWnd0+1nZzMcYW61ODJEkLouz/9u/GJs8iW2juHYVjh3H8fmkdndkre39DPZ8ptdZGZtuJ817mHb8wOwvMnsxQPSA0BMEBCxP'
        'lNOduCKy4CR+EcumdfRIfuFkuOMzsT10NqRhokboShlAfOwRhAgIdIEiQXfQ0yZv7pU/kLBrfn7ePcWsnbWzGHMshcX/8d8Nv/Oy'
        'tfoYa2Gh8V7TH15fX/+pcMMnObFGRHK5Mw8l8JTxGzH8IHKGiqALDhDLlSBcIZHEv8dlvX5syjAynsx65gIIRZkwABCADCIoWSO6'
        'FNxH3fcqfdnxuW65ul/2CcvzdswZY8RSOPdn/3Pr9Weiz7AV1YSE0F+53T/xBV57K9vt4HXtVilrhdrwYuCfhfB45FnKAInCMidc'
        'KeJXoMMgkk7URIICvCVLoPCxRwAiFEDJajHJW+4r3W9z/03HnXPdkbqpArMsb8VMMUYshf0X/nr2x98amz3Voipr9Sc2hV132c47'
        '69HVpl6eqjJlC97Ks7qFcxF/Y+HPoh0mexQFYjkjhCtF4ZeRMrERFpBGzAxIUAeQrAKEjz1CgMPbgMlKsfE0jrQj6bPJ7q2a7TPz'
        'jZBaYSjPs5ClaBFL4fzxV7pHn8pPvBZ7c94Z8c2747abuXJDyttEClLVhF4SY9NBk+Fps29a+EmwC6ZGBEBcJEC4iFhOhCtF/Puo'
        'KCSxT7XBBBi8DesDAogBCGyYhqAo6zldaT3SjQlfaPipXrWy6FWEt+JQK48WPVqGpXB2drY6/Zpee96mztnERLju9mzzLtAdEkmE'
        'umHZVB5TB50cR8lvhvBECO8G9EUBBEAIcECCAQSEAZCKgsS+MRdKKJdasDkqAIYButhYGoZnsq4DnrbRDyR8seb++X5rod8zUzsO'
        '5VkWshSYYyksikLeoGkggWSICBEXSQJJAGXdb1ITOZKFuRCeNX438qcZT8i6QKAHQmAJJmeAjBAGfpkAAgQcIAYAEW6wJjcX4A1W'
        '1ekO4YuO26tm5Xy3KylmWR6zRTEGCUtiURT4EJIAkKzqMnmCsjxWIRwL+HHkjyJfhU2CogeCYAUmJyEjhIGBpQhwM0ukKA0lba/9'
        'k45POW7ol+1er0talmUxi3mWhxAkYSksigIfThLJum4klM18O88z1hneNj5p4UmzI4YFyMAIOSi3WnRzw8DA0owKYl9spbTL092N'
        'H0zY5Zgo5hZSSjHEPM9DDFmWBQuSsBQWRYEPJ4mkJ3dXmRbAkFsr50zkYdpThufNTpDTZI8AZG4N6JRhYGBJMqidkLk2Nr5Pukvc'
        '2/jqXk9lVRMMMeR5HheFSBJXgEVR4MMQctGohJRS5d1G1rKRnGVmb5DPG56mvW520myGIhRkCUiAYWBgCYQMPtxgNPl1jfYD+8Hr'
        'qtSZn+95kpmFGPI8j4tCxJVhURT4ECTd3cwAeWqqlOpGTWqGWqFlDTVHHWM4jPBc4BsB8+ZO1GAjazAw8AtIQZRAwKQI5O4d8lpv'
        'diffn7ATYWXdWLfbq+vEEEIMeZZnWRZCIIkrw6Io8OEkkQQgT8lVNalu6iwgD4ihpk6Tb8leC3wrYso0TyyApdjgEoIgBgYgQhST'
        'ADCXhqBhYdx1HXxH8l3gNUmdfuVlVSdPZhazmMVskZmRxJVhURS4MpLquq6qqklNlmWdTg4tUDPS+cATGU8Qp4wXoK7Q4BISJDEw'
        '4AbR2ThBDrtWyFfT1jW+07guNatjnOiVzXy/7wLoMTDP8hBClmUkccVYFAWuEJEuaRbVTQghLgoie9CMsSDnwAWqpPfxPuE9wsDH'
        'mhEUJUBsSaPQqHw4xBXuQ/JOv0zdqgcTGYIxiyHGGBZZIIkrxqIocGVodPeUUtM0VVmRjLEVYwjWSFWuwMsAAAJGSURBVF2yR/bF'
        'kqrN+7hEkIRLhIGPMYKkiRIoZMIQNCzPg7WbFKW81+/3626IZhajWZbFRcECL8EVY1EUuAIkQSzy5HVTN4vqxlOTRWu12haDGV1y'
        'l+Ck8HNahJ8jBj62KJJYJHMHEcxywLrduW63F7PMvU+mEGIIrTzmMcYQAkl8RCyKAleApCAAcjWpSYua5O6gjIaLBNJoMOF9wiJh'
        'YAAiIJAwudwFQWLV1KlJMUapJJHFPFg7xhgizYwkPiIWRYErQKNcAAR58kUpJZeSp6ZpUkpyDyHEEBlMEv4dEhcJEAY+vgQBIGWS'
        'UqpTU7skBoJZDEIdaHnWCWyHzBhkZvjoWBQFPipCLndPP9e4PCWHoEWQ8CsJAwMgcRHBRYJIxhBpCCGaGWkxBDPDVWFRFPhIiEVy'
        'LUqLPHlK7kqeILhcEoSBgQ9FLCJpNEEkQwhmFkIgabT34KqwKAp8JARJudxdkl8iKXkCIAmCu2Ng4DIILgJII2nBSAYL/ABcFRZF'
        'gY+EIKn3uH4B9HMuDAxcBsFFAGm8xC7BIoIgrhaLosBHRKMkCJIAuLskAIIWAZALAwOXR/A94HvMDP/BWBQFPhKCJC7RIhcASSQl'
        'AZBEEgMDH0ISAF4iiSQAkvi7wKIo8BGRxCWSAEjCB0giiYGBDyEJAEl8AEn8XWBRFBgYWC5YFAUGBpYLFkWBgYHlgkVRYGBguWBR'
        'FBgYWC5YFAUGBpYLFkWBgYHlgkVRYGBguWBRFBgYWC5YFAUGBpYLFkWBgYHl4v8DI+P2vq6gOmsAAAAASUVORK5CYII='
    ),
    'stamp_no_button': (
        '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'
    ),
}


class _HeartbeatFile:
    """Throttled cross-process heartbeat for the outer supervisor."""

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
                # Telemetry failure must not interrupt controller input.  If
                # it persists, the parent will recover this child process.
                pass


class FFmpegCapture:
    """Read Macro6's configured capture source and retain its latest frame."""

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

    def snapshot(self) -> FrameSnapshot:
        with self._lock:
            frame = (
                None
                if self._snapshot.frame is None
                else self._snapshot.frame.copy()
            )
            raw_frame = (
                None
                if self._snapshot.raw_frame is None
                else self._snapshot.raw_frame.copy()
            )
            return FrameSnapshot(
                self._snapshot.sequence,
                frame,
                self._snapshot.error,
                raw_frame,
            )

    def _store(self, frame: np.ndarray | None, error: str = "") -> None:
        with self._lock:
            self._snapshot = FrameSnapshot(
                self._snapshot.sequence + 1,
                None if frame is None else self._preview_frame(frame),
                error,
                frame,
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

    def start(self, worker_errors: list[BaseException]) -> None:
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
                self._store(frame)
                self.heartbeat.beat(force=True)
                latest_timestamp = source.latest_frame_ts
                while not self.stop_event.is_set():
                    frame = source.read_next(
                        after_ts=latest_timestamp,
                        timeout_seconds=max(
                            2.0,
                            settings.read_timeout_seconds,
                        ),
                    )
                    if frame is None:
                        if not self.stop_event.is_set():
                            self._store(
                                None,
                                str(
                                    source.last_error
                                    or "FFmpeg视频流停止刷新"
                                ),
                            )
                            self.stop_event.set()
                        return
                    latest_timestamp = source.latest_frame_ts
                    self._store(frame)
                    self.heartbeat.beat()
            except BaseException as exc:
                self._store(None, str(exc))
                worker_errors.append(exc)
                self.stop_event.set()

        self._thread = threading.Thread(
            target=reader,
            name="macro6-ffmpeg-capture",
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


class _MacroInterrupted(Exception):
    """Internal control-flow signal raised by the keyboard 0 command."""


class _CodeTimerExpired(Exception):
    """Restart automatic macro1 after 15 minutes without a CODE update."""


class _BusinessDateChanged(Exception):
    """Restart automatic macro1 when Beijing's 05:00 day changes."""


class _BusinessDateHold(Exception):
    """Freeze controller activity at CURSOR during the 04:50 hold window."""


class _SevereCodeTimerExpired(Exception):
    """Restart automatic macro1 after 30 minutes regardless of cursor."""


class _CodeRecognitionTimedOut(Exception):
    """Restart after a visible CODE panel yields no new six-character CODE."""


class _NetworkConnectionError(Exception):
    """Restart an open round after the Switch connection-error popup."""


class _NeverExpireCodeTimer:
    """Use shared STAMP navigation without enabling Macro1's 15m restart."""

    @staticmethod
    def timer_expired() -> bool:
        return False


_NEVER_EXPIRE_CODE_TIMER = _NeverExpireCodeTimer()


REOPEN_REASON_TEXT = {
    "normal_reopen": "正常重开",
    "network_connection_error": "网络异常重开",
    "date_change_reopen": "日期切换重开",
}


def _reopen_reason_text(reason: str) -> str:
    return REOPEN_REASON_TEXT.get(str(reason), str(reason))


class _CursorGatedCodeTimer:
    """Combine regular, business-date, and severe restart triggers."""

    def __init__(
        self,
        code_recognizer: CodeRecognizer,
        visual_state: _PokopiaVisualState,
    ) -> None:
        self._code_recognizer = code_recognizer
        self._visual_state = visual_state
        self._armed = True
        self.last_trigger_cursor = "INVALID"
        self.last_trigger_reason = ""
        self._observed_operational_date = _beijing_operational_date()
        self._last_code_revision = code_recognizer.revision()
        network_visible, network_revision = (
            visual_state.network_error_state()
        )
        self._last_network_error_revision = network_revision - int(
            network_visible
        )
        elapsed = code_recognizer.timer_elapsed_seconds()
        self._severe_anchor_monotonic = time.monotonic() - (
            elapsed if elapsed is not None else 0.0
        )

    def arm(self) -> None:
        """Enable the timer after a CODE is ready for stamp checking."""
        # A business-day boundary crossed while read/load/reopen was
        # disarmed has already occurred inside an existing reopen and must
        # not inject another reopen immediately after the new CODE.
        self._observed_operational_date = _beijing_operational_date()
        self._armed = True
        self.last_trigger_cursor = "INVALID"
        _, self._last_network_error_revision = (
            self._visual_state.network_error_state()
        )

    def disarm(self) -> None:
        """Suppress stale timer expiry throughout read/load/restart work."""
        self._armed = False

    def timer_expired(self) -> bool:
        # A complete read/load/reopen is already the recovery action.  While
        # disarmed, no timer-derived exception may interrupt it and inject a
        # second restart.  arm() consumes any date boundary crossed during
        # that existing reopen before enabling checks again.
        if not self._armed:
            return False

        current_date = _beijing_operational_date()
        if current_date != self._observed_operational_date:
            self._observed_operational_date = current_date
            self.last_trigger_reason = "business_date_changed"
            return True

        if (
            _in_business_date_hold_window()
            and self._visual_state.immediate_cursor() != "INVALID"
        ):
            self.last_trigger_cursor = self._visual_state.immediate_cursor()
            self.last_trigger_reason = "business_date_hold"
            return True

        _, network_revision = self._visual_state.network_error_state()
        if network_revision > self._last_network_error_revision:
            self._last_network_error_revision = network_revision
            self.last_trigger_reason = "network_connection_error"
            return True

        revision = self._code_recognizer.revision()
        if revision != self._last_code_revision:
            self._last_code_revision = revision
            elapsed = self._code_recognizer.timer_elapsed_seconds()
            self._severe_anchor_monotonic = time.monotonic() - (
                elapsed if elapsed is not None else 0.0
            )

        now = time.monotonic()
        if (
            now - self._severe_anchor_monotonic
            >= SEVERE_CODE_TIMER_RESTART_SECONDS
        ):
            # Start a fresh severe-watchdog window for the restart attempt.
            self._severe_anchor_monotonic = now
            self.last_trigger_reason = "severe_timeout"
            return True

        if not self._code_recognizer.timer_expired():
            return False
        cursor = self._visual_state.cursor()
        if cursor == "INVALID":
            return False
        self.last_trigger_cursor = cursor
        self.last_trigger_reason = "cursor_timeout"
        return True


class _NetworkConnectionErrorGuard:
    """Expose only a new connection-error popup through the shared guard."""

    def __init__(self, visual_state: _PokopiaVisualState) -> None:
        self._visual_state = visual_state
        self._armed = True
        self.last_trigger_reason = ""
        visible, revision = visual_state.network_error_state()
        self._last_revision = revision - int(visible)

    def arm(self) -> None:
        self._armed = True
        _, self._last_revision = self._visual_state.network_error_state()

    def disarm(self) -> None:
        self._armed = False

    def timer_expired(self) -> bool:
        _, revision = self._visual_state.network_error_state()
        if not self._armed or revision <= self._last_revision:
            return False
        self._last_revision = revision
        self.last_trigger_reason = "network_connection_error"
        return True


@dataclass(frozen=True)
class RunCounterSnapshot:
    operational_date: str
    daily_runs: int
    total_runs: int
    daily_player_entries: int
    total_player_entries: int
    daily_player_entry_failures: int
    total_player_entry_failures: int


class Macro6RunCounter:
    """Persistent CODE updates grouped by Beijing 05:00 operating days."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self._lock = threading.Lock()
        self._daily_runs: dict[str, int] = {}
        self._total_runs = 0
        self._daily_player_entries: dict[str, int] = {}
        self._total_player_entries = 0
        self._daily_player_entry_failures: dict[str, int] = {}
        self._total_player_entry_failures = 0
        self._daily_player_visits: dict[str, dict[str, int]] = {}
        self._total_player_visits: dict[str, int] = {}
        self._daily_player_completed_tasks: dict[str, dict[str, int]] = {}
        self._total_player_completed_tasks: dict[str, int] = {}
        self._daily_player_returned_before_arrival: dict[
            str, dict[str, int]
        ] = {}
        self._total_player_returned_before_arrival: dict[str, int] = {}
        self._daily_player_room_closed_before_arrival: dict[
            str, dict[str, int]
        ] = {}
        self._total_player_room_closed_before_arrival: dict[str, int] = {}
        self._daily_player_network_error_before_arrival: dict[
            str, dict[str, int]
        ] = {}
        self._total_player_network_error_before_arrival: dict[str, int] = {}
        self._load()

    @staticmethod
    def _date_key() -> str:
        return _beijing_operational_date()

    def _load(self) -> None:
        if not self.path.is_file():
            return
        try:
            raw = json.loads(self.path.read_text(encoding="utf-8"))
            if raw.get("metric") != "completed_macro_code_updates":
                _timestamped_log(
                    "检测到旧版CODE计数；新规则只统计宏执行后的新CODE，"
                    "旧计数不会并入新计数。"
                )
                return
            total_runs = max(0, int(raw.get("total_runs", 0)))
            daily_raw = raw.get("daily_runs", {})
            if not isinstance(daily_raw, dict):
                raise ValueError("daily_runs must be an object")
            daily_runs = {
                str(date): max(0, int(count))
                for date, count in daily_raw.items()
            }
            self._daily_runs = daily_runs
            self._total_runs = max(total_runs, sum(daily_runs.values()))
            daily_player_raw = raw.get("daily_player_entries", {})
            if not isinstance(daily_player_raw, dict):
                raise ValueError("daily_player_entries must be an object")
            self._daily_player_entries = {
                str(date): max(0, int(count))
                for date, count in daily_player_raw.items()
            }
            self._total_player_entries = max(
                0,
                int(raw.get("total_player_entries", 0)),
                sum(self._daily_player_entries.values()),
            )
            daily_failure_raw = raw.get("daily_player_entry_failures", {})
            if not isinstance(daily_failure_raw, dict):
                raise ValueError(
                    "daily_player_entry_failures must be an object"
                )
            self._daily_player_entry_failures = {
                str(date): max(0, int(count))
                for date, count in daily_failure_raw.items()
            }
            self._total_player_entry_failures = max(
                0,
                int(raw.get("total_player_entry_failures", 0)),
                sum(self._daily_player_entry_failures.values()),
            )

            def nested_player_counts(key: str) -> dict[str, dict[str, int]]:
                value = raw.get(key, {})
                if not isinstance(value, dict):
                    raise ValueError(f"{key} must be an object")
                result: dict[str, dict[str, int]] = {}
                for date, players in value.items():
                    if not isinstance(players, dict):
                        continue
                    cleaned = {
                        str(name): max(0, int(count))
                        for name, count in players.items()
                        if str(name).strip()
                    }
                    if cleaned:
                        result[str(date)] = cleaned
                return result

            def total_player_counts(
                key: str,
                daily: dict[str, dict[str, int]],
            ) -> dict[str, int]:
                value = raw.get(key, {})
                if not isinstance(value, dict):
                    raise ValueError(f"{key} must be an object")
                stored = {
                    str(name): max(0, int(count))
                    for name, count in value.items()
                    if str(name).strip()
                }
                summed: dict[str, int] = {}
                for players in daily.values():
                    for name, count in players.items():
                        summed[name] = summed.get(name, 0) + count
                return {
                    name: max(stored.get(name, 0), summed.get(name, 0))
                    for name in set(stored) | set(summed)
                }

            self._daily_player_visits = nested_player_counts(
                "daily_player_visits"
            )
            self._total_player_visits = total_player_counts(
                "total_player_visits",
                self._daily_player_visits,
            )
            self._daily_player_completed_tasks = nested_player_counts(
                "daily_player_completed_tasks"
            )
            self._total_player_completed_tasks = total_player_counts(
                "total_player_completed_tasks",
                self._daily_player_completed_tasks,
            )
            self._daily_player_returned_before_arrival = nested_player_counts(
                "daily_player_returned_before_arrival"
            )
            self._total_player_returned_before_arrival = total_player_counts(
                "total_player_returned_before_arrival",
                self._daily_player_returned_before_arrival,
            )
            self._daily_player_room_closed_before_arrival = (
                nested_player_counts(
                    "daily_player_room_closed_before_arrival"
                )
            )
            self._total_player_room_closed_before_arrival = (
                total_player_counts(
                    "total_player_room_closed_before_arrival",
                    self._daily_player_room_closed_before_arrival,
                )
            )
            self._daily_player_network_error_before_arrival = (
                nested_player_counts(
                    "daily_player_network_error_before_arrival"
                )
            )
            self._total_player_network_error_before_arrival = (
                total_player_counts(
                    "total_player_network_error_before_arrival",
                    self._daily_player_network_error_before_arrival,
                )
            )
            migration_needed = False
            if not any(
                key in raw
                for key in (
                    "daily_player_visits",
                    "total_player_visits",
                    "daily_player_completed_tasks",
                    "total_player_completed_tasks",
                )
            ):
                self._backfill_player_rankings_from_room_logs()
                migration_needed = True
            if not any(
                key in raw
                for key in (
                    "daily_player_returned_before_arrival",
                    "total_player_returned_before_arrival",
                    "daily_player_room_closed_before_arrival",
                    "total_player_room_closed_before_arrival",
                )
            ):
                self._backfill_player_failure_rankings_from_room_logs()
                migration_needed = True
            if not any(
                key in raw
                for key in (
                    "daily_player_network_error_before_arrival",
                    "total_player_network_error_before_arrival",
                )
            ):
                # Older summaries did not retain enough information to split
                # network-error freezes from ordinary reopen freezes.  Start
                # the new category at zero without rewriting old statistics.
                migration_needed = True
            if migration_needed:
                self._save_locked()
        except Exception as exc:
            _timestamped_log(
                f"macro6 永久计数文件读取失败，将从0继续统计：{exc}"
            )

    def _backfill_player_rankings_from_room_logs(self) -> None:
        """One-time v7 migration from existing per-reopen player summaries."""
        archive_root = self.path.with_name(CODE_ARCHIVE_DIRNAME)
        if not archive_root.is_dir():
            return
        migrated_visits = 0
        migrated_tasks = 0
        for record_path in sorted(
            archive_root.glob("STAMP_*/PLAYERS_*.jsonl")
        ):
            folder_name = record_path.parent.name
            if not folder_name.startswith("STAMP_"):
                continue
            date_key = folder_name.removeprefix("STAMP_")
            try:
                with record_path.open("r", encoding="utf-8-sig") as source:
                    records = tuple(source)
            except OSError:
                continue
            for line in records:
                try:
                    record = json.loads(line)
                except (TypeError, ValueError, json.JSONDecodeError):
                    continue
                if (
                    not isinstance(record, dict)
                    or record.get("status") != "room_reopen_summary"
                ):
                    continue
                visits = record.get("players", ())
                if not isinstance(visits, list):
                    continue
                for visit in visits:
                    if (
                        not isinstance(visit, dict)
                        or visit.get("result") != "entered"
                    ):
                        continue
                    player_name = str(visit.get("player_name", "")).strip()
                    if not player_name:
                        continue
                    daily_visits = self._daily_player_visits.setdefault(
                        date_key,
                        {},
                    )
                    daily_visits[player_name] = (
                        daily_visits.get(player_name, 0) + 1
                    )
                    migrated_visits += 1
                    try:
                        completed_tasks = min(
                            3,
                            max(
                                0,
                                int(
                                    visit.get(
                                        "available_reward_tasks",
                                        visit.get(
                                            "completed_reward_tasks",
                                            0,
                                        ),
                                    )
                                ),
                            ),
                        )
                    except (TypeError, ValueError):
                        completed_tasks = 0
                    if completed_tasks:
                        daily_tasks = (
                            self._daily_player_completed_tasks.setdefault(
                                date_key,
                                {},
                            )
                        )
                        daily_tasks[player_name] = (
                            daily_tasks.get(player_name, 0) + completed_tasks
                        )
                        migrated_tasks += completed_tasks

        for players in self._daily_player_visits.values():
            for player_name, count in players.items():
                self._total_player_visits[player_name] = (
                    self._total_player_visits.get(player_name, 0) + count
                )
        for players in self._daily_player_completed_tasks.values():
            for player_name, count in players.items():
                self._total_player_completed_tasks[player_name] = (
                    self._total_player_completed_tasks.get(player_name, 0)
                    + count
                )
        _timestamped_log(
            "玩家排行统计已从既有每轮日志完成一次性迁移："
            f"来访{migrated_visits}次，参与完成任务{migrated_tasks}次。"
        )

    def _backfill_player_failure_rankings_from_room_logs(self) -> None:
        """One-time v8 migration of named failures from room summaries."""
        archive_root = self.path.with_name(CODE_ARCHIVE_DIRNAME)
        if not archive_root.is_dir():
            return
        returned_count = 0
        closed_count = 0
        network_error_count = 0
        for record_path in sorted(
            archive_root.glob("STAMP_*/PLAYERS_*.jsonl")
        ):
            folder_name = record_path.parent.name
            if not folder_name.startswith("STAMP_"):
                continue
            date_key = folder_name.removeprefix("STAMP_")
            try:
                with record_path.open("r", encoding="utf-8-sig") as source:
                    records = tuple(source)
            except OSError:
                continue
            for line in records:
                try:
                    record = json.loads(line)
                except (TypeError, ValueError, json.JSONDecodeError):
                    continue
                if (
                    not isinstance(record, dict)
                    or record.get("status") != "room_reopen_summary"
                ):
                    continue
                visits = record.get("players", ())
                if not isinstance(visits, list):
                    continue
                for visit in visits:
                    if (
                        not isinstance(visit, dict)
                        or visit.get("result") != "entry_failed"
                    ):
                        continue
                    player_name = str(visit.get("player_name", "")).strip()
                    if not player_name:
                        continue
                    failure_reason = str(visit.get("failure_reason") or "")
                    if failure_reason == "network_error_before_arrival":
                        daily = self._daily_player_network_error_before_arrival
                        network_error_count += 1
                    elif bool(visit.get("ended_by_room_freeze", False)):
                        daily = self._daily_player_room_closed_before_arrival
                        closed_count += 1
                    else:
                        daily = self._daily_player_returned_before_arrival
                        returned_count += 1
                    players = daily.setdefault(date_key, {})
                    players[player_name] = players.get(player_name, 0) + 1

        for players in self._daily_player_returned_before_arrival.values():
            for player_name, count in players.items():
                self._total_player_returned_before_arrival[player_name] = (
                    self._total_player_returned_before_arrival.get(
                        player_name,
                        0,
                    )
                    + count
                )
        for players in self._daily_player_room_closed_before_arrival.values():
            for player_name, count in players.items():
                self._total_player_room_closed_before_arrival[player_name] = (
                    self._total_player_room_closed_before_arrival.get(
                        player_name,
                        0,
                    )
                    + count
                )
        for players in self._daily_player_network_error_before_arrival.values():
            for player_name, count in players.items():
                self._total_player_network_error_before_arrival[player_name] = (
                    self._total_player_network_error_before_arrival.get(
                        player_name,
                        0,
                    )
                    + count
                )
        _timestamped_log(
            "玩家失败排行已从既有每轮日志完成一次性迁移："
            f"到达前返回{returned_count}次，"
            f"房间关闭时未抵达{closed_count}次，"
            f"网络连接错误时未抵达{network_error_count}次。"
        )

    def _save_locked(self) -> None:
        payload = {
            "version": 9,
            "metric": "completed_macro_code_updates",
            "timezone": "Asia/Shanghai (UTC+08:00)",
            "day_boundary": "05:00",
            "total_runs": self._total_runs,
            "daily_runs": dict(sorted(self._daily_runs.items())),
            "total_player_entries": self._total_player_entries,
            "daily_player_entries": dict(
                sorted(self._daily_player_entries.items())
            ),
            "total_player_entry_failures": (
                self._total_player_entry_failures
            ),
            "daily_player_entry_failures": dict(
                sorted(self._daily_player_entry_failures.items())
            ),
            "total_player_visits": dict(
                sorted(self._total_player_visits.items())
            ),
            "daily_player_visits": {
                date: dict(sorted(players.items()))
                for date, players in sorted(self._daily_player_visits.items())
            },
            "total_player_completed_tasks": dict(
                sorted(self._total_player_completed_tasks.items())
            ),
            "daily_player_completed_tasks": {
                date: dict(sorted(players.items()))
                for date, players in sorted(
                    self._daily_player_completed_tasks.items()
                )
            },
            "total_player_returned_before_arrival": dict(
                sorted(self._total_player_returned_before_arrival.items())
            ),
            "daily_player_returned_before_arrival": {
                date: dict(sorted(players.items()))
                for date, players in sorted(
                    self._daily_player_returned_before_arrival.items()
                )
            },
            "total_player_room_closed_before_arrival": dict(
                sorted(self._total_player_room_closed_before_arrival.items())
            ),
            "daily_player_room_closed_before_arrival": {
                date: dict(sorted(players.items()))
                for date, players in sorted(
                    self._daily_player_room_closed_before_arrival.items()
                )
            },
            "total_player_network_error_before_arrival": dict(
                sorted(self._total_player_network_error_before_arrival.items())
            ),
            "daily_player_network_error_before_arrival": {
                date: dict(sorted(players.items()))
                for date, players in sorted(
                    self._daily_player_network_error_before_arrival.items()
                )
            },
            "updated_at_utc": datetime.now(timezone.utc)
            .isoformat(timespec="seconds")
            .replace("+00:00", "Z"),
        }
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary_path = self.path.with_suffix(self.path.suffix + ".tmp")
        temporary_path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        temporary_path.replace(self.path)

    def snapshot(self) -> RunCounterSnapshot:
        date_key = self._date_key()
        with self._lock:
            return RunCounterSnapshot(
                operational_date=date_key,
                daily_runs=self._daily_runs.get(date_key, 0),
                total_runs=self._total_runs,
                daily_player_entries=self._daily_player_entries.get(
                    date_key,
                    0,
                ),
                total_player_entries=self._total_player_entries,
                daily_player_entry_failures=(
                    self._daily_player_entry_failures.get(date_key, 0)
                ),
                total_player_entry_failures=(
                    self._total_player_entry_failures
                ),
            )

    def record_code_update(self) -> RunCounterSnapshot:
        date_key = self._date_key()
        with self._lock:
            self._daily_runs[date_key] = self._daily_runs.get(date_key, 0) + 1
            self._total_runs += 1
            snapshot = RunCounterSnapshot(
                operational_date=date_key,
                daily_runs=self._daily_runs[date_key],
                total_runs=self._total_runs,
                daily_player_entries=self._daily_player_entries.get(
                    date_key,
                    0,
                ),
                total_player_entries=self._total_player_entries,
                daily_player_entry_failures=(
                    self._daily_player_entry_failures.get(date_key, 0)
                ),
                total_player_entry_failures=(
                    self._total_player_entry_failures
                ),
            )
            try:
                self._save_locked()
            except OSError as exc:
                _timestamped_log(f"macro6 永久计数保存失败：{exc}")
            return snapshot

    def record_player_entered(self, player_name: str) -> RunCounterSnapshot:
        """Persist one distinct successful player arrival."""
        date_key = self._date_key()
        normalized_name = str(player_name).strip()
        with self._lock:
            self._daily_player_entries[date_key] = (
                self._daily_player_entries.get(date_key, 0) + 1
            )
            self._total_player_entries += 1
            if normalized_name:
                daily_players = self._daily_player_visits.setdefault(
                    date_key,
                    {},
                )
                daily_players[normalized_name] = (
                    daily_players.get(normalized_name, 0) + 1
                )
                self._total_player_visits[normalized_name] = (
                    self._total_player_visits.get(normalized_name, 0) + 1
                )
            snapshot = RunCounterSnapshot(
                operational_date=date_key,
                daily_runs=self._daily_runs.get(date_key, 0),
                total_runs=self._total_runs,
                daily_player_entries=self._daily_player_entries[date_key],
                total_player_entries=self._total_player_entries,
                daily_player_entry_failures=(
                    self._daily_player_entry_failures.get(date_key, 0)
                ),
                total_player_entry_failures=(
                    self._total_player_entry_failures
                ),
            )
            try:
                self._save_locked()
            except OSError as exc:
                _timestamped_log(f"玩家进入永久计数保存失败：{exc}")
            return snapshot

    def record_player_completed_tasks(
        self,
        player_task_counts: dict[str, int],
    ) -> None:
        """Persist completed task participation for every arrived player."""
        increments = {
            str(name).strip(): max(0, int(count))
            for name, count in player_task_counts.items()
            if str(name).strip() and int(count) > 0
        }
        if not increments:
            return
        date_key = self._date_key()
        with self._lock:
            daily_players = self._daily_player_completed_tasks.setdefault(
                date_key,
                {},
            )
            for name, count in increments.items():
                daily_players[name] = daily_players.get(name, 0) + count
                self._total_player_completed_tasks[name] = (
                    self._total_player_completed_tasks.get(name, 0) + count
                )
            try:
                self._save_locked()
            except OSError as exc:
                _timestamped_log(f"玩家参与任务永久计数保存失败：{exc}")

    def record_player_entry_failures(
        self,
        amount: int = 1,
        *,
        player_names: tuple[str, ...] = (),
        reason: str = "",
    ) -> RunCounterSnapshot:
        """Persist failed player visits that ended before arrival."""
        increment = max(0, int(amount))
        normalized_names = tuple(
            name
            for name in (str(item).strip() for item in player_names)
            if name
        )
        date_key = self._date_key()
        with self._lock:
            self._daily_player_entry_failures[date_key] = (
                self._daily_player_entry_failures.get(date_key, 0) + increment
            )
            self._total_player_entry_failures += increment
            if reason == "left_before_arrival":
                daily_by_name = self._daily_player_returned_before_arrival
                total_by_name = self._total_player_returned_before_arrival
            elif reason == "room_closed_before_arrival":
                daily_by_name = self._daily_player_room_closed_before_arrival
                total_by_name = self._total_player_room_closed_before_arrival
            elif reason == "network_error_before_arrival":
                daily_by_name = self._daily_player_network_error_before_arrival
                total_by_name = self._total_player_network_error_before_arrival
            else:
                daily_by_name = None
                total_by_name = None
            if daily_by_name is not None and total_by_name is not None:
                daily_players = daily_by_name.setdefault(date_key, {})
                for player_name in normalized_names:
                    daily_players[player_name] = (
                        daily_players.get(player_name, 0) + 1
                    )
                    total_by_name[player_name] = (
                        total_by_name.get(player_name, 0) + 1
                    )
            snapshot = RunCounterSnapshot(
                operational_date=date_key,
                daily_runs=self._daily_runs.get(date_key, 0),
                total_runs=self._total_runs,
                daily_player_entries=self._daily_player_entries.get(
                    date_key,
                    0,
                ),
                total_player_entries=self._total_player_entries,
                daily_player_entry_failures=(
                    self._daily_player_entry_failures[date_key]
                ),
                total_player_entry_failures=(
                    self._total_player_entry_failures
                ),
            )
            try:
                self._save_locked()
            except OSError as exc:
                _timestamped_log(f"玩家进入失败永久计数保存失败：{exc}")
            return snapshot


@dataclass(frozen=True)
class _PreviousRoundResult:
    outcome: str
    elapsed_seconds: int | None = None


def _announcement_deadline(now: datetime) -> datetime:
    """Cap the normal 15-minute deadline at the next Beijing 05:00."""
    normal_deadline = now + timedelta(seconds=CODE_TIMER_RESTART_SECONDS)
    next_day_boundary = now.replace(
        hour=DAILY_BOUNDARY_HOUR_BEIJING,
        minute=0,
        second=0,
        microsecond=0,
    )
    if now >= next_day_boundary:
        next_day_boundary += timedelta(days=1)
    return min(normal_deadline, next_day_boundary)


class _RoundAnnouncementTracker:
    """Build Xiaohongshu text and retain the preceding round's result."""

    def __init__(self) -> None:
        self._round_started_monotonic: float | None = None
        self._restart_started_monotonic: float | None = None
        self._previous_result: _PreviousRoundResult | None = None

    def ensure_restart_started(self) -> None:
        """Start timing a full read/load/reopen sequence exactly once."""
        if self._restart_started_monotonic is None:
            self._restart_started_monotonic = time.monotonic()

    def mark_stamp_full(self) -> None:
        started_at = self._round_started_monotonic
        if started_at is None:
            self._previous_result = _PreviousRoundResult("filled_unknown")
        else:
            self._previous_result = _PreviousRoundResult(
                "filled",
                max(0, round(time.monotonic() - started_at)),
            )
        self._round_started_monotonic = None
        self._restart_started_monotonic = time.monotonic()

    def mark_timeout(self) -> None:
        self._previous_result = _PreviousRoundResult("timeout")
        self._round_started_monotonic = None
        self._restart_started_monotonic = time.monotonic()

    def mark_business_date_changed(self) -> None:
        started_at = self._round_started_monotonic
        elapsed_seconds = (
            max(0, round(time.monotonic() - started_at))
            if started_at is not None
            else 0
        )
        self._previous_result = _PreviousRoundResult(
            "business_date_changed",
            elapsed_seconds,
        )
        self._round_started_monotonic = None
        self._restart_started_monotonic = time.monotonic()

    def mark_severe_timeout(self) -> None:
        self._previous_result = _PreviousRoundResult("severe_timeout")
        self._round_started_monotonic = None
        self._restart_started_monotonic = time.monotonic()

    def mark_network_connection_error(
        self,
        fallback_elapsed_seconds: float | None = None,
    ) -> None:
        """Close the round with its elapsed time after a network popup."""
        started_at = self._round_started_monotonic
        if started_at is not None:
            elapsed_seconds = max(
                0,
                round(time.monotonic() - started_at),
            )
        else:
            elapsed_seconds = max(
                0,
                round(float(fallback_elapsed_seconds or 0.0)),
            )
        self._previous_result = _PreviousRoundResult(
            "network_connection_error",
            elapsed_seconds,
        )
        self._round_started_monotonic = None
        self._restart_started_monotonic = time.monotonic()

    def begin_round(
        self,
        code: str,
        daily_round: int,
        previous_completed_tasks: int,
        previous_player_entries: int,
    ) -> str:
        now = datetime.now(BEIJING_TIMEZONE)
        deadline = _announcement_deadline(now)
        previous = self._previous_result
        if previous is None:
            previous_line = "（首次接入当前轮次，暂无上轮完整记录）"
        elif previous.outcome == "timeout":
            previous_line = "（上轮15分钟未盖满，自动开启新一轮）"
        elif previous.outcome == "business_date_changed":
            elapsed = max(0, int(previous.elapsed_seconds or 0))
            minutes, seconds = divmod(elapsed, 60)
            previous_line = (
                "（日期变更，自动重开。上轮共开放时间"
                f"{minutes}分{seconds}秒）"
            )
        elif previous.outcome == "severe_timeout":
            previous_line = (
                "（严重超时，自动重开。）"
            )
        elif previous.outcome == "network_connection_error":
            elapsed = max(0, int(previous.elapsed_seconds or 0))
            minutes, seconds = divmod(elapsed, 60)
            previous_line = (
                f"（上轮开放时长{minutes}分{seconds}秒，"
                "发生网络波动断开，重新开启新的一轮）"
            )
        elif previous.outcome == "filled" and previous.elapsed_seconds is not None:
            minutes, seconds = divmod(previous.elapsed_seconds, 60)
            previous_line = (
                f"（上轮共花费{minutes}分{seconds}秒盖满，自动开启新一轮）"
            )
        else:
            previous_line = "（上轮已盖满，未记录到完整耗时，自动开启新一轮）"

        task_count = min(3, max(0, int(previous_completed_tasks)))
        player_count = max(0, int(previous_player_entries))
        previous_line = (
            previous_line[:-1]
            + f"，上轮任务完成情况{task_count}/3"
            + f"，上轮共服务百变怪{player_count}只）"
        )

        restart_started_at = self._restart_started_monotonic
        if restart_started_at is None:
            restart_line = "（此次为中途接入，未记录重开耗时）"
        else:
            restart_seconds = max(
                0,
                round(time.monotonic() - restart_started_at),
            )
            restart_minutes, restart_remainder = divmod(restart_seconds, 60)
            restart_line = (
                f"（此次重开耗时{restart_minutes}分"
                f"{restart_remainder}秒）"
            )

        self._previous_result = None
        self._restart_started_monotonic = None
        self._round_started_monotonic = time.monotonic()
        return "\n".join(
            (
                "[满当当集章的城镇-梦幻章车播报]",
                f"现在是：[{now:%Y-%m-%d %H:%M:%S}]",
                f"当前轮次：[{code}]（今日第{max(1, daily_round)}轮）",
                previous_line,
                restart_line,
                f"预计开到盖满或[{deadline:%Y-%m-%d %H:%M:%S}]",
                "大家把握好时间！谨防两车之间的间隙夹人！"
                "如果确认无法进入，请私聊反馈对应密语，感谢各位协助！",
                "大家也可在页面上传自己的开门码",
            )
        )


def _log_run_counter(snapshot: RunCounterSnapshot) -> None:
    _timestamped_log(
        f"CODE 更新次数｜北京时间早5点业务日 {snapshot.operational_date}｜"
        f"当日 {snapshot.daily_runs} 次｜总计 {snapshot.total_runs} 次；"
        f"进入玩家｜本日 {snapshot.daily_player_entries} 人｜"
        f"总计 {snapshot.total_player_entries} 人；"
        f"进入失败｜本日 {snapshot.daily_player_entry_failures} 人｜"
        f"总计 {snapshot.total_player_entry_failures} 人。"
    )


def _embedded_code_chime_wav() -> bytes:
    """Decode the bundled SoundReality notification WAV once per process."""
    global _CODE_CHIME_WAV_BYTES
    if _CODE_CHIME_WAV_BYTES is None:
        _CODE_CHIME_WAV_BYTES = zlib.decompress(
            base64.b64decode(_EMBEDDED_CODE_CHIME_WAV_ZLIB_BASE64)
        )
    return _CODE_CHIME_WAV_BYTES


def _play_code_change_chime() -> None:
    """Play the real 8.04-second ding-dong three times asynchronously."""
    if os.name != "nt":
        return

    def worker() -> None:
        with _CODE_CHIME_LOCK:
            try:
                audio = _embedded_code_chime_wav()
                for _ in range(CODE_CHIME_REPEAT_COUNT):
                    winsound.PlaySound(audio, winsound.SND_MEMORY)
                _timestamped_log(
                    "CODE 变化提示音已播放：真实叮咚铃声×3。"
                )
            except Exception as exc:
                _timestamped_log(f"CODE 变化提示音播放失败：{exc}")

    threading.Thread(
        target=worker,
        name="macro6-code-chime",
        daemon=True,
    ).start()


class _PokopiaVisualState:
    """Thread-safe visual flags shared by preview and macro workers."""

    def __init__(self) -> None:
        self.save_finished_event = threading.Event()
        self._lock = threading.Lock()
        self._cursor = "INVALID"
        self._immediate_cursor = "INVALID"
        self._last_valid_cursor_monotonic = 0.0
        self._stamp = "INVALID"
        self._reward = False
        self._reward_lines = 0
        self._completed_tasks = 0
        self._code_panel = False
        self._able_access = False
        self._black_screen = False
        self._connect_ok = False
        self._network_error = False
        self._network_error_revision = 0
        self._reopen_stage = "INVALID"
        self._reward_archive: StampCodeArchive | None = None
        self._room_player_tracker: RoomPlayerTracker | None = None

    def set_reward_archive(self, archive: StampCodeArchive) -> None:
        self._reward_archive = archive

    def set_room_player_tracker(self, tracker: RoomPlayerTracker) -> None:
        self._room_player_tracker = tracker

    def update(self, detection: PokopiaDetection) -> None:
        if detection.save_finished:
            self.save_finished_event.set()
        else:
            self.save_finished_event.clear()
        now = time.monotonic()
        with self._lock:
            self._immediate_cursor = detection.cursor
            if detection.cursor != "INVALID":
                self._cursor = detection.cursor
                self._last_valid_cursor_monotonic = now
            elif (
                now - self._last_valid_cursor_monotonic
                >= CURSOR_INVALID_GRACE_SECONDS
            ):
                self._cursor = "INVALID"
            self._stamp = detection.stamp
            self._reward = detection.reward
            if detection.reward:
                # Preserve the strongest observation across the continuous
                # page so one transition frame cannot turn a three-row page
                # into the one-row fallback immediately before B is pressed.
                self._reward_lines = max(
                    self._reward_lines,
                    detection.reward_lines,
                )
            else:
                self._reward_lines = 0
            self._code_panel = detection.code_panel
            self._able_access = detection.able_access
            self._black_screen = detection.black_screen
            self._connect_ok = detection.connect_ok
            if detection.network_error and not self._network_error:
                self._network_error_revision += 1
            self._network_error = detection.network_error
            self._reopen_stage = detection.reopen_stage

    def cursor(self) -> str:
        with self._lock:
            return self._cursor

    def immediate_cursor(self) -> str:
        """Return this frame's cursor without the INVALID grace period."""
        with self._lock:
            return self._immediate_cursor

    def stamp(self) -> str:
        with self._lock:
            return self._stamp

    def able_access(self) -> bool:
        with self._lock:
            return self._able_access

    def black_screen(self) -> bool:
        with self._lock:
            return self._black_screen

    def connect_ok(self) -> bool:
        with self._lock:
            return self._connect_ok

    def network_error_state(self) -> tuple[bool, int]:
        """Return current visibility and a rising-edge revision counter."""
        with self._lock:
            return self._network_error, self._network_error_revision

    def reopen_stage(self) -> str:
        with self._lock:
            return self._reopen_stage

    def code_panel(self) -> bool:
        with self._lock:
            return self._code_panel

    def cursor_and_reward(self) -> tuple[str, bool]:
        with self._lock:
            return self._cursor, self._reward

    def reward_lines(self) -> int:
        with self._lock:
            return self._reward_lines

    def add_completed_tasks(self, lines: int) -> tuple[int, int]:
        """Add one REWARD page once, capped at the three daily tasks."""
        with self._lock:
            added = min(max(0, int(lines)), 3 - self._completed_tasks)
            self._completed_tasks += added
            completed = self._completed_tasks
            archive = self._reward_archive
        if archive is not None:
            archive.record_reward_tasks(
                detected_lines=max(0, int(lines)),
                added_lines=added,
                completed_tasks=completed,
            )
        tracker = self._room_player_tracker
        if tracker is not None and added > 0:
            tracker.record_reward(added)
        return added, completed

    def completed_tasks(self) -> int:
        with self._lock:
            return self._completed_tasks

    def reset_completed_tasks(self) -> None:
        with self._lock:
            self._completed_tasks = 0

    def take_completed_tasks_for_new_code(self) -> int:
        """Return the preceding round's count and initialize the new round."""
        with self._lock:
            completed = self._completed_tasks
            self._completed_tasks = 0
            return completed

    def screen_state(self) -> tuple[bool, str, bool, str]:
        with self._lock:
            return (
                self._code_panel,
                self._cursor,
                self._reward,
                self._stamp,
            )


class _HeartbeatMacroContext(_MacroContext):
    """Macro context that proves the controller worker is still progressing."""

    def __init__(
        self,
        controller: SerialRemoteController,
        stop_event: threading.Event,
        pause_state: _PauseState,
        heartbeat: _HeartbeatFile,
    ) -> None:
        super().__init__(controller, stop_event, pause_state)
        self.macro6_part_triggers = {
            part_number: threading.Event() for part_number in (1, 2, 3, 4)
        }
        self.heartbeat = heartbeat
        self.macro_interrupt_event = threading.Event()
        self.macro_running_event = threading.Event()
        self.macro_trigger_lock = threading.Lock()
        self.operation_lock_event = threading.Event()
        self.manual_screenshot_event = threading.Event()
        self._reopen_screenshot_callback = None
        self._active_macro_key = 0
        self._automatic_restart_guard: _CursorGatedCodeTimer | None = None
        self._room_player_tracker: RoomPlayerTracker | None = None
        self._disconnect_on_next_home_reason = ""
        self._controller_activity_lock = threading.Lock()
        self._last_controller_input_monotonic = time.monotonic()
        self._idle_keepalive_in_progress = False
        self._web_publisher: PokopiaWebStatePublisher | None = None
        self._web_phase_key = "starting"

    def note_controller_input(self) -> None:
        """Reset the five-minute inactivity timer after a real button input."""
        with self._controller_activity_lock:
            self._last_controller_input_monotonic = time.monotonic()

    def _service_controller_idle_keepalive(self) -> bool:
        """Send UP then DOWN after five minutes with no controller input."""
        now = time.monotonic()
        with self._controller_activity_lock:
            if self._idle_keepalive_in_progress:
                return True
            if self.active_bits:
                self._last_controller_input_monotonic = now
                return True
            if (
                now - self._last_controller_input_monotonic
                < CONTROLLER_IDLE_KEEPALIVE_SECONDS
            ):
                return True
            self._idle_keepalive_in_progress = True
            # Reserve the next interval immediately so another service point
            # cannot begin a duplicate pair while this pair is being sent.
            self._last_controller_input_monotonic = now

        _timestamped_log(
            "手柄连续300秒无按键输入：发送DPAD上、DPAD下保持画面活跃。"
        )
        button_held = False
        try:
            for bit_index, gap_ms in (
                (BIT_DPAD_UP, CONTROLLER_IDLE_KEEPALIVE_GAP_MS),
                (BIT_DPAD_DOWN, 0),
            ):
                self._raise_if_macro_interrupted()
                self.heartbeat.beat()
                self.controller.send_bits(1 << bit_index)
                button_held = True
                if self.stop_event.wait(
                    CONTROLLER_IDLE_KEEPALIVE_HOLD_MS / 1000.0
                ):
                    return False
                self.controller.release()
                button_held = False
                if gap_ms and self.stop_event.wait(gap_ms / 1000.0):
                    return False
            return not self.stop_event.is_set()
        finally:
            if button_held:
                with contextlib.suppress(Exception):
                    self.controller.release()
            with self._controller_activity_lock:
                self._last_controller_input_monotonic = time.monotonic()
                self._idle_keepalive_in_progress = False

    def set_room_player_tracker(self, tracker: RoomPlayerTracker) -> None:
        self._room_player_tracker = tracker

    def set_web_publisher(self, publisher: PokopiaWebStatePublisher) -> None:
        self._web_publisher = publisher

    def set_web_phase(
        self,
        key: str,
        label: str,
        *,
        status_key: str | None = None,
        status_label: str | None = None,
        tone: str | None = None,
    ) -> None:
        self._web_phase_key = str(key)
        publisher = self._web_publisher
        if publisher is not None:
            publisher.set_phase(
                key,
                label,
                status_key=status_key,
                status_label=status_label,
                tone=tone,
            )

    def web_phase_key(self) -> str:
        return self._web_phase_key

    def set_web_announcement(self, text: str) -> None:
        publisher = self._web_publisher
        if publisher is not None:
            publisher.set_announcement(text)

    def set_web_code_ready(
        self,
        code: str,
        *,
        code_unknown: bool,
        revision: int,
    ) -> None:
        """Publish the new CODE and open status in one atomic state merge."""
        key = "code_unknown_ready" if code_unknown else "code_ready"
        label = (
            "CODE未能自动识别，请根据截图读取；正在返回STAMP页面"
            if code_unknown
            else "六位CODE已识别，正在返回STAMP页面"
        )
        self._web_phase_key = key
        publisher = self._web_publisher
        if publisher is not None:
            publisher.update(
                code=str(code),
                code_unknown=bool(code_unknown),
                code_revision=max(0, int(revision)),
                phase={"key": key, "label": label},
                status={
                    "key": "open",
                    "label": "开放中",
                    "tone": "green",
                },
            )

    def set_reopen_screenshot_callback(self, callback) -> None:
        self._reopen_screenshot_callback = callback

    def record_reopen_screenshot(self, reason: str) -> None:
        """Synchronously archive the last frame before reopen input begins."""
        callback = self._reopen_screenshot_callback
        if callback is not None:
            callback(reason)

    def mark_room_disconnected(self, reopen_reason: str) -> None:
        tracker = self._room_player_tracker
        if tracker is not None:
            tracker.mark_room_disconnected(reopen_reason)

    def take_previous_round_player_entries(self) -> int:
        tracker = self._room_player_tracker
        if tracker is None:
            return 0
        return tracker.take_previous_round_entries()

    def arm_room_disconnect_on_next_home(
        self,
        reopen_reason: str = "normal_reopen",
    ) -> None:
        """Freeze the room at the first HOME actually sent by a reopen."""
        self._disconnect_on_next_home_reason = str(
            reopen_reason or "normal_reopen"
        )

    def disarm_room_disconnect_on_next_home(self) -> None:
        self._disconnect_on_next_home_reason = ""

    def tap(self, bit_index: int, hold_ms: int, gap_ms: int) -> bool:
        if bit_index == BIT_HOME and self._disconnect_on_next_home_reason:
            reopen_reason = self._disconnect_on_next_home_reason
            self._disconnect_on_next_home_reason = ""
            self.mark_room_disconnected(reopen_reason)
        return super().tap(bit_index, hold_ms, gap_ms)

    def toggle_operation_lock(self) -> bool:
        """Toggle keyboard admission and discard any pending macro trigger."""
        with self.macro_trigger_lock:
            if self.operation_lock_event.is_set():
                self.operation_lock_event.clear()
                return False
            self.operation_lock_event.set()
            for trigger in self.macro6_part_triggers.values():
                trigger.clear()
            return True

    def operation_locked(self) -> bool:
        return self.operation_lock_event.is_set()

    def set_active_macro_key(self, macro_key: int) -> None:
        with self.macro_trigger_lock:
            self._active_macro_key = int(macro_key)

    def current_macro_key(self) -> int:
        with self.macro_trigger_lock:
            return self._active_macro_key

    def set_automatic_restart_guard(
        self,
        guard: _CursorGatedCodeTimer | None,
    ) -> None:
        self._automatic_restart_guard = guard

    def _raise_if_macro_interrupted(self) -> None:
        if self.macro_interrupt_event.is_set():
            raise _MacroInterrupted
        guard = self._automatic_restart_guard
        if guard is not None:
            _raise_if_code_timer_expired(guard)

    def send_active_bits(self) -> None:
        self._raise_if_macro_interrupted()
        self.heartbeat.beat()
        if self.active_bits:
            self.note_controller_input()
        super().send_active_bits()

    def _raw_wait_ms(self, duration_ms: int) -> bool:
        remaining = max(0, duration_ms) / 1000.0
        while remaining > 0:
            self._raise_if_macro_interrupted()
            self.heartbeat.beat()
            if self.stop_event.is_set():
                return False
            wait_seconds = min(remaining, 0.05)
            started = time.monotonic()
            if self.stop_event.wait(wait_seconds):
                return False
            remaining -= max(0.0, time.monotonic() - started)
        self._raise_if_macro_interrupted()
        return not self.stop_event.is_set()

    def _service_pause(self) -> bool:
        self._raise_if_macro_interrupted()
        self.heartbeat.beat()
        if not (
            self.pause_state.is_paused()
            or self.pause_state.has_resume_detection()
        ):
            return True

        saved_bits = self.active_bits
        self.controller.release()
        while True:
            while self.pause_state.is_paused():
                self._raise_if_macro_interrupted()
                self.heartbeat.beat()
                if not self._service_controller_idle_keepalive():
                    return False
                if self.status_callback is not None:
                    self.status_callback(False)
                if self.stop_event.wait(0.05):
                    return False

            if self.pause_state.consume_resume_detection():
                _timestamped_log("宏已恢复，不发送A，直接继续暂停前的原序列。")
                if self.pause_state.is_paused():
                    self.controller.release()
                    continue
            break

        self._raise_if_macro_interrupted()
        self.active_bits = saved_bits
        self.send_active_bits()
        return True

    def wait_ms(self, duration_ms: int) -> bool:
        remaining = max(0, duration_ms) / 1000.0
        while remaining > 0:
            self._raise_if_macro_interrupted()
            self.heartbeat.beat()
            if not self._service_controller_idle_keepalive():
                return False
            if not self._service_pause():
                return False
            if self.status_callback is not None:
                self.status_callback(False)
            started_wait = self.active_monotonic()
            wait_seconds = min(remaining, 0.05)
            if self.stop_event.wait(wait_seconds):
                return False
            if not self.pause_state.is_paused():
                remaining -= max(
                    0.0,
                    self.active_monotonic() - started_wait,
                )
        self._raise_if_macro_interrupted()
        self.heartbeat.beat()
        if not self._service_controller_idle_keepalive():
            return False
        return self._service_pause()


def _wait_for_visual_state(
    context: _HeartbeatMacroContext,
    predicate,
) -> bool:
    while not predicate():
        if not context.wait_ms(100):
            return False
    return True


def _raise_if_code_timer_expired(code_recognizer: CodeRecognizer) -> None:
    if not code_recognizer.timer_expired():
        return
    reason = getattr(code_recognizer, "last_trigger_reason", "")
    if reason == "network_connection_error":
        raise _NetworkConnectionError
    if reason == "business_date_hold":
        raise _BusinessDateHold
    if reason == "business_date_changed":
        raise _BusinessDateChanged
    if reason == "severe_timeout":
        raise _SevereCodeTimerExpired
    raise _CodeTimerExpired


def _wait_ms_with_code_timer(
    context: _HeartbeatMacroContext,
    code_recognizer: CodeRecognizer,
    duration_ms: int,
) -> bool:
    remaining_ms = max(0, duration_ms)
    while remaining_ms > 0:
        _raise_if_code_timer_expired(code_recognizer)
        chunk_ms = min(remaining_ms, 100)
        if not context.wait_ms(chunk_ms):
            return False
        remaining_ms -= chunk_ms
    _raise_if_code_timer_expired(code_recognizer)
    return True


def _wait_for_visual_state_with_code_timer(
    context: _HeartbeatMacroContext,
    code_recognizer: CodeRecognizer,
    predicate,
) -> bool:
    while not predicate():
        _raise_if_code_timer_expired(code_recognizer)
        if not context.wait_ms(100):
            return False
    _raise_if_code_timer_expired(code_recognizer)
    return True


def _wait_for_business_date_boundary(
    context: _HeartbeatMacroContext,
    operational_date: str,
) -> bool:
    """Wait for 05:00 while retaining the 300-second idle keepalive."""
    while _beijing_operational_date() == operational_date:
        # The ordinary wait path maintains heartbeat/pause/interrupt handling
        # and deliberately retains the five-minute DPAD UP/DOWN keepalive.
        if not context.wait_ms(250):
            return False
    return True


def _navigate_until_cursor_stamp(
    context: _HeartbeatMacroContext,
    visual_state: _PokopiaVisualState,
    code_recognizer: CodeRecognizer,
    *,
    accept_stamp_page: bool = True,
) -> bool:
    """Check cursor every 3 seconds and correct each newly observed state."""
    cursor_steps = {
        "CONNECT": (BIT_DPAD_UP, BIT_DPAD_RIGHT, BIT_DPAD_DOWN),
        "CHALLENGE": (BIT_DPAD_RIGHT, BIT_DPAD_DOWN),
        "SHOP": (BIT_DPAD_DOWN,),
        "ARCHE": (BIT_DPAD_LEFT,),
        "RECEIVE": (BIT_DPAD_RIGHT,),
    }
    handled_action = ""
    reward_page_armed = True
    reward_absent_since: float | None = None
    next_cursor_check_at = time.monotonic()
    while True:
        _raise_if_code_timer_expired(code_recognizer)
        if accept_stamp_page and visual_state.stamp() in {"YES", "NO"}:
            # The user may open the stamp page manually while the periodic
            # timer is still running.  Hand control back to the stamp state
            # machine instead of waiting forever for a cursor to reappear.
            return True

        now = time.monotonic()
        if not reward_page_armed:
            _, reward_visible = visual_state.cursor_and_reward()
            if reward_visible:
                reward_absent_since = None
            elif reward_absent_since is None:
                reward_absent_since = now
            elif now - reward_absent_since >= 1.0:
                reward_page_armed = True
                reward_absent_since = None

        if now < next_cursor_check_at:
            if not context.wait_ms(100):
                return False
            continue
        next_cursor_check_at = now + CURSOR_CORRECTION_INTERVAL_SECONDS

        cursor, reward = visual_state.cursor_and_reward()
        if cursor == "STAMP":
            return True

        action = "REWARD" if cursor == "INVALID" and reward else cursor
        if action not in cursor_steps and action != "REWARD":
            handled_action = ""
        elif action == "REWARD" and not reward_page_armed:
            # The B-dismissed page is still visible (or has not remained
            # absent for a full second), so it cannot be counted again.
            handled_action = ""
        elif action != handled_action:
            handled_action = action
            if action == "REWARD":
                _timestamped_log(
                    "CURSOR:INVALID 且 REWARD:TRUE："
                    "等待2000ms后按B。"
                )
                if not _wait_ms_with_code_timer(
                    context,
                    code_recognizer,
                    2000,
                ):
                    return False
                detected_lines = visual_state.reward_lines()
                # REWARD has already remained visible for five seconds here;
                # use one as the safe minimum for the known one-line page.
                added, completed_tasks = visual_state.add_completed_tasks(
                    max(1, detected_lines)
                )
                _timestamped_log(
                    f"REWARD页面识别到{max(1, detected_lines)}条任务；"
                    f"本次计入{added}条，任务已完成{completed_tasks}/3。"
                )
                reward_page_armed = False
                reward_absent_since = None
                # Keep this action latched while the same REWARD page remains
                # visible.  The one-second post-B gap lets it disappear before
                # a later REWARD page is eligible to be handled.
                if not context.tap(BIT_B, hold_ms=50, gap_ms=1000):
                    return False
            else:
                _timestamped_log(f"CURSOR:{action}：执行 STAMP 定位纠偏。")
                for bit_index in cursor_steps[action]:
                    _raise_if_code_timer_expired(code_recognizer)
                    if not context.tap(bit_index, hold_ms=50, gap_ms=500):
                        return False

        if not context.wait_ms(100):
            return False


def _wait_for_reopen_stages(
    context: _HeartbeatMacroContext,
    visual_state: _PokopiaVisualState,
    *,
    start_stage: str = "GO_GAME",
) -> bool:
    """Advance from the current expected Switch save-data stage."""
    stages = (
        "GO_GAME",
        "DOWNLOAD",
        "CLOSE_GAME",
        "OVERWRITE",
    )
    try:
        start_index = stages.index(start_stage)
    except ValueError as exc:
        raise ValueError(f"未知重开阶段入口：{start_stage}") from exc
    for expected in stages[start_index:]:
        context.set_web_phase(
            "waiting_reopen_stage",
            f"正在等待重开阶段：{expected}",
        )
        _timestamped_log(f"开始等待重开阶段：{expected}。")
        while visual_state.reopen_stage() != expected:
            if not context.wait_ms(100):
                return False
        _timestamped_log(f"已识别重开阶段：{expected}。")

        if expected == "GO_GAME":
            if not context.wait_ms(1000):
                return False
            if not context.tap(BIT_A, hold_ms=50, gap_ms=1000):
                return False
            if not context.tap(BIT_A, hold_ms=50, gap_ms=500):
                return False
            if not context.tap(BIT_A, hold_ms=50, gap_ms=0):
                return False
        elif expected in {"DOWNLOAD", "CLOSE_GAME"}:
            if not context.tap(BIT_A, hold_ms=50, gap_ms=0):
                return False
        else:
            if not context.wait_ms(500):
                return False
            if not context.tap(BIT_DPAD_UP, hold_ms=50, gap_ms=500):
                return False
            if not context.tap(BIT_A, hold_ms=50, gap_ms=0):
                return False
    return True


def _run_smart_macro_6_part_1(
    context: _HeartbeatMacroContext,
    visual_state: _PokopiaVisualState,
    *,
    start_at_home: bool = False,
) -> bool:
    """Run Macro6's reopen/read-load segment without an external signature dependency.

    A network-error reopen starts at HOME.  Every other reopen first performs
    the normal escape inputs before reaching that same HOME/read-load path.
    Keep this implementation local to the watchdog so deploying this file on
    its own cannot accidentally call an older smart_macro_gamepad function.
    """
    if not start_at_home:
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
        for _ in range(6):
            if not context.tap(BIT_B, hold_ms=50, gap_ms=500):
                return False

    steps = (
        *((
            (BIT_PLUS, 500),
            (BIT_MINUS, 2000),
            (BIT_DPAD_UP, 500),
            (BIT_A, 5000),
        ) if not start_at_home else ()),
        (BIT_HOME, 500),
        (BIT_DPAD_DOWN, 500),
        (BIT_DPAD_LEFT, 500),
        (BIT_DPAD_LEFT, 500),
        (BIT_A, 1500),
        *((BIT_DPAD_DOWN, 500) for _ in range(6)),
        (BIT_A, 500),
        *((BIT_DPAD_DOWN, 500) for _ in range(3)),
        (BIT_A, 500),
        (BIT_A, 0),
    )
    for bit_index, gap_ms in steps:
        if not context.tap(bit_index, hold_ms=50, gap_ms=gap_ms):
            return False
    return _wait_for_reopen_stages(context, visual_state)


def _run_original_macro1_chain(
    context: _HeartbeatMacroContext,
    visual_state: _PokopiaVisualState,
) -> bool:
    """Run old macro1, then the current key-4 sequence after save success."""
    context.record_reopen_screenshot("manual_key_3_read_load")
    completed = _run_smart_macro_6_part_1(context, visual_state)
    if completed:
        _timestamped_log(
            "旧1宏动作已完成，开始等待 POK_SAV_STATE=true。"
        )
    while completed and not visual_state.save_finished_event.is_set():
        completed = context.wait_ms(100)
    if completed:
        _timestamped_log(
            "POK_SAV_STATE 已变为 true，等待 1000ms 后执行旧2宏。"
        )
        completed = context.wait_ms(1000)
    if completed:
        _timestamped_log(
            "开始执行与按键4一致的新版旧2宏，包含CURSOR循环确认。"
        )
        completed = _run_macro4_with_cursor_retry(
            context,
            visual_state,
        )
    return completed


def _run_macro12_after_download(
    context: _HeartbeatMacroContext,
    visual_state: _PokopiaVisualState,
) -> bool:
    """Key 1/2 post-download sequence gated by a complete black-screen cycle."""
    context.set_web_phase(
        "launching_game",
        "存档已下载，正在返回游戏并等待黑屏",
        status_key="reopening",
        status_label="重开中",
        tone="amber",
    )
    _timestamped_log("POK_SAV_STATE=true：执行 HOME→A×4。")
    if not context.tap(BIT_HOME, hold_ms=50, gap_ms=500):
        return False
    for _ in range(4):
        if not context.tap(BIT_A, hold_ms=50, gap_ms=500):
            return False

    _timestamped_log("HOME→A×4 已完成，等待 BLACK SCREEN=ON。")
    context.set_web_phase("waiting_black_on", "正在等待游戏进入黑屏")
    if not _wait_for_visual_state(context, visual_state.black_screen):
        return False
    _timestamped_log("检测到 BLACK SCREEN=ON，等待黑屏消失。")
    return _run_macro12_after_black_started(context, visual_state)


def _run_macro12_after_black_started(
    context: _HeartbeatMacroContext,
    visual_state: _PokopiaVisualState,
) -> bool:
    """Continue key 1/2 after BLACK SCREEN has already appeared."""
    context.set_web_phase("waiting_black_off", "已进入黑屏，正在等待游戏加载完成")
    if not _wait_for_visual_state(
        context,
        lambda: not visual_state.black_screen(),
    ):
        return False
    _timestamped_log("BLACK SCREEN 已变为OFF，等待3000ms。")
    context.set_web_phase("post_load_delay", "黑屏已消失，等待画面稳定")
    if not context.wait_ms(3000):
        return False
    _timestamped_log(
        "黑屏消失后3000ms已到，执行后续 A→等待→B×5→上→A 序列。"
    )

    if not context.tap(BIT_A, hold_ms=50, gap_ms=20000):
        return False
    context.set_web_phase("opening_room", "正在进入联机入口并定位光标")
    if not _repeat_open_segment_until_cursor(context, visual_state):
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
    if not context.tap(BIT_PLUS, hold_ms=50, gap_ms=0):
        return False
    return _wait_for_connect_ok_and_submit(context, visual_state)


def _wait_for_connect_ok_and_submit(
    context: _HeartbeatMacroContext,
    visual_state: _PokopiaVisualState,
) -> bool:
    """Wait for CONNECT_OK/CODE while giving a new network error priority."""
    network_visible_at_start, network_revision_at_start = (
        visual_state.network_error_state()
    )

    def raise_if_network_error() -> None:
        visible, revision = visual_state.network_error_state()
        if (
            visible
            or network_visible_at_start
            or revision > network_revision_at_start
        ):
            _timestamped_log(
                "按+后的连接阶段检测到Switch连接错误弹窗："
                "立即结束等待并进入网络异常重开。"
            )
            raise _NetworkConnectionError

    context.set_web_phase(
        "waiting_connect_ok",
        "已提交开门，正在等待连接完成",
        status_key="waiting",
        status_label="等待连接",
        tone="amber",
    )
    _timestamped_log("已按+，持续等待 CONNECT_OK=ON 后按A进入CODE页面。")
    while not visual_state.connect_ok():
        raise_if_network_error()
        if not context.wait_ms(100):
            return False
    raise_if_network_error()
    _timestamped_log("检测到 CONNECT_OK=ON，按A并等待500ms。")
    context.set_web_phase("submitting_connection", "连接已开始，正在进入CODE页面")
    submitted_at = context.active_monotonic()
    if not context.tap(BIT_A, hold_ms=50, gap_ms=500):
        return False
    while not visual_state.code_panel():
        raise_if_network_error()
        if (
            context.active_monotonic() - submitted_at
            >= CONNECT_OK_CODE_RETRY_SECONDS
            and visual_state.connect_ok()
        ):
            _timestamped_log(
                "按A后30秒仍未检测到CODE画面，且CONNECT_OK仍为ON："
                "再次按A并重新开始30秒等待。"
            )
            submitted_at = context.active_monotonic()
            if not context.tap(BIT_A, hold_ms=50, gap_ms=500):
                return False
        if not context.wait_ms(100):
            return False
    raise_if_network_error()
    _timestamped_log("已检测到CODE画面，进入新CODE识别等待状态。")
    context.set_web_phase("waiting_code_ocr", "CODE画面已出现，正在识别六位密语")
    return True


def _repeat_open_segment_until_cursor(
    context: _HeartbeatMacroContext,
    visual_state: _PokopiaVisualState,
) -> bool:
    """Repeat B×5→up→A until any valid CURSOR becomes visible."""
    attempt = 0
    while not context.stop_event.is_set():
        attempt += 1
        _timestamped_log(
            f"开门CURSOR确认第{attempt}轮：执行B×5→左摇杆上→A。"
        )
        for _ in range(5):
            if not context.tap(BIT_B, hold_ms=50, gap_ms=500):
                return False
        if not context.move_stick(BIT_LSTICK_UP, duration_ms=100):
            return False
        context.center_stick()
        if not context.wait_ms(500):
            return False
        if not context.tap(BIT_A, hold_ms=50, gap_ms=3000):
            return False

        cursor = visual_state.cursor()
        if cursor != "INVALID":
            _timestamped_log(
                f"开门CURSOR确认成功：CURSOR:{cursor}，继续后续序列。"
            )
            return True
        _timestamped_log(
            "等待3000ms后仍未检测到CURSOR，重新执行开门确认段。"
        )
    return False


def _run_macro12_download_chain(
    context: _HeartbeatMacroContext,
    visual_state: _PokopiaVisualState,
    *,
    start_at_home: bool = False,
    reopen_reason: str | None = None,
) -> bool:
    """Run key 1/2's first part, then use the black-screen-gated ending."""
    effective_reopen_reason = str(
        reopen_reason
        or (
            "network_connection_error"
            if start_at_home
            else "normal_reopen"
        )
    )
    context.set_web_phase(
        "reopening",
        (
            "网络异常后从HOME重开"
            if start_at_home
            else (
                "业务日期切换，正在执行完整读档重开"
                if effective_reopen_reason == "date_change_reopen"
                else "正在结束上轮并执行完整读档重开"
            )
        ),
        status_key="reopening",
        status_label=(
            "日期切换重开"
            if effective_reopen_reason == "date_change_reopen"
            else "重开中"
        ),
        tone="amber",
    )
    context.record_reopen_screenshot(effective_reopen_reason)
    context.arm_room_disconnect_on_next_home(
        effective_reopen_reason
    )
    try:
        completed = _run_smart_macro_6_part_1(
            context,
            visual_state,
            start_at_home=start_at_home,
        )
    finally:
        # An interruption before HOME must not freeze a later unrelated HOME.
        context.disarm_room_disconnect_on_next_home()
    if completed:
        _timestamped_log(
            "按键1/2前段已完成，开始等待 POK_SAV_STATE=true。"
        )
        context.set_web_phase("waiting_save", "正在等待存档下载完成对勾")
    while completed and not visual_state.save_finished_event.is_set():
        completed = context.wait_ms(100)
    if not completed:
        return False
    return _run_macro12_after_download(context, visual_state)


def _run_macro4_with_cursor_retry(
    context: _HeartbeatMacroContext,
    visual_state: _PokopiaVisualState,
) -> bool:
    """Run the former key-2 macro, retrying its open segment until CURSOR."""
    for bit_index in (BIT_HOME, BIT_A, BIT_A, BIT_A, BIT_A):
        if not context.tap(bit_index, hold_ms=50, gap_ms=500):
            return False
    if not context.wait_ms(45000):
        return False
    if not context.tap(BIT_A, hold_ms=50, gap_ms=20000):
        return False

    if not _repeat_open_segment_until_cursor(context, visual_state):
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
    if not context.tap(BIT_PLUS, hold_ms=50, gap_ms=0):
        return False
    network_reopen = False
    while not context.stop_event.is_set():
        try:
            if network_reopen:
                return _run_macro12_download_chain(
                    context,
                    visual_state,
                    start_at_home=True,
                    reopen_reason="network_connection_error",
                )
            return _wait_for_connect_ok_and_submit(context, visual_state)
        except _NetworkConnectionError:
            network_reopen = True
            context.set_web_phase(
                "network_error_reopen",
                "检测到网络波动，正在从HOME重新开放",
                status_key="error",
                status_label="网络异常重开",
                tone="red",
            )
            _timestamped_log(
                "按键3/4在按+后的连接阶段检测到网络错误："
                "跳过前置退出动作，直接从HOME执行网络异常重开。"
            )
    return False


def _leave_code_page_and_locate_stamp(
    context: _HeartbeatMacroContext,
    visual_state: _PokopiaVisualState,
    timer_guard,
) -> bool:
    """Leave the CODE page with B/up/right/down and locate STAMP."""
    context.set_web_phase("locating_stamp", "新轮次已建立，正在定位STAMP光标")
    _timestamped_log("等待 5000ms 后执行 B→上→右→下并定位 STAMP。")
    if not _wait_ms_with_code_timer(context, timer_guard, 5000):
        return False
    for bit_index in (
        BIT_B,
        BIT_DPAD_UP,
        BIT_DPAD_RIGHT,
        BIT_DPAD_DOWN,
    ):
        _raise_if_code_timer_expired(timer_guard)
        if not context.tap(bit_index, hold_ms=50, gap_ms=500):
            return False

    _timestamped_log("导航动作完成，等待 CURSOR:STAMP。")
    if not _navigate_until_cursor_stamp(
        context,
        visual_state,
        timer_guard,
    ):
        return False
    return True


def _run_stamp_check_until_no(
    context: _HeartbeatMacroContext,
    visual_state: _PokopiaVisualState,
    timer_guard,
    *,
    arm_timer_before_poll: bool = False,
) -> bool:
    """Navigate to STAMP and keep polling until STAMP:NO."""
    if not _leave_code_page_and_locate_stamp(
        context,
        visual_state,
        timer_guard,
    ):
        return False

    if arm_timer_before_poll:
        timer_guard.arm()
        _timestamped_log(
            "已进入STAMP循环检测，15分钟/30分钟保护现在重新启用。"
        )
    context.set_web_phase(
        "stamp_cycle",
        "当前轮次开放中，持续检查梦幻章与玩家状态",
        status_key="open",
        status_label="开放中",
        tone="green",
    )
    return _poll_stamp_until_no(context, visual_state, timer_guard)


def _poll_stamp_until_no(
    context: _HeartbeatMacroContext,
    visual_state: _PokopiaVisualState,
    timer_guard,
) -> bool:
    """At CURSOR:STAMP, poll continuously until the stamp page reports NO."""
    context.set_web_phase(
        "stamp_cycle",
        "当前轮次开放中，持续检查梦幻章与玩家状态",
        status_key="open",
        status_label="开放中",
        tone="green",
    )
    next_a_at = time.monotonic() + STAMP_CHECK_INTERVAL_SECONDS
    stamp_check_due = False
    resume_detection_at: float | None = None
    pending_departure = ""
    pending_departure_since: float | None = None
    actionable_departures = {
        "REWARD",
        "CONNECT",
        "CHALLENGE",
        "SHOP",
        "ARCHE",
        "RECEIVE",
    }
    while not context.stop_event.is_set():
        now = time.monotonic()
        if resume_detection_at is not None:
            # Opening the stamp page needs a short, fixed transition delay.
            # This is not the periodic check timer: after it expires, visual
            # detection resumes and a failed page transition starts a fresh
            # 10-second cycle.
            if now < resume_detection_at:
                _raise_if_code_timer_expired(timer_guard)
                if not context.wait_ms(100):
                    return False
                continue
            resume_detection_at = None
            stamp_check_due = False
            next_a_at = now + STAMP_CHECK_INTERVAL_SECONDS

        stamp_state = visual_state.stamp()
        if stamp_state == "NO":
            _timestamped_log("检测到 STAMP:NO，梦幻章检查完成。")
            return True
        if stamp_state == "YES":
            _timestamped_log("检测到 STAMP:YES，立即按B退出梦幻章页面。")
            if not context.tap(BIT_B, hold_ms=50, gap_ms=500):
                return False
            if not _navigate_until_cursor_stamp(
                context,
                visual_state,
                timer_guard,
                accept_stamp_page=False,
            ):
                return False
            stamp_check_due = False
            next_a_at = time.monotonic() + STAMP_CHECK_INTERVAL_SECONDS
            continue

        _raise_if_code_timer_expired(timer_guard)
        cursor, reward = visual_state.cursor_and_reward()
        now = time.monotonic()
        if now >= next_a_at:
            stamp_check_due = True

        if cursor != "STAMP":
            observed_departure = (
                "REWARD" if cursor == "INVALID" and reward else cursor
            )
            if observed_departure not in actionable_departures:
                # Plain INVALID is a transition/recognition state, not a
                # correction target.  In particular, opening the stamp page
                # commonly passes through INVALID before STAMP:YES/NO.
                pending_departure = ""
                pending_departure_since = None
                if not context.wait_ms(100):
                    return False
                continue
            if observed_departure != pending_departure:
                pending_departure = observed_departure
                pending_departure_since = now
            stable_seconds = (
                0.0
                if pending_departure_since is None
                else now - pending_departure_since
            )
            if stable_seconds < CURSOR_CORRECTION_INTERVAL_SECONDS:
                if not context.wait_ms(100):
                    return False
                continue
            _timestamped_log(
                "光标连续3000ms未处于STAMP，执行纠偏："
                f"CURSOR:{observed_departure}。"
            )
            if not _navigate_until_cursor_stamp(
                context,
                visual_state,
                timer_guard,
            ):
                return False
            pending_departure = ""
            pending_departure_since = None
            if not stamp_check_due:
                next_a_at = time.monotonic() + STAMP_CHECK_INTERVAL_SECONDS
            continue
        pending_departure = ""
        pending_departure_since = None
        if stamp_check_due:
            # Re-read all mutually exclusive visual flags atomically at the
            # exact point where A would be sent.  A cursor value captured
            # earlier in this loop must never authorize the button press.
            _, confirmed_cursor, _, confirmed_stamp = visual_state.screen_state()
            if confirmed_cursor != "STAMP" or confirmed_stamp in {"YES", "NO"}:
                _timestamped_log(
                    "STAMP检查周期已到，但即时画面不是CURSOR:STAMP；"
                    "保持到期状态，等待下一次确认STAMP后立即按A。"
                )
                continue
            _timestamped_log("CURSOR:STAMP 周期达到7000ms，按A进入梦幻章页面。")
            if not context.tap(BIT_A, hold_ms=50, gap_ms=0):
                return False
            stamp_check_due = False
            _timestamped_log("按A后等待2000ms，再恢复持续画面检测。")
            resume_detection_at = time.monotonic() + STAMP_PAGE_OPEN_DELAY_SECONDS

        if not context.wait_ms(100):
            return False
    return False


def _resume_from_current_stamp_page(
    context: _HeartbeatMacroContext,
    visual_state: _PokopiaVisualState,
    timer_guard,
) -> bool:
    """Handle the currently visible STAMP:YES/NO page, then keep polling."""
    stamp_state = visual_state.stamp()
    if stamp_state == "NO":
        _timestamped_log("动态起点检测到 STAMP:NO。")
        return True
    if stamp_state != "YES":
        if not _wait_for_visual_state_with_code_timer(
            context,
            timer_guard,
            lambda: visual_state.stamp() in {"YES", "NO"},
        ):
            return False
        return _resume_from_current_stamp_page(
            context,
            visual_state,
            timer_guard,
        )

    _timestamped_log("动态起点检测到 STAMP:YES，按B退出梦幻章页面。")
    if not context.tap(BIT_B, hold_ms=50, gap_ms=500):
        return False
    if not _navigate_until_cursor_stamp(
        context,
        visual_state,
        timer_guard,
        accept_stamp_page=False,
    ):
        return False
    return _poll_stamp_until_no(context, visual_state, timer_guard)


def _run_dynamic_macro_start(
    context: _HeartbeatMacroContext,
    visual_state: _PokopiaVisualState,
    code_recognizer: CodeRecognizer,
    run_counter: Macro6RunCounter,
    announcement_tracker: _RoundAnnouncementTracker,
    macro_key: int,
    timer_guard,
) -> tuple[bool, bool]:
    """Resume key 1 or 2 from the mutually exclusive screen now visible."""
    code_panel, cursor, reward, stamp = visual_state.screen_state()
    if code_panel:
        if not _wait_for_visual_state(
            context,
            lambda: bool(code_recognizer.snapshot())
            or code_recognizer.is_unknown(),
        ):
            return True, False
        entry_code = (
            "未知"
            if code_recognizer.is_unknown()
            else code_recognizer.snapshot()
        )
        if macro_key == 1:
            code_recognizer.record_current("按键1动态入口", force=True)
            _timestamped_log("按键1动态起点：CODE画面；入口不计数。")
        else:
            _timestamped_log(
                "按键2动态起点：CODE画面；"
                "入口CODE不写日志、不截图、不计数。"
            )
        entry_round = run_counter.snapshot().daily_runs + 1
        announcement_text = announcement_tracker.begin_round(
            entry_code,
            entry_round,
            visual_state.take_completed_tasks_for_new_code(),
            context.take_previous_round_player_entries(),
        )
        context.set_web_announcement(announcement_text)
        _queue_code_input_to_chrome(
            entry_code,
            macro_key,
            announcement_text,
        )
        return True, _run_stamp_check_until_no(
            context,
            visual_state,
            timer_guard,
        )
    if stamp in {"YES", "NO"}:
        _timestamped_log(
            f"按键{macro_key}动态起点：STAMP:{stamp} 画面。"
        )
        return True, _resume_from_current_stamp_page(
            context,
            visual_state,
            timer_guard,
        )
    if cursor != "INVALID":
        _timestamped_log(
            f"按键{macro_key}动态起点：CURSOR:{cursor} 画面。"
        )
        completed = _navigate_until_cursor_stamp(
            context,
            visual_state,
            timer_guard,
        )
        if completed:
            completed = _poll_stamp_until_no(
                context,
                visual_state,
                timer_guard,
            )
        return True, completed
    if reward:
        _timestamped_log(
            f"按键{macro_key}动态起点：REWARD:TRUE，"
            "等待5000ms后按B并继续纠偏。"
        )
        completed = _navigate_until_cursor_stamp(
            context,
            visual_state,
            timer_guard,
        )
        if completed:
            completed = _poll_stamp_until_no(
                context,
                visual_state,
                timer_guard,
            )
        return True, completed
    return False, True


def _queue_code_input_to_chrome(
    code: str,
    macro_key: int,
    expanded_text: str,
) -> None:
    """Fill Chrome asynchronously; browser failures must not stop the macro."""

    def worker() -> None:
        try:
            result = input_code_to_chrome(
                code,
                expanded_text=expanded_text,
                allow_unknown=code == "未知",
            )
        except Exception as exc:
            _timestamped_log(
                f"按键{macro_key}未能把CODE {code}输入Chrome：{exc}"
            )
            return
        _timestamped_log(
            f"按键{macro_key}已在Chrome页面 {result.page_title!r} "
            f"输入并发送 {result.text!r}。"
        )

    threading.Thread(
        target=worker,
        name=f"macro6-chrome-code-{macro_key}",
        daemon=True,
    ).start()


def _wait_for_new_code_and_count(
    context: _HeartbeatMacroContext,
    visual_state: _PokopiaVisualState,
    code_recognizer: CodeRecognizer,
    baseline_revision: int,
    run_counter: Macro6RunCounter,
    announcement_tracker: _RoundAnnouncementTracker,
    macro_key: int,
) -> bool:
    """Wait for a new CODE, or continue after ten failed OCR attempts."""
    code_panel_started_at: float | None = None
    while code_recognizer.revision() <= baseline_revision:
        if visual_state.code_panel():
            now = context.active_monotonic()
            if code_panel_started_at is None:
                code_panel_started_at = now
                _timestamped_log(
                    f"按键{macro_key}已检测到绿色CODE面板；"
                    "OCR每1秒重试，连续10次失败后以未知CODE继续。"
                )
            elif now - code_panel_started_at >= CODE_OCR_TIMEOUT_SECONDS:
                raise _CodeRecognitionTimedOut
        if not context.wait_ms(100):
            return False
    code_recognizer.end_ocr_round()
    code_unknown = code_recognizer.is_unknown()
    code = "未知" if code_unknown else code_recognizer.snapshot()
    if not code_unknown:
        # Finish the CODE screenshot before exposing this revision publicly,
        # so visitors never receive the prior round's image with the new CODE.
        code_recognizer.record_current(f"按键{macro_key}执行后新CODE")
    context.set_web_code_ready(
        "" if code_unknown else code,
        code_unknown=code_unknown,
        revision=code_recognizer.revision(),
    )
    count = run_counter.record_code_update()
    _log_run_counter(count)
    if code_unknown:
        _timestamped_log(
            f"按键{macro_key}连续{CODE_OCR_UNKNOWN_AFTER_FAILURES}次未能识别"
            "六位CODE：本轮按未知CODE继续，网页显示？？？？？？并展示截图。"
        )
    else:
        _timestamped_log(f"按键{macro_key}检测到执行后新CODE：{code}。")
    announcement_text = announcement_tracker.begin_round(
        code,
        count.daily_runs,
        visual_state.take_completed_tasks_for_new_code(),
        context.take_previous_round_player_entries(),
    )
    context.set_web_announcement(announcement_text)
    _queue_code_input_to_chrome(
        code,
        macro_key,
        announcement_text,
    )
    return True


def _resume_connect_ok_to_new_code(
    context: _HeartbeatMacroContext,
    visual_state: _PokopiaVisualState,
    code_recognizer: CodeRecognizer,
    run_counter: Macro6RunCounter,
    announcement_tracker: _RoundAnnouncementTracker,
    macro_key: int,
) -> bool:
    """Continue an already-open CONNECT_OK screen through new-CODE handling."""
    baseline_revision = code_recognizer.revision()
    announcement_tracker.ensure_restart_started()
    _timestamped_log(
        f"按键{macro_key}动态起点：CONNECT_OK=ON；"
        "按A继续重开最后阶段并等待新CODE。"
    )
    if not _wait_for_connect_ok_and_submit(context, visual_state):
        return False
    code_recognizer.begin_ocr_round(macro_key)
    return _wait_for_new_code_and_count(
        context,
        visual_state,
        code_recognizer,
        baseline_revision,
        run_counter,
        announcement_tracker,
        macro_key,
    )


def _resume_reopen_visual_entry_to_new_code(
    context: _HeartbeatMacroContext,
    visual_state: _PokopiaVisualState,
    code_recognizer: CodeRecognizer,
    run_counter: Macro6RunCounter,
    announcement_tracker: _RoundAnnouncementTracker,
    macro_key: int,
    entry: str,
) -> bool:
    """Resume an already-running read/load flow from its visible stage."""
    baseline_revision = code_recognizer.revision()
    announcement_tracker.ensure_restart_started()
    context.set_web_phase(
        "resuming_reopen",
        f"从当前画面继续重开：{entry}",
        status_key="reopening",
        status_label="重开中",
        tone="amber",
    )

    if entry == "BLACK_SCREEN":
        _timestamped_log(
            f"按键{macro_key}动态起点：BLACK SCREEN=ON；"
            "等待黑屏消失后继续开门流程。"
        )
        completed = _run_macro12_after_black_started(
            context,
            visual_state,
        )
    elif entry == "SAVE_FINISHED":
        _timestamped_log(
            f"按键{macro_key}动态起点：POK_SAV_STATE=true；"
            "直接执行HOME→A×4并等待黑屏。"
        )
        completed = _run_macro12_after_download(context, visual_state)
    elif entry in REOPEN_STAGE_FEATURES:
        _timestamped_log(
            f"按键{macro_key}动态起点：REOPEN_STAGE:{entry}；"
            "从当前存档阶段继续，不重复此前步骤。"
        )
        completed = _wait_for_reopen_stages(
            context,
            visual_state,
            start_stage=entry,
        )
        if completed:
            context.set_web_phase("waiting_save", "正在等待存档下载完成对勾")
            _timestamped_log(
                "存档阶段操作已完成，持续等待 POK_SAV_STATE=true。"
            )
        while completed and not visual_state.save_finished_event.is_set():
            completed = context.wait_ms(100)
        if completed:
            completed = _run_macro12_after_download(context, visual_state)
    else:
        raise ValueError(f"未知动态重开入口：{entry}")

    if not completed:
        return False
    code_recognizer.begin_ocr_round(macro_key)
    _timestamped_log(
        f"按键{macro_key}已从{entry}续接至CODE画面，等待新CODE。"
    )
    return _wait_for_new_code_and_count(
        context,
        visual_state,
        code_recognizer,
        baseline_revision,
        run_counter,
        announcement_tracker,
        macro_key,
    )


def _current_reopen_visual_entry(
    visual_state: _PokopiaVisualState,
) -> str:
    """Return the highest-priority resumable read/load screen."""
    if visual_state.black_screen():
        return "BLACK_SCREEN"
    if visual_state.save_finished_event.is_set():
        return "SAVE_FINISHED"
    stage = visual_state.reopen_stage()
    return stage if stage in REOPEN_STAGE_FEATURES else ""


def _run_automatic_macro1(
    context: _HeartbeatMacroContext,
    visual_state: _PokopiaVisualState,
    code_recognizer: CodeRecognizer,
    run_counter: Macro6RunCounter,
) -> bool:
    """Repeat the former macro1 chain whenever the stamp page reports NO."""
    visual_state.reset_completed_tasks()
    code_recognizer.reset_timer_anchor()
    timer_guard = _CursorGatedCodeTimer(code_recognizer, visual_state)
    context.set_automatic_restart_guard(timer_guard)
    _timestamped_log(
        "按键1已启动：所有动态入口的15分钟/30分钟保护从00:00开始计时；"
        "15分钟仍仅在有效CURSOR时触发，30分钟不判断画面。"
    )
    announcement_tracker = _RoundAnnouncementTracker()
    dynamic_start_pending = True
    network_reopen_from_home = False
    pending_reopen_reason = "normal_reopen"
    while not context.stop_event.is_set():
        try:
            if dynamic_start_pending:
                dynamic_start_pending = False
                if (
                    visual_state.connect_ok()
                    and not visual_state.code_panel()
                ):
                    # CONNECT_OK is already part of an in-progress reopen.  Do
                    # not discard it by starting read/load from the beginning.
                    timer_guard.disarm()
                    if not _resume_connect_ok_to_new_code(
                        context,
                        visual_state,
                        code_recognizer,
                        run_counter,
                        announcement_tracker,
                        1,
                    ):
                        return False
                    _timestamped_log(
                        "按键1从CONNECT_OK取得新CODE；进入STAMP循环检测时"
                        "重新启用15分钟/30分钟保护。"
                    )
                    if not _run_stamp_check_until_no(
                        context,
                        visual_state,
                        timer_guard,
                        arm_timer_before_poll=True,
                    ):
                        return False
                    announcement_tracker.mark_stamp_full()
                    timer_guard.disarm()
                    _timestamped_log(
                        "按键1从CONNECT_OK续接的轮次检测到STAMP:NO，"
                        "返回自动宏开头执行完整流程。"
                    )
                    continue
                reopen_entry = _current_reopen_visual_entry(visual_state)
                if reopen_entry:
                    # A visible save/load stage means a reopen is already in
                    # progress.  Continue from that exact screen instead of
                    # issuing another escape/HOME sequence.
                    timer_guard.disarm()
                    if not _resume_reopen_visual_entry_to_new_code(
                        context,
                        visual_state,
                        code_recognizer,
                        run_counter,
                        announcement_tracker,
                        1,
                        reopen_entry,
                    ):
                        return False
                    _timestamped_log(
                        f"按键1从{reopen_entry}取得新CODE；进入STAMP循环检测时"
                        "重新启用15分钟/30分钟保护。"
                    )
                    if not _run_stamp_check_until_no(
                        context,
                        visual_state,
                        timer_guard,
                        arm_timer_before_poll=True,
                    ):
                        return False
                    announcement_tracker.mark_stamp_full()
                    timer_guard.disarm()
                    _timestamped_log(
                        f"按键1从{reopen_entry}续接的轮次检测到STAMP:NO，"
                        "返回自动宏开头执行完整流程。"
                    )
                    continue
                handled, completed = _run_dynamic_macro_start(
                    context,
                    visual_state,
                    code_recognizer,
                    run_counter,
                    announcement_tracker,
                    1,
                    timer_guard,
                )
                if handled:
                    if not completed:
                        return False
                    announcement_tracker.mark_stamp_full()
                    timer_guard.disarm()
                    _timestamped_log(
                        "动态起点已检测到 STAMP:NO，"
                        "关闭15分钟/30分钟触发并返回宏最开头执行完整流程。"
                    )
                    continue

            # From here until a new CODE is obtained, read/load/restart is
            # already in progress.  Cursor screens passed on that route must
            # never inject a second 15/30-minute restart.
            timer_guard.disarm()
            announcement_tracker.ensure_restart_started()
            code_revision_before_cycle = code_recognizer.revision()
            if not _run_macro12_download_chain(
                context,
                visual_state,
                start_at_home=network_reopen_from_home,
                reopen_reason=pending_reopen_reason,
            ):
                return False
            network_reopen_from_home = False
            pending_reopen_reason = "normal_reopen"
            code_recognizer.begin_ocr_round(1)
            _timestamped_log("旧宏组合已完成，继续等待 Pokopia CODE 变化。")
            if not _wait_for_new_code_and_count(
                context,
                visual_state,
                code_recognizer,
                code_revision_before_cycle,
                run_counter,
                announcement_tracker,
                1,
            ):
                return False

            _timestamped_log(
                "检测到 CODE 变化；重开收尾期间继续关闭15分钟/30分钟保护，"
                "进入STAMP循环检测时再重新启用。"
            )
            if not _run_stamp_check_until_no(
                context,
                visual_state,
                timer_guard,
                arm_timer_before_poll=True,
            ):
                return False
            announcement_tracker.mark_stamp_full()
            timer_guard.disarm()
            _timestamped_log(
                "检测到 STAMP:NO，关闭15分钟/30分钟触发并返回自动宏开头"
                "执行完整流程。"
            )
        except _NetworkConnectionError:
            announcement_tracker.mark_network_connection_error(
                code_recognizer.timer_elapsed_seconds()
            )
            timer_guard.disarm()
            network_reopen_from_home = True
            pending_reopen_reason = "network_connection_error"
            _timestamped_log(
                "检测到Switch连接错误弹窗：结束本轮并记录开放时长；"
                "网络异常重开跳过HOME之前的全部动作，直接从HOME开始完整读档/重开。"
            )
            context.set_web_phase(
                "network_error_reopen",
                "检测到网络波动，正在关闭本轮并重新开放",
                status_key="error",
                status_label="网络异常重开",
                tone="red",
            )
            continue
        except _BusinessDateHold:
            hold_date = _beijing_operational_date()
            hold_cursor = timer_guard.last_trigger_cursor
            timer_guard.disarm()
            context.set_web_phase(
                "waiting_date_boundary",
                "04:50业务日保护：已暂停控制，等待05:00重开",
                status_key="waiting",
                status_label="等待日期切换",
                tone="amber",
            )
            _timestamped_log(
                "北京时间04:50–05:00保护窗内首次确认"
                f"CURSOR:{hold_cursor}；暂停宏流程并等待05:00；"
                "保留300秒DPAD上下空闲保活。"
            )
            if not _wait_for_business_date_boundary(context, hold_date):
                return False
            announcement_tracker.mark_business_date_changed()
            network_reopen_from_home = False
            pending_reopen_reason = "date_change_reopen"
            _timestamped_log(
                "北京时间已到05:00：结束保护等待，执行一次日期切换重开。"
            )
            context.set_web_phase(
                "date_change_reopen",
                "业务日期已切换，正在重新开放",
                status_key="reopening",
                status_label="日期切换重开",
                tone="amber",
            )
            continue
        except _BusinessDateChanged:
            announcement_tracker.mark_business_date_changed()
            timer_guard.disarm()
            network_reopen_from_home = False
            pending_reopen_reason = "date_change_reopen"
            _timestamped_log(
                "北京时间已到05:00，业务日期发生变更；"
                "当前不在重开流程，立即执行一次日期切换重开。"
            )
            context.set_web_phase(
                "date_change_reopen",
                "业务日期变更，正在重新开放",
                status_key="reopening",
                status_label="日期切换重开",
                tone="amber",
            )
            continue
        except _SevereCodeTimerExpired:
            announcement_tracker.mark_severe_timeout()
            timer_guard.disarm()
            network_reopen_from_home = False
            pending_reopen_reason = "normal_reopen"
            _timestamped_log(
                "CODE更新已严重超时30分钟；不判断CURSOR，"
                "立即返回自动宏开头执行完整读档/重开。"
            )
            context.set_web_phase(
                "severe_timeout_reopen",
                "严重超时，正在重新开放并等待检查",
                status_key="error",
                status_label="严重超时重开",
                tone="red",
            )
            continue
        except _CodeTimerExpired:
            trigger_cursor = timer_guard.last_trigger_cursor
            announcement_tracker.mark_timeout()
            timer_guard.disarm()
            network_reopen_from_home = False
            pending_reopen_reason = "normal_reopen"
            _timestamped_log(
                "CODE 更新计时已达到15分钟，且当前为"
                f"CURSOR:{trigger_cursor}；关闭再次触发并返回自动宏开头"
                "执行完整流程。"
            )
            context.set_web_phase(
                "timeout_reopen",
                "本轮已到15分钟，正在自动重新开放",
                status_key="reopening",
                status_label="超时重开",
                tone="amber",
            )
            continue
        except _CodeRecognitionTimedOut:
            timer_guard.disarm()
            code_recognizer.record_ocr_timeout_error(1)
            _timestamped_log(
                "绿色CODE面板持续3分钟仍未识别到有效新CODE；"
                "不计数、不发送，立即返回宏最开头重新执行完整读档/重开。"
            )
            context.set_web_phase(
                "code_ocr_timeout_reopen",
                "CODE识别超时，正在重新开放",
                status_key="error",
                status_label="识别异常重开",
                tone="red",
            )
            continue
    return False


def _run_dynamic_macro2_once(
    context: _HeartbeatMacroContext,
    visual_state: _PokopiaVisualState,
    code_recognizer: CodeRecognizer,
    run_counter: Macro6RunCounter,
) -> bool:
    """Resume dynamically, run former key 2, then stop at the new CODE."""
    announcement_tracker = _RoundAnnouncementTracker()
    network_guard = _NetworkConnectionErrorGuard(visual_state)
    network_reopen_from_home = False
    context.set_automatic_restart_guard(network_guard)
    try:
        _raise_if_code_timer_expired(network_guard)
        if visual_state.connect_ok() and not visual_state.code_panel():
            if not _resume_connect_ok_to_new_code(
                context,
                visual_state,
                code_recognizer,
                run_counter,
                announcement_tracker,
                2,
            ):
                return False
            if code_recognizer.is_unknown():
                _timestamped_log(
                    "按键2从CONNECT_OK取得未知CODE：执行B→上→右→下，"
                    "返回CURSOR:STAMP后停止。"
                )
                if not _leave_code_page_and_locate_stamp(
                    context,
                    visual_state,
                    network_guard,
                ):
                    return False
            _timestamped_log(
                "按键2已从CONNECT_OK继续并取得新CODE，单次流程结束。"
            )
            network_guard.disarm()
            return True
        reopen_entry = _current_reopen_visual_entry(visual_state)
        if reopen_entry:
            if not _resume_reopen_visual_entry_to_new_code(
                context,
                visual_state,
                code_recognizer,
                run_counter,
                announcement_tracker,
                2,
                reopen_entry,
            ):
                return False
            if code_recognizer.is_unknown():
                _timestamped_log(
                    f"按键2从{reopen_entry}取得未知CODE：执行B→上→右→下，"
                    "返回CURSOR:STAMP后停止。"
                )
                if not _leave_code_page_and_locate_stamp(
                    context,
                    visual_state,
                    network_guard,
                ):
                    return False
            _timestamped_log(
                f"按键2已从{reopen_entry}继续并取得新CODE，单次流程结束。"
            )
            network_guard.disarm()
            return True
        handled, completed = _run_dynamic_macro_start(
            context,
            visual_state,
            code_recognizer,
            run_counter,
            announcement_tracker,
            2,
            network_guard,
        )
        if handled:
            if not completed:
                return False
            announcement_tracker.mark_stamp_full()
            _timestamped_log(
                "按键2动态入口的STAMP检查已完成，开始执行原按键2内容。"
            )
    except _CodeRecognitionTimedOut:
        code_recognizer.record_ocr_timeout_error(2)
        _timestamped_log(
            "按键2从CONNECT_OK进入的绿色CODE面板持续3分钟仍未识别成功；"
            "改为执行一次完整读档/重开。"
        )
    except _NetworkConnectionError:
        announcement_tracker.mark_network_connection_error(
            code_recognizer.timer_elapsed_seconds()
        )
        network_reopen_from_home = True
        _timestamped_log(
            "按键2检测到Switch连接错误弹窗：结束本轮并记录开放时长；"
            "跳过HOME之前的全部动作，直接从HOME读档/重开；"
            "取得新CODE后停止。"
        )
    network_guard.disarm()

    while not context.stop_event.is_set():
        baseline_revision = code_recognizer.revision()
        announcement_tracker.ensure_restart_started()
        try:
            if not _run_macro12_download_chain(
                context,
                visual_state,
                start_at_home=network_reopen_from_home,
            ):
                return False
        except _NetworkConnectionError:
            announcement_tracker.mark_network_connection_error(
                code_recognizer.timer_elapsed_seconds()
            )
            network_reopen_from_home = True
            context.set_web_phase(
                "network_error_reopen",
                "检测到网络波动，正在关闭本轮并重新开放",
                status_key="error",
                status_label="网络异常重开",
                tone="red",
            )
            _timestamped_log(
                "按键2在按+后的连接阶段检测到网络错误："
                "结束本轮并直接从HOME重新执行读档/开门；"
                "取得新CODE后停止。"
            )
            continue
        network_reopen_from_home = False
        code_recognizer.begin_ocr_round(2)
        _timestamped_log("按键2原内容已完成，等待执行后新CODE。")
        try:
            if not _wait_for_new_code_and_count(
                context,
                visual_state,
                code_recognizer,
                baseline_revision,
                run_counter,
                announcement_tracker,
                2,
            ):
                return False
        except _CodeRecognitionTimedOut:
            code_recognizer.record_ocr_timeout_error(2)
            _timestamped_log(
                "按键2绿色CODE面板持续3分钟仍未识别到有效新CODE；"
                "不计数、不发送，重新执行一次完整读档/重开。"
            )
            continue
        if code_recognizer.is_unknown():
            _timestamped_log(
                "按键2本轮CODE为未知：按要求执行B→上→右→下，"
                "返回CURSOR:STAMP后再停止。"
            )
            if not _leave_code_page_and_locate_stamp(
                context,
                visual_state,
                network_guard,
            ):
                return False
        break
    if code_recognizer.is_unknown():
        _timestamped_log(
            "按键2未知CODE截图已播报，且已返回CURSOR:STAMP；"
            "停止并等待下一次手动按2。"
        )
    else:
        _timestamped_log(
            "按键2已到达新CODE画面，停止并返回待机；"
            "需要再次手动按2才能继续。"
        )
    return True


def _run_macro6_watchdog_loop(
    context: _HeartbeatMacroContext,
    visual_state: _PokopiaVisualState,
    code_recognizer: CodeRecognizer,
    run_counter: Macro6RunCounter,
) -> None:
    """Dispatch automatic macro1, former macro1, and former macro2."""
    _timestamped_log(
        "smart macro6 watchdog 已待机："
        "1=动态起点自动循环宏，"
        "2=动态起点单轮宏→停在新CODE，"
        "3=原宏1，4=原宏2。"
    )
    context.set_web_phase(
        "standby",
        "watchdog已就绪，等待宏按键",
        status_key="standby",
        status_label="待机",
        tone="gray",
    )
    while not context.stop_event.is_set():
        context.heartbeat.beat()
        selected_part = 0
        with context.macro_trigger_lock:
            for part_number, trigger in context.macro6_part_triggers.items():
                if trigger.is_set():
                    selected_part = part_number
                    break
            if selected_part:
                for trigger in context.macro6_part_triggers.values():
                    trigger.clear()
                context.macro_interrupt_event.clear()
                context._active_macro_key = selected_part
                context.macro_running_event.set()
        if not selected_part:
            if not context._service_controller_idle_keepalive():
                return
            context.stop_event.wait(0.05)
            continue

        _timestamped_log(
            f"smart macro6 watchdog 开始执行按键 {selected_part} 宏。"
        )
        context.set_web_phase(
            "macro_started",
            f"按键{selected_part}宏已启动，正在判断当前入口",
            status_key="running",
            status_label="流程执行中",
            tone="amber",
        )
        completed = False
        interrupted = False
        try:
            if selected_part == 1:
                completed = _run_automatic_macro1(
                    context,
                    visual_state,
                    code_recognizer,
                    run_counter,
                )
            elif selected_part == 2:
                completed = _run_dynamic_macro2_once(
                    context,
                    visual_state,
                    code_recognizer,
                    run_counter,
                )
            elif selected_part == 3:
                completed = _run_original_macro1_chain(
                    context,
                    visual_state,
                )
            else:
                completed = _run_macro4_with_cursor_retry(
                    context,
                    visual_state,
                )
        except _MacroInterrupted:
            interrupted = True
        finally:
            context.active_bits = 0
            context.set_automatic_restart_guard(None)
            with contextlib.suppress(Exception):
                context.controller.release()
            with context.macro_trigger_lock:
                # Defensive clear: no macro trigger received during this run
                # is allowed to survive into the next idle state.
                for trigger in context.macro6_part_triggers.values():
                    trigger.clear()
                context._active_macro_key = 0
                context.macro_running_event.clear()
                context.macro_interrupt_event.clear()

        if interrupted:
            _timestamped_log(
                f"smart macro6 watchdog 按键 {selected_part} 宏已被 0 强制中断，"
                "所有待处理宏指令已丢弃，返回待机。"
            )
            context.set_web_phase(
                "interrupted",
                f"按键{selected_part}宏已被手动中断",
                status_key="standby",
                status_label="已中断",
                tone="gray",
            )
            continue
        if not completed:
            return
        if selected_part == 2:
            _timestamped_log("按键2已停在执行后新CODE，返回手动待机。")
            context.set_web_phase(
                "single_round_ready",
                "单次宏已取得新CODE，等待下一次手动操作",
                status_key="open",
                status_label="开放中",
                tone="green",
            )
        elif selected_part == 3:
            _timestamped_log("按键3原宏1执行完成，返回待机。")
            context.set_web_phase(
                "standby",
                "读档宏已完成，等待操作",
                status_key="standby",
                status_label="待机",
                tone="gray",
            )
        else:
            _timestamped_log("按键4原宏2执行完成，返回待机。")
            context.set_web_phase(
                "standby",
                "开门宏已完成，等待操作",
                status_key="standby",
                status_label="待机",
                tone="gray",
            )


def _handle_watchdog_control_key(
    key: str,
    context: _HeartbeatMacroContext,
    pause_state: _PauseState,
    run_counter: Macro6RunCounter,
) -> bool:
    """Handle Macro6 keyboard admission before normal controller keys."""
    normalized = key.lower()
    if normalized == "f12":
        locked = context.toggle_operation_lock()
        if locked:
            if not context.macro_running_event.is_set():
                context.active_bits = 0
                with contextlib.suppress(Exception):
                    context.controller.release()
            _timestamped_log(
                "操作锁已开启：宏继续运行；除F12解锁外，"
                "所有键盘操作均已锁定。"
            )
        else:
            _timestamped_log("操作锁已解除：键盘控制恢复。")
        return True

    if context.operation_locked():
        _timestamped_log(f"操作锁已开启，忽略按键 {key!r}。按F12解除锁定。")
        return True

    if normalized == "f10":
        context.manual_screenshot_event.set()
        _timestamped_log("收到 F10：已请求保存当前原始采集画面。")
        return True

    if normalized == "0":
        if not context.macro_running_event.is_set():
            _timestamped_log("收到 0：当前没有正在执行的 Macro6 指令。")
            return True
        context.macro_interrupt_event.set()
        context.active_bits = 0
        with contextlib.suppress(Exception):
            context.controller.release()
        _timestamped_log(
            "收到 0：正在强制中断当前 Macro6 指令；"
            "中断完成后可重新接收 1/2/3/4。"
        )
        return True

    if normalized in {"\\", "|", "｜"}:
        _log_run_counter(run_counter.snapshot())
        return True

    if normalized in {"\x1b", "esc", "escape"}:
        context.stop_event.set()
        return False

    if normalized == "q":
        return _handle_terminal_control_key(key, context, pause_state)

    if normalized == "p":
        paused = pause_state.toggle()
        if paused:
            _timestamped_log(
                "宏已暂停；可使用手动键盘控制。再次按 P 恢复。"
            )
        else:
            _timestamped_log(
                "收到恢复指令；不发送A，直接继续暂停前的原序列。"
            )
        return True

    if normalized in {"1", "2", "3", "4"}:
        part_number = int(normalized)
        with context.macro_trigger_lock:
            already_pending = any(
                trigger.is_set()
                for trigger in context.macro6_part_triggers.values()
            )
            if context.macro_running_event.is_set() or already_pending:
                _timestamped_log(
                    f"忽略 Macro6 按键 {part_number}："
                    "当前宏正在执行或已有宏指令待执行。"
                )
                return True
            context.macro6_part_triggers[part_number].set()
        _timestamped_log(f"已触发 smart macro6 按键 {part_number} 宏。")
        return True

    # All regular manual keys intentionally retain the original behavior,
    # including while any Macro6 command is running.
    if (
        key in {"'", '"'}
        or key.upper() in _MANUAL_ARROW_BITS
        or normalized in _MANUAL_KEY_BITS
    ):
        context.note_controller_input()
    return _handle_terminal_control_key(key, context, pause_state)


def _read_macro6_windows_terminal_key() -> str:
    """Read Macro6 console keys, including the locally owned F10 shortcut."""
    key = msvcrt.getwch()
    if key not in {"\x00", "\xe0"}:
        return key
    extended = msvcrt.getwch()
    return {
        "H": "UP",
        "P": "DOWN",
        "K": "LEFT",
        "M": "RIGHT",
        "D": "F10",
        "\x86": "F12",
    }.get(extended, "")


def _listen_for_watchdog_keyboard(
    context: _HeartbeatMacroContext,
    pause_state: _PauseState,
    run_counter: Macro6RunCounter,
    worker_errors: list[BaseException],
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
                key = _read_macro6_windows_terminal_key()
                if key and not _handle_watchdog_control_key(
                    key,
                    context,
                    pause_state,
                    run_counter,
                ):
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
                if key and not _handle_watchdog_control_key(
                    key,
                    context,
                    pause_state,
                    run_counter,
                ):
                    return
        finally:
            termios.tcsetattr(
                input_fd,
                termios.TCSADRAIN,
                previous_terminal_mode,
            )
    except BaseException as exc:
        worker_errors.append(exc)
        context.stop_event.set()


@dataclass(frozen=True)
class TemplateMatch:
    detected: bool
    edge_similarity: float
    color_error: float


@dataclass(frozen=True)
class PokopiaDetection:
    save_finished: bool
    able_access: bool
    black_screen: bool
    connect_ok: bool
    code_panel: bool
    cursor: str
    reward: bool
    reward_lines: int
    stamp: str
    network_error: bool
    reopen_stage: str = "INVALID"


class StampCodeArchive:
    """Save CODE-change frames and append the corresponding daily UTC log."""

    def __init__(self, root: Path) -> None:
        self.root = root
        self._lock = threading.Lock()

    @staticmethod
    def _stamp_date(now_utc: datetime) -> str:
        return _beijing_operational_date(now_utc)

    @staticmethod
    def _save_png(path: Path, frame: np.ndarray) -> bool:
        try:
            success, encoded = cv2.imencode(".png", frame)
            if not success:
                return False
            encoded.tofile(str(path))
            return True
        except Exception as exc:
            _timestamped_log(f"CODE 更新截图保存失败：{exc}")
            return False

    def record(
        self,
        code: str,
        frame: np.ndarray,
        detection: PokopiaDetection,
    ) -> None:
        now_utc = datetime.now(timezone.utc)
        stamp_date = self._stamp_date(now_utc)
        folder_name = f"STAMP_{stamp_date}"
        folder = self.root / folder_name
        filename = (
            "STAMP_"
            + now_utc.strftime("%Y%m%dT%H%M%S_%fZ")
            + f"_{code}.png"
        )
        screenshot_path = folder / filename
        record_path = folder / f"{folder_name}.jsonl"
        with self._lock:
            try:
                folder.mkdir(parents=True, exist_ok=True)
                screenshot_saved = self._save_png(screenshot_path, frame)
                record = {
                    "changed_at_utc": now_utc.isoformat(
                        timespec="milliseconds"
                    ).replace("+00:00", "Z"),
                    "stamp_operational_date": stamp_date,
                    "day_boundary_beijing": "05:00",
                    "code": code,
                    "screenshot": filename if screenshot_saved else "未保存",
                    "detection": {
                        "POK_SAV_STATE": detection.save_finished,
                        "BLACK_SCREEN": detection.black_screen,
                        "CONNECT_OK": detection.connect_ok,
                        "CODE_PANEL": detection.code_panel,
                        "CURSOR": detection.cursor,
                        "REWARD": detection.reward,
                        "REWARD_LINES": detection.reward_lines,
                        "STAMP": detection.stamp,
                        "NETWORK_ERROR": detection.network_error,
                        "REOPEN_STAGE": detection.reopen_stage,
                    },
                }
                with record_path.open("a", encoding="utf-8") as output:
                    output.write(
                        json.dumps(record, ensure_ascii=False) + "\n"
                    )
                _timestamped_log(
                    f"CODE 更新归档完成：{folder_name}/{filename}"
                )
            except Exception as exc:
                _timestamped_log(f"CODE 更新每日记录保存失败：{exc}")

    def record_error(
        self,
        error: str,
        *,
        macro_key: int,
        retained_code: str,
        failure_count: int,
        frame: np.ndarray | None,
        detection: PokopiaDetection | None,
        ocr_results: tuple[str, ...] = (),
        screenshot_suppressed_reason: str = "",
    ) -> None:
        """Append an error event to the same operational-day JSONL log."""
        now_utc = datetime.now(timezone.utc)
        stamp_date = self._stamp_date(now_utc)
        folder_name = f"STAMP_{stamp_date}"
        folder = self.root / folder_name
        record_path = folder / f"{folder_name}.jsonl"
        screenshot_filename = (
            "ERROR_"
            + now_utc.strftime("%Y%m%dT%H%M%S_%fZ")
            + f"_{error}.png"
        )
        screenshot_path = folder / screenshot_filename
        with self._lock:
            try:
                folder.mkdir(parents=True, exist_ok=True)
                screenshot_saved = (
                    frame is not None
                    and self._save_png(screenshot_path, frame)
                )
                record = {
                    "recorded_at_utc": now_utc.isoformat(
                        timespec="milliseconds"
                    ).replace("+00:00", "Z"),
                    "stamp_operational_date": stamp_date,
                    "day_boundary_beijing": "05:00",
                    "status": "error",
                    "error": error,
                    "macro_key": int(macro_key),
                    "code": retained_code,
                    "timeout_seconds": CODE_OCR_TIMEOUT_SECONDS,
                    "ocr_retry_seconds": CODE_OCR_RETRY_SECONDS,
                    "ocr_unknown_after_failures": (
                        CODE_OCR_UNKNOWN_AFTER_FAILURES
                    ),
                    "ocr_failure_count": max(0, int(failure_count)),
                    "ocr_results": list(ocr_results),
                    "screenshot": (
                        screenshot_filename if screenshot_saved else "未保存"
                    ),
                    "screenshot_suppressed_reason": (
                        screenshot_suppressed_reason or None
                    ),
                    "detection": (
                        {
                            "POK_SAV_STATE": detection.save_finished,
                            "BLACK_SCREEN": detection.black_screen,
                            "CONNECT_OK": detection.connect_ok,
                            "CODE_PANEL": detection.code_panel,
                            "CURSOR": detection.cursor,
                            "REWARD": detection.reward,
                            "REWARD_LINES": detection.reward_lines,
                            "STAMP": detection.stamp,
                            "NETWORK_ERROR": detection.network_error,
                            "REOPEN_STAGE": detection.reopen_stage,
                        }
                        if detection is not None
                        else None
                    ),
                }
                with record_path.open("a", encoding="utf-8") as output:
                    output.write(json.dumps(record, ensure_ascii=False) + "\n")
                _timestamped_log(
                    f"CODE识别error已记录（{error}）：{folder_name}/"
                    f"{record_path.name}；截图：{record['screenshot']}"
                )
            except Exception as exc:
                _timestamped_log(f"CODE识别error记录保存失败：{exc}")

    def record_manual_screenshot(
        self,
        frame: np.ndarray,
        detection: PokopiaDetection,
    ) -> None:
        """Save a manually requested unmodified capture frame."""
        now_utc = datetime.now(timezone.utc)
        stamp_date = self._stamp_date(now_utc)
        folder_name = f"STAMP_{stamp_date}"
        folder = self.root / folder_name
        filename = (
            "SCREENSHOT_"
            + now_utc.strftime("%Y%m%dT%H%M%S_%fZ")
            + ".png"
        )
        screenshot_path = folder / filename
        record_path = folder / f"{folder_name}.jsonl"
        with self._lock:
            try:
                folder.mkdir(parents=True, exist_ok=True)
                screenshot_saved = self._save_png(screenshot_path, frame)
                record = {
                    "recorded_at_utc": now_utc.isoformat(
                        timespec="milliseconds"
                    ).replace("+00:00", "Z"),
                    "stamp_operational_date": stamp_date,
                    "day_boundary_beijing": "05:00",
                    "status": "manual_screenshot",
                    "screenshot": filename if screenshot_saved else "未保存",
                    "detection": {
                        "POK_SAV_STATE": detection.save_finished,
                        "BLACK_SCREEN": detection.black_screen,
                        "CONNECT_OK": detection.connect_ok,
                        "CODE_PANEL": detection.code_panel,
                        "CURSOR": detection.cursor,
                        "REWARD": detection.reward,
                        "REWARD_LINES": detection.reward_lines,
                        "STAMP": detection.stamp,
                        "NETWORK_ERROR": detection.network_error,
                        "REOPEN_STAGE": detection.reopen_stage,
                    },
                }
                with record_path.open("a", encoding="utf-8") as output:
                    output.write(json.dumps(record, ensure_ascii=False) + "\n")
                _timestamped_log(
                    f"watchdog 原始画面截图已归档：{folder_name}/{filename}"
                )
            except Exception as exc:
                _timestamped_log(f"watchdog 原始画面截图保存失败：{exc}")

    def record_round_end_screenshot(
        self,
        frame: np.ndarray,
        code: str,
        run_count: RunCounterSnapshot,
        reason: str,
        macro_key: int,
    ) -> None:
        """Archive the unmodified frame synchronously before a reopen starts."""
        now_utc = datetime.now(timezone.utc)
        stamp_date = self._stamp_date(now_utc)
        folder_name = f"STAMP_{stamp_date}"
        folder = self.root / folder_name
        safe_code = re.sub(r"[^0-9A-Za-z_-]+", "_", code.strip()) or "EMPTY"
        filename = (
            "SCREENSHOT_"
            + now_utc.strftime("%Y%m%dT%H%M%S_%fZ")
            + f"_{safe_code}_结束.png"
        )
        screenshot_path = folder / filename
        record_path = folder / f"{folder_name}.jsonl"
        with self._lock:
            try:
                folder.mkdir(parents=True, exist_ok=True)
                screenshot_saved = self._save_png(screenshot_path, frame)
                record = {
                    "recorded_at_utc": now_utc.isoformat(
                        timespec="milliseconds"
                    ).replace("+00:00", "Z"),
                    "stamp_operational_date": stamp_date,
                    "status": "round_end_before_reopen",
                    "round_today": run_count.daily_runs,
                    "round_total": run_count.total_runs,
                    "code": code,
                    "reason": reason,
                    "reason_text": _reopen_reason_text(reason),
                    "macro_key": int(macro_key),
                    "screenshot": filename if screenshot_saved else "未保存",
                }
                with record_path.open("a", encoding="utf-8") as output:
                    output.write(json.dumps(record, ensure_ascii=False) + "\n")
                _timestamped_log(
                    "重开前原始截图已归档："
                    f"本日第{run_count.daily_runs}轮、总第"
                    f"{run_count.total_runs}轮、CODE={code or '<空>'}、"
                    f"原因={_reopen_reason_text(reason)}；"
                    f"{folder_name}/{record['screenshot']}"
                )
            except Exception as exc:
                _timestamped_log(f"重开前原始截图保存失败：{exc}")

    def record_player_notification_failure(
        self,
        frame: np.ndarray,
        reason: str,
    ) -> None:
        """Save one raw frame when the globe passes but its banner fails."""
        now_utc = datetime.now(timezone.utc)
        stamp_date = self._stamp_date(now_utc)
        folder_name = f"STAMP_{stamp_date}"
        folder = self.root / folder_name
        safe_reason = re.sub(r"[^0-9A-Za-z_-]+", "_", reason).strip("_")
        safe_reason = safe_reason or "right_banner_failed"
        filename = (
            "ERROR_"
            + now_utc.strftime("%Y%m%dT%H%M%S_%fZ")
            + f"_player_notification_{safe_reason}.png"
        )
        screenshot_path = folder / filename
        record_path = folder / f"{folder_name}.jsonl"
        with self._lock:
            try:
                folder.mkdir(parents=True, exist_ok=True)
                screenshot_saved = self._save_png(screenshot_path, frame)
                record = {
                    "recorded_at_utc": now_utc.isoformat(
                        timespec="milliseconds"
                    ).replace("+00:00", "Z"),
                    "stamp_operational_date": stamp_date,
                    "day_boundary_beijing": "05:00",
                    "status": "player_notification_detection_error",
                    "error": reason,
                    "globe_detected": True,
                    "right_banner_detected": False,
                    "screenshot": filename if screenshot_saved else "未保存",
                }
                with record_path.open("a", encoding="utf-8") as output:
                    output.write(json.dumps(record, ensure_ascii=False) + "\n")
                _timestamped_log(
                    "玩家通知右侧识别失败截图已归档："
                    f"{folder_name}/{filename}"
                )
            except Exception as exc:
                _timestamped_log(f"玩家通知失败截图保存失败：{exc}")

    def record_player_name_crop(
        self,
        player_name: str,
        name_crop: np.ndarray,
        captured_at_utc: datetime,
        notification_status: str,
    ) -> None:
        """Save the repaired OCR input and append its result to an index."""
        if name_crop is None or name_crop.size == 0:
            return
        capture_time = captured_at_utc
        if capture_time.tzinfo is None:
            capture_time = capture_time.replace(tzinfo=timezone.utc)
        capture_time = capture_time.astimezone(timezone.utc)
        # This is filename safety only. Keep the recognized name itself
        # unchanged, including punctuation that is legal in Windows paths.
        safe_name = re.sub(
            r'[<>:"/\\|?*\x00-\x1f]+',
            "_",
            player_name,
        ).strip(" .")
        safe_name = (safe_name or "EMPTY")[:80]
        folder = self.root / "Name"
        base_name = (
            capture_time.strftime("%Y%m%dT%H%M%S_%fZ")
            + f"_{safe_name}"
        )
        with self._lock:
            try:
                folder.mkdir(parents=True, exist_ok=True)
                screenshot_path = folder / f"{base_name}.png"
                suffix = 2
                while screenshot_path.exists():
                    screenshot_path = folder / f"{base_name}_{suffix}.png"
                    suffix += 1
                if self._save_png(screenshot_path, name_crop):
                    record_time = capture_time.isoformat(
                        timespec="milliseconds"
                    ).replace("+00:00", "Z")
                    record_path = folder / "姓名识别记录.txt"
                    with record_path.open("a", encoding="utf-8") as output:
                        output.write(
                            f"{record_time}\t{screenshot_path.name}\t"
                            f"识别名={player_name}\t状态={notification_status}\n"
                        )
                    _timestamped_log(
                        "玩家姓名右边缘修复裁剪已归档："
                        f"Name/{screenshot_path.name}"
                    )
                else:
                    _timestamped_log("玩家姓名原始裁剪编码失败。")
            except Exception as exc:
                _timestamped_log(f"玩家姓名原始裁剪保存失败：{exc}")

    def record_reward_tasks(
        self,
        *,
        detected_lines: int,
        added_lines: int,
        completed_tasks: int,
    ) -> None:
        """Append one accepted REWARD page to the operational-day log."""
        now_utc = datetime.now(timezone.utc)
        now_beijing = now_utc.astimezone(BEIJING_TIMEZONE)
        stamp_date = self._stamp_date(now_utc)
        folder_name = f"STAMP_{stamp_date}"
        folder = self.root / folder_name
        record_path = folder / f"{folder_name}.jsonl"
        with self._lock:
            try:
                folder.mkdir(parents=True, exist_ok=True)
                record = {
                    "recorded_at_utc": now_utc.isoformat(
                        timespec="milliseconds"
                    ).replace("+00:00", "Z"),
                    "recorded_at_beijing": now_beijing.isoformat(
                        timespec="milliseconds"
                    ),
                    "stamp_operational_date": stamp_date,
                    "day_boundary_beijing": "05:00",
                    "status": "reward_tasks_detected",
                    "reward_lines_detected": min(
                        3,
                        max(0, int(detected_lines)),
                    ),
                    "reward_lines_added": min(
                        3,
                        max(0, int(added_lines)),
                    ),
                    "round_tasks_completed": min(
                        3,
                        max(0, int(completed_tasks)),
                    ),
                    "round_tasks_total": 3,
                }
                with record_path.open("a", encoding="utf-8") as output:
                    output.write(json.dumps(record, ensure_ascii=False) + "\n")
                _timestamped_log(
                    "REWARD任务日志已写入："
                    f"检测{record['reward_lines_detected']}条，"
                    f"增加{record['reward_lines_added']}条，"
                    f"本轮累计{record['round_tasks_completed']}/3。"
                )
            except Exception as exc:
                _timestamped_log(f"REWARD任务日志保存失败：{exc}")

    def record_player_status(
        self,
        *,
        player_name: str,
        visit_index_for_player: int,
        notification_status: str,
        room_status: str,
        action: str,
        room_players: tuple[tuple[str, str], ...],
        ocr_player_name: str | None = None,
        capture_sequence: int | None = None,
        captured_at_utc: datetime | None = None,
        segment_id: int | None = None,
        processing_delay_ms: int | None = None,
    ) -> None:
        """Append one recognized player arrival/departure notification."""
        now_utc = datetime.now(timezone.utc)
        now_beijing = now_utc.astimezone(BEIJING_TIMEZONE)
        stamp_date = self._stamp_date(now_utc)
        folder_name = f"STAMP_{stamp_date}"
        folder = self.root / folder_name
        record_path = folder / f"{folder_name}.jsonl"
        with self._lock:
            try:
                folder.mkdir(parents=True, exist_ok=True)
                record = {
                    "record_time": now_beijing.replace(
                        tzinfo=None
                    ).isoformat(timespec="milliseconds"),
                    "status": "player_room_status",
                    "player_name": player_name,
                    "ocr_player_name": ocr_player_name or player_name,
                    "visit_index_for_player": visit_index_for_player,
                    "notification_status": notification_status,
                    "room_status": room_status,
                    "action": action,
                    "capture_sequence": capture_sequence,
                    "capture_time": (
                        captured_at_utc.astimezone(BEIJING_TIMEZONE)
                        .replace(tzinfo=None)
                        .isoformat(timespec="milliseconds")
                        if captured_at_utc is not None
                        else None
                    ),
                    "segment_id": segment_id,
                    "processing_delay_ms": processing_delay_ms,
                    "room_players": [
                        {"name": name, "status": status}
                        for name, status in room_players
                    ],
                }
                with record_path.open("a", encoding="utf-8") as output:
                    output.write(json.dumps(record, ensure_ascii=False) + "\n")
                _timestamped_log(
                    f"玩家状态已记录：{player_name}{notification_status}；"
                    f"房间状态={room_status or '已移除'}。"
                )
            except Exception as exc:
                _timestamped_log(f"玩家状态日志保存失败：{exc}")

    def record_player_reward(
        self,
        *,
        reward_lines_added: int,
        players: tuple[dict[str, object], ...],
    ) -> None:
        """Append one accepted REWARD event for every active player visit."""
        if not players:
            return
        now_utc = datetime.now(timezone.utc)
        now_beijing = now_utc.astimezone(BEIJING_TIMEZONE)
        stamp_date = self._stamp_date(now_utc)
        folder_name = f"STAMP_{stamp_date}"
        folder = self.root / folder_name
        record_path = folder / f"{folder_name}.jsonl"
        with self._lock:
            try:
                folder.mkdir(parents=True, exist_ok=True)
                record = {
                    "record_time": now_beijing.replace(
                        tzinfo=None
                    ).isoformat(timespec="milliseconds"),
                    "status": "player_reward_progress",
                    "reward_lines_added": min(
                        3,
                        max(0, int(reward_lines_added)),
                    ),
                    "players": list(players),
                }
                with record_path.open("a", encoding="utf-8") as output:
                    output.write(json.dumps(record, ensure_ascii=False) + "\n")
            except Exception as exc:
                _timestamped_log(f"玩家任务进度日志保存失败：{exc}")

    def record_room_reopen_summary(
        self,
        visits: tuple[dict[str, object], ...],
        frozen_at_utc: datetime,
        round_player_entries: int,
        round_player_entry_failures: int,
        *,
        reopen_reason: str = "normal_reopen",
        network_error_player_names: tuple[str, ...] = (),
    ) -> int:
        """Write every distinct player visit closed by one reopen."""
        frozen_at_beijing = frozen_at_utc.astimezone(BEIJING_TIMEZONE)
        stamp_date = self._stamp_date(frozen_at_utc)
        folder_name = f"STAMP_{stamp_date}"
        folder = self.root / folder_name
        record_path = folder / f"PLAYERS_{stamp_date}.jsonl"
        with self._lock:
            try:
                folder.mkdir(parents=True, exist_ok=True)
                reopen_index = 1
                if record_path.exists():
                    with record_path.open("r", encoding="utf-8") as source:
                        reopen_index += sum(
                            1
                            for line in source
                            if '"status": "room_reopen_summary"' in line
                        )
                record = {
                    "record_time": frozen_at_beijing.replace(
                        tzinfo=None
                    ).isoformat(timespec="milliseconds"),
                    "status": "room_reopen_summary",
                    "reopen_index_today": reopen_index,
                    "summary": f"今日第{reopen_index}次重开",
                    "round_player_entries": max(
                        0,
                        int(round_player_entries),
                    ),
                    "round_summary": (
                        f"本轮一共{max(0, int(round_player_entries))}名玩家进入"
                    ),
                    "round_player_entry_failures": max(
                        0,
                        int(round_player_entry_failures),
                    ),
                    "round_failure_summary": (
                        "本轮一共"
                        f"{max(0, int(round_player_entry_failures))}次进入失败"
                    ),
                    "reopen_reason": str(reopen_reason),
                    "reopen_reason_text": _reopen_reason_text(
                        reopen_reason
                    ),
                    "network_connection_error": {
                        "detected": (
                            reopen_reason == "network_connection_error"
                        ),
                        "players_not_arrived": list(
                            network_error_player_names
                        ),
                        "count": len(network_error_player_names),
                    },
                    "players": list(visits),
                }
                with record_path.open("a", encoding="utf-8") as output:
                    output.write(json.dumps(record, ensure_ascii=False) + "\n")
                _timestamped_log(
                    f"今日第{reopen_index}次重开玩家记录已写入："
                    f"原因={_reopen_reason_text(reopen_reason)}，"
                    f"共{len(visits)}次进入尝试，"
                    f"成功{max(0, int(round_player_entries))}人，"
                    f"失败{max(0, int(round_player_entry_failures))}人。"
                )
                return reopen_index
            except Exception as exc:
                _timestamped_log(f"重开玩家汇总日志保存失败：{exc}")
                return 0


@dataclass
class _PlayerVisit:
    player_name: str
    visit_index_for_player: int
    incoming_at_utc: datetime
    arrived_at_utc: datetime | None = None
    ended_at_utc: datetime | None = None
    ended_by_freeze: bool = False
    ended_by_network_error: bool = False
    available_rewards: int = 0
    completed_rewards: int = 0

    def as_log_record(self) -> dict[str, object]:
        end = self.ended_at_utc or datetime.now(timezone.utc)
        successful = self.arrived_at_utc is not None
        loading_end = self.arrived_at_utc or end
        loading_seconds = max(
            0,
            round((loading_end - self.incoming_at_utc).total_seconds()),
        )
        room_seconds = (
            max(0, round((end - self.arrived_at_utc).total_seconds()))
            if self.arrived_at_utc is not None
            else None
        )

        def local_time(value: datetime | None) -> str | None:
            if value is None:
                return None
            return value.astimezone(BEIJING_TIMEZONE).replace(
                tzinfo=None
            ).isoformat(timespec="seconds")

        return {
            "player_name": self.player_name,
            "visit_index_for_player": self.visit_index_for_player,
            "result": "entered" if successful else "entry_failed",
            "failure_reason": (
                None
                if successful
                else (
                    "network_error_before_arrival"
                    if self.ended_by_network_error
                    else (
                        "room_closed_before_arrival"
                        if self.ended_by_freeze
                        else "left_before_arrival"
                    )
                )
            ),
            "failure_reason_text": (
                None
                if successful
                else (
                    "网络连接错误时仍未抵达"
                    if self.ended_by_network_error
                    else (
                        "房间关闭时仍未抵达"
                        if self.ended_by_freeze
                        else "到达前返回"
                    )
                )
            ),
            "incoming": local_time(self.incoming_at_utc),
            "arrived": local_time(self.arrived_at_utc),
            "ended": local_time(end),
            "loading_seconds": loading_seconds,
            "room_seconds": room_seconds,
            "ended_by_room_freeze": self.ended_by_freeze,
            "ended_by_network_error": self.ended_by_network_error,
            "available_reward_tasks": min(3, self.available_rewards),
            "completed_reward_tasks": min(3, self.completed_rewards),
            "description": (
                f"{self.player_name}，加载所用时长{loading_seconds}秒，"
                "进入失败（"
                + (
                    "网络连接错误时仍未抵达"
                    if self.ended_by_network_error
                    else (
                        "房间关闭时仍未抵达"
                        if self.ended_by_freeze
                        else "到达前返回"
                    )
                )
                + "）"
                if not successful
                else (
                    f"{self.player_name}，本次进入时间"
                    f"{local_time(self.arrived_at_utc)}，加载所用时长"
                    f"{loading_seconds}秒，在房间内总共时长{room_seconds}秒，"
                    f"在房间期间可获取{min(3, self.available_rewards)}个任务奖励，"
                    f"在房间期间做了{min(3, self.completed_rewards)}个任务"
                )
            ),
        }


class RoomPlayerTracker:
    """Maintain the players currently travelling to or inside the room."""

    _NOTIFICATION_TEXT = {
        "incoming": "即将抵达。",
        "arrived": "已抵达。",
        "left": "回去了。",
    }

    def __init__(
        self,
        archive: StampCodeArchive,
        run_counter: Macro6RunCounter,
    ) -> None:
        self._archive = archive
        self._run_counter = run_counter
        self._lock = threading.Lock()
        self._players: dict[str, str] = {}
        self._disconnected_players: tuple[tuple[str, str], ...] = ()
        self._showing_disconnected_room = False
        self._round_generation = 0
        self._active_visits: dict[str, _PlayerVisit] = {}
        self._round_visits: list[_PlayerVisit] = []
        self._visit_counts_by_player: dict[str, int] = {}
        self._round_player_entries = 0
        self._round_player_entry_failures = 0
        self._previous_round_player_entries = 0
        self._last_event_sequence_by_player: dict[str, int] = {}
        self._left_sequence_by_player: dict[str, int] = {}

    def _canonical_name_locked(
        self,
        player_name: str,
        status: str,
    ) -> str:
        folded_name = player_name.casefold()
        candidate_names = list(self._players)
        for departed_name in self._left_sequence_by_player:
            if departed_name not in self._players:
                candidate_names.append(departed_name)
        for existing_name in candidate_names:
            if existing_name.casefold() == folded_name:
                return existing_name

        containment_pool = (
            candidate_names
            if status == "left"
            else (
                list(self._left_sequence_by_player)
                if status == "arrived"
                else []
            )
        )
        if containment_pool:
            # Leave banners can be very short.  OCR may preserve only the
            # first/last glyph (百變怪 -> 百, 哈哈 -> 哈).  A unique containment
            # match among players actually in this room is stronger evidence
            # than the symmetric SequenceMatcher ratio, which unfairly
            # penalizes a correctly recognized but truncated name.
            containment_matches = [
                existing_name
                for existing_name in containment_pool
                if (
                    folded_name in existing_name.casefold()
                    or existing_name.casefold() in folded_name
                )
            ]
            if len(containment_matches) == 1:
                return containment_matches[0]
            if len(containment_matches) > 1:
                # A one-glyph OCR result must never guess between multiple
                # live players that all contain that glyph.
                return player_name

        best_name = player_name
        best_ratio = 0.0
        second_ratio = 0.0
        for existing_name in candidate_names:
            ratio = difflib.SequenceMatcher(
                None,
                folded_name,
                existing_name.casefold(),
            ).ratio()
            if ratio > best_ratio:
                second_ratio = best_ratio
                best_name = existing_name
                best_ratio = ratio
            elif ratio > second_ratio:
                second_ratio = ratio
        if best_ratio >= 0.78:
            return best_name
        if (
            status == "left"
            and best_ratio >= 0.60
            and best_ratio - second_ratio >= 0.15
        ):
            return best_name
        return player_name

    def apply(
        self,
        player_name: str,
        status: str,
        *,
        expected_generation: int | None = None,
        capture_sequence: int | None = None,
        captured_at_utc: datetime | None = None,
        segment_id: int | None = None,
        processing_delay_ms: int | None = None,
    ) -> bool:
        if status not in self._NOTIFICATION_TEXT:
            return False
        with self._lock:
            if (
                expected_generation is not None
                and expected_generation != self._round_generation
            ):
                return False
            if self._showing_disconnected_room:
                return False
            canonical_name = self._canonical_name_locked(player_name, status)
            if status == "left" and canonical_name != player_name:
                _timestamped_log(
                    "离开通知玩家名自动匹配："
                    f"OCR={player_name!r} → 房间内={canonical_name!r}。"
                )
            if capture_sequence is not None:
                sequence = int(capture_sequence)
                previous_sequence = self._last_event_sequence_by_player.get(
                    canonical_name,
                    0,
                )
                if sequence <= previous_sequence:
                    _timestamped_log(
                        "拒绝玩家陈旧事件："
                        f"{canonical_name}/{status} capture_sequence="
                        f"{sequence} <= {previous_sequence}。"
                    )
                    return False
                if (
                    status == "arrived"
                    and canonical_name in self._left_sequence_by_player
                ):
                    self._last_event_sequence_by_player[
                        canonical_name
                    ] = sequence
                    _timestamped_log(
                        "拒绝离开后的迟到抵达事件："
                        f"{canonical_name} capture_sequence={sequence}；"
                        "必须先识别到更新的即将抵达。"
                    )
                    return False
                if status == "incoming":
                    self._left_sequence_by_player.pop(
                        canonical_name,
                        None,
                    )
            previous_status = self._players.get(canonical_name)
            now_utc = captured_at_utc or datetime.now(timezone.utc)
            if now_utc.tzinfo is None:
                now_utc = now_utc.replace(tzinfo=timezone.utc)
            visit_index_for_player = 0
            player_entered = False
            player_entry_failed = False
            arrival_task_increment = 0
            if status == "incoming":
                room_status = "路上"
                changed = previous_status != room_status
                self._players[canonical_name] = room_status
                action = "add_or_update"
                if canonical_name not in self._active_visits:
                    visit_index = (
                        self._visit_counts_by_player.get(canonical_name, 0) + 1
                    )
                    self._visit_counts_by_player[canonical_name] = visit_index
                    self._active_visits[canonical_name] = _PlayerVisit(
                        player_name=canonical_name,
                        visit_index_for_player=visit_index,
                        incoming_at_utc=now_utc,
                    )
                visit_index_for_player = self._active_visits[
                    canonical_name
                ].visit_index_for_player
            elif status == "arrived":
                room_status = "到达"
                changed = previous_status != room_status
                self._players[canonical_name] = room_status
                action = "add_or_update"
                visit = self._active_visits.get(canonical_name)
                if visit is None:
                    visit_index = (
                        self._visit_counts_by_player.get(canonical_name, 0) + 1
                    )
                    self._visit_counts_by_player[canonical_name] = visit_index
                    visit = _PlayerVisit(
                        player_name=canonical_name,
                        visit_index_for_player=visit_index,
                        incoming_at_utc=now_utc,
                    )
                    self._active_visits[canonical_name] = visit
                if visit.arrived_at_utc is None:
                    visit.arrived_at_utc = now_utc
                    self._round_player_entries += 1
                    player_entered = True
                    arrival_task_increment = max(
                        0,
                        visit.available_rewards - visit.completed_rewards,
                    )
                    visit.completed_rewards = visit.available_rewards
                visit_index_for_player = visit.visit_index_for_player
            else:
                room_status = ""
                changed = canonical_name in self._players
                self._players.pop(canonical_name, None)
                action = "remove"
                visit = self._active_visits.pop(canonical_name, None)
                if visit is not None:
                    visit.ended_at_utc = now_utc
                    self._round_visits.append(visit)
                    visit_index_for_player = visit.visit_index_for_player
                    if visit.arrived_at_utc is None:
                        self._round_player_entry_failures += 1
                        player_entry_failed = True
            if capture_sequence is not None:
                sequence = int(capture_sequence)
                self._last_event_sequence_by_player[canonical_name] = sequence
                if status == "left" and changed:
                    self._left_sequence_by_player[canonical_name] = sequence
            snapshot = tuple(self._players.items())
        # A distinct visible notification is useful evidence even if the
        # resulting room state was already identical, so it is always logged.
        self._archive.record_player_status(
            player_name=canonical_name,
            visit_index_for_player=visit_index_for_player,
            notification_status=self._NOTIFICATION_TEXT[status],
            room_status=room_status,
            action=action,
            room_players=snapshot,
            ocr_player_name=player_name,
            capture_sequence=capture_sequence,
            captured_at_utc=captured_at_utc,
            segment_id=segment_id,
            processing_delay_ms=processing_delay_ms,
        )
        if player_entered:
            count = self._run_counter.record_player_entered(canonical_name)
            if arrival_task_increment > 0:
                self._run_counter.record_player_completed_tasks(
                    {canonical_name: arrival_task_increment}
                )
                _timestamped_log(
                    f"玩家{canonical_name}确认已抵达："
                    f"补计加载期间参与完成任务{arrival_task_increment}次。"
                )
            _timestamped_log(
                f"玩家成功进入累计：本轮{self.current_round_entries()}人，"
                f"本日{count.daily_player_entries}人，"
                f"总计{count.total_player_entries}人。"
            )
        if player_entry_failed:
            count = self._run_counter.record_player_entry_failures(
                player_names=(canonical_name,),
                reason="left_before_arrival",
            )
            _timestamped_log(
                f"玩家进入失败累计：本轮"
                f"{self.current_round_entry_failures()}人，"
                f"本日{count.daily_player_entry_failures}人，"
                f"总计{count.total_player_entry_failures}人。"
            )
        return changed

    def record_reward(self, task_count: int) -> None:
        """Accumulate accepted REWARD rows in each active visit window."""
        count = min(3, max(0, int(task_count)))
        if count <= 0:
            return
        with self._lock:
            if self._showing_disconnected_room:
                return
            completed_task_increments: dict[str, int] = {}
            for visit in self._active_visits.values():
                visit.available_rewards = min(
                    3,
                    visit.available_rewards + count,
                )
                if visit.arrived_at_utc is not None:
                    completed_before = visit.completed_rewards
                    visit.completed_rewards = min(
                        3,
                        visit.completed_rewards + count,
                    )
                    completed_increment = (
                        visit.completed_rewards - completed_before
                    )
                    if completed_increment > 0:
                        completed_task_increments[visit.player_name] = (
                            completed_task_increments.get(
                                visit.player_name,
                                0,
                            )
                            + completed_increment
                        )
            player_progress = tuple(
                {
                    "player_name": visit.player_name,
                    "visit_index_for_player": visit.visit_index_for_player,
                    "room_status": (
                        "到达" if visit.arrived_at_utc is not None else "路上"
                    ),
                    "available_reward_tasks": visit.available_rewards,
                    "completed_reward_tasks": visit.completed_rewards,
                }
                for visit in self._active_visits.values()
            )
        self._run_counter.record_player_completed_tasks(
            completed_task_increments
        )
        self._archive.record_player_reward(
            reward_lines_added=count,
            players=player_progress,
        )

    def mark_room_disconnected(
        self,
        reopen_reason: str = "normal_reopen",
    ) -> None:
        """Freeze and close the room exactly when the first HOME is sent."""
        frozen_at_utc = datetime.now(timezone.utc)
        network_error_reopen = reopen_reason == "network_connection_error"
        with self._lock:
            if self._showing_disconnected_room:
                return
            self._disconnected_players = tuple(self._players.items())
            self._showing_disconnected_room = True
            frozen_failures = 0
            frozen_failure_names: list[str] = []
            for visit in self._active_visits.values():
                visit.ended_at_utc = frozen_at_utc
                visit.ended_by_freeze = True
                visit.ended_by_network_error = network_error_reopen
                self._round_visits.append(visit)
                if visit.arrived_at_utc is None:
                    frozen_failures += 1
                    frozen_failure_names.append(visit.player_name)
            self._round_player_entry_failures += frozen_failures
            self._active_visits.clear()
            visits = tuple(
                visit.as_log_record() for visit in self._round_visits
            )
            self._round_visits.clear()
            round_player_entries = self._round_player_entries
            round_player_entry_failures = self._round_player_entry_failures
            self._previous_round_player_entries = round_player_entries
        if frozen_failures:
            self._run_counter.record_player_entry_failures(
                frozen_failures,
                player_names=tuple(frozen_failure_names),
                reason=(
                    "network_error_before_arrival"
                    if network_error_reopen
                    else "room_closed_before_arrival"
                ),
            )
        reopen_index = self._archive.record_room_reopen_summary(
            visits,
            frozen_at_utc,
            round_player_entries,
            round_player_entry_failures,
            reopen_reason=reopen_reason,
            network_error_player_names=(
                tuple(frozen_failure_names)
                if network_error_reopen
                else ()
            ),
        )
        if network_error_reopen and frozen_failure_names:
            _timestamped_log(
                "网络连接错误时尚未抵达的玩家："
                + "、".join(frozen_failure_names)
                + "；已归入网络连接错误，不计入普通重开冻结分类。"
            )
        _timestamped_log(
            "重开第一个HOME即将按下：已冻结断开时房间内玩家列表"
            f"并结算今日第{reopen_index or '?'}次重开。"
        )

    def clear_for_new_code(self) -> None:
        with self._lock:
            self._players.clear()
            self._disconnected_players = ()
            self._showing_disconnected_room = False
            self._round_generation += 1
            self._active_visits.clear()
            self._round_visits.clear()
            self._visit_counts_by_player.clear()
            self._round_player_entries = 0
            self._round_player_entry_failures = 0
            self._last_event_sequence_by_player.clear()
            self._left_sequence_by_player.clear()
        _timestamped_log(
            "检测到新CODE：已切回房间内玩家，并清空上一房间列表。"
        )

    def snapshot(self) -> tuple[tuple[str, str], ...]:
        with self._lock:
            return tuple(self._players.items())

    def current_round_entries(self) -> int:
        with self._lock:
            return self._round_player_entries

    def current_round_entry_failures(self) -> int:
        with self._lock:
            return self._round_player_entry_failures

    def take_previous_round_entries(self) -> int:
        with self._lock:
            count = self._previous_round_player_entries
            self._previous_round_player_entries = 0
            return count

    def generation(self) -> int:
        with self._lock:
            return self._round_generation

    def display_snapshot(
        self,
    ) -> tuple[bool, tuple[tuple[str, str], ...]]:
        with self._lock:
            if self._showing_disconnected_room:
                return True, self._disconnected_players
            return False, tuple(self._players.items())


def _fractional_crop(
    frame: np.ndarray,
    roi: tuple[float, float, float, float],
) -> np.ndarray:
    height, width = frame.shape[:2]
    x1, y1, x2, y2 = roi
    left = max(0, min(width - 1, round(x1 * width)))
    top = max(0, min(height - 1, round(y1 * height)))
    right = max(left + 1, min(width, round(x2 * width)))
    bottom = max(top + 1, min(height, round(y2 * height)))
    return frame[top:bottom, left:right]


def _edge_similarity(first: np.ndarray, second: np.ndarray) -> float:
    size = (240, 96)
    first_gray = cv2.cvtColor(
        cv2.resize(first, size, interpolation=cv2.INTER_AREA),
        cv2.COLOR_BGR2GRAY,
    )
    second_gray = cv2.cvtColor(
        cv2.resize(second, size, interpolation=cv2.INTER_AREA),
        cv2.COLOR_BGR2GRAY,
    )
    first_edges = cv2.Canny(first_gray, 50, 150)
    second_edges = cv2.Canny(second_gray, 50, 150)
    edge_count = int(np.count_nonzero(first_edges)) + int(
        np.count_nonzero(second_edges)
    )
    if edge_count == 0:
        return 0.0
    kernel = np.ones((3, 3), dtype=np.uint8)
    first_near = cv2.dilate(first_edges, kernel)
    second_near = cv2.dilate(second_edges, kernel)
    matched = int(
        np.count_nonzero((first_edges > 0) & (second_near > 0))
    ) + int(np.count_nonzero((second_edges > 0) & (first_near > 0)))
    return min(1.0, matched / edge_count)


def _color_error(first: np.ndarray, second: np.ndarray) -> float:
    size = (240, 96)
    first_resized = cv2.resize(first, size, interpolation=cv2.INTER_AREA)
    second_resized = cv2.resize(second, size, interpolation=cv2.INTER_AREA)
    return float(
        np.mean(
            np.abs(
                first_resized.astype(np.float32)
                - second_resized.astype(np.float32)
            )
        )
        / 255.0
    )


class PokopiaDetector:
    """Recognizers backed only by ROI features embedded in this source."""

    def __init__(self) -> None:
        self.templates: dict[str, np.ndarray] = {}
        invalid: list[str] = []
        for name, encoded in _EMBEDDED_TEMPLATE_PNG_BASE64.items():
            try:
                raw = np.frombuffer(base64.b64decode(encoded), dtype=np.uint8)
                template = cv2.imdecode(raw, cv2.IMREAD_COLOR)
            except Exception:
                template = None
            if template is None:
                invalid.append(name)
            else:
                self.templates[name] = template
        if invalid or len(self.templates) != 14:
            raise RuntimeError(
                "Pokopia watchdog 内嵌特征损坏："
                + "；".join(invalid or [f"数量={len(self.templates)}"])
            )
        packed_network_error = np.frombuffer(
            zlib.decompress(
                base64.b64decode(
                    NETWORK_ERROR_CLOSE_MASK_ZLIB_BASE64
                )
            ),
            dtype=np.uint8,
        )
        network_error_count = int(
            np.prod(NETWORK_ERROR_CLOSE_TEMPLATE_SHAPE)
        )
        self._network_error_close_mask = (
            np.unpackbits(packed_network_error)[:network_error_count]
            .reshape(NETWORK_ERROR_CLOSE_TEMPLATE_SHAPE)
            .astype(np.float32)
        )
        self._network_error_streak = 0
        self._reopen_stage_features: dict[
            str,
            tuple[tuple[float, float, float, float], np.ndarray],
        ] = {}
        reopen_feature_size = int(np.prod(REOPEN_STAGE_FEATURE_SHAPE))
        for stage_name, (roi, encoded) in REOPEN_STAGE_FEATURES.items():
            raw_feature = zlib.decompress(base64.b64decode(encoded))
            feature = np.frombuffer(raw_feature, dtype=np.uint8)
            if feature.size != reopen_feature_size:
                raise RuntimeError(
                    f"重开阶段内嵌特征损坏：{stage_name}，"
                    f"长度={feature.size}"
                )
            self._reopen_stage_features[stage_name] = (
                roi,
                feature.reshape(REOPEN_STAGE_FEATURE_SHAPE).astype(np.float32),
            )
        self._reopen_stage_candidate = "INVALID"
        self._reopen_stage_candidate_frames = 0

    def _match(
        self,
        frame: np.ndarray,
        template_name: str,
        roi: tuple[float, float, float, float],
        *,
        edge_threshold: float = FIXED_TEXT_EDGE_THRESHOLD,
        color_error_threshold: float = 0.20,
    ) -> TemplateMatch:
        current = _fractional_crop(frame, roi)
        reference = self.templates[template_name]
        edge = _edge_similarity(current, reference)
        error = _color_error(current, reference)
        return TemplateMatch(
            detected=(
                edge >= edge_threshold
                and error <= color_error_threshold
            ),
            edge_similarity=edge,
            color_error=error,
        )

    def _detect_save_finished(self, frame: np.ndarray) -> bool:
        # Detect the blue completed-download check icon, not its accompanying
        # Chinese status text.  Requiring both edge shape and color prevents
        # unrelated solid-blue screens from being accepted.
        return self._match(
            frame,
            "save_check",
            SAVE_CHECK_ROI,
            edge_threshold=0.70,
            color_error_threshold=0.16,
        ).detected

    @staticmethod
    def _detect_exact_black(frame: np.ndarray) -> bool:
        center = _fractional_crop(frame, BLACK_ROI)
        if not center.size:
            return False
        first_pixel = center[0, 0]
        # The supplied DirectShow PNGs encode digital black as 07/07/07
        # (limited-range video black).  Accept only that exact uniform value
        # or literal 00/00/00; every other value or colored pixel is OFF.
        uniform = bool(np.all(center == first_pixel))
        neutral = bool(first_pixel[0] == first_pixel[1] == first_pixel[2])
        return uniform and neutral and int(first_pixel[0]) in {0, 7}

    @staticmethod
    def _detect_code_panel(frame: np.ndarray) -> bool:
        crop = _fractional_crop(frame, CODE_PANEL_ROI)
        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        deep_blue = (
            (hsv[:, :, 0] >= 105)
            & (hsv[:, :, 0] <= 140)
            & (hsv[:, :, 1] >= 80)
            & (hsv[:, :, 2] >= 100)
            & (hsv[:, :, 2] <= 240)
        )
        channel_range = crop.max(axis=2) - crop.min(axis=2)
        white = (crop.min(axis=2) >= 200) & (channel_range <= 45)
        return bool(deep_blue.mean() >= 0.35 and white.mean() >= 0.01)

    def _detect_cursor(self, frame: np.ndarray) -> str:
        current = _fractional_crop(frame, CURSOR_AREA_ROI)
        candidates: list[tuple[float, float, str]] = []
        for template_name, label in (
            ("cursor_connect", "CONNECT"),
            ("cursor_challenge", "CHALLENGE"),
            ("cursor_shop", "SHOP"),
            ("cursor_arche", "ARCHE"),
            ("cursor_receive", "RECEIVE"),
            ("cursor_stamp", "STAMP"),
        ):
            reference = self.templates[template_name]
            candidates.append(
                (
                    _color_error(current, reference),
                    _edge_similarity(current, reference),
                    label,
                )
            )
        candidates.sort(key=lambda item: item[0])
        best_error, best_edge, best_label = candidates[0]
        second_error = candidates[1][0]
        if (
            best_error > 0.08
            or best_edge < 0.80
            or second_error - best_error < 0.015
        ):
            return "INVALID"
        return best_label

    def _detect_stamp(self, frame: np.ndarray) -> str:
        context_match = self._match(
            frame,
            "stamp_yes_context",
            STAMP_CONTEXT_ROI,
            edge_threshold=STAMP_CONTEXT_EDGE_THRESHOLD,
            color_error_threshold=0.20,
        )
        if not context_match.detected:
            return "INVALID"

        button = _fractional_crop(frame, STAMP_BUTTON_ROI)
        yes_error = _color_error(
            button,
            self.templates["stamp_yes_button"],
        )
        no_error = _color_error(
            button,
            self.templates["stamp_no_button"],
        )
        if min(yes_error, no_error) > 0.20 or abs(yes_error - no_error) < 0.02:
            return "INVALID"
        return "YES" if yes_error < no_error else "NO"

    @staticmethod
    def _detect_reward_line_count(frame: np.ndarray) -> int:
        """Count 1-3 proportional-font task rows on a REWARD page."""
        crop = _fractional_crop(frame, REWARD_LIST_ROI)
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        channel_spread = crop.max(axis=2) - crop.min(axis=2)
        # Task text and reward values are dark, nearly neutral gray.  The
        # pale purple mascot and colorful background are excluded by the
        # channel-spread constraint.
        dark_neutral = (gray < 145) & (channel_spread < 48)
        row_pixel_counts = np.count_nonzero(dark_neutral, axis=1)
        active_rows = np.flatnonzero(row_pixel_counts >= 8)
        if active_rows.size == 0:
            return 0

        line_count = 1
        previous_row = int(active_rows[0])
        maximum_intra_line_gap = max(3, round(crop.shape[0] * 0.02))
        for row in active_rows[1:]:
            current_row = int(row)
            if current_row - previous_row > maximum_intra_line_gap:
                line_count += 1
            previous_row = current_row
        return min(3, line_count)

    def _detect_network_error(self, frame: np.ndarray) -> bool:
        """Match the fixed neutral-white “关闭” button glyphs."""
        crop = _fractional_crop(frame, NETWORK_ERROR_CLOSE_ROI)
        spread = crop.max(axis=2).astype(np.int16) - crop.min(
            axis=2
        ).astype(np.int16)
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        neutral_white = ((gray >= 175) & (spread <= 38)).astype(np.uint8)
        normalized = cv2.resize(
            neutral_white * 255,
            (
                NETWORK_ERROR_CLOSE_TEMPLATE_SHAPE[1],
                NETWORK_ERROR_CLOSE_TEMPLATE_SHAPE[0],
            ),
            interpolation=cv2.INTER_AREA,
        ).astype(np.float32) / 255.0
        denominator = float(
            normalized.sum() + self._network_error_close_mask.sum()
        )
        dice = (
            2.0
            * float(
                np.minimum(
                    normalized,
                    self._network_error_close_mask,
                ).sum()
            )
            / denominator
            if denominator > 0.0
            else 0.0
        )
        if dice >= NETWORK_ERROR_CLOSE_MIN_DICE:
            self._network_error_streak += 1
        else:
            self._network_error_streak = 0
        return self._network_error_streak >= NETWORK_ERROR_CONFIRM_FRAMES

    def _detect_reopen_stage(self, frame: np.ndarray) -> str:
        best_stage = "INVALID"
        best_score = -1.0
        feature_width = REOPEN_STAGE_FEATURE_SHAPE[1]
        feature_height = REOPEN_STAGE_FEATURE_SHAPE[0]
        for stage_name, (roi, reference) in self._reopen_stage_features.items():
            crop = _fractional_crop(frame, roi)
            if stage_name == "OVERWRITE":
                # Use colour as a required gate before comparing structure.
                # This explicitly rejects the supplied disabled
                # “无法覆盖数据” screen instead of accepting its nearly
                # identical glyph/layout through grayscale correlation.
                blue = crop[:, :, 0].astype(np.int16)
                green = crop[:, :, 1].astype(np.int16)
                red = crop[:, :, 2].astype(np.int16)
                bright_orange_red = (
                    (red >= 150)
                    & ((red - green) >= 80)
                    & (green <= 100)
                    & (blue <= 80)
                )
                if (
                    float(np.count_nonzero(bright_orange_red))
                    / float(bright_orange_red.size)
                    < REOPEN_OVERWRITE_MIN_BRIGHT_RED_RATIO
                ):
                    continue
            current = cv2.resize(
                cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY),
                (feature_width, feature_height),
                interpolation=cv2.INTER_AREA,
            ).astype(np.float32)
            score = float(
                cv2.matchTemplate(
                    current,
                    reference,
                    cv2.TM_CCOEFF_NORMED,
                )[0, 0]
            )
            if np.isfinite(score) and score > best_score:
                best_stage = stage_name
                best_score = score

        if best_score < REOPEN_STAGE_MIN_CORRELATION:
            self._reopen_stage_candidate = "INVALID"
            self._reopen_stage_candidate_frames = 0
            return "INVALID"
        if best_stage == self._reopen_stage_candidate:
            self._reopen_stage_candidate_frames += 1
        else:
            self._reopen_stage_candidate = best_stage
            self._reopen_stage_candidate_frames = 1
        if self._reopen_stage_candidate_frames < REOPEN_STAGE_CONFIRM_FRAMES:
            return "INVALID"
        return best_stage

    def detect(self, frame: np.ndarray) -> PokopiaDetection:
        if frame is None or frame.size == 0 or frame.ndim != 3:
            return PokopiaDetection(
                False,
                False,
                False,
                False,
                False,
                "INVALID",
                False,
                0,
                "INVALID",
                False,
            )
        if frame.shape[:2] != (CAPTURE_HEIGHT, CAPTURE_WIDTH):
            frame = cv2.resize(
                frame,
                (CAPTURE_WIDTH, CAPTURE_HEIGHT),
                interpolation=cv2.INTER_AREA,
            )
        reward = self._match(frame, "reward", REWARD_ROI).detected
        return PokopiaDetection(
            save_finished=self._detect_save_finished(frame),
            able_access=self._match(
                frame,
                "able_access",
                ABLE_ACCESS_ROI,
            ).detected,
            black_screen=self._detect_exact_black(frame),
            connect_ok=self._match(frame, "connect", CONNECT_ROI).detected,
            code_panel=self._detect_code_panel(frame),
            cursor=self._detect_cursor(frame),
            reward=reward,
            reward_lines=(
                self._detect_reward_line_count(frame) if reward else 0
            ),
            stamp=self._detect_stamp(frame),
            network_error=self._detect_network_error(frame),
            reopen_stage=self._detect_reopen_stage(frame),
        )


def _parse_six_character_code(raw_text: str) -> str:
    normalized = unicodedata.normalize(
        "NFKC",
        str(raw_text or ""),
    ).upper().translate(_CODE_FORBIDDEN_GLYPH_TRANSLATION)
    matches = re.findall(r"(?<![A-Z0-9])[A-Z0-9]{6}(?![A-Z0-9])", normalized)
    for match in matches:
        if all(character in CODE_ALLOWED_CHARACTERS for character in match):
            return match
    compact = re.sub(r"[^A-Z0-9]", "", normalized)
    if len(compact) != 6:
        return ""
    return (
        compact
        if all(character in CODE_ALLOWED_CHARACTERS for character in compact)
        else ""
    )


def _embedded_code_glyph_templates(
) -> tuple[tuple[str, np.ndarray, float, float, float], ...]:
    packed_maps = np.frombuffer(
        zlib.decompress(
            base64.b64decode(_CODE_GLYPH_TEMPLATE_MAP_ZLIB_BASE64)
        ),
        dtype=np.uint8,
    )
    map_value_count = int(np.prod(_CODE_GLYPH_TEMPLATE_SHAPE))
    map_values = np.empty(packed_maps.size * 2, dtype=np.uint8)
    map_values[0::2] = packed_maps >> 4
    map_values[1::2] = packed_maps & 0x0F
    maps = (
        map_values[:map_value_count]
        .reshape(_CODE_GLYPH_TEMPLATE_SHAPE)
        .astype(np.float32)
        / 15.0
    )
    aspect_stats = (
        np.frombuffer(
            zlib.decompress(
                base64.b64decode(_CODE_GLYPH_TEMPLATE_ASPECT_ZLIB_BASE64)
            ),
            dtype="<u2",
        )
        .reshape((len(_CODE_GLYPH_TEMPLATE_LABELS), 3))
        .astype(np.float32)
        / 10000.0
    )
    return tuple(
        (
            label,
            reference,
            float(stats[0]),
            float(stats[1]),
            float(stats[2]),
        )
        for label, reference, stats in zip(
            _CODE_GLYPH_TEMPLATE_LABELS, maps, aspect_stats
        )
    )


def _score_embedded_code_glyph(
    glyph: np.ndarray,
    templates: tuple[tuple[str, np.ndarray, float, float, float], ...],
) -> tuple[str, float, float, float] | None:
    """Return label, shape error, confidence margin, and combined cost."""
    glyph_rows = np.flatnonzero(glyph.sum(axis=1) > 0)
    if glyph_rows.size == 0:
        return None
    glyph = glyph[glyph_rows[0] : glyph_rows[-1] + 1]
    glyph_aspect = glyph.shape[1] / max(1, glyph.shape[0])
    normalized = cv2.resize(
        glyph * 255,
        (24, 32),
        interpolation=cv2.INTER_AREA,
    ).astype(np.float32) / 255.0
    scores: dict[str, tuple[float, float]] = {}
    for label, reference, median_aspect, low_aspect, high_aspect in templates:
        shape_error = float(np.mean(np.abs(normalized - reference)))
        outside_width = (
            low_aspect - glyph_aspect
            if glyph_aspect < low_aspect
            else glyph_aspect - high_aspect
            if glyph_aspect > high_aspect
            else 0.0
        )
        lower_right_error = float(
            np.mean(np.abs(normalized[23:32, 14:24] - reference[23:32, 14:24]))
        )
        # Each character has one canonical pixel-probability map plus its
        # observed proportional-font width range.  The lower-right quadrant
        # is weighted separately so Q's protruding tail cannot collapse into
        # the closed oval of 0.
        combined_cost = (
            shape_error
            + 0.02 * abs(glyph_aspect - median_aspect)
            + 0.04 * outside_width
            + 0.06 * lower_right_error
        )
        scores[label] = (combined_cost, shape_error)
    ranked = sorted(
        (combined_cost, label, shape_error)
        for label, (combined_cost, shape_error) in scores.items()
    )
    if len(ranked) < 2:
        return None
    best_cost, best_label, best_shape_error = ranked[0]
    margin = ranked[1][0] - best_cost
    return best_label, best_shape_error, margin, best_cost


def _partition_embedded_code_run(
    mask: np.ndarray,
    left: int,
    right: int,
    character_count: int,
    templates: tuple[tuple[str, np.ndarray, float, float, float], ...],
) -> tuple[float, tuple[tuple[str, float, float], ...]] | None:
    """Find variable-width glyph boundaries inside one connected run."""
    block = mask[:, left : right + 1]
    width = block.shape[1]
    font_height = max(1, mask.shape[0])
    minimum_width = max(2, int(font_height * 0.16))
    maximum_width = max(minimum_width, int(np.ceil(font_height * 1.65)))
    if not minimum_width * character_count <= width <= maximum_width * character_count:
        return None

    score_cache: dict[
        tuple[int, int], tuple[str, float, float, float] | None
    ] = {}

    def score_segment(start: int, end: int):
        key = (start, end)
        if key not in score_cache:
            score_cache[key] = _score_embedded_code_glyph(
                block[:, start:end],
                templates,
            )
        return score_cache[key]

    states: dict[
        int, tuple[float, tuple[tuple[str, float, float], ...]]
    ] = {0: (0.0, ())}
    for part_index in range(character_count):
        next_states: dict[
            int, tuple[float, tuple[tuple[str, float, float], ...]]
        ] = {}
        remaining_parts = character_count - part_index - 1
        for start, (current_cost, current_result) in states.items():
            smallest_end = start + minimum_width
            largest_end = min(width, start + maximum_width)
            for end in range(smallest_end, largest_end + 1):
                remaining_width = width - end
                if not (
                    minimum_width * remaining_parts
                    <= remaining_width
                    <= maximum_width * remaining_parts
                ):
                    continue
                scored = score_segment(start, end)
                if scored is None:
                    continue
                label, shape_error, margin, segment_cost = scored
                candidate = (
                    current_cost + segment_cost,
                    current_result + ((label, shape_error, margin),),
                )
                previous = next_states.get(end)
                if previous is None or candidate[0] < previous[0]:
                    next_states[end] = candidate
        states = next_states
        if not states:
            return None
    return states.get(width)


def _code_run_character_allocations(
    run_count: int,
    total_characters: int = 6,
) -> tuple[tuple[int, ...], ...]:
    """Enumerate positive character counts for each visible column run."""
    allocations: list[tuple[int, ...]] = []

    def visit(remaining_runs: int, remaining_characters: int, prefix):
        if remaining_runs == 1:
            allocations.append(prefix + (remaining_characters,))
            return
        maximum_here = remaining_characters - remaining_runs + 1
        for count in range(1, maximum_here + 1):
            visit(
                remaining_runs - 1,
                remaining_characters - count,
                prefix + (count,),
            )

    if 1 <= run_count <= total_characters:
        visit(run_count, total_characters, ())
    return tuple(allocations)


def _recognize_embedded_code_glyphs(crop: np.ndarray) -> str:
    """Recognize supplied Pokopia glyphs without installed OCR fonts."""
    if crop is None or crop.size == 0 or crop.ndim != 3:
        return ""
    bgr = crop[:, :, :3]
    channel_range = bgr.max(axis=2) - bgr.min(axis=2)
    mask = ((bgr.min(axis=2) >= 170) & (channel_range <= 80)).astype(
        np.uint8
    )
    active_rows = np.flatnonzero(mask.sum(axis=1) > 0)
    active_columns = np.flatnonzero(mask.sum(axis=0) > 0)
    if active_rows.size == 0 or active_columns.size == 0:
        return ""
    mask = mask[
        active_rows[0] : active_rows[-1] + 1,
        active_columns[0] : active_columns[-1] + 1,
    ]

    column_active = np.flatnonzero(mask.sum(axis=0) > 0)
    runs: list[list[int]] = []
    for column in column_active:
        value = int(column)
        if not runs or value > runs[-1][1] + 1:
            runs.append([value, value])
        else:
            runs[-1][1] = value
    if not runs:
        return ""

    if len(runs) > 6:
        return ""

    templates = _embedded_code_glyph_templates()
    partition_cache: dict[
        tuple[int, int],
        tuple[float, tuple[tuple[str, float, float], ...]] | None,
    ] = {}
    best_partition = None
    for allocation in _code_run_character_allocations(len(runs)):
        total_cost = 0.0
        result: tuple[tuple[str, float, float], ...] = ()
        valid = True
        for run_index, ((left, right), character_count) in enumerate(
            zip(runs, allocation)
        ):
            cache_key = (run_index, character_count)
            if cache_key not in partition_cache:
                partition_cache[cache_key] = _partition_embedded_code_run(
                    mask,
                    left,
                    right,
                    character_count,
                    templates,
                )
            partition = partition_cache[cache_key]
            if partition is None:
                valid = False
                break
            partition_cost, partition_result = partition
            total_cost += partition_cost
            result += partition_result
        if valid and (
            best_partition is None or total_cost < best_partition[0]
        ):
            best_partition = (total_cost, result)
    if best_partition is None:
        return ""
    recognized = best_partition[1]
    if any(
        shape_error > 0.26 or margin < 0.008
        for _, shape_error, margin in recognized
    ):
        return ""
    return "".join(label for label, _, _ in recognized)


def _macro6_windows_ocr(
    image_path: Path,
    language_tag: str = "en-US",
) -> str:
    """Run Windows Runtime OCR locally without any Macro5 dependency."""
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
$file = Await-Result ([Windows.Storage.StorageFile]::GetFileFromPathAsync($env:MACRO6_OCR_IMAGE)) ([Windows.Storage.StorageFile])
$stream = Await-Result ($file.OpenAsync([Windows.Storage.FileAccessMode]::Read)) ([Windows.Storage.Streams.IRandomAccessStream])
$decoder = Await-Result ([Windows.Graphics.Imaging.BitmapDecoder]::CreateAsync($stream)) ([Windows.Graphics.Imaging.BitmapDecoder])
$bitmap = Await-Result ($decoder.GetSoftwareBitmapAsync()) ([Windows.Graphics.Imaging.SoftwareBitmap])
$language = New-Object Windows.Globalization.Language($env:MACRO6_OCR_LANGUAGE)
$engine = [Windows.Media.Ocr.OcrEngine]::TryCreateFromLanguage($language)
if ($null -eq $engine) { $engine = [Windows.Media.Ocr.OcrEngine]::TryCreateFromUserProfileLanguages() }
if ($null -eq $engine) { exit 2 }
$result = Await-Result ($engine.RecognizeAsync($bitmap)) ([Windows.Media.Ocr.OcrResult])
Write-Output $result.Text
'''
    environment = dict(os.environ)
    environment["MACRO6_OCR_IMAGE"] = str(image_path.resolve())
    environment["MACRO6_OCR_LANGUAGE"] = language_tag or "en-US"
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
            creationflags=getattr(
                subprocess,
                "BELOW_NORMAL_PRIORITY_CLASS",
                0,
            ),
        )
        return completed.stdout.strip() if completed.returncode == 0 else ""
    except Exception:
        return ""


def _embedded_player_status_templates() -> np.ndarray:
    """Decode the independent white-status glyph probability maps."""
    packed = np.frombuffer(
        zlib.decompress(
            base64.b64decode(PLAYER_STATUS_TEMPLATE_MAP_ZLIB_BASE64)
        ),
        dtype=np.uint8,
    )
    values = np.empty(packed.size * 2, dtype=np.uint8)
    values[0::2] = packed >> 4
    values[1::2] = packed & 0x0F
    count = int(np.prod(PLAYER_STATUS_TEMPLATE_SHAPE))
    return (
        values[:count]
        .reshape(PLAYER_STATUS_TEMPLATE_SHAPE)
        .astype(np.float32)
        / 15.0
    )


def _embedded_player_recycle_notice() -> np.ndarray:
    """Decode the fixed recycle-bin system-notice glyph mask."""
    values = np.frombuffer(
        zlib.decompress(
            base64.b64decode(PLAYER_RECYCLE_NOTICE_ZLIB_BASE64)
        ),
        dtype=np.uint8,
    )
    count = int(np.prod(PLAYER_RECYCLE_NOTICE_SHAPE))
    if values.size != count:
        raise ValueError("embedded recycle notice has an invalid size")
    return (
        values.reshape(PLAYER_RECYCLE_NOTICE_SHAPE).astype(np.float32)
        / 255.0
    )


def _embedded_player_connection_icon() -> np.ndarray:
    """Decode the compact globe-network icon probability map."""
    packed = np.frombuffer(
        zlib.decompress(
            base64.b64decode(PLAYER_CONNECTION_ICON_MAP_ZLIB_BASE64)
        ),
        dtype=np.uint8,
    )
    values = np.empty(packed.size * 2, dtype=np.uint8)
    values[0::2] = packed >> 4
    values[1::2] = packed & 0x0F
    count = int(np.prod(PLAYER_CONNECTION_ICON_SHAPE))
    return (
        values[:count]
        .reshape(PLAYER_CONNECTION_ICON_SHAPE)
        .astype(np.float32)
        / 15.0
    )


def _embedded_player_connection_icon_color() -> np.ndarray:
    """Decode the embedded three-channel BGR network-icon template."""
    values = np.frombuffer(
        zlib.decompress(
            base64.b64decode(PLAYER_CONNECTION_ICON_COLOR_ZLIB_BASE64)
        ),
        dtype=np.uint8,
    )
    count = int(np.prod(PLAYER_CONNECTION_ICON_COLOR_SHAPE))
    if values.size != count:
        raise ValueError("invalid embedded player connection icon color map")
    return (
        values.reshape(PLAYER_CONNECTION_ICON_COLOR_SHAPE).astype(np.float32)
        / 255.0
    )


def _normalize_player_ocr_text(raw_text: str) -> str:
    text = unicodedata.normalize("NFKC", str(raw_text or ""))
    for fixed_status in ("即将抵达", "已抵达", "回去了", "回去", "抵达"):
        text = text.replace(fixed_status, "")
    text = re.sub(r"[\s。．.，,：:；;]+", "", text)
    text = text.strip("-_—·|[]【】()（）")
    if not 1 <= len(text) <= 24:
        return ""
    return text


def _lower_current_thread_priority() -> None:
    """Keep OCR work below interactive Windows input and foreground apps."""
    if sys.platform != "win32":
        return
    try:
        import ctypes

        kernel32 = ctypes.windll.kernel32
        current_thread = kernel32.GetCurrentThread()
        # THREAD_PRIORITY_BELOW_NORMAL = -1.  This changes only the calling
        # worker thread, not capture, controller, or watchdog display threads.
        kernel32.SetThreadPriority(current_thread, -1)
    except Exception:
        pass


_PLAYER_RAPIDOCR_LOCK = threading.Lock()
_PLAYER_RAPIDOCR_ENGINE = None
_PLAYER_RAPIDOCR_INITIALIZED = False
_PLAYER_RAPIDOCR_ERROR_LOGGED = False
_PLAYER_NAME_GLYPH_ATLAS: np.ndarray | None = None


def _embedded_player_name_glyph_atlas() -> np.ndarray:
    """Decode the compact per-character four-bit orange glyph atlas."""
    global _PLAYER_NAME_GLYPH_ATLAS
    if _PLAYER_NAME_GLYPH_ATLAS is not None:
        return _PLAYER_NAME_GLYPH_ATLAS
    packed = np.frombuffer(
        zlib.decompress(
            base64.b64decode(PLAYER_NAME_GLYPH_ATLAS_ZLIB_BASE64)
        ),
        dtype=np.uint8,
    )
    value_count = (
        len(PLAYER_NAME_GLYPH_LABELS)
        * PLAYER_NAME_GLYPH_SHAPE[0]
        * PLAYER_NAME_GLYPH_SHAPE[1]
    )
    quantized = np.empty(value_count, dtype=np.uint8)
    quantized[0::2] = packed >> 4
    quantized[1::2] = packed & 0x0F
    _PLAYER_NAME_GLYPH_ATLAS = (
        quantized.reshape(
            len(PLAYER_NAME_GLYPH_LABELS),
            *PLAYER_NAME_GLYPH_SHAPE,
        ).astype(np.float32)
        / 15.0
    )
    return _PLAYER_NAME_GLYPH_ATLAS


def _recognize_embedded_player_name_glyphs(
    player_name_crop: np.ndarray,
) -> tuple[str, float, str]:
    """Recognize confident full-width glyph slots before invoking OCR."""
    if player_name_crop.size == 0 or player_name_crop.ndim != 3:
        return "", 0.0, ""
    blue, green, red = cv2.split(player_name_crop[:, :, :3])
    orange_mask = (
        (red > 180)
        & (green > 70)
        & (green < 205)
        & (blue < 100)
        & (red.astype(np.int16) > green.astype(np.int16) + 35)
    ).astype(np.uint8)
    rows, columns = np.where(orange_mask)
    if rows.size < 24:
        return "", 0.0, ""
    glyph_line = orange_mask[
        rows.min() : rows.max() + 1,
        columns.min() : columns.max() + 1,
    ]
    height, width = glyph_line.shape
    if height <= 0:
        return "", 0.0, ""
    aspect = width / height
    slot_count = int(round(aspect / 1.08))
    if not 1 <= slot_count <= 12:
        return "", 0.0, ""
    slot_aspect = aspect / slot_count
    if abs(slot_aspect - 1.08) > 0.20:
        return "", 0.0, ""

    atlas = _embedded_player_name_glyph_atlas()
    edges = np.linspace(0, width, slot_count + 1).round().astype(int)
    recognized: list[str] = []
    errors: list[float] = []
    margins: list[float] = []
    for index in range(slot_count):
        slot = glyph_line[:, edges[index] : edges[index + 1]]
        if slot.size == 0:
            return "", 0.0, ""
        normalized = cv2.resize(
            slot * 255,
            (PLAYER_NAME_GLYPH_SHAPE[1], PLAYER_NAME_GLYPH_SHAPE[0]),
            interpolation=cv2.INTER_AREA,
        ).astype(np.float32) / 255.0
        scores = np.mean(
            np.abs(atlas - normalized[None, :, :]),
            axis=(1, 2),
        )
        ranking = np.argsort(scores)
        best_index = int(ranking[0])
        best_error = float(scores[best_index])
        second_error = float(scores[int(ranking[1])])
        recognized.append(PLAYER_NAME_GLYPH_LABELS[best_index])
        errors.append(best_error)
        margins.append(second_error - best_error)

    maximum_error = max(errors)
    minimum_margin = min(margins)
    if maximum_error > 0.16 or minimum_margin < 0.04:
        return "", 0.0, (
            f"glyph-rejected(max_error={maximum_error:.3f}, "
            f"min_margin={minimum_margin:.3f}, slots={slot_count})"
        )
    text_value = "".join(recognized)
    confidence = max(0.0, min(0.99, 1.0 - maximum_error * 2.5))
    return text_value, confidence, (
        f"glyph-atlas={text_value}(max_error={maximum_error:.3f}, "
        f"min_margin={minimum_margin:.3f}, slots={slot_count})"
    )


def _player_name_ocr_variants(
    player_name_crop: np.ndarray,
) -> tuple[tuple[str, np.ndarray], ...]:
    """Build OCR inputs from the untouched, original-resolution name crop."""
    if player_name_crop.size == 0:
        return ()
    height = max(1, int(player_name_crop.shape[0]))
    scale = max(1.0, 96.0 / height)
    enlarged = cv2.resize(
        player_name_crop,
        None,
        fx=scale,
        fy=scale,
        interpolation=cv2.INTER_CUBIC,
    )
    # Preserve antialiased edges: orange ink has red clearly above blue/green,
    # while the pale banner background has almost no such channel difference.
    blue, green, red = cv2.split(enlarged.astype(np.float32))
    ink_strength = np.maximum(red - blue, red - green)
    strong_orange = (
        (red > 180)
        & (green > 70)
        & (green < 205)
        & (blue < 100)
        & (red > green + 35)
    )
    # Only pixels connected to genuine orange ink may reach OCR. This removes
    # the dark first stroke of the following status text and colored HUD lines
    # that sometimes remain inside the geometrically cropped right margin.
    orange_neighborhood = cv2.dilate(
        strong_orange.astype(np.uint8),
        np.ones((3, 3), dtype=np.uint8),
        iterations=1,
    ).astype(bool)
    orange_support = (ink_strength > 8.0) & orange_neighborhood
    orange_original = np.full_like(enlarged, 255)
    orange_original[orange_support] = enlarged[orange_support]

    positive = ink_strength[ink_strength > 8.0]
    reference = float(np.percentile(positive, 92)) if positive.size else 1.0
    soft_ink = np.clip(
        ink_strength / max(24.0, reference),
        0.0,
        1.0,
    ) * orange_neighborhood
    soft_mask = (255.0 * (1.0 - soft_ink)).astype(np.uint8)
    _, binary_mask = cv2.threshold(
        soft_mask,
        0,
        255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU,
    )
    soft_bgr = cv2.cvtColor(soft_mask, cv2.COLOR_GRAY2BGR)
    binary_bgr = cv2.cvtColor(binary_mask, cv2.COLOR_GRAY2BGR)
    return (
        # Restore the original antialiased crop that gave the early player
        # OCR its best Chinese accuracy.  Geometry below is measured from
        # orange pixels only, so a dark status stroke cannot make a longer
        # OCR string valid.
        ("raw-original", enlarged),
        ("orange-original", orange_original),
        ("orange-soft", soft_bgr),
        ("orange-binary", binary_bgr),
    )


def _single_latin_player_name_fallback(
    player_name_crop: np.ndarray,
) -> str:
    """Recognize a strongly shaped single uppercase L without invoking OCR."""
    if (
        player_name_crop.size == 0
        or player_name_crop.ndim != 3
    ):
        return ""
    blue, green, red = cv2.split(player_name_crop[:, :, :3])
    red_i = red.astype(np.int16)
    green_i = green.astype(np.int16)
    orange_mask = (
        (red > 180)
        & (green > 70)
        & (green < 205)
        & (blue < 100)
        & (red_i > green_i + 35)
    )
    rows, columns = np.where(orange_mask)
    if rows.size < 24:
        return ""
    glyph = orange_mask[
        rows.min() : rows.max() + 1,
        columns.min() : columns.max() + 1,
    ]
    height, width = glyph.shape
    if height < 12 or width < 5 or width > height * 0.75:
        return ""

    upper_bottom = max(1, int(round(height * 0.75)))
    left_end = max(1, int(round(width * 0.35)))
    upper = glyph[:upper_bottom]
    upper_ink = int(upper.sum())
    if upper_ink <= 0:
        return ""
    upper_right_ratio = float(upper[:, left_end:].sum()) / upper_ink
    stem_row_ratio = float(
        np.mean(np.any(upper[:, :left_end], axis=1))
    )
    foot = glyph[max(0, height - max(2, int(round(height * 0.20)))) :]
    foot_width_ratio = float(np.mean(np.any(foot, axis=0)))
    if (
        upper_right_ratio <= 0.08
        and stem_row_ratio >= 0.90
        and foot_width_ratio >= 0.70
    ):
        return "L"
    return ""


def _player_name_orange_aspect(
    player_name_image: np.ndarray,
) -> float | None:
    """Measure only the visible orange name width divided by glyph height."""
    if player_name_image.size == 0 or player_name_image.ndim != 3:
        return None
    blue, green, red = cv2.split(player_name_image[:, :, :3])
    red_i = red.astype(np.int16)
    green_i = green.astype(np.int16)
    orange_mask = (
        (red > 180)
        & (green > 70)
        & (green < 205)
        & (blue < 100)
        & (red_i > green_i + 35)
    )
    rows, columns = np.where(orange_mask)
    if rows.size < 24:
        return None
    ink_width = int(columns.max() - columns.min() + 1)
    ink_height = int(rows.max() - rows.min() + 1)
    return ink_width / ink_height if ink_height > 0 else None


def _player_name_character_advance(character: str) -> float:
    """Estimate Nintendo UI advance width without assuming equal glyphs."""
    if character == "\u200d" or unicodedata.combining(character):
        return 0.0
    if 0xFE00 <= ord(character) <= 0xFE0F:
        return 0.0
    east_asian_width = unicodedata.east_asian_width(character)
    if east_asian_width in {"W", "F"}:
        return 1.0
    if character in "ilI1|![](){}'`.,:;_":
        return 0.32
    if character in "MWmw@%&":
        return 0.85
    if character.isdigit():
        return 0.58
    if "A" <= character <= "Z":
        return 0.62
    if "a" <= character <= "z":
        return 0.52
    if east_asian_width == "A":
        return 0.75
    return 0.58


def _player_name_geometry_error(
    text_value: str,
    orange_aspect: float | None,
) -> float:
    """Return relative disagreement between OCR text and orange ink width."""
    if not text_value or orange_aspect is None or orange_aspect <= 0.0:
        return 1.0
    advance = sum(
        _player_name_character_advance(character)
        for character in text_value
    )
    if advance <= 0.0:
        return 1.0
    predicted_aspect = advance * 1.08
    return abs(predicted_aspect - orange_aspect) / max(
        predicted_aspect,
        orange_aspect,
    )


def _player_name_candidate_is_complete(
    text_value: str,
    orange_aspect: float | None,
) -> bool:
    """Accept only text whose character advances fit the orange name width."""
    return _player_name_geometry_error(text_value, orange_aspect) <= 0.195


def _player_name_character_count(text_value: str) -> int:
    """Count visible name characters without assuming equal glyph widths."""
    return sum(
        1
        for character in text_value
        if character != "\u200d"
        and not unicodedata.combining(character)
        and not 0xFE00 <= ord(character) <= 0xFE0F
    )


def _player_name_group_rank(
    group: list[tuple[str, float, str]],
    orange_aspect: float | None,
) -> tuple[int, float, float]:
    """Restore source consensus first; width and confidence break ties."""
    source_count = len({item[2] for item in group})
    maximum_confidence = max(item[1] for item in group)
    return (
        source_count,
        -_player_name_geometry_error(group[0][0], orange_aspect),
        maximum_confidence,
    )


def _player_rapidocr_engine():
    """Lazily create the independent generic player-name OCR engine."""
    global _PLAYER_RAPIDOCR_ENGINE
    global _PLAYER_RAPIDOCR_INITIALIZED
    global _PLAYER_RAPIDOCR_ERROR_LOGGED
    with _PLAYER_RAPIDOCR_LOCK:
        if _PLAYER_RAPIDOCR_INITIALIZED:
            return _PLAYER_RAPIDOCR_ENGINE
        _PLAYER_RAPIDOCR_INITIALIZED = True
        _lower_current_thread_priority()
        try:
            from rapidocr_onnxruntime import RapidOCR

            # The name boundary has already been isolated by orange color.
            # Recognition-only mode is both faster and essential for a
            # one-character name such as Q, which a text detector may ignore.
            _PLAYER_RAPIDOCR_ENGINE = RapidOCR(
                use_text_det=False,
                use_angle_cls=False,
                text_score=0.30,
                intra_op_num_threads=PLAYER_OCR_INTRA_OP_THREADS,
                inter_op_num_threads=PLAYER_OCR_INTER_OP_THREADS,
            )
            _timestamped_log(
                "玩家名通用OCR已启用（单核、低优先级原始裁剪模式）。"
            )
        except Exception as exc:
            if not _PLAYER_RAPIDOCR_ERROR_LOGGED:
                _PLAYER_RAPIDOCR_ERROR_LOGGED = True
                _timestamped_log(
                    "玩家名通用OCR不可用，将回退Windows OCR："
                    f"{type(exc).__name__}: {exc}"
                )
        return _PLAYER_RAPIDOCR_ENGINE


def _rapidocr_text_items(value) -> list[tuple[str, float, float]]:
    """Return (text, confidence, left) across RapidOCR API result shapes."""
    items: list[tuple[str, float, float]] = []

    def visit(node) -> None:
        if isinstance(node, dict):
            text_value = node.get("text") or node.get("txt")
            if isinstance(text_value, str):
                score_value = node.get("score", node.get("confidence", 0.0))
                try:
                    confidence = float(score_value)
                except (TypeError, ValueError):
                    confidence = 0.0
                items.append((text_value, confidence, float(len(items))))
                return
            for child in node.values():
                visit(child)
            return
        if not isinstance(node, (list, tuple)):
            return
        if len(node) >= 3 and isinstance(node[1], str):
            try:
                confidence = float(node[2])
            except (TypeError, ValueError):
                confidence = 0.0
            left = float(len(items))
            try:
                points = np.asarray(node[0], dtype=np.float32)
                if points.ndim >= 2 and points.shape[-1] >= 2:
                    left = float(points[..., 0].min())
            except Exception:
                pass
            items.append((node[1], confidence, left))
            return
        for child in node:
            visit(child)

    visit(value)
    return items


def _rapidocr_player_candidates(
    image: np.ndarray,
) -> list[tuple[str, float]]:
    engine = _player_rapidocr_engine()
    if engine is None:
        return []
    try:
        result = engine(image)
    except Exception as exc:
        _timestamped_log(f"玩家名RapidOCR执行失败：{exc}")
        return []
    # Old RapidOCR returns (result, elapsed); parse only its first item.
    payload = result[0] if isinstance(result, tuple) and result else result
    items = _rapidocr_text_items(payload)
    candidates: list[tuple[str, float]] = []
    for raw_text, confidence, _ in items:
        text_value = _normalize_player_ocr_text(raw_text)
        if text_value:
            candidates.append((text_value, confidence))
    if len(items) > 1:
        ordered = sorted(items, key=lambda item: item[2])
        joined = _normalize_player_ocr_text(
            "".join(item[0] for item in ordered)
        )
        if joined:
            candidates.append(
                (joined, min(float(item[1]) for item in ordered))
            )
    return candidates


def _recognize_player_name_variants(
    variants: tuple[tuple[str, np.ndarray], ...],
) -> tuple[str, float, str]:
    """Combine OCR evidence only after checking the orange name geometry."""
    observations: list[tuple[str, float, str]] = []
    rejected_observations: list[tuple[str, float, str]] = []
    orange_aspect = (
        _player_name_orange_aspect(variants[0][1])
        if variants
        else None
    )

    def source_consensus() -> bool:
        groups: dict[str, set[str]] = {}
        for text_value, _, source in observations:
            groups.setdefault(text_value.casefold(), set()).add(source)
        return any(len(sources) >= 2 for sources in groups.values())

    for variant_name, image in variants:
        for text_value, confidence in _rapidocr_player_candidates(image):
            observation = (
                text_value,
                float(confidence),
                f"rapid/{variant_name}",
            )
            if _player_name_candidate_is_complete(
                text_value,
                orange_aspect,
            ):
                observations.append(observation)
            else:
                rejected_observations.append(observation)
        # Restore the early stable behavior: two geometry-compatible variants
        # are enough. A shorter “彩彩” cannot stop “八彩彩” because its total
        # character advance does not fit the measured three-glyph width.
        if source_consensus():
            break

    generic_groups: dict[str, list[tuple[str, float, str]]] = {}
    for observation in observations:
        generic_groups.setdefault(observation[0].casefold(), []).append(
            observation
        )
    generic_is_reliable = any(
        len({item[2] for item in group}) >= 2
        or max(item[1] for item in group) >= 0.78
        for group in generic_groups.values()
    )

    # Windows OCR is the low-confidence fallback. Stop as soon as two
    # geometry-compatible sources agree instead of always starting every
    # PowerShell/language/variant combination.
    if not generic_is_reliable and sys.platform == "win32":
        with tempfile.TemporaryDirectory(
            prefix="macro6-player-ocr-"
        ) as raw_dir:
            raw_path = Path(raw_dir)
            for variant_index, (variant_name, image) in enumerate(variants):
                image_path = raw_path / f"player-name-{variant_index}.png"
                success, encoded = cv2.imencode(".png", image)
                if not success:
                    continue
                encoded.tofile(str(image_path))
                for language_tag in ("zh-CN", "en-US"):
                    text_value = _normalize_player_ocr_text(
                        _macro6_windows_ocr(image_path, language_tag)
                    )
                    if not text_value:
                        continue
                    observation = (
                        text_value,
                        0.45,
                        f"windows-{language_tag}/{variant_name}",
                    )
                    if _player_name_candidate_is_complete(
                        text_value,
                        orange_aspect,
                    ):
                        observations.append(observation)
                    else:
                        rejected_observations.append(observation)
                if source_consensus():
                    break
    if not observations:
        if rejected_observations:
            # Never revive a gross width mismatch merely because it is the
            # longest string. Keep retrying later frames instead of recording
            # a four-glyph hallucination inside a two-glyph orange region.
            rejected_groups: dict[
                str, list[tuple[str, float, str]]
            ] = {}
            for observation in rejected_observations:
                rejected_groups.setdefault(
                    observation[0].casefold(), []
                ).append(observation)
            closest = min(
                rejected_groups.values(),
                key=lambda group: _player_name_geometry_error(
                    group[0][0],
                    orange_aspect,
                ),
            )
            diagnostics = ", ".join(
                f"{source}={text_value}({score:.2f}, width-error="
                f"{_player_name_geometry_error(text_value, orange_aspect):.3f})"
                for text_value, score, source in rejected_observations
            )
            diagnostics += f", rejected-best={closest[0][0]}"
            return "", 0.0, diagnostics
        diagnostics = ", ".join(
            f"{source}={text_value}({score:.2f}, incomplete)"
            for text_value, score, source in rejected_observations
        )
        return "", 0.0, diagnostics

    grouped: dict[str, list[tuple[str, float, str]]] = {}
    for observation in observations:
        grouped.setdefault(observation[0].casefold(), []).append(observation)
    winner = max(
        grouped.values(),
        key=lambda group: _player_name_group_rank(group, orange_aspect),
    )
    best = max(winner, key=lambda item: item[1])
    source_count = len({item[2] for item in winner})
    confidence = min(
        0.99,
        float(best[1]) + 0.04 * max(0, source_count - 1),
    )
    diagnostics = ", ".join(
        f"{source}={text_value}({score:.2f}, width-error="
        f"{_player_name_geometry_error(text_value, orange_aspect):.3f})"
        for text_value, score, source in observations
    )
    if rejected_observations:
        rejected_diagnostics = ", ".join(
            f"{source}={text_value}({score:.2f}, width-rejected="
            f"{_player_name_geometry_error(text_value, orange_aspect):.3f})"
            for text_value, score, source in rejected_observations
        )
        diagnostics = f"{diagnostics}, {rejected_diagnostics}"
    return best[0], confidence, diagnostics


@dataclass(frozen=True)
class _PlayerBannerCapture:
    sequence: int
    captured_at_utc: datetime
    captured_monotonic: float
    crop: np.ndarray
    source_height: int
    source_width: int
    room_generation: int


@dataclass(frozen=True)
class _PlayerOcrWork:
    capture: _PlayerBannerCapture
    sample: tuple[str, np.ndarray, float, tuple[str, int, int]]
    segment_id: int


class PlayerNotificationRecognizer:
    """Recognize orange player names and an independent fixed status font."""

    def __init__(
        self,
        tracker: RoomPlayerTracker,
        archive: StampCodeArchive,
    ) -> None:
        self._tracker = tracker
        self._archive = archive
        self._connection_icon = _embedded_player_connection_icon()
        self._connection_icon_color = (
            _embedded_player_connection_icon_color()
        )
        self._recycle_notice = _embedded_player_recycle_notice()
        self._templates = _embedded_player_status_templates()
        self._lock = threading.Lock()
        self._analysis_busy = False
        self._analysis_queue = deque()
        self._busy = False
        self._work_queue = deque()
        self._last_banner_signature: tuple[str, int, int] | None = None
        self._active_segment_id: int | None = None
        self._next_segment_id = 0
        self._active_segment_frames = 0
        self._next_capture_sequence = 0
        self._last_applied_capture_sequence = 0
        self._segment_vote_counts: dict[
            int, dict[tuple[str, str], int]
        ] = {}
        self._accepted_segment_ids: set[int] = set()
        self._last_request_monotonic = 0.0
        self._absent_since: float | None = None
        self._last_accepted_event: tuple[str, str] | None = None
        self._right_failure_streak = 0
        self._failure_capture_armed = True
        self._failure_capture_requested = False
        self._failure_capture_reason = ""
        self._icon_visible = False
        self._notification_visible = False
        # Load the official PaddleOCR model before the first notification so
        # its one-time initialization does not consume a short-lived banner.
        threading.Thread(
            target=self._warm_up_name_recognizer,
            name="macro6-player-ocr-warmup",
            daemon=True,
        ).start()

    @staticmethod
    def _warm_up_name_recognizer() -> None:
        try:
            warm_up_player_name_recognizer()
            _timestamped_log(
                f"玩家名识别接口已就绪：{PLAYER_NAME_MODEL}。"
            )
        except PlayerNameRecognizerUnavailable as exc:
            _timestamped_log(f"玩家名识别接口不可用：{exc}")

    def _detect_connection_icon(self, frame: np.ndarray) -> bool:
        """Perform only the small, coarse globe check on the UI thread."""
        if frame is None or frame.size == 0 or frame.ndim != 3:
            return False
        icon_crop = _fractional_crop(frame, PLAYER_CONNECTION_ICON_ROI)
        icon_spread = icon_crop.max(axis=2).astype(np.int16) - icon_crop.min(
            axis=2
        ).astype(np.int16)
        icon_gray = cv2.cvtColor(icon_crop, cv2.COLOR_BGR2GRAY)
        icon_dark_mask = (
            (icon_gray < 150)
            & (icon_spread < 60)
        ).astype(np.uint8)
        normalized_icon = cv2.resize(
            icon_dark_mask * 255,
            (
                PLAYER_CONNECTION_ICON_SHAPE[1],
                PLAYER_CONNECTION_ICON_SHAPE[0],
            ),
            interpolation=cv2.INTER_AREA,
        ).astype(np.float32) / 255.0
        icon_error = float(
            np.mean(np.abs(normalized_icon - self._connection_icon))
        )
        normalized_icon_color = cv2.resize(
            icon_crop[:, :, :3],
            (
                PLAYER_CONNECTION_ICON_COLOR_SHAPE[1],
                PLAYER_CONNECTION_ICON_COLOR_SHAPE[0],
            ),
            interpolation=cv2.INTER_AREA,
        ).astype(np.float32) / 255.0
        icon_color_error = float(
            np.mean(
                np.abs(
                    normalized_icon_color - self._connection_icon_color
                )
            )
        )
        return (
            icon_error <= 0.055
            or (icon_error <= 0.075 and icon_color_error <= 0.088)
        )

    def _extract_banner(
        self,
        crop: np.ndarray,
        source_height: int,
        source_width: int,
    ) -> tuple[
        str,
        np.ndarray,
        float,
        tuple[str, int, int],
    ] | None:
        """Analyze one copied banner crop entirely off the UI thread."""
        if crop is None or crop.size == 0 or crop.ndim != 3:
            return None
        scale_x = CAPTURE_WIDTH / max(1.0, float(source_width))
        scale_y = CAPTURE_HEIGHT / max(1.0, float(source_height))
        blue, green, red = cv2.split(crop[:, :, :3])
        red_i = red.astype(np.int16)
        green_i = green.astype(np.int16)
        orange_mask = (
            (red > 180)
            & (green > 70)
            & (green < 205)
            & (blue < 100)
            & (red_i > green_i + 35)
        )
        active_orange_columns = np.flatnonzero(orange_mask.sum(axis=0) > 0)
        if active_orange_columns.size == 0:
            return None

        # The widened ROI reaches the physical right edge so very long names
        # are not truncated.  Keep only the first tightly spaced orange run;
        # this excludes unrelated orange HUD elements farther to the right.
        orange_runs: list[list[int]] = []
        run_gap = max(2, int(round(12.0 / scale_x)))
        for column in active_orange_columns:
            value = int(column)
            if not orange_runs or value > orange_runs[-1][1] + run_gap:
                orange_runs.append([value, value])
            else:
                orange_runs[-1][1] = value
        orange_start, orange_end = orange_runs[0]
        orange_mask[:, :orange_start] = False
        orange_mask[:, orange_end + 1 :] = False
        orange_rows, orange_columns = np.where(orange_mask)
        normalized_orange_pixels = (
            float(orange_columns.size) * scale_x * scale_y
        )
        if normalized_orange_pixels < 12.0:
            return None

        orange_left = int(orange_columns.min())
        orange_right = int(orange_columns.max())
        orange_top = int(orange_rows.min())
        orange_bottom = int(orange_rows.max())
        normalized_orange_width = (
            float(orange_right - orange_left + 1) * scale_x
        )
        normalized_orange_height = (
            float(orange_bottom - orange_top + 1) * scale_y
        )
        # A valid player name can be a single narrow character such as Q.
        if normalized_orange_width < 4.0 or normalized_orange_height < 6.0:
            return None

        spread = crop.max(axis=2).astype(np.int16) - crop.min(
            axis=2
        ).astype(np.int16)
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        neutral_mask = (
            (spread < 38)
            & (gray > 45)
            & (gray < 205)
        )
        status_mask = np.zeros_like(neutral_mask, dtype=np.uint8)
        status_gap = max(2, int(round(3.0 / scale_x)))
        status_left = min(crop.shape[1], orange_right + status_gap)
        # Fixed status text is at most 64 normalized pixels wide.  The whole
        # banner ROI reaches the screen edge for long names, while this local
        # post-name window avoids unrelated HUD text beyond the banner.
        status_right = min(
            crop.shape[1], orange_right + int(round(100.0 / scale_x))
        )
        status_top = max(
            0,
            orange_top - int(round(4.0 / scale_y)),
        )
        status_bottom = min(
            crop.shape[0],
            orange_bottom + int(round(6.0 / scale_y)),
        )
        status_mask[
            status_top:status_bottom,
            status_left:status_right,
        ] = neutral_mask[
            status_top:status_bottom,
            status_left:status_right,
        ]
        status_rows, status_columns = np.where(status_mask)
        normalized_status_pixels = (
            float(status_columns.size) * scale_x * scale_y
        )
        if normalized_status_pixels < 30.0:
            return None
        status_glyphs = status_mask[
            status_rows.min() : status_rows.max() + 1,
            status_columns.min() : status_columns.max() + 1,
        ].astype(np.uint8)
        normalized_status = cv2.resize(
            status_glyphs * 255,
            (PLAYER_STATUS_TEMPLATE_SHAPE[2], PLAYER_STATUS_TEMPLATE_SHAPE[1]),
            interpolation=cv2.INTER_AREA,
        ).astype(np.float32) / 255.0
        scores = np.mean(
            np.abs(self._templates - normalized_status[None, :, :]),
            axis=(1, 2),
        )
        status_width = int(
            round(
                (status_columns.max() - status_columns.min() + 1) * scale_x
            )
        )
        if 56 <= status_width <= 80:
            # “即将抵达。” is consistently wider than both three-character
            # statuses.  Width is stable even when transition antialiasing
            # makes its normalized glyph bitmap resemble “已抵达。” briefly.
            incoming_index = PLAYER_STATUS_LABELS.index("incoming")
            competing_indices = tuple(
                index
                for index, label in enumerate(PLAYER_STATUS_LABELS)
                if label != "incoming"
            )
            best_other_index = min(
                competing_indices,
                key=lambda index: float(scores[index]),
            )
            # A partially faded short status can acquire a wide gray tail.
            # Let a clearly superior fixed glyph template override width;
            # this repairs “呆呆 已抵达。” without weakening true incoming
            # samples, whose incoming-template errors are near zero.
            if (
                float(scores[best_other_index]) <= 0.30
                and float(
                    scores[incoming_index] - scores[best_other_index]
                ) >= 0.06
            ):
                best_index = int(best_other_index)
                best_score = float(scores[best_index])
                margin = float(
                    scores[incoming_index] - scores[best_index]
                )
            else:
                best_index = incoming_index
                best_score = float(scores[best_index])
                margin = min(
                    float(scores[index] - scores[best_index])
                    for index in competing_indices
                )
            if best_score > 0.43:
                return None
        elif 40 <= status_width <= 55:
            short_indices = tuple(
                index
                for index, label in enumerate(PLAYER_STATUS_LABELS)
                if label in {"arrived", "left"}
            )
            short_ranking = sorted(
                short_indices,
                key=lambda index: float(scores[index]),
            )
            best_index = int(short_ranking[0])
            best_score = float(scores[best_index])
            margin = float(scores[int(short_ranking[1])] - best_score)
            # Width already proves this is one of the two short fixed
            # statuses.  Always keep the better B64 match when its absolute
            # error is plausible; the outer two-frame vote rejects a fleeting
            # transition rather than discarding a short “已抵达” banner.
            if best_score > 0.38:
                return None
        else:
            return None

        # Use color only to find the proportional player-name boundary, then
        # give the generic OCR the original antialiased glyph pixels.  The
        # crop ends before the independent gray/white fixed-status segment.
        name_top = max(0, orange_top - int(round(4.0 / scale_y)))
        name_bottom = min(
            crop.shape[0], orange_bottom + int(round(5.0 / scale_y))
        )
        name_left = max(0, orange_left - int(round(3.0 / scale_x)))
        name_right = min(
            crop.shape[1], orange_right + int(round(3.0 / scale_x))
        )
        player_name_crop = crop[
            name_top:name_bottom,
            name_left:name_right,
        ]
        # Fingerprint the fixed-width name area instead of tightly cropping
        # and stretching every name to the same width.  This preserves both
        # the glyph shapes and their real start/end positions, so a globe that
        # stays visible while player A's banner changes to B's is still a new
        # segment.  Multiply the bool mask by 255 before thresholding; without
        # this, the old >=96 test reduced every signature to an empty bitmap.
        signature_mask = cv2.resize(
            orange_mask.astype(np.uint8) * 255,
            (
                PLAYER_NOTIFICATION_FINGERPRINT_SHAPE[1],
                PLAYER_NOTIFICATION_FINGERPRINT_SHAPE[0],
            ),
            interpolation=cv2.INTER_AREA,
        )
        signature_bits = np.packbits(signature_mask >= 64)
        visual_signature = (
            PLAYER_STATUS_LABELS[best_index],
            int(round(normalized_orange_width)),
            int.from_bytes(signature_bits.tobytes(), "big"),
        )
        return (
            PLAYER_STATUS_LABELS[best_index],
            player_name_crop.copy(),
            best_score,
            visual_signature,
        )

    def _is_recycle_box_notice(
        self,
        crop: np.ndarray,
        source_height: int,
        source_width: int,
    ) -> bool:
        """Identify the fixed non-player “回收箱中存有物品。” notice."""
        if crop is None or crop.size == 0 or crop.ndim != 3:
            return False
        banner_left = round(PLAYER_NOTIFICATION_ROI[0] * source_width)
        banner_top = round(PLAYER_NOTIFICATION_ROI[1] * source_height)
        x1 = round(PLAYER_RECYCLE_NOTICE_ROI[0] * source_width) - banner_left
        y1 = round(PLAYER_RECYCLE_NOTICE_ROI[1] * source_height) - banner_top
        x2 = round(PLAYER_RECYCLE_NOTICE_ROI[2] * source_width) - banner_left
        y2 = round(PLAYER_RECYCLE_NOTICE_ROI[3] * source_height) - banner_top
        x1 = max(0, min(crop.shape[1], x1))
        x2 = max(x1 + 1, min(crop.shape[1], x2))
        y1 = max(0, min(crop.shape[0], y1))
        y2 = max(y1 + 1, min(crop.shape[0], y2))
        notice_crop = crop[y1:y2, x1:x2]
        if notice_crop.size == 0:
            return False
        spread = notice_crop.max(axis=2).astype(np.int16) - notice_crop.min(
            axis=2
        ).astype(np.int16)
        gray = cv2.cvtColor(notice_crop, cv2.COLOR_BGR2GRAY)
        glyph_mask = ((spread < 42) & (gray < 190)).astype(np.uint8) * 255
        normalized = cv2.resize(
            glyph_mask,
            (
                PLAYER_RECYCLE_NOTICE_SHAPE[1],
                PLAYER_RECYCLE_NOTICE_SHAPE[0],
            ),
            interpolation=cv2.INTER_AREA,
        ).astype(np.float32) / 255.0
        error = float(np.mean(np.abs(normalized - self._recycle_notice)))
        return error <= PLAYER_RECYCLE_NOTICE_MAX_ERROR

    @staticmethod
    def _same_banner_segment(
        previous: tuple[str, int, int] | None,
        current: tuple[str, int, int],
    ) -> bool:
        if previous is None or previous[0] != current[0]:
            return False
        if abs(previous[1] - current[1]) > 1:
            return False
        changed_bits = (previous[2] ^ current[2]).bit_count()
        foreground_bits = (previous[2] | current[2]).bit_count()
        allowed_changes = min(
            PLAYER_NOTIFICATION_FINGERPRINT_MAX_BIT_CHANGES,
            max(
                1,
                int(
                    foreground_bits
                    * PLAYER_NOTIFICATION_FINGERPRINT_MAX_CHANGE_RATIO
                ),
            ),
        )
        return changed_bits <= allowed_changes

    def request(self, frame: np.ndarray) -> None:
        # Keep the display thread deliberately small: only inspect the globe,
        # then copy one narrow raw banner crop at the configured sample rate.
        # All text masks, status matching, fingerprints, OCR variants, and OCR
        # itself run in the two background stages below.
        icon_visible = self._detect_connection_icon(frame)
        now = time.monotonic()
        enqueue_banner = False
        capture_failure = False
        failure_reason = ""
        with self._lock:
            self._icon_visible = icon_visible
            if self._failure_capture_requested:
                self._failure_capture_requested = False
                capture_failure = True
                failure_reason = self._failure_capture_reason
            if not icon_visible:
                self._notification_visible = False
                if self._absent_since is None:
                    self._absent_since = now
                elif now - self._absent_since >= 0.50:
                    if (
                        not self._analysis_busy
                        and not self._analysis_queue
                        and not self._busy
                        and not self._work_queue
                    ):
                        self._last_accepted_event = None
                        self._segment_vote_counts.clear()
                    self._failure_capture_armed = True
                    self._last_banner_signature = None
                    self._active_segment_id = None
                    self._active_segment_frames = 0
            else:
                self._absent_since = None
                if (
                    now - self._last_request_monotonic
                    >= PLAYER_NOTIFICATION_SCAN_SECONDS
                ):
                    self._last_request_monotonic = now
                    enqueue_banner = True

        if capture_failure:
            failure_frame = frame.copy()
            threading.Thread(
                target=self._archive.record_player_notification_failure,
                args=(
                    failure_frame,
                    failure_reason or "right_banner_failed",
                ),
                name="macro6-player-notification-error-screenshot",
                daemon=True,
            ).start()

        if not enqueue_banner:
            return
        source_height, source_width = frame.shape[:2]
        banner_crop = _fractional_crop(
            frame, PLAYER_NOTIFICATION_ROI
        ).copy()
        room_generation = self._tracker.generation()
        start_analysis = False
        with self._lock:
            self._next_capture_sequence += 1
            capture = _PlayerBannerCapture(
                sequence=self._next_capture_sequence,
                captured_at_utc=datetime.now(timezone.utc),
                captured_monotonic=now,
                crop=banner_crop,
                source_height=source_height,
                source_width=source_width,
                room_generation=room_generation,
            )
            self._analysis_queue.append(capture)
            if not self._analysis_busy:
                self._analysis_busy = True
                start_analysis = True
        if start_analysis:
            threading.Thread(
                target=self._analysis_loop,
                name="macro6-player-notification-analysis",
                daemon=True,
            ).start()

    def _analysis_loop(self) -> None:
        """Segment copied banner crops without ever blocking video display."""
        while True:
            with self._lock:
                if not self._analysis_queue:
                    self._analysis_busy = False
                    return
                capture = self._analysis_queue.popleft()
            try:
                sample = self._extract_banner(
                    capture.crop,
                    capture.source_height,
                    capture.source_width,
                )
                recycle_box_notice = (
                    sample is None
                    and self._is_recycle_box_notice(
                        capture.crop,
                        capture.source_height,
                        capture.source_width,
                    )
                )
            except Exception as exc:
                _timestamped_log(f"玩家横幅后台分析失败：{exc}")
                sample = None
                recycle_box_notice = False

            start_ocr = False
            with self._lock:
                self._notification_visible = (
                    self._icon_visible and sample is not None
                )
                if sample is None:
                    if recycle_box_notice:
                        # The globe also prefixes this fixed system notice.
                        # It is neither a player event nor a detection error.
                        self._right_failure_streak = 0
                        self._failure_capture_armed = True
                        self._last_banner_signature = None
                        self._active_segment_id = None
                        self._active_segment_frames = 0
                        continue
                    if not self._icon_visible:
                        self._right_failure_streak = 0
                        continue
                    self._right_failure_streak += 1
                    if (
                        self._right_failure_streak >= 2
                        and self._failure_capture_armed
                    ):
                        self._failure_capture_armed = False
                        self._failure_capture_requested = True
                        self._failure_capture_reason = "right_banner_failed"
                    if self._right_failure_streak >= 2:
                        # Treat a short blank transition as a boundary even
                        # when the globe itself remains continuously visible.
                        self._last_banner_signature = None
                        self._active_segment_id = None
                        self._active_segment_frames = 0
                    continue

                self._right_failure_streak = 0
                visual_signature = sample[3]
                if not self._same_banner_segment(
                    self._last_banner_signature,
                    visual_signature,
                ):
                    self._next_segment_id += 1
                    self._active_segment_id = self._next_segment_id
                    self._active_segment_frames = 0
                    # Anchor each segment to its first valid shape.  Comparing
                    # against the previous frame alone could walk through a
                    # crossfade without ever declaring a player change.
                    self._last_banner_signature = visual_signature
                segment_id = self._active_segment_id
                if segment_id is None:
                    continue
                if segment_id in self._accepted_segment_ids:
                    continue
                if (
                    self._active_segment_frames
                    >= PLAYER_NOTIFICATION_FRAMES_PER_SEGMENT
                ):
                    continue
                self._active_segment_frames += 1
                self._work_queue.append(
                    _PlayerOcrWork(
                        capture=capture,
                        sample=sample,
                        segment_id=segment_id,
                    )
                )
                if not self._busy:
                    self._busy = True
                    start_ocr = True
            if start_ocr:
                threading.Thread(
                    target=self._recognition_loop,
                    name="macro6-player-notification-ocr",
                    daemon=True,
                ).start()

    def _recognition_loop(self) -> None:
        _lower_current_thread_priority()
        while True:
            with self._lock:
                if not self._work_queue:
                    self._busy = False
                    return
                work = self._work_queue.popleft()
                sample = work.sample
                segment_id = work.segment_id
                if segment_id in self._accepted_segment_ids:
                    continue
            status, player_name_crop, status_score, visual_signature = sample
            try:
                repaired_name_crop = repair_player_name_right_edge(
                    player_name_crop
                )
                diagnostics = ""
                try:
                    player_name = recognize_player_name(repaired_name_crop)
                except PlayerNameRecognizerUnavailable as exc:
                    player_name = ""
                    diagnostics = str(exc)
                if not player_name:
                    # This banner has already passed the globe, status and
                    # orange-name-region checks.  Re-running every queued
                    # frame after all OCR paths returned empty only blocks
                    # newer player events and floods the terminal.  Archive
                    # one untouched name crop as "未知", then consume this
                    # visual segment without applying it to the room tracker.
                    with self._lock:
                        self._last_applied_capture_sequence = max(
                            self._last_applied_capture_sequence,
                            work.capture.sequence,
                        )
                        self._last_accepted_event = None
                        self._accepted_segment_ids.add(segment_id)
                        self._segment_vote_counts.pop(segment_id, None)
                        self._work_queue = deque(
                            item
                            for item in self._work_queue
                            if item.segment_id != segment_id
                        )
                        if len(self._accepted_segment_ids) > 256:
                            cutoff = self._next_segment_id - 128
                            self._accepted_segment_ids = {
                                item
                                for item in self._accepted_segment_ids
                                if item >= cutoff
                            }
                        self._failure_capture_armed = True
                    _timestamped_log(
                        f"检测到玩家通知，但{PLAYER_NAME_MODEL}未取得"
                        "玩家名；已将姓名裁剪记为“未知”并舍弃该横幅。"
                    )
                    if diagnostics:
                        _timestamped_log(f"玩家名OCR候选：{diagnostics}")
                    self._archive.record_player_name_crop(
                        "未知",
                        repaired_name_crop,
                        work.capture.captured_at_utc,
                        status,
                    )
                    continue
                event = (player_name.casefold(), status)
                with self._lock:
                    if event == self._last_accepted_event:
                        self._last_applied_capture_sequence = max(
                            self._last_applied_capture_sequence,
                            work.capture.sequence,
                        )
                        self._accepted_segment_ids.add(segment_id)
                        self._work_queue = deque(
                            item
                            for item in self._work_queue
                            if item.segment_id != segment_id
                        )
                        continue
                    votes = self._segment_vote_counts.setdefault(
                        segment_id, {}
                    )
                    votes[event] = votes.get(event, 0) + 1
                    vote_count = votes[event]
                    if vote_count < 2:
                        continue
                    if (
                        work.capture.sequence
                        <= self._last_applied_capture_sequence
                    ):
                        self._accepted_segment_ids.add(segment_id)
                        self._segment_vote_counts.pop(segment_id, None)
                        self._work_queue = deque(
                            item
                            for item in self._work_queue
                            if item.segment_id != segment_id
                        )
                        _timestamped_log(
                            "丢弃乱序玩家通知："
                            f"capture_sequence={work.capture.sequence}，"
                            "不晚于已应用序号"
                            f"{self._last_applied_capture_sequence}。"
                        )
                        continue
                    self._last_applied_capture_sequence = (
                        work.capture.sequence
                    )
                    self._last_accepted_event = event
                    self._accepted_segment_ids.add(segment_id)
                    self._segment_vote_counts.pop(segment_id, None)
                    # Once two frames agree, discard only this segment's
                    # remaining redundant work.  Frames already queued for a
                    # newer player remain FIFO-ordered and run immediately.
                    self._work_queue = deque(
                        item
                        for item in self._work_queue
                        if item.segment_id != segment_id
                    )
                    if len(self._accepted_segment_ids) > 256:
                        cutoff = self._next_segment_id - 128
                        self._accepted_segment_ids = {
                            item
                            for item in self._accepted_segment_ids
                            if item >= cutoff
                        }
                    self._failure_capture_armed = True
                _timestamped_log(
                    f"玩家通知识别：{player_name} / {status} "
                    f"(status_score={status_score:.3f}, "
                    f"name_model={PLAYER_NAME_MODEL}, votes={vote_count}, "
                    f"capture_sequence={work.capture.sequence}, "
                    f"segment_id={segment_id}, "
                    "processing_delay_ms="
                    f"{max(0, round((time.monotonic() - work.capture.captured_monotonic) * 1000))})。"
                )
                if diagnostics:
                    _timestamped_log(f"玩家名OCR候选：{diagnostics}")
                applied = self._tracker.apply(
                    player_name,
                    status,
                    expected_generation=work.capture.room_generation,
                    capture_sequence=work.capture.sequence,
                    captured_at_utc=work.capture.captured_at_utc,
                    segment_id=segment_id,
                    processing_delay_ms=max(
                        0,
                        round(
                            (
                                time.monotonic()
                                - work.capture.captured_monotonic
                            )
                            * 1000
                        ),
                    ),
                )
                # Save exactly the right-edge-repaired image sent to OCR, not
                # the full frame or a later OCR model tensor.
                if applied:
                    self._archive.record_player_name_crop(
                        player_name,
                        repaired_name_crop,
                        work.capture.captured_at_utc,
                        status,
                    )
            except Exception as exc:
                _timestamped_log(f"玩家通知识别失败：{exc}")

    def notification_visible(self) -> bool:
        with self._lock:
            return self._notification_visible

    def icon_visible(self) -> bool:
        with self._lock:
            return self._icon_visible


def _ocr_code_image_detailed(
    image_path: Path,
) -> tuple[str, tuple[tuple[str, str], ...]]:
    raw_results: list[tuple[str, str]] = []
    executable = shutil.which("tesseract")
    if executable is not None:
        try:
            completed = subprocess.run(
                [
                    executable,
                    str(image_path),
                    "stdout",
                    "--psm",
                    "7",
                    "-c",
                    f"tessedit_char_whitelist={CODE_OCR_WHITELIST}",
                ],
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="ignore",
                timeout=10,
                check=False,
            )
            raw_results.append(("tesseract", completed.stdout))
            code = _parse_six_character_code(completed.stdout)
            if code:
                return code, tuple(raw_results)
        except Exception as exc:
            raw_results.append(("tesseract-error", str(exc)))
    else:
        raw_results.append(("tesseract", "<未安装>"))
    windows_raw = _macro6_windows_ocr(image_path, language_tag="en-US")
    raw_results.append(("windows-en-US", windows_raw))
    return _parse_six_character_code(windows_raw), tuple(raw_results)


def _ocr_code_image(image_path: Path) -> str:
    """Backward-compatible CODE OCR result without diagnostic details."""
    return _ocr_code_image_detailed(image_path)[0]


def _ocr_debug_text(raw: str) -> str:
    compact = " ".join(str(raw or "").split())
    if not compact:
        return "<空>"
    return compact[:120]


def _build_code_ocr_variants(
    crop: np.ndarray,
) -> tuple[tuple[str, np.ndarray], ...]:
    """Build complementary OCR inputs for white CODE text on a blue panel."""
    enlarged = cv2.resize(
        crop,
        None,
        fx=4.0,
        fy=4.0,
        interpolation=cv2.INTER_CUBIC,
    )
    gray = cv2.cvtColor(enlarged, cv2.COLOR_BGR2GRAY)

    _, fixed_binary = cv2.threshold(gray, 180, 255, cv2.THRESH_BINARY)
    _, otsu_binary = cv2.threshold(
        gray,
        0,
        255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU,
    )
    hsv = cv2.cvtColor(enlarged, cv2.COLOR_BGR2HSV)
    _, saturation, value = cv2.split(hsv)
    white_text = (
        (saturation <= 105)
        & (value >= 165)
    ).astype(np.uint8) * 255

    def monochrome_canvas(text_mask: np.ndarray) -> np.ndarray:
        return cv2.copyMakeBorder(
            255 - text_mask,
            40,
            40,
            40,
            40,
            cv2.BORDER_CONSTANT,
            value=255,
        )

    color_canvas = cv2.copyMakeBorder(
        enlarged,
        40,
        40,
        40,
        40,
        cv2.BORDER_CONSTANT,
        value=(255, 255, 255),
    )
    return (
        ("white-mask", monochrome_canvas(white_text)),
        ("fixed-180", monochrome_canvas(fixed_binary)),
        ("otsu", monochrome_canvas(otsu_binary)),
        ("color", color_canvas),
    )


class CodeRecognizer:
    """Run slow OCR away from the preview and controller threads."""

    def __init__(
        self,
        archive: StampCodeArchive | None = None,
    ) -> None:
        self._lock = threading.Lock()
        self._archive = archive
        self._code = ""
        self._code_unknown = False
        self._last_reported_code = ""
        self._revision = 0
        self._last_change_monotonic: float | None = None
        self._latest_frame: np.ndarray | None = None
        self._latest_detection: PokopiaDetection | None = None
        self._latest_attempt_frame: np.ndarray | None = None
        self._latest_attempt_detection: PokopiaDetection | None = None
        self._last_recorded_revision = 0
        self._last_recorded_code = ""
        self._in_flight = False
        self._last_attempt = 0.0
        self._failure_count = 0
        self._last_failure_log = 0.0
        self._ocr_round_active = False
        self._ocr_round_macro_key = 0
        self._ocr_round_failure_count = 0
        self._ocr_round_error_screenshot_recorded = False
        self._blocked_until_code_panel_absent = False
        self._latest_ocr_diagnostics: tuple[str, ...] = ()

    def snapshot(self) -> str:
        with self._lock:
            return self._code

    def revision(self) -> int:
        with self._lock:
            return self._revision

    def is_unknown(self) -> bool:
        with self._lock:
            return self._code_unknown

    def note_code_panel_absent(self) -> None:
        """Permit OCR for the next CODE panel after an unknown-code fallback."""
        with self._lock:
            self._blocked_until_code_panel_absent = False

    def reset_timer_anchor(self) -> None:
        """Start both Macro1 watchdog windows without recording a CODE."""
        with self._lock:
            self._last_change_monotonic = time.monotonic()

    def begin_ocr_round(self, macro_key: int) -> None:
        """Start one ten-attempt CODE recognition window for this restart."""
        with self._lock:
            self._ocr_round_active = True
            self._ocr_round_macro_key = int(macro_key)
            self._ocr_round_failure_count = 0
            self._ocr_round_error_screenshot_recorded = False
            self._latest_ocr_diagnostics = ()

    def end_ocr_round(self) -> None:
        with self._lock:
            self._ocr_round_active = False

    def record_ocr_timeout_error(self, macro_key: int) -> None:
        with self._lock:
            archive = self._archive
            retained_code = self._code
            failure_count = self._failure_count
            screenshot_already_recorded = (
                self._ocr_round_error_screenshot_recorded
            )
            if not screenshot_already_recorded:
                self._ocr_round_error_screenshot_recorded = True
            self._ocr_round_active = False
            frame = (
                None
                if screenshot_already_recorded
                else self._latest_attempt_frame
            )
            detection = self._latest_attempt_detection
            ocr_results = self._latest_ocr_diagnostics
        if archive is not None:
            archive.record_error(
                "code_ocr_timeout",
                macro_key=macro_key,
                retained_code=retained_code,
                failure_count=failure_count,
                frame=frame,
                detection=detection,
                ocr_results=ocr_results,
                screenshot_suppressed_reason=(
                    "本轮已保存过一次CODE识别error截图"
                    if screenshot_already_recorded
                    else ""
                ),
            )

    def timer_expired(self) -> bool:
        with self._lock:
            changed_at = self._last_change_monotonic
        return (
            changed_at is not None
            and time.monotonic() - changed_at >= CODE_TIMER_RESTART_SECONDS
        )

    def timer_elapsed_seconds(self) -> float | None:
        with self._lock:
            changed_at = self._last_change_monotonic
        if changed_at is None:
            return None
        return max(0.0, time.monotonic() - changed_at)

    def record_current(self, reason: str, *, force: bool = False) -> bool:
        """Write the latest CODE once, only when a macro requests a record."""
        with self._lock:
            if (
                not self._code
                or (
                    not force
                    and self._revision <= self._last_recorded_revision
                )
            ):
                return False
            code = self._code
            revision = self._revision
            previous_code = "" if force else self._last_recorded_code
            frame = self._latest_frame
            detection = self._latest_detection
            self._last_recorded_revision = revision
            self._last_recorded_code = code

        if previous_code:
            _timestamped_log(
                f"{reason}：记录 Pokopia CODE "
                f"{previous_code} -> {code}。"
            )
        else:
            _timestamped_log(f"{reason}：记录 Pokopia CODE {code}。")
        if (
            self._archive is not None
            and frame is not None
            and detection is not None
        ):
            self._archive.record(code, frame, detection)
        return True

    def request(
        self,
        frame: np.ndarray,
        detection: PokopiaDetection,
        *,
        suppress_chime: bool = False,
    ) -> None:
        now = time.monotonic()
        with self._lock:
            if (
                self._in_flight
                or self._blocked_until_code_panel_absent
                or now - self._last_attempt < CODE_OCR_RETRY_SECONDS
            ):
                return
            self._in_flight = True
            self._last_attempt = now
        crop = _fractional_crop(frame, CODE_TEXT_ROI).copy()
        threading.Thread(
            target=self._recognize,
            args=(crop, frame.copy(), detection, suppress_chime),
            name="macro6-code-ocr",
            daemon=True,
        ).start()

    def _recognize(
        self,
        crop: np.ndarray,
        source_frame: np.ndarray | None = None,
        detection: PokopiaDetection | None = None,
        suppress_chime: bool = False,
    ) -> None:
        code = ""
        successful_variant = ""
        should_report = False
        previous_code = ""
        failure_count = 0
        round_failure_count = 0
        should_log_failure = False
        should_archive_failure = False
        archive: StampCodeArchive | None = None
        archive_macro_key = 0
        archive_frame: np.ndarray | None = None
        archive_detection: PokopiaDetection | None = None
        ocr_diagnostics: list[str] = []
        try:
            code = _recognize_embedded_code_glyphs(crop)
            if code:
                successful_variant = "embedded-pokopia-glyph"
                ocr_diagnostics.append(f"embedded-glyph={code}")
            else:
                with tempfile.TemporaryDirectory(
                    prefix="macro6-code-ocr-"
                ) as raw_dir:
                    for variant_name, ocr_image in _build_code_ocr_variants(
                        crop
                    ):
                        image_path = Path(raw_dir) / f"code-{variant_name}.png"
                        success, encoded = cv2.imencode(".png", ocr_image)
                        if not success:
                            continue
                        encoded.tofile(str(image_path))
                        ocr_candidate, raw_results = (
                            _ocr_code_image_detailed(image_path)
                        )
                        ocr_diagnostics.extend(
                            f"{variant_name}/{engine}={_ocr_debug_text(raw)}"
                            for engine, raw in raw_results
                        )
                        if ocr_candidate:
                            ocr_diagnostics.append(
                                f"{variant_name}/candidate={ocr_candidate}"
                            )
                    # Generic OCR is diagnostic only.  A CODE is allowed to
                    # update state exclusively when all six glyphs matched
                    # the embedded Pokopia character library.
        finally:
            with self._lock:
                previous_code = self._code
                if source_frame is not None:
                    self._latest_attempt_frame = source_frame
                    self._latest_attempt_detection = detection
                self._latest_ocr_diagnostics = tuple(ocr_diagnostics)
                if code:
                    self._failure_count = 0
                    self._code = code
                    self._code_unknown = False
                    if code != self._last_reported_code:
                        self._last_reported_code = code
                        self._revision += 1
                        self._last_change_monotonic = time.monotonic()
                        self._latest_frame = source_frame
                        self._latest_detection = detection
                        should_report = True
                        self._ocr_round_active = False
                else:
                    self._failure_count += 1
                    failure_count = self._failure_count
                    if self._ocr_round_active:
                        self._ocr_round_failure_count += 1
                        round_failure_count = self._ocr_round_failure_count
                    if (
                        self._ocr_round_active
                        and self._ocr_round_failure_count
                        >= CODE_OCR_UNKNOWN_AFTER_FAILURES
                        and not self._ocr_round_error_screenshot_recorded
                    ):
                        self._ocr_round_error_screenshot_recorded = True
                        should_archive_failure = True
                        archive = self._archive
                        archive_macro_key = self._ocr_round_macro_key
                        archive_frame = source_frame
                        archive_detection = detection
                        self._ocr_round_active = False
                        self._blocked_until_code_panel_absent = True
                    now = time.monotonic()
                    if (
                        now - self._last_failure_log
                        >= CODE_OCR_FAILURE_LOG_INTERVAL_SECONDS
                    ):
                        self._last_failure_log = now
                        should_log_failure = True
                self._in_flight = False
            if should_archive_failure and archive is not None:
                archive.record_error(
                    "code_ocr_ten_failures_unknown",
                    macro_key=archive_macro_key,
                    retained_code=previous_code,
                    failure_count=round_failure_count,
                    frame=archive_frame,
                    detection=archive_detection,
                    ocr_results=tuple(ocr_diagnostics),
                )
            if should_archive_failure:
                # Publish the unknown revision only after its screenshot has
                # finished writing.  The local server can therefore never
                # attach an older unknown round's image to this revision.
                with self._lock:
                    self._code = ""
                    self._code_unknown = True
                    self._last_reported_code = ""
                    self._revision += 1
                    self._last_change_monotonic = time.monotonic()
                    self._latest_frame = source_frame
                    self._latest_detection = detection
            if should_log_failure:
                retained = previous_code or "<空>"
                if should_archive_failure:
                    _timestamped_log(
                        "CODE面板连续10次未得到六码；已保存本轮唯一error截图，"
                        f"不再沿用旧CODE {retained}，改以未知CODE继续正常流程。"
                    )
                else:
                    _timestamped_log(
                        "CODE面板已识别，但四种OCR方案均未得到六位码；"
                        f"连续失败{failure_count}次，保留CODE {retained}并继续重试。"
                    )
                _timestamped_log(
                    "CODE OCR原始结果：" + "｜".join(ocr_diagnostics)
                )
            if should_report:
                _timestamped_log(
                    f"CODE OCR更新：{previous_code or '<空>'} -> {code} "
                    f"（方案：{successful_variant}）。"
                )
                # The first valid assignment (empty -> CODE) is also a CODE
                # change and therefore uses the same audible notification.
                if suppress_chime:
                    _timestamped_log(
                        "按键1检测到CODE变化：已取消声音播报。"
                    )
                else:
                    _play_code_change_chime()


def _draw_roi(
    display: np.ndarray,
    roi: tuple[float, float, float, float],
    active: bool,
) -> None:
    height, width = display.shape[:2]
    x1, y1, x2, y2 = roi
    color = (0, 220, 0) if active else (0, 0, 255)
    cv2.rectangle(
        display,
        (round(x1 * width), round(y1 * height)),
        (round(x2 * width), round(y2 * height)),
        color,
        2,
    )


_WATCHDOG_FONT_CACHE: dict[int, ImageFont.FreeTypeFont | ImageFont.ImageFont] = {}


def _watchdog_unicode_font(size: int):
    cached = _WATCHDOG_FONT_CACHE.get(size)
    if cached is not None:
        return cached
    candidates = (
        Path(os.environ.get("WINDIR", r"C:\Windows"))
        / "Fonts"
        / "msyh.ttc",
        Path(os.environ.get("WINDIR", r"C:\Windows"))
        / "Fonts"
        / "msyhbd.ttc",
        Path("/System/Library/Fonts/PingFang.ttc"),
        Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"),
    )
    for candidate in candidates:
        if not candidate.is_file():
            continue
        with contextlib.suppress(Exception):
            font = ImageFont.truetype(str(candidate), size=size)
            _WATCHDOG_FONT_CACHE[size] = font
            return font
    font = ImageFont.load_default()
    _WATCHDOG_FONT_CACHE[size] = font
    return font


def _draw_room_players_panel(
    display: np.ndarray,
    room_players: tuple[tuple[str, str], ...],
    room_disconnected: bool,
    round_player_entries: int,
    daily_player_entries: int,
    total_player_entries: int,
    round_player_entry_failures: int,
    daily_player_entry_failures: int,
    total_player_entry_failures: int,
) -> None:
    """Draw a UTF-8 room roster in the upper-right watchdog area."""
    shown_players = room_players[:3]
    panel_width = min(500, max(320, display.shape[1] // 3))
    player_row_count = max(1, len(shown_players))
    panel_height = 52 + 34 * player_row_count + 68
    left = max(0, display.shape[1] - panel_width - 12)
    top = min(display.shape[0] - panel_height, 135)
    right = min(display.shape[1], left + panel_width)
    bottom = min(display.shape[0], top + panel_height)
    panel = display[top:bottom, left:right].copy()
    overlay = panel.copy()
    overlay[:] = (0, 0, 0)
    cv2.addWeighted(overlay, 0.72, panel, 0.28, 0, panel)

    rgb = cv2.cvtColor(panel, cv2.COLOR_BGR2RGB)
    image = Image.fromarray(rgb)
    drawer = ImageDraw.Draw(image)
    title_font = _watchdog_unicode_font(23)
    item_font = _watchdog_unicode_font(21)
    panel_title = (
        "断开时房间内玩家：" if room_disconnected else "房间内玩家："
    )
    drawer.text((16, 8), panel_title, font=title_font, fill=(255, 255, 255))
    if shown_players:
        for index, (player_name, status) in enumerate(shown_players):
            drawer.text(
                (22, 42 + 33 * index),
                f"{player_name}({status})",
                font=item_font,
                fill=(255, 220, 120) if status == "路上" else (150, 255, 170),
            )
    else:
        drawer.text((22, 42), "（无）", font=item_font, fill=(190, 190, 190))
    drawer.text(
        (22, 42 + 33 * player_row_count),
        (
            f"本轮{max(0, round_player_entries)}人 "
            f"本日{max(0, daily_player_entries)}人 "
            f"总计{max(0, total_player_entries)}人"
        ),
        font=item_font,
        fill=(120, 220, 255),
    )
    drawer.text(
        (22, 42 + 33 * (player_row_count + 1)),
        (
            f"失败 本轮{max(0, round_player_entry_failures)}人 "
            f"本日{max(0, daily_player_entry_failures)}人 "
            f"总计{max(0, total_player_entry_failures)}人"
        ),
        font=item_font,
        fill=(255, 170, 120),
    )
    panel[:] = cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR)
    display[top:bottom, left:right] = panel


def render_watchdog_overlay(
    frame: np.ndarray,
    detection: PokopiaDetection,
    code: str,
    run_count: RunCounterSnapshot,
    code_timer_elapsed: float | None,
    completed_tasks: int,
    room_players: tuple[tuple[str, str], ...],
    room_disconnected: bool,
    round_player_entries: int,
    round_player_entry_failures: int,
    player_icon_visible: bool,
    player_notification_visible: bool,
    operation_locked: bool = False,
) -> np.ndarray:
    display = frame.copy()
    roi_states = (
        (SAVE_CHECK_ROI, detection.save_finished),
        (ABLE_ACCESS_ROI, detection.able_access),
        (BLACK_ROI, detection.black_screen),
        (CONNECT_ROI, detection.connect_ok),
        (CODE_PANEL_ROI, detection.code_panel),
        (CURSOR_AREA_ROI, detection.cursor != "INVALID"),
        (STAMP_BUTTON_ROI, detection.stamp in {"YES", "NO"}),
        (REWARD_ROI, detection.reward),
        (REWARD_LIST_ROI, detection.reward and detection.reward_lines > 0),
        (NETWORK_ERROR_CLOSE_ROI, detection.network_error),
        (PLAYER_CONNECTION_ICON_ROI, player_icon_visible),
        (PLAYER_NOTIFICATION_ROI, player_notification_visible),
        *((
            (
                REOPEN_STAGE_FEATURES[detection.reopen_stage][0],
                True,
            ),
        ) if detection.reopen_stage in REOPEN_STAGE_FEATURES else ()),
    )
    for roi, active in roi_states:
        _draw_roi(display, roi, active)

    if code_timer_elapsed is None:
        code_timer_text = "--:--"
    else:
        elapsed_seconds = max(0, int(code_timer_elapsed))
        elapsed_minutes, elapsed_remainder = divmod(elapsed_seconds, 60)
        code_timer_text = f"{elapsed_minutes:02d}:{elapsed_remainder:02d}"
    timer_limit_minutes, timer_limit_remainder = divmod(
        CODE_TIMER_RESTART_SECONDS,
        60,
    )
    timer_limit_text = (
        f"{timer_limit_minutes:02d}:{timer_limit_remainder:02d}"
    )
    lines = (
        f"POK_SAV_STATE: {'true' if detection.save_finished else 'FALSE'}",
        f"ABLE_ACCESS: {'ON' if detection.able_access else 'OFF'}",
        f"BLACK SCREEN: {'ON' if detection.black_screen else 'OFF'}",
        f"CONNECT_OK: {'ON' if detection.connect_ok else 'OFF'}",
        f"CODE: {code}",
        f"CODE TIMER: {code_timer_text} / {timer_limit_text}",
        f"OPERATION LOCK: {'ON' if operation_locked else 'OFF'}",
        f"CURSOR: {detection.cursor}",
        (
            f"REWARD: {'TRUE' if detection.reward else 'FALSE'}  "
            f"LINES:{detection.reward_lines}"
        ),
        f"TASKS COMPLETED: {min(3, max(0, completed_tasks))}/3",
        f"STAMP: {detection.stamp}",
        f"NETWORK ERROR: {'TRUE' if detection.network_error else 'FALSE'}",
        f"REOPEN STAGE: {detection.reopen_stage}",
        (
            "UPDATES  "
            f"TODAY:{run_count.daily_runs}  "
            f"TOTAL:{run_count.total_runs}"
        ),
    )
    panel_height = 14 + 25 * len(lines)
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.58
    font_thickness = 2
    panel_width = min(
        display.shape[1] - 8,
        max(
            cv2.getTextSize(
                line,
                font,
                font_scale,
                font_thickness,
            )[0][0]
            for line in lines
        )
        + 32,
    )
    overlay = display.copy()
    cv2.rectangle(
        overlay,
        (8, 8),
        (panel_width, panel_height),
        (0, 0, 0),
        -1,
    )
    cv2.addWeighted(overlay, 0.60, display, 0.40, 0.0, display)
    for index, line in enumerate(lines):
        value_on = line.endswith(("true", "TRUE", "ON", "YES")) or (
            line.startswith("CODE:") and bool(code)
        ) or line.startswith(("CODE TIMER:", "UPDATES"))
        color = (0, 255, 0) if value_on else (0, 170, 255)
        cv2.putText(
            display,
            line,
            (16, 30 + index * 25),
            font,
            font_scale,
            color,
            font_thickness,
            cv2.LINE_AA,
        )
    _draw_room_players_panel(
        display,
        room_players,
        room_disconnected,
        round_player_entries,
        run_count.daily_player_entries,
        run_count.total_player_entries,
        round_player_entry_failures,
        run_count.daily_player_entry_failures,
        run_count.total_player_entry_failures,
    )
    return display


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Smart Macro6 with a hard-recovery controller watchdog."
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
        help=f"输入规格；默认读取配置（回退为 {DEFAULT_CAPTURE_SPEC}）",
    )
    parser.add_argument(
        "--capture-fps",
        type=int,
        default=0,
        help=f"采集帧率；0表示读取配置（回退为 {DEFAULT_CAPTURE_FPS}）",
    )
    parser.add_argument(
        "--capture-timeout",
        type=float,
        default=0.0,
        help=(
            "每种输入规格等待首帧秒数；0表示读取配置"
            f"（回退为 {DEFAULT_CAPTURE_READ_TIMEOUT_SECONDS:g}）"
        ),
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
    if child.poll() is not None:
        return
    if os.name == "nt":
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
    """Restart the child when either macro or capture stops responding."""
    with tempfile.TemporaryDirectory(prefix="macro6-watchdog-") as raw_dir:
        heartbeat_path = Path(raw_dir) / "macro.heartbeat"
        capture_heartbeat_path = Path(raw_dir) / "capture.heartbeat"
        restart_count = 0

        while True:
            for stale_path in (heartbeat_path, capture_heartbeat_path):
                with contextlib.suppress(OSError):
                    stale_path.unlink()
            environment = dict(os.environ)
            environment[SUPERVISED_CHILD_ENV] = "1"
            environment[MACRO_HEARTBEAT_ENV] = str(heartbeat_path)
            environment[CAPTURE_HEARTBEAT_ENV] = str(capture_heartbeat_path)
            command = [
                sys.executable,
                str(Path(__file__).resolve()),
                *sys.argv[1:],
            ]
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
                            # Bad arguments, imports, or serial setup need a
                            # visible fix instead of an endless restart loop.
                            return return_code
                        reason = f"子程序异常退出（代码{return_code}）"
                        break

                    macro_age = _heartbeat_age(
                        heartbeat_path,
                        time.time(),
                    )
                    capture_age = _heartbeat_age(
                        capture_heartbeat_path,
                        time.time(),
                    )
                    if not monitoring_ready:
                        # Serial discovery may legitimately wait forever.  The
                        # watchdog is armed after both workers report in.
                        if macro_age is not None and capture_age is not None:
                            monitoring_ready = True
                            _timestamped_log(
                                "macro6 外层防卡死监控已就绪："
                                "宏线程和视频采集均已有心跳。"
                            )
                    else:
                        if (
                            macro_age is None
                            or macro_age > MACRO_HEARTBEAT_TIMEOUT_SECONDS
                        ):
                            reason = (
                                "宏线程超过"
                                f"{MACRO_HEARTBEAT_TIMEOUT_SECONDS:g}秒无心跳"
                            )
                            break
                        if (
                            capture_age is None
                            or capture_age > CAPTURE_HEARTBEAT_TIMEOUT_SECONDS
                        ):
                            reason = (
                                "视频采集超过"
                                f"{CAPTURE_HEARTBEAT_TIMEOUT_SECONDS:g}秒未刷新"
                            )
                            break
                    time.sleep(0.5)
            except KeyboardInterrupt:
                _terminate_supervised_child(child)
                return 130

            restart_count += 1
            _timestamped_log(
                f"检测到 macro6 程序卡死：{reason}。正在强制完整重启"
                f"（第{restart_count}次）；重启后回到 1/2/3/4 待机。"
            )
            _terminate_supervised_child(child)
            time.sleep(RESTART_DELAY_SECONDS)


def main() -> int:
    args = parse_args()
    config_path = _resolve_config_path(args.config)
    stop_event = threading.Event()
    pause_state = _PauseState()
    worker_errors: list[BaseException] = []
    heartbeat = _HeartbeatFile(MACRO_HEARTBEAT_ENV)
    capture_heartbeat = _HeartbeatFile(CAPTURE_HEARTBEAT_ENV)
    controller = None
    capture: FFmpegCapture | None = None
    web_publisher: PokopiaWebStatePublisher | None = None

    try:
        web_publisher = PokopiaWebStatePublisher()
        detector = PokopiaDetector()
        visual_state = _PokopiaVisualState()
        run_counter = Macro6RunCounter(
            Path(__file__).resolve().with_name(RUN_COUNTER_FILENAME)
        )
        code_archive = StampCodeArchive(
            Path(__file__).resolve().with_name(CODE_ARCHIVE_DIRNAME)
        )
        visual_state.set_reward_archive(code_archive)
        code_recognizer = CodeRecognizer(code_archive)
        room_player_tracker = RoomPlayerTracker(code_archive, run_counter)
        visual_state.set_room_player_tracker(room_player_tracker)
        player_notification_recognizer = PlayerNotificationRecognizer(
            room_player_tracker,
            code_archive,
        )
        _log_run_counter(run_counter.snapshot())
        capture_settings = resolve_capture_settings(args, config_path)
        capture = FFmpegCapture(
            capture_settings,
            stop_event,
            capture_heartbeat,
        )
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
        context = _HeartbeatMacroContext(
            controller,
            stop_event,
            pause_state,
            heartbeat,
        )
        context.macro_profile_name = "macro6"
        context.set_room_player_tracker(room_player_tracker)
        context.set_web_publisher(web_publisher)

        def record_reopen_screenshot(reason: str) -> None:
            snapshot = capture.snapshot()
            frame = (
                snapshot.raw_frame
                if snapshot.raw_frame is not None
                else snapshot.frame
            )
            if frame is None:
                _timestamped_log(
                    "重开前无可用采集帧，未能保存本轮结束截图。"
                )
                return
            code_archive.record_round_end_screenshot(
                frame,
                code_recognizer.snapshot(),
                run_counter.snapshot(),
                reason,
                context.current_macro_key(),
            )

        context.set_reopen_screenshot_callback(record_reopen_screenshot)

        def controller_worker() -> None:
            try:
                heartbeat.beat(force=True)
                # Macro6 starts directly in idle.  Unlike the former Smart
                # Gamepad path, watchdog startup sends no A-button detection
                # sequence to the console.
                _run_macro6_watchdog_loop(
                    context,
                    visual_state,
                    code_recognizer,
                    run_counter,
                )
            except BaseException as exc:
                worker_errors.append(exc)
                stop_event.set()
            finally:
                if context.status_callback is not None:
                    context.status_callback(True)
                    context.status_callback = None
                    _commit_status_line()

        workers = [
            threading.Thread(
                target=controller_worker,
                name="macro6-watchdog-controller",
                daemon=True,
            ),
            threading.Thread(
                target=_listen_for_watchdog_keyboard,
                args=(context, pause_state, run_counter, worker_errors),
                name="macro6-watchdog-keyboard",
                daemon=True,
            ),
        ]
        for worker in workers:
            worker.start()

        print(
            f"\n智能 macro6 watchdog 已启动：{selected_port}。\n"
            f"构建版本：{MACRO6_BUILD_ID}；"
            f"内置CODE字符模型：{len(_CODE_GLYPH_TEMPLATE_LABELS)}个。\n"
            f"实际源码：{Path(__file__).resolve()}\n"
            "宏按键说明：1=循环刷章；2=单次刷章；"
            "3=读档；4=开门；F12=操作锁；0=中断\n"
            "其他按键：F10=界面截图；\\|=查看记录；"
            "P=暂停/恢复；Q/Esc=退出。\n"
            "玩家动态识别：即将抵达=路上；已抵达=到达；"
            "回去了=从房间列表移除。\n"
            "手柄保活：连续300秒无实际手柄输入时自动按上、下各一次。\n"
            "执行期间新的 1/2/3/4 会被丢弃；"
            "普通手动控制键在宏执行期间仍然有效。"
            "外层防卡死监控已启用：宏线程或采集画面无响应时会完整重启；"
            "重启后安全返回 1/2/3/4 待机，不自动重放中断段。\n"
        )
        window_title = (
            f"{capture_settings.device_label} - Smart Macro6 Watchdog"
        )
        cv2.namedWindow(window_title, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window_title, CAPTURE_WIDTH, CAPTURE_HEIGHT)

        code_panel_detected_since: float | None = None
        room_code_revision = code_recognizer.revision()
        last_web_update_monotonic = 0.0
        while not stop_event.is_set():
            snapshot = capture.snapshot()
            if snapshot.frame is None:
                time.sleep(0.03)
                continue

            detection = detector.detect(snapshot.frame)
            visual_state.update(detection)
            player_notification_recognizer.request(
                snapshot.raw_frame
                if snapshot.raw_frame is not None
                else snapshot.frame
            )
            if detection.code_panel:
                now = time.monotonic()
                if code_panel_detected_since is None:
                    code_panel_detected_since = now
                elif (
                    now - code_panel_detected_since
                    >= CODE_PANEL_STABILIZATION_SECONDS
                ):
                    code_recognizer.request(
                        snapshot.frame,
                        detection,
                        suppress_chime=context.current_macro_key() == 1,
                    )
            else:
                code_panel_detected_since = None
                code_recognizer.note_code_panel_absent()
            current_code_revision = code_recognizer.revision()
            if current_code_revision != room_code_revision:
                room_code_revision = current_code_revision
                room_player_tracker.clear_for_new_code()
            room_disconnected, displayed_room_players = (
                room_player_tracker.display_snapshot()
            )
            count_snapshot = run_counter.snapshot()
            timer_elapsed = code_recognizer.timer_elapsed_seconds()
            now_web_update = time.monotonic()
            if now_web_update - last_web_update_monotonic >= 0.25:
                last_web_update_monotonic = now_web_update
                timer_active = context.web_phase_key() == "stamp_cycle"
                web_fields = dict(
                    macro_key=context.current_macro_key(),
                    operation_locked=context.operation_locked(),
                    room_disconnected=room_disconnected,
                    players=[
                        {"name": name, "status": status}
                        for name, status in displayed_room_players
                    ],
                    counts={
                        "operational_date": count_snapshot.operational_date,
                        "daily_runs": count_snapshot.daily_runs,
                        "total_runs": count_snapshot.total_runs,
                        "round_player_entries": room_player_tracker.current_round_entries(),
                        "round_player_entry_failures": room_player_tracker.current_round_entry_failures(),
                        "daily_player_entries": count_snapshot.daily_player_entries,
                        "total_player_entries": count_snapshot.total_player_entries,
                        "daily_player_entry_failures": count_snapshot.daily_player_entry_failures,
                        "total_player_entry_failures": count_snapshot.total_player_entry_failures,
                    },
                    tasks={
                        "completed": visual_state.completed_tasks(),
                        "total": 3,
                    },
                    timer={
                        "elapsed_seconds": (
                            round(timer_elapsed, 1)
                            if timer_elapsed is not None
                            else None
                        ),
                        "limit_seconds": CODE_TIMER_RESTART_SECONDS,
                        "remaining_seconds": (
                            max(0.0, round(CODE_TIMER_RESTART_SECONDS - timer_elapsed, 1))
                            if timer_elapsed is not None and timer_active
                            else None
                        ),
                        "active": timer_active,
                    },
                    detection={
                        "save_finished": detection.save_finished,
                        "black_screen": detection.black_screen,
                        "connect_ok": detection.connect_ok,
                        "code_panel": detection.code_panel,
                        "cursor": detection.cursor,
                        "reward": detection.reward,
                        "reward_lines": detection.reward_lines,
                        "stamp": detection.stamp,
                        "network_error": detection.network_error,
                    },
                )
                # While the macro worker is finishing the new screenshot and
                # announcement, keep the prior public CODE/status together.
                # set_web_code_ready() publishes the new tuple atomically.
                if context.web_phase_key() != "waiting_code_ocr":
                    web_fields.update(
                        code=code_recognizer.snapshot(),
                        code_unknown=code_recognizer.is_unknown(),
                        code_revision=current_code_revision,
                    )
                web_publisher.update(**web_fields)
            display = render_watchdog_overlay(
                snapshot.frame,
                detection,
                (
                    "未知"
                    if code_recognizer.is_unknown()
                    else code_recognizer.snapshot()
                ),
                count_snapshot,
                timer_elapsed,
                visual_state.completed_tasks(),
                displayed_room_players,
                room_disconnected,
                room_player_tracker.current_round_entries(),
                room_player_tracker.current_round_entry_failures(),
                player_notification_recognizer.icon_visible(),
                player_notification_recognizer.notification_visible(),
                context.operation_locked(),
            )
            if context.manual_screenshot_event.is_set():
                context.manual_screenshot_event.clear()
                code_archive.record_manual_screenshot(
                    snapshot.raw_frame
                    if snapshot.raw_frame is not None
                    else snapshot.frame,
                    detection,
                )
            cv2.imshow(window_title, display)

            key_code = cv2.waitKeyEx(1)
            key = {
                2490368: "UP",
                2621440: "DOWN",
                2424832: "LEFT",
                2555904: "RIGHT",
                7929856: "F10",
                8060928: "F12",
                27: "ESC",
            }.get(key_code, "")
            if not key and 0 <= key_code <= 255:
                key = chr(key_code)
            if key and not _handle_watchdog_control_key(
                key,
                context,
                pause_state,
                run_counter,
            ):
                break
            window_visible = 1.0
            with contextlib.suppress(Exception):
                window_visible = cv2.getWindowProperty(
                    window_title,
                    cv2.WND_PROP_VISIBLE,
                )
            if window_visible < 1:
                if context.operation_locked():
                    cv2.namedWindow(window_title, cv2.WINDOW_NORMAL)
                    cv2.resizeWindow(
                        window_title,
                        CAPTURE_WIDTH,
                        CAPTURE_HEIGHT,
                    )
                    _timestamped_log(
                        "操作锁已开启，忽略关闭 watchdog 窗口操作。"
                    )
                else:
                    stop_event.set()
                    break
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
        if web_publisher is not None:
            web_publisher.close(clean_exit=True)
        with contextlib.suppress(Exception):
            cv2.destroyAllWindows()


if __name__ == "__main__":
    if os.name == "nt":
        # Console-window close still terminates the process, while Ctrl+C can
        # no longer bypass the operation lock.
        signal.signal(signal.SIGINT, signal.SIG_IGN)
    if os.environ.get(SUPERVISED_CHILD_ENV) == "1":
        raise SystemExit(main())
    raise SystemExit(run_with_hard_recovery_supervisor())
