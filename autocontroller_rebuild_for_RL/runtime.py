from __future__ import annotations

import contextlib
import datetime as dt
import importlib
import json
import msvcrt
import os
import re
import select
import signal
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import urllib.parse
import urllib.request
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Set, Tuple

import cv2
import numpy as np


REPO_ROOT = Path(__file__).resolve().parent.parent
TABLETURF_SIM_ROOT = REPO_ROOT / "tableturf_sim"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(TABLETURF_SIM_ROOT) not in sys.path:
    sys.path.insert(0, str(TABLETURF_SIM_ROOT))

from src.assets.tableturf_types import Map_PointBit, Map_PointMask
from switch_connect.ui.terminal_select import choose_with_arrows
from switch_connect.virtual_gamepad.device_discovery import list_serial_port_labels, parse_device_from_label
from switch_connect.virtual_gamepad.input_mapper import (
    BIT_A,
    BIT_B,
    BIT_DPAD_DOWN,
    BIT_DPAD_LEFT,
    BIT_DPAD_RIGHT,
    BIT_DPAD_UP,
    BIT_HOME,
    BIT_L,
    BIT_PLUS,
    BIT_R,
    BIT_X,
    BIT_Y,
    REMOTE_INPUT_BITS,
    RemoteStep,
)
from switch_connect.virtual_gamepad.serial_controller import SerialRemoteController
from tableturf_vision.hand_card_detector import SLOT_NAMES as HAND_CARD_SLOT_NAMES, detect_hand_cards
from tableturf_vision.map_state_detector import MapStateTracker, detect_map_state, detect_map_state_from_frames
from tableturf_vision.mapper_preview import _match_card
from tableturf_vision.playable_detector import detect_draw_banner, detect_lose_banner, detect_playable_banner, detect_win_banner
from tableturf_vision.reference_matcher import detect_map_from_frame, load_map_info, map_name_cn_to_id
from tableturf_vision.settlement_map_state_detector import (
    analyze_settlement_map_state,
    analyze_settlement_map_state_from_frame_api,
)
from tableturf_vision.sp_detector import get_enemy_sp_count_frame, get_sp_count_frame
from tableturf_vision.tableturf_mapper import _load_layout
from vision_capture.adapter import (
    FFmpegCaptureSource,
    auto_detect_capture_device_name,
    is_usb_capture_device_name,
    list_avfoundation_video_devices,
)
from vision_capture.state_types import ObservedState
from src.engine.env_core import GameState, PlayerState, legal_actions
from src.engine.loaders import MAP_PADDING, load_map
from src.strategy import nn_loader
from src.strategy.registry import choose_action_from_strategy_id
from src.utils.common_utils import create_card_from_id
from src.utils.common_utils import _card_cells_on_map
from src.view.gamepad_ui import _find_local_view_rightbottom_self_special_anchor


class MissingInterfaceError(RuntimeError):
    """Raised when the orchestration flow needs fields not provided by current interfaces."""

    def __init__(self, missing_fields: Sequence[str]):
        self.missing_fields = list(missing_fields)
        joined = ", ".join(self.missing_fields)
        super().__init__(f"Missing required interface fields: {joined}")


class TargetWinGoalReached(RuntimeError):
    """Raised when the temporary/session target win count has been reached."""


MAP_STATE_MULTI_FRAME_DEFAULT = 5
MAP_STATE_MULTI_FRAME_PLAYABLE = 10
MAP_STATE_MULTI_FRAME_POLL_INTERVAL_SECONDS = 0.08
MAP_STATE_MULTI_FRAME_TIMEOUT_MARGIN_SECONDS = 2.0
SETTLEMENT_MULTI_FRAME_COUNT = 30
SETTLEMENT_MULTI_FRAME_POLL_INTERVAL_SECONDS = 0.08
SETTLEMENT_MULTI_FRAME_TIMEOUT_MARGIN_SECONDS = 2.0
CAPTURE_RECOVERY_RETRY_SECONDS = 5.0
CAPTURE_RECOVERY_MAX_RETRIES = 12
CAPTURE_RECOVERY_TOTAL_TIMEOUT_SECONDS = 60.0


_NO_PROXY_OPENER = urllib.request.build_opener(urllib.request.ProxyHandler({}))


def _urlopen_local_direct(req_or_url: Any, timeout: float):
    url = req_or_url.full_url if isinstance(req_or_url, urllib.request.Request) else str(req_or_url)
    parsed = urllib.parse.urlparse(str(url))
    host = str(parsed.hostname or "").strip().lower()
    if host in {"127.0.0.1", "localhost", "::1"}:
        return _NO_PROXY_OPENER.open(req_or_url, timeout=timeout)
    return urllib.request.urlopen(req_or_url, timeout=timeout)


def _unique_debug_image_path(out_dir: Path, prefix: str, idx: int) -> Path:
    ts = dt.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    return out_dir / f"{prefix}_{ts}_{idx:05d}.png"


def _load_callable(ref: str) -> Callable[..., Any]:
    if ":" not in ref:
        raise ValueError(f"callable ref must be module:function, got: {ref}")
    module_name, func_name = ref.split(":", 1)
    module = importlib.import_module(module_name)
    func = getattr(module, func_name, None)
    if func is None or not callable(func):
        raise ValueError(f"callable not found: {ref}")
    return func


def _load_json_obj(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def _write_json_obj(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _usb_capture_device_names() -> List[str]:
    return [name for name in _all_video_device_names() if is_usb_capture_device_name(name)]


def _all_video_device_names() -> List[str]:
    names = [str(name).strip() for name in list_avfoundation_video_devices()]
    seen: Set[str] = set()
    out: List[str] = []
    for name in names:
        if not name or name in seen:
            continue
        seen.add(name)
        out.append(name)
    return out


def _board_label_to_mask(label: str) -> int:
    mapping = {
        "invalid": int(Map_PointMask.NotMap),
        "empty": int(Map_PointMask.Empty),
        "p1_fill": int(Map_PointMask.P1Normal),
        "p1_special": int(Map_PointMask.P1Special),
        "p1_special_activated": int(Map_PointMask.P1SpActive),
        "p2_fill": int(Map_PointMask.P2Normal),
        "p2_special": int(Map_PointMask.P2Special),
        "p2_special_activated": int(Map_PointMask.P2SpActive),
        "conflict": int(Map_PointMask.Conflict),
        "changed": int(Map_PointMask.Empty),
    }
    return mapping.get(label, int(Map_PointMask.Empty))


def _extract_board_grid(board_labels: List[List[str]]) -> List[List[int]]:
    return [[_board_label_to_mask(label) for label in row] for row in board_labels]


def _pad_board_labels_to_engine_dims(map_id: str, board_labels: List[List[str]]) -> List[List[str]]:
    game_map = load_map(map_id)
    target_h = int(game_map.height)
    target_w = int(game_map.width)
    src_h = len(board_labels)
    src_w = len(board_labels[0]) if src_h else 0
    if src_h == target_h and src_w == target_w:
        return [row[:] for row in board_labels]
    out = [["invalid" for _ in range(target_w)] for _ in range(target_h)]
    for y in range(src_h):
        for x in range(src_w):
            oy = y + MAP_PADDING
            ox = x + MAP_PADDING
            if 0 <= oy < target_h and 0 <= ox < target_w:
                out[oy][ox] = str(board_labels[y][x])
    return out


def _pad_grid_to_engine_dims(map_id: str, grid: List[List[int]]) -> List[List[int]]:
    game_map = load_map(map_id)
    target_h = int(game_map.height)
    target_w = int(game_map.width)
    src_h = len(grid)
    src_w = len(grid[0]) if src_h else 0
    if src_h == target_h and src_w == target_w:
        return [row[:] for row in grid]
    pad = MAP_PADDING
    out = [[int(Map_PointMask.NotMap) for _ in range(target_w)] for _ in range(target_h)]
    for y in range(src_h):
        for x in range(src_w):
            oy = y + pad
            ox = x + pad
            if 0 <= oy < target_h and 0 <= ox < target_w:
                out[oy][ox] = int(grid[y][x])
    return out


def _slot_rank(slot_name: str) -> int:
    try:
        return HAND_CARD_SLOT_NAMES.index(slot_name)
    except ValueError:
        return len(HAND_CARD_SLOT_NAMES)


def _map_info_by_id(map_id: str) -> Dict[str, Any]:
    for row in load_map_info():
        if str(row.get("id", "")) == str(map_id):
            return row
    raise ValueError(f"map info not found for map_id={map_id}")


def _map_state_to_board_labels(map_id: str, map_state_result: Dict[str, Any]) -> List[List[str]]:
    map_info = _map_info_by_id(map_id)
    point_type = map_info["point_type"]
    labels = [["invalid" if int(cell) == 0 else "transparent" for cell in row] for row in point_type]
    for cell in map_state_result.get("cells", []):
        row = int(cell["json_row"])
        col = int(cell["json_col"])
        labels[row][col] = str(cell["label"])
    return labels


def _pad_map_match_payload(map_id: str, map_match: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(map_match)
    template_board = out.get("board_from_template")
    if isinstance(template_board, dict):
        labels = template_board.get("labels")
        if isinstance(labels, list):
            out["board_from_template"] = {
                **template_board,
                "labels": _pad_board_labels_to_engine_dims(map_id, labels),
            }
    return out


def _compact_settlement_map_state_for_replay(map_id: str, settlement_result: Dict[str, Any]) -> Dict[str, Any]:
    raw_labels = _map_state_to_board_labels(map_id, settlement_result)
    return {
        "map_name": str(settlement_result.get("map_name", "") or ""),
        "reference_point_count": int(settlement_result.get("reference_point_count", 0) or 0),
        "counts": dict(settlement_result.get("counts", {}) or {}),
        "board_labels": raw_labels,
        "map_grid": _extract_board_grid(raw_labels),
    }


def _normalize_hand_slots(hand_result: Dict[str, Any]) -> List[Dict[str, Any]]:
    slots = list(hand_result.get("slots", []))
    slots.sort(key=lambda slot: _slot_rank(str(slot.get("slot", ""))))
    return slots


def _unpad_grid_for_replay(map_id: str, grid: List[List[int]]) -> List[List[int]]:
    game_map = load_map(map_id)
    target_h = int(game_map.height)
    target_w = int(game_map.width)
    src_h = len(grid)
    src_w = len(grid[0]) if src_h else 0
    if src_h == max(0, target_h - MAP_PADDING * 2) and src_w == max(0, target_w - MAP_PADDING * 2):
        return [[int(cell) for cell in row] for row in grid]
    if src_h == target_h and src_w == target_w and target_h > MAP_PADDING * 2 and target_w > MAP_PADDING * 2:
        return [
            [int(cell) for cell in row[MAP_PADDING:target_w - MAP_PADDING]]
            for row in grid[MAP_PADDING:target_h - MAP_PADDING]
        ]
    return [[int(cell) for cell in row] for row in grid]


def _engine_xy_to_replay_xy(map_id: str, x: Optional[int], y: Optional[int], grid: List[List[int]]) -> Tuple[Optional[int], Optional[int]]:
    if x is None or y is None:
        return (None, None)
    game_map = load_map(map_id)
    target_h = int(game_map.height)
    target_w = int(game_map.width)
    src_h = len(grid)
    src_w = len(grid[0]) if src_h else 0
    if src_h == target_h and src_w == target_w and target_h > MAP_PADDING * 2 and target_w > MAP_PADDING * 2:
        return (int(x) - MAP_PADDING, int(y) - MAP_PADDING)
    return (int(x), int(y))


def _compute_board_scores_from_grid(grid: List[List[int]]) -> Tuple[int, int]:
    p1 = 0
    p2 = 0
    for row in grid:
        for cell in row:
            mask = int(cell)
            if (mask & int(Map_PointBit.IsValid)) == 0:
                continue
            is_p1 = (mask & int(Map_PointBit.IsP1)) != 0
            is_p2 = (mask & int(Map_PointBit.IsP2)) != 0
            if is_p1 and not is_p2:
                p1 += 1
            elif is_p2 and not is_p1:
                p2 += 1
    return p1, p2


def _format_score_broadcast(p1_score: int, p2_score: int) -> str:
    diff = int(p1_score) - int(p2_score)
    if diff > 0:
        lead_text = f"我方领先 {diff} 格"
    elif diff < 0:
        lead_text = f"对方领先 {abs(diff)} 格"
    else:
        lead_text = "当前平分"
    return f"比分播报：我方 {int(p1_score)}，对方 {int(p2_score)}。{lead_text}。"


def _resolve_serial_port(configured_port: str, pick_serial: bool) -> str:
    if configured_port.strip():
        return configured_port.strip()
    if not pick_serial:
        labels = list_serial_port_labels()
        if len(labels) == 1:
            return parse_device_from_label(labels[0])
    labels = list_serial_port_labels()
    if not labels:
        raise RuntimeError("No serial ports found for virtual gamepad")
    picked = choose_with_arrows(labels, "Select virtual gamepad serial port")
    if not picked:
        raise RuntimeError("Serial port selection cancelled")
    return parse_device_from_label(picked)


@dataclass
class ManualVisionFields:
    selected_hand_index: Optional[int] = None
    cursor_xy: Optional[Tuple[int, int]] = None
    rotation: Optional[int] = None
    p1_sp: Optional[int] = None


@dataclass
class SupplementalState:
    selected_hand_index: Optional[int] = None
    cursor_xy: Optional[Tuple[int, int]] = None
    rotation: Optional[int] = None
    p1_sp: Optional[int] = None

    @classmethod
    def from_payload(cls, payload: Dict[str, Any]) -> "SupplementalState":
        cursor_xy = payload.get("cursor_xy")
        cursor_tuple = tuple(cursor_xy) if isinstance(cursor_xy, (list, tuple)) and len(cursor_xy) == 2 else None
        return cls(
            selected_hand_index=payload.get("selected_hand_index"),
            cursor_xy=cursor_tuple,
            rotation=payload.get("rotation"),
            p1_sp=payload.get("p1_sp"),
        )


@dataclass
class ControllerConfig:
    frame_api_url: str = ""
    frame_api_auto_start: bool = True
    frame_api_health_url: str = ""
    frame_api_launch_script: str = "vision_capture/preview_stream_opencv.py"
    frame_api_launch_config: str = "vision_capture/capture_config.json"
    frame_api_startup_seconds: float = 15.0
    capture_device_name: str = ""
    capture_width: int = 1920
    capture_height: int = 1080
    capture_fps: int = 30
    capture_pixel_format: str = ""
    capture_read_timeout_seconds: float = 5.0
    capture_drain_ms: int = 60
    serial_port: str = ""
    pick_serial: bool = True
    wait_press_hold_ms: int = 110
    wait_press_gap_ms: int = 1300
    playable_poll_seconds: float = 0.35
    max_turns: int = 12
    continuous_run: bool = True
    target_win_count: int = 1
    progress_timeout_seconds: float = 60.0
    strategy_id: str = "default:aggressive:high"
    strategy_id_by_map: Dict[str, str] = field(default_factory=dict)
    strategy_id_by_map_name: Dict[str, str] = field(default_factory=dict)
    policy_config_json: str = "autocontroller_rebuild_for_RL/strategy_policy.example.json"
    strict_missing_interfaces: bool = True
    layout_json: str = "tableturf_vision/tableturf_layout.json"
    manual_fields: ManualVisionFields = field(default_factory=ManualVisionFields)
    supplemental_state_provider: str = ""
    debug_ui_enabled: bool = True
    save_debug_frames: bool = True
    debug_frame_dir: str = "autocontroller_rebuild_for_RL/debug_runtime/screenshot"
    log_file: str = "autocontroller_rebuild_for_RL/debug_runtime/log/autocontroller.log"
    global_stats_file: str = "autocontroller_rebuild_for_RL/map_battle_stats.json"
    battle_replay_dir: str = "autocontroller_rebuild_for_RL/replays"
    enable_battle_replay: bool = True
    stats_mode: str = "map"
    aggregate_stats_file: str = ""
    aggregate_stats_name: str = "session"
    map_state_frame_count_default: int = MAP_STATE_MULTI_FRAME_DEFAULT
    map_state_frame_count_playable: int = MAP_STATE_MULTI_FRAME_PLAYABLE
    settlement_frame_count: int = SETTLEMENT_MULTI_FRAME_COUNT
    preserve_p1_special_history: bool = False
    clone_jelly_split_action_execution: bool = False
    clone_jelly_sp_attack_up_right: bool = False

    @classmethod
    def from_json(cls, path: Path) -> "ControllerConfig":
        data = json.loads(path.read_text(encoding="utf-8"))
        manual_fields = ManualVisionFields(**data.get("manual_fields", {}))
        payload = dict(data)
        payload["manual_fields"] = manual_fields
        return cls(**payload)


@dataclass
class ParsedTurnState:
    map_id: str
    map_name: str
    hand_card_numbers: List[int]
    map_grid: List[List[int]]
    playable_result: Dict[str, Any]
    analysis_result: Dict[str, Any]
    card_matches: List[Dict[str, Any]]
    turn: int
    selected_hand_index: Optional[int]
    cursor_xy: Optional[Tuple[int, int]]
    rotation: int
    p1_sp: int
    p2_sp: int

    def to_observed_state(self) -> ObservedState:
        return ObservedState(
            map_id=self.map_id,
            hand_card_numbers=self.hand_card_numbers,
            p1_sp=self.p1_sp,
            turn=self.turn,
            map_grid=self.map_grid,
            selected_hand_index=self.selected_hand_index,
            cursor_xy=self.cursor_xy,
            rotation=self.rotation,
        )


@dataclass
class ResolvedStrategy:
    mode: str
    label: str
    strategy_id: str = ""
    checkpoint_file: str = ""
    source: str = ""


class RuntimeLogWriter:
    def __init__(self, path: Path):
        self._path = self._session_log_path(path)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()

    @staticmethod
    def _session_log_path(path: Path) -> Path:
        if path.parent.name == "debug_runtime":
            path = path.parent / "log" / path.name
        ts = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
        stem = path.stem or "autocontroller"
        suffix = path.suffix or ".log"
        return path.with_name(f"{stem}_{ts}{suffix}")

    @property
    def path(self) -> Path:
        return self._path

    def write(self, message: str, tag: str = "SYSTEM") -> None:
        normalized_tag = str(tag or "SYSTEM").strip().upper() or "SYSTEM"
        line = f"{dt.datetime.now().strftime('%Y-%m-%d %H:%M:%S')} [{normalized_tag}] {message}\n"
        with self._lock:
            self._path.open("a", encoding="utf-8").write(line)


def _resolve_debug_screenshot_dir(path_value: str) -> Path:
    path = Path(path_value)
    if not path.is_absolute():
        path = REPO_ROOT / path
    if path.name == "debug_runtime":
        path = path / "screenshot"
    return path


def _resolve_debug_log_path(path_value: str) -> Path:
    path = Path(path_value)
    if not path.is_absolute():
        path = REPO_ROOT / path
    if path.parent.name == "debug_runtime":
        path = path.parent / "log" / path.name
    return path


def _resolve_debug_tmp_dir(screenshot_dir: Path) -> Path:
    if screenshot_dir.name == "screenshot":
        return screenshot_dir.parent / "tmp"
    return screenshot_dir / "tmp"


class BattleReplayWriter:
    def __init__(self, out_dir: Path):
        self._out_dir = out_dir
        self._out_dir.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self.reset()

    def reset(self) -> None:
        self._active = False
        self._battle_index = 0
        self._started_at = ""
        self._meta: Dict[str, Any] = {
            "map_id": "",
            "p1_deck": [],
            "p2_deck": [],
            "replay_version": 2,
            "rng_seed": None,
            "p1_draw_order": None,
            "p2_draw_order": None,
        }
        self._moves: List[Dict[str, Any]] = []
        self._extras: Dict[str, Any] = {
            "battle_status": "incomplete",
            "map_name": "",
            "map_info": None,
            "settlement_map_state": None,
            "coordinate_system": "raw_map_without_padding",
            "missing_fields": [
                "meta.p1_deck",
                "meta.p2_deck",
                "meta.rng_seed",
                "meta.p1_draw_order",
                "meta.p2_draw_order",
                "result.p1_score",
                "result.p2_score",
                "last move p1_sp_after/p2_sp_after may be unavailable without a post-battle state frame",
                "P2 turns and hidden hand information",
            ],
        }

    @staticmethod
    def _safe_name(name: str, default: str = "NA", limit: int = 20) -> str:
        raw = "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in str(name))
        raw = raw.strip("_") or default
        return raw[:limit]

    def start_battle(self, battle_index: int) -> None:
        with self._lock:
            self.reset()
            self._active = True
            self._battle_index = int(battle_index)
            self._started_at = dt.datetime.now().isoformat(timespec="seconds")

    def record_turn(self, state: ParsedTurnState, action: Any) -> None:
        with self._lock:
            if not self._active:
                return
            if not self._meta["map_id"]:
                self._meta["map_id"] = str(state.map_id)
            self._extras["map_name"] = str(state.map_name)
            with contextlib.suppress(Exception):
                self._extras["map_info"] = _map_info_by_id(state.map_id)
            card_idx = None
            if action.card_number is not None:
                for idx, number in enumerate(state.hand_card_numbers):
                    if int(number) == int(action.card_number):
                        card_idx = idx
                        break
            replay_x, replay_y = _engine_xy_to_replay_xy(state.map_id, action.x, action.y, state.map_grid)
            move = {
                "turn": int(state.turn),
                "player": "P1",
                "phase": "select",
                "pass": bool(action.pass_turn),
                "card_idx": card_idx,
                "card_number": int(action.card_number) if action.card_number is not None else None,
                "used_sp": bool(action.use_sp_attack) if action.card_number is not None else None,
                "x": replay_x,
                "y": replay_y,
                "rotation": int(action.rotation) if action.rotation is not None else None,
                "valid_action": None if action.surrender else True,
                "invalid_reason": "surrender" if action.surrender else None,
                "p1_sp_before": int(state.p1_sp),
                "p1_sp_after": None,
                "p2_sp_before": int(state.p2_sp),
                "p2_sp_after": None,
                "hand_card_numbers": [int(v) for v in state.hand_card_numbers],
                "map_grid": _unpad_grid_for_replay(state.map_id, state.map_grid),
            }
            self._moves.append(move)

    def complete_previous_move_after_state(self, state: ParsedTurnState) -> None:
        with self._lock:
            if not self._active or not self._moves:
                return
            prev = self._moves[-1]
            if prev.get("p1_sp_after") is None:
                prev["p1_sp_after"] = int(state.p1_sp)
            if prev.get("p2_sp_after") is None:
                prev["p2_sp_after"] = int(state.p2_sp)

    def discard_last_move(self) -> None:
        with self._lock:
            if not self._active or not self._moves:
                return
            self._moves.pop()

    def finalize(
        self,
        *,
        battle_status: str,
        winner: Optional[str],
        rounds_played: Optional[int],
        map_id: str,
        map_name: str,
        settlement_map_state: Optional[Dict[str, Any]] = None,
    ) -> Optional[Path]:
        with self._lock:
            if not self._active:
                return None
            if map_id and not self._meta["map_id"]:
                self._meta["map_id"] = str(map_id)
            self._extras["battle_status"] = str(battle_status)
            self._extras["map_name"] = str(map_name or self._extras.get("map_name", ""))
            if settlement_map_state is not None:
                self._extras["settlement_map_state"] = settlement_map_state
            self._extras["generated_at"] = dt.datetime.now().isoformat(timespec="seconds")
            self._extras["battle_index"] = int(self._battle_index)
            self._extras["started_at"] = self._started_at
            payload = {
                "meta": dict(self._meta),
                "moves": list(self._moves),
                "result": {
                    "p1_score": None,
                    "p2_score": None,
                    "winner": winner,
                    "rounds_played": int(rounds_played) if rounds_played is not None else None,
                },
                "autocontroller": dict(self._extras),
            }
            ts = self._started_at.replace("-", "").replace(":", "").replace("T", "_")[:15] if self._started_at else dt.datetime.now().strftime("%Y%m%d_%H%M%S")
            map_part = self._safe_name(map_id or map_name, default="UnknownMap")
            status_part = self._safe_name(battle_status, default="status")
            path = self._out_dir / f"Replay_{ts}_battle{self._battle_index:04d}_{map_part}_{status_part}.json"
            path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
            self._active = False
            return path


class NoOpBattleReplayWriter:
    def start_battle(self, battle_index: int) -> None:
        del battle_index

    def record_turn(self, state: ParsedTurnState, action: Any) -> None:
        del state, action

    def complete_previous_move_after_state(self, state: ParsedTurnState) -> None:
        del state

    def discard_last_move(self) -> None:
        return

    def finalize(
        self,
        *,
        battle_status: str,
        winner: Optional[str],
        rounds_played: Optional[int],
        map_id: str,
        map_name: str,
        settlement_map_state: Optional[Dict[str, Any]] = None,
    ) -> Optional[Path]:
        del battle_status, winner, rounds_played, map_id, map_name, settlement_map_state
        return None


class GlobalMapStatsWriter:
    def __init__(self, path: Path):
        self._path = path
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        ordered = list(map_name_cn_to_id().keys())
        self._order = [str(name) for name in ordered]
        self._stats: Dict[str, Dict[str, int]] = {
            name: {"Battles": 0, "Wins": 0, "Errors": 0}
            for name in self._order
        }
        self._load_existing()
        self._write()

    def _load_existing(self) -> None:
        if not self._path.exists():
            return
        try:
            payload = json.loads(self._path.read_text(encoding="utf-8"))
            maps = payload.get("maps", {}) if isinstance(payload, dict) else {}
            if not isinstance(maps, dict):
                return
            for map_name, row_payload in maps.items():
                if not isinstance(row_payload, dict):
                    continue
                row = self._stats.setdefault(str(map_name), {"Battles": 0, "Wins": 0, "Errors": 0})
                for key in ("Battles", "Wins", "Errors"):
                    try:
                        row[key] = int(row_payload.get(key, row.get(key, 0)))
                    except Exception:
                        continue
        except Exception:
            return

    def _payload(self) -> Dict[str, Any]:
        names = list(self._order)
        for name in self._stats:
            if name not in names:
                names.append(name)
        maps: Dict[str, Dict[str, int]] = {}
        for name in names:
            row = self._stats.get(name, {"Battles": 0, "Wins": 0, "Errors": 0})
            maps[name] = {
                "Battles": int(row.get("Battles", 0)),
                "Wins": int(row.get("Wins", 0)),
                "Errors": int(row.get("Errors", 0)),
            }
        return {
            "format": "map_battle_stats_v1",
            "maps": maps,
        }

    def _write(self) -> None:
        self._path.write_text(json.dumps(self._payload(), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    def increment(self, map_name: str, key: str, amount: int = 1) -> None:
        if key not in {"Battles", "Wins", "Errors"}:
            raise ValueError(f"unsupported stats key: {key}")
        name = str(map_name).strip()
        if not name:
            return
        with self._lock:
            row = self._stats.setdefault(name, {"Battles": 0, "Wins": 0, "Errors": 0})
            row[key] = int(row.get(key, 0)) + int(amount)
            self._write()


class AggregateStatsWriter:
    def __init__(self, path: Path, name: str):
        self._path = path
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._name = str(name or "session")
        self._stats: Dict[str, int] = {"Battles": 0, "Wins": 0, "Errors": 0}
        self._load_existing()
        self._write()

    def _load_existing(self) -> None:
        if not self._path.exists():
            return
        try:
            payload = json.loads(self._path.read_text(encoding="utf-8"))
        except Exception:
            return
        stats = payload.get("stats", {}) if isinstance(payload, dict) else {}
        if not isinstance(stats, dict):
            return
        for key in ("Battles", "Wins", "Errors"):
            try:
                self._stats[key] = int(stats.get(key, self._stats[key]))
            except Exception:
                continue

    def _write(self) -> None:
        payload = {
            "format": "aggregate_battle_stats_v1",
            "name": self._name,
            "stats": {
                "Battles": int(self._stats.get("Battles", 0)),
                "Wins": int(self._stats.get("Wins", 0)),
                "Errors": int(self._stats.get("Errors", 0)),
            },
        }
        self._path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    def increment(self, map_name: str, key: str, amount: int = 1) -> None:
        del map_name
        if key not in {"Battles", "Wins", "Errors"}:
            raise ValueError(f"unsupported stats key: {key}")
        with self._lock:
            self._stats[key] = int(self._stats.get(key, 0)) + int(amount)
            self._write()


_STRATEGIC_MODEL_CACHE: Dict[str, Any] = {}
_BASE_MODEL_CACHE: Dict[str, Any] = {}


def _json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_safe(v) for v in value]
    if hasattr(value, "__dict__"):
        return {
            "__type__": value.__class__.__name__,
            **{str(k): _json_safe(v) for k, v in vars(value).items()},
        }
    return repr(value)


class HttpJpegCaptureSource:
    def __init__(self, frame_api_url: str):
        self.frame_api_url = frame_api_url
        self.last_error: Optional[str] = None
        self._cache_path = Path(tempfile.gettempdir()) / "tableturf_http_frame_latest.jpg"

    def read_latest(self, timeout_seconds: float = 5.0, drain_ms: int = 0) -> Optional[np.ndarray]:
        del drain_ms
        req = urllib.request.Request(self.frame_api_url, headers={"Cache-Control": "no-cache"})
        deadline = time.monotonic() + max(0.5, float(timeout_seconds))
        last_error = "HTTP_FRAME_FETCH_FAILED:unknown"
        while time.monotonic() < deadline:
            try:
                with _urlopen_local_direct(req, timeout=min(1.0, max(0.3, float(timeout_seconds)))) as resp:
                    payload = resp.read()
            except Exception as exc:
                last_error = f"HTTP_FRAME_FETCH_FAILED:{exc}"
                time.sleep(0.15)
                continue
            try:
                self._cache_path.write_bytes(payload)
            except Exception as exc:
                self.last_error = f"HTTP_FRAME_SAVE_FAILED:{exc}"
                return None
            frame = cv2.imread(str(self._cache_path), cv2.IMREAD_COLOR)
            if frame is None or frame.size == 0:
                last_error = "HTTP_FRAME_DECODE_FAILED"
                time.sleep(0.1)
                continue
            self.last_error = None
            return frame
        self.last_error = last_error
        return None

    def read_with_fallbacks(
        self,
        timeout_seconds: float = 5.0,
        fallback_specs: Optional[List[Dict[str, object]]] = None,
    ) -> Optional[np.ndarray]:
        del fallback_specs
        return self.read_latest(timeout_seconds=timeout_seconds, drain_ms=0)

    def stop(self) -> None:
        return


class FrameApiAutoLauncher:
    def __init__(self, config: ControllerConfig):
        self._config = config
        self._proc: Optional[subprocess.Popen] = None
        self._last_launch_output: str = ""

    def _launch_script_path(self) -> Path:
        return _resolve_repo_path(self._config.frame_api_launch_script)

    def _launch_config_path(self) -> Path:
        return _resolve_repo_path(self._config.frame_api_launch_config)

    def _launch_cmd(self) -> List[str]:
        return [sys.executable, str(self._launch_script_path()), "--config", str(self._launch_config_path())]

    def _cleanup_stale_preview_processes(self) -> None:
        if os.name == "nt":
            return
        script_path = str(self._launch_script_path())
        try:
            proc = subprocess.run(
                ["pgrep", "-af", script_path],
                capture_output=True,
                text=True,
                check=False,
            )
        except Exception:
            return

        current_pid = self._proc.pid if self._proc is not None and self._proc.poll() is None else None
        for line in str(proc.stdout or "").splitlines():
            parts = line.strip().split(maxsplit=1)
            if not parts:
                continue
            try:
                pid = int(parts[0])
            except Exception:
                continue
            cmdline = parts[1] if len(parts) > 1 else ""
            if pid == current_pid:
                continue
            if script_path not in cmdline:
                continue
            with contextlib.suppress(Exception):
                os.kill(pid, signal.SIGTERM)
        time.sleep(0.5)
        try:
            proc = subprocess.run(
                ["pgrep", "-af", script_path],
                capture_output=True,
                text=True,
                check=False,
            )
        except Exception:
            return
        for line in str(proc.stdout or "").splitlines():
            parts = line.strip().split(maxsplit=1)
            if not parts:
                continue
            try:
                pid = int(parts[0])
            except Exception:
                continue
            cmdline = parts[1] if len(parts) > 1 else ""
            if pid == current_pid:
                continue
            if script_path not in cmdline:
                continue
            with contextlib.suppress(Exception):
                os.kill(pid, signal.SIGKILL)

    def _health_url(self) -> str:
        explicit = str(self._config.frame_api_health_url or "").strip()
        if explicit:
            return explicit
        base = str(self._config.frame_api_url or "").strip()
        if base.endswith("/frame.jpg"):
            return base[:-10] + "/health"
        if base.endswith("/frame.jpeg"):
            return base[:-11] + "/health"
        return base.rstrip("/") + "/health"

    def _ensure_launch_target_selected(self) -> None:
        launch_config_path = self._launch_config_path()
        launch_cfg = _load_json_obj(launch_config_path)
        configured_name = str(launch_cfg.get("device_name", "") or self._config.capture_device_name or "").strip()
        if configured_name and configured_name.lower() != "invalid":
            self._config.capture_device_name = configured_name
            return

        all_devices = _all_video_device_names()
        available_usb = [name for name in all_devices if is_usb_capture_device_name(name)]
        configured_allow_non_usb = bool(launch_cfg.get("allow_non_usb", False))
        if configured_name and configured_name in all_devices and (
            is_usb_capture_device_name(configured_name) or configured_allow_non_usb
        ):
            return
        choices = available_usb or all_devices
        allow_non_usb = not bool(available_usb)
        if not choices:
            raise RuntimeError("NO_VIDEO_CAPTURE_DEVICE_AVAILABLE")
        picked = ""
        if sys.stdin.isatty() and sys.stdout.isatty():
            title = "未识别到采集卡，请从全部视频设备中手动选择" if allow_non_usb else "选择可用的采集卡设备"
            picked = str(choose_with_arrows(choices, title) or "").strip()
        else:
            if allow_non_usb:
                raise RuntimeError("VIDEO_CAPTURE_MANUAL_SELECTION_REQUIRED")
            picked = str(choices[0]).strip()
        if not picked:
            raise RuntimeError("CAPTURE_DEVICE_SELECTION_CANCELLED")
        launch_cfg["device_name"] = picked
        launch_cfg["pick_device"] = False
        launch_cfg["allow_non_usb"] = bool(allow_non_usb or not is_usb_capture_device_name(picked))
        _write_json_obj(launch_config_path, launch_cfg)
        self._config.capture_device_name = picked
        runtime_config_path = str(getattr(self._config, "_runtime_config_path", "") or "").strip()
        if runtime_config_path:
            runtime_payload = _load_json_obj(Path(runtime_config_path))
            runtime_payload["capture_device_name"] = picked
            _write_json_obj(Path(runtime_config_path), runtime_payload)

    def is_ready(self, timeout_seconds: float = 1.0) -> bool:
        health_url = self._health_url()
        if health_url:
            req = urllib.request.Request(health_url, headers={"Cache-Control": "no-cache"})
            try:
                with _urlopen_local_direct(req, timeout=max(0.3, timeout_seconds)) as resp:
                    payload = json.loads(resp.read().decode("utf-8", errors="ignore"))
                if bool(payload.get("has_frame")):
                    return True
            except Exception:
                pass
        frame_url = str(self._config.frame_api_url or "").strip()
        if not frame_url:
            return False
        req = urllib.request.Request(frame_url, headers={"Cache-Control": "no-cache"})
        try:
            with _urlopen_local_direct(req, timeout=max(0.3, timeout_seconds)) as resp:
                content_type = str(resp.headers.get("Content-Type", "")).lower()
                payload = resp.read(32)
            return bool(payload) and ("jpeg" in content_type or "image/" in content_type)
        except Exception:
            return False

    def ensure_started(self) -> None:
        if not str(self._config.frame_api_url or "").strip():
            return
        if self.is_ready(timeout_seconds=0.8):
            return
        if not self._config.frame_api_auto_start:
            raise RuntimeError("FRAME_API_NOT_READY_AND_AUTO_START_DISABLED")
        self._ensure_launch_target_selected()

        if self._proc is not None and self._proc.poll() is None:
            self.stop()
        self._cleanup_stale_preview_processes()
        cmd = self._launch_cmd()
        self._last_launch_output = ""
        self._proc = subprocess.Popen(
            cmd,
            cwd=str(REPO_ROOT),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            stdin=subprocess.DEVNULL,
            text=True,
            encoding="utf-8",
            errors="ignore",
        )
        deadline = time.monotonic() + max(1.0, float(self._config.frame_api_startup_seconds))
        while time.monotonic() < deadline:
            if self._proc.poll() is not None:
                output = ""
                with contextlib.suppress(Exception):
                    output = str(self._proc.stdout.read() if self._proc.stdout is not None else "").strip()
                self._last_launch_output = output
                if output:
                    tail = " | ".join([line.strip() for line in output.splitlines()[-6:] if line.strip()])
                    raise RuntimeError(f"FRAME_API_PROCESS_EXITED_EARLY:{tail}")
                raise RuntimeError("FRAME_API_PROCESS_EXITED_EARLY")
            if self.is_ready(timeout_seconds=0.8):
                return
            time.sleep(0.2)
        raise RuntimeError("FRAME_API_START_TIMEOUT")

    def restart(self) -> None:
        self.stop()
        self.ensure_started()

    def stop(self) -> None:
        if self._proc is None:
            return
        proc = self._proc
        self._proc = None
        if proc.poll() is not None:
            return
        with contextlib.suppress(Exception):
            proc.terminate()
        try:
            proc.wait(timeout=1.5)
        except Exception:
            with contextlib.suppress(Exception):
                proc.kill()


def _resolve_repo_path(path_str: str) -> Path:
    path = Path(path_str)
    if not path.is_absolute():
        path = REPO_ROOT / path
    return path


def _load_json_if_exists(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"json root must be object: {path}")
    return data


def _latest_checkpoint_in_dir(dir_path: Path) -> Path:
    cands = sorted(dir_path.glob("ppo_tableturf_u*.pt"))
    if not cands:
        raise FileNotFoundError(f"no PPO checkpoint found in {dir_path}")
    return cands[-1]


def _resolve_checkpoint_from_entry(entry: Dict[str, Any]) -> Tuple[Path, str]:
    for key in ("checkpoint_file", "checkpoint", "pt"):
        value = str(entry.get(key, "")).strip()
        if value:
            resolved = _resolve_repo_path(value)
            if not resolved.exists():
                raise FileNotFoundError(f"checkpoint file not found: {resolved}")
            return resolved, key

    for key in ("training_summary", "eval_summary"):
        value = str(entry.get(key, "")).strip()
        if not value:
            continue
        summary_path = _resolve_repo_path(value)
        if not summary_path.exists():
            raise FileNotFoundError(f"summary file not found: {summary_path}")
        payload = json.loads(summary_path.read_text(encoding="utf-8"))
        save_dir = str(payload.get("save_dir", "")).strip()
        if save_dir:
            return _latest_checkpoint_in_dir(_resolve_repo_path(save_dir)), key
        return _latest_checkpoint_in_dir(summary_path.parent), key

    for key in ("checkpoint_dir", "save_dir", "dir"):
        value = str(entry.get(key, "")).strip()
        if value:
            resolved_dir = _resolve_repo_path(value)
            if not resolved_dir.exists():
                raise FileNotFoundError(f"checkpoint dir not found: {resolved_dir}")
            return _latest_checkpoint_in_dir(resolved_dir), key

    raise ValueError(f"ppo entry missing checkpoint path fields: {entry}")


def _resolve_policy_entry(raw: Any, source: str) -> ResolvedStrategy:
    if isinstance(raw, str):
        return ResolvedStrategy(mode="strategy_id", label=raw, strategy_id=raw, source=source)
    if not isinstance(raw, dict):
        raise ValueError(f"invalid policy entry from {source}: {raw!r}")

    mode = str(raw.get("mode", "") or raw.get("type", "") or "").strip().lower()
    strategy_id = str(raw.get("strategy_id", "")).strip()
    fallback_strategy_id = str(raw.get("fallback_strategy_id", "")).strip()
    fallback_label = str(raw.get("fallback_label", "")).strip()
    if strategy_id:
        label = str(raw.get("label", "") or strategy_id)
        return ResolvedStrategy(mode="strategy_id", label=label, strategy_id=strategy_id, source=source)

    if mode in {"ppo", "checkpoint", "nn", "strategic_ppo", "strategic", "strategic_checkpoint"} or any(
        str(raw.get(k, "")).strip() for k in ("checkpoint_file", "checkpoint", "pt", "training_summary", "eval_summary", "checkpoint_dir", "save_dir", "dir")
    ):
        try:
            checkpoint_file, path_kind = _resolve_checkpoint_from_entry(raw)
        except FileNotFoundError:
            if fallback_strategy_id:
                return ResolvedStrategy(
                    mode="strategy_id",
                    label=fallback_label or fallback_strategy_id,
                    strategy_id=fallback_strategy_id,
                    source=f"{source}:fallback_missing_checkpoint",
                )
            raise
        label = str(raw.get("label", "") or f"ppo:{checkpoint_file.name}")
        resolved_mode = "ppo_checkpoint"
        checkpoint_name = checkpoint_file.name.lower()
        if mode in {"strategic_ppo", "strategic", "strategic_checkpoint"} or checkpoint_name.startswith("strategic_ppo_"):
            resolved_mode = "strategic_ppo_checkpoint"
        return ResolvedStrategy(
            mode=resolved_mode,
            label=label,
            checkpoint_file=str(checkpoint_file),
            source=f"{source}:{path_kind}",
        )

    raise ValueError(f"unsupported policy entry from {source}: {raw}")


def _load_policy_config(config: ControllerConfig) -> Dict[str, Any]:
    path = str(config.policy_config_json or "").strip()
    if not path:
        return {}
    return _load_json_if_exists(_resolve_repo_path(path))


class ASpamWorker:
    def __init__(self, controller: SerialRemoteController, hold_ms: int, gap_ms: int):
        self._controller = controller
        self._hold_ms = hold_ms
        self._gap_ms = gap_ms
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None

    def start(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, name="a-spam-worker", daemon=True)
        self._thread.start()

    def _run(self) -> None:
        step = RemoteStep(bits=(1 << BIT_A), hold_ms=self._hold_ms, gap_ms=self._gap_ms)
        while not self._stop.is_set():
            self._controller.run_steps([step])

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)


def _press_button(steps: List[RemoteStep], bit_index: int, hold_ms: int = 50, gap_ms: int = 100) -> None:
    steps.append(RemoteStep(bits=(1 << bit_index), hold_ms=hold_ms, gap_ms=gap_ms))


def _move_axis(steps: List[RemoteStep], dx: int, dy: int, move_hold_ms: int = 50) -> None:
    if dx > 0:
        for _ in range(dx):
            _press_button(steps, BIT_DPAD_RIGHT, hold_ms=move_hold_ms, gap_ms=100)
    elif dx < 0:
        for _ in range(-dx):
            _press_button(steps, BIT_DPAD_LEFT, hold_ms=move_hold_ms, gap_ms=100)
    if dy > 0:
        for _ in range(dy):
            _press_button(steps, BIT_DPAD_DOWN, hold_ms=move_hold_ms, gap_ms=100)
    elif dy < 0:
        for _ in range(-dy):
            _press_button(steps, BIT_DPAD_UP, hold_ms=move_hold_ms, gap_ms=100)


def _alternating_axis_tokens(
    dx: int,
    dy: int,
    hold_ms: int = 1,
    gap_ms: int = 0,
) -> List[str]:
    tokens: List[str] = []
    _append_alternating_axis_csv(tokens, dx, "DRIGHT", "LRIGHT", "DLEFT", "LLEFT")
    _append_alternating_axis_csv(tokens, dy, "DDOWN", "LDOWN", "DUP", "LUP")
    if gap_ms > 0:
        out: List[str] = []
        for idx in range(0, len(tokens), 2):
            out.extend(tokens[idx : idx + 2])
            out.extend(["NOTHING", str(int(gap_ms))])
        tokens = out
    return tokens


def build_alternating_axis_sequence_csv(
    dx: int,
    dy: int,
    hold_ms: int = 50,
    gap_ms: int = 0,
) -> str:
    """Return a smart-sequence CSV using alternating DPad / left-stick moves.

    Kept as a spare movement encoder for long straight-line moves.
    It is not called by the current runtime flow.
    """
    return ",".join(_alternating_axis_tokens(dx=dx, dy=dy, hold_ms=hold_ms, gap_ms=gap_ms))


def _append_csv_command(parts: List[str], token: str, duration: int) -> None:
    parts.extend([str(token).upper(), str(max(0, int(duration)))])


def _append_csv_press(parts: List[str], token: str, duration: int = 1, nothing_after: Optional[int] = None) -> None:
    _append_csv_command(parts, token, duration)
    if nothing_after is not None:
        _append_csv_command(parts, "NOTHING", nothing_after)


@dataclass
class _AlternatingMoveState:
    next_kind: str = "dpad"


@dataclass
class ActionCommandPlan:
    menu_phase_csv: str = ""
    selection_phase_csv: str = ""
    map_phase_csv: str = ""

    def non_empty_phases(self) -> List[Tuple[str, str]]:
        phases: List[Tuple[str, str]] = []
        if self.menu_phase_csv:
            phases.append(("menu", self.menu_phase_csv))
        if self.selection_phase_csv:
            phases.append(("selection", self.selection_phase_csv))
        if self.map_phase_csv:
            phases.append(("map", self.map_phase_csv))
        return phases


def _phase_transition_delay_seconds(current_phase: str, next_phase: str) -> float:
    current_name = str(current_phase or "").strip().lower()
    next_name = str(next_phase or "").strip().lower()
    if current_name == "selection" and next_name == "map":
        return 0.3
    return 0.1


def _append_alternating_direction_csv(
    parts: List[str],
    state: "_AlternatingMoveState",
    *,
    dpad_token: str,
    stick_token: str,
) -> None:
    use_dpad = str(state.next_kind).lower() != "stick"
    _append_csv_command(parts, dpad_token if use_dpad else stick_token, 1)
    state.next_kind = "stick" if use_dpad else "dpad"


def _append_alternating_axis_csv(
    parts: List[str],
    state: "_AlternatingMoveState",
    delta: int,
    positive_dpad: str,
    positive_stick: str,
    negative_dpad: str,
    negative_stick: str,
) -> None:
    count = abs(int(delta))
    if count <= 0:
        return
    dpad_token, stick_token = (
        (positive_dpad, positive_stick) if int(delta) > 0 else (negative_dpad, negative_stick)
    )
    for idx in range(count):
        _append_alternating_direction_csv(
            parts,
            state,
            dpad_token=dpad_token,
            stick_token=stick_token,
        )


def _append_map_move_csv(parts: List[str], state: "_AlternatingMoveState", dx: int, dy: int) -> None:
    _append_alternating_axis_csv(
        parts,
        state,
        dx,
        positive_dpad="DRIGHT",
        positive_stick="LRIGHT",
        negative_dpad="DLEFT",
        negative_stick="LLEFT",
    )
    _append_alternating_axis_csv(
        parts,
        state,
        dy,
        positive_dpad="DDOWN",
        positive_stick="LDOWN",
        negative_dpad="DUP",
        negative_stick="LUP",
    )


def _append_card_selection_csv(parts: List[str], state: "_AlternatingMoveState", from_index: int, to_index: int) -> None:
    from_x, from_y = _card_grid_xy(from_index)
    to_x, to_y = _card_grid_xy(to_index)
    dx = int(to_x) - int(from_x)
    dy = int(to_y) - int(from_y)
    _append_alternating_axis_csv(
        parts,
        state,
        dx,
        positive_dpad="DRIGHT",
        positive_stick="LRIGHT",
        negative_dpad="DLEFT",
        negative_stick="LLEFT",
    )
    _append_alternating_axis_csv(
        parts,
        state,
        dy,
        positive_dpad="DDOWN",
        positive_stick="LDOWN",
        negative_dpad="DUP",
        negative_stick="LUP",
    )


def _card_grid_xy(index: int) -> Tuple[int, int]:
    idx = max(0, int(index))
    return (idx % 2, idx // 2)


def _move_card_selection(steps: List[RemoteStep], from_index: int, to_index: int) -> None:
    from_x, from_y = _card_grid_xy(from_index)
    to_x, to_y = _card_grid_xy(to_index)
    _move_axis(steps, dx=to_x - from_x, dy=to_y - from_y)


def compile_action_menu_selection_steps(
    action,
    obs: ObservedState,
    *,
    sp_attack_up_right: bool = False,
) -> List[RemoteStep]:
    hand = list(obs.hand_card_numbers or [])
    if not hand:
        raise ValueError("hand_card_numbers is empty")
    action_card = action.card_number if action.card_number is not None else hand[0]
    if action_card not in hand:
        raise ValueError(f"card {action.card_number} not in observed hand {hand}")

    steps: List[RemoteStep] = []

    if bool(getattr(action, "surrender", False)):
        _press_button(steps, BIT_PLUS, hold_ms=120, gap_ms=2000)
        _press_button(steps, BIT_DPAD_RIGHT, hold_ms=120, gap_ms=2000)
        _press_button(steps, BIT_A, hold_ms=120, gap_ms=200)
        return steps

    selected_hand_index = int(obs.selected_hand_index or 0)

    if bool(getattr(action, "pass_turn", False)):
        _press_button(steps, BIT_DPAD_DOWN)
        _press_button(steps, BIT_DPAD_DOWN)
        _press_button(steps, BIT_A, hold_ms=50, gap_ms=100)
        target_idx = hand.index(action_card)
        _move_card_selection(steps, from_index=0, to_index=target_idx)
        _press_button(steps, BIT_A, hold_ms=50, gap_ms=100)
        return steps

    if bool(getattr(action, "use_sp_attack", False)):
        sp_pool = _sp_pick_pool(obs)
        if action_card not in sp_pool:
            raise ValueError(f"card {action_card} not available in sp pick pool {sp_pool}")
        start_card = sp_pool[0]
        start_idx = hand.index(start_card)
        target_idx = hand.index(action_card)
        if sp_attack_up_right:
            _press_button(steps, BIT_DPAD_UP)
        else:
            _press_button(steps, BIT_DPAD_DOWN)
            _press_button(steps, BIT_DPAD_DOWN)
        _press_button(steps, BIT_DPAD_RIGHT)
        _press_button(steps, BIT_A, hold_ms=50, gap_ms=100)
        _move_card_selection(steps, from_index=start_idx, to_index=target_idx)
        _press_button(steps, BIT_A, hold_ms=50, gap_ms=100)
        return steps

    target_idx = hand.index(action_card)
    _move_card_selection(steps, from_index=selected_hand_index, to_index=target_idx)
    _press_button(steps, BIT_A, hold_ms=50, gap_ms=100)
    return steps


def compile_action_map_phase_csv(action, obs: ObservedState) -> str:
    if bool(getattr(action, "surrender", False)) or bool(getattr(action, "pass_turn", False)):
        return ""

    cursor_x, cursor_y = _initial_ui_anchor_for_map(obs)
    parts: List[str] = []
    cw_steps = int(action.rotation) % 4
    ccw_steps = (-int(action.rotation)) % 4
    if cw_steps <= ccw_steps:
        for _ in range(cw_steps):
            _append_csv_press(parts, "X", 1, 2)
    else:
        for _ in range(ccw_steps):
            _append_csv_press(parts, "Y", 1, 2)

    target_x, target_y = _action_target_ui_xy(action, obs)
    move_state = _AlternatingMoveState()
    _append_map_move_csv(parts, move_state, dx=int(target_x) - int(cursor_x), dy=int(target_y) - int(cursor_y))
    _append_csv_press(parts, "A", 1, 20)
    return _csv_from_parts(parts)


def compile_action_to_runtime_steps(action, obs: ObservedState) -> List[RemoteStep]:
    steps = compile_action_menu_selection_steps(action, obs)
    if bool(getattr(action, "surrender", False)) or bool(getattr(action, "pass_turn", False)):
        return steps

    cursor_x, cursor_y = _initial_ui_anchor_for_map(obs)
    cw_steps = int(action.rotation) % 4
    ccw_steps = (-int(action.rotation)) % 4
    if cw_steps <= ccw_steps:
        for _ in range(cw_steps):
            _press_button(steps, BIT_X)
    else:
        for _ in range(ccw_steps):
            _press_button(steps, BIT_Y)

    target_x, target_y = _action_target_ui_xy(action, obs)
    _move_axis(steps, dx=int(target_x) - int(cursor_x), dy=int(target_y) - int(cursor_y))
    _press_button(steps, BIT_A, hold_ms=50, gap_ms=100)
    return steps


def _build_state_from_observation(obs: ObservedState) -> GameState:
    game_map = load_map(obs.map_id)
    if obs.map_grid is not None:
        game_map.grid = _pad_grid_to_engine_dims(obs.map_id, obs.map_grid)
    p1_hand = [create_card_from_id(n) for n in obs.hand_card_numbers]
    p1 = PlayerState(deck_ids=[], draw_pile=[], hand=p1_hand, sp=obs.p1_sp)
    p2 = PlayerState(deck_ids=[], draw_pile=[], hand=[], sp=0)
    return GameState(map=game_map, players={"P1": p1, "P2": p2}, turn=obs.turn)


def _initial_engine_anchor_for_map(obs: ObservedState) -> Tuple[int, int]:
    game_map = load_map(obs.map_id)
    pad = MAP_PADDING
    if game_map.width > pad * 2 and game_map.height > pad * 2:
        logical_w = game_map.width - pad * 2
        logical_h = game_map.height - pad * 2
        view_x0 = pad
        view_y0 = pad
    else:
        logical_w = game_map.width
        logical_h = game_map.height
        view_x0 = 0
        view_y0 = 0
    anchor_local = _find_local_view_rightbottom_self_special_anchor(
        game_map=game_map,
        is_p1=True,
        view_x0=view_x0,
        view_y0=view_y0,
        view_w=logical_w,
        view_h=logical_h,
        flip_180=False,
    )
    return (int(anchor_local[0] + view_x0), int(anchor_local[1] + view_y0))


def _engine_xy_to_ui_xy(x: int, y: int, map_id: str, map_grid: Optional[List[List[int]]] = None) -> Tuple[int, int]:
    game_map = load_map(map_id)
    target_h = int(game_map.height)
    target_w = int(game_map.width)
    raw_h = len(map_grid) if map_grid else max(0, target_h - MAP_PADDING * 2)
    raw_w = len(map_grid[0]) if map_grid and map_grid[0] else max(0, target_w - MAP_PADDING * 2)
    if target_h == raw_h and target_w == raw_w:
        return (int(x), int(y))
    return (int(x) - MAP_PADDING, int(y) - MAP_PADDING)


def _initial_ui_anchor_for_map(obs: ObservedState) -> Tuple[int, int]:
    anchor_x, anchor_y = _initial_engine_anchor_for_map(obs)
    return _engine_xy_to_ui_xy(anchor_x, anchor_y, obs.map_id, obs.map_grid)


def _action_target_ui_xy(action: Any, obs: ObservedState) -> Tuple[int, int]:
    if action.x is None or action.y is None:
        raise ValueError("non-pass action requires x/y")
    return _engine_xy_to_ui_xy(int(action.x), int(action.y), obs.map_id, obs.map_grid)


def choose_action_from_strategy(obs: ObservedState, strategy_id: str):
    state = _build_state_from_observation(obs)
    return choose_action_from_strategy_id(state=state, player="P1", strategy_id=strategy_id)


def _base_load_model(checkpoint_file: str):
    try:
        import torch
    except Exception as exc:  # pragma: no cover
        raise RuntimeError(f"torch import failed: {exc}") from exc

    ckpt = str(checkpoint_file)
    if ckpt in _BASE_MODEL_CACHE:
        return _BASE_MODEL_CACHE[ckpt], torch

    from GST_RL.networks import PolicyValueNet

    model = PolicyValueNet(map_channels=6, scalar_dim=6, action_feature_dim=12)
    obj = torch.load(ckpt, map_location="cpu")
    state_dict = obj["model"] if isinstance(obj, dict) and "model" in obj else obj
    if not isinstance(state_dict, dict):
        raise RuntimeError("invalid checkpoint content")
    model.load_state_dict(state_dict)
    model.eval()
    _BASE_MODEL_CACHE[ckpt] = model
    return model, torch


def _base_encode_state(state: GameState, player: str) -> Tuple[np.ndarray, np.ndarray]:
    game_map = state.map
    obs = np.zeros((6, game_map.height, game_map.width), dtype=np.float32)
    is_p1_agent = player == "P1"
    for y in range(game_map.height):
        for x in range(game_map.width):
            mask = int(game_map.get_point(x, y))
            is_p1 = (mask & int(Map_PointBit.IsP1)) != 0
            is_p2 = (mask & int(Map_PointBit.IsP2)) != 0
            obs[0, y, x] = 1.0 if (mask & int(Map_PointBit.IsValid)) else 0.0
            obs[1, y, x] = 1.0 if (is_p1 if is_p1_agent else is_p2) else 0.0
            obs[2, y, x] = 1.0 if (is_p2 if is_p1_agent else is_p1) else 0.0
            obs[3, y, x] = 1.0 if (mask & int(Map_PointBit.IsSp)) else 0.0
            obs[4, y, x] = 1.0 if (mask & int(Map_PointBit.IsSupplySp)) else 0.0
            obs[5, y, x] = 1.0 if (is_p1 and is_p2) else 0.0

    p1_score, p2_score = _strategic_compute_scores(state)
    own = state.players[player]
    opp = state.players["P2" if player == "P1" else "P1"]
    own_score = p1_score if is_p1_agent else p2_score
    opp_score = p2_score if is_p1_agent else p1_score
    scalar = np.array(
        [
            state.turn / max(1, state.max_turns),
            own.sp / 20.0,
            opp.sp / 20.0,
            (own_score - opp_score) / 100.0,
            len(own.draw_pile) / 15.0,
            len(opp.draw_pile) / 15.0,
        ],
        dtype=np.float32,
    )
    return obs, scalar


def _base_encode_actions(state: GameState, player: str, legal_action_dicts: List[Dict[str, Any]]) -> np.ndarray:
    ps = state.players[player]
    hand = ps.hand
    hand_index = {card.Number: idx for idx, card in enumerate(hand)}
    w = max(1, state.map.width - 1)
    h = max(1, state.map.height - 1)
    feats: List[np.ndarray] = []
    for action in legal_action_dicts:
        card_no = int(action.get("card_number"))
        card = next((c for c in hand if c.Number == card_no), None)
        if card is None:
            raise RuntimeError(f"card #{card_no} not in hand")
        rotation = int(action.get("rotation", 0))
        x = action.get("x")
        y = action.get("y")
        cell_count, sp_count = nn_loader._card_cell_stats(card, rotation)
        feats.append(
            np.array(
                [
                    hand_index[card.Number] / 3.0,
                    1.0 if bool(action.get("pass_turn", False)) else 0.0,
                    1.0 if bool(action.get("use_sp_attack", False)) else 0.0,
                    rotation / 3.0,
                    (float(x) / w) if x is not None else 0.0,
                    (float(y) / h) if y is not None else 0.0,
                    card.CardPoint / 20.0,
                    card.SpecialCost / 10.0,
                    cell_count / 64.0,
                    sp_count / 64.0,
                    ps.sp / 20.0,
                    state.turn / max(1, state.max_turns),
                ],
                dtype=np.float32,
            )
        )
    if not feats:
        return np.zeros((1, 12), dtype=np.float32)
    return np.stack(feats, axis=0)


def _choose_action_from_base_checkpoint(state: GameState, checkpoint_file: str, player: str = "P1") -> Dict[str, Any]:
    legal = legal_actions(state, player)
    if not legal:
        raise RuntimeError("legal_actions is empty")
    legal_action_dicts = [{
        "player": a.player,
        "card_number": a.card_number,
        "surrender": a.surrender,
        "pass_turn": a.pass_turn,
        "use_sp_attack": a.use_sp_attack,
        "rotation": a.rotation,
        "x": a.x,
        "y": a.y,
    } for a in legal]
    model, torch = _base_load_model(checkpoint_file)
    map_obs, scalar_obs = _base_encode_state(state, player)
    action_feats = _base_encode_actions(state, player, legal_action_dicts)
    with torch.no_grad():
        logits, _ = model.forward_single(
            torch.as_tensor(map_obs, dtype=torch.float32),
            torch.as_tensor(scalar_obs, dtype=torch.float32),
            torch.as_tensor(action_feats, dtype=torch.float32),
        )
        idx = int(torch.argmax(logits).item())
    idx = max(0, min(idx, len(legal_action_dicts) - 1))
    return dict(legal_action_dicts[idx])


def _strategic_load_model(checkpoint_file: str):
    try:
        import torch
    except Exception as exc:  # pragma: no cover
        raise RuntimeError(f"torch import failed: {exc}") from exc

    ckpt = str(checkpoint_file)
    if ckpt in _STRATEGIC_MODEL_CACHE:
        return _STRATEGIC_MODEL_CACHE[ckpt], torch

    from GST_RL.strategic_networks import StrategicPolicyValueNet

    model = StrategicPolicyValueNet(map_channels=6, scalar_dim=14, action_feature_dim=12)
    obj = torch.load(ckpt, map_location="cpu")
    state_dict = obj["model"] if isinstance(obj, dict) and "model" in obj else obj
    if not isinstance(state_dict, dict):
        raise RuntimeError("invalid strategic checkpoint content")
    model.load_state_dict(state_dict)
    model.eval()
    _STRATEGIC_MODEL_CACHE[ckpt] = model
    return model, torch


def _strategic_is_valid(mask: int) -> bool:
    return (mask & int(Map_PointBit.IsValid)) != 0


def _strategic_has_owner(mask: int, player: str) -> bool:
    bit = Map_PointBit.IsP1 if player == "P1" else Map_PointBit.IsP2
    return (mask & int(bit)) != 0


def _strategic_is_empty(mask: int) -> bool:
    if not _strategic_is_valid(mask):
        return False
    return not _strategic_has_owner(mask, "P1") and not _strategic_has_owner(mask, "P2")


def _strategic_is_sp(mask: int) -> bool:
    return (mask & int(Map_PointBit.IsSp)) != 0


def _iter_state_cells(state: GameState) -> Iterable[Tuple[int, int, int]]:
    game_map = state.map
    for y in range(game_map.height):
        for x in range(game_map.width):
            yield x, y, int(game_map.get_point(x, y))


def _strategic_neighbors4(x: int, y: int) -> Iterable[Tuple[int, int]]:
    return ((x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1))


def _strategic_compute_scores(state: GameState) -> Tuple[int, int]:
    p1 = 0
    p2 = 0
    for _, _, mask in _iter_state_cells(state):
        if not _strategic_is_valid(mask):
            continue
        is_p1 = _strategic_has_owner(mask, "P1")
        is_p2 = _strategic_has_owner(mask, "P2")
        if is_p1 and not is_p2:
            p1 += 1
        elif is_p2 and not is_p1:
            p2 += 1
    return p1, p2


def _strategic_reachable_empty_stats(state: GameState, player: str) -> Tuple[int, int, int]:
    reachable_total = 0
    largest_reachable = 0
    largest_locked = 0
    visited: Set[Tuple[int, int]] = set()
    game_map = state.map

    for y in range(game_map.height):
        for x in range(game_map.width):
            if (x, y) in visited:
                continue
            mask = int(game_map.get_point(x, y))
            if not _strategic_is_empty(mask):
                continue
            queue = deque([(x, y)])
            visited.add((x, y))
            size = 0
            touches_player = False
            while queue:
                cx, cy = queue.popleft()
                size += 1
                for nx, ny in _strategic_neighbors4(cx, cy):
                    if nx < 0 or ny < 0 or nx >= game_map.width or ny >= game_map.height:
                        continue
                    nm = int(game_map.get_point(nx, ny))
                    if _strategic_has_owner(nm, player):
                        touches_player = True
                    if (nx, ny) in visited or not _strategic_is_empty(nm):
                        continue
                    visited.add((nx, ny))
                    queue.append((nx, ny))
            if touches_player:
                reachable_total += size
                largest_reachable = max(largest_reachable, size)
            else:
                largest_locked = max(largest_locked, size)
    return reachable_total, largest_reachable, largest_locked


def _strategic_frontier_count(state: GameState, player: str) -> int:
    other = "P2" if player == "P1" else "P1"
    count = 0
    for x, y, mask in _iter_state_cells(state):
        if not _strategic_has_owner(mask, player):
            continue
        for nx, ny in _strategic_neighbors4(x, y):
            if nx < 0 or ny < 0 or nx >= state.map.width or ny >= state.map.height:
                continue
            nm = int(state.map.get_point(nx, ny))
            if _strategic_is_empty(nm) or _strategic_has_owner(nm, other):
                count += 1
                break
    return count


def _strategic_is_frontier_cell(state: GameState, x: int, y: int, player: str) -> bool:
    other = "P2" if player == "P1" else "P1"
    for nx, ny in _strategic_neighbors4(x, y):
        if nx < 0 or ny < 0 or nx >= state.map.width or ny >= state.map.height:
            continue
        nm = int(state.map.get_point(nx, ny))
        if _strategic_is_empty(nm) or _strategic_has_owner(nm, other):
            return True
    return False


def _strategic_sp_breach_risk(state: GameState, attacker: str, defender: str) -> float:
    attacker_sp = float(state.players[attacker].sp)
    if attacker_sp < 3:
        return 0.0
    attack_scale = 2.0 if attacker_sp >= 6 else 1.0
    risk = 0.0
    for x, y, mask in _iter_state_cells(state):
        if not _strategic_has_owner(mask, defender):
            continue
        if not _strategic_is_frontier_cell(state, x, y, defender):
            continue
        adjacent_enemy_sp = False
        for nx, ny in _strategic_neighbors4(x, y):
            if nx < 0 or ny < 0 or nx >= state.map.width or ny >= state.map.height:
                continue
            nm = int(state.map.get_point(nx, ny))
            if _strategic_has_owner(nm, attacker) and _strategic_is_sp(nm):
                adjacent_enemy_sp = True
                break
        if not adjacent_enemy_sp:
            continue
        support = 0
        for nx, ny in _strategic_neighbors4(x, y):
            if nx < 0 or ny < 0 or nx >= state.map.width or ny >= state.map.height:
                continue
            nm = int(state.map.get_point(nx, ny))
            if _strategic_has_owner(nm, defender):
                support += 1
        if support <= 1:
            risk += 1.5 * attack_scale
        elif support == 2:
            risk += 0.8 * attack_scale
        else:
            risk += 0.25 * attack_scale
    return risk


def _strategic_compute_metrics(state: GameState) -> Dict[str, float]:
    valid_cells = 0
    empty_cells = 0
    for _, _, mask in _iter_state_cells(state):
        if not _strategic_is_valid(mask):
            continue
        valid_cells += 1
        if _strategic_is_empty(mask):
            empty_cells += 1
    p1_reach, p1_largest, p1_locked = _strategic_reachable_empty_stats(state, "P1")
    p2_reach, p2_largest, p2_locked = _strategic_reachable_empty_stats(state, "P2")
    p1_score, p2_score = _strategic_compute_scores(state)
    return {
        "valid_cells": float(valid_cells),
        "empty_cells": float(empty_cells),
        "turn_ratio": state.turn / max(1, state.max_turns),
        "p1_sp": float(state.players["P1"].sp),
        "p2_sp": float(state.players["P2"].sp),
        "p1_score": float(p1_score),
        "p2_score": float(p2_score),
        "score_diff": float(p1_score - p2_score),
        "p1_reachable_empty": float(p1_reach),
        "p2_reachable_empty": float(p2_reach),
        "p1_largest_reachable": float(p1_largest),
        "p2_largest_reachable": float(p2_largest),
        "p1_largest_locked": float(p1_locked),
        "p2_largest_locked": float(p2_locked),
        "p1_frontier": float(_strategic_frontier_count(state, "P1")),
        "p2_frontier": float(_strategic_frontier_count(state, "P2")),
        "enemy_breach_risk": float(_strategic_sp_breach_risk(state, attacker="P2", defender="P1")),
        "own_breach_chance": float(_strategic_sp_breach_risk(state, attacker="P1", defender="P2")),
        "p1_draw_ratio": len(state.players["P1"].draw_pile) / 15.0,
        "p2_draw_ratio": len(state.players["P2"].draw_pile) / 15.0,
    }


def _strategic_encode_state(state: GameState, player: str) -> Tuple[np.ndarray, np.ndarray]:
    game_map = state.map
    obs = np.zeros((6, game_map.height, game_map.width), dtype=np.float32)
    metrics = _strategic_compute_metrics(state)
    is_p1_agent = player == "P1"
    for y in range(game_map.height):
        for x in range(game_map.width):
            mask = int(game_map.get_point(x, y))
            is_p1 = (mask & int(Map_PointBit.IsP1)) != 0
            is_p2 = (mask & int(Map_PointBit.IsP2)) != 0
            obs[0, y, x] = 1.0 if _strategic_is_valid(mask) else 0.0
            obs[1, y, x] = 1.0 if (is_p1 if is_p1_agent else is_p2) else 0.0
            obs[2, y, x] = 1.0 if (is_p2 if is_p1_agent else is_p1) else 0.0
            obs[3, y, x] = 1.0 if (mask & int(Map_PointBit.IsSp)) else 0.0
            obs[4, y, x] = 1.0 if (mask & int(Map_PointBit.IsSupplySp)) else 0.0
            obs[5, y, x] = 1.0 if (is_p1 and is_p2) else 0.0

    valid = max(1.0, metrics["valid_cells"])
    scalar = np.array(
        [
            metrics["turn_ratio"],
            metrics["p1_sp"] / 20.0,
            metrics["p2_sp"] / 20.0,
            metrics["score_diff"] / 100.0,
            metrics["p1_draw_ratio"],
            metrics["p2_draw_ratio"],
            metrics["p1_reachable_empty"] / valid,
            metrics["p2_reachable_empty"] / valid,
            metrics["p1_largest_reachable"] / valid,
            metrics["p2_largest_reachable"] / valid,
            metrics["enemy_breach_risk"] / valid,
            metrics["own_breach_chance"] / valid,
            metrics["p1_largest_locked"] / valid,
            metrics["p2_largest_locked"] / valid,
        ],
        dtype=np.float32,
    )
    return obs, scalar


def _strategic_encode_actions(state: GameState, player: str, legal_action_dicts: List[Dict[str, Any]]) -> np.ndarray:
    ps = state.players[player]
    hand = ps.hand
    hand_index = {card.Number: idx for idx, card in enumerate(hand)}
    feats: List[np.ndarray] = []
    for action in legal_action_dicts:
        card_no = int(action.get("card_number"))
        card = next((c for c in hand if c.Number == card_no), None)
        if card is None:
            raise RuntimeError(f"card #{card_no} not in hand")
        rotation = int(action.get("rotation", 0))
        x = action.get("x")
        y = action.get("y")
        cell_count, sp_count = nn_loader._card_cell_stats(card, rotation)
        feats.append(
            np.array(
                [
                    hand_index[card.Number] / 3.0,
                    1.0 if bool(action.get("pass_turn", False)) else 0.0,
                    1.0 if bool(action.get("use_sp_attack", False)) else 0.0,
                    rotation / 3.0,
                    (float(x) / max(1, state.map.width - 1)) if x is not None else 0.0,
                    (float(y) / max(1, state.map.height - 1)) if y is not None else 0.0,
                    card.CardPoint / 20.0,
                    card.SpecialCost / 10.0,
                    cell_count / 64.0,
                    sp_count / 64.0,
                    ps.sp / 20.0,
                    state.turn / max(1, state.max_turns),
                ],
                dtype=np.float32,
            )
        )
    if not feats:
        return np.zeros((1, 12), dtype=np.float32)
    return np.stack(feats, axis=0)


def _choose_action_from_strategic_checkpoint(state: GameState, checkpoint_file: str, player: str = "P1") -> Dict[str, Any]:
    legal = legal_actions(state, player)
    if not legal:
        raise RuntimeError("legal_actions is empty")
    legal_action_dicts = [{
        "player": a.player,
        "card_number": a.card_number,
        "surrender": a.surrender,
        "pass_turn": a.pass_turn,
        "use_sp_attack": a.use_sp_attack,
        "rotation": a.rotation,
        "x": a.x,
        "y": a.y,
    } for a in legal]
    model, torch = _strategic_load_model(checkpoint_file)
    map_obs, scalar_obs = _strategic_encode_state(state, player)
    action_feats = _strategic_encode_actions(state, player, legal_action_dicts)
    with torch.no_grad():
        logits, _ = model.forward_single(
            torch.as_tensor(map_obs, dtype=torch.float32),
            torch.as_tensor(scalar_obs, dtype=torch.float32),
            torch.as_tensor(action_feats, dtype=torch.float32),
        )
        idx = int(torch.argmax(logits).item())
    idx = max(0, min(idx, len(legal_action_dicts) - 1))
    return dict(legal_action_dicts[idx])


def choose_action_from_resolved_strategy(obs: ObservedState, resolved: ResolvedStrategy):
    state = _build_state_from_observation(obs)
    if resolved.mode == "strategy_id":
        return choose_action_from_strategy_id(state=state, player="P1", strategy_id=resolved.strategy_id)
    if resolved.mode == "ppo_checkpoint":
        payload = _choose_action_from_base_checkpoint(state, resolved.checkpoint_file, player="P1")
        for action in legal_actions(state, "P1"):
            if (
                action.card_number == payload.get("card_number")
                and action.pass_turn == bool(payload.get("pass_turn", False))
                and action.use_sp_attack == bool(payload.get("use_sp_attack", False))
                and action.rotation == int(payload.get("rotation", 0))
                and action.x == payload.get("x")
                and action.y == payload.get("y")
            ):
                return action
        raise RuntimeError(f"ppo strategy returned non-legal action: {resolved.checkpoint_file}")
    if resolved.mode == "strategic_ppo_checkpoint":
        payload = _choose_action_from_strategic_checkpoint(state, resolved.checkpoint_file, player="P1")
        for action in legal_actions(state, "P1"):
            if (
                action.card_number == payload.get("card_number")
                and action.pass_turn == bool(payload.get("pass_turn", False))
                and action.use_sp_attack == bool(payload.get("use_sp_attack", False))
                and action.rotation == int(payload.get("rotation", 0))
                and action.x == payload.get("x")
                and action.y == payload.get("y")
            ):
                return action
        raise RuntimeError(f"strategic ppo returned non-legal action: {resolved.checkpoint_file}")
    raise ValueError(f"unsupported resolved strategy mode: {resolved.mode}")


def resolve_strategy_id(
    config: ControllerConfig,
    map_id: str,
    map_name: str = "",
) -> str:
    map_id_use = str(map_id or "")
    map_name_use = str(map_name or "")
    if map_id_use and map_id_use in config.strategy_id_by_map:
        return str(config.strategy_id_by_map[map_id_use])
    if map_name_use and map_name_use in config.strategy_id_by_map_name:
        return str(config.strategy_id_by_map_name[map_name_use])
    return str(config.strategy_id)


def resolve_strategy(
    config: ControllerConfig,
    map_id: str,
    map_name: str = "",
) -> ResolvedStrategy:
    policy = _load_policy_config(config)
    maps = policy.get("maps", {}) if isinstance(policy.get("maps", {}), dict) else {}
    defaults = policy.get("default")
    source_base = str(config.policy_config_json or "")

    def _embedded_fallback(source_suffix: str) -> ResolvedStrategy:
        strategy_id = resolve_strategy_id(config, map_id=map_id, map_name=map_name) or "default:aggressive:high"
        return ResolvedStrategy(
            mode="strategy_id",
            label=strategy_id,
            strategy_id=strategy_id,
            source=f"{source_base}:{source_suffix}:embedded_fallback",
        )

    for key in (str(map_id or ""), str(map_name or "")):
        if key and key in maps:
            try:
                return _resolve_policy_entry(maps[key], f"{source_base}:{key}")
            except (FileNotFoundError, ValueError):
                if defaults is not None:
                    try:
                        return _resolve_policy_entry(defaults, f"{source_base}:default_after_{key}_fallback")
                    except (FileNotFoundError, ValueError):
                        return _embedded_fallback(f"{key}_fallback")
                return _embedded_fallback(f"{key}_fallback")

    if defaults is not None:
        try:
            return _resolve_policy_entry(defaults, f"{source_base}:default")
        except (FileNotFoundError, ValueError):
            return _embedded_fallback("default_fallback")

    strategy_id = resolve_strategy_id(config, map_id=map_id, map_name=map_name)
    return ResolvedStrategy(mode="strategy_id", label=strategy_id, strategy_id=strategy_id, source="embedded_config")


def _sp_pick_pool(obs: ObservedState) -> List[int]:
    state = _build_state_from_observation(obs)
    actions = legal_actions(state, "P1")
    seen: set[int] = set()
    ordered: List[int] = []
    for hand_card in obs.hand_card_numbers:
        for action in actions:
            card_number = action.card_number
            if (
                card_number == hand_card
                and action.use_sp_attack
                and card_number not in seen
            ):
                ordered.append(int(card_number))
                seen.add(int(card_number))
                break
    return ordered


def _csv_from_parts(parts: List[str]) -> str:
    return ",".join(parts) if parts else ""


def compile_action_command_plan(action, obs: ObservedState) -> ActionCommandPlan:
    if bool(getattr(action, "surrender", False)):
        return ActionCommandPlan(menu_phase_csv="PLUS,1,NOTHING,20,DRIGHT,1,NOTHING,20,A,1,NOTHING,20")

    hand = obs.hand_card_numbers
    if not hand:
        raise ValueError("hand_card_numbers is empty")
    action_card = action.card_number if action.card_number is not None else hand[0]
    if action_card not in hand:
        raise ValueError(f"card {action.card_number} not in observed hand {hand}")

    move_state = _AlternatingMoveState()
    menu_parts: List[str] = []
    selection_parts: List[str] = []
    map_parts: List[str] = []
    if action.pass_turn:
        target_idx = hand.index(action_card)
        _append_alternating_direction_csv(menu_parts, move_state, dpad_token="DUP", stick_token="LUP")
        _append_csv_press(menu_parts, "A", 1, 2)
        _append_card_selection_csv(selection_parts, move_state, from_index=0, to_index=target_idx)
        _append_csv_press(selection_parts, "A", 1, 20)
        return ActionCommandPlan(
            menu_phase_csv=_csv_from_parts(menu_parts),
            selection_phase_csv=_csv_from_parts(selection_parts),
            map_phase_csv="",
        )

    if action.use_sp_attack:
        sp_pool = _sp_pick_pool(obs)
        if action_card not in sp_pool:
            raise ValueError(f"card {action_card} not available in sp pick pool {sp_pool}")
        start_card = sp_pool[0]
        start_idx = hand.index(start_card)
        target_idx = hand.index(action_card)
        _append_alternating_direction_csv(menu_parts, move_state, dpad_token="DUP", stick_token="LUP")
        _append_alternating_direction_csv(menu_parts, move_state, dpad_token="DRIGHT", stick_token="LRIGHT")
        _append_csv_press(menu_parts, "A", 1, 2)
        _append_card_selection_csv(selection_parts, move_state, from_index=start_idx, to_index=target_idx)
        _append_csv_press(selection_parts, "A", 1, 1)
    else:
        target_idx = hand.index(action_card)
        _append_card_selection_csv(selection_parts, move_state, from_index=int(obs.selected_hand_index or 0), to_index=target_idx)
        _append_csv_press(selection_parts, "A", 1, 2)

    cursor_x, cursor_y = _initial_ui_anchor_for_map(obs)

    cw_steps = int(action.rotation) % 4
    ccw_steps = (-int(action.rotation)) % 4
    if cw_steps <= ccw_steps:
        for _ in range(cw_steps):
            _append_csv_press(map_parts, "X", 1, 2)
    else:
        for _ in range(ccw_steps):
            _append_csv_press(map_parts, "Y", 1, 2)

    target_x, target_y = _action_target_ui_xy(action, obs)
    _append_map_move_csv(map_parts, move_state, dx=int(target_x) - int(cursor_x), dy=int(target_y) - int(cursor_y))
    _append_csv_press(map_parts, "A", 1, 20)
    return ActionCommandPlan(
        menu_phase_csv=_csv_from_parts(menu_parts),
        selection_phase_csv=_csv_from_parts(selection_parts),
        map_phase_csv=_csv_from_parts(map_parts),
    )


def compile_action_with_defaults(action, obs: ObservedState) -> str:
    plan = compile_action_command_plan(action, obs)
    merged: List[str] = []
    for _phase_name, command_csv in plan.non_empty_phases():
        if command_csv:
            merged.extend([part for part in command_csv.split(",") if part.strip()])
    return ",".join(merged)


class TerminalDebugUI:
    def __init__(self, runtime: "AutoControllerRuntime"):
        self._runtime = runtime
        self._thread: Optional[threading.Thread] = None
        self._stop = threading.Event()
        self._interactive = False
        self._first_frame = True
        self._last_render_body = ""
        self._started = False

    def start(self) -> None:
        if self._started:
            return
        if not self._runtime.config.debug_ui_enabled:
            return
        self._interactive = bool(sys.stdin.isatty() and sys.stdout.isatty())
        if not self._interactive:
            return
        self._started = True
        self._stop.clear()
        self._first_frame = True
        self._last_render_body = ""
        self._thread = threading.Thread(target=self._run, name="terminal-debug-ui", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        if not self._started:
            return
        self._started = False
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=1.5)
        self._thread = None
        was_interactive = self._interactive
        if was_interactive:
            with contextlib.suppress(Exception):
                self._write_output("\r\033[H\033[J")
                self._flush_output()
        self._interactive = False

    def _poll_key(self) -> None:
        if not self._interactive:
            return
        if not msvcrt.kbhit():
            return
        chars = ""
        while msvcrt.kbhit():
            ch = msvcrt.getwch()
            if ch in ("\x00", "\xe0"):
                nxt = msvcrt.getwch()
                arrow_map = {
                    "H": "\x1b[A",
                    "P": "\x1b[B",
                    "M": "\x1b[C",
                    "K": "\x1b[D",
                }
                chars += arrow_map.get(nxt, "")
                continue
            chars += ch
        idx = 0
        while idx < len(chars):
            if chars[idx : idx + 3] == "\x1b[A":
                self._runtime.send_manual_controller_input("DUP")
                idx += 3
                continue
            if chars[idx : idx + 3] == "\x1b[B":
                self._runtime.send_manual_controller_input("DDOWN")
                idx += 3
                continue
            if chars[idx : idx + 3] == "\x1b[C":
                self._runtime.send_manual_controller_input("DRIGHT")
                idx += 3
                continue
            if chars[idx : idx + 3] == "\x1b[D":
                self._runtime.send_manual_controller_input("DLEFT")
                idx += 3
                continue
            ch = chars[idx]
            idx += 1
            if ch in ("\r", "\n"):
                self._runtime.send_manual_controller_input("HOME")
                continue
            if ch == "=":
                self._runtime.adjust_turn_index(+1)
                continue
            if ch == "-":
                self._runtime.adjust_turn_index(-1)
                continue
            key = ch.lower()
            if key == "p":
                self._runtime.toggle_pause()
            elif key == "r":
                self._runtime.restart_battle_waiting()
            elif key == "t":
                self._runtime.surrender_and_restart_waiting()
            elif key == "q":
                self._runtime.request_stop("user_requested_quit")
            elif key == "z":
                self._runtime.send_manual_controller_input("A")
            elif key == "x":
                self._runtime.send_manual_controller_input("B")
            elif key == "a":
                self._runtime.send_manual_controller_input("Y")
            elif key == "s":
                self._runtime.send_manual_controller_input("X")
            elif key == "c":
                self._runtime.send_manual_controller_input("L")
            elif key == "d":
                self._runtime.send_manual_controller_input("PLUS")

    def _render(self) -> None:
        state = self._runtime.debug_snapshot()
        term_size = shutil.get_terminal_size((120, 32))
        width = max(40, int(term_size.columns))
        height = max(12, int(term_size.lines))
        if self._runtime.config.continuous_run:
            wins_text = str(state["wins"])
        else:
            wins_text = f"{state['wins']}/{max(1, int(self._runtime.config.target_win_count))}"

        def _fit(text: str) -> str:
            s = str(text)
            if len(s) <= width:
                return s
            if width <= 3:
                return s[:width]
            return s[: width - 3] + "..."

        lines = [
            _fit("keys: p=pause/resume  r=restart-battle  t=surrender+restart  q=quit  -=turn-1  ==turn+1  arrows=dpad  enter=HOME  z=A  x=B  a=Y  s=X  c=L  d=+"),
            "",
            _fit(f"status: {state['status']}"),
            _fit(f"phase: {state['phase']}"),
            _fit(f"turn: {state['turn']}"),
            _fit(f"map: {state['map_id']}"),
            _fit(f"wins: {wins_text}"),
            _fit(f"battles: {state['battles']}"),
            _fit(f"pending result: {state['pending_result_check']}"),
            _fit(f"playable: {state['playable']}"),
            _fit(f"hand: {state['hand']}"),
            _fit(f"sp: {state['p1_sp']}"),
            _fit(f"action: {state['last_action']}"),
            _fit(f"strategy: {state['strategy_id']}"),
            _fit(f"strategy source: {state['strategy_source']}"),
            _fit(f"serial: {state['serial_port']}"),
            _fit(f"frame: {state['last_frame_path']}"),
            _fit(f"analysis: {state['last_analysis_path']}"),
            _fit(f"last error: {state['last_error']}"),
            _fit(f"updated: {state['updated_at']}"),
            "",
            "recent events:",
        ]
        reserved = len(lines)
        event_slots = max(1, height - reserved - 1)
        for item in list(state["events"])[-event_slots:]:
            lines.append(_fit(f"  {item}"))
        lines = lines[: max(1, height - 1)]
        body = "\n".join(lines)
        if not self._first_frame and body == self._last_render_body:
            return
        if self._first_frame:
            self._write_output("\r\033[2J\033[H" + body)
            self._first_frame = False
        else:
            # Write the new frame before erasing its tail. CMD otherwise exposes
            # a blank frame between ESC[J and the following body write.
            self._write_output("\r\033[H" + body + "\033[J")
        self._last_render_body = body
        self._flush_output()

    def _write_output(self, text: str) -> None:
        sys.stdout.write(text)

    def _flush_output(self) -> None:
        sys.stdout.flush()

    def _run(self) -> None:
        last_render = 0.0
        while not self._stop.is_set():
            self._poll_key()
            now = time.monotonic()
            if now - last_render >= 0.2:
                self._render()
                last_render = now
            time.sleep(0.05)


class AutoControllerRuntime:
    def __init__(self, config: ControllerConfig):
        self.config = config
        self.serial_port = _resolve_serial_port(config.serial_port, config.pick_serial)
        self.controller = SerialRemoteController(port=self.serial_port)
        self.vision = _FrameVisionPipeline(
            config,
            on_capture_recovery_pause=self._suspend_timeout_accumulation,
            on_capture_recovery_event=self._push_vision_error_event,
            on_capture_interactive_prompt=self._set_capture_interactive_prompt,
        )
        log_path = _resolve_debug_log_path(config.log_file)
        self._logger = RuntimeLogWriter(log_path)
        if str(config.stats_mode or "map").strip().lower() == "aggregate":
            aggregate_path = Path(config.aggregate_stats_file or "autocontroller_rebuild_for_RL/clone_jelly_state.json")
            if not aggregate_path.is_absolute():
                aggregate_path = REPO_ROOT / aggregate_path
            self._global_stats = AggregateStatsWriter(aggregate_path, config.aggregate_stats_name)
        else:
            stats_path = Path(config.global_stats_file)
            if not stats_path.is_absolute():
                stats_path = REPO_ROOT / stats_path
            self._global_stats = GlobalMapStatsWriter(stats_path)
        if bool(config.enable_battle_replay):
            replay_dir = Path(config.battle_replay_dir)
            if not replay_dir.is_absolute():
                replay_dir = REPO_ROOT / replay_dir
            self._battle_replay = BattleReplayWriter(replay_dir)
        else:
            self._battle_replay = NoOpBattleReplayWriter()
        self.turn_index = 1
        self.win_count = 0
        self.battle_count = 0
        self._pending_result_check = False
        self._battle_started = False
        self._current_battle_map_name = ""
        self._current_battle_stats_recorded = False
        self._turn_progress_failure_streak = 0
        self._last_progress_ts = time.monotonic()
        self._playable_seen_since: Optional[float] = None
        self._non_playable_seen_since: Optional[float] = None
        self._pause_started_ts: Optional[float] = None
        self._wait_a_enabled = True
        self._wait_silent_logged = False
        self._force_wait_a_reactivation = False
        self._closed = False
        self._paused = threading.Event()
        self._stop_requested = threading.Event()
        self._restart_battle_requested = threading.Event()
        self._manual_surrender_requested = threading.Event()
        self._manual_surrender_trigger_after_ts: Optional[float] = None
        self._manual_surrender_wait_logged = False
        self._status_lock = threading.Lock()
        self._status: Dict[str, Any] = {
            "status": "initializing",
            "phase": "boot",
            "turn": 0,
            "map_id": "",
            "wins": 0,
            "battles": 0,
            "pending_result_check": False,
            "playable": False,
            "hand": "",
            "p1_sp": 0,
            "last_action": "",
            "strategy_id": self.config.strategy_id,
            "strategy_source": "",
            "serial_port": self.serial_port,
            "last_frame_path": "",
            "last_analysis_path": "",
            "last_error": "",
            "updated_at": "",
            "events": [],
        }
        self._debug_ui = TerminalDebugUI(self)
        self._debug_ui.start()
        self._logger.write("自动控制器启动，已初始化串口、采集与调试界面。", tag="SYSTEM")
        self._push_event("runtime_ready", tag="SYSTEM")
        self._set_status(status="running", phase="idle")

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._debug_ui.stop()
        self.vision.close()
        self.controller.close()

    def _timestamp(self) -> str:
        return dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    def _set_status(self, **kwargs: Any) -> None:
        with self._status_lock:
            self._status.update(kwargs)
            self._status["updated_at"] = self._timestamp()

    def _set_capture_interactive_prompt(self, active: bool) -> None:
        if bool(active):
            self._debug_ui.stop()
        else:
            self._debug_ui.start()

    def _push_event(self, message: str, tag: str = "SYSTEM") -> None:
        with self._status_lock:
            events = list(self._status.get("events", []))
            events.append(f"{self._timestamp()} {message}")
            self._status["events"] = events[-48:]
            self._status["updated_at"] = self._timestamp()
        self._logger.write(message, tag=tag)

    def _push_vision_error_event(self, message: str) -> None:
        self._push_event(message, tag="VISION_ERROR")

    def _mark_progress(self, reason: str) -> None:
        self._last_progress_ts = time.monotonic()
        self._logger.write(f"进度推进：{reason}", tag="GAMEROUND")

    def _reset_turn_progress_failure_streak(self, reason: str = "") -> None:
        previous = int(getattr(self, "_turn_progress_failure_streak", 0))
        self._turn_progress_failure_streak = 0
        if previous > 0 and reason:
            self._logger.write(f"回合推进失败累计已清零（此前连续失败 {previous} 次）：{reason}", tag="GAMEROUND")

    def _record_turn_progress_failure(self, reason: str) -> bool:
        self._turn_progress_failure_streak = int(getattr(self, "_turn_progress_failure_streak", 0)) + 1
        count = self._turn_progress_failure_streak
        self._push_event(f"回合推进失败累计：{count}/3。原因：{reason}", tag="TIMEOUT")
        self._logger.write(f"回合推进失败累计 {count}/3。原因：{reason}", tag="TIMEOUT")
        return count >= 3

    def _suspend_timeout_accumulation(self, seconds: float, reason: str = "") -> None:
        delta = max(0.0, float(seconds))
        if delta <= 0.0:
            return
        self._last_progress_ts += delta
        if self._playable_seen_since is not None:
            self._playable_seen_since += delta
        if self._non_playable_seen_since is not None:
            self._non_playable_seen_since += delta
        if reason:
            self._logger.write(f"采集恢复等待 {delta:.1f}s：{reason}", tag="VISION_ERROR")

    def _stats_map_name(self) -> str:
        if str(self._current_battle_map_name or "").strip():
            return str(self._current_battle_map_name).strip()
        return str(getattr(self.vision, "_map_name", "") or "").strip()

    def _increment_stats(self, key: str, amount: int = 1, map_name: str = "") -> None:
        stats_mode = str(self.config.stats_mode or "map").strip().lower()
        target_name = str(map_name or self._stats_map_name()).strip()
        self._global_stats.increment(target_name, key, amount)
        if stats_mode == "aggregate":
            self._logger.write(f"{self.config.aggregate_stats_name} 统计更新：{key} +{amount}。", tag="STATISTIC")
        elif target_name:
            self._logger.write(f"地图统计更新：{target_name} {key} +{amount}。", tag="STATISTIC")

    def _set_pending_result_check(self, value: bool, reason: str) -> None:
        self._pending_result_check = bool(value)
        self._set_status(pending_result_check=bool(value))

    def _current_replay_map_id(self) -> str:
        return str(getattr(self.vision, "_map_id", "") or "")

    def _reset_after_battle_resolution(self, reason: str) -> None:
        self._mark_progress(f"battle_reset:{reason}")
        self._battle_started = False
        self._reset_turn_progress_failure_streak("battle_reset")
        self.turn_index = 1
        self._wait_a_enabled = True
        self._wait_silent_logged = False
        self._force_wait_a_reactivation = True
        self._reset_playable_state_timers()
        self.vision.reset_battle_context()
        self._current_battle_map_name = ""
        self._current_battle_stats_recorded = False
        self._set_status(
            phase="waiting_playable",
            turn=1,
            map_id="",
            hand="",
            p1_sp=0,
            last_action="",
            strategy_id=self.config.strategy_id,
            strategy_source="",
        )
        self._logger.write(f"本局已重置上下文，准备进入下一局等待阶段。reason={reason}", tag="GAMEROUND")

    def _analyze_settlement_result(self, frame) -> Optional[Dict[str, Any]]:
        map_name = self._stats_map_name()
        map_id = self._current_replay_map_id()
        if not map_name:
            return None
        try:
            if str(self.config.frame_api_url or "").strip():
                timeout_seconds = max(
                    3.0,
                    float(max(1, int(self.config.settlement_frame_count))) / max(1.0, float(self.config.capture_fps))
                    + float(SETTLEMENT_MULTI_FRAME_TIMEOUT_MARGIN_SECONDS),
                )
                if int(self.config.settlement_frame_count) <= 1:
                    result = analyze_settlement_map_state(frame, map_name)
                else:
                    result = self.vision.run_frame_api_operation(
                        lambda: analyze_settlement_map_state_from_frame_api(
                            map_name=map_name,
                            frame_url=str(self.config.frame_api_url).strip(),
                            sample_count=int(self.config.settlement_frame_count),
                            poll_interval_seconds=float(SETTLEMENT_MULTI_FRAME_POLL_INTERVAL_SECONDS),
                            timeout_seconds=float(timeout_seconds),
                        ),
                        "Settlement frame API",
                    )
            else:
                result = analyze_settlement_map_state(frame, map_name)
            if map_id:
                return _compact_settlement_map_state_for_replay(map_id, result)
            return {
                "map_name": str(result.get("map_name", "") or ""),
                "reference_point_count": int(result.get("reference_point_count", 0) or 0),
                "counts": dict(result.get("counts", {}) or {}),
            }
        except Exception as exc:
            self._logger.write(f"结算地图状态识别失败：map={map_name} error={exc}", tag="VISION")
            return {
                "map_name": map_name,
                "analysis_error": str(exc),
            }

    def _finalize_battle_replay(
        self,
        battle_status: str,
        winner: Optional[str],
        settlement_map_state: Optional[Dict[str, Any]] = None,
    ) -> None:
        path = self._battle_replay.finalize(
            battle_status=battle_status,
            winner=winner,
            rounds_played=min(max(0, int(self.turn_index) - 1), int(self.config.max_turns)),
            map_id=self._current_replay_map_id(),
            map_name=self._stats_map_name(),
            settlement_map_state=settlement_map_state,
        )
        if path is not None:
            self._logger.write(f"已写入回放日志：{path}", tag="REPLAY")

    def _check_progress_timeout(self, phase: str) -> None:
        if not self._battle_started:
            return
        timeout = max(1.0, float(self.config.progress_timeout_seconds))
        elapsed = time.monotonic() - self._last_progress_ts
        if elapsed < timeout:
            return
        self._push_event(f"progress_timeout phase={phase} elapsed={elapsed:.1f}s，触发投降并重开。", tag="TIMEOUT")
        self._run_surrender_sequence()
        self._battle_started = False
        self._set_pending_result_check(False, f"progress_timeout:{phase}")
        self.vision.reset_battle_context()
        raise RuntimeError("BATTLE_PROGRESS_TIMEOUT_SURRENDER")

    def _check_wait_timeout(self, phase: str) -> None:
        raise NotImplementedError("_check_wait_timeout is replaced by playable/non-playable state timers")

    def _reset_playable_state_timers(self) -> None:
        self._playable_seen_since = None
        self._non_playable_seen_since = None

    def _update_playable_state_timers(self, is_playable: bool) -> None:
        now = time.monotonic()
        if is_playable:
            if self._playable_seen_since is None:
                self._playable_seen_since = now
            self._non_playable_seen_since = None
        else:
            if self._non_playable_seen_since is None:
                self._non_playable_seen_since = now
            self._playable_seen_since = None

    def _check_playable_state_timeout(self, is_playable: bool, phase: str) -> None:
        timeout = max(1.0, float(self.config.progress_timeout_seconds))
        start_ts = self._playable_seen_since if is_playable else self._non_playable_seen_since
        if start_ts is None:
            return
        elapsed = time.monotonic() - start_ts
        if elapsed < timeout:
            return
        state_name = "playable" if is_playable else "non_playable"
        self._push_event(
            f"state_timeout phase={phase} state={state_name} elapsed={elapsed:.1f}s，触发投降并重开。",
            tag="TIMEOUT",
        )
        self._logger.write(f"状态 {state_name} 持续超过阈值，执行投降序列并重置到等待阶段。", tag="TIMEOUT")
        self._run_surrender_sequence()
        self._battle_started = False
        self._set_pending_result_check(False, f"state_timeout:{phase}:{state_name}")
        self.vision.reset_battle_context()
        self._reset_playable_state_timers()
        raise RuntimeError("WAIT_PLAYABLE_TIMEOUT_SURRENDER")

    def _wait_for_next_turn_playable(self, action_sent_successfully: bool) -> bool:
        self._push_event("进入下一回合确认阶段：固定等待 3 秒，不进行 playable 检测。", tag="GAMEROUND")
        start_ts = time.monotonic()
        self._sleep_with_pause(3.0)
        self._suspend_timeout_accumulation(time.monotonic() - start_ts)
        self._reset_playable_state_timers()
        seen_non_playable = False
        warned_continuous_playable = False
        recovery_count_at_entry = int(getattr(self.vision, "capture_recovery_success_count", 0))
        while True:
            self._wait_if_paused()
            self._ensure_not_stopped()
            self._check_progress_timeout("wait_next_turn_playable")
            self._set_status(phase="waiting_next_turn_playable", turn=self.turn_index)
            playable_result = self.vision.detect_playable()
            self._set_status(
                playable=bool(playable_result.get("playable")),
                p1_sp=self.vision.last_sp_count,
                last_frame_path=self.vision.last_frame_path,
                last_analysis_path=self.vision.last_analysis_path,
            )
            is_playable = bool(playable_result.get("playable"))
            self._update_playable_state_timers(is_playable)
            recovery_happened_during_wait = (
                int(getattr(self.vision, "capture_recovery_success_count", 0)) > recovery_count_at_entry
            )
            if is_playable:
                if seen_non_playable:
                    self._push_event("已重新检测到 playable，确认进入下一回合。", tag="GAMEROUND")
                    self._mark_progress("重新检测到可出牌状态，确认本回合已成功推进。")
                    self._reset_playable_state_timers()
                    return True
                if recovery_happened_during_wait:
                    if bool(action_sent_successfully):
                        self._push_event(
                            "检测到本轮等待期间发生视频流恢复，且当前 playable=true；本回合动作已成功发送，按已提交处理。",
                            tag="GAMEROUND",
                        )
                        self._logger.write(
                            "本轮等待期间存在视频流恢复，可能错过了中间 playable=false 窗口；因动作已发送成功，按本回合已提交处理。",
                            tag="GAMEROUND",
                        )
                        self._mark_progress("视频流恢复后按本回合已提交成功推进。")
                        self._reset_playable_state_timers()
                        return True
                    self._push_event(
                        "检测到本轮等待期间发生视频流恢复，但本回合动作未成功发送；按未提交处理并重试本回合。",
                        tag="GAMEROUND",
                    )
                    self._logger.write(
                        "本轮等待期间存在视频流恢复，且动作未发送成功；按本回合未提交处理。",
                        tag="GAMEROUND",
                    )
                    self._reset_playable_state_timers()
                    return False
                if not warned_continuous_playable:
                    self._push_event(
                        "出牌后持续检测到 playable，尚未观察到中间的不可出牌阶段，判定本回合未成功提交，准备重新执行本回合。",
                        tag="GAMEROUND",
                    )
                    self._logger.write(
                        "出牌后 playable 持续为 true，未先变为 false；判定为本回合未成功提交，将返回重新识别并重试本回合。",
                        tag="GAMEROUND",
                    )
                    warned_continuous_playable = True
                    self._reset_playable_state_timers()
                    return False
            else:
                if not seen_non_playable:
                    self._push_event("出牌后已观察到 playable=false，开始等待下一回合重新出现 playable。", tag="GAMEROUND")
                seen_non_playable = True
            self._check_playable_state_timeout(is_playable, "wait_next_turn_playable")
            time.sleep(max(0.1, self.config.playable_poll_seconds))

    def debug_snapshot(self) -> Dict[str, Any]:
        with self._status_lock:
            return dict(self._status)

    def toggle_pause(self) -> None:
        if self._paused.is_set():
            self.resume()
        else:
            self._pause_started_ts = time.monotonic()
            self._paused.set()
            self._set_status(status="paused")
            self._push_event("paused_by_user", tag="USER")

    def resume(self) -> None:
        if self._paused.is_set():
            paused_for = 0.0
            if self._pause_started_ts is not None:
                paused_for = max(0.0, time.monotonic() - self._pause_started_ts)
                self._last_progress_ts += paused_for
                if self._playable_seen_since is not None:
                    self._playable_seen_since += paused_for
                if self._non_playable_seen_since is not None:
                    self._non_playable_seen_since += paused_for
            self._pause_started_ts = None
            self._paused.clear()
            self._wait_a_enabled = True
            self._wait_silent_logged = False
            self._force_wait_a_reactivation = True
            self._set_status(status="running")
            self._push_event(
                f"resumed_by_user，已重新进入按 A 等待与 playable 检测状态。暂停时长 {paused_for:.1f}s 已从超时计时中扣除。",
                tag="USER",
            )

    def restart_battle_waiting(self) -> None:
        self._restart_battle_requested.set()
        self._manual_surrender_requested.clear()
        self._manual_surrender_trigger_after_ts = None
        self._manual_surrender_wait_logged = False
        paused_for = 0.0
        if self._paused.is_set():
            if self._pause_started_ts is not None:
                paused_for = max(0.0, time.monotonic() - self._pause_started_ts)
                self._last_progress_ts += paused_for
                if self._playable_seen_since is not None:
                    self._playable_seen_since += paused_for
                if self._non_playable_seen_since is not None:
                    self._non_playable_seen_since += paused_for
            self._pause_started_ts = None
            self._paused.clear()
        self._set_pending_result_check(False, "manual_restart_battle")
        self._battle_started = False
        self.turn_index = 1
        self._wait_a_enabled = True
        self._wait_silent_logged = False
        self._force_wait_a_reactivation = True
        self._reset_playable_state_timers()
        self.vision.reset_battle_context()
        self._current_battle_map_name = ""
        self._current_battle_stats_recorded = False
        self._set_status(
            status="running",
            phase="waiting_playable",
            turn=1,
            map_id="",
            hand="",
            p1_sp=0,
            last_action="",
            playable=False,
            pending_result_check=False,
            strategy_id=self.config.strategy_id,
            strategy_source="",
            last_error="",
        )
        self._push_event(
            f"manual_battle_restart，已清空本局进行中状态并重新进入按 A 等待 playable。暂停时长 {paused_for:.1f}s 已从超时计时中扣除。",
            tag="USER",
        )
        self._logger.write("用户触发重开当前对局：已清空本局进行中状态，重新进入按 A 等待 playable 阶段。", tag="USER")

    def surrender_and_restart_waiting(self) -> None:
        if not self._battle_started:
            self.restart_battle_waiting()
            self._push_event("当前不在对局内，已直接重置到按 A 等待 playable 状态。", tag="USER")
            return
        self._restart_battle_requested.set()
        self._manual_surrender_requested.set()
        delay_seconds = 15.0
        self._manual_surrender_trigger_after_ts = time.monotonic() + delay_seconds
        self._manual_surrender_wait_logged = False
        self._push_event(
            f"已请求手动投降并重开：先等待 {int(delay_seconds)} 秒让当前动作收尾，再执行投降并重置到初始等待状态。",
            tag="USER",
        )
        self._logger.write(
            f"用户触发手动投降并重开：已设置 {int(delay_seconds)} 秒缓冲，缓冲结束后执行投降并重置到按 A 等待 playable。",
            tag="USER",
        )

    def _execute_manual_surrender_restart(self) -> None:
        self._manual_surrender_requested.clear()
        self._manual_surrender_trigger_after_ts = None
        self._manual_surrender_wait_logged = False
        self._restart_battle_requested.clear()
        self._set_status(phase="manual_surrender_restarting", last_error="")
        self._push_event("已中断当前局内状态机，准备执行手动投降并重新开始。", tag="USER")
        had_pending_result = bool(self._pending_result_check)
        with contextlib.suppress(Exception):
            self._run_surrender_sequence()
        self._set_pending_result_check(False, "manual_surrender_restart")
        if self._battle_started or had_pending_result:
            self._finalize_battle_replay("timeout_surrender", None)
        self._reset_after_battle_resolution("manual_surrender_restart")

    def adjust_turn_index(self, delta: int) -> None:
        if delta == 0:
            return
        old_turn = int(self.turn_index)
        new_turn = max(1, min(int(self.config.max_turns), old_turn + int(delta)))
        if new_turn == old_turn:
            self._push_event(f"turn_adjust_ignored current={old_turn} delta={delta}", tag="USER")
            return
        self.turn_index = new_turn
        self._set_status(turn=self.turn_index)
        self._push_event(f"turn_adjusted {old_turn}->{new_turn}", tag="USER")
        self._logger.write(f"用户手动调整当前回合：从第 {old_turn} 回合改为第 {new_turn} 回合。", tag="USER")

    def request_stop(self, reason: str, tag: str = "SYSTEM") -> None:
        self._stop_requested.set()
        self._set_status(status="stopping", last_error=reason)
        self._push_event(reason, tag=tag)

    def _current_serial_port_labels(self) -> List[str]:
        return list_serial_port_labels()

    def _serial_port_supports_switch_link(self, port: str) -> bool:
        target = str(port or "").strip()
        if not target:
            return False
        try:
            ctl = SerialRemoteController(port=target, timeout=0.1)
        except Exception:
            return False
        try:
            return bool(ctl.probe_firmware(timeout_seconds=1.2))
        finally:
            with contextlib.suppress(Exception):
                ctl.close()

    def _serial_port_is_available(self, port: str) -> bool:
        target = str(port or "").strip()
        if not target:
            return False
        return any(parse_device_from_label(label) == target for label in self._current_serial_port_labels())

    def _reconnect_controller_if_needed(self) -> bool:
        current_port = str(self.serial_port or "").strip()
        if current_port and self._serial_port_is_available(current_port):
            try:
                self.controller.release()
                return False
            except Exception:
                pass
        labels = self._current_serial_port_labels()
        if not labels:
            self._logger.write("switch_link 串口检查失败：当前没有可用串口，程序即将退出。", tag="CONTROLLER_ERROR")
            self._push_event("switch_link_unavailable_exit", tag="CONTROLLER_ERROR")
            self._set_status(last_error="NO_AVAILABLE_SWITCH_LINK_PORT", phase="error")
            raise RuntimeError("NO_AVAILABLE_SWITCH_LINK_PORT")
        next_port = ""
        for label in labels:
            candidate = parse_device_from_label(label)
            if self._serial_port_supports_switch_link(candidate):
                next_port = candidate
                break
        if not next_port:
            self._logger.write(
                "switch_link 串口检查失败：检测到串口存在，但没有可握手的虚拟手柄串口，程序即将退出。",
                tag="CONTROLLER_ERROR",
            )
            self._push_event("switch_link_unavailable_exit", tag="CONTROLLER_ERROR")
            self._set_status(last_error="NO_AVAILABLE_SWITCH_LINK_PORT", phase="error")
            raise RuntimeError("NO_AVAILABLE_SWITCH_LINK_PORT")
        with contextlib.suppress(Exception):
            self.controller.close()
        self.controller = SerialRemoteController(port=next_port)
        self.serial_port = next_port
        self.config.serial_port = next_port
        self.config.pick_serial = False
        runtime_config_path = str(getattr(self.config, "_runtime_config_path", "") or "").strip()
        if runtime_config_path:
            runtime_payload = _load_json_obj(Path(runtime_config_path))
            runtime_payload["serial_port"] = next_port
            runtime_payload["pick_serial"] = False
            _write_json_obj(Path(runtime_config_path), runtime_payload)
        self._set_status(serial_port=self.serial_port)
        self._logger.write(f"检测到原 switch_link 串口不可用，已自动切换到当前可用串口：{next_port}。", tag="CONTROLLER")
        self._push_event(f"switch_link_reconnected:{next_port}", tag="CONTROLLER")
        return True

    def _run_controller_with_reconnect(self, action: Callable[[], None], *, context: str) -> None:
        try:
            action()
            return
        except Exception as exc:
            self._logger.write(f"{context} 发送失败，开始检查 switch_link 串口状态。error={exc}", tag="CONTROLLER_ERROR")
        reconnected = self._reconnect_controller_if_needed()
        if reconnected:
            self._logger.write(f"switch_link 串口已恢复，重试发送：{context}", tag="CONTROLLER")
        else:
            self._logger.write(f"switch_link 串口表面可用，重试发送：{context}", tag="CONTROLLER")
        try:
            action()
        except Exception as exc:
            self._logger.write(f"{context} 重试后仍失败：{exc}", tag="CONTROLLER_ERROR")
            raise RuntimeError("SWITCH_LINK_UNAVAILABLE_EXIT")

    def send_manual_controller_input(self, token: str, hold_ms: int = 50, gap_ms: int = 100) -> None:
        token_upper = str(token).upper()
        if token_upper in {"PLUS", "HOME"}:
            self.controller.send_smart_sequence_csv_blocking(f"{token_upper},1", timeout_seconds=2.0)
            self._push_event(f"manual_controller_input:{token}", tag="USER")
            return
        special_timing = {
            "L": (80, 120),
            "R": (80, 120),
        }
        if token_upper in special_timing:
            press_ms, release_gap_ms = special_timing[token_upper]
        else:
            press_ms = max(20, int(hold_ms))
            release_gap_ms = max(40, int(gap_ms))
        bit_map = dict(REMOTE_INPUT_BITS)
        bit_map.update(
            {
                "LUP": REMOTE_INPUT_BITS["LSTICK_UP"],
                "LDOWN": REMOTE_INPUT_BITS["LSTICK_DOWN"],
                "LLEFT": REMOTE_INPUT_BITS["LSTICK_LEFT"],
                "LRIGHT": REMOTE_INPUT_BITS["LSTICK_RIGHT"],
                "RUP": REMOTE_INPUT_BITS["RSTICK_UP"],
                "RDOWN": REMOTE_INPUT_BITS["RSTICK_DOWN"],
                "RLEFT": REMOTE_INPUT_BITS["RSTICK_LEFT"],
                "RRIGHT": REMOTE_INPUT_BITS["RSTICK_RIGHT"],
                "DUP": REMOTE_INPUT_BITS["DPAD_UP"],
                "DDOWN": REMOTE_INPUT_BITS["DPAD_DOWN"],
                "DLEFT": REMOTE_INPUT_BITS["DPAD_LEFT"],
                "DRIGHT": REMOTE_INPUT_BITS["DPAD_RIGHT"],
            }
        )
        bit_index = bit_map.get(token_upper)
        if bit_index is None:
            raise ValueError(f"unsupported manual controller token: {token}")
        self.controller.run_steps(
            [RemoteStep(bits=(1 << bit_index), hold_ms=press_ms, gap_ms=release_gap_ms)]
        )
        self._push_event(f"manual_controller_input:{token}", tag="USER")

    def _press_home_and_stop(self, reason: str, tag: str = "SYSTEM") -> None:
        self._set_status(phase="stopping", last_error=reason)
        self._push_event(reason, tag=tag)
        self.controller.send_smart_sequence_csv_blocking("HOME,1", timeout_seconds=2.0)
        time.sleep(0.3)
        self.request_stop(reason, tag=tag)

    def _prompt_next_target_after_goal_reached(self) -> bool:
        reached = int(self.win_count)
        should_restart_ui = False
        self._logger.write(f"已达成本次目标胜场：{reached}。等待用户选择是否继续对战。", tag="SYSTEM")
        self._push_event(f"已达成目标胜场 {reached}。等待用户选择是否继续。", tag="SYSTEM")
        self._set_status(status="paused", phase="target_reached", last_error="", wins=reached)
        self._debug_ui.stop()
        try:
            choice = choose_with_arrows(
                ["继续对战", "结束返回终端"],
                title=f"已达成目标胜场：{reached}。是否继续对战？（使用 ↑/↓ 选择，回车确认）",
                footer="",
            )
            if choice != "继续对战":
                self._logger.write(f"用户选择结束本次自动对战。已达成胜场：{reached}。", tag="USER")
                self._set_status(status="stopped", phase="stopped")
                return False

            self.vision.ensure_capture_ready()
            while True:
                raw = input("请输入新的目标胜场（>=0，直接回车表示按配置继续运行）：").strip()
                if raw == "":
                    original_continuous_run = bool(getattr(self.config, "_original_continuous_run", self.config.continuous_run))
                    original_target_win_count = int(
                        getattr(self.config, "_original_target_win_count", self.config.target_win_count)
                    )
                    self.config.continuous_run = original_continuous_run
                    self.config.target_win_count = original_target_win_count
                    self._logger.write("用户未输入新的目标胜场，已恢复为配置文件中的原始运行模式。", tag="USER")
                    self._push_event("用户未输入新的目标胜场，已恢复为配置文件中的原始运行模式。", tag="USER")
                    self._set_status(status="running", phase="idle", last_error="")
                    should_restart_ui = True
                    return True
                try:
                    value = int(raw)
                except ValueError:
                    print(f"无效输入：{raw}，请输入 >=0 的整数，或直接回车按配置继续运行。")
                    continue
                if value < 0:
                    print("目标胜场不能为负数。")
                    continue
                if value == 0:
                    self.config.continuous_run = True
                    self._logger.write(f"用户选择继续对战，并切换为无限循环模式。当前累计胜场={reached}。", tag="USER")
                    self._push_event("用户选择继续对战，后续改为无限循环。", tag="USER")
                else:
                    self.config.continuous_run = False
                    self.config.target_win_count = reached + value
                    self._logger.write(
                        f"用户选择继续对战，并设置新的目标：还需再赢 {value} 局；"
                        f"新的累计停止胜场={self.config.target_win_count}。",
                        tag="USER",
                    )
                    self._push_event(
                        f"用户选择继续对战，新的目标为再赢 {value} 局（累计 {self.config.target_win_count} 胜停止）。",
                        tag="USER",
                    )
                self._set_status(status="running", phase="idle", last_error="")
                should_restart_ui = True
                return True
        finally:
            if should_restart_ui and not self._closed:
                self._debug_ui.start()

    def _run_surrender_sequence(self) -> None:
        self._push_event("执行投降：按手柄 +，等待 2 秒，再按右，等待 0.5 秒，按 A；再等待 2 秒后再按一次 A，最后等待 5 秒。", tag="CONTROLLER")
        self._run_controller_with_reconnect(
            lambda: self.controller.send_smart_sequence_csv_blocking("PLUS,1", timeout_seconds=2.0),
            context="投降序列:PLUS",
        )
        time.sleep(2.0)
        self._run_controller_with_reconnect(
            lambda: self.controller.run_steps([RemoteStep(bits=(1 << BIT_DPAD_RIGHT), hold_ms=120, gap_ms=0)]),
            context="投降序列:DRIGHT",
        )
        time.sleep(0.5)
        self._run_controller_with_reconnect(
            lambda: self.controller.run_steps([RemoteStep(bits=(1 << BIT_A), hold_ms=120, gap_ms=0)]),
            context="投降序列:A",
        )
        time.sleep(2.0)
        self._run_controller_with_reconnect(
            lambda: self.controller.run_steps([RemoteStep(bits=(1 << BIT_A), hold_ms=120, gap_ms=0)]),
            context="投降序列:A_2nd",
        )
        self._push_event("投降序列完成，冷却等待 5 秒后再继续后续检测。", tag="GAMEROUND")
        self._sleep_with_pause(5.0)

    def _run_resume_reactivation_sequence(self) -> None:
        self._push_event("恢复后执行手柄 R x3，每次间隔 1 秒，然后回到按 A 等待与 playable 检测。", tag="CONTROLLER")
        for idx in range(3):
            self.controller.run_steps([RemoteStep(bits=(1 << BIT_R), hold_ms=50, gap_ms=100)])
            if idx < 2:
                time.sleep(1.0)

    def _execute_battle_action(self, action, observed_state: ObservedState) -> None:
        if bool(self.config.clone_jelly_split_action_execution):
            selection_steps = compile_action_menu_selection_steps(
                action,
                observed_state,
                sp_attack_up_right=bool(self.config.clone_jelly_sp_attack_up_right),
            )
            if selection_steps:
                self.controller.run_steps(selection_steps)
            map_phase_csv = compile_action_map_phase_csv(action, observed_state)
            if map_phase_csv:
                time.sleep(0.1)
                self.controller.send_smart_sequence_csv_blocking(map_phase_csv, timeout_seconds=15.0)
            return
        battle_steps = compile_action_to_runtime_steps(action, observed_state)
        self.controller.run_steps(battle_steps)

    def _finalize_previous_battle_as_not_win(self, reason: str) -> None:
        if not self._pending_result_check:
            return
        self._set_pending_result_check(False, f"finalize_previous_battle_as_not_win:{reason}")
        self._push_event(reason, tag="GAMEROUND")
        self._logger.write("上一局在重新进入可出牌前未检测到明确 win/lose/draw 标志，按结果不确定处理。", tag="GAMEROUND")
        self._finalize_battle_replay("uncertain", None)
        self._current_battle_map_name = ""
        self._current_battle_stats_recorded = False

    def _record_battle_result_from_frame(
        self,
        frame,
        lose_result: Dict[str, Any],
        win_result: Dict[str, Any],
        draw_result: Dict[str, Any],
    ) -> None:
        if not self._pending_result_check:
            return
        if bool(lose_result.get("lose")):
            settlement_map_state = self._analyze_settlement_result(frame)
            if isinstance(settlement_map_state, dict) and isinstance(settlement_map_state.get("map_grid"), list):
                p1_score, p2_score = _compute_board_scores_from_grid(settlement_map_state["map_grid"])
                self._logger.write(f"结算比分：{_format_score_broadcast(p1_score, p2_score)}", tag="GAMEROUND")
            self._set_pending_result_check(False, "settlement_detected_lose_banner")
            self.win_count += 1
            stats_map_name = self._stats_map_name()
            if stats_map_name or str(self.config.stats_mode or "").strip().lower() == "aggregate":
                self._increment_stats("Wins", 1, stats_map_name)
            self._set_status(wins=self.win_count)
            self._push_event(f"battle_result=win total_wins={self.win_count}", tag="GAMEROUND")
            self._mark_progress("检测到敌方战败标志，本局记为胜利。")
            self._finalize_battle_replay("win", "P1", settlement_map_state=settlement_map_state)
            self._reset_after_battle_resolution("win")
            if (not self.config.continuous_run) and self.win_count >= max(1, int(self.config.target_win_count)):
                raise TargetWinGoalReached("target_win_count_reached")
            return
        if bool(win_result.get("win")):
            settlement_map_state = self._analyze_settlement_result(frame)
            if isinstance(settlement_map_state, dict) and isinstance(settlement_map_state.get("map_grid"), list):
                p1_score, p2_score = _compute_board_scores_from_grid(settlement_map_state["map_grid"])
                self._logger.write(f"结算比分：{_format_score_broadcast(p1_score, p2_score)}", tag="GAMEROUND")
            self._set_pending_result_check(False, "settlement_detected_win_banner")
            self._push_event("battle_result=lose", tag="GAMEROUND")
            self._mark_progress("检测到我方战败标志，本局记为战败。")
            self._finalize_battle_replay("lose", "P2", settlement_map_state=settlement_map_state)
            self._reset_after_battle_resolution("lose")
            return
        if bool(draw_result.get("draw")):
            settlement_map_state = self._analyze_settlement_result(frame)
            if isinstance(settlement_map_state, dict) and isinstance(settlement_map_state.get("map_grid"), list):
                p1_score, p2_score = _compute_board_scores_from_grid(settlement_map_state["map_grid"])
                self._logger.write(f"结算比分：{_format_score_broadcast(p1_score, p2_score)}", tag="GAMEROUND")
            self._set_pending_result_check(False, "settlement_detected_draw_banner")
            self._push_event("battle_result=draw", tag="GAMEROUND")
            self._mark_progress("检测到平局标志，本局记为平局。")
            self._finalize_battle_replay("draw", "draw", settlement_map_state=settlement_map_state)
            self._reset_after_battle_resolution("draw")
            return

    def _wait_if_paused(self) -> None:
        while self._paused.is_set() and not self._stop_requested.is_set():
            time.sleep(0.1)

    def _sleep_with_pause(self, seconds: float) -> None:
        remaining = max(0.0, float(seconds))
        deadline = time.monotonic() + remaining
        while remaining > 0:
            self._wait_if_paused()
            self._ensure_not_stopped()
            chunk = min(0.1, remaining)
            time.sleep(chunk)
            remaining = max(0.0, deadline - time.monotonic())

    def _ensure_not_stopped(self) -> None:
        if self._stop_requested.is_set():
            raise KeyboardInterrupt("stop requested")
        if self._manual_surrender_requested.is_set():
            trigger_ts = self._manual_surrender_trigger_after_ts
            if trigger_ts is None or time.monotonic() >= trigger_ts:
                raise RuntimeError("MANUAL_BATTLE_SURRENDER_RESTART")
            raise RuntimeError("MANUAL_BATTLE_SURRENDER_WAIT")
        if self._restart_battle_requested.is_set():
            raise RuntimeError("MANUAL_BATTLE_RESTART")

    def wait_until_playable(self) -> Dict[str, Any]:
        self._set_status(phase="waiting_playable", playable=False)
        self._push_event(
            "开始等待进入可出牌状态。每次按 A 前都会先检查当前帧，避免在已可出牌时误按 A 选中第一张卡。",
            tag="GAMEROUND",
        )
        self._mark_progress("进入等待可出牌阶段。")
        self._reset_playable_state_timers()
        wait_a_step = [RemoteStep(bits=(1 << BIT_A), hold_ms=self.config.wait_press_hold_ms, gap_ms=0)]
        next_wait_a_ts = time.monotonic()
        while True:
            self._wait_if_paused()
            self._ensure_not_stopped()
            frame, playable_result, lose_result, win_result, draw_result = self.vision.inspect_wait_frame()
            self._set_status(
                phase="waiting_playable",
                playable=bool(playable_result.get("playable")),
                last_frame_path=self.vision.last_frame_path,
                last_analysis_path=self.vision.last_analysis_path,
                p1_sp=self.vision.last_sp_count,
                pending_result_check=self._pending_result_check,
                wins=self.win_count,
                battles=self.battle_count,
            )
            is_playable = bool(playable_result.get("playable"))
            self._update_playable_state_timers(is_playable)
            if self._pending_result_check:
                self.vision.save_settlement_poll_capture(frame)
                self._record_battle_result_from_frame(frame, lose_result, win_result, draw_result)
                self._ensure_not_stopped()
            if is_playable:
                if self._pending_result_check:
                    self._set_pending_result_check(False, "playable_detected_before_settlement_resolved")
                    self._push_event("battle_result_unresolved_on_playable", tag="GAMEROUND")
                    self._logger.write(
                        "已重新进入可出牌状态，但上一局未检测到明确 win/lose/draw 标志；不按 playable 推断胜负，本局结果记为 uncertain。",
                        tag="GAMEROUND",
                    )
                    self._finalize_battle_replay("uncertain", None)
                    self._reset_after_battle_resolution("uncertain")
                    self._push_event("上一局结果未解析，已清除旧地图上下文，下一局将重新检测地图。", tag="GAMEROUND")
                self._push_event("playable_detected", tag="VISION")
                self._mark_progress("检测到可出牌状态，停止继续按 A。")
                self._wait_a_enabled = False
                self._wait_silent_logged = False
                self._reset_playable_state_timers()
                with contextlib.suppress(Exception):
                    self._run_controller_with_reconnect(
                        lambda: self.controller.run_steps([RemoteStep(bits=(1 << BIT_B), hold_ms=50, gap_ms=0)]),
                        context="playable_detected_post_disable_a:B",
                    )
                self._sleep_with_pause(0.5)
                return playable_result
            self._check_playable_state_timeout(False, "wait_until_playable")
            if self._wait_a_enabled:
                self._wait_silent_logged = False
                if self._force_wait_a_reactivation:
                    next_wait_a_ts = 0.0
                    self._force_wait_a_reactivation = False
                now = time.monotonic()
                if now >= next_wait_a_ts:
                    self.controller.run_steps(wait_a_step)
                    next_wait_a_ts = now + max(0.1, self.config.wait_press_gap_ms / 1000.0) + 0.1
            else:
                if not self._wait_silent_logged:
                    self._push_event("当前未进入可出牌状态，但已进入静默等待阶段，本轮不再自动按 A。", tag="GAMEROUND")
                    self._wait_silent_logged = True
            time.sleep(max(0.05, self.config.playable_poll_seconds))

    def play_one_battle(self) -> None:
        self.turn_index = 1
        self._reset_turn_progress_failure_streak("new_battle")
        try:
            self.wait_until_playable()
            self._current_battle_map_name = ""
            self._current_battle_stats_recorded = False
            self.vision.reset_battle_context()
            self.battle_count += 1
            self._battle_replay.start_battle(self.battle_count)
            self._battle_started = True
            self._mark_progress("新对局开始。")
            self._set_status(phase="battle_started", turn=1, battles=self.battle_count)
            self._push_event("battle_started", tag="GAMEROUND")
            while self.turn_index <= self.config.max_turns:
                self._wait_if_paused()
                self._ensure_not_stopped()
                self._check_progress_timeout("battle_turn_loop")
                state = self.vision.parse_turn_state(turn_index=self.turn_index)
                self._battle_replay.complete_previous_move_after_state(state)
                if (not self._current_battle_stats_recorded) and state.map_name:
                    self._current_battle_map_name = str(state.map_name)
                    self._increment_stats("Battles", 1, self._current_battle_map_name)
                    self._current_battle_stats_recorded = True
                observed_state = state.to_observed_state()
                resolved_strategy = resolve_strategy(self.config, state.map_id, state.map_name)
                p1_score_now, p2_score_now = _compute_board_scores_from_grid(state.map_grid)
                self._logger.write(
                    f"第 {self.turn_index} 回合识别完成：地图={state.map_name}({state.map_id})，手牌={state.hand_card_numbers}，SP={state.p1_sp}。"
                , tag="VISION")
                self._logger.write(f"第 {self.turn_index} 回合{_format_score_broadcast(p1_score_now, p2_score_now)}", tag="GAMEROUND")
                self._set_status(
                    phase="planning_action",
                    turn=self.turn_index,
                    map_id=state.map_id,
                    playable=bool(state.playable_result.get("playable")),
                    hand=",".join(str(n) for n in state.hand_card_numbers),
                    p1_sp=state.p1_sp,
                    last_frame_path=self.vision.last_frame_path,
                    last_analysis_path=self.vision.last_analysis_path,
                    last_error="",
                    strategy_id=resolved_strategy.label,
                    strategy_source=resolved_strategy.source,
                )
                action = choose_action_from_resolved_strategy(observed_state, resolved_strategy)
                action_text = (
                    f"card={action.card_number} rot={action.rotation} "
                    f"xy=({action.x},{action.y}) pass={action.pass_turn} sp={action.use_sp_attack} surrender={action.surrender}"
                )
                self._set_status(last_action=action_text)
                self._push_event(f"action_turn_{self.turn_index}: {action_text}", tag="STRATEGY")
                self._logger.write(
                    f"第 {self.turn_index} 回合采用策略 {resolved_strategy.label}，来源={resolved_strategy.source}，动作={action_text}。",
                    tag="STRATEGY",
                )
                self._battle_replay.record_turn(state, action)
                self._set_status(phase="executing_action")
                action_sent_successfully = False
                try:
                    self._execute_battle_action(action, observed_state)
                    action_sent_successfully = True
                except Exception:
                    self._logger.write("当前出牌序列发送异常，开始检查 switch_link 串口状态，并保留在当前回合重试。", tag="CONTROLLER_ERROR")
                    reconnected = self._reconnect_controller_if_needed()
                    if reconnected:
                        self._logger.write("switch_link 串口已恢复，当前回合将重新识别并重试。", tag="CONTROLLER")
                    else:
                        self._logger.write("switch_link 串口检查正常，当前回合将重新识别并重试。", tag="CONTROLLER")
                    self._battle_replay.discard_last_move()
                    self._set_status(phase="retrying_turn", last_error="ACTION_SEND_FAILED_RETRY_TURN")
                    self._push_event(f"第 {self.turn_index} 回合动作发送失败，保留在当前回合并重试。", tag="CONTROLLER_ERROR")
                    if self._record_turn_progress_failure("action_send_failed"):
                        self._push_event("连续 3 次回合推进失败，触发投降并重开。", tag="TIMEOUT")
                        self._logger.write("连续 3 次回合推进失败（动作发送失败/未提交），执行投降并重开。", tag="TIMEOUT")
                        self._run_surrender_sequence()
                        self._battle_started = False
                        self._set_pending_result_check(False, "turn_progress_failure_streak_surrender")
                        self.vision.reset_battle_context()
                        self._reset_playable_state_timers()
                        raise RuntimeError("TURN_PROGRESS_FAILURE_STREAK_SURRENDER")
                    time.sleep(max(0.1, self.config.playable_poll_seconds))
                    continue
                self.vision.remember_executed_specials(action)
                if self.turn_index < self.config.max_turns:
                    committed = self._wait_for_next_turn_playable(action_sent_successfully=action_sent_successfully)
                    if not committed:
                        self._battle_replay.discard_last_move()
                        self._set_status(phase="retrying_turn", last_error="ACTION_NOT_COMMITTED_RETRY_TURN")
                        self._push_event(f"第 {self.turn_index} 回合未成功提交，准备重新识别并重试本回合。", tag="GAMEROUND")
                        self._logger.write(
                            f"第 {self.turn_index} 回合动作未成功提交，本回合不推进，重新识别当前局面后重试。",
                            tag="GAMEROUND",
                        )
                        if self._record_turn_progress_failure("action_not_committed"):
                            self._push_event("连续 3 次回合推进失败，触发投降并重开。", tag="TIMEOUT")
                            self._logger.write("连续 3 次回合推进失败（动作发送失败/未提交），执行投降并重开。", tag="TIMEOUT")
                            self._run_surrender_sequence()
                            self._battle_started = False
                            self._set_pending_result_check(False, "turn_progress_failure_streak_surrender")
                            self.vision.reset_battle_context()
                            self._reset_playable_state_timers()
                            raise RuntimeError("TURN_PROGRESS_FAILURE_STREAK_SURRENDER")
                        time.sleep(max(0.1, self.config.playable_poll_seconds))
                        continue
                self._reset_turn_progress_failure_streak("turn_advanced")
                self.turn_index += 1
                self._mark_progress(f"已完成一次动作执行，进入回合索引 {self.turn_index}。")
                self._set_status(phase="turn_complete", turn=min(self.turn_index, self.config.max_turns))
                time.sleep(max(0.1, self.config.playable_poll_seconds))
            self._push_event("battle_complete", tag="GAMEROUND")
            self._set_pending_result_check(True, "battle_complete_after_turn_12")
            self._battle_started = False
            self._wait_a_enabled = True
            self._wait_silent_logged = False
            self._logger.write(
                f"12 回合结束，进入结算检查阶段。当前累计局数={self.battle_count}，累计胜场={self.win_count}。",
                tag="GAMEROUND",
            )
            self._set_status(phase="battle_complete", battles=self.battle_count)
            self._push_event("结算阶段冷却：固定等待 7 秒，结束后再重启按 A 与 playable 检测。", tag="GAMEROUND")
            self._sleep_with_pause(7.0)
        except TargetWinGoalReached:
            raise
        except RuntimeError as exc:
            if str(exc) == "MANUAL_BATTLE_RESTART":
                self._restart_battle_requested.clear()
                self._set_status(phase="waiting_playable", last_error="")
                self._push_event("已中断当前局内状态机，重新回到新的按 A 等待 playable 状态。", tag="USER")
                return
            if str(exc) == "MANUAL_BATTLE_SURRENDER_WAIT":
                remaining = max(
                    0.0,
                    float(self._manual_surrender_trigger_after_ts or time.monotonic()) - time.monotonic(),
                )
                self._set_status(phase="manual_surrender_waiting", last_error="")
                if not self._manual_surrender_wait_logged:
                    self._push_event(
                        f"手动投降缓冲中：等待 {remaining:.1f}s 后执行投降。缓冲期间暂停 playable 检测与对局主流程。",
                        tag="USER",
                    )
                    self._manual_surrender_wait_logged = True
                if remaining > 0.0:
                    time.sleep(remaining)
                self._execute_manual_surrender_restart()
                return
            if str(exc) == "MANUAL_BATTLE_SURRENDER_RESTART":
                self._execute_manual_surrender_restart()
                return
            if str(exc) in {"NO_AVAILABLE_SWITCH_LINK_PORT", "SWITCH_LINK_UNAVAILABLE_EXIT"}:
                self._set_status(phase="error", last_error=str(exc), status="stopping")
                self._push_event("程序已安全退出：switch_link 串口不可用或无法恢复，请检查 CP2104/串口连接后重新启动。", tag="CONTROLLER_ERROR")
                self._logger.write("switch_link 串口不可用，已安全退出。", tag="CONTROLLER_ERROR")
                self.request_stop("switch_link_unavailable_exit", tag="CONTROLLER_ERROR")
                return
            if "recovery window" in str(exc) and "Capture frame" in str(exc):
                self._set_status(phase="error", last_error=str(exc), status="stopping")
                self._push_event(f"视频流在恢复窗口内未能恢复，程序安全退出：{exc}", tag="VISION_ERROR")
                self._logger.write(f"视频流恢复失败，已达到恢复窗口上限，准备安全退出：{exc}", tag="VISION_ERROR")
                self.request_stop("capture_frame_unavailable_exit", tag="VISION_ERROR")
                return
            if str(exc) in {
                "BATTLE_PROGRESS_TIMEOUT_SURRENDER",
                "WAIT_PLAYABLE_TIMEOUT_SURRENDER",
                "TURN_PROGRESS_FAILURE_STREAK_SURRENDER",
            }:
                if self._current_battle_stats_recorded and (
                    self._current_battle_map_name or str(self.config.stats_mode or "").strip().lower() == "aggregate"
                ):
                    self._increment_stats("Errors", 1, self._current_battle_map_name)
                    self._logger.write("本局超时投降，已计入错误统计。", tag="STATISTIC")
                self._finalize_battle_replay("timeout_surrender", None)
                self._set_pending_result_check(False, f"runtime_error:{exc}")
                self._set_status(phase="battle_timeout_restarting", last_error=str(exc))
                self._push_event("对局因长时间无推进而投降，已重置状态，准备重新从等待阶段开始。", tag="TIMEOUT")
                self._reset_after_battle_resolution("timeout_surrender")
                return
            if str(exc) == "BATTLE_ACTION_SEQUENCE_ABORTED":
                self._set_status(phase="battle_action_aborted", last_error=str(exc))
                self._push_event("当前出牌序列已中止，已重置状态并重新回到等待阶段。", tag="CONTROLLER_ERROR")
                self._reset_after_battle_resolution("battle_action_sequence_aborted")
                return
            self._set_status(last_error=str(exc), phase="error")
            self._push_event(f"error: {exc}", tag="SYSTEM")
            raise
        except Exception as exc:
            self._finalize_battle_replay("error", None)
            self._set_status(last_error=str(exc), phase="error")
            self._push_event(f"error: {exc}", tag="SYSTEM")
            raise

    def run_forever(self) -> None:
        try:
            while True:
                self._ensure_not_stopped()
                try:
                    self.play_one_battle()
                except TargetWinGoalReached:
                    if not self._prompt_next_target_after_goal_reached():
                        break
        except KeyboardInterrupt:
            last_error = ""
            with self._status_lock:
                last_error = str(self._status.get("last_error", "") or "")
            if last_error == "capture_frame_unavailable_exit":
                self._push_event("程序已安全退出：视频流在恢复窗口内未能恢复，请检查采集卡/代理/VPN 状态后重新启动。", tag="VISION_ERROR")
            else:
                self._push_event("runtime_stopped", tag="SYSTEM")
            self._set_status(status="stopped", phase="stopped")


class _FrameVisionPipeline:
    def __init__(
        self,
        config: ControllerConfig,
        on_capture_recovery_pause: Optional[Callable[[float, str], None]] = None,
        on_capture_recovery_event: Optional[Callable[[str], None]] = None,
        on_capture_interactive_prompt: Optional[Callable[[bool], None]] = None,
    ):
        self._config = config
        self._on_capture_recovery_pause = on_capture_recovery_pause
        self._on_capture_recovery_event = on_capture_recovery_event
        self._on_capture_interactive_prompt = on_capture_interactive_prompt
        layout_path = Path(config.layout_json)
        if not layout_path.is_absolute():
            layout_path = REPO_ROOT / layout_path
        self._layout = _load_layout(layout_path)
        frame_api_url = str(config.frame_api_url or "").strip()
        if frame_api_url:
            self._frame_api_launcher = FrameApiAutoLauncher(config)
            self._frame_api_launcher.ensure_started()
            self._capture = HttpJpegCaptureSource(frame_api_url=frame_api_url)
            device_name = f"http:{frame_api_url}"
        else:
            self._frame_api_launcher = None
            device_name = config.capture_device_name.strip() or auto_detect_capture_device_name(prefer_usb=True)
            if not device_name:
                raise RuntimeError("No capture device available")
            self._capture = FFmpegCaptureSource(
                device_name=device_name,
                width=config.capture_width,
                height=config.capture_height,
                fps=config.capture_fps,
                pixel_format=config.capture_pixel_format,
                strict_usb_only=False,
            )
        self._supplemental_provider = (
            _load_callable(config.supplemental_state_provider) if config.supplemental_state_provider else None
        )
        self._map_id: Optional[str] = None
        self._map_name: Optional[str] = None
        self._map_tracker: Optional[MapStateTracker] = None
        self._persisted_p1_special_positions: Set[Tuple[int, int]] = set()
        self._persisted_p2_special_positions: Set[Tuple[int, int]] = set()
        self._persisted_conflict_positions: Set[Tuple[int, int]] = set()
        self.last_sp_count = 0
        self.last_enemy_sp_count = 0
        debug_dir = _resolve_debug_screenshot_dir(config.debug_frame_dir)
        self._debug_dir = debug_dir
        self._debug_dir.mkdir(parents=True, exist_ok=True)
        self._tmp_dir = _resolve_debug_tmp_dir(self._debug_dir)
        self._tmp_dir.mkdir(parents=True, exist_ok=True)
        self.last_frame_path = ""
        self.last_analysis_path = ""
        self._playable_shot_index = 0
        self._settlement_shot_index = 0
        self._capture_recovery_success_count = 0

    def close(self) -> None:
        self._capture.stop()
        if self._frame_api_launcher is not None:
            self._frame_api_launcher.stop()

    def reset_battle_context(self) -> None:
        self._map_id = None
        self._map_name = None
        self._map_tracker = None
        self._persisted_p1_special_positions.clear()
        self._persisted_p2_special_positions.clear()
        self._persisted_conflict_positions.clear()

    def ensure_capture_ready(self) -> None:
        if self._frame_api_launcher is not None:
            self._frame_api_launcher.ensure_started()

    def run_frame_api_operation(self, operation: Callable[[], Any], description: str) -> Any:
        if self._frame_api_launcher is None:
            return operation()

        last_error = "unknown error"
        recovery_started_at = time.monotonic()
        failed_rounds = 0
        restart_count = 0

        while True:
            round_started_at = time.monotonic()
            try:
                result = operation()
                if failed_rounds > 0 and self._on_capture_recovery_event is not None:
                    self._on_capture_recovery_event(
                        f"{description} 已恢复成功（此前失败 {failed_rounds} 轮，重启预览 {restart_count} 次）。"
                    )
                if failed_rounds > 0:
                    self._capture_recovery_success_count += 1
                return result
            except Exception as exc:
                last_error = str(exc)
                failed_rounds += 1

            elapsed = time.monotonic() - recovery_started_at
            if elapsed >= float(CAPTURE_RECOVERY_TOTAL_TIMEOUT_SECONDS) or failed_rounds >= int(CAPTURE_RECOVERY_MAX_RETRIES):
                break

            if self._on_capture_recovery_event is not None:
                self._on_capture_recovery_event(
                    f"{description} 失败，准备在 {CAPTURE_RECOVERY_RETRY_SECONDS:.0f}s 后重试 "
                    f"(本轮累计失败 {failed_rounds} 次，总恢复 {elapsed:.1f}s)：{last_error}"
                )

            try:
                if self._on_capture_interactive_prompt is not None:
                    self._on_capture_interactive_prompt(True)
                self._frame_api_launcher.ensure_started()
                restart_count += 1
                if self._on_capture_recovery_event is not None:
                    self._on_capture_recovery_event(
                        f"{description} 本轮已尝试恢复/拉起视频流（第 {restart_count} 次）。"
                    )
            except Exception as exc:
                last_error = f"{last_error}; FRAME_API_RESTART_FAILED:{exc}"
                if self._on_capture_recovery_event is not None:
                    self._on_capture_recovery_event(f"恢复/拉起视频流失败：{exc}")
            finally:
                if self._on_capture_interactive_prompt is not None:
                    self._on_capture_interactive_prompt(False)

            round_elapsed = time.monotonic() - round_started_at
            remaining = float(CAPTURE_RECOVERY_TOTAL_TIMEOUT_SECONDS) - (time.monotonic() - recovery_started_at)
            if remaining <= 0:
                if self._on_capture_recovery_pause is not None and round_elapsed > 0:
                    self._on_capture_recovery_pause(float(round_elapsed), f"{description} 恢复失败轮次")
                break
            sleep_seconds = min(max(0.0, float(CAPTURE_RECOVERY_RETRY_SECONDS) - round_elapsed), max(0.0, remaining))
            total_paused = max(0.0, round_elapsed) + max(0.0, sleep_seconds)
            if self._on_capture_recovery_pause is not None and total_paused > 0:
                self._on_capture_recovery_pause(float(total_paused), f"{description} 恢复等待")
            time.sleep(float(sleep_seconds))

        raise RuntimeError(
            f"{description} unavailable after {int(CAPTURE_RECOVERY_TOTAL_TIMEOUT_SECONDS)}s recovery window: {last_error}"
        )

    def _read_latest_frame(self):
        fallback_specs = [
            {"width": self._config.capture_width, "height": self._config.capture_height, "pixel_format": self._config.capture_pixel_format},
            {"width": 1920, "height": 1080, "pixel_format": ""},
            {"width": 1280, "height": 720, "pixel_format": ""},
            {"width": 1280, "height": 720, "pixel_format": self._config.capture_pixel_format},
            {"width": 1920, "height": 1080, "pixel_format": "uyvy422"},
            {"width": 1280, "height": 720, "pixel_format": "uyvy422"},
            {"width": 1280, "height": 720, "pixel_format": "nv12"},
            {"width": 1920, "height": 1080, "pixel_format": "nv12"},
            {"width": 1280, "height": 720, "pixel_format": "yuyv422"},
            {"width": 1920, "height": 1080, "pixel_format": "yuyv422"},
        ]
        def _op():
            frame = self._capture.read_with_fallbacks(
                timeout_seconds=self._config.capture_read_timeout_seconds,
                fallback_specs=fallback_specs,
            )
            if frame is None:
                raise RuntimeError(str(self._capture.last_error or "unknown error"))
            return frame

        return self.run_frame_api_operation(_op, "Capture frame")

    @property
    def capture_recovery_success_count(self) -> int:
        return int(self._capture_recovery_success_count)

    def _save_debug_snapshot(self, frame, payload: Dict[str, Any]) -> None:
        if not self._config.save_debug_frames:
            return
        latest_png = self._tmp_dir / "latest_frame.png"
        latest_json = self._tmp_dir / "latest_analysis.json"
        cv2.imwrite(str(latest_png), frame)
        latest_json.write_text(json.dumps(_json_safe(payload), ensure_ascii=False, indent=2), encoding="utf-8")
        self.last_frame_path = str(latest_png)
        self.last_analysis_path = str(latest_json)

    def _save_playable_capture(self, frame) -> None:
        # Temporarily disabled:
        # do not generate extra playable-triggered screenshots for now.
        # if not self._config.save_debug_frames:
        #     return
        # self._playable_shot_index += 1
        # path = _unique_debug_image_path(self._debug_dir, "capture", self._playable_shot_index)
        # cv2.imwrite(str(path), frame)
        return

    def save_settlement_poll_capture(self, frame) -> None:
        # Temporarily disabled:
        # do not generate extra settlement polling screenshots for now.
        # if not self._config.save_debug_frames:
        #     return
        # self._settlement_shot_index += 1
        # path = _unique_debug_image_path(self._debug_dir, "settlement_poll", self._settlement_shot_index)
        # cv2.imwrite(str(path), frame)
        return

    def detect_playable(self) -> Dict[str, Any]:
        frame = self._read_latest_frame()
        result = detect_playable_banner(frame)
        self.last_sp_count = int(get_sp_count_frame(frame))
        self.last_enemy_sp_count = int(get_enemy_sp_count_frame(frame))
        result["frame_shape"] = [int(frame.shape[0]), int(frame.shape[1]), int(frame.shape[2])]
        result["p1_sp"] = int(self.last_sp_count)
        result["p2_sp"] = int(self.last_enemy_sp_count)
        # if result.get("playable"):
        #     self._save_playable_capture(frame)
        self._save_debug_snapshot(
            frame,
            {
                "kind": "playable_poll",
                "playable_result": result,
                "p1_sp": int(self.last_sp_count),
                "p2_sp": int(self.last_enemy_sp_count),
            },
        )
        return result

    def inspect_wait_frame(self):
        frame = self._read_latest_frame()
        playable_result = detect_playable_banner(frame)
        lose_result = detect_lose_banner(frame)
        win_result = detect_win_banner(frame)
        draw_result = detect_draw_banner(frame)
        self.last_sp_count = int(get_sp_count_frame(frame))
        self.last_enemy_sp_count = int(get_enemy_sp_count_frame(frame))
        # if playable_result.get("playable"):
        #     self._save_playable_capture(frame)
        self._save_debug_snapshot(
            frame,
            {
                "kind": "wait_poll",
                "playable_result": playable_result,
                "lose_result": lose_result,
                "win_result": win_result,
                "draw_result": draw_result,
                "p1_sp": int(self.last_sp_count),
                "p2_sp": int(self.last_enemy_sp_count),
            },
        )
        return frame, playable_result, lose_result, win_result, draw_result

    def _detect_map_identity(self, frame) -> Dict[str, Any]:
        return detect_map_from_frame(frame, layout=self._layout)

    def _detect_map_identity_stable(self, first_frame) -> Dict[str, Any]:
        attempts: List[Dict[str, Any]] = []
        frames = [first_frame]
        for _ in range(2):
            time.sleep(0.08)
            frames.append(self._read_latest_frame())
        for frame in frames:
            result = dict(self._detect_map_identity(frame))
            attempts.append(result)

        best_by_map: Dict[str, Dict[str, Any]] = {}
        for idx, item in enumerate(attempts):
            map_id = str(item.get("map_id", "") or "")
            score = float(item.get("score", 0.0) or 0.0)
            if not map_id:
                continue
            slot = best_by_map.setdefault(
                map_id,
                {"count": 0, "best_score": -1.0, "best": item, "attempt_indices": []},
            )
            slot["count"] = int(slot["count"]) + 1
            slot["attempt_indices"].append(idx)
            if score > float(slot["best_score"]):
                slot["best_score"] = score
                slot["best"] = item

        if not best_by_map:
            fallback = attempts[0] if attempts else {}
            fallback["stability_attempts"] = attempts
            return fallback

        ranked = sorted(
            best_by_map.values(),
            key=lambda row: (int(row["count"]), float(row["best_score"])),
            reverse=True,
        )
        chosen = dict(ranked[0]["best"])
        chosen["stability_attempts"] = attempts
        chosen["stability_vote_count"] = int(ranked[0]["count"])
        chosen["stability_best_score"] = float(ranked[0]["best_score"])
        return chosen

    def _collect_recent_frames(self, first_frame, sample_count: int = 5, poll_interval_seconds: float = 0.08) -> List[np.ndarray]:
        frames = [first_frame]
        while len(frames) < max(1, int(sample_count)):
            time.sleep(max(0.01, float(poll_interval_seconds)))
            frames.append(self._read_latest_frame())
        return frames

    def _supplemental_state(self, frame, analysis_result: Dict[str, Any], turn_index: int) -> SupplementalState:
        if self._supplemental_provider is not None:
            payload = self._supplemental_provider(
                frame=frame,
                analysis_result=analysis_result,
                turn_index=turn_index,
                map_id=self._map_id,
            )
            if payload is None:
                return SupplementalState()
            if not isinstance(payload, dict):
                raise ValueError("supplemental_state_provider must return a dict or None")
            return SupplementalState.from_payload(payload)
        return SupplementalState(
            selected_hand_index=self._config.manual_fields.selected_hand_index,
            cursor_xy=self._config.manual_fields.cursor_xy,
            rotation=self._config.manual_fields.rotation,
            p1_sp=self._config.manual_fields.p1_sp,
        )

    def _apply_p1_special_history(self, raw_labels: List[List[str]]) -> List[List[str]]:
        if not bool(self._config.preserve_p1_special_history):
            return raw_labels
        cleared_labels = {
            "empty",
            "changed",
            "transparent",
        }
        out = [list(row) for row in raw_labels]
        for y, row in enumerate(out):
            for x, label in enumerate(row):
                pos = (int(x), int(y))
                if label in {"p1_special", "p1_special_activated"}:
                    self._persisted_p1_special_positions.add(pos)
                    self._persisted_p2_special_positions.discard(pos)
                    self._persisted_conflict_positions.discard(pos)
                    continue
                if label in {"p2_special", "p2_special_activated"}:
                    self._persisted_p1_special_positions.discard(pos)
                    self._persisted_p2_special_positions.add(pos)
                    self._persisted_conflict_positions.discard(pos)
                    continue
                if label == "conflict":
                    self._persisted_p1_special_positions.discard(pos)
                    self._persisted_p2_special_positions.discard(pos)
                    self._persisted_conflict_positions.add(pos)
                    continue
                if pos in self._persisted_conflict_positions:
                    if label == "p1_fill":
                        out[y][x] = "conflict"
                        continue
                    if label in cleared_labels:
                        self._persisted_conflict_positions.discard(pos)
                        # continue evaluating possible p1_special history cleanup below
                if pos in self._persisted_p1_special_positions:
                    if label == "p1_fill":
                        out[y][x] = "p1_special"
                        continue
                    if label in cleared_labels:
                        self._persisted_p1_special_positions.discard(pos)
                if pos in self._persisted_p2_special_positions:
                    if label == "p2_fill":
                        out[y][x] = "p2_special"
                        continue
                    if label in cleared_labels:
                        self._persisted_p2_special_positions.discard(pos)
        return out

    def remember_executed_specials(self, action: Any) -> None:
        if not bool(self._config.preserve_p1_special_history):
            return
        if bool(getattr(action, "pass_turn", False)) or bool(getattr(action, "surrender", False)):
            return
        if getattr(action, "x", None) is None or getattr(action, "y", None) is None:
            return
        try:
            card = create_card_from_id(int(action.card_number))
            cells = _card_cells_on_map(card, int(action.x), int(action.y), int(action.rotation or 0))
        except Exception:
            return
        for x, y, cell_type in cells:
            if int(cell_type) == 2:
                self._persisted_p1_special_positions.add((int(x - MAP_PADDING), int(y - MAP_PADDING)))

    def parse_turn_state(self, turn_index: int) -> ParsedTurnState:
        frame = self._read_latest_frame()
        playable_result = detect_playable_banner(frame)
        map_match: Dict[str, Any] = {}
        if self._map_id is None:
            map_match = dict(self._detect_map_identity_stable(frame))
            self._map_id = str(map_match.get("map_id", "") or "")
            self._map_name = str(map_match.get("map_name_zh", "") or "")
            if self._map_id:
                map_match = _pad_map_match_payload(self._map_id, map_match)
            if self._map_name:
                self._map_tracker = MapStateTracker(self._map_name)
        if not self._map_id:
            raise MissingInterfaceError(["map_id(tableturf_vision map match failed)"])
        if not self._map_name:
            name_map = {str(v): str(k) for k, v in map_name_cn_to_id().items()}
            self._map_name = name_map.get(self._map_id, "")
        if not self._map_name:
            raise MissingInterfaceError(["map_name(tableturf_vision map name missing)"])
        if self._map_tracker is None:
            self._map_tracker = MapStateTracker(self._map_name)

        hand_result = detect_hand_cards(frame)
        ordered_slots = _normalize_hand_slots(hand_result)
        card_matches = []
        hand_card_numbers: List[int] = []
        for slot in ordered_slots:
            match = _match_card(slot["matrix"])
            card_matches.append(
                {
                    "slot": slot.get("slot"),
                    "counts": slot.get("counts"),
                    "matrix": slot.get("matrix"),
                    "match": match,
                }
            )
            if match.get("number") is not None:
                hand_card_numbers.append(int(match["number"]))
        if len(hand_card_numbers) != 4:
            raise MissingInterfaceError(["hand_card_numbers(card recognition incomplete)"])

        self.last_sp_count = int(get_sp_count_frame(frame))
        self.last_enemy_sp_count = int(get_enemy_sp_count_frame(frame))
        raw_map_state_result = detect_map_state(frame, self._map_name)
        state_frame_count = (
            max(1, int(self._config.map_state_frame_count_playable))
            if bool(playable_result.get("playable"))
            else max(1, int(self._config.map_state_frame_count_default))
        )
        state_frame_count_used = 1
        if str(self._config.frame_api_url or "").strip() and state_frame_count > 1:
            frame_api_timeout_seconds = max(
                3.0,
                float(state_frame_count) / max(1.0, float(self._config.capture_fps))
                + float(MAP_STATE_MULTI_FRAME_TIMEOUT_MARGIN_SECONDS),
            )
            map_state_result = self.run_frame_api_operation(
                lambda: self._map_tracker.update_frame_api(
                    frame_url=str(self._config.frame_api_url).strip(),
                    sample_count=int(state_frame_count),
                    poll_interval_seconds=MAP_STATE_MULTI_FRAME_POLL_INTERVAL_SECONDS,
                    timeout_seconds=frame_api_timeout_seconds,
                ),
                "Map-state frame API",
            )
            state_frame_count_used = int(map_state_result.get("frame_count_used", state_frame_count) or state_frame_count)
        elif state_frame_count > 1:
            state_frames = self._collect_recent_frames(
                frame,
                sample_count=state_frame_count,
                poll_interval_seconds=MAP_STATE_MULTI_FRAME_POLL_INTERVAL_SECONDS,
            )
            map_state_result = self._map_tracker.update_frames(state_frames)
            state_frame_count_used = len(state_frames)
        else:
            map_state_result = self._map_tracker.update_frame(frame)
            state_frame_count_used = 1
        board_labels_raw = self._apply_p1_special_history(_map_state_to_board_labels(self._map_id, map_state_result))
        board_labels = _pad_board_labels_to_engine_dims(self._map_id, board_labels_raw)
        engine_map_grid = _extract_board_grid(board_labels)
        analysis_result = {
            "kind": "turn_state",
            "map_match": map_match or {"map_id": self._map_id, "map_name_zh": self._map_name},
            "raw_map_state": raw_map_state_result,
            "map_state": map_state_result,
            "map_state_frame_count": int(state_frame_count_used),
            "map_state_frame_count_target": int(state_frame_count),
            "board": {
                "raw_labels": board_labels_raw,
                "labels": board_labels,
            },
            "hand_cards": {
                "slots": ordered_slots,
            },
            "sp": {
                "p1_sp": int(self.last_sp_count),
                "p2_sp": int(self.last_enemy_sp_count),
            },
        }
        self._save_debug_snapshot(frame, analysis_result)

        supplemental = self._supplemental_state(frame, analysis_result, turn_index)
        missing: List[str] = []
        selected_hand_index = supplemental.selected_hand_index if supplemental.selected_hand_index is not None else 0
        rotation = supplemental.rotation if supplemental.rotation is not None else 0
        p1_sp = supplemental.p1_sp if supplemental.p1_sp is not None else int(self.last_sp_count)
        cursor_xy = supplemental.cursor_xy if supplemental.cursor_xy is not None else _initial_ui_anchor_for_map(
            ObservedState(
                map_id=self._map_id,
                hand_card_numbers=hand_card_numbers,
                p1_sp=int(p1_sp),
                turn=turn_index,
                map_grid=engine_map_grid,
            )
        )
        if missing and self._config.strict_missing_interfaces:
            raise MissingInterfaceError(missing)

        return ParsedTurnState(
            map_id=self._map_id,
            map_name=self._map_name,
            hand_card_numbers=hand_card_numbers,
            map_grid=engine_map_grid,
            playable_result=playable_result,
            analysis_result=analysis_result,
            card_matches=card_matches,
            turn=turn_index,
            selected_hand_index=int(selected_hand_index),
            cursor_xy=(int(cursor_xy[0]), int(cursor_xy[1])),
            rotation=int(rotation),
            p1_sp=int(p1_sp),
            p2_sp=int(self.last_enemy_sp_count),
        )


def load_config(path: str) -> ControllerConfig:
    cfg_path = Path(path)
    if not cfg_path.is_absolute():
        cfg_path = REPO_ROOT / cfg_path
    return ControllerConfig.from_json(cfg_path)


def apply_clone_jelly_profile(config: ControllerConfig) -> ControllerConfig:
    config.strategy_id = "module:autocontroller_clone_jelly_strategy"
    config.strategy_id_by_map = {}
    config.strategy_id_by_map_name = {}
    config.policy_config_json = ""
    config.progress_timeout_seconds = 30.0
    config.enable_battle_replay = False
    config.stats_mode = "aggregate"
    config.aggregate_stats_name = "clone_jelly"
    config.aggregate_stats_file = "autocontroller_rebuild_for_RL/clone_jelly_state.json"
    config.log_file = "autocontroller_rebuild_for_RL/debug_runtime/log/clone_jelly.log"
    config.map_state_frame_count_default = 1
    config.map_state_frame_count_playable = 1
    config.settlement_frame_count = 1
    config.preserve_p1_special_history = True
    config.clone_jelly_split_action_execution = False
    config.clone_jelly_sp_attack_up_right = False
    return config
