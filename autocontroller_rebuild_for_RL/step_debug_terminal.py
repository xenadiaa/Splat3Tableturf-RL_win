from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List, Tuple

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from autocontroller_rebuild_for_RL.runtime import (
    BIT_A,
    ControllerConfig,
    MissingInterfaceError,
    MAP_PADDING,
    RemoteStep,
    _FrameVisionPipeline,
    _action_target_ui_xy,
    _append_alternating_axis_csv,
    _append_card_selection_csv,
    _engine_xy_to_ui_xy,
    _resolve_serial_port,
    choose_action_from_resolved_strategy,
    compile_action_map_phase_csv,
    compile_action_menu_selection_steps,
    compile_action_to_runtime_steps,
    compile_action_with_defaults,
    load_config,
    resolve_strategy,
)
from src.assets.tableturf_types import Map_PointBit
from src.utils.common_utils import create_card_from_id
from src.utils.localization import lookup_card_name_zh
from tableturf_vision.mapper_preview import CLR_RESET
from switch_connect.virtual_gamepad.serial_controller import SerialRemoteController
from switch_connect.virtual_gamepad.input_mapper import REMOTE_INPUT_BITS


BIT_NAMES: Dict[int, str] = {
    bit_index: name for name, bit_index in REMOTE_INPUT_BITS.items()
}

VISION_RGB: Dict[str, Tuple[int, int, int]] = {
    "empty": (0, 0, 0),
    "p1_fill": (255, 255, 0),
    "p1_special": (255, 192, 0),
    "p2_fill": (0, 112, 192),
    "p2_special": (0, 176, 240),
    "conflict": (128, 128, 128),
    "p1_special_activated": (255, 192, 0),
    "p2_special_activated": (0, 176, 240),
    "planned_fill": (80, 255, 120),
    "planned_special": (0, 255, 180),
    "planned_anchor": (255, 64, 64),
}

VISION_TOKEN_MAP: Dict[str, str] = {
    "invalid": "  ",
    "empty": "[]",
    "p1_fill": "[]",
    "p1_special": "[]",
    "p2_fill": "[]",
    "p2_special": "[]",
    "conflict": "[]",
    "p1_special_activated": "/\\",
    "p2_special_activated": "/\\",
    "planned_fill": "[]",
    "planned_special": "[]",
    "planned_anchor": "<>",
}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Single-step terminal debugger for Tableturf auto controller")
    parser.add_argument(
        "--config",
        default="autocontroller_rebuild_for_RL/runtime_config.example.json",
        help="config json path",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=0,
        help="0 means no limit",
    )
    return parser.parse_args()


def _wait_for_enter(prompt: str) -> str:
    try:
        return input(prompt).strip().lower()
    except EOFError:
        return "q"


def _print_block(title: str, payload) -> None:
    print(f"\n=== {title} ===")
    if isinstance(payload, str):
        print(payload)
        return
    print(json.dumps(payload, ensure_ascii=False, indent=2))


def _build_wait_a_step(config: ControllerConfig) -> List[RemoteStep]:
    return [RemoteStep(bits=(1 << BIT_A), hold_ms=config.wait_press_hold_ms, gap_ms=config.wait_press_gap_ms)]


def _rgb(text: str, rgb: Tuple[int, int, int]) -> str:
    r, g, b = rgb
    return f"\033[38;2;{r};{g};{b}m{text}{CLR_RESET}"


def _mask_to_label(mask: int) -> str:
    mask_i = int(mask)
    if (mask_i & int(Map_PointBit.IsValid)) == 0:
        return "invalid"
    is_p1 = (mask_i & int(Map_PointBit.IsP1)) != 0
    is_p2 = (mask_i & int(Map_PointBit.IsP2)) != 0
    is_sp = (mask_i & int(Map_PointBit.IsSp)) != 0
    is_supply = (mask_i & int(Map_PointBit.IsSupplySp)) != 0
    if is_p1 and is_p2:
        return "conflict"
    if is_p1 and is_sp:
        return "p1_special_activated"
    if is_p2 and is_sp:
        return "p2_special_activated"
    if is_p1:
        return "p1_fill"
    if is_p2:
        return "p2_fill"
    if is_supply:
        return "empty"
    return "empty"


def _label_to_token(label: str) -> str:
    token = VISION_TOKEN_MAP.get(label, "??")
    if token.strip() == "":
        return token
    return _rgb(token, VISION_RGB.get(label, (255, 255, 255)))


def _normalize_board_label(label: str) -> str:
    if label == "transparent":
        return "empty"
    return str(label)


def _trim_padding_grid(grid: List[List[int]]) -> List[List[int]]:
    if not grid:
        return grid
    height = len(grid)
    width = len(grid[0]) if height else 0
    if height <= MAP_PADDING * 2 or width <= MAP_PADDING * 2:
        return grid
    return [row[MAP_PADDING : width - MAP_PADDING] for row in grid[MAP_PADDING : height - MAP_PADDING]]


def _format_board_lines(grid: List[List[int]]) -> List[str]:
    grid_use = _trim_padding_grid(grid)
    if not grid_use:
        return ["<empty board>"]
    width = len(grid_use[0])
    header = "   " + " ".join(f"{x:02d}" for x in range(width))
    lines = [header]
    for y, row in enumerate(grid_use):
        lines.append(f"{y:02d} " + " ".join(_label_to_token(_mask_to_label(cell)) for cell in row))
    return lines


def _format_board_label_lines(labels_2d: List[List[str]]) -> List[str]:
    if not labels_2d:
        return ["<empty board>"]
    width = len(labels_2d[0])
    header = "   " + " ".join(f"{x:02d}" for x in range(width))
    lines = [header]
    for y, row in enumerate(labels_2d):
        lines.append(f"{y:02d} " + " ".join(_label_to_token(_normalize_board_label(label)) for label in row))
    return lines


def _overlay_action_preview(grid: List[List[int]], card_number: int, rotation: int, x: int, y: int) -> List[str]:
    grid_use = _trim_padding_grid(grid)
    if not grid_use:
        return ["<empty board>"]
    overlay = [[_mask_to_label(cell) for cell in row] for row in grid_use]
    card = create_card_from_id(int(card_number))
    matrix = card.get_square_matrix(int(rotation))
    link_x, link_y = card.get_link_pos(int(rotation))
    ui_x = int(x) - MAP_PADDING
    ui_y = int(y) - MAP_PADDING
    top_left_x = int(ui_x) - int(link_x)
    top_left_y = int(ui_y) - int(link_y)

    for row_idx, row in enumerate(matrix):
        for col_idx, value in enumerate(row):
            if int(value) == 0:
                continue
            board_x = top_left_x + col_idx
            board_y = top_left_y + row_idx
            if board_y < 0 or board_y >= len(overlay):
                continue
            if board_x < 0 or board_x >= len(overlay[board_y]):
                continue
            overlay[board_y][board_x] = "planned_special" if int(value) == 2 else "planned_fill"
    if 0 <= ui_y < len(overlay) and 0 <= ui_x < len(overlay[ui_y]):
        overlay[ui_y][ui_x] = "planned_anchor"

    width = len(overlay[0])
    header = "   " + " ".join(f"{col:02d}" for col in range(width))
    lines = [header]
    for row_idx, row in enumerate(overlay):
        lines.append(f"{row_idx:02d} " + " ".join(_label_to_token(label) for label in row))
    return lines


def _card_preview_lines(card_number: int, rotation: int) -> List[str]:
    card = create_card_from_id(int(card_number))
    matrix = card.get_square_matrix(int(rotation))
    link_x, link_y = card.get_link_pos(int(rotation))
    lines: List[str] = [f"card={card_number} name={_card_display_name(card_number)} rotation={rotation} link_pos={link_x, link_y}"]
    for row_idx, row in enumerate(matrix):
        parts: List[str] = []
        for col_idx, value in enumerate(row):
            if col_idx == link_x and row_idx == link_y and int(value) == 0:
                parts.append("@")
            elif col_idx == link_x and row_idx == link_y and int(value) == 2:
                parts.append("$")
            elif col_idx == link_x and row_idx == link_y:
                parts.append("@")
            elif int(value) == 2:
                parts.append("*")
            elif int(value) != 0:
                parts.append("#")
            else:
                parts.append(".")
        lines.append("".join(parts))
    return lines


def _print_lines_block(title: str, lines: List[str]) -> None:
    print(f"\n=== {title} ===")
    for line in lines:
        print(line)


def _board_legend_lines() -> List[str]:
    return [
        f"{_label_to_token('empty')} empty",
        f"{_label_to_token('p1_fill')} p1_fill",
        f"{_label_to_token('p1_special_activated')} p1_special_activated (shows as /\\)",
        f"{_label_to_token('p2_fill')} p2_fill",
        f"{_label_to_token('p2_special_activated')} p2_special_activated (shows as /\\)",
        f"{_label_to_token('conflict')} conflict",
        f"{_label_to_token('planned_fill')} planned_fill",
        f"{_label_to_token('planned_special')} planned_special",
        f"{_label_to_token('planned_anchor')} planned_anchor",
        "   invalid",
    ]


def _card_display_name(card_number: int) -> str:
    card = create_card_from_id(int(card_number))
    zh = lookup_card_name_zh(getattr(card, "name", "")) or ""
    if zh:
        return zh
    if getattr(card, "name", ""):
        return str(card.name)
    return f"Card#{card_number}"


def _hand_detection_rows(hand_card_numbers: List[int]) -> List[Dict[str, object]]:
    slot_names = ["left_top", "right_top", "left_bottom", "right_bottom"]
    rows: List[Dict[str, object]] = []
    for idx, card_number in enumerate(hand_card_numbers):
        rows.append(
            {
                "slot": slot_names[idx] if idx < len(slot_names) else f"slot_{idx}",
                "card_number": int(card_number),
                "card_name": _card_display_name(int(card_number)),
            }
        )
    return rows


def _action_summary(action, state) -> Dict[str, object]:
    card_name = _card_display_name(int(action.card_number)) if action.card_number is not None else ""
    action_kind = "surrender" if action.surrender else "pass_turn" if action.pass_turn else "sp_attack" if action.use_sp_attack else "normal_place"
    summary = {
        "action_kind": action_kind,
        "card_number": action.card_number,
        "card_name": card_name,
        "pass_turn": bool(action.pass_turn),
        "use_sp_attack": bool(action.use_sp_attack),
        "surrender": bool(action.surrender),
        "rotation": int(action.rotation),
    }
    if action.x is not None and action.y is not None:
        summary["engine_xy"] = [int(action.x), int(action.y)]
        summary["ui_xy"] = list(_engine_xy_to_ui_xy(int(action.x), int(action.y), state.map_id, state.map_grid))
    return summary


def _compact_card_matches(card_matches: List[Dict[str, object]]) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    for item in card_matches:
        match = item.get("match", {}) if isinstance(item.get("match"), dict) else {}
        rows.append(
            {
                "slot": item.get("slot"),
                "counts": item.get("counts"),
                "match_number": match.get("number"),
                "match_name": _card_display_name(int(match["number"])) if match.get("number") is not None else "",
                "match_rotation": match.get("rotation"),
                "match_score": match.get("score"),
            }
        )
    return rows


def _step_to_name(step: RemoteStep) -> str:
    for bit, name in BIT_NAMES.items():
        if step.bits & (1 << bit):
            return name
    return f"bits={step.bits}"


def _action_phase_breakdown(action, state, obs) -> Dict[str, object]:
    card_number = int(action.card_number) if action.card_number is not None else None
    hand = list(obs.hand_card_numbers or [])
    selection_steps: List[str] = []
    rotation_steps: List[str] = []
    map_move_steps: List[str] = []
    confirm_steps: List[str] = []

    if action.surrender:
        return {
            "selection_phase": ["Plus", "DPadRight", "A"],
            "rotation_phase": [],
            "map_move_phase": [],
            "confirm_phase": [],
        }

    if action.pass_turn:
        selection_steps.extend(["DPadDown", "DPadDown", "A"])
        if card_number in hand:
            idx = hand.index(card_number)
            selection_steps.extend(_selection_path_names(0, idx))
        selection_steps.append("A")
        return {
            "selection_phase": selection_steps,
            "rotation_phase": [],
            "map_move_phase": [],
            "confirm_phase": [],
        }

    if action.use_sp_attack:
        selection_steps.extend(["DPadDown", "DPadDown", "DPadRight", "A"])
        sp_pool = [int(x) for x in obs.hand_card_numbers if x is not None]
        if card_number in sp_pool and sp_pool:
            start_idx = hand.index(sp_pool[0]) if sp_pool[0] in hand else 0
            target_idx = hand.index(card_number) if card_number in hand else start_idx
            selection_steps.extend(_selection_path_names(start_idx, target_idx))
        selection_steps.append("A")
    else:
        if card_number in hand:
            idx = hand.index(card_number)
            selection_steps.extend(_selection_path_names(int(obs.selected_hand_index or 0), idx))
        selection_steps.append("A")

    cw_steps = int(action.rotation) % 4
    ccw_steps = (-int(action.rotation)) % 4
    if cw_steps <= ccw_steps:
        rotation_steps.extend(["X"] * cw_steps)
    else:
        rotation_steps.extend(["Y"] * ccw_steps)

    if action.x is not None and action.y is not None:
        target_x, target_y = _action_target_ui_xy(action, obs)
        cursor_x, cursor_y = state.cursor_xy if state.cursor_xy is not None else (0, 0)
        dx = int(target_x) - int(cursor_x)
        dy = int(target_y) - int(cursor_y)
        map_move_steps.extend(_axis_path_names(dx, dy))
        confirm_steps.append("A")

    return {
        "selection_phase": selection_steps,
        "rotation_phase": rotation_steps,
        "map_move_phase": map_move_steps,
        "confirm_phase": confirm_steps,
    }


def _selection_path_names(from_index: int, to_index: int) -> List[str]:
    parts: List[str] = []
    _append_card_selection_csv(parts, from_index, to_index)
    return _tokens_to_names(parts)


def divmod_like_card(index: int) -> Tuple[int, int]:
    idx = max(0, int(index))
    return (idx % 2, idx // 2)


def _axis_path_names(dx: int, dy: int) -> List[str]:
    parts: List[str] = []
    _append_alternating_axis_csv(parts, dx, "DRIGHT", "LRIGHT", "DLEFT", "LLEFT")
    _append_alternating_axis_csv(parts, dy, "DDOWN", "LDOWN", "DUP", "LUP")
    return _tokens_to_names(parts)


def _tokens_to_names(parts: List[str]) -> List[str]:
    token_names = {
        "DUP": "DPadUp",
        "DDOWN": "DPadDown",
        "DLEFT": "DPadLeft",
        "DRIGHT": "DPadRight",
        "LUP": "LeftStickUp",
        "LDOWN": "LeftStickDown",
        "LLEFT": "LeftStickLeft",
        "LRIGHT": "LeftStickRight",
        "A": "A",
        "B": "B",
        "X": "X",
        "Y": "Y",
        "PLUS": "Plus",
        "LOOP": "Loop",
        "NOTHING": "Nothing",
    }
    names: List[str] = []
    for idx in range(0, len(parts), 2):
        token = str(parts[idx]).upper()
        duration = str(parts[idx + 1]) if idx + 1 < len(parts) else ""
        label = token_names.get(token, token)
        names.append(f"{label}x{duration}" if token == "LOOP" else label)
    return names


def main() -> int:
    args = _parse_args()
    config = load_config(args.config)
    serial_port = _resolve_serial_port(config.serial_port, config.pick_serial)
    vision = _FrameVisionPipeline(config)
    controller = SerialRemoteController(port=serial_port)
    turn_index = 1
    step_count = 0
    wait_a_enabled = True

    print("单步调试终端已启动。每一步都会先展示识别结果、策略输入输出、即将发送的按键。")
    print("按回车执行本步，输入 q 后回车退出。")
    _print_lines_block("图例", _board_legend_lines())

    try:
        while True:
            if args.max_steps and step_count >= args.max_steps:
                print("\n达到 max-steps，停止。")
                return 0

            playable_result = vision.detect_playable()
            _print_block(
                "当前检测",
                {
                    "playable": bool(playable_result.get("playable")),
                    "sp_count": vision.last_sp_count,
                    "frame_shape": playable_result.get("frame_shape"),
                    "last_frame_path": vision.last_frame_path,
                    "last_analysis_path": vision.last_analysis_path,
                    "turn_index": turn_index,
                },
            )

            if not playable_result.get("playable"):
                if wait_a_enabled:
                    steps = _build_wait_a_step(config)
                    _print_block("即将执行", {"kind": "wait_playable_press_a_once"})
                else:
                    steps = []
                    _print_block("即将执行", {"kind": "silent_wait_no_press_a"})
                cmd = _wait_for_enter("回车执行本步，q 退出：")
                if cmd == "q":
                    return 0
                if steps:
                    controller.run_steps(steps)
                    result_kind = "wait_playable_press_a_once"
                else:
                    result_kind = "silent_wait_no_press_a"
                _print_block("本步结果", {"executed": True, "kind": result_kind})
                cmd = _wait_for_enter("回车继续下一步，q 退出：")
                if cmd == "q":
                    return 0
                step_count += 1
                continue

            state = vision.parse_turn_state(turn_index=turn_index)
            wait_a_enabled = False
            observed_state = state.to_observed_state()
            resolved_strategy = resolve_strategy(config, state.map_id, state.map_name)
            action = choose_action_from_resolved_strategy(observed_state, resolved_strategy)
            battle_steps = compile_action_to_runtime_steps(action, observed_state)

            _print_block(
                "识别输入/输出",
                {
                    "map_id": state.map_id,
                    "map_name": state.map_name,
                    "turn": state.turn,
                    "p1_sp": state.p1_sp,
                    "hand_card_numbers": state.hand_card_numbers,
                    "selected_hand_index": state.selected_hand_index,
                    "cursor_xy": state.cursor_xy,
                    "rotation": state.rotation,
                    "card_matches": _compact_card_matches(state.card_matches),
                    "last_frame_path": vision.last_frame_path,
                    "last_analysis_path": vision.last_analysis_path,
                },
            )
            _print_block("手牌检测结果", _hand_detection_rows(state.hand_card_numbers))
            board_raw_labels = []
            tracker_board_labels = []
            if isinstance(state.analysis_result.get("board"), dict):
                raw_payload = state.analysis_result["board"].get("raw_labels")
                if isinstance(raw_payload, list):
                    tracker_board_labels = raw_payload
            if isinstance(state.analysis_result.get("raw_map_state"), dict):
                cells = state.analysis_result["raw_map_state"].get("cells")
                if isinstance(cells, list):
                    raw_grid: Dict[Tuple[int, int], str] = {}
                    max_row = -1
                    max_col = -1
                    for cell in cells:
                        row = int(cell.get("json_row", -1))
                        col = int(cell.get("json_col", -1))
                        label = str(cell.get("label", "empty"))
                        if row >= 0 and col >= 0:
                            raw_grid[(row, col)] = label
                            max_row = max(max_row, row)
                            max_col = max(max_col, col)
                    if max_row >= 0 and max_col >= 0:
                        board_raw_labels = [
                            [raw_grid.get((r, c), "invalid") for c in range(max_col + 1)]
                            for r in range(max_row + 1)
                        ]
            if board_raw_labels:
                _print_lines_block("原始地图直观图", _format_board_label_lines(board_raw_labels))
            if tracker_board_labels:
                _print_lines_block("修正后地图直观图", _format_board_label_lines(tracker_board_labels))
            if not board_raw_labels and not tracker_board_labels:
                _print_lines_block("当前地图直观图", _format_board_lines(state.map_grid))
            _print_block(
                "策略输出",
                {
                    "strategy_label": resolved_strategy.label,
                    "strategy_mode": resolved_strategy.mode,
                    "strategy_source": resolved_strategy.source,
                    "strategy_id": resolved_strategy.strategy_id,
                    "checkpoint_file": resolved_strategy.checkpoint_file,
                    "player": action.player,
                    "card_number": action.card_number,
                    "pass_turn": action.pass_turn,
                    "use_sp_attack": action.use_sp_attack,
                    "rotation": action.rotation,
                    "x": action.x,
                    "y": action.y,
                    "ui_xy": (
                        _engine_xy_to_ui_xy(int(action.x), int(action.y), state.map_id, state.map_grid)
                        if action.x is not None and action.y is not None
                        else None
                    ),
                },
            )
            _print_block("本回合策略摘要", _action_summary(action, state))
            _print_block("动作拆解", _action_phase_breakdown(action, state, observed_state))
            if not action.pass_turn and not action.surrender and action.x is not None and action.y is not None and action.card_number is not None:
                _print_lines_block(
                    "将要放置的卡牌",
                    _card_preview_lines(int(action.card_number), int(action.rotation)),
                )
                _print_lines_block(
                    "策略落点直观图",
                    _overlay_action_preview(state.map_grid, int(action.card_number), int(action.rotation), int(action.x), int(action.y)),
                )
                _print_block(
                    "落点坐标说明",
                    {
                        "engine_xy": [int(action.x), int(action.y)],
                        "ui_xy": list(_action_target_ui_xy(action, observed_state)),
                        "cursor_ui_xy": list(state.cursor_xy) if state.cursor_xy is not None else None,
                    },
                )
            cmd = _wait_for_enter("回车执行本步，q 退出：")
            if cmd == "q":
                return 0

            if bool(config.clone_jelly_split_action_execution):
                selection_steps = compile_action_menu_selection_steps(
                    action,
                    observed_state,
                    sp_attack_up_right=bool(config.clone_jelly_sp_attack_up_right),
                )
                if selection_steps:
                    controller.run_steps(selection_steps)
                map_phase_csv = compile_action_map_phase_csv(action, observed_state)
                if map_phase_csv:
                    time.sleep(0.1)
                    controller.send_smart_sequence_csv_blocking(map_phase_csv, timeout_seconds=15.0)
            else:
                controller.run_steps(battle_steps)
            _print_block(
                "本步结果",
                {
                    "executed": True,
                    "turn_index": turn_index,
                    "strategy_label": resolved_strategy.label,
                    "strategy_mode": resolved_strategy.mode,
                    "action_card": action.card_number,
                    "pass_turn": action.pass_turn,
                    "use_sp_attack": action.use_sp_attack,
                },
            )

            turn_index += 1
            if turn_index > config.max_turns:
                turn_index = 1
                wait_a_enabled = True
                _print_block("对战循环", {"battle_reset": True, "next_turn_index": turn_index})

            cmd = _wait_for_enter("回车继续下一步，q 退出：")
            if cmd == "q":
                return 0
            step_count += 1
    except MissingInterfaceError as exc:
        _print_block("缺失接口", {"missing_fields": exc.missing_fields})
        return 2
    finally:
        vision.close()
        controller.close()


if __name__ == "__main__":
    raise SystemExit(main())
