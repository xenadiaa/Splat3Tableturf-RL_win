from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

from vision_capture.state_types import ObservedState


BIT_A = 0
BIT_B = 1
BIT_X = 2
BIT_Y = 3
BIT_DPAD_UP = 22
BIT_DPAD_DOWN = 23
BIT_DPAD_LEFT = 24
BIT_DPAD_RIGHT = 25


@dataclass
class RemoteStep:
    bits: int
    hold_ms: int = 90
    gap_ms: int = 40


def _bit(i: int) -> int:
    return 1 << i


def _press_button(steps: List[RemoteStep], bit_index: int, hold_ms: int = 90, gap_ms: int = 40) -> None:
    steps.append(RemoteStep(bits=_bit(bit_index), hold_ms=hold_ms, gap_ms=gap_ms))


def _move_axis(steps: List[RemoteStep], dx: int, dy: int, move_hold_ms: int = 80) -> None:
    bit_map: Dict[str, int] = {
        "left": BIT_DPAD_LEFT,
        "right": BIT_DPAD_RIGHT,
        "up": BIT_DPAD_UP,
        "down": BIT_DPAD_DOWN,
    }
    if dx > 0:
        for _ in range(dx):
            _press_button(steps, bit_map["right"], hold_ms=move_hold_ms, gap_ms=30)
    elif dx < 0:
        for _ in range(-dx):
            _press_button(steps, bit_map["left"], hold_ms=move_hold_ms, gap_ms=30)
    if dy > 0:
        for _ in range(dy):
            _press_button(steps, bit_map["down"], hold_ms=move_hold_ms, gap_ms=30)
    elif dy < 0:
        for _ in range(-dy):
            _press_button(steps, bit_map["up"], hold_ms=move_hold_ms, gap_ms=30)


def compile_action_to_remote_steps(action, obs: ObservedState) -> List[RemoteStep]:
    """Compile selected action into remote button presses (0xFF-mode)."""
    if obs.selected_hand_index is None or obs.cursor_xy is None:
        raise ValueError("selected_hand_index and cursor_xy are required for mapping")

    hand = obs.hand_card_numbers
    if action.card_number not in hand:
        raise ValueError(f"card {action.card_number} not in observed hand {hand}")

    steps: List[RemoteStep] = []
    target_idx = hand.index(action.card_number)
    delta_idx = target_idx - obs.selected_hand_index
    _move_axis(steps, dx=delta_idx, dy=0)

    rot_delta = (action.rotation - obs.rotation) % 4
    for _ in range(rot_delta):
        _press_button(steps, BIT_Y)

    if action.pass_turn:
        _press_button(steps, BIT_B)
        return steps

    if action.x is None or action.y is None:
        raise ValueError("non-pass action requires x/y")

    dx = int(action.x) - int(obs.cursor_xy[0])
    dy = int(action.y) - int(obs.cursor_xy[1])
    _move_axis(steps, dx=dx, dy=dy)

    if action.use_sp_attack:
        _press_button(steps, BIT_X)
    _press_button(steps, BIT_A, hold_ms=110, gap_ms=60)
    return steps
