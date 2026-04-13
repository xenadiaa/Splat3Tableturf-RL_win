from __future__ import annotations

from typing import Dict, List


def infer(observed_state: Dict[str, object]) -> Dict[str, object]:
    """
    Example NN-like policy API.
    Replace this with real model inference.
    Input: ObservedState dict
    Output: Action dict
    """
    hand: List[int] = list(observed_state.get("hand_card_numbers", []))  # type: ignore[arg-type]
    if not hand:
        return {"player": "P1", "pass_turn": True, "card_number": None}

    # Naive fallback: pick first card, keep current cursor as target.
    cursor = observed_state.get("cursor_xy") or [0, 0]
    x = int(cursor[0])
    y = int(cursor[1])
    return {
        "player": "P1",
        "card_number": int(hand[0]),
        "pass_turn": False,
        "use_sp_attack": False,
        "rotation": int(observed_state.get("rotation", 0)),
        "x": x,
        "y": y,
    }

