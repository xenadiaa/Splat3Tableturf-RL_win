"""Example automated player strategy: choose a random legal action."""

from __future__ import annotations

from typing import Dict, List


def choose_action(state, player: str, legal_actions: List[dict], context: Dict[str, object]) -> Dict[str, object]:
    if not legal_actions:
        raise RuntimeError("legal_actions is empty")
    idx = state.rng.randrange(len(legal_actions))
    chosen = legal_actions[idx]
    return {
        "card_number": chosen.get("card_number"),
        "pass_turn": bool(chosen.get("pass_turn", False)),
        "use_sp_attack": bool(chosen.get("use_sp_attack", False)),
        "rotation": int(chosen.get("rotation", 0)),
        "x": chosen.get("x"),
        "y": chosen.get("y"),
    }
