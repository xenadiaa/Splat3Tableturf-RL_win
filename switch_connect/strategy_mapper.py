from __future__ import annotations

import random
from typing import Dict, Optional

from src.engine.env_core import GameState, PlayerState, _score_bot_action, legal_actions
from src.engine.loaders import load_map
from src.utils.common_utils import create_card_from_id
from vision_capture.state_types import ObservedState


DEFAULT_STYLE_BY_MAP: Dict[str, str] = {
    "Square": "aggressive",
    "ManySp": "conservative",
}


def _build_state_from_observation(obs: ObservedState) -> GameState:
    game_map = load_map(obs.map_id)
    if obs.map_grid is not None:
        if len(obs.map_grid) != game_map.height or any(len(r) != game_map.width for r in obs.map_grid):
            raise ValueError("map_grid size mismatch with map_id")
        game_map.grid = [row[:] for row in obs.map_grid]

    p1_hand = [create_card_from_id(n) for n in obs.hand_card_numbers]
    p1 = PlayerState(deck_ids=[], draw_pile=[], hand=p1_hand, sp=obs.p1_sp)
    p2 = PlayerState(deck_ids=[], draw_pile=[], hand=[], sp=0)
    return GameState(map=game_map, players={"P1": p1, "P2": p2}, turn=obs.turn)


def choose_action_from_observation(
    obs: ObservedState,
    style: Optional[str] = None,
    level: str = "high",
    rng_seed: Optional[int] = None,
):
    """Reuse existing env_core legal-actions and scoring on observed state."""
    if level not in {"low", "mid", "high"}:
        raise ValueError("level must be one of: low, mid, high")

    style_use = style or DEFAULT_STYLE_BY_MAP.get(obs.map_id, "balanced")
    if style_use not in {"balanced", "aggressive", "conservative"}:
        raise ValueError("style must be one of: balanced, aggressive, conservative")

    state = _build_state_from_observation(obs)
    actions = legal_actions(state, "P1")
    if not actions:
        raise RuntimeError("no legal actions generated from observation")

    scored = [(float(_score_bot_action(state, "P1", a, style_use, level)), a) for a in actions]
    scored.sort(key=lambda x: x[0], reverse=True)
    if level == "high":
        return scored[0][1]

    rng = random.Random(rng_seed)
    if level == "mid":
        top_k = max(1, len(scored) // 5)
    else:
        top_k = max(1, len(scored) // 2)
    return rng.choice([a for _, a in scored[:top_k]])
