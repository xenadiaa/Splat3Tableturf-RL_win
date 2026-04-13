from __future__ import annotations

from typing import Optional

from switch_connect.strategy_mapper import choose_action_from_observation
from vision_capture.state_types import ObservedState


def choose_action_engine(
    obs: ObservedState,
    style: Optional[str] = None,
    level: str = "high",
):
    return choose_action_from_observation(obs=obs, style=style, level=level)
