from __future__ import annotations

from pathlib import Path
import sys
from typing import Optional, Tuple

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tableturf_sim.src.engine.env_core import _choose_bot_action, step

if __package__:
    from .rl_env import EnvStep, TableturfRLEnv
else:
    from rl_env import EnvStep, TableturfRLEnv  # type: ignore


class OpeningAggressiveTableturfEnv(TableturfRLEnv):
    """
    Warm-start env:
    - first `opening_turns` full turns are played by an aggressive heuristic for P1
    - P2 remains the normal 1P-mode bot throughout
    - PPO only starts from the post-opening state
    """

    def __init__(
        self,
        map_id: str = "Square",
        p1_deck_selector: Optional[str] = None,
        p2_deck_selector: Optional[str] = None,
        bot_style: str = "aggressive",
        bot_level: str = "high",
        seed: Optional[int] = 42,
        opening_turns: int = 3,
    ) -> None:
        super().__init__(
            map_id=map_id,
            p1_deck_selector=p1_deck_selector,
            p2_deck_selector=p2_deck_selector,
            bot_style=bot_style,
            bot_level=bot_level,
            seed=seed,
        )
        self.opening_turns = max(0, int(opening_turns))

    def reset(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        map_obs, scalar_obs, action_feats = super().reset()
        if self.state is None:
            raise RuntimeError("Environment not reset")

        for _ in range(self.opening_turns):
            if self.state.done:
                break
            action = _choose_bot_action(self.state, "P1")
            ok, reason, result = step(self.state, action)
            if not ok:
                raise RuntimeError(f"Opening aggressive action failed: {reason} {result}")

        map_obs, scalar_obs = self._encode_state()
        action_feats = self._encode_actions() if not self.state.done else np.zeros((1, self.action_feature_dim), dtype=np.float32)
        return map_obs, scalar_obs, action_feats

    def step(self, action_index: int) -> EnvStep:
        return super().step(action_index)
