from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import random
import sys
from typing import Dict, List, Optional, Tuple

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

try:
    from tableturf_sim.src.assets.types import Card_Single, Map_PointBit
except Exception:
    from tableturf_sim.src.assets.tableturf_types import Card_Single, Map_PointBit
from tableturf_sim.src.engine.env_core import Action, GameState, init_state, legal_actions, step

if __package__:
    from .rl_env import _max_map_size_with_padding, resolve_deck_numbers
else:
    from rl_env import _max_map_size_with_padding, resolve_deck_numbers  # type: ignore


@dataclass
class PendingActionResult:
    done: bool
    resolved: bool
    reward_p1: float
    reward_p2: float
    info: Dict[str, object]


class TableturfSelfPlayEnv:
    """Two-player environment wrapper for shared-policy PPO self-play."""

    def __init__(
        self,
        map_id: str,
        p1_deck_selector: str,
        p2_deck_selector: str,
        seed: Optional[int] = 42,
    ) -> None:
        self.map_id = map_id
        self.p1_deck_rowid = p1_deck_selector
        self.p2_deck_rowid = p2_deck_selector
        self.p1_deck_ids = resolve_deck_numbers(p1_deck_selector, fallback_index=0)
        self.p2_deck_ids = resolve_deck_numbers(p2_deck_selector, fallback_index=1)
        self.base_seed = seed
        self._rng = random.Random(seed)
        self.max_w, self.max_h = _max_map_size_with_padding()
        self.state: Optional[GameState] = None

    @property
    def map_shape(self) -> Tuple[int, int, int]:
        return (6, self.max_h, self.max_w)

    @property
    def scalar_dim(self) -> int:
        return 6

    @property
    def action_feature_dim(self) -> int:
        return 12

    def reset(self) -> None:
        seed = self._rng.randint(0, 2**31 - 1) if self.base_seed is not None else None
        self.state = init_state(
            map_id=self.map_id,
            p1_deck_ids=self.p1_deck_ids,
            p2_deck_ids=self.p2_deck_ids,
            seed=seed,
            mode="2P",
            log_path="/dev/null",
        )

    def observe(self, player: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[Action]]:
        if self.state is None:
            raise RuntimeError("Environment not reset")
        map_obs, scalar_obs = self._encode_state(player)
        actions, action_features = self._encode_actions(player)
        return map_obs, scalar_obs, action_features, actions

    def submit(self, action: Action, prev_score_diff: float) -> PendingActionResult:
        if self.state is None:
            raise RuntimeError("Environment not reset")
        ok, reason, result = step(self.state, action)
        if not ok:
            return PendingActionResult(
                done=False,
                resolved=False,
                reward_p1=-1.0 if action.player == "P1" else 0.0,
                reward_p2=-1.0 if action.player == "P2" else 0.0,
                info={"ok": False, "reason": reason, "result": result},
            )
        resolved = reason == "TURN_RESOLVED"
        if not resolved:
            return PendingActionResult(
                done=False,
                resolved=False,
                reward_p1=0.0,
                reward_p2=0.0,
                info={"ok": True, "reason": reason, "result": result},
            )

        new_diff = self.score_diff()
        reward_p1 = (new_diff - prev_score_diff) / 10.0 - 0.01
        reward_p2 = -reward_p1
        done = bool(self.state.done)
        if done:
            if self.state.winner == "P1":
                reward_p1 += 1.0
                reward_p2 -= 1.0
            elif self.state.winner == "P2":
                reward_p1 -= 1.0
                reward_p2 += 1.0
        return PendingActionResult(
            done=done,
            resolved=True,
            reward_p1=float(reward_p1),
            reward_p2=float(reward_p2),
            info={"ok": True, "reason": reason, "result": result, "winner": self.state.winner if done else ""},
        )

    def score_diff(self) -> float:
        p1, p2 = self._compute_scores()
        return float(p1 - p2)

    def _encode_state(self, player: str) -> Tuple[np.ndarray, np.ndarray]:
        if self.state is None:
            raise RuntimeError("Environment not reset")
        obs = np.zeros((6, self.max_h, self.max_w), dtype=np.float32)
        own = "P1" if player == "P1" else "P2"
        opp = "P2" if player == "P1" else "P1"
        game_map = self.state.map
        for y in range(game_map.height):
            for x in range(game_map.width):
                m = int(game_map.get_point(x, y))
                obs[0, y, x] = 1.0 if (m & int(Map_PointBit.IsValid)) else 0.0
                obs[1, y, x] = 1.0 if self._has_owner(m, own) else 0.0
                obs[2, y, x] = 1.0 if self._has_owner(m, opp) else 0.0
                obs[3, y, x] = 1.0 if (m & int(Map_PointBit.IsSp)) else 0.0
                obs[4, y, x] = 1.0 if (m & int(Map_PointBit.IsSupplySp)) else 0.0
                obs[5, y, x] = 1.0 if self._has_owner(m, "P1") and self._has_owner(m, "P2") else 0.0

        own_score, opp_score = self._compute_scores_perspective(player)
        scalar = np.array(
            [
                self.state.turn / max(1, self.state.max_turns),
                self.state.players[own].sp / 20.0,
                self.state.players[opp].sp / 20.0,
                (own_score - opp_score) / 100.0,
                len(self.state.players[own].draw_pile) / 15.0,
                len(self.state.players[opp].draw_pile) / 15.0,
            ],
            dtype=np.float32,
        )
        return obs, scalar

    def _encode_actions(self, player: str) -> Tuple[List[Action], np.ndarray]:
        if self.state is None:
            raise RuntimeError("Environment not reset")
        actions = legal_actions(self.state, player)
        if not actions:
            return [], np.zeros((1, self.action_feature_dim), dtype=np.float32)
        hand = self.state.players[player].hand
        hand_index = {card.Number: idx for idx, card in enumerate(hand)}
        feats: List[np.ndarray] = []
        for action in actions:
            card = next((c for c in hand if c.Number == action.card_number), None)
            if card is None:
                continue
            cell_count, sp_count = self._card_cell_stats(card, action.rotation)
            x_norm = (action.x / max(1, self.max_w - 1)) if action.x is not None else 0.0
            y_norm = (action.y / max(1, self.max_h - 1)) if action.y is not None else 0.0
            feats.append(
                np.array(
                    [
                        hand_index[card.Number] / 3.0,
                        1.0 if action.pass_turn else 0.0,
                        1.0 if action.use_sp_attack else 0.0,
                        action.rotation / 3.0,
                        x_norm,
                        y_norm,
                        card.CardPoint / 20.0,
                        card.SpecialCost / 10.0,
                        cell_count / 64.0,
                        sp_count / 64.0,
                        self.state.players[player].sp / 20.0,
                        self.state.turn / max(1, self.state.max_turns),
                    ],
                    dtype=np.float32,
                )
            )
        if len(feats) != len(actions):
            raise RuntimeError("Failed to encode one or more legal actions")
        return actions, np.stack(feats, axis=0)

    def _card_cell_stats(self, card: Card_Single, rotation: int) -> Tuple[int, int]:
        matrix = card.get_square_matrix(rotation)
        cell_count = 0
        sp_count = 0
        for row in matrix:
            for value in row:
                if value != 0:
                    cell_count += 1
                if value == 2:
                    sp_count += 1
        return cell_count, sp_count

    def _compute_scores(self) -> Tuple[int, int]:
        if self.state is None:
            raise RuntimeError("Environment not reset")
        p1 = 0
        p2 = 0
        game_map = self.state.map
        for y in range(game_map.height):
            for x in range(game_map.width):
                m = int(game_map.get_point(x, y))
                if (m & int(Map_PointBit.IsValid)) == 0:
                    continue
                is_p1 = self._has_owner(m, "P1")
                is_p2 = self._has_owner(m, "P2")
                if is_p1 and not is_p2:
                    p1 += 1
                elif is_p2 and not is_p1:
                    p2 += 1
        return p1, p2

    def _compute_scores_perspective(self, player: str) -> Tuple[int, int]:
        p1, p2 = self._compute_scores()
        return (p1, p2) if player == "P1" else (p2, p1)

    def _has_owner(self, mask: int, player: str) -> bool:
        bit = Map_PointBit.IsP1 if player == "P1" else Map_PointBit.IsP2
        return (mask & int(bit)) != 0
