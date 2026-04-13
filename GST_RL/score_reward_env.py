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
from tableturf_sim.src.utils.common_utils import _card_cells_on_map

if __package__:
    from .rl_env import _max_map_size_with_padding, auto_deck_rowids_for_map, resolve_deck_numbers
else:
    from rl_env import _max_map_size_with_padding, auto_deck_rowids_for_map, resolve_deck_numbers  # type: ignore


@dataclass
class ScoreRewardStep:
    map_obs: np.ndarray
    scalar_obs: np.ndarray
    action_features: np.ndarray
    done: bool
    reward: float
    info: Dict[str, object]


class TableturfScoreRewardEnv:
    """Single-agent env with score-first reward and SP misuse penalty."""

    def __init__(
        self,
        map_id: str = "Square",
        p1_deck_selector: Optional[str] = None,
        p2_deck_selector: Optional[str] = None,
        bot_style: str = "aggressive",
        bot_level: str = "high",
        seed: Optional[int] = 42,
    ) -> None:
        self.map_id = map_id
        auto_p1_rowid, auto_p2_rowid = auto_deck_rowids_for_map(self.map_id)
        p1_pick = p1_deck_selector or auto_p1_rowid
        p2_pick = p2_deck_selector or auto_p2_rowid
        self.p1_deck_rowid = p1_pick
        self.p2_deck_rowid = p2_pick
        self.p1_deck_ids = resolve_deck_numbers(p1_pick, fallback_index=0)
        self.p2_deck_ids = resolve_deck_numbers(p2_pick, fallback_index=1)
        self.bot_style = bot_style
        self.bot_level = bot_level
        self.base_seed = seed
        self._rng = random.Random(seed)
        self.max_w, self.max_h = _max_map_size_with_padding()
        self.state: Optional[GameState] = None
        self._cached_actions: List[Action] = []

    @property
    def map_shape(self) -> Tuple[int, int, int]:
        return (6, self.max_h, self.max_w)

    @property
    def scalar_dim(self) -> int:
        return 6

    @property
    def action_feature_dim(self) -> int:
        return 12

    def reset(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        seed = self._rng.randint(0, 2**31 - 1) if self.base_seed is not None else None
        self.state = init_state(
            map_id=self.map_id,
            p1_deck_ids=self.p1_deck_ids,
            p2_deck_ids=self.p2_deck_ids,
            seed=seed,
            mode="1P",
            bot_style=self.bot_style,
            bot_level=self.bot_level,
            log_path="/dev/null",
        )
        map_obs, scalar_obs = self._encode_state()
        action_feats = self._encode_actions()
        return map_obs, scalar_obs, action_feats

    def step(self, action_index: int) -> ScoreRewardStep:
        if self.state is None:
            raise RuntimeError("Environment not reset")
        if action_index < 0 or action_index >= len(self._cached_actions):
            raise IndexError(f"action index out of range: {action_index}")

        action = self._cached_actions[action_index]
        prev_p1, prev_p2 = self._compute_scores()
        prev_objective = self._score_objective(prev_p1, prev_p2)
        sp_overlap = self._sp_overlap_counts(action)

        ok, reason, result = step(self.state, action)
        if not ok:
            map_obs, scalar_obs = self._encode_state()
            action_feats = self._encode_actions()
            return ScoreRewardStep(
                map_obs=map_obs,
                scalar_obs=scalar_obs,
                action_features=action_feats,
                done=False,
                reward=-1.0,
                info={"ok": False, "reason": reason, "result": result},
            )

        done = bool(self.state.done)
        p1_score, p2_score = self._compute_scores()
        new_objective = self._score_objective(p1_score, p2_score)
        reward = (new_objective - prev_objective) * 10.0

        own_overlap = sp_overlap["own_overlap"]
        enemy_overlap = sp_overlap["enemy_overlap"]
        sp_penalty = 0.0
        if action.use_sp_attack and own_overlap > enemy_overlap:
            sp_penalty = 0.06 * float(own_overlap - enemy_overlap)
            reward -= sp_penalty

        terminal = 0.0
        if done:
            if self.state.winner == "P1":
                terminal = 0.05
            elif self.state.winner == "P2":
                terminal = -0.05
            reward += terminal

        map_obs, scalar_obs = self._encode_state()
        action_feats = self._encode_actions() if not done else np.zeros((1, self.action_feature_dim), dtype=np.float32)
        return ScoreRewardStep(
            map_obs=map_obs,
            scalar_obs=scalar_obs,
            action_features=action_feats,
            done=done,
            reward=float(reward),
            info={
                "ok": True,
                "reason": reason,
                "result": result,
                "winner": self.state.winner if done else "",
                "p1_score": p1_score,
                "p2_score": p2_score,
                "score_objective": new_objective,
                "sp_overlap": sp_overlap,
                "sp_penalty": sp_penalty,
                "terminal_bonus": terminal,
            },
        )

    def _encode_state(self) -> Tuple[np.ndarray, np.ndarray]:
        if self.state is None:
            raise RuntimeError("Environment not reset")
        obs = np.zeros((6, self.max_h, self.max_w), dtype=np.float32)
        game_map = self.state.map
        for y in range(game_map.height):
            for x in range(game_map.width):
                m = int(game_map.get_point(x, y))
                obs[0, y, x] = 1.0 if (m & int(Map_PointBit.IsValid)) else 0.0
                obs[1, y, x] = 1.0 if (m & int(Map_PointBit.IsP1)) else 0.0
                obs[2, y, x] = 1.0 if (m & int(Map_PointBit.IsP2)) else 0.0
                obs[3, y, x] = 1.0 if (m & int(Map_PointBit.IsSp)) else 0.0
                obs[4, y, x] = 1.0 if (m & int(Map_PointBit.IsSupplySp)) else 0.0
                obs[5, y, x] = 1.0 if ((m & int(Map_PointBit.IsP1)) and (m & int(Map_PointBit.IsP2))) else 0.0
        p1_score, p2_score = self._compute_scores()
        scalar = np.array(
            [
                self.state.turn / max(1, self.state.max_turns),
                self.state.players["P1"].sp / 20.0,
                self.state.players["P2"].sp / 20.0,
                (p1_score - p2_score) / 100.0,
                len(self.state.players["P1"].draw_pile) / 15.0,
                len(self.state.players["P2"].draw_pile) / 15.0,
            ],
            dtype=np.float32,
        )
        return obs, scalar

    def _encode_actions(self) -> np.ndarray:
        if self.state is None:
            raise RuntimeError("Environment not reset")
        self._cached_actions = legal_actions(self.state, "P1")
        if not self._cached_actions:
            return np.zeros((1, self.action_feature_dim), dtype=np.float32)
        hand = self.state.players["P1"].hand
        hand_index = {card.Number: idx for idx, card in enumerate(hand)}
        feats: List[np.ndarray] = []
        for action in self._cached_actions:
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
                        self.state.players["P1"].sp / 20.0,
                        self.state.turn / max(1, self.state.max_turns),
                    ],
                    dtype=np.float32,
                )
            )
        if len(feats) != len(self._cached_actions):
            raise RuntimeError("Failed to encode one or more legal actions")
        return np.stack(feats, axis=0)

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
                is_p1 = (m & int(Map_PointBit.IsP1)) != 0
                is_p2 = (m & int(Map_PointBit.IsP2)) != 0
                if is_p1 and not is_p2:
                    p1 += 1
                elif is_p2 and not is_p1:
                    p2 += 1
        return p1, p2

    def _score_objective(self, p1_score: int, p2_score: int) -> float:
        return 0.7 * (p1_score / 100.0) + 0.25 * ((p1_score - p2_score) / 100.0)

    def _sp_overlap_counts(self, action: Action) -> Dict[str, int]:
        if self.state is None:
            raise RuntimeError("Environment not reset")
        if action.pass_turn or not action.use_sp_attack or action.x is None or action.y is None:
            return {"own_overlap": 0, "enemy_overlap": 0, "empty_overlap": 0}
        card = next((c for c in self.state.players["P1"].hand if c.Number == action.card_number), None)
        if card is None:
            return {"own_overlap": 0, "enemy_overlap": 0, "empty_overlap": 0}
        own_overlap = 0
        enemy_overlap = 0
        empty_overlap = 0
        for x, y, _cell_type in _card_cells_on_map(card, int(action.x), int(action.y), action.rotation):
            if not (0 <= x < self.state.map.width and 0 <= y < self.state.map.height):
                continue
            m = int(self.state.map.get_point(x, y))
            if (m & int(Map_PointBit.IsP1)) != 0:
                own_overlap += 1
            elif (m & int(Map_PointBit.IsP2)) != 0:
                enemy_overlap += 1
            else:
                empty_overlap += 1
        return {"own_overlap": own_overlap, "enemy_overlap": enemy_overlap, "empty_overlap": empty_overlap}

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
