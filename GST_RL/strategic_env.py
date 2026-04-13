from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import math
from pathlib import Path
import random
import sys
from typing import Dict, Iterable, List, Optional, Set, Tuple

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
    from .rl_env import (
        _max_map_size_with_padding,
        auto_deck_rowids_for_map,
        level_npc_profiles_for_map,
        list_map_ids,
        map_name_by_id,
        player_map_deck_selector,
        resolve_deck_numbers,
    )
else:
    from rl_env import (  # type: ignore
        _max_map_size_with_padding,
        auto_deck_rowids_for_map,
        level_npc_profiles_for_map,
        list_map_ids,
        map_name_by_id,
        player_map_deck_selector,
        resolve_deck_numbers,
    )


STYLE_MAP = {
    "Aggressive": "aggressive",
    "Balance": "balanced",
    "AccumulateSpecial": "conservative",
}


@dataclass
class StrategicEnvStep:
    map_obs: np.ndarray
    scalar_obs: np.ndarray
    action_features: np.ndarray
    done: bool
    reward: float
    info: Dict[str, object]


class StrategicTableturfEnv:
    """A separate env wrapper with situation-dependent shaping rewards."""

    def __init__(
        self,
        map_id: str = "ManySp",
        p1_deck_selector: Optional[str] = None,
        p2_deck_selector: Optional[str] = None,
        bot_style: str = "balanced",
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
        return 14

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
        metrics = self._compute_metrics()
        map_obs, scalar_obs = self._encode_state(metrics)
        action_feats = self._encode_actions()
        return map_obs, scalar_obs, action_feats

    def step(self, action_index: int) -> StrategicEnvStep:
        if self.state is None:
            raise RuntimeError("Environment not reset")
        if action_index < 0 or action_index >= len(self._cached_actions):
            raise IndexError(f"action index out of range: {action_index}")

        prev_metrics = self._compute_metrics()
        prev_legal = len(self._cached_actions)
        action = self._cached_actions[action_index]
        prev_card = self._card_from_action(action)

        ok, reason, result = step(self.state, action)
        if not ok:
            metrics = self._compute_metrics()
            map_obs, scalar_obs = self._encode_state(metrics)
            action_feats = self._encode_actions()
            return StrategicEnvStep(
                map_obs=map_obs,
                scalar_obs=scalar_obs,
                action_features=action_feats,
                done=False,
                reward=-1.0,
                info={"ok": False, "reason": reason, "result": result},
            )

        done = bool(self.state.done)
        post_metrics = self._compute_metrics()
        map_obs, scalar_obs = self._encode_state(post_metrics)
        action_feats = self._encode_actions() if not done else np.zeros((1, self.action_feature_dim), dtype=np.float32)
        post_legal = len(self._cached_actions) if not done else 0

        reward, components, phase_weights = self._strategic_reward(
            prev_metrics=prev_metrics,
            post_metrics=post_metrics,
            action=action,
            card=prev_card,
            prev_legal=prev_legal,
            post_legal=post_legal,
            done=done,
        )
        info: Dict[str, object] = {
            "ok": True,
            "reason": reason,
            "result": result,
            "reward_components": components,
            "phase_weights": phase_weights,
            "metrics_before": self._metrics_summary(prev_metrics),
            "metrics_after": self._metrics_summary(post_metrics),
        }
        if done:
            info["winner"] = self.state.winner
        return StrategicEnvStep(
            map_obs=map_obs,
            scalar_obs=scalar_obs,
            action_features=action_feats,
            done=done,
            reward=float(reward),
            info=info,
        )

    def _encode_state(self, metrics: Optional[Dict[str, float]] = None) -> Tuple[np.ndarray, np.ndarray]:
        if self.state is None:
            raise RuntimeError("Environment not reset")
        if metrics is None:
            metrics = self._compute_metrics()

        obs = np.zeros((6, self.max_h, self.max_w), dtype=np.float32)
        game_map = self.state.map
        for y in range(game_map.height):
            for x in range(game_map.width):
                m = int(game_map.get_point(x, y))
                obs[0, y, x] = 1.0 if self._is_valid(m) else 0.0
                obs[1, y, x] = 1.0 if (m & int(Map_PointBit.IsP1)) else 0.0
                obs[2, y, x] = 1.0 if (m & int(Map_PointBit.IsP2)) else 0.0
                obs[3, y, x] = 1.0 if (m & int(Map_PointBit.IsSp)) else 0.0
                obs[4, y, x] = 1.0 if (m & int(Map_PointBit.IsSupplySp)) else 0.0
                obs[5, y, x] = 1.0 if ((m & int(Map_PointBit.IsP1)) and (m & int(Map_PointBit.IsP2))) else 0.0

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

    def _strategic_reward(
        self,
        prev_metrics: Dict[str, float],
        post_metrics: Dict[str, float],
        action: Action,
        card: Optional[Card_Single],
        prev_legal: int,
        post_legal: int,
        done: bool,
    ) -> Tuple[float, Dict[str, float], Dict[str, float]]:
        phase = self._phase_weights(prev_metrics)
        base_score = (post_metrics["score_diff"] - prev_metrics["score_diff"]) / 10.0 - 0.01

        delta_p1_reach = max(0.0, post_metrics["p1_reachable_empty"] - prev_metrics["p1_reachable_empty"])
        delta_p1_largest = max(0.0, post_metrics["p1_largest_reachable"] - prev_metrics["p1_largest_reachable"])
        delta_p2_reach = max(0.0, prev_metrics["p2_reachable_empty"] - post_metrics["p2_reachable_empty"])
        delta_p2_largest = max(0.0, prev_metrics["p2_largest_reachable"] - post_metrics["p2_largest_reachable"])
        occupancy_potential = math.log1p(post_metrics["p1_largest_reachable"]) / math.log1p(max(2.0, post_metrics["valid_cells"]))

        breakthrough = 0.018 * delta_p1_reach * (0.7 + occupancy_potential) + 0.028 * delta_p1_largest
        if delta_p1_reach > 0 and post_legal > prev_legal:
            breakthrough += 0.012 * float(post_legal - prev_legal)

        compression = 0.018 * delta_p2_reach + 0.025 * delta_p2_largest
        if prev_metrics["p2_largest_locked"] > post_metrics["p2_largest_locked"]:
            compression += 0.01 * (prev_metrics["p2_largest_locked"] - post_metrics["p2_largest_locked"])

        fortify = 0.035 * max(0.0, prev_metrics["enemy_breach_risk"] - post_metrics["enemy_breach_risk"])
        if prev_metrics["p2_sp"] < 3:
            fortify *= 0.4

        sp_setup = 0.0
        for threshold, bonus in ((3, 0.10), (6, 0.18)):
            if prev_metrics["p1_sp"] < threshold <= post_metrics["p1_sp"]:
                sp_setup += bonus
        if not action.use_sp_attack and post_metrics["p1_sp"] >= 3 and phase["finish"] < 0.45:
            sp_setup += 0.02 * max(0.0, post_metrics["own_breach_chance"] - prev_metrics["own_breach_chance"])

        sp_spent = max(0.0, prev_metrics["p1_sp"] - post_metrics["p1_sp"])
        enemy_area_removed = max(0.0, prev_metrics["p2_score"] - post_metrics["p2_score"])
        own_area_gained = max(0.0, post_metrics["p1_score"] - prev_metrics["p1_score"])
        sp_attack = 0.0
        if action.use_sp_attack:
            sp_attack += 0.050 * enemy_area_removed
            sp_attack += 0.025 * own_area_gained
            sp_attack += 0.018 * delta_p1_reach
            if sp_spent > 0:
                sp_attack -= 0.012 * sp_spent
            net_recovery = max(0.0, post_metrics["p1_sp"] - (prev_metrics["p1_sp"] - sp_spent))
            sp_attack += 0.020 * net_recovery
            if sp_spent >= 6 and enemy_area_removed >= 4:
                sp_attack += 0.15
            elif sp_spent >= 3 and enemy_area_removed >= 2 and delta_p2_reach > 0:
                sp_attack += 0.08

        card_pressure = 0.0
        if card is not None and not action.pass_turn and not action.use_sp_attack:
            cell_count, sp_count = self._card_cell_stats(card, action.rotation)
            if cell_count >= 8 and (compression > 0 or breakthrough > 0):
                card_pressure += 0.02 + 0.005 * sp_count

        weighted_pressure = phase["pressure"] * (compression + breakthrough + card_pressure)
        weighted_build = phase["build"] * (0.60 * breakthrough + fortify + sp_setup)
        weighted_finish = phase["finish"] * (0.40 * compression + 0.50 * fortify + sp_attack)

        terminal = 0.0
        if done and self.state is not None:
            if self.state.winner == "P1":
                terminal = 1.0
            elif self.state.winner == "P2":
                terminal = -1.0

        reward = base_score + weighted_pressure + weighted_build + weighted_finish + terminal
        components = {
            "base_score": float(base_score),
            "breakthrough": float(breakthrough),
            "compression": float(compression),
            "fortify": float(fortify),
            "sp_setup": float(sp_setup),
            "sp_attack": float(sp_attack),
            "card_pressure": float(card_pressure),
            "weighted_pressure": float(weighted_pressure),
            "weighted_build": float(weighted_build),
            "weighted_finish": float(weighted_finish),
            "terminal": float(terminal),
            "total": float(reward),
        }
        return reward, components, phase

    def _phase_weights(self, metrics: Dict[str, float]) -> Dict[str, float]:
        turn = metrics["turn_ratio"]
        valid = max(1.0, metrics["valid_cells"])
        pressure = max(0.05, 1.1 - 1.6 * turn)
        build = 0.25 + max(0.0, 1.0 - abs(turn - 0.55) / 0.30)
        finish = max(0.05, (turn - 0.58) * 2.3)

        compression_live = (metrics["p2_reachable_empty"] / valid) > 0.10 or metrics["enemy_breach_risk"] > 1.0
        breakthrough_live = (metrics["p1_largest_locked"] / valid) > 0.12 and metrics["p1_reachable_empty"] < metrics["p1_largest_locked"]
        if compression_live:
            pressure += 0.35
        else:
            build += 0.25
        if breakthrough_live:
            pressure += 0.20
            build += 0.10
        if metrics["p1_sp"] < 3:
            build += 0.15
        if metrics["p1_sp"] >= 3 and (turn > 0.68 or metrics["own_breach_chance"] > 0.8):
            finish += 0.45
        if turn > 0.82:
            finish += 0.75

        total = pressure + build + finish
        return {
            "pressure": pressure / total,
            "build": build / total,
            "finish": finish / total,
        }

    def _compute_metrics(self) -> Dict[str, float]:
        if self.state is None:
            raise RuntimeError("Environment not reset")
        valid_cells = 0
        empty_cells = 0
        p1_cells: Set[Tuple[int, int]] = set()
        p2_cells: Set[Tuple[int, int]] = set()
        for x, y, m in self._iter_map_cells():
            if not self._is_valid(m):
                continue
            valid_cells += 1
            if self._has_owner(m, "P1"):
                p1_cells.add((x, y))
            if self._has_owner(m, "P2"):
                p2_cells.add((x, y))
            if self._is_empty_cell(m):
                empty_cells += 1

        p1_reach, p1_largest, p1_locked = self._reachable_empty_stats("P1")
        p2_reach, p2_largest, p2_locked = self._reachable_empty_stats("P2")
        p1_score, p2_score = self._compute_scores()
        metrics = {
            "valid_cells": float(valid_cells),
            "empty_cells": float(empty_cells),
            "turn_ratio": self.state.turn / max(1, self.state.max_turns),
            "p1_sp": float(self.state.players["P1"].sp),
            "p2_sp": float(self.state.players["P2"].sp),
            "p1_score": float(p1_score),
            "p2_score": float(p2_score),
            "score_diff": float(p1_score - p2_score),
            "p1_reachable_empty": float(p1_reach),
            "p2_reachable_empty": float(p2_reach),
            "p1_largest_reachable": float(p1_largest),
            "p2_largest_reachable": float(p2_largest),
            "p1_largest_locked": float(p1_locked),
            "p2_largest_locked": float(p2_locked),
            "p1_frontier": float(self._frontier_count("P1")),
            "p2_frontier": float(self._frontier_count("P2")),
            "enemy_breach_risk": float(self._sp_breach_risk(attacker="P2", defender="P1")),
            "own_breach_chance": float(self._sp_breach_risk(attacker="P1", defender="P2")),
            "p1_draw_ratio": len(self.state.players["P1"].draw_pile) / 15.0,
            "p2_draw_ratio": len(self.state.players["P2"].draw_pile) / 15.0,
        }
        return metrics

    def _iter_map_cells(self) -> Iterable[Tuple[int, int, int]]:
        if self.state is None:
            raise RuntimeError("Environment not reset")
        game_map = self.state.map
        for y in range(game_map.height):
            for x in range(game_map.width):
                yield x, y, int(game_map.get_point(x, y))

    def _reachable_empty_stats(self, player: str) -> Tuple[int, int, int]:
        if self.state is None:
            raise RuntimeError("Environment not reset")
        reachable_total = 0
        largest_reachable = 0
        largest_locked = 0
        visited: Set[Tuple[int, int]] = set()
        game_map = self.state.map

        for y in range(game_map.height):
            for x in range(game_map.width):
                if (x, y) in visited:
                    continue
                m = int(game_map.get_point(x, y))
                if not self._is_empty_cell(m):
                    continue
                queue = deque([(x, y)])
                visited.add((x, y))
                component: List[Tuple[int, int]] = []
                touches_player = False
                while queue:
                    cx, cy = queue.popleft()
                    component.append((cx, cy))
                    for nx, ny in self._neighbors4(cx, cy):
                        if nx < 0 or ny < 0 or nx >= game_map.width or ny >= game_map.height:
                            continue
                        nm = int(game_map.get_point(nx, ny))
                        if self._has_owner(nm, player):
                            touches_player = True
                        if (nx, ny) in visited or not self._is_empty_cell(nm):
                            continue
                        visited.add((nx, ny))
                        queue.append((nx, ny))
                size = len(component)
                if touches_player:
                    reachable_total += size
                    largest_reachable = max(largest_reachable, size)
                else:
                    largest_locked = max(largest_locked, size)
        return reachable_total, largest_reachable, largest_locked

    def _frontier_count(self, player: str) -> int:
        if self.state is None:
            raise RuntimeError("Environment not reset")
        count = 0
        other = "P2" if player == "P1" else "P1"
        for x, y, m in self._iter_map_cells():
            if not self._has_owner(m, player):
                continue
            for nx, ny in self._neighbors4(x, y):
                if nx < 0 or ny < 0 or nx >= self.state.map.width or ny >= self.state.map.height:
                    continue
                nm = int(self.state.map.get_point(nx, ny))
                if self._is_empty_cell(nm) or self._has_owner(nm, other):
                    count += 1
                    break
        return count

    def _sp_breach_risk(self, attacker: str, defender: str) -> float:
        if self.state is None:
            raise RuntimeError("Environment not reset")
        attacker_sp = float(self.state.players[attacker].sp)
        if attacker_sp < 3:
            return 0.0
        attack_scale = 2.0 if attacker_sp >= 6 else 1.0
        risk = 0.0
        for x, y, m in self._iter_map_cells():
            if not self._has_owner(m, defender):
                continue
            if not self._is_frontier_cell(x, y, defender):
                continue
            adjacent_enemy_sp = False
            for nx, ny in self._neighbors4(x, y):
                if nx < 0 or ny < 0 or nx >= self.state.map.width or ny >= self.state.map.height:
                    continue
                nm = int(self.state.map.get_point(nx, ny))
                if self._has_owner(nm, attacker) and self._is_sp_cell(nm):
                    adjacent_enemy_sp = True
                    break
            if not adjacent_enemy_sp:
                continue
            support = 0
            for nx, ny in self._neighbors4(x, y):
                if nx < 0 or ny < 0 or nx >= self.state.map.width or ny >= self.state.map.height:
                    continue
                nm = int(self.state.map.get_point(nx, ny))
                if self._has_owner(nm, defender):
                    support += 1
            if support <= 1:
                risk += 1.5 * attack_scale
            elif support == 2:
                risk += 0.8 * attack_scale
            else:
                risk += 0.25 * attack_scale
        return risk

    def _is_frontier_cell(self, x: int, y: int, player: str) -> bool:
        if self.state is None:
            raise RuntimeError("Environment not reset")
        other = "P2" if player == "P1" else "P1"
        for nx, ny in self._neighbors4(x, y):
            if nx < 0 or ny < 0 or nx >= self.state.map.width or ny >= self.state.map.height:
                continue
            nm = int(self.state.map.get_point(nx, ny))
            if self._is_empty_cell(nm) or self._has_owner(nm, other):
                return True
        return False

    def _neighbors4(self, x: int, y: int) -> Iterable[Tuple[int, int]]:
        return ((x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1))

    def _is_valid(self, mask: int) -> bool:
        return (mask & int(Map_PointBit.IsValid)) != 0

    def _is_empty_cell(self, mask: int) -> bool:
        if not self._is_valid(mask):
            return False
        return not self._has_owner(mask, "P1") and not self._has_owner(mask, "P2")

    def _has_owner(self, mask: int, player: str) -> bool:
        bit = Map_PointBit.IsP1 if player == "P1" else Map_PointBit.IsP2
        return (mask & int(bit)) != 0

    def _is_sp_cell(self, mask: int) -> bool:
        return (mask & int(Map_PointBit.IsSp)) != 0

    def _card_from_action(self, action: Action) -> Optional[Card_Single]:
        if self.state is None:
            raise RuntimeError("Environment not reset")
        return next((c for c in self.state.players["P1"].hand if c.Number == action.card_number), None)

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
        p1 = 0
        p2 = 0
        for _, _, m in self._iter_map_cells():
            if not self._is_valid(m):
                continue
            is_p1 = self._has_owner(m, "P1")
            is_p2 = self._has_owner(m, "P2")
            if is_p1 and not is_p2:
                p1 += 1
            elif is_p2 and not is_p1:
                p2 += 1
        return p1, p2

    def _metrics_summary(self, metrics: Dict[str, float]) -> Dict[str, float]:
        return {
            "turn_ratio": round(metrics["turn_ratio"], 4),
            "score_diff": round(metrics["score_diff"], 4),
            "p1_sp": round(metrics["p1_sp"], 4),
            "p2_sp": round(metrics["p2_sp"], 4),
            "p1_reachable_empty": round(metrics["p1_reachable_empty"], 4),
            "p2_reachable_empty": round(metrics["p2_reachable_empty"], 4),
            "enemy_breach_risk": round(metrics["enemy_breach_risk"], 4),
            "own_breach_chance": round(metrics["own_breach_chance"], 4),
        }

