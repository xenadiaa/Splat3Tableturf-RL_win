"""Strategic NN loader for strategic_ppo checkpoints."""

from __future__ import annotations

from collections import deque
from pathlib import Path
import sys
from typing import Deque, Dict, Iterable, List, Set, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.assets.tableturf_types import Map_PointBit  # noqa: E402

_MODEL_CACHE: Dict[str, object] = {}
_MAX_SHAPE: Tuple[int, int] | None = None


def _load_torch():
    try:
        import torch
    except Exception as exc:  # pragma: no cover
        raise RuntimeError(f"torch import failed: {exc}") from exc
    return torch


def is_strategic_checkpoint(checkpoint_file: str) -> bool:
    torch = _load_torch()
    obj = torch.load(str(checkpoint_file), map_location="cpu")
    state_dict = obj["model"] if isinstance(obj, dict) and "model" in obj else obj
    if not isinstance(state_dict, dict):
        return False
    keys = set(state_dict.keys())
    return (
        "state_encoder.map_net.6.weight" in keys
        or "action_head.4.weight" in keys
        or "value_head.4.weight" in keys
    )


def _max_map_shape() -> Tuple[int, int]:
    global _MAX_SHAPE
    if _MAX_SHAPE is None:
        from GST_RL.rl_env import _max_map_size_with_padding

        _MAX_SHAPE = _max_map_size_with_padding()
    return _MAX_SHAPE


def _load_model(checkpoint_file: str):
    torch = _load_torch()
    ckpt = str(checkpoint_file)
    if ckpt in _MODEL_CACHE:
        return _MODEL_CACHE[ckpt], torch

    from GST_RL.strategic_networks import StrategicPolicyValueNet

    model = StrategicPolicyValueNet(map_channels=6, scalar_dim=14, action_feature_dim=12)
    obj = torch.load(ckpt, map_location="cpu")
    state_dict = obj["model"] if isinstance(obj, dict) and "model" in obj else obj
    if not isinstance(state_dict, dict):
        raise RuntimeError("invalid checkpoint content")
    model.load_state_dict(state_dict)
    model.eval()
    _MODEL_CACHE[ckpt] = model
    return model, torch


def _is_valid(mask: int) -> bool:
    return (mask & int(Map_PointBit.IsValid)) != 0


def _has_owner(mask: int, player: str) -> bool:
    bit = Map_PointBit.IsP1 if player == "P1" else Map_PointBit.IsP2
    return (mask & int(bit)) != 0


def _is_empty_cell(mask: int) -> bool:
    if not _is_valid(mask):
        return False
    return not _has_owner(mask, "P1") and not _has_owner(mask, "P2")


def _is_sp_cell(mask: int) -> bool:
    return (mask & int(Map_PointBit.IsSp)) != 0


def _neighbors4(x: int, y: int) -> Iterable[Tuple[int, int]]:
    return ((x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1))


def _iter_map_cells(state) -> Iterable[Tuple[int, int, int]]:
    for y in range(state.map.height):
        for x in range(state.map.width):
            yield x, y, int(state.map.get_point(x, y))


def _compute_scores(state) -> Tuple[int, int]:
    p1 = 0
    p2 = 0
    for _, _, m in _iter_map_cells(state):
        if not _is_valid(m):
            continue
        is_p1 = _has_owner(m, "P1")
        is_p2 = _has_owner(m, "P2")
        if is_p1 and not is_p2:
            p1 += 1
        elif is_p2 and not is_p1:
            p2 += 1
    return p1, p2


def _reachable_empty_stats(state, player: str) -> Tuple[int, int, int]:
    reachable_total = 0
    largest_reachable = 0
    largest_locked = 0
    visited: Set[Tuple[int, int]] = set()
    game_map = state.map

    for y in range(game_map.height):
        for x in range(game_map.width):
            if (x, y) in visited:
                continue
            m = int(game_map.get_point(x, y))
            if not _is_empty_cell(m):
                continue
            queue: Deque[Tuple[int, int]] = deque([(x, y)])
            visited.add((x, y))
            component: List[Tuple[int, int]] = []
            touches_player = False
            while queue:
                cx, cy = queue.popleft()
                component.append((cx, cy))
                for nx, ny in _neighbors4(cx, cy):
                    if nx < 0 or ny < 0 or nx >= game_map.width or ny >= game_map.height:
                        continue
                    nm = int(game_map.get_point(nx, ny))
                    if _has_owner(nm, player):
                        touches_player = True
                    if (nx, ny) in visited or not _is_empty_cell(nm):
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


def _is_frontier_cell(state, x: int, y: int, player: str) -> bool:
    other = "P2" if player == "P1" else "P1"
    for nx, ny in _neighbors4(x, y):
        if nx < 0 or ny < 0 or nx >= state.map.width or ny >= state.map.height:
            continue
        nm = int(state.map.get_point(nx, ny))
        if _is_empty_cell(nm) or _has_owner(nm, other):
            return True
    return False


def _sp_breach_risk(state, attacker: str, defender: str) -> float:
    attacker_sp = float(state.players[attacker].sp)
    if attacker_sp < 3:
        return 0.0
    attack_scale = 2.0 if attacker_sp >= 6 else 1.0
    risk = 0.0
    for x, y, m in _iter_map_cells(state):
        if not _has_owner(m, defender):
            continue
        if not _is_frontier_cell(state, x, y, defender):
            continue
        adjacent_enemy_sp = False
        for nx, ny in _neighbors4(x, y):
            if nx < 0 or ny < 0 or nx >= state.map.width or ny >= state.map.height:
                continue
            nm = int(state.map.get_point(nx, ny))
            if _has_owner(nm, attacker) and _is_sp_cell(nm):
                adjacent_enemy_sp = True
                break
        if not adjacent_enemy_sp:
            continue
        support = 0
        for nx, ny in _neighbors4(x, y):
            if nx < 0 or ny < 0 or nx >= state.map.width or ny >= state.map.height:
                continue
            nm = int(state.map.get_point(nx, ny))
            if _has_owner(nm, defender):
                support += 1
        if support <= 1:
            risk += 1.5 * attack_scale
        elif support == 2:
            risk += 0.8 * attack_scale
        else:
            risk += 0.25 * attack_scale
    return risk


def _compute_metrics(state) -> Dict[str, float]:
    valid_cells = 0
    empty_cells = 0
    for _, _, m in _iter_map_cells(state):
        if not _is_valid(m):
            continue
        valid_cells += 1
        if _is_empty_cell(m):
            empty_cells += 1

    p1_reach, p1_largest, p1_locked = _reachable_empty_stats(state, "P1")
    p2_reach, p2_largest, p2_locked = _reachable_empty_stats(state, "P2")
    p1_score, p2_score = _compute_scores(state)
    return {
        "valid_cells": float(valid_cells),
        "empty_cells": float(empty_cells),
        "turn_ratio": state.turn / max(1, state.max_turns),
        "p1_sp": float(state.players["P1"].sp),
        "p2_sp": float(state.players["P2"].sp),
        "p1_score": float(p1_score),
        "p2_score": float(p2_score),
        "score_diff": float(p1_score - p2_score),
        "p1_reachable_empty": float(p1_reach),
        "p2_reachable_empty": float(p2_reach),
        "p1_largest_reachable": float(p1_largest),
        "p2_largest_reachable": float(p2_largest),
        "p1_largest_locked": float(p1_locked),
        "p2_largest_locked": float(p2_locked),
        "enemy_breach_risk": float(_sp_breach_risk(state, attacker="P2", defender="P1")),
        "own_breach_chance": float(_sp_breach_risk(state, attacker="P1", defender="P2")),
        "p1_draw_ratio": len(state.players["P1"].draw_pile) / 15.0,
        "p2_draw_ratio": len(state.players["P2"].draw_pile) / 15.0,
    }


def _encode_state(state) -> Tuple[List[List[List[float]]], List[float]]:
    metrics = _compute_metrics(state)
    max_w, max_h = _max_map_shape()
    obs = [[[0.0 for _ in range(max_w)] for _ in range(max_h)] for _ in range(6)]
    for y in range(state.map.height):
        for x in range(state.map.width):
            m = int(state.map.get_point(x, y))
            obs[0][y][x] = 1.0 if _is_valid(m) else 0.0
            obs[1][y][x] = 1.0 if (m & int(Map_PointBit.IsP1)) else 0.0
            obs[2][y][x] = 1.0 if (m & int(Map_PointBit.IsP2)) else 0.0
            obs[3][y][x] = 1.0 if (m & int(Map_PointBit.IsSp)) else 0.0
            obs[4][y][x] = 1.0 if (m & int(Map_PointBit.IsSupplySp)) else 0.0
            obs[5][y][x] = 1.0 if ((m & int(Map_PointBit.IsP1)) and (m & int(Map_PointBit.IsP2))) else 0.0

    valid = max(1.0, metrics["valid_cells"])
    scalar = [
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
    ]
    return obs, scalar


def _card_cell_stats(card, rotation: int) -> Tuple[int, int]:
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


def _encode_actions(state, player: str, legal_actions: List[dict]) -> List[List[float]]:
    hand = state.players[player].hand
    hand_index = {card.Number: idx for idx, card in enumerate(hand)}
    max_w, max_h = _max_map_shape()
    feats: List[List[float]] = []
    for action in legal_actions:
        card = next((c for c in hand if c.Number == int(action.get("card_number"))), None)
        if card is None:
            raise RuntimeError(f"card #{action.get('card_number')} not in hand")
        cell_count, sp_count = _card_cell_stats(card, int(action.get("rotation", 0)))
        x = action.get("x")
        y = action.get("y")
        feats.append(
            [
                hand_index[card.Number] / 3.0,
                1.0 if bool(action.get("pass_turn", False)) else 0.0,
                1.0 if bool(action.get("use_sp_attack", False)) else 0.0,
                int(action.get("rotation", 0)) / 3.0,
                (float(x) / max(1, max_w - 1)) if x is not None else 0.0,
                (float(y) / max(1, max_h - 1)) if y is not None else 0.0,
                card.CardPoint / 20.0,
                card.SpecialCost / 10.0,
                cell_count / 64.0,
                sp_count / 64.0,
                state.players[player].sp / 20.0,
                state.turn / max(1, state.max_turns),
            ]
        )
    return feats or [[0.0] * 12]


def choose_action(state, player: str, legal_actions: List[dict], context: Dict[str, object]) -> Dict[str, object]:
    checkpoint = str(context.get("checkpoint_file", "")).strip()
    if not checkpoint:
        raise RuntimeError("checkpoint_file missing in context")
    if not Path(checkpoint).exists():
        raise RuntimeError(f"checkpoint_file not found: {checkpoint}")
    if not legal_actions:
        raise RuntimeError("legal_actions is empty")

    model, torch = _load_model(checkpoint)
    map_obs, scalar_obs = _encode_state(state)
    action_feats = _encode_actions(state, player, legal_actions)
    with torch.no_grad():
        logits, _ = model.forward_single(
            torch.as_tensor(map_obs, dtype=torch.float32),
            torch.as_tensor(scalar_obs, dtype=torch.float32),
            torch.as_tensor(action_feats, dtype=torch.float32),
        )
        idx = int(torch.argmax(logits).item())
    idx = max(0, min(idx, len(legal_actions) - 1))
    chosen = dict(legal_actions[idx])
    return {
        "card_number": chosen.get("card_number"),
        "pass_turn": bool(chosen.get("pass_turn", False)),
        "use_sp_attack": bool(chosen.get("use_sp_attack", False)),
        "rotation": int(chosen.get("rotation", 0)),
        "x": chosen.get("x"),
        "y": chosen.get("y"),
    }
