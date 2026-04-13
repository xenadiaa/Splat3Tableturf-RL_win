"""Unified NN loader for strategy .pt checkpoints."""

from __future__ import annotations

from pathlib import Path
import sys
from typing import Dict, List, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.assets.tableturf_types import Map_PointBit  # noqa: E402


_MODEL_CACHE: Dict[str, object] = {}


def _load_model(checkpoint_file: str):
    try:
        import torch
    except Exception as exc:  # pragma: no cover
        raise RuntimeError(f"torch import failed: {exc}") from exc

    ckpt = str(checkpoint_file)
    if ckpt in _MODEL_CACHE:
        return _MODEL_CACHE[ckpt], torch

    from GST_RL.networks import PolicyValueNet

    model = PolicyValueNet(map_channels=6, scalar_dim=6, action_feature_dim=12)
    obj = torch.load(ckpt, map_location="cpu")
    state_dict = obj["model"] if isinstance(obj, dict) and "model" in obj else obj
    if not isinstance(state_dict, dict):
        raise RuntimeError("invalid checkpoint content")
    model.load_state_dict(state_dict)
    model.eval()
    _MODEL_CACHE[ckpt] = model
    return model, torch


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


def _compute_scores(game_map) -> Tuple[int, int]:
    p1 = 0
    p2 = 0
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


def _encode_state(state, player: str) -> Tuple[List[List[List[float]]], List[float]]:
    game_map = state.map
    obs = [[[0.0 for _ in range(game_map.width)] for _ in range(game_map.height)] for _ in range(6)]
    is_p1_agent = player == "P1"
    for y in range(game_map.height):
        for x in range(game_map.width):
            m = int(game_map.get_point(x, y))
            is_p1 = (m & int(Map_PointBit.IsP1)) != 0
            is_p2 = (m & int(Map_PointBit.IsP2)) != 0
            obs[0, y, x] = 1.0 if (m & int(Map_PointBit.IsValid)) else 0.0
            obs[1, y, x] = 1.0 if (is_p1 if is_p1_agent else is_p2) else 0.0
            obs[2, y, x] = 1.0 if (is_p2 if is_p1_agent else is_p1) else 0.0
            obs[3, y, x] = 1.0 if (m & int(Map_PointBit.IsSp)) else 0.0
            obs[4, y, x] = 1.0 if (m & int(Map_PointBit.IsSupplySp)) else 0.0
            obs[5, y, x] = 1.0 if (is_p1 and is_p2) else 0.0

    p1_score, p2_score = _compute_scores(game_map)
    own = state.players[player]
    opp = state.players["P2" if player == "P1" else "P1"]
    own_score = p1_score if is_p1_agent else p2_score
    opp_score = p2_score if is_p1_agent else p1_score
    scalar = [
        state.turn / max(1, state.max_turns),
        own.sp / 20.0,
        opp.sp / 20.0,
        (own_score - opp_score) / 100.0,
        len(own.draw_pile) / 15.0,
        len(opp.draw_pile) / 15.0,
    ]
    return obs, scalar


def _encode_actions(state, player: str, legal_actions: List[dict]) -> List[List[float]]:
    ps = state.players[player]
    hand = ps.hand
    hand_index = {card.Number: idx for idx, card in enumerate(hand)}
    w = max(1, state.map.width - 1)
    h = max(1, state.map.height - 1)
    feats = []
    for action in legal_actions:
        card_no = int(action.get("card_number"))
        card = next((c for c in hand if c.Number == card_no), None)
        if card is None:
            raise RuntimeError(f"card #{card_no} not in hand")
        rotation = int(action.get("rotation", 0))
        x = action.get("x")
        y = action.get("y")
        cell_count, sp_count = _card_cell_stats(card, rotation)
        feats.append(
            [
                hand_index[card.Number] / 3.0,
                1.0 if bool(action.get("pass_turn", False)) else 0.0,
                1.0 if bool(action.get("use_sp_attack", False)) else 0.0,
                rotation / 3.0,
                (float(x) / w) if x is not None else 0.0,
                (float(y) / h) if y is not None else 0.0,
                card.CardPoint / 20.0,
                card.SpecialCost / 10.0,
                cell_count / 64.0,
                sp_count / 64.0,
                ps.sp / 20.0,
                state.turn / max(1, state.max_turns),
            ]
        )
    if not feats:
        return [[0.0] * 12]
    return feats


def choose_action(state, player: str, legal_actions: List[dict], context: Dict[str, object]) -> Dict[str, object]:
    checkpoint = str(context.get("checkpoint_file", "")).strip()
    if not checkpoint:
        raise RuntimeError("checkpoint_file missing in context")
    if not Path(checkpoint).exists():
        raise RuntimeError(f"checkpoint_file not found: {checkpoint}")
    if not legal_actions:
        raise RuntimeError("legal_actions is empty")

    model, torch = _load_model(checkpoint)
    map_obs, scalar_obs = _encode_state(state, player)
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
