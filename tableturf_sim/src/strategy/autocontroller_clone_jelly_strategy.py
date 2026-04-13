"""AutoControllerHelper-inspired Clone Jelly strategy.

This module ports the high-level phase logic from AutoControllerHelper's
`ThreeTwelveSp` Tableturf AI into the simulator strategy system:

- early game: use small cards to extend toward the opponent
- mid game: build special points until 3 SP is available
- late game: preserve the 12-point finisher for the last turn
- final turn: spend 3 SP on the 12-point finisher when possible

The simulator has richer state than the original vision-driven C++ bot, so this
port uses exact legal actions and board state while keeping the same overall
decision priorities.
"""

from __future__ import annotations

from copy import deepcopy
from typing import Dict, Iterable, List, Optional, Tuple

from src.assets.tableturf_types import Map_PointBit, Map_PointMask
from src.engine.env_core import _card_cells_on_map, _find_card_in_hand
from src.engine.loaders import load_map
from src.utils.common_utils import create_card_from_id, validate_place_card_action

STRATEGY_LABEL = "AutoController Clone Jelly"
PREDICT_FINISHER_CARD_ID = 73  # SpMultiMissile
LEAST_REPLACE_TO_WIN = 120
LAST_TURN_EXPAND_TURF = 9


def _payload_to_card(state, player: str, payload: Dict[str, object]):
    return _find_card_in_hand(state.players[player], payload.get("card_number"))


def _payload_cells(state, player: str, payload: Dict[str, object]) -> List[Tuple[int, int, int]]:
    card = _payload_to_card(state, player, payload)
    if card is None or bool(payload.get("pass_turn", False)):
        return []
    return _card_cells_on_map(
        card,
        int(payload.get("x", 0)),
        int(payload.get("y", 0)),
        int(payload.get("rotation", 0)),
    )


def _neighbors8(x: int, y: int) -> Iterable[Tuple[int, int]]:
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            if dx == 0 and dy == 0:
                continue
            yield x + dx, y + dy


def _owner_bits(player: str) -> Tuple[int, int]:
    if player == "P1":
        return int(Map_PointBit.IsP1), int(Map_PointBit.IsP2)
    return int(Map_PointBit.IsP2), int(Map_PointBit.IsP1)


def _card_point(state, player: str, payload: Dict[str, object]) -> int:
    card = _payload_to_card(state, player, payload)
    return int(card.CardPoint) if card is not None else -1


def _card_sp_cost(state, player: str, payload: Dict[str, object]) -> int:
    card = _payload_to_card(state, player, payload)
    return int(card.SpecialCost) if card is not None else 0


def _finisher_card_numbers(state, player: str) -> List[int]:
    nums: List[int] = []
    for cid in state.players[player].deck_ids:
        card = create_card_from_id(cid)
        if int(card.CardPoint) == 12:
            nums.append(card.Number)
    return nums


def _finisher_card(state, player: str):
    for cid in state.players[player].deck_ids:
        card = create_card_from_id(cid)
        if int(card.CardPoint) == 12:
            return card
    return None


def _predict_finisher_card():
    return create_card_from_id(PREDICT_FINISHER_CARD_ID)


def _project_score_swing(state, player: str, payload: Dict[str, object]) -> float:
    own_bit, opp_bit = _owner_bits(player)
    swing = 0.0
    for x, y, _cell_type in _payload_cells(state, player, payload):
        m = int(state.map.get_point(x, y))
        had_own = (m & own_bit) != 0
        had_opp = (m & opp_bit) != 0
        if had_opp and not had_own:
            swing += 2.0
        elif not had_own and not had_opp:
            swing += 1.0
    return swing


def _normal_owner_mask(player: str, cell_type: int) -> int:
    if player == "P1":
        return int(Map_PointMask.P1Special if int(cell_type) == 2 else Map_PointMask.P1Normal)
    return int(Map_PointMask.P2Special if int(cell_type) == 2 else Map_PointMask.P2Normal)


def _placed_occupancy(state, player: str, payload: Dict[str, object]) -> Tuple[set[Tuple[int, int]], set[Tuple[int, int]]]:
    occupied = set()
    special = set()
    for x, y, cell_type in _payload_cells(state, player, payload):
        occupied.add((x, y))
        if int(cell_type) == 2:
            special.add((x, y))
    return occupied, special


def _special_points_after_action(state, player: str, payload: Dict[str, object]) -> List[Tuple[int, int]]:
    own_bit, _opp_bit = _owner_bits(player)
    occupied, placed_special = _placed_occupancy(state, player, payload)
    out: List[Tuple[int, int]] = []
    seen = set()
    for y in range(state.map.height):
        for x in range(state.map.width):
            m = int(state.map.get_point(x, y))
            if (m & own_bit) != 0 and (m & int(Map_PointBit.IsSp)) != 0:
                out.append((x, y))
                seen.add((x, y))
    for pos in placed_special:
        if pos not in seen and pos in occupied:
            out.append(pos)
    return out


def _is_blocked_after_action(state, player: str, payload: Dict[str, object], x: int, y: int) -> bool:
    if not (0 <= x < state.map.width and 0 <= y < state.map.height):
        return True
    occupied, _special = _placed_occupancy(state, player, payload)
    if (x, y) in occupied:
        return True
    m = int(state.map.get_point(x, y))
    if (m & int(Map_PointBit.IsValid)) == 0:
        return True
    return (m & (int(Map_PointBit.IsP1) | int(Map_PointBit.IsP2))) != 0


def _build_special_score(state, player: str, payload: Dict[str, object]) -> int:
    score = 0
    for x, y in _special_points_after_action(state, player, payload):
        surround_count = 0
        for nx, ny in _neighbors8(x, y):
            if _is_blocked_after_action(state, player, payload, nx, ny):
                surround_count += 1
        score += sum(range(surround_count))
        if surround_count == 8:
            score += 40
    return score


def _build_special_delta(state, player: str, payload: Dict[str, object]) -> int:
    return _build_special_score(state, player, payload) - _build_special_score(state, player, {"pass_turn": True})


def _home_anchor(state, player: str) -> Tuple[float, float]:
    if str(state.map.map_id) == "Rectangle":
        return (4.0, 22.0)
    try:
        base_map = load_map(state.map.map_id)
    except Exception:
        base_map = state.map
    own_bit, _opp_bit = _owner_bits(player)
    own_special_cells: List[Tuple[int, int]] = []
    for y in range(base_map.height):
        for x in range(base_map.width):
            m = int(base_map.get_point(x, y))
            if (m & own_bit) != 0 and (m & int(Map_PointBit.IsSp)) != 0:
                own_special_cells.append((x, y))
    if not own_special_cells:
        return 0.0, 0.0
    if str(base_map.map_id) == "ManyHole":
        own_special_cells.sort(key=lambda p: (p[0], -p[1]))
        x, y = own_special_cells[0]
        return float(x), float(y)
    own_special_cells.sort(key=lambda p: (p[0], p[1]), reverse=True)
    x, y = own_special_cells[0]
    return float(x), float(y)


def _rotation_penalty(payload: Dict[str, object]) -> int:
    rotation = int(payload.get("rotation", 0))
    return rotation if rotation % 2 == 0 else 1


def _special_anchor(state, player: str, payload: Dict[str, object]) -> Tuple[float, float]:
    _occupied, placed_special = _placed_occupancy(state, player, payload)
    if placed_special:
        x, y = next(iter(placed_special))
        return float(x), float(y)
    cells = _payload_cells(state, player, payload)
    if not cells:
        return 0.0, 0.0
    return (
        sum(x for x, _y, _t in cells) / len(cells),
        sum(y for _x, y, _t in cells) / len(cells),
    )


def _least_moves_score(state, player: str, payload: Dict[str, object]) -> float:
    hx, hy = _home_anchor(state, player)
    ax, ay = _special_anchor(state, player, payload)
    return 10.0 - abs(ax - hx) - abs(ay - hy) - float(_rotation_penalty(payload))


def _expand_turf_score(state, player: str, payload: Dict[str, object]) -> float:
    ax, ay = _special_anchor(state, player, payload)
    map_mid_x = (state.map.width - 1) / 2.0
    forward = -ay if player == "P1" else ay
    return 30.0 + forward - abs(ax - map_mid_x)


def _best_by_score(actions: List[dict], scorer) -> Dict[str, object]:
    best = None
    best_score = None
    for action in actions:
        score = scorer(action)
        if best_score is None or score > best_score:
            best = action
            best_score = score
    if best is None:
        raise RuntimeError("no action available")
    return best


def _action_key(payload: Dict[str, object]) -> Tuple[object, ...]:
    return (
        payload.get("card_number"),
        bool(payload.get("pass_turn", False)),
        bool(payload.get("use_sp_attack", False)),
        int(payload.get("rotation", 0)),
        payload.get("x"),
        payload.get("y"),
    )


def _count_turf(game_map, player: str) -> int:
    own_bit, _opp_bit = _owner_bits(player)
    total = 0
    for y in range(game_map.height):
        for x in range(game_map.width):
            m = int(game_map.get_point(x, y))
            if (m & own_bit) != 0:
                total += 1
    return total


def _cover_enemy_turf_score(before_map, after_map, player: str) -> int:
    own_before = _count_turf(before_map, player)
    own_after = _count_turf(after_map, player)
    opp_player = "P2" if player == "P1" else "P1"
    opp_before = _count_turf(before_map, opp_player)
    opp_after = _count_turf(after_map, opp_player)
    return 100 + (opp_before - opp_after) + (own_after - own_before)


def _apply_normal_action_to_map(state, player: str, payload: Dict[str, object]):
    game_map = deepcopy(state.map)
    for x, y, cell_type in _payload_cells(state, player, payload):
        game_map.set_point(x, y, _normal_owner_mask(player, cell_type))
    return game_map


def _apply_action_to_map(state, player: str, payload: Dict[str, object]):
    game_map = deepcopy(state.map)
    for x, y, cell_type in _payload_cells(state, player, payload):
        game_map.set_point(x, y, _normal_owner_mask(player, cell_type))
    return game_map


def _finisher_enemy_cover_count(game_map, finisher_card, player: str, x: int, y: int, rotation: int) -> int:
    _own_bit, opp_bit = _owner_bits(player)
    cells = _card_cells_on_map(finisher_card, x, y, rotation)
    covered = 0
    for cx, cy, _cell_type in cells:
        m = int(game_map.get_point(cx, cy))
        if (m & opp_bit) != 0:
            covered += 1
    return covered


def _max_finisher_special_score(game_map, finisher_card, player: str) -> Optional[int]:
    best: Optional[int] = None
    is_p1 = player == "P1"
    for rotation in (0, 1, 2, 3):
        for y in range(game_map.height):
            for x in range(game_map.width):
                ok, _reason, cells = validate_place_card_action(
                    card=finisher_card,
                    game_map=game_map,
                    anchor_x=x,
                    anchor_y=y,
                    rotation=rotation,
                    is_p1=is_p1,
                    use_sp_attack=True,
                )
                if not ok:
                    continue
                finisher_map = deepcopy(game_map)
                for cx, cy, cell_type in cells:
                    finisher_map.set_point(cx, cy, _normal_owner_mask(player, cell_type))
                score = _cover_enemy_turf_score(game_map, finisher_map, player)
                if score <= int(finisher_card.CardPoint):
                    continue
                if best is None or score > best:
                    best = score
    return best


def _future_finisher_priority(
    payload: Dict[str, object],
    current_finisher_score: Optional[int],
    future_finisher_scores: Dict[Tuple[object, ...], Optional[int]],
) -> int:
    if current_finisher_score is None:
        return 0
    predicted = future_finisher_scores.get(_action_key(payload))
    if predicted is None:
        return 0
    if predicted > current_finisher_score:
        return 1000 + int(predicted)
    return 0


def _max_predict_score(game_map, player: str) -> int:
    finisher_card = _predict_finisher_card()
    score = _max_finisher_special_score(game_map, finisher_card, player)
    return int(score) if score is not None else 0


def choose_action(state, player: str, legal_actions: List[dict], context: Dict[str, object]) -> Dict[str, object]:
    if not legal_actions:
        raise RuntimeError("legal_actions is empty")

    ps = state.players[player]
    turns_left = max(1, int(state.max_turns) - int(state.turn) + 1)
    finishers = set(_finisher_card_numbers(state, player))
    finisher_card = _finisher_card(state, player)
    predict_finisher_card = _predict_finisher_card()
    finisher_in_hand = {c.Number for c in ps.hand if c.Number in finishers}

    non_pass = [a for a in legal_actions if not bool(a.get("pass_turn", False))]
    normal = [a for a in non_pass if not bool(a.get("use_sp_attack", False))]
    sp_actions = [a for a in non_pass if bool(a.get("use_sp_attack", False))]

    # Final turn: spend 3 SP on the 12-point finisher whenever possible.
    if state.turn >= state.max_turns:
        finisher_sp = [
            a for a in sp_actions
            if a.get("card_number") in finisher_in_hand and _card_sp_cost(state, player, a) <= ps.sp
        ]
        if ps.sp >= 3 and finisher_sp:
            return _best_by_score(
                finisher_sp,
                lambda a: (
                    _project_score_swing(state, player, a),
                    _card_point(state, player, a),
                ),
            )
        usable_sp = [a for a in sp_actions if _card_sp_cost(state, player, a) <= ps.sp]
        if usable_sp:
            return _best_by_score(usable_sp, lambda a: (_project_score_swing(state, player, a), _card_point(state, player, a)))
        if normal:
            return _best_by_score(normal, lambda a: (_project_score_swing(state, player, a), _card_point(state, player, a)))
        return legal_actions[0]

    # Before the final turn, preserve the 12-point finisher when alternatives exist.
    if finisher_in_hand:
        keep_finisher = [
            a for a in legal_actions
            if bool(a.get("pass_turn", False)) or a.get("card_number") not in finisher_in_hand
        ]
        if keep_finisher:
            legal_actions = keep_finisher
            non_pass = [a for a in legal_actions if not bool(a.get("pass_turn", False))]
            normal = [a for a in non_pass if not bool(a.get("use_sp_attack", False))]
            sp_actions = [a for a in non_pass if bool(a.get("use_sp_attack", False))]

    # AutoController's ThreeTwelveSp mode avoids >4-point cards before the final turn.
    small_normal = [a for a in normal if _card_point(state, player, a) <= 4]
    if small_normal:
        normal = small_normal

    current_finisher_score: Optional[int] = None
    future_finisher_scores: Dict[Tuple[object, ...], Optional[int]] = {}
    if finisher_card is not None and ps.sp >= 3 and state.turn < state.max_turns:
        current_finisher_score = _max_predict_score(state.map, player)
        for action in normal:
            preview_map = _apply_normal_action_to_map(state, player, action)
            future_finisher_scores[_action_key(action)] = _max_predict_score(preview_map, player)

        if current_finisher_score is not None:
            preserved_normal: List[dict] = []
            for action in normal:
                predicted = future_finisher_scores.get(_action_key(action))
                if predicted is None or predicted >= current_finisher_score:
                    preserved_normal.append(action)
            if preserved_normal:
                normal = preserved_normal

    # Spend excess SP on small cards if we are already above the finisher threshold.
    if ps.sp > 3 and (current_finisher_score or 0) < LEAST_REPLACE_TO_WIN:
        extra_sp = [
            a for a in sp_actions
            if _card_sp_cost(state, player, a) <= ps.sp
            and _card_point(state, player, a) <= (4 if ps.sp >= 5 else 3)
        ]
        profitable_sp = []
        for action in extra_sp:
            preview_map = _apply_action_to_map(state, player, action)
            score = _cover_enemy_turf_score(state.map, preview_map, player)
            if score > int(_card_point(state, player, action)):
                profitable_sp.append(action)
        if profitable_sp:
            return _best_by_score(
                profitable_sp,
                lambda a: (
                    _cover_enemy_turf_score(state.map, _apply_action_to_map(state, player, a), player),
                    -_card_sp_cost(state, player, a),
                    _card_point(state, player, a),
                ),
            )

    # Match BrianUuu's phase split:
    # - turns_left >= 9: ExpandTurf
    # - turns_left < 9: BuildSpecial, unless late-game LeastMoves condition applies
    if turns_left >= LAST_TURN_EXPAND_TURF and normal:
        return _best_by_score(
            normal,
            lambda a: (
                _future_finisher_priority(a, current_finisher_score, future_finisher_scores),
                _expand_turf_score(state, player, a),
                _card_point(state, player, a),
                _least_moves_score(state, player, a),
            ),
        )

    if normal and ((turns_left == 2 or (current_finisher_score or 0) >= LEAST_REPLACE_TO_WIN) and ps.sp >= 3):
        return _best_by_score(
            normal,
            lambda a: (
                _future_finisher_priority(a, current_finisher_score, future_finisher_scores),
                _least_moves_score(state, player, a),
                _card_point(state, player, a),
            ),
        )

    # Otherwise, build special like BrianUuu's helper.
    if normal:
        return _best_by_score(
            normal,
            lambda a: (
                _future_finisher_priority(a, current_finisher_score, future_finisher_scores),
                _build_special_delta(state, player, a),
                _build_special_score(state, player, a),
                _card_point(state, player, a),
            ),
        )

    usable_sp = [a for a in sp_actions if _card_sp_cost(state, player, a) <= ps.sp]
    if usable_sp:
        return _best_by_score(
            usable_sp,
            lambda a: (
                _cover_enemy_turf_score(state.map, _apply_action_to_map(state, player, a), player),
                -_card_sp_cost(state, player, a),
                _card_point(state, player, a),
            ),
        )

    return legal_actions[0]
