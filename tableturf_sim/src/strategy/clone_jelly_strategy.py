"""CloneJelly deck strategy.

Heuristic:
- Preserve the 12-point finisher until the final turn; outside the final turn it has
  the lowest priority.
- Early game (first ~4 turns): extend toward the opponent home with small cards.
- Mid / late game before final turn: switch to SP-building, reduce empty neighbors
  around own special points, and target at least 3 SP. Extra SP may be spent after
  that threshold, but normal SP-building remains preferred.
- Final turn: 12-point 3SP finisher has top priority; choose the SP attack position
  with the largest score swing.
"""

from __future__ import annotations

from collections import deque
from typing import Dict, Iterable, List, Optional, Tuple

from src.assets.tableturf_types import Map_PointBit
from src.engine.env_core import _card_cells_on_map, _find_card_in_hand, _owner_cells, _centroid
from src.utils.common_utils import create_card_from_id


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


def _project_score_swing(state, player: str, payload: Dict[str, object]) -> float:
    own_bit = int(Map_PointBit.IsP1) if player == "P1" else int(Map_PointBit.IsP2)
    opp_bit = int(Map_PointBit.IsP2) if player == "P1" else int(Map_PointBit.IsP1)
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


def _neighbors8(x: int, y: int) -> Iterable[Tuple[int, int]]:
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            if dx == 0 and dy == 0:
                continue
            yield x + dx, y + dy


def _sp_activation_count(state, player: str, payload: Dict[str, object]) -> int:
    own_bit = int(Map_PointBit.IsP1) if player == "P1" else int(Map_PointBit.IsP2)
    placed = {(x, y) for x, y, _ in _payload_cells(state, player, payload)}
    activated = 0
    for y in range(state.map.height):
        for x in range(state.map.width):
            m = int(state.map.get_point(x, y))
            if (m & own_bit) == 0 or (m & int(Map_PointBit.IsSp)) == 0:
                continue
            if (m & int(Map_PointBit.IsSupplySp)) != 0:
                continue
            empty_before = 0
            empty_after = 0
            for nx, ny in _neighbors8(x, y):
                if not (0 <= nx < state.map.width and 0 <= ny < state.map.height):
                    continue
                nm = int(state.map.get_point(nx, ny))
                if (nm & int(Map_PointBit.IsValid)) == 0:
                    continue
                occupied_before = (nm & (int(Map_PointBit.IsP1) | int(Map_PointBit.IsP2))) != 0
                occupied_after = occupied_before or ((nx, ny) in placed)
                if not occupied_before:
                    empty_before += 1
                if not occupied_after:
                    empty_after += 1
            if empty_before > 0 and empty_after == 0:
                activated += 1
    return activated


def _special_empty_reduction(state, player: str, payload: Dict[str, object]) -> int:
    own_bit = int(Map_PointBit.IsP1) if player == "P1" else int(Map_PointBit.IsP2)
    placed = {(x, y) for x, y, _ in _payload_cells(state, player, payload)}
    reduced = 0
    for y in range(state.map.height):
        for x in range(state.map.width):
            m = int(state.map.get_point(x, y))
            if (m & own_bit) == 0 or (m & int(Map_PointBit.IsSp)) == 0:
                continue
            if (m & int(Map_PointBit.IsSupplySp)) != 0:
                continue
            empty_before = 0
            empty_after = 0
            for nx, ny in _neighbors8(x, y):
                if not (0 <= nx < state.map.width and 0 <= ny < state.map.height):
                    continue
                nm = int(state.map.get_point(nx, ny))
                if (nm & int(Map_PointBit.IsValid)) == 0:
                    continue
                occupied_before = (nm & (int(Map_PointBit.IsP1) | int(Map_PointBit.IsP2))) != 0
                occupied_after = occupied_before or ((nx, ny) in placed)
                if not occupied_before:
                    empty_before += 1
                if not occupied_after:
                    empty_after += 1
            if empty_after < empty_before:
                reduced += empty_before - empty_after
    return reduced


def _advancement_score(state, player: str, payload: Dict[str, object]) -> float:
    own_cells = _owner_cells(state, is_p1=(player == "P1"))
    opp_cells = _owner_cells(state, is_p1=(player != "P1"))
    own_cx, own_cy = _centroid(own_cells, state.map.width, state.map.height)
    opp_cx, opp_cy = _centroid(opp_cells, state.map.width, state.map.height)
    vec_x = opp_cx - own_cx
    vec_y = opp_cy - own_cy
    norm = (vec_x * vec_x + vec_y * vec_y) ** 0.5 or 1.0
    ux, uy = vec_x / norm, vec_y / norm
    cells = _payload_cells(state, player, payload)
    if not cells:
        return -1e9
    act_cx = sum(x for x, _, _ in cells) / len(cells)
    act_cy = sum(y for _, y, _ in cells) / len(cells)
    return (act_cx - own_cx) * ux + (act_cy - own_cy) * uy


def _region_extension_score(state, player: str, payload: Dict[str, object]) -> float:
    own_bit = int(Map_PointBit.IsP1) if player == "P1" else int(Map_PointBit.IsP2)
    opp_bit = int(Map_PointBit.IsP2) if player == "P1" else int(Map_PointBit.IsP1)
    cells = _payload_cells(state, player, payload)
    if not cells:
        return -1e9

    own_cells = _owner_cells(state, is_p1=(player == "P1"))
    opp_cells = _owner_cells(state, is_p1=(player != "P1"))
    own_cx, own_cy = _centroid(own_cells, state.map.width, state.map.height)

    # Target the opponent's broad occupied region rather than only the global centroid.
    opp_coords = []
    for y in range(state.map.height):
        for x in range(state.map.width):
            m = int(state.map.get_point(x, y))
            if (m & opp_bit) != 0:
                opp_coords.append((x, y))
    if opp_coords:
        opp_min_x = min(x for x, _ in opp_coords)
        opp_max_x = max(x for x, _ in opp_coords)
        opp_min_y = min(y for _, y in opp_coords)
        opp_max_y = max(y for _, y in opp_coords)
        opp_cx = (opp_min_x + opp_max_x) / 2.0
        opp_cy = (opp_min_y + opp_max_y) / 2.0
        opp_span = (opp_max_x - opp_min_x + 1) + (opp_max_y - opp_min_y + 1)
    else:
        opp_cx, opp_cy = _centroid(opp_cells, state.map.width, state.map.height)
        opp_span = 0.0
    vec_x = opp_cx - own_cx
    vec_y = opp_cy - own_cy
    norm = (vec_x * vec_x + vec_y * vec_y) ** 0.5 or 1.0
    ux, uy = vec_x / norm, vec_y / norm

    placed = {(x, y) for x, y, _ in cells}
    act_cx = sum(x for x, _, _ in cells) / len(cells)
    act_cy = sum(y for _, y, _ in cells) / len(cells)
    forward = (act_cx - own_cx) * ux + (act_cy - own_cy) * uy
    target_closeness = -((act_cx - opp_cx) ** 2 + (act_cy - opp_cy) ** 2) ** 0.5
    map_mid_x = (state.map.width - 1) / 2.0
    center_bias = -abs(act_cx - map_mid_x)
    edge_touch = 0
    for x, y in placed:
        if x <= 1 or x >= state.map.width - 2:
            edge_touch += 1

    frontier = set()
    for x, y in placed:
        for nx, ny in _neighbors8(x, y):
            if not (0 <= nx < state.map.width and 0 <= ny < state.map.height):
                continue
            if (nx, ny) in placed:
                continue
            m = int(state.map.get_point(nx, ny))
            if (m & int(Map_PointBit.IsValid)) == 0:
                continue
            if (m & (own_bit | opp_bit)) != 0:
                continue
            frontier.add((nx, ny))

    seen = set(frontier)
    q = deque((x, y, 0) for x, y in frontier)
    region_score = 0.0
    region_size = 0
    enemy_edge = 0
    while q:
        x, y, depth = q.popleft()
        region_size += 1
        proj = (x - own_cx) * ux + (y - own_cy) * uy
        depth_weight = max(0.15, 1.0 - 0.12 * depth)
        region_score += proj * depth_weight
        for nx, ny in _neighbors8(x, y):
            if not (0 <= nx < state.map.width and 0 <= ny < state.map.height):
                continue
            nm = int(state.map.get_point(nx, ny))
            if (nm & int(Map_PointBit.IsValid)) == 0:
                continue
            if (nm & opp_bit) != 0:
                enemy_edge += 1
                continue
            if (nm & own_bit) != 0:
                continue
            if (nx, ny) in placed or (nx, ny) in seen:
                continue
            seen.add((nx, ny))
            q.append((nx, ny, depth + 1))

    return (
        (1.4 * forward)
        + (0.26 * region_score)
        + (0.32 * region_size)
        + (0.9 * enemy_edge)
        + (1.2 * target_closeness)
        + (0.2 * opp_span)
        + (1.35 * center_bias)
        - (0.9 * edge_touch)
    )


def _finisher_card_numbers(state, player: str) -> List[int]:
    nums: List[int] = []
    for cid in state.players[player].deck_ids:
        card = create_card_from_id(cid)
        if card.CardPoint == 12:
            nums.append(card.Number)
    return nums


def _card_point(state, player: str, payload: Dict[str, object]) -> int:
    card = _payload_to_card(state, player, payload)
    return int(card.CardPoint) if card is not None else -1


def _card_sp_cost(state, player: str, payload: Dict[str, object]) -> int:
    card = _payload_to_card(state, player, payload)
    return int(card.SpecialCost) if card is not None else 0


def _non_final_clone_jelly_priority(state, player: str, payload: Dict[str, object]) -> int:
    point = _card_point(state, player, payload)
    order = {
        4: 5,
        3: 4,
        2: 3,
        1: 2,
        12: 1,
    }
    return order.get(point, 0)


def _best_by_score(state, player: str, actions: List[dict], scorer) -> Dict[str, object]:
    best = None
    best_score = None
    for a in actions:
        score = scorer(a)
        if best_score is None or score > best_score:
            best = a
            best_score = score
    if best is None:
        raise RuntimeError("no action available")
    return best


def choose_action(state, player: str, legal_actions: List[dict], context: Dict[str, object]) -> Dict[str, object]:
    if not legal_actions:
        raise RuntimeError("legal_actions is empty")

    ps = state.players[player]
    finishers = set(_finisher_card_numbers(state, player))
    finisher_in_hand = {c.Number for c in ps.hand if c.Number in finishers}

    non_pass = [a for a in legal_actions if not bool(a.get("pass_turn", False))]
    normal = [a for a in non_pass if not bool(a.get("use_sp_attack", False))]
    sp_actions = [a for a in non_pass if bool(a.get("use_sp_attack", False))]
    extension_turns = int(context.get("extension_turns", 4)) if context else 4

    # Final turn: use the 12-point finisher SP attack if possible.
    if state.turn >= state.max_turns:
        finisher_sp = [
            a for a in sp_actions
            if a.get("card_number") in finisher_in_hand
        ]
        finisher_sp = [a for a in finisher_sp if _card_sp_cost(state, player, a) <= ps.sp]
        if ps.sp >= 3 and finisher_sp:
            return _best_by_score(
                state,
                player,
                finisher_sp,
                lambda a: (
                    _project_score_swing(state, player, a),
                    _card_point(state, player, a),
                ),
            )
        if sp_actions:
            return _best_by_score(state, player, sp_actions, lambda a: _project_score_swing(state, player, a))
        if normal:
            return _best_by_score(
                state,
                player,
                normal,
                lambda a: (
                    _payload_to_card(state, player, a).CardPoint if _payload_to_card(state, player, a) else -1,
                    _project_score_swing(state, player, a),
                ),
            )
        return legal_actions[0]

    # Preserve the finisher before the final turn whenever possible.
    if finisher_in_hand:
        keep_finisher = [a for a in legal_actions if bool(a.get("pass_turn", False)) or a.get("card_number") not in finisher_in_hand]
        if keep_finisher:
            legal_actions = keep_finisher
            non_pass = [a for a in legal_actions if not bool(a.get("pass_turn", False))]
            normal = [a for a in non_pass if not bool(a.get("use_sp_attack", False))]
            sp_actions = [a for a in non_pass if bool(a.get("use_sp_attack", False))]

    if state.turn < state.max_turns and non_pass:
        best_priority = max(_non_final_clone_jelly_priority(state, player, a) for a in non_pass)
        prioritized = [a for a in non_pass if _non_final_clone_jelly_priority(state, player, a) == best_priority]
        if prioritized:
            non_pass = prioritized
            normal = [a for a in non_pass if not bool(a.get("use_sp_attack", False))]
            sp_actions = [a for a in non_pass if bool(a.get("use_sp_attack", False))]

    # Early game: extend toward enemy home using small cards.
    if state.turn <= extension_turns and normal:
        return _best_by_score(
            state,
            player,
            normal,
            lambda a: (
                _region_extension_score(state, player, a),
                -_card_point(state, player, a),
            ),
        )

    # Mid / late before final turn:
    # - before 3 SP: build toward 3 SP
    # - after reaching 3 SP: resume extending toward enemy home
    if normal:
        if ps.sp >= 3:
            return _best_by_score(
                state,
                player,
                normal,
                lambda a: (
                    _region_extension_score(state, player, a),
                    _sp_activation_count(state, player, a),
                    -_card_point(state, player, a),
                ),
            )
        return _best_by_score(
            state,
            player,
            normal,
            lambda a: (
                _sp_activation_count(state, player, a),
                _special_empty_reduction(state, player, a),
                1 if ps.sp < 3 else 0,
                -_card_point(state, player, a),
                _region_extension_score(state, player, a),
            ),
        )

    # No normal action available: after 3 SP is available, extra SP may be spent.
    if sp_actions:
        usable = [a for a in sp_actions if _card_sp_cost(state, player, a) <= ps.sp]
        if usable:
            return _best_by_score(
                state,
                player,
                usable,
                lambda a: (
                    1 if ps.sp > 3 else 0,
                    _project_score_swing(state, player, a),
                    _sp_activation_count(state, player, a),
                ),
            )

    return legal_actions[0]
