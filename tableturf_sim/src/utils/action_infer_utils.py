"""Infer player actions from before/after map snapshots."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict
import json
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

from ..assets.tableturf_types import GameMap, Map_PointBit
from ..engine.env_core import Action, BotConfig, GameState, PlayerState, step
from ..engine.loaders import MAP_PADDING, load_map
from .common_utils import create_card_from_id, validate_place_card_action
from .infer_config import infer_mode_flags
from .npc_match_utils import npc_card_pool_by_map


def _all_card_ids() -> List[int]:
    project_root = Path(__file__).resolve().parents[2]
    path = project_root / "data" / "cards" / "MiniGameCardInfo.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    return sorted(int(item["Number"]) for item in data)


def _dedupe_keep_order(values: Sequence[int]) -> List[int]:
    seen = set()
    out: List[int] = []
    for v in values:
        i = int(v)
        if i in seen:
            continue
        seen.add(i)
        out.append(i)
    return out


def _build_state_from_snapshot(
    map_id: str,
    before_grid: List[List[int]],
    p1_cards: List[int],
    p2_cards: List[int],
    p1_sp: int = 0,
    p2_sp: int = 0,
    turn: int = 1,
) -> GameState:
    base_map = load_map(map_id)
    expected_h = base_map.height - MAP_PADDING * 2
    expected_w = base_map.width - MAP_PADDING * 2
    if len(before_grid) != expected_h or any(len(r) != expected_w for r in before_grid):
        raise ValueError(
            f"before_grid shape mismatch for map {map_id}: expected {expected_h}x{expected_w}, "
            f"got {len(before_grid)}x{(len(before_grid[0]) if before_grid else 0)}"
        )
    grid = [row[:] for row in base_map.grid]
    for y, row in enumerate(before_grid):
        for x, val in enumerate(row):
            grid[y + MAP_PADDING][x + MAP_PADDING] = int(val)

    game_map = GameMap(
        map_id=base_map.map_id,
        name=base_map.name,
        ename=base_map.ename,
        width=base_map.width,
        height=base_map.height,
        grid=grid,
    )
    return GameState(
        map=game_map,
        players={
            "P1": PlayerState(
                deck_ids=list(p1_cards),
                draw_pile=[],
                hand=[create_card_from_id(cid) for cid in p1_cards],
                sp=int(p1_sp),
            ),
            "P2": PlayerState(
                deck_ids=list(p2_cards),
                draw_pile=[],
                hand=[create_card_from_id(cid) for cid in p2_cards],
                sp=int(p2_sp),
            ),
        },
        turn=int(turn),
        max_turns=12,
        mode="2P",
        pending_actions={},
        done=False,
        seed=None,
        bot_config=BotConfig(style="aggressive", level="high"),
        bot_nn_spec=None,
        log_path=None,
    )


def _extract_unpadded_grid(game_map: GameMap) -> List[List[int]]:
    return [
        [int(game_map.get_point(x + MAP_PADDING, y + MAP_PADDING)) for x in range(game_map.width - MAP_PADDING * 2)]
        for y in range(game_map.height - MAP_PADDING * 2)
    ]


def _cell_flags(mask: int) -> Dict[str, bool]:
    m = int(mask)
    return {
        "valid": (m & int(Map_PointBit.IsValid)) != 0,
        "p1": (m & int(Map_PointBit.IsP1)) != 0,
        "p2": (m & int(Map_PointBit.IsP2)) != 0,
        "sp": (m & int(Map_PointBit.IsSp)) != 0,
    }


def _diff_sets(
    before_grid: List[List[int]],
    after_grid: List[List[int]],
) -> Dict[str, Set[Tuple[int, int]]]:
    out: Dict[str, Set[Tuple[int, int]]] = {
        "new_fill_p1": set(),
        "new_special_p1": set(),
        "lost_fill_p1": set(),
        "lost_special_p1": set(),
        "new_fill_p2": set(),
        "new_special_p2": set(),
        "lost_fill_p2": set(),
        "lost_special_p2": set(),
        "new_conflict": set(),
        "lost_conflict": set(),
        "changed_p2_to_p1": set(),
        "changed_p1_to_p2": set(),
        "all_changed": set(),
    }
    for y, row in enumerate(before_grid):
        for x, before in enumerate(row):
            after = int(after_grid[y][x])
            before_f = _cell_flags(before)
            after_f = _cell_flags(after)
            if int(before) != int(after):
                out["all_changed"].add((x + MAP_PADDING, y + MAP_PADDING))

            before_conflict = before_f["p1"] and before_f["p2"]
            after_conflict = after_f["p1"] and after_f["p2"]
            if after_conflict and not before_conflict:
                out["new_conflict"].add((x + MAP_PADDING, y + MAP_PADDING))
            if before_conflict and not after_conflict:
                out["lost_conflict"].add((x + MAP_PADDING, y + MAP_PADDING))

            before_p1_only = before_f["p1"] and not before_f["p2"]
            before_p2_only = before_f["p2"] and not before_f["p1"]
            after_p1_only = after_f["p1"] and not after_f["p2"]
            after_p2_only = after_f["p2"] and not after_f["p1"]

            if before_p1_only and before_f["sp"]:
                if not (after_p1_only and after_f["sp"]):
                    out["lost_special_p1"].add((x + MAP_PADDING, y + MAP_PADDING))
            if before_p1_only and not before_f["sp"]:
                if not (after_p1_only and not after_f["sp"]):
                    out["lost_fill_p1"].add((x + MAP_PADDING, y + MAP_PADDING))
            if before_p2_only and before_f["sp"]:
                if not (after_p2_only and after_f["sp"]):
                    out["lost_special_p2"].add((x + MAP_PADDING, y + MAP_PADDING))
            if before_p2_only and not before_f["sp"]:
                if not (after_p2_only and not after_f["sp"]):
                    out["lost_fill_p2"].add((x + MAP_PADDING, y + MAP_PADDING))

            if after_p1_only and after_f["sp"] and not (before_p1_only and before_f["sp"]):
                out["new_special_p1"].add((x + MAP_PADDING, y + MAP_PADDING))
            if after_p1_only and not after_f["sp"] and not (before_p1_only and not before_f["sp"]):
                out["new_fill_p1"].add((x + MAP_PADDING, y + MAP_PADDING))
            if after_p2_only and after_f["sp"] and not (before_p2_only and before_f["sp"]):
                out["new_special_p2"].add((x + MAP_PADDING, y + MAP_PADDING))
            if after_p2_only and not after_f["sp"] and not (before_p2_only and not before_f["sp"]):
                out["new_fill_p2"].add((x + MAP_PADDING, y + MAP_PADDING))

            if before_p2_only and after_p1_only:
                out["changed_p2_to_p1"].add((x + MAP_PADDING, y + MAP_PADDING))
            if before_p1_only and after_p2_only:
                out["changed_p1_to_p2"].add((x + MAP_PADDING, y + MAP_PADDING))
    return out


def _side_diff(diff: Dict[str, Set[Tuple[int, int]]], player: str) -> Dict[str, Set[Tuple[int, int]]]:
    if player == "P1":
        return {
            "new_fill_self": diff["new_fill_p1"],
            "new_special_self": diff["new_special_p1"],
            "lost_fill_self": diff["lost_fill_p1"],
            "lost_special_self": diff["lost_special_p1"],
            "new_fill_enemy": diff["new_fill_p2"],
            "new_special_enemy": diff["new_special_p2"],
            "lost_fill_enemy": diff["lost_fill_p2"],
            "lost_special_enemy": diff["lost_special_p2"],
            "changed_enemy_to_self": diff["changed_p2_to_p1"],
            "changed_self_to_enemy": diff["changed_p1_to_p2"],
            "new_conflict": diff["new_conflict"],
            "all_changed": diff["all_changed"],
        }
    return {
        "new_fill_self": diff["new_fill_p2"],
        "new_special_self": diff["new_special_p2"],
        "lost_fill_self": diff["lost_fill_p2"],
        "lost_special_self": diff["lost_special_p2"],
        "new_fill_enemy": diff["new_fill_p1"],
        "new_special_enemy": diff["new_special_p1"],
        "lost_fill_enemy": diff["lost_fill_p1"],
        "lost_special_enemy": diff["lost_special_p1"],
        "changed_enemy_to_self": diff["changed_p1_to_p2"],
        "changed_self_to_enemy": diff["changed_p2_to_p1"],
        "new_conflict": diff["new_conflict"],
        "all_changed": diff["all_changed"],
    }


def _stage_specs(
    hand: Optional[Sequence[int]],
    deck: Optional[Sequence[int]],
    sp: Optional[int],
    only_sp: bool = False,
) -> List[Dict[str, object]]:
    specs: List[Dict[str, object]] = []
    full_pool = _all_card_ids()

    hand_ids = _dedupe_keep_order(hand or [])
    deck_ids = _dedupe_keep_order(deck or [])
    deck_extra = [cid for cid in deck_ids if cid not in hand_ids]

    def add_stage(label: str, card_ids: Sequence[int], allow_normal: bool, allow_sp: bool, sp_value: int, source: str) -> None:
        ids = _dedupe_keep_order(card_ids)
        if not ids:
            return
        specs.append(
            {
                "label": label,
                "card_ids": ids,
                "allow_normal": allow_normal,
                "allow_sp": allow_sp,
                "sp_value": int(sp_value),
                "source": source,
            }
        )

    inferred_sp = max((create_card_from_id(cid).SpecialCost for cid in (hand_ids or deck_ids or full_pool)), default=10)
    sp_for_search = int(sp) if sp is not None else inferred_sp

    if hand_ids and not only_sp:
        add_stage("hand_normal", hand_ids, True, False, int(sp or 0), "hand")
    add_stage("hand_sp", hand_ids, False, True, sp_for_search, "hand")
    if deck_extra and not only_sp:
        add_stage("deck_normal", deck_extra, True, False, int(sp or 0), "deck")
    add_stage("deck_sp", deck_extra, False, True, sp_for_search, "deck")
    if not only_sp:
        add_stage("all_cards_normal", full_pool, True, False, int(sp or 0), "all_cards")
    add_stage("all_cards_sp", full_pool, False, True, sp_for_search, "all_cards")
    return specs


def _enumerate_actions_for_stage(
    state: GameState,
    player: str,
    stage: Dict[str, object],
    side_diff: Dict[str, Set[Tuple[int, int]]],
    global_diff: Dict[str, Set[Tuple[int, int]]],
) -> List[Dict[str, object]]:
    ps = state.players[player]
    is_p1 = player == "P1"
    ps.sp = int(stage["sp_value"])
    actions: List[Dict[str, object]] = []
    no_self_change = (
        not side_diff["new_fill_self"]
        and not side_diff["new_special_self"]
        and not side_diff["new_conflict"]
    )
    if no_self_change:
        return [
            {
                "action": Action(player=player, card_number=None, pass_turn=True),
                "placed_all": set(),
                "placed_fill": set(),
                "placed_special": set(),
                "special_on_conflict": False,
                "overlap_new_fill_self_count": 0,
                "overlap_new_special_self_count": 0,
                "overlap_new_conflict_count": 0,
                "reason_tags": ["pass_no_self_change"],
            }
        ]
    target_self_diff = (
        set(side_diff["new_fill_self"])
        | set(side_diff["new_special_self"])
        | set(side_diff["new_conflict"])
    )
    both_no_new_special = not global_diff["new_special_p1"] and not global_diff["new_special_p2"]
    has_new_conflict = bool(global_diff["new_conflict"])
    for cid in stage["card_ids"]:
        card = next((c for c in ps.hand if c.Number == int(cid)), None)
        if card is None:
            continue
        card_has_special = any(cell == 2 for row in card.square_2d_0 for cell in row)
        if both_no_new_special and not has_new_conflict and card_has_special:
            continue
        if side_diff["new_special_self"] and not card_has_special:
            continue
        for rot in (0, 1, 2, 3):
            for y in range(state.map.height):
                for x in range(state.map.width):
                    if stage["allow_normal"]:
                        ok_n, reason_n, cells_n = validate_place_card_action(
                            card=card,
                            game_map=state.map,
                            anchor_x=x,
                            anchor_y=y,
                            rotation=rot,
                            is_p1=is_p1,
                            use_sp_attack=False,
                        )
                        if ok_n:
                            action = Action(
                                player=player,
                                card_number=card.Number,
                                pass_turn=False,
                                use_sp_attack=False,
                                rotation=rot,
                                x=x,
                                y=y,
                            )
                            info = _single_action_match_info(action, cells_n, side_diff, global_diff)
                            if info is not None:
                                actions.append(info)
                    if stage["allow_sp"] and ps.sp >= card.SpecialCost:
                        ok_s, reason_s, cells_s = validate_place_card_action(
                            card=card,
                            game_map=state.map,
                            anchor_x=x,
                            anchor_y=y,
                            rotation=rot,
                            is_p1=is_p1,
                            use_sp_attack=True,
                        )
                        if ok_s:
                            action = Action(
                                player=player,
                                card_number=card.Number,
                                pass_turn=False,
                                use_sp_attack=True,
                                rotation=rot,
                                x=x,
                                y=y,
                            )
                            info = _single_action_match_info(action, cells_s, side_diff, global_diff)
                            if info is not None:
                                actions.append(info)
    return actions


def _single_action_match_info(
    action: Action,
    cells: List[Tuple[int, int, int]],
    side_diff: Dict[str, Set[Tuple[int, int]]],
    global_diff: Dict[str, Set[Tuple[int, int]]],
) -> Optional[Dict[str, object]]:
    placed_all = {(x, y) for x, y, _ in cells}
    placed_fill = {(x, y) for x, y, cell_type in cells if cell_type == 1}
    placed_special = {(x, y) for x, y, cell_type in cells if cell_type == 2}
    target_self_diff = set(side_diff["new_fill_self"]) | set(side_diff["new_special_self"]) | set(side_diff["new_conflict"])

    if side_diff["changed_enemy_to_self"] and not action.use_sp_attack:
        return None
    if side_diff["new_special_self"]:
        if not placed_special:
            return None
        if not side_diff["new_special_self"].issubset(placed_special):
            return None
    both_no_new_special = not global_diff["new_special_p1"] and not global_diff["new_special_p2"]
    has_new_conflict = bool(global_diff["new_conflict"])
    if both_no_new_special and not has_new_conflict and placed_special:
        return None
    if target_self_diff and not target_self_diff.issubset(placed_all):
        return None
    if not target_self_diff and side_diff["all_changed"] and placed_all.isdisjoint(side_diff["all_changed"]):
        return None

    special_on_conflict = bool(placed_special & global_diff["new_conflict"])
    if both_no_new_special and has_new_conflict:
        needed_overlap = set(side_diff["new_fill_self"]) | set(side_diff["new_conflict"])
        if needed_overlap and placed_all.isdisjoint(needed_overlap):
            return None

    return {
        "action": action,
        "placed_all": placed_all,
        "placed_fill": placed_fill,
        "placed_special": placed_special,
        "special_on_conflict": special_on_conflict,
        "overlap_new_fill_self_count": len(placed_all & side_diff["new_fill_self"]),
        "overlap_new_special_self_count": len(placed_special & side_diff["new_special_self"]),
        "overlap_new_conflict_count": len(placed_all & side_diff["new_conflict"]),
        "reason_tags": [
            tag
            for tag, ok in [
                ("enemy_to_self_requires_sp", bool(side_diff["changed_enemy_to_self"]) and action.use_sp_attack),
                ("match_new_special", bool(side_diff["new_special_self"]) and bool(placed_special & side_diff["new_special_self"])),
                ("special_on_conflict_possible", special_on_conflict),
                ("conflict_overlap", bool(placed_all & side_diff["new_conflict"])),
            ]
            if ok
        ],
    }


def _jsonable_single_match(info: Dict[str, object]) -> Dict[str, object]:
    out: Dict[str, object] = {}
    for key, value in info.items():
        if key == "action":
            continue
        if isinstance(value, set):
            out[key] = sorted(value)
        else:
            out[key] = value
    return out


def infer_actions_from_map_transition(
    map_id: str,
    before_grid: List[List[int]],
    after_grid: List[List[int]],
    p1_deck: Optional[List[int]] = None,
    p2_deck: Optional[List[int]] = None,
    p1_hand: Optional[List[int]] = None,
    p2_hand: Optional[List[int]] = None,
    p1_sp: Optional[int] = None,
    p2_sp: Optional[int] = None,
    turn: int = 1,
    max_results: int = 128,
) -> Dict[str, object]:
    """Infer legal action pairs from a before/after map transition.

    Required:
    - map_id
    - before_grid
    - after_grid

    Optional search constraints:
    - p1_deck / p2_deck
    - p1_hand / p2_hand
    - p1_sp / p2_sp

    Search order per side:
    1. hand normal
    2. hand sp
    3. deck-extra normal
    4. deck-extra sp
    5. all cards normal
    6. all cards sp
    """
    if not map_id:
        raise ValueError("map_id is required")
    if before_grid is None or after_grid is None:
        raise ValueError("before_grid and after_grid are required")

    p1_base_pool = _dedupe_keep_order((p1_hand or []) + (p1_deck or []))
    p2_base_pool = _dedupe_keep_order((p2_hand or []) + (p2_deck or []))
    if not p1_base_pool:
        p1_base_pool = _all_card_ids()
    if not p2_base_pool:
        p2_base_pool = _all_card_ids()

    state = _build_state_from_snapshot(
        map_id=map_id,
        before_grid=before_grid,
        p1_cards=p1_base_pool,
        p2_cards=p2_base_pool,
        p1_sp=int(p1_sp or 0),
        p2_sp=int(p2_sp or 0),
        turn=turn,
    )
    raw_after = _build_state_from_snapshot(
        map_id=map_id,
        before_grid=after_grid,
        p1_cards=p1_base_pool,
        p2_cards=p2_base_pool,
        p1_sp=int(p1_sp or 0),
        p2_sp=int(p2_sp or 0),
        turn=turn,
    )
    target_grid = _extract_unpadded_grid(raw_after.map)

    diff = _diff_sets(before_grid, after_grid)
    p1_side_diff = _side_diff(diff, "P1")
    p2_side_diff = _side_diff(diff, "P2")
    p1_specs = _stage_specs(p1_hand, p1_deck, p1_sp, only_sp=bool(p1_side_diff["changed_enemy_to_self"]))
    p2_specs = _stage_specs(p2_hand, p2_deck, p2_sp, only_sp=bool(p2_side_diff["changed_enemy_to_self"]))
    matches: List[Dict[str, object]] = []
    searched_pairs = 0

    for p1_stage in p1_specs:
        for p2_stage in p2_specs:
            trial_state = deepcopy(state)
            trial_state.players["P1"].hand = [create_card_from_id(cid) for cid in p1_stage["card_ids"]]
            trial_state.players["P2"].hand = [create_card_from_id(cid) for cid in p2_stage["card_ids"]]
            p1_actions = _enumerate_actions_for_stage(trial_state, "P1", p1_stage, p1_side_diff, diff)
            p2_actions = _enumerate_actions_for_stage(trial_state, "P2", p2_stage, p2_side_diff, diff)
            searched_pairs += len(p1_actions) * len(p2_actions)
            for p1_info in p1_actions:
                for p2_info in p2_actions:
                    p1_action = p1_info["action"]
                    p2_action = p2_info["action"]
                    trial = deepcopy(trial_state)
                    ok1, _, _ = step(trial, p1_action)
                    if not ok1:
                        continue
                    ok2, _, payload = step(trial, p2_action)
                    if not ok2:
                        continue
                    if _extract_unpadded_grid(trial.map) != target_grid:
                        continue
                    matches.append(
                        {
                            "p1_action": asdict(p1_action),
                            "p2_action": asdict(p2_action),
                            "p1_single_match": _jsonable_single_match(p1_info),
                            "p2_single_match": _jsonable_single_match(p2_info),
                            "p1_stage": {k: v for k, v in p1_stage.items() if k != "card_ids"},
                            "p2_stage": {k: v for k, v in p2_stage.items() if k != "card_ids"},
                            "result": payload,
                        }
                    )
                    if len(matches) >= max_results:
                        return {
                            "ok": True,
                            "truncated": True,
                            "searched_action_pairs": searched_pairs,
                            "match_count": len(matches),
                            "matches": matches,
                        }

    return {
        "ok": True,
        "truncated": False,
        "searched_action_pairs": searched_pairs,
        "diff_summary": {k: sorted(v) for k, v in diff.items()},
        "match_count": len(matches),
        "matches": matches,
    }


def infer_other_action_from_known_action(
    map_id: str,
    before_grid: List[List[int]],
    after_grid: List[List[int]],
    known_player: str,
    known_action: Dict[str, object],
    p1_deck: Optional[List[int]] = None,
    p2_deck: Optional[List[int]] = None,
    p1_hand: Optional[List[int]] = None,
    p2_hand: Optional[List[int]] = None,
    p1_sp: Optional[int] = None,
    p2_sp: Optional[int] = None,
    turn: int = 1,
) -> Dict[str, object]:
    """
    Infer the opponent action from a known one-side action plus before/after map snapshots.

    Stub interface for future implementation.
    Current behavior:
    - exposes a stable public API
    - always returns a single fallback candidate: opponent pass
    """
    if known_player not in ("P1", "P2"):
        raise ValueError("known_player must be P1 or P2")
    other_player = "P2" if known_player == "P1" else "P1"
    inferred_pool = None
    used_pool_source = "provided"
    effective_p1_deck = list(p1_deck or [])
    effective_p2_deck = list(p2_deck or [])
    if other_player == "P1" and not effective_p1_deck:
        inferred_pool = npc_card_pool_by_map(map_id)
        effective_p1_deck = [int(x) for x in inferred_pool["card_numbers"]]
        used_pool_source = "map_npc_union"
    elif other_player == "P2" and not effective_p2_deck:
        inferred_pool = npc_card_pool_by_map(map_id)
        effective_p2_deck = [int(x) for x in inferred_pool["card_numbers"]]
        used_pool_source = "map_npc_union"
    return {
        "ok": True,
        "implemented": False,
        "mode": "infer_other_action_from_known_action",
        "map_id": map_id,
        "turn": int(turn),
        "known_player": known_player,
        "known_action": dict(known_action),
        "other_player": other_player,
        "p1_deck_effective": effective_p1_deck,
        "p2_deck_effective": effective_p2_deck,
        "other_action_card_pool_source": used_pool_source,
        "other_action_card_pool": (inferred_pool["card_numbers"] if inferred_pool else (effective_p1_deck if other_player == "P1" else effective_p2_deck)),
        "map_npc_pool_info": inferred_pool,
        "candidates": [
            asdict(Action(player=other_player, card_number=None, pass_turn=True)),
        ],
        "message": "stub only; currently returns opponent pass_turn, but missing opponent deck now falls back to the union of all NPC deck cards on this map",
    }


def infer_map_from_known_actions(
    map_id: str,
    before_grid: List[List[int]],
    after_grid: List[List[int]],
    p1_action: Optional[Dict[str, object]] = None,
    p2_action: Optional[Dict[str, object]] = None,
    p1_deck: Optional[List[int]] = None,
    p2_deck: Optional[List[int]] = None,
    p1_hand: Optional[List[int]] = None,
    p2_hand: Optional[List[int]] = None,
    p1_sp: Optional[int] = None,
    p2_sp: Optional[int] = None,
    turn: int = 1,
) -> Dict[str, object]:
    """
    Infer/validate map transition from known actions.

    Stub interface for future implementation.
    Current behavior:
    - exposes a stable public API
    - returns the provided before/after grids unchanged
    """
    return {
        "ok": True,
        "implemented": False,
        "mode": "infer_map_from_known_actions",
        "map_id": map_id,
        "turn": int(turn),
        "p1_action": dict(p1_action or {}),
        "p2_action": dict(p2_action or {}),
        "before_grid": before_grid,
        "after_grid": after_grid,
        "message": "stub only; currently echoes provided before/after grids",
    }


def infer_deck_from_played_and_hand(
    player: str,
    played_card_numbers: Optional[List[int]] = None,
    hand_card_numbers: Optional[List[int]] = None,
    deck_size: int = 15,
) -> Dict[str, object]:
    """
    Infer a player's deck from observed played cards + current hand.

    Rules:
    - dedupe by card number
    - if unique count == deck_size, deck is fully resolved
    - if unique count < deck_size, deck is partial/incomplete
    - if unique count > deck_size, input is contradictory
    """
    if player not in ("P1", "P2"):
        raise ValueError("player must be P1 or P2")
    played = _dedupe_keep_order(played_card_numbers or [])
    hand = _dedupe_keep_order(hand_card_numbers or [])
    merged = _dedupe_keep_order(list(played) + list(hand))
    if len(merged) > int(deck_size):
        return {
            "ok": False,
            "implemented": True,
            "mode": "infer_deck_from_played_and_hand",
            "player": player,
            "deck_size": int(deck_size),
            "played_card_numbers": played,
            "hand_card_numbers": hand,
            "inferred_deck_ids": merged,
            "resolved": False,
            "error": "UNIQUE_CARD_COUNT_EXCEEDS_DECK_SIZE",
        }
    return {
        "ok": True,
        "implemented": True,
        "mode": "infer_deck_from_played_and_hand",
        "player": player,
        "deck_size": int(deck_size),
        "played_card_numbers": played,
        "hand_card_numbers": hand,
        "unique_count": len(merged),
        "inferred_deck_ids": merged,
        "resolved": len(merged) == int(deck_size),
        "remaining_unknown_count": max(0, int(deck_size) - len(merged)),
        "message": (
            "deck fully resolved"
            if len(merged) == int(deck_size)
            else "deck still incomplete; more played/hand cards required"
        ),
    }


def infer(
    mode: str,
    map_id: str,
    before_grid: Optional[List[List[int]]] = None,
    after_grid: Optional[List[List[int]]] = None,
    p1_deck: Optional[List[int]] = None,
    p2_deck: Optional[List[int]] = None,
    p1_hand: Optional[List[int]] = None,
    p2_hand: Optional[List[int]] = None,
    p1_sp: Optional[int] = None,
    p2_sp: Optional[int] = None,
    turn: int = 1,
    max_results: int = 128,
    known_player: Optional[str] = None,
    known_action: Optional[Dict[str, object]] = None,
    p1_action: Optional[Dict[str, object]] = None,
    p2_action: Optional[Dict[str, object]] = None,
) -> Dict[str, object]:
    """
    Unified inference entry with explicit mode enum.

    Supported modes:
    - map_to_both_actions
    - map_plus_one_action_to_other
    - both_actions_to_map
    - played_plus_hand_to_deck
    """
    flags = infer_mode_flags(mode)

    if not flags["enable_exact"] and not flags["enable_fuzzy"]:
        return {
            "ok": False,
            "mode": mode,
            "implemented": False,
            "error": "INFERENCE_DISABLED_BY_CONFIG",
            "inference_flags": flags,
        }

    if mode == "map_to_both_actions":
        if before_grid is None or after_grid is None:
            raise ValueError("before_grid and after_grid are required for mode map_to_both_actions")
        result = infer_actions_from_map_transition(
            map_id=map_id,
            before_grid=before_grid,
            after_grid=after_grid,
            p1_deck=p1_deck,
            p2_deck=p2_deck,
            p1_hand=p1_hand,
            p2_hand=p2_hand,
            p1_sp=p1_sp,
            p2_sp=p2_sp,
            turn=turn,
            max_results=max_results,
        )
        result["inference_flags"] = flags
        result["inference_path"] = "exact_only" if not flags["enable_fuzzy"] else "exact_then_fuzzy"
        return result
    if mode == "map_plus_one_action_to_other":
        if before_grid is None or after_grid is None:
            raise ValueError("before_grid and after_grid are required for mode map_plus_one_action_to_other")
        if not known_player:
            raise ValueError("known_player is required for mode map_plus_one_action_to_other")
        result = infer_other_action_from_known_action(
            map_id=map_id,
            before_grid=before_grid,
            after_grid=after_grid,
            known_player=known_player,
            known_action=dict(known_action or {}),
            p1_deck=p1_deck,
            p2_deck=p2_deck,
            p1_hand=p1_hand,
            p2_hand=p2_hand,
            p1_sp=p1_sp,
            p2_sp=p2_sp,
            turn=turn,
        )
        result["inference_flags"] = flags
        result["inference_path"] = "exact_only" if not flags["enable_fuzzy"] else "exact_then_fuzzy"
        return result
    if mode == "both_actions_to_map":
        if before_grid is None:
            raise ValueError("before_grid is required for mode both_actions_to_map")
        result = infer_map_from_known_actions(
            map_id=map_id,
            before_grid=before_grid,
            after_grid=after_grid or before_grid,
            p1_action=p1_action,
            p2_action=p2_action,
            p1_deck=p1_deck,
            p2_deck=p2_deck,
            p1_hand=p1_hand,
            p2_hand=p2_hand,
            p1_sp=p1_sp,
            p2_sp=p2_sp,
            turn=turn,
        )
        result["inference_flags"] = flags
        result["inference_path"] = "exact_only" if not flags["enable_fuzzy"] else "exact_then_fuzzy"
        return result
    if mode == "played_plus_hand_to_deck":
        if not known_player:
            raise ValueError("known_player/player is required for mode played_plus_hand_to_deck")
        result = infer_deck_from_played_and_hand(
            player=known_player,
            played_card_numbers=[int(x) for x in (known_action or {}).get("played_card_numbers", [])],
            hand_card_numbers=[int(x) for x in (known_action or {}).get("hand_card_numbers", [])],
            deck_size=int((known_action or {}).get("deck_size", 15)),
        )
        result["inference_flags"] = flags
        result["inference_path"] = "exact_only" if not flags["enable_fuzzy"] else "exact_then_fuzzy"
        return result
    raise ValueError(f"unsupported infer mode: {mode}")


def infer_with_mode(
    mode: str,
    map_id: str,
    before_grid: Optional[List[List[int]]] = None,
    after_grid: Optional[List[List[int]]] = None,
    p1_deck: Optional[List[int]] = None,
    p2_deck: Optional[List[int]] = None,
    p1_hand: Optional[List[int]] = None,
    p2_hand: Optional[List[int]] = None,
    p1_sp: Optional[int] = None,
    p2_sp: Optional[int] = None,
    turn: int = 1,
    max_results: int = 128,
    known_player: Optional[str] = None,
    known_action: Optional[Dict[str, object]] = None,
    p1_action: Optional[Dict[str, object]] = None,
    p2_action: Optional[Dict[str, object]] = None,
) -> Dict[str, object]:
    # Backward-compatible alias.
    return infer(
        mode=mode,
        map_id=map_id,
        before_grid=before_grid,
        after_grid=after_grid,
        p1_deck=p1_deck,
        p2_deck=p2_deck,
        p1_hand=p1_hand,
        p2_hand=p2_hand,
        p1_sp=p1_sp,
        p2_sp=p2_sp,
        turn=turn,
        max_results=max_results,
        known_player=known_player,
        known_action=known_action,
        p1_action=p1_action,
        p2_action=p2_action,
    )
