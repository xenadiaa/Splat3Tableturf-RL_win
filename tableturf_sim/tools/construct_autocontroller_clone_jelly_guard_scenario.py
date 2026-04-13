from __future__ import annotations

from copy import deepcopy
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.assets.tableturf_types import Map_PointBit, Map_PointMask
from src.engine.env_core import init_state, legal_actions
from src.engine.loaders import load_map
from src.strategy.autocontroller_clone_jelly_strategy import (
    _apply_action_to_map,
    _max_finisher_enemy_cover,
    choose_action,
)
from src.utils.common_utils import create_card_from_id, validate_place_card_action


MAP_ID = "Square"
MISSILE_ID = 73          # SpMultiMissile
SUCTION_BOMB_ID = 57     # BombSuction
SPLASH_BOMB_ID = 56      # BombSplash
SHAKE_ID = 141           # Assumption for the requested "小鲑鱼"
HAND_IDS = [SUCTION_BOMB_ID, MISSILE_ID, SHAKE_ID, SPLASH_BOMB_ID]
FILLER_IDS = [61, 62, 58, 63, 69, 64, 16, 17, 25, 26, 12]


def _payload_from_action(action) -> dict:
    return {
        "player": action.player,
        "card_number": action.card_number,
        "surrender": action.surrender,
        "pass_turn": action.pass_turn,
        "use_sp_attack": action.use_sp_attack,
        "rotation": action.rotation,
        "x": action.x,
        "y": action.y,
    }


def _render_board(game_map) -> str:
    rows = []
    for y in range(game_map.height):
        chars = []
        for x in range(game_map.width):
            m = int(game_map.get_point(x, y))
            if (m & int(Map_PointBit.IsValid)) == 0:
                chars.append(" ")
            elif (m & int(Map_PointBit.IsP1)) and (m & int(Map_PointBit.IsSp)):
                chars.append("S")
            elif (m & int(Map_PointBit.IsP1)):
                chars.append("o")
            elif (m & int(Map_PointBit.IsP2)) and (m & int(Map_PointBit.IsSp)):
                chars.append("X")
            elif (m & int(Map_PointBit.IsP2)):
                chars.append("x")
            else:
                chars.append(".")
        rows.append(f"{y:02d} {''.join(chars)}")
    return "\n".join(rows)


def _build_base_state():
    deck = HAND_IDS + FILLER_IDS
    state = init_state(MAP_ID, deck, deck, seed=1, mode="2P")
    state.turn = 11
    state.players["P1"].sp = 5
    state.players["P2"].sp = 0
    state.players["P1"].hand = [create_card_from_id(i) for i in HAND_IDS]
    state.players["P1"].draw_pile = []
    state.players["P2"].hand = [create_card_from_id(i) for i in FILLER_IDS[:4]]
    state.players["P2"].draw_pile = []
    state.map = deepcopy(load_map(MAP_ID))

    # Keep a small P1 region alive near the lower side so normal actions remain legal.
    for x in range(10, 13):
        state.map.set_point(x, 16, int(Map_PointMask.P1Normal))
    state.map.set_point(11, 15, int(Map_PointMask.P1Special))
    return state


def _place_enemy_missile(state) -> set[tuple[int, int]]:
    finisher = create_card_from_id(MISSILE_ID)
    ok, _reason, cells = validate_place_card_action(
        finisher,
        state.map,
        anchor_x=11,
        anchor_y=8,
        rotation=2,
        is_p1=False,
        use_sp_attack=False,
    )
    if not ok:
        raise RuntimeError("fixed enemy missile placement is not legal on the current base map")
    for x, y, cell_type in cells:
        state.map.set_point(x, y, int(Map_PointMask.P2Special if cell_type == 2 else Map_PointMask.P2Normal))
    return {(x, y) for x, y, _ in cells}


def _adjacent_candidates(state, enemy_cells: set[tuple[int, int]]) -> list[tuple[int, int]]:
    out: list[tuple[int, int]] = []
    seen = set()
    for ex, ey in sorted(enemy_cells):
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                if dx == 0 and dy == 0:
                    continue
                px, py = ex + dx, ey + dy
                if not (0 <= px < state.map.width and 0 <= py < state.map.height):
                    continue
                if (px, py) in seen:
                    continue
                m = int(state.map.get_point(px, py))
                if (m & int(Map_PointBit.IsValid)) == 0:
                    continue
                if (m & (int(Map_PointBit.IsP1) | int(Map_PointBit.IsP2))) != 0:
                    continue
                seen.add((px, py))
                out.append((px, py))
    return out


def main() -> int:
    state = _build_base_state()
    finisher = create_card_from_id(MISSILE_ID)
    enemy_cells = _place_enemy_missile(state)

    for px, py in _adjacent_candidates(state, enemy_cells):
        candidate = deepcopy(state)
        candidate.map.set_point(px, py, int(Map_PointMask.P1Special))
        current_cover = _max_finisher_enemy_cover(candidate.map, finisher, "P1")
        if current_cover < 12:
            continue

        actions = legal_actions(candidate, "P1")
        payloads = [_payload_from_action(a) for a in actions]
        harmful = []
        for payload in payloads:
            if bool(payload.get("pass_turn", False)):
                continue
            if not bool(payload.get("use_sp_attack", False)):
                continue
            if int(payload.get("card_number") or -1) == MISSILE_ID:
                continue
            cost = create_card_from_id(int(payload["card_number"])).SpecialCost
            if candidate.players["P1"].sp - cost < 3:
                continue
            future_map = _apply_action_to_map(candidate, "P1", payload)
            future_cover = _max_finisher_enemy_cover(future_map, finisher, "P1")
            if future_cover < 12:
                harmful.append((payload, future_cover))

        if not harmful:
            continue

        chosen = choose_action(candidate, "P1", payloads, {})
        print("scenario_found=true")
        print("map_id=Square")
        print("turn=11")
        print("p1_sp=5")
        print("enemy_missile_anchor=(11,8), rotation=2")
        print(f"extra_p1_special=({px},{py})")
        print(f"current_finisher_enemy_cover={current_cover}")
        print(f"harmful_sp_actions={len(harmful)}")
        print("first_harmful_action=", harmful[0][0], "future_cover=", harmful[0][1])
        print("strategy_choice=", chosen)
        print()
        print("Legend: o=P1, S=P1 special, x=P2, X=P2 special, .=empty")
        print(_render_board(candidate.map))
        return 0

    print("scenario_found=false")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
