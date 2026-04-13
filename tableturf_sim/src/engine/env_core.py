"""Core turn engine for Tableturf battle flow.

Features:
- 2P sync submit/resolve
- 1P mode with P2 bot (9 strategies: 3 styles x 3 levels)
- structured turn/event logging (JSONL) for replay/RL usage
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime
import importlib.util
import json
from pathlib import Path
import random
from typing import Dict, List, Optional, Tuple
from uuid import uuid4

from ..assets.tableturf_types import Card_Single, GameMap, Map_PointBit, Map_PointMask
from ..utils.common_utils import (
    _card_cells_on_map,  # internal helper reused here
    activate_special_points_and_gain_sp,
    create_card_from_id,
    validate_place_card_action,
)
from .loaders import load_map


MAX_TURNS = 12
PLAYER_IDS = ("P1", "P2")
BOT_STYLES = ("balanced", "aggressive", "conservative")
BOT_LEVELS = ("low", "mid", "high")


@dataclass
class BotConfig:
    style: str = "balanced"   # balanced/aggressive/conservative
    level: str = "mid"        # low/mid/high


@dataclass
class PlayerState:
    deck_ids: List[int]
    draw_pile: List[Card_Single]
    hand: List[Card_Single]
    sp: int = 0


@dataclass
class Action:
    player: str
    card_number: Optional[int] = None
    surrender: bool = False
    pass_turn: bool = False
    use_sp_attack: bool = False
    rotation: int = 0
    x: Optional[int] = None
    y: Optional[int] = None


@dataclass
class GameState:
    map: GameMap
    players: Dict[str, PlayerState]
    turn: int = 1
    max_turns: int = MAX_TURNS
    mode: str = "2P"
    pending_actions: Dict[str, Action] = field(default_factory=dict)
    done: bool = False
    winner: Optional[str] = None
    reason: Optional[str] = None
    seed: Optional[int] = None
    rng: random.Random = field(default_factory=random.Random)
    bot_config: BotConfig = field(default_factory=BotConfig)
    bot_nn_spec: Optional[Dict[str, object]] = None
    log_path: Optional[str] = None
    event_seq: int = 0


def _default_log_path(seed: Optional[int]) -> str:
    # kept for backward compatibility; prefer _build_log_path(...)
    root = Path(__file__).resolve().parent.parent.parent
    log_dir = root / "data" / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    suffix = uuid4().hex[:8]
    return str(log_dir / f"Tableturf_{ts}_{suffix}.jsonl")


def _safe_name(name: str, max_len: int = 16) -> str:
    raw = "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in str(name))
    raw = raw.strip("_") or "NA"
    return raw[:max_len]


def _build_log_path(map_id: str, p1_name: str, p2_name: str) -> str:
    root = Path(__file__).resolve().parent.parent.parent
    log_dir = root / "data" / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    suffix = uuid4().hex[:8]
    fname = (
        f"Tableturf_{ts}_{_safe_name(map_id,12)}_{_safe_name(p1_name,12)}_{_safe_name(p2_name,12)}_{suffix}.jsonl"
    )
    return str(log_dir / fname)


def _log_event(state: GameState, event: str, payload: Dict[str, object]) -> None:
    if not state.log_path:
        return
    state.event_seq += 1
    record = {
        "seq": state.event_seq,
        "utc": datetime.utcnow().isoformat(timespec="seconds") + "Z",
        "event": event,
        "turn": state.turn,
        "payload": payload,
    }
    with open(state.log_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def _write_log_meta_once(state: GameState, meta: Dict[str, object]) -> None:
    if not state.log_path:
        return
    with open(state.log_path, "w", encoding="utf-8") as f:
        f.write(
            json.dumps(
                {
                    "seq": 0,
                    "utc": datetime.utcnow().isoformat(timespec="seconds") + "Z",
                    "event": "meta",
                    "payload": meta,
                },
                ensure_ascii=False,
            )
            + "\n"
        )


def _validate_deck_ids(deck_ids: List[int]) -> None:
    if len(deck_ids) != 15:
        raise ValueError(f"deck must have exactly 15 cards, got {len(deck_ids)}")
    if len(set(deck_ids)) != len(deck_ids):
        raise ValueError("deck has duplicated card numbers")


def _validate_mode(mode: str) -> None:
    if mode not in ("1P", "2P"):
        raise ValueError(f"unsupported mode: {mode}")


def _validate_bot_config(bot_config: BotConfig) -> None:
    if bot_config.style not in BOT_STYLES:
        raise ValueError(f"unsupported bot style: {bot_config.style}")
    if bot_config.level not in BOT_LEVELS:
        raise ValueError(f"unsupported bot level: {bot_config.level}")


def _build_player_state(deck_ids: List[int], rng: random.Random) -> PlayerState:
    cards = [create_card_from_id(cid) for cid in deck_ids]
    rng.shuffle(cards)
    hand = cards[:4]
    draw_pile = cards[4:]
    return PlayerState(deck_ids=list(deck_ids), draw_pile=draw_pile, hand=hand, sp=0)


def init_state(
    map_id: str,
    p1_deck_ids: List[int],
    p2_deck_ids: List[int],
    seed: Optional[int] = None,
    mode: str = "2P",
    bot_style: str = "balanced",
    bot_level: str = "mid",
    bot_nn_spec: Optional[Dict[str, object]] = None,
    log_path: Optional[str] = None,
    p1_player_id: str = "P1",
    p2_player_id: str = "P2",
    p1_player_name: str = "P1",
    p2_player_name: str = "P2",
    p1_deck_name: Optional[str] = None,
    p2_deck_name: Optional[str] = None,
) -> GameState:
    """Initialize game state and draw 4 random cards for each player."""
    _validate_deck_ids(p1_deck_ids)
    _validate_deck_ids(p2_deck_ids)
    _validate_mode(mode)
    bot_config = BotConfig(style=bot_style, level=bot_level)
    _validate_bot_config(bot_config)

    rng = random.Random(seed)
    game_map = load_map(map_id)
    players = {
        "P1": _build_player_state(p1_deck_ids, rng),
        "P2": _build_player_state(p2_deck_ids, rng),
    }
    state = GameState(
        map=game_map,
        players=players,
        turn=1,
        max_turns=MAX_TURNS,
        mode=mode,
        pending_actions={},
        done=False,
        seed=seed,
        rng=rng,
        bot_config=bot_config,
        bot_nn_spec=bot_nn_spec,
        log_path=log_path or _build_log_path(map_id, p1_player_name, p2_player_name),
    )
    _write_log_meta_once(
        state,
        {
            "map_id": map_id,
            "mode": mode,
            "seed": seed,
            "log_path": state.log_path,
            "bot_strategy": {
                "style": bot_style if mode == "1P" else "",
                "level": bot_level if mode == "1P" else "",
                "nn_spec": bot_nn_spec if mode == "1P" else {},
            },
            "players": {
                "P1": {
                    "player_id": p1_player_id,
                    "player_name": p1_player_name,
                    "deck_name": p1_deck_name or "",
                    "deck_ids": list(p1_deck_ids),
                },
                "P2": {
                    "player_id": p2_player_id,
                    "player_name": p2_player_name,
                    "deck_name": p2_deck_name or "",
                    "deck_ids": list(p2_deck_ids),
                },
            },
        },
    )
    _log_event(
        state,
        "init_state",
        {
            "p1_hand": [c.Number for c in state.players["P1"].hand],
            "p2_hand": [c.Number for c in state.players["P2"].hand],
        },
    )
    return state


def _find_card_in_hand(player_state: PlayerState, card_number: Optional[int]) -> Optional[Card_Single]:
    if card_number is None:
        return None
    for c in player_state.hand:
        if c.Number == card_number:
            return c
    return None


def _to_owner_mask(player: str, cell_type: int) -> int:
    if player == "P1":
        return int(Map_PointMask.P1Special if cell_type == 2 else Map_PointMask.P1Normal)
    return int(Map_PointMask.P2Special if cell_type == 2 else Map_PointMask.P2Normal)


def _priority(card_point: int, is_special_cell: bool, use_sp_attack: bool) -> float:
    # 地格归属只与卡牌点数、是否为 special 格有关，
    # 与本次是否使用 SP 攻击无关。保留 use_sp_attack 参数仅为了兼容调用点。
    _ = use_sp_attack
    return 30.0 - card_point + (30.0 if is_special_cell else 0.0)


def _resolve_contested_cell(
    p_action: Action,
    p_card: Card_Single,
    p_cell_type: int,
    e_action: Action,
    e_card: Card_Single,
    e_cell_type: int,
) -> int:
    # special 仅可被 special 顶掉
    if p_cell_type == 2 and e_cell_type != 2:
        return _to_owner_mask("P1", 2)
    if e_cell_type == 2 and p_cell_type != 2:
        return _to_owner_mask("P2", 2)

    p_score = _priority(p_card.CardPoint, p_cell_type == 2, p_action.use_sp_attack)
    e_score = _priority(e_card.CardPoint, e_cell_type == 2, e_action.use_sp_attack)
    eps = 1e-6
    if abs(p_score - e_score) < eps:
        if p_cell_type == 2 and e_cell_type == 2:
            return int(Map_PointMask.ConflictSp)
        return int(Map_PointMask.Conflict)
    if p_score > e_score:
        return _to_owner_mask("P1", p_cell_type)
    return _to_owner_mask("P2", e_cell_type)


def _apply_action_effect(state: GameState, player: str, action: Action) -> Tuple[Optional[Card_Single], List[Tuple[int, int, int]]]:
    ps = state.players[player]
    card = _find_card_in_hand(ps, action.card_number)
    if card is None:
        return None, []
    if action.pass_turn:
        return card, []
    cells = _card_cells_on_map(card, int(action.x), int(action.y), action.rotation)
    return card, cells


def _remove_card_from_hand_and_draw(ps: PlayerState, card_number: int) -> None:
    idx = next((i for i, c in enumerate(ps.hand) if c.Number == card_number), None)
    if idx is None:
        raise ValueError(f"card #{card_number} not found in hand")
    if ps.draw_pile:
        # Keep 0/1/2/3 hand slot stable: replace in-place instead of shifting.
        ps.hand[idx] = ps.draw_pile.pop(0)
    else:
        # No draw remains (endgame), remove card as fallback.
        ps.hand.pop(idx)


def _compute_scores(game_map: GameMap) -> Tuple[int, int]:
    p1 = 0
    p2 = 0
    for y in range(game_map.height):
        for x in range(game_map.width):
            m = int(game_map.get_point(x, y))
            is_valid = (m & int(Map_PointBit.IsValid)) != 0
            if not is_valid:
                continue
            is_p1 = (m & int(Map_PointBit.IsP1)) != 0
            is_p2 = (m & int(Map_PointBit.IsP2)) != 0
            if is_p1 and not is_p2:
                p1 += 1
            elif is_p2 and not is_p1:
                p2 += 1
    return p1, p2


def _update_done_and_winner(state: GameState) -> None:
    if state.turn > state.max_turns:
        state.done = True
    if not state.done:
        return
    p1, p2 = _compute_scores(state.map)
    if p1 > p2:
        state.winner = "P1"
    elif p2 > p1:
        state.winner = "P2"
    else:
        state.winner = "draw"


def validate_action(state: GameState, action: Action) -> Tuple[bool, str]:
    if state.done:
        return False, "GAME_ALREADY_DONE"
    if action.player not in PLAYER_IDS:
        return False, "INVALID_PLAYER"
    if action.player in state.pending_actions:
        return False, "PLAYER_ALREADY_SUBMITTED"

    if action.surrender:
        return True, "OK"

    ps = state.players[action.player]
    card = _find_card_in_hand(ps, action.card_number)
    if card is None:
        return False, "CARD_NOT_IN_HAND"

    if action.pass_turn:
        return True, "OK"

    if action.x is None or action.y is None:
        return False, "MISSING_POSITION"
    if action.rotation not in (0, 1, 2, 3):
        return False, "INVALID_ROTATION"

    if action.use_sp_attack and ps.sp < card.SpecialCost:
        return False, "SP_NOT_ENOUGH"

    ok, reason, _ = validate_place_card_action(
        card=card,
        game_map=state.map,
        anchor_x=int(action.x),
        anchor_y=int(action.y),
        rotation=action.rotation,
        is_p1=(action.player == "P1"),
        use_sp_attack=action.use_sp_attack,
    )
    return ok, reason


def submit_action(state: GameState, action: Action) -> Tuple[bool, str]:
    ok, reason = validate_action(state, action)
    if not ok:
        _log_event(state, "submit_rejected", {"player": action.player, "reason": reason, "action": asdict(action)})
        return False, reason
    state.pending_actions[action.player] = action
    _log_event(state, "submit_accepted", {"player": action.player, "action": asdict(action)})
    return True, "OK"


def _resolve_surrender(state: GameState, action: Action) -> Dict[str, object]:
    """立即处理投降，直接结束对局。"""
    loser = action.player
    winner = "P2" if loser == "P1" else "P1"
    state.done = True
    state.reason = "SURRENDER"
    state.winner = winner
    state.pending_actions.clear()
    p1_score, p2_score = _compute_scores(state.map)
    result = {
        "turn": state.turn,
        "reason": "SURRENDER",
        "loser": loser,
        "winner": winner,
        "p1_score": p1_score,
        "p2_score": p2_score,
        "done": True,
    }
    _log_event(
        state,
        "surrender_resolved",
        {
            "action": asdict(action),
            "result": result,
            "p1_hand": [c.Number for c in state.players["P1"].hand],
            "p2_hand": [c.Number for c in state.players["P2"].hand],
        },
    )
    return result


def _resolve_turn(state: GameState) -> Dict[str, object]:
    p_action = state.pending_actions.get("P1")
    e_action = state.pending_actions.get("P2")
    if p_action is None or e_action is None:
        raise RuntimeError("resolve_turn requires both P1 and P2 actions")

    p_card, p_cells = _apply_action_effect(state, "P1", p_action)
    e_card, e_cells = _apply_action_effect(state, "P2", e_action)
    if p_card is None or e_card is None:
        raise RuntimeError("card not found while resolving turn")

    p_by_pos = {(x, y): cell_type for x, y, cell_type in p_cells}
    e_by_pos = {(x, y): cell_type for x, y, cell_type in e_cells}
    all_pos = set(p_by_pos.keys()) | set(e_by_pos.keys())

    # 应用格子归属
    for x, y in all_pos:
        p_cell = p_by_pos.get((x, y))
        e_cell = e_by_pos.get((x, y))
        if p_cell is not None and e_cell is not None:
            new_mask = _resolve_contested_cell(
                p_action=p_action,
                p_card=p_card,
                p_cell_type=p_cell,
                e_action=e_action,
                e_card=e_card,
                e_cell_type=e_cell,
            )
        elif p_cell is not None:
            new_mask = _to_owner_mask("P1", p_cell)
        else:
            new_mask = _to_owner_mask("P2", e_cell)  # type: ignore[arg-type]
        state.map.set_point(x, y, int(new_mask))

    # SP 变化：先扣SP/跳过加SP，再做special激活加SP
    if p_action.pass_turn:
        state.players["P1"].sp += 1
    elif p_action.use_sp_attack:
        state.players["P1"].sp -= p_card.SpecialCost
    if e_action.pass_turn:
        state.players["P2"].sp += 1
    elif e_action.use_sp_attack:
        state.players["P2"].sp -= e_card.SpecialCost

    p1_gain = activate_special_points_and_gain_sp(state.map, is_p1=True)
    p2_gain = activate_special_points_and_gain_sp(state.map, is_p1=False)
    state.players["P1"].sp += p1_gain
    state.players["P2"].sp += p2_gain

    # 移除使用/弃置的手牌并抽1张
    _remove_card_from_hand_and_draw(state.players["P1"], p_action.card_number)
    _remove_card_from_hand_and_draw(state.players["P2"], e_action.card_number)

    state.pending_actions.clear()
    state.turn += 1
    _update_done_and_winner(state)

    p1_score, p2_score = _compute_scores(state.map)
    result = {
        "turn": state.turn,
        "p1_sp_gain": p1_gain,
        "p2_sp_gain": p2_gain,
        "p1_sp": state.players["P1"].sp,
        "p2_sp": state.players["P2"].sp,
        "p1_score": p1_score,
        "p2_score": p2_score,
        "done": state.done,
        "winner": state.winner,
    }
    _log_event(
        state,
        "turn_resolved",
        {
            "p1_action": asdict(p_action),
            "p2_action": asdict(e_action),
            "result": result,
            "p1_hand": [c.Number for c in state.players["P1"].hand],
            "p2_hand": [c.Number for c in state.players["P2"].hand],
        },
    )
    return result


def _owner_cells(state: GameState, is_p1: bool) -> List[Tuple[int, int, bool]]:
    owner_bit = int(Map_PointBit.IsP1) if is_p1 else int(Map_PointBit.IsP2)
    out: List[Tuple[int, int, bool]] = []
    for y in range(state.map.height):
        for x in range(state.map.width):
            m = int(state.map.get_point(x, y))
            if (m & owner_bit) == 0:
                continue
            is_sp = (m & int(Map_PointBit.IsSp)) != 0
            out.append((x, y, is_sp))
    return out


def _centroid(cells: List[Tuple[int, int, bool]], width: int, height: int) -> Tuple[float, float]:
    if not cells:
        return (width / 2.0, height / 2.0)
    sx = sum(x for x, _, _ in cells)
    sy = sum(y for _, y, _ in cells)
    n = len(cells)
    return (sx / n, sy / n)


def _action_cells(action: Action, card: Card_Single) -> List[Tuple[int, int, int]]:
    if action.pass_turn or action.x is None or action.y is None:
        return []
    return _card_cells_on_map(card, action.x, action.y, action.rotation)


def _shape_span(card: Card_Single, rotation: int) -> int:
    mat = card.get_square_matrix(rotation)
    xs: List[int] = []
    ys: List[int] = []
    for y in range(8):
        for x in range(8):
            if mat[y][x] != 0:
                xs.append(x)
                ys.append(y)
    if not xs:
        return 0
    return max(max(xs) - min(xs) + 1, max(ys) - min(ys) + 1)


def _highest_point_actions(actions: List[Action], player_state: PlayerState) -> List[Action]:
    card_point_by_no = {c.Number: c.CardPoint for c in player_state.hand}
    place_actions = [a for a in actions if not a.pass_turn and a.card_number in card_point_by_no]
    if not place_actions:
        return actions
    max_point = max(card_point_by_no[a.card_number] for a in place_actions)
    top = [a for a in place_actions if card_point_by_no[a.card_number] == max_point]
    return top or actions


def _suppress_sp_attack_in_early_mid(actions: List[Action], turn: int) -> List[Action]:
    # In early/mid game, if a normal placement exists, do not use SP attack.
    if turn > 8:
        return actions
    normal_actions = [a for a in actions if not a.pass_turn and not a.use_sp_attack]
    return normal_actions or actions


def _card_has_special(card: Card_Single) -> bool:
    return any(cell == 2 for row in card.square_2d_0 for cell in row)


def _is_no_special_12(card: Card_Single) -> bool:
    return card.CardPoint == 12 and not _card_has_special(card)


def _aggressive_low_point_threshold() -> int:
    return 6


def _choose_aggressive_reserved_card(ps: PlayerState) -> Optional[Card_Single]:
    deck_cards = [create_card_from_id(cid) for cid in ps.deck_ids]
    deck_targets = [c for c in deck_cards if _is_no_special_12(c)]
    if len(deck_targets) < 2:
        return None
    hand_targets = [c for c in ps.hand if _is_no_special_12(c)]
    if not hand_targets:
        return None
    # Prefer keeping the 3-SP finisher in hand for the last turn.
    hand_targets.sort(key=lambda c: (abs(c.SpecialCost - 3), c.SpecialCost, c.Number))
    return hand_targets[0]


def _filter_aggressive_actions(state: GameState, player: str, actions: List[Action]) -> List[Action]:
    ps = state.players[player]
    filtered = list(actions)
    reserved = _choose_aggressive_reserved_card(ps)

    # Keep at least one 12-point no-special card in hand until the last two turns.
    if state.turn < state.max_turns - 1 and reserved is not None:
        reserve_ids = {c.Number for c in ps.hand if _is_no_special_12(c)}
        if len(reserve_ids) <= 1:
            kept = [a for a in filtered if a.pass_turn or a.card_number not in reserve_ids]
        else:
            kept = [a for a in filtered if a.pass_turn or a.card_number != reserved.Number]
        if kept:
            filtered = kept

    # Bank SP until the final two turns. Before that, only spend the surplus above 3 SP,
    # and skip 1-SP attacks entirely.
    if state.turn < state.max_turns - 1:
        normal_or_pass = [a for a in filtered if a.pass_turn or not a.use_sp_attack]
        normal_cards = {a.card_number for a in filtered if not a.pass_turn and not a.use_sp_attack}
        sp_ok: List[Action] = []
        if ps.sp > 3:
            for a in filtered:
                if a.pass_turn or not a.use_sp_attack:
                    continue
                card = _find_card_in_hand(ps, a.card_number)
                if card is None:
                    continue
                if card.CardPoint <= _aggressive_low_point_threshold():
                    cells = _action_cells(a, card)
                    exc_value = _small_card_sp_exception_value(state, player, a, card, cells)
                    if a.card_number in normal_cards or exc_value < 2.0:
                        continue
                    if ps.sp - card.SpecialCost >= 3:
                        sp_ok.append(a)
                    continue
                if card.SpecialCost <= 1:
                    continue
                if ps.sp - card.SpecialCost >= 3:
                    sp_ok.append(a)
        filtered = normal_or_pass + sp_ok

    # Small cards should be used to set up/activate special points through normal placement,
    # not SP attacks, whenever they can be placed normally.
    normal_cards = {a.card_number for a in filtered if not a.pass_turn and not a.use_sp_attack}
    kept = []
    for a in filtered:
        if a.pass_turn or not a.use_sp_attack:
            kept.append(a)
            continue
        card = _find_card_in_hand(ps, a.card_number)
        if card is None:
            kept.append(a)
            continue
        if card.CardPoint <= _aggressive_low_point_threshold() and a.card_number in normal_cards:
            continue
        kept.append(a)
    filtered = kept
    return filtered or actions


def _empty_neighbors_after_action(state: GameState, placed: set[Tuple[int, int]], x: int, y: int) -> Tuple[int, int]:
    empty_neighbors_before = 0
    empty_neighbors_after = 0
    for nx, ny in _neighbors8(x, y):
        if not (0 <= nx < state.map.width and 0 <= ny < state.map.height):
            continue
        nm = int(state.map.get_point(nx, ny))
        if (nm & int(Map_PointBit.IsValid)) == 0:
            continue
        occupied_before = (nm & (int(Map_PointBit.IsP1) | int(Map_PointBit.IsP2))) != 0
        occupied_after = occupied_before or ((nx, ny) in placed)
        if not occupied_before:
            empty_neighbors_before += 1
        if not occupied_after:
            empty_neighbors_after += 1
    return empty_neighbors_before, empty_neighbors_after


def _project_score_swing(state: GameState, player: str, action: Action, cells: List[Tuple[int, int, int]]) -> float:
    own_bit = int(Map_PointBit.IsP1) if player == "P1" else int(Map_PointBit.IsP2)
    opp_bit = int(Map_PointBit.IsP2) if player == "P1" else int(Map_PointBit.IsP1)
    swing = 0.0
    for x, y, _cell_type in cells:
        m = int(state.map.get_point(x, y))
        had_own = (m & own_bit) != 0
        had_opp = (m & opp_bit) != 0
        if had_opp and not had_own:
            swing += 2.0
        elif not had_own and not had_opp:
            swing += 1.0
    return swing


def _maximize_sp_swing_actions(state: GameState, player: str, actions: List[Action]) -> List[Action]:
    sp_actions = [a for a in actions if not a.pass_turn and a.use_sp_attack]
    if not sp_actions:
        return actions
    ps = state.players[player]
    best_swing = None
    best_sp: List[Action] = []
    for action in sp_actions:
        card = _find_card_in_hand(ps, action.card_number)
        if card is None:
            continue
        cells = _action_cells(action, card)
        swing = _project_score_swing(state, player, action, cells)
        if best_swing is None or swing > best_swing:
            best_swing = swing
            best_sp = [action]
        elif swing == best_swing:
            best_sp.append(action)
    if best_swing is None:
        return actions
    non_sp = [a for a in actions if a.pass_turn or not a.use_sp_attack]
    return non_sp + best_sp


def _final_turn_actions(state: GameState, player: str, actions: List[Action]) -> List[Action]:
    if state.turn < state.max_turns:
        return actions
    ps = state.players[player]
    sp_actions = [a for a in actions if not a.pass_turn and a.use_sp_attack]
    if sp_actions:
        best_swing = None
        best_sp: List[Action] = []
        for action in sp_actions:
            card = _find_card_in_hand(ps, action.card_number)
            if card is None:
                continue
            swing = _project_score_swing(state, player, action, _action_cells(action, card))
            if best_swing is None or swing > best_swing:
                best_swing = swing
                best_sp = [action]
            elif swing == best_swing:
                best_sp.append(action)
        if best_swing is None:
            return actions
        normal_actions = [a for a in actions if not a.pass_turn and not a.use_sp_attack]
        better_normals: List[Action] = []
        for action in normal_actions:
            card = _find_card_in_hand(ps, action.card_number)
            if card is None:
                continue
            swing = _project_score_swing(state, player, action, _action_cells(action, card))
            if swing > best_swing:
                better_normals.append(action)
        return (best_sp + better_normals) or actions

    normal_actions = [a for a in actions if not a.pass_turn and not a.use_sp_attack]
    if not normal_actions:
        return actions
    card_point_by_no = {c.Number: c.CardPoint for c in ps.hand}
    max_point = max(card_point_by_no.get(a.card_number, -1) for a in normal_actions)
    top = [a for a in normal_actions if card_point_by_no.get(a.card_number, -1) == max_point]
    return top or actions


def _neighbors8(x: int, y: int) -> List[Tuple[int, int]]:
    out: List[Tuple[int, int]] = []
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            if dx == 0 and dy == 0:
                continue
            out.append((x + dx, y + dy))
    return out


def _valid_map_bit(m: int) -> bool:
    return (m & int(Map_PointBit.IsValid)) != 0


def _aggressive_action_metrics(
    state: GameState,
    player: str,
    action: Action,
    cells: List[Tuple[int, int, int]],
) -> Dict[str, float]:
    opp_bit = int(Map_PointBit.IsP2) if player == "P1" else int(Map_PointBit.IsP1)
    own_bit = int(Map_PointBit.IsP1) if player == "P1" else int(Map_PointBit.IsP2)

    touch_enemy = 0
    surround_enemy = 0
    own_fill_hole = 0
    sp_enemy_cover = 0

    for x, y, _cell_type in cells:
        here = int(state.map.get_point(x, y))
        if action.use_sp_attack and (here & opp_bit) != 0 and (here & int(Map_PointBit.IsSp)) == 0:
            sp_enemy_cover += 1

        enemy_adj = 0
        own_adj = 0
        for nx, ny in _neighbors8(x, y):
            if not (0 <= nx < state.map.width and 0 <= ny < state.map.height):
                continue
            m = int(state.map.get_point(nx, ny))
            if not _valid_map_bit(m):
                continue
            if (m & opp_bit) != 0:
                enemy_adj += 1
            elif (m & own_bit) != 0:
                own_adj += 1

        if enemy_adj > 0:
            touch_enemy += 1
        if enemy_adj >= 2:
            surround_enemy += 1
        if enemy_adj == 0 and own_adj >= 5:
            own_fill_hole += 1

    return {
        "touch_enemy": float(touch_enemy),
        "surround_enemy": float(surround_enemy),
        "own_fill_hole": float(own_fill_hole),
        "sp_enemy_cover": float(sp_enemy_cover),
    }


def _sp_setup_potential(state: GameState, player: str, cells: List[Tuple[int, int, int]]) -> float:
    own_bit = int(Map_PointBit.IsP1) if player == "P1" else int(Map_PointBit.IsP2)
    placed = {(x, y) for x, y, _ in cells}
    gain = 0.0
    for y in range(state.map.height):
        for x in range(state.map.width):
            m = int(state.map.get_point(x, y))
            if (m & own_bit) == 0 or (m & int(Map_PointBit.IsSp)) == 0:
                continue
            if (m & int(Map_PointBit.IsSupplySp)) != 0:
                continue
            blocked = True
            for nx, ny in _neighbors8(x, y):
                if not (0 <= nx < state.map.width and 0 <= ny < state.map.height):
                    continue
                nm = int(state.map.get_point(nx, ny))
                if (nx, ny) in placed:
                    continue
                if (nm & int(Map_PointBit.IsValid)) == 0:
                    continue
                if (nm & (int(Map_PointBit.IsP1) | int(Map_PointBit.IsP2))) == 0:
                    blocked = False
                    break
            if blocked:
                gain += 1.0
    return gain


def _sp_activation_counts(state: GameState, player: str, cells: List[Tuple[int, int, int]]) -> int:
    own_bit = int(Map_PointBit.IsP1) if player == "P1" else int(Map_PointBit.IsP2)
    placed = {(x, y) for x, y, _ in cells}
    activated = 0
    for y in range(state.map.height):
        for x in range(state.map.width):
            m = int(state.map.get_point(x, y))
            if (m & own_bit) == 0 or (m & int(Map_PointBit.IsSp)) == 0:
                continue
            if (m & int(Map_PointBit.IsSupplySp)) != 0:
                continue
            empty_neighbors_before, empty_neighbors_after = _empty_neighbors_after_action(state, placed, x, y)
            if empty_neighbors_before > 0 and empty_neighbors_after == 0:
                activated += 1
    for x, y, cell_type in cells:
        if cell_type != 2:
            continue
        empty_neighbors_before, empty_neighbors_after = _empty_neighbors_after_action(state, placed, x, y)
        if empty_neighbors_before > 0 and empty_neighbors_after == 0:
            activated += 1
    return activated


def _new_special_fully_enclosed(state: GameState, cells: List[Tuple[int, int, int]]) -> int:
    placed = {(x, y) for x, y, _ in cells}
    enclosed = 0
    for x, y, cell_type in cells:
        if cell_type != 2:
            continue
        _before, after = _empty_neighbors_after_action(state, placed, x, y)
        if after == 0:
            enclosed += 1
    return enclosed


def _small_card_sp_exception_value(
    state: GameState,
    player: str,
    action: Action,
    card: Card_Single,
    cells: List[Tuple[int, int, int]],
) -> float:
    metrics = _aggressive_action_metrics(state, player, action, cells)
    activated = _sp_activation_counts(state, player, cells)
    enclosed_new_special = _new_special_fully_enclosed(state, cells)
    net_sp_gain = activated - card.SpecialCost
    return (2.0 * net_sp_gain) + (metrics["sp_enemy_cover"] / 3.0) + (1.0 if enclosed_new_special > 0 else 0.0)


def _score_bot_action(
    state: GameState,
    player: str,
    action: Action,
    style: str,
    level: str,
    aggressive_ctx: Optional[Dict[str, object]] = None,
) -> float:
    # Temporary policy routing: conservative shares aggressive heuristic
    # until its dedicated logic is rewritten.
    if style == "conservative":
        style = "aggressive"

    ps = state.players[player]
    card = _find_card_in_hand(ps, action.card_number)
    if card is None:
        return -1e9
    if action.pass_turn:
        # 允许但默认弱于出牌；保守风格略放宽
        base = -8.0 if style != "conservative" else -3.0
        return base + (2.0 if ps.sp <= 2 else 0.0)

    own_cells = _owner_cells(state, is_p1=(player == "P1"))
    opp_cells = _owner_cells(state, is_p1=(player != "P1"))
    own_cx, own_cy = _centroid(own_cells, state.map.width, state.map.height)
    opp_cx, opp_cy = _centroid(opp_cells, state.map.width, state.map.height)

    cells = _action_cells(action, card)
    if not cells:
        return -1e9
    act_cx = sum(x for x, _, _ in cells) / len(cells)
    act_cy = sum(y for _, y, _ in cells) / len(cells)
    vec_x = opp_cx - own_cx
    vec_y = opp_cy - own_cy
    norm = (vec_x * vec_x + vec_y * vec_y) ** 0.5 or 1.0
    ux, uy = vec_x / norm, vec_y / norm
    adv = (act_cx - own_cx) * ux + (act_cy - own_cy) * uy  # 向对手方向推进值

    special_count = sum(1 for _, _, t in cells if t == 2)
    span = _shape_span(card, action.rotation)
    point = card.CardPoint
    sp_cost = card.SpecialCost

    turn_phase = "early" if state.turn <= 4 else ("mid" if state.turn <= 8 else "late")
    lvl = {"low": 0.7, "mid": 1.0, "high": 1.4}[level]
    is_final_turn = state.turn >= state.max_turns

    if style == "aggressive":
        metrics = _aggressive_action_metrics(state, player, action, cells)
        normal_exists = bool(aggressive_ctx.get("normal_exists", True)) if aggressive_ctx else True
        sp_setup_gain = _sp_setup_potential(state, player, cells)
        sp_activated = _sp_activation_counts(state, player, cells)
        reserved_no_special12 = bool(aggressive_ctx.get("reserved_no_special12", False)) if aggressive_ctx else False
        if is_final_turn:
            # Final turn: maximize immediate score swing first, then use point/SP traits as tie-breakers.
            swing = _project_score_swing(state, player, action, cells)
            score = (12.0 * lvl) * swing + (3.5 * lvl) * point + (1.0 * lvl) * span + 0.3 * special_count
            score += 1.5 * metrics["touch_enemy"] + 1.2 * metrics["surround_enemy"]
            if action.use_sp_attack:
                score += (4.0 * lvl) + (4.0 * lvl) * metrics["sp_enemy_cover"] - 0.05 * sp_cost
                if reserved_no_special12 and point == 12 and special_count == 0 and ps.sp >= 3:
                    score += (16.0 * lvl) + (8.0 * lvl) * metrics["sp_enemy_cover"]
            else:
                score += 0.4 * adv
            return score
        # 激进：优先大牌推进，并贴紧/围堵；若贴不上，再回头补自己围住的空格。
        score = (4.0 * lvl) * adv + (1.8 * lvl) * point + (0.9 * lvl) * span
        score += (3.8 * lvl) * metrics["touch_enemy"]
        score += (2.8 * lvl) * metrics["surround_enemy"]
        score += 0.25 * special_count
        if point <= 4:
            score += (-3.2 * lvl) * adv
            score += (-3.0 * lvl) * metrics["touch_enemy"]
            score += (-2.0 * lvl) * metrics["surround_enemy"]
            score += (12.0 * lvl) * sp_activated + (3.0 * lvl) * sp_setup_gain + 2.8 * metrics["own_fill_hole"]
        elif point <= _aggressive_low_point_threshold():
            score += (-1.8 * lvl) * adv
            score += (-1.6 * lvl) * metrics["touch_enemy"]
            score += (-0.8 * lvl) * metrics["surround_enemy"]
            score += (8.0 * lvl) * sp_activated + (2.2 * lvl) * sp_setup_gain + 2.0 * metrics["own_fill_hole"]
        if metrics["touch_enemy"] <= 0:
            score += 1.7 * metrics["own_fill_hole"]
        if action.use_sp_attack:
            score -= 0.2 * sp_cost
            score += 0.2
            if not normal_exists:
                score += (6.0 * lvl) * metrics["sp_enemy_cover"]
            else:
                score += (1.0 * lvl) * metrics["sp_enemy_cover"]
    else:
        # balanced：继承 aggressive 基础骨架，但中后期小牌更强调净 SP 收益。
        metrics = _aggressive_action_metrics(state, player, action, cells)
        normal_exists = bool(aggressive_ctx.get("normal_exists", True)) if aggressive_ctx else True
        sp_setup_gain = _sp_setup_potential(state, player, cells)
        sp_activated = _sp_activation_counts(state, player, cells)
        net_sp_gain = float(sp_activated)
        if action.use_sp_attack:
            net_sp_gain -= float(sp_cost)
        score = (3.0 * lvl) * adv + (1.2 * lvl) * point + (0.8 * lvl) * span + 0.25 * special_count
        score += (2.2 * lvl) * metrics["touch_enemy"] + (1.5 * lvl) * metrics["surround_enemy"]
        if point <= _aggressive_low_point_threshold() and turn_phase in ("mid", "late"):
            if net_sp_gain > 0:
                score += (9.0 * lvl) * net_sp_gain
            elif net_sp_gain == 0 and sp_activated > 0:
                score += (4.0 * lvl) * sp_activated
            score += (1.4 * lvl) * sp_setup_gain
            if action.use_sp_attack:
                score += 0.45 * metrics["sp_enemy_cover"]
        if action.use_sp_attack:
            score -= 0.15 * sp_cost
            if not normal_exists:
                score += (3.2 * lvl) * metrics["sp_enemy_cover"]
            else:
                score += (0.9 * lvl) * metrics["sp_enemy_cover"]
    return score


def _choose_bot_action(state: GameState, player: str = "P2") -> Action:
    ps = state.players[player]
    actions = legal_actions(state, player)
    if not actions:
        # 理论上不会空；兜底用首张手牌跳过
        return Action(player=player, card_number=ps.hand[0].Number, pass_turn=True)

    filtered_actions = actions
    if state.bot_config.style in ("aggressive", "balanced", "conservative"):
        filtered_actions = _final_turn_actions(state, player, filtered_actions)
        if state.bot_config.style in ("aggressive", "balanced"):
            filtered_actions = _filter_aggressive_actions(state, player, filtered_actions)
            filtered_actions = _highest_point_actions(filtered_actions, ps)
        else:
            filtered_actions = _highest_point_actions(filtered_actions, ps)
            filtered_actions = _suppress_sp_attack_in_early_mid(filtered_actions, state.turn)
        filtered_actions = _maximize_sp_swing_actions(state, player, filtered_actions)
        filtered_actions = _final_turn_actions(state, player, filtered_actions)

    nn_pick, nn_reason = _choose_bot_action_from_nn(state, player, filtered_actions)
    if nn_pick is not None:
        _log_event(
            state,
            "bot_action_selected_nn",
            {
                "player": player,
                "reason": nn_reason,
                "action": asdict(nn_pick),
            },
        )
        return nn_pick
    if state.bot_nn_spec:
        _log_event(
            state,
            "bot_action_nn_fallback",
            {
                "player": player,
                "reason": nn_reason or "NO_NN_SPEC",
            },
        )

    aggressive_ctx = None
    if state.bot_config.style in ("aggressive", "balanced"):
        aggressive_ctx = {
            "normal_exists": any((not a.pass_turn) and (not a.use_sp_attack) for a in filtered_actions),
            "reserved_no_special12": any(
                (
                    (card := _find_card_in_hand(ps, a.card_number)) is not None
                    and _is_no_special_12(card)
                )
                for a in filtered_actions
                if not a.pass_turn and a.use_sp_attack
            ),
        }

    scored: List[Tuple[float, Action]] = []
    for a in filtered_actions:
        scored.append(
            (
                _score_bot_action(
                    state,
                    player,
                    a,
                    state.bot_config.style,
                    state.bot_config.level,
                    aggressive_ctx=aggressive_ctx,
                ),
                a,
            )
        )
    scored.sort(key=lambda x: x[0], reverse=True)

    # low/mid/high 对应不同“近似最优”采样范围
    if state.bot_config.level == "high":
        pick = scored[0][1]
    elif state.bot_config.level == "mid":
        top_k = max(1, len(scored) // 5)
        pick = state.rng.choice([a for _, a in scored[:top_k]])
    else:
        top_k = max(1, len(scored) // 2)
        pick = state.rng.choice([a for _, a in scored[:top_k]])

    _log_event(
        state,
        "bot_action_selected",
        {
            "player": player,
            "style": state.bot_config.style,
            "level": state.bot_config.level,
            "action": asdict(pick),
            "best_score": scored[0][0],
        },
    )
    return pick


def choose_default_strategy_action(state: GameState, player: str, style: str, level: str) -> Action:
    """Public helper for default heuristic strategy selection on either side."""
    ps = state.players[player]
    actions = legal_actions(state, player)
    if not actions:
        return Action(player=player, card_number=ps.hand[0].Number, pass_turn=True)

    filtered_actions = actions
    if style in ("aggressive", "balanced", "conservative"):
        filtered_actions = _final_turn_actions(state, player, filtered_actions)
        if style in ("aggressive", "balanced"):
            filtered_actions = _filter_aggressive_actions(state, player, filtered_actions)
            filtered_actions = _highest_point_actions(filtered_actions, ps)
        else:
            filtered_actions = _highest_point_actions(filtered_actions, ps)
            filtered_actions = _suppress_sp_attack_in_early_mid(filtered_actions, state.turn)
        filtered_actions = _maximize_sp_swing_actions(state, player, filtered_actions)
        filtered_actions = _final_turn_actions(state, player, filtered_actions)

    aggressive_ctx = None
    if style in ("aggressive", "balanced"):
        aggressive_ctx = {
            "normal_exists": any((not a.pass_turn) and (not a.use_sp_attack) for a in filtered_actions),
            "reserved_no_special12": any(
                (
                    (card := _find_card_in_hand(ps, a.card_number)) is not None
                    and _is_no_special_12(card)
                )
                for a in filtered_actions
                if not a.pass_turn and a.use_sp_attack
            ),
        }

    scored: List[Tuple[float, Action]] = []
    for a in filtered_actions:
        scored.append((_score_bot_action(state, player, a, style, level, aggressive_ctx=aggressive_ctx), a))
    scored.sort(key=lambda x: x[0], reverse=True)

    if level == "high":
        return scored[0][1]
    if level == "mid":
        top_k = max(1, len(scored) // 5)
        return state.rng.choice([a for _, a in scored[:top_k]])
    top_k = max(1, len(scored) // 2)
    return state.rng.choice([a for _, a in scored[:top_k]])


def _match_action_from_payload(actions: List[Action], payload: Dict[str, object]) -> Optional[Action]:
    def _norm_bool(v: object, default: bool = False) -> bool:
        return bool(v) if v is not None else default

    target = {
        "card_number": int(payload.get("card_number")) if payload.get("card_number") is not None else None,
        "pass_turn": _norm_bool(payload.get("pass_turn"), False),
        "use_sp_attack": _norm_bool(payload.get("use_sp_attack"), False),
        "rotation": int(payload.get("rotation", 0)),
        "x": int(payload.get("x")) if payload.get("x") is not None else None,
        "y": int(payload.get("y")) if payload.get("y") is not None else None,
    }
    for a in actions:
        if (
            a.card_number == target["card_number"]
            and a.pass_turn == target["pass_turn"]
            and a.use_sp_attack == target["use_sp_attack"]
            and a.rotation == target["rotation"]
            and a.x == target["x"]
            and a.y == target["y"]
        ):
            return a
    return None


def _choose_bot_action_from_nn(
    state: GameState,
    player: str,
    actions: List[Action],
) -> Tuple[Optional[Action], str]:
    """
    NN strategy interface (optional):
    - type=file_action_json, action_file=<path>
      File contains one action payload dict.
    - type=python_module, module_file=<path>, function=<name|choose_action>
      function signature: fn(state, player, legal_actions, context) -> action payload dict

    On any failure, returns (None, reason) and caller falls back to default 3x3 policy.
    """
    spec = state.bot_nn_spec or {}
    if not spec:
        return None, "NO_SPEC"

    spec_type = str(spec.get("type", "")).strip()
    if not spec_type:
        return None, "SPEC_TYPE_EMPTY"

    if spec_type == "file_action_json":
        action_file = str(spec.get("action_file", "")).strip()
        if not action_file:
            return None, "ACTION_FILE_EMPTY"
        path = Path(action_file)
        if not path.exists():
            return None, "ACTION_FILE_NOT_FOUND"
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return None, "ACTION_FILE_BAD_JSON"
        if not isinstance(payload, dict):
            return None, "ACTION_FILE_NOT_DICT"
        pick = _match_action_from_payload(actions, payload)
        if pick is None:
            return None, "ACTION_NOT_LEGAL"
        return pick, "FILE_ACTION"

    if spec_type == "python_module":
        module_file = str(spec.get("module_file", "")).strip()
        fn_name = str(spec.get("function", "choose_action")).strip() or "choose_action"
        if not module_file:
            return None, "MODULE_FILE_EMPTY"
        path = Path(module_file)
        if not path.exists():
            return None, "MODULE_FILE_NOT_FOUND"
        try:
            mod_name = f"tableturf_nn_{path.stem}"
            module_spec = importlib.util.spec_from_file_location(mod_name, str(path))
            if module_spec is None or module_spec.loader is None:
                return None, "MODULE_SPEC_FAILED"
            module = importlib.util.module_from_spec(module_spec)
            module_spec.loader.exec_module(module)
            fn = getattr(module, fn_name, None)
            if fn is None:
                return None, "MODULE_FN_NOT_FOUND"
            payload = fn(
                state=state,
                player=player,
                legal_actions=[asdict(a) for a in actions],
                context=dict(spec),
            )
            if not isinstance(payload, dict):
                return None, "MODULE_FN_BAD_RETURN"
            pick = _match_action_from_payload(actions, payload)
            if pick is None:
                return None, "ACTION_NOT_LEGAL"
            return pick, "MODULE_ACTION"
        except Exception:
            return None, "MODULE_EXCEPTION"

    return None, f"UNSUPPORTED_SPEC_TYPE:{spec_type}"


def step(state: GameState, action: Action) -> Tuple[bool, str, Dict[str, object]]:
    """
    Submit one player's action.
    - 返回 action 是否有效。
    - 若双方动作尚未齐全，返回等待状态。
    - 若双方齐全，自动 resolve 本回合并返回回合结果。
    - 1P 模式下，P1 提交后会自动生成 P2 电脑动作并结算。
    """
    if state.mode == "1P" and action.player != "P1":
        return False, "ONLY_P1_CAN_SUBMIT_IN_1P_MODE", {}

    ok, reason = submit_action(state, action)
    if not ok:
        return False, reason, {"waiting_for": [p for p in PLAYER_IDS if p not in state.pending_actions]}

    if action.surrender:
        result = _resolve_surrender(state, action)
        return True, "GAME_OVER_SURRENDER", result

    waiting = [p for p in PLAYER_IDS if p not in state.pending_actions]
    if state.mode == "1P" and waiting == ["P2"]:
        bot_action = _choose_bot_action(state, "P2")
        bok, breason = submit_action(state, bot_action)
        if not bok:
            return False, f"BOT_SUBMIT_FAILED:{breason}", {}
        waiting = [p for p in PLAYER_IDS if p not in state.pending_actions]

    if waiting:
        return True, "ACCEPTED_WAITING_OTHER_PLAYER", {"waiting_for": waiting}

    result = _resolve_turn(state)
    return True, "TURN_RESOLVED", result


def legal_actions(state: GameState, player: str) -> List[Action]:
    """
    枚举玩家所有可行动作（包含跳过）。
    用于AI/前端提示。动作空间较大时可按需裁剪。
    """
    if state.done:
        return []
    if player not in PLAYER_IDS:
        raise ValueError(f"invalid player: {player}")
    ps = state.players[player]
    is_p1 = player == "P1"
    actions: List[Action] = []
    for card in ps.hand:
        actions.append(Action(player=player, card_number=card.Number, pass_turn=True))
        for rot in (0, 1, 2, 3):
            for y in range(state.map.height):
                for x in range(state.map.width):
                    ok_n, _, _ = validate_place_card_action(
                        card=card,
                        game_map=state.map,
                        anchor_x=x,
                        anchor_y=y,
                        rotation=rot,
                        is_p1=is_p1,
                        use_sp_attack=False,
                    )
                    if ok_n:
                        actions.append(
                            Action(
                                player=player,
                                card_number=card.Number,
                                pass_turn=False,
                                use_sp_attack=False,
                                rotation=rot,
                                x=x,
                                y=y,
                            )
                        )
                    if ps.sp >= card.SpecialCost:
                        ok_s, _, _ = validate_place_card_action(
                            card=card,
                            game_map=state.map,
                            anchor_x=x,
                            anchor_y=y,
                            rotation=rot,
                            is_p1=is_p1,
                            use_sp_attack=True,
                        )
                        if ok_s:
                            actions.append(
                                Action(
                                    player=player,
                                    card_number=card.Number,
                                    pass_turn=False,
                                    use_sp_attack=True,
                                    rotation=rot,
                                    x=x,
                                    y=y,
                                )
                            )
    return actions
