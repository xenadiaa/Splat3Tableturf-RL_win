#!/usr/bin/env python3
"""Minimal playable CLI wired to env_core."""

from __future__ import annotations

import argparse
from typing import Tuple

from src.engine.env_core import Action, init_state, step
from src.utils.deck_utils import deck_display_name, load_deck_cards_by_rowid
from src.utils.localization import lookup_card_name_zh
from src.utils.player_deck_utils import (
    get_player_deck_card_numbers,
    get_player_deck_name,
)
from src.view.map_view import render_point_type_grid_lines


def _print_map(state) -> None:
    print("\nMap:")
    for line in render_point_type_grid_lines(state.map.grid):
        print(line)


def _print_player_state(state, player: str) -> None:
    ps = state.players[player]
    hand_text = []
    for c in ps.hand:
        zh = lookup_card_name_zh(c.name) or "未找到中文"
        hand_text.append(f"{c.Number}:{c.name}[{zh}]")
    print(f"{player} SP={ps.sp} Hand={{" + ", ".join(hand_text) + "}}")


def _parse_action_input(player: str, raw: str) -> Action:
    """
    Supported formats:
    - pass <card_number>
    - play <card_number> <rotation(0|1|2|3)> <x> <y> [sp]
    """
    parts = raw.strip().split()
    if not parts:
        raise ValueError("empty input")
    cmd = parts[0].lower()
    if cmd == "pass":
        if len(parts) != 2:
            raise ValueError("pass format: pass <card_number>")
        return Action(player=player, card_number=int(parts[1]), pass_turn=True)
    if cmd == "play":
        if len(parts) not in (5, 6):
            raise ValueError("play format: play <card_number> <rot> <x> <y> [sp]")
        card_number = int(parts[1])
        rot = int(parts[2])
        x = int(parts[3])
        y = int(parts[4])
        use_sp = len(parts) == 6 and parts[5].lower() == "sp"
        return Action(
            player=player,
            card_number=card_number,
            pass_turn=False,
            use_sp_attack=use_sp,
            rotation=rot,
            x=x,
            y=y,
        )
    raise ValueError("unknown cmd, use pass/play")


def _prompt_action(state, player: str) -> Action:
    while True:
        try:
            raw = input(
                f"{player} action (pass <card> | play <card> <rot> <x> <y> [sp]): "
            )
            return _parse_action_input(player, raw)
        except Exception as exc:
            print(f"input error: {exc}")


def _resolve_deck(index: int, npc_rowid: str) -> Tuple[str, list[int], str]:
    rowid = str(npc_rowid or "").strip()
    if rowid:
        cards = load_deck_cards_by_rowid(rowid)
        deck_ids = [int(c["number"]) for c in cards]
        deck_name = deck_display_name(rowid)
        return deck_name, deck_ids, f"npc_rowid:{rowid}"
    deck_name = get_player_deck_name(index)
    deck_ids = get_player_deck_card_numbers(index, require_full_15=True)
    return deck_name, deck_ids, f"player_deck:{index}"


def run_game(
    map_id: str,
    p1_deck_index: int,
    p2_deck_index: int,
    p1_npc_deck_rowid: str,
    p2_npc_deck_rowid: str,
    seed: int | None,
    mode: str,
    bot_style: str,
    bot_level: str,
    p1_player_id: str,
    p2_player_id: str,
    p1_player_name: str,
    p2_player_name: str,
) -> None:
    p1_deck_name, p1_deck, p1_deck_src = _resolve_deck(p1_deck_index, p1_npc_deck_rowid)
    p2_deck_name, p2_deck, p2_deck_src = _resolve_deck(p2_deck_index, p2_npc_deck_rowid)
    state = init_state(
        map_id=map_id,
        p1_deck_ids=p1_deck,
        p2_deck_ids=p2_deck,
        seed=seed,
        mode=mode,
        bot_style=bot_style,
        bot_level=bot_level,
        p1_player_id=p1_player_id,
        p2_player_id=p2_player_id,
        p1_player_name=p1_player_name,
        p2_player_name=p2_player_name,
        p1_deck_name=p1_deck_name,
        p2_deck_name=p2_deck_name,
    )
    print(
        f"Start game map={map_id}, mode={mode}, "
        f"P1={p1_player_name}[{p1_player_id}] deck={p1_deck_src}({p1_deck_name}), "
        f"P2={p2_player_name}[{p2_player_id}] deck={p2_deck_src}({p2_deck_name}), "
        f"bot={bot_style}/{bot_level}"
    )

    while not state.done:
        print(f"\n=== Turn {state.turn}/{state.max_turns} ===")
        _print_map(state)
        _print_player_state(state, "P1")
        _print_player_state(state, "P2")

        # P1 submit
        while True:
            a1 = _prompt_action(state, "P1")
            ok, reason, payload = step(state, a1)
            if ok:
                print(f"P1 accepted: {reason} {payload}")
                break
            print(f"P1 invalid: {reason} {payload}")

        # P2 submit: 2P 需要人工输入，1P 由服务器自动生成并结算
        if mode == "2P":
            while True:
                a2 = _prompt_action(state, "P2")
                ok, reason, payload = step(state, a2)
                if ok:
                    print(f"P2 accepted: {reason}")
                    if reason == "TURN_RESOLVED":
                        print(f"turn result: {payload}")
                    else:
                        print(payload)
                    break
                print(f"P2 invalid: {reason} {payload}")
        else:
            # 1P：重复提交 P1 直到回合结算（无效动作时重试）
            if reason == "TURN_RESOLVED":
                print(f"turn result: {payload}")
            else:
                # 如果未结算，继续要求 P1 输入直到结算
                while reason != "TURN_RESOLVED" and not state.done:
                    a1b = _prompt_action(state, "P1")
                    ok, reason, payload = step(state, a1b)
                    if not ok:
                        print(f"P1 invalid: {reason} {payload}")
                        continue
                if reason == "TURN_RESOLVED":
                    print(f"turn result: {payload}")

    print("\n=== Game Over ===")
    _print_map(state)
    print(f"winner={state.winner}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Minimal Tableturf CLI")
    parser.add_argument("--map", default="Painted", help="Map id, e.g. Painted")
    parser.add_argument("--mode", choices=["1P", "2P"], default="2P", help="Game mode")
    parser.add_argument("--p1-deck", type=int, default=0, help="Player deck index 0~32")
    parser.add_argument("--p2-deck", type=int, default=1, help="Player deck index 0~32")
    parser.add_argument(
        "--p1-npc-deck-rowid",
        default="",
        help="Use NPC preset deck rowid for P1 (e.g. MiniGame_Aori), overrides --p1-deck",
    )
    parser.add_argument(
        "--p2-npc-deck-rowid",
        default="",
        help="Use NPC preset deck rowid for P2 (e.g. MiniGame_Hotaru), overrides --p2-deck",
    )
    parser.add_argument("--p1-id", default="P1", help="P1 player id")
    parser.add_argument("--p2-id", default="P2", help="P2 player id")
    parser.add_argument("--p1-name", default="P1", help="P1 player name")
    parser.add_argument("--p2-name", default="P2", help="P2 player name")
    parser.add_argument("--bot-style", choices=["balanced", "aggressive", "conservative"], default="balanced")
    parser.add_argument("--bot-level", choices=["low", "mid", "high"], default="mid")
    parser.add_argument("--seed", type=int, default=None, help="Random seed")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    run_game(
        map_id=args.map,
        p1_deck_index=args.p1_deck,
        p2_deck_index=args.p2_deck,
        p1_npc_deck_rowid=args.p1_npc_deck_rowid,
        p2_npc_deck_rowid=args.p2_npc_deck_rowid,
        seed=args.seed,
        mode=args.mode,
        bot_style=args.bot_style,
        bot_level=args.bot_level,
        p1_player_id=args.p1_id,
        p2_player_id=args.p2_id,
        p1_player_name=args.p1_name,
        p2_player_name=args.p2_name,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
