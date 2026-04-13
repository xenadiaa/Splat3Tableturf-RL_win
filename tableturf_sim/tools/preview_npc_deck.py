#!/usr/bin/env python3
"""Preview NPC strategy + deck, or preview deck directly."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List, Tuple

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.deck_utils import (  # noqa: E402
    extract_npc_strategy,
    find_deck,
    find_npc,
    load_deck_cards_by_rowid,
)


SYMBOL_MAP = {0: ".", 1: "#", 2: "*"}


def matrix_to_lines(matrix: List[List[int]], link_pos: Tuple[int, int]) -> List[str]:
    x0, y0 = link_pos
    lines: List[str] = []
    for y, row in enumerate(matrix):
        chars = [SYMBOL_MAP.get(v, "?") for v in row]
        if 0 <= x0 < 8 and 0 <= y0 < 8 and y == y0:
            chars[x0] = "c"
        lines.append("".join(chars))
    return lines


def print_deck(deck_rowid: str, title: str) -> None:
    cards = load_deck_cards_by_rowid(deck_rowid)
    print(title)
    print(f"deck_rowid: {deck_rowid}")
    print(f"card_count: {len(cards)}")
    print("legend: '.'=Empty, '#'=Fill, '*'=Special, 'c'=centerpos(link_pos_0)")
    print()
    for item in cards:
        card = item["card"]
        print(
            f"[{item['slot']:02d}] #{item['number']:03d} {item['name']} "
            f"(point={item['card_point']}, sp={item['special_cost']}, link_pos_0={card.link_pos_0})"
        )
        for line in matrix_to_lines(card.square_2d_0, card.link_pos_0):
            print(line)
        print()


def preview_npc(query: str) -> None:
    idx, npc = find_npc(query)
    print("=== NPC Preview ===")
    print(f"npc_index: {idx}")
    print(f"npc_id: {npc.get('__RowId')}")
    print(f"npc_name: {npc.get('Name')}")
    print(f"npc_order: {npc.get('Order')}")
    print()

    strategy_rows = extract_npc_strategy(npc)
    print("strategy(level -> ai/map/deck):")
    for s in strategy_rows:
        print(
            f"- level={s['level']} | ai={s['ai_type']}({s['ai_style_zh']}) "
            f"| map={s['map_id']} | deck={s['deck_rowid']}"
        )
    print()

    seen = set()
    for s in strategy_rows:
        deck_rowid = s["deck_rowid"]
        if deck_rowid in seen:
            continue
        seen.add(deck_rowid)
        print_deck(deck_rowid, f"=== Deck Preview ({deck_rowid}) ===")


def preview_deck(query: str) -> None:
    idx, deck = find_deck(query)
    deck_rowid = deck["__RowId"]
    print("=== Deck Preview ===")
    print(f"deck_index: {idx}")
    print(f"deck_rowid: {deck_rowid}")
    print()
    print_deck(deck_rowid, f"=== Deck Cards ({deck_rowid}) ===")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Preview npc/deck: strategy + deck list + card 0deg matrix."
    )
    parser.add_argument(
        "mode",
        choices=["npc", "deck", "npc/deck"],
        help="Preview target type.",
    )
    parser.add_argument(
        "query",
        help="NPC/Deck query by name/id/index(number).",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        if args.mode == "npc":
            preview_npc(args.query)
        elif args.mode == "deck":
            preview_deck(args.query)
        else:
            try:
                preview_npc(args.query)
            except Exception:
                preview_deck(args.query)
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
