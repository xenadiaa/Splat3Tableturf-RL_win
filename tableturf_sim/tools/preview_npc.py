#!/usr/bin/env python3
"""Preview NPC strategy and deck cards by npc query."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List, Tuple

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.deck_utils import (  # noqa: E402
    deck_display_name,
    extract_npc_strategy,
    find_npc,
    load_deck_cards_by_rowid,
    npc_name_zh,
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


def print_deck(deck_rowid: str) -> None:
    cards = load_deck_cards_by_rowid(deck_rowid)
    print(f"deck_id: {deck_rowid}")
    print(f"deck_name: {deck_display_name(deck_rowid)}")
    print(f"card_count: {len(cards)}")
    print("legend: '.'=Empty, '#'=Fill, '*'=Special, 'c'=centerpos(link_pos_0)")
    print()
    for item in cards:
        card = item["card"]
        print(
            f"[{item['slot']:02d}] #{item['number']:03d} {item['name']} "
            f"[{item.get('name_zh') or '未找到中文'}] "
            f"(point={item['card_point']}, sp={item['special_cost']}, link_pos_0={card.link_pos_0})"
        )
        for line in matrix_to_lines(card.square_2d_0, card.link_pos_0):
            print(line)
        print()


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Preview NPC by query(name/id/index/order/namehash)."
    )
    parser.add_argument("query", help="NPC query: name / id / number")
    args = parser.parse_args()

    try:
        idx, npc = find_npc(args.query)
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    print("=== NPC Preview ===")
    print(f"npc_index: {idx}")
    print(f"npc_id: {npc.get('__RowId')}")
    print(f"npc_name: {npc.get('Name')}")
    print(f"npc_name_zh: {npc_name_zh(npc.get('Name')) or '(not found)'}")
    print(f"npc_order: {npc.get('Order')}")
    print()

    strategies = extract_npc_strategy(npc)
    print("strategy(level -> ai/map/deck):")
    for s in strategies:
        print(
            f"- level={s['level']} | ai={s['ai_type']}({s['ai_style_zh']}) | "
            f"map={s['map_id']} | deck_id={s['deck_rowid']} | deck_name={deck_display_name(s['deck_rowid'])}"
        )
    print()

    seen = set()
    for s in strategies:
        deck_rowid = s["deck_rowid"]
        if deck_rowid in seen:
            continue
        seen.add(deck_rowid)
        print(f"=== Deck Cards ({deck_rowid}) ===")
        print_deck(deck_rowid)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
