#!/usr/bin/env python3
"""Preview deck cards by deck query."""

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
    find_deck,
    gyml_to_rowid,
    load_deck_cards_by_rowid,
    load_npc_data,
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


def npc_names_using_deck(deck_rowid: str) -> List[str]:
    out: List[str] = []
    for npc in load_npc_data():
        deck_paths = npc.get("Deck", [])
        if any(gyml_to_rowid(p) == deck_rowid for p in deck_paths):
            out.append(npc.get("Name", npc.get("__RowId", "")))
    return sorted(set(out))


def npc_names_using_deck_zh(deck_rowid: str) -> List[str]:
    names = npc_names_using_deck(deck_rowid)
    out = []
    for n in names:
        out.append(f"{n}({npc_name_zh(n) or '未找到中文'})")
    return out


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Preview deck by query(name/id/index)."
    )
    parser.add_argument("query", help="Deck query: name / id / number")
    args = parser.parse_args()

    try:
        idx, deck = find_deck(args.query)
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    deck_rowid = deck["__RowId"]
    deck_name = deck_display_name(deck_rowid)
    cards = load_deck_cards_by_rowid(deck_rowid)
    owners = npc_names_using_deck(deck_rowid)
    owners_zh = npc_names_using_deck_zh(deck_rowid)

    print("=== Deck Preview ===")
    print(f"deck_index: {idx}")
    print(f"deck_id: {deck_rowid}")
    print(f"deck_name: {deck_name}")
    print(f"npc_name: {', '.join(owners) if owners else '(none)'}")
    print(f"npc_name_zh: {', '.join(owners_zh) if owners_zh else '(none)'}")
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
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
