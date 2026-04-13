#!/usr/bin/env python3
"""Load a card by id and print its structure plus 4 rotation previews."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Iterable, List, Tuple

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.common_utils import create_card_from_id  # noqa: E402
from src.utils.localization import lookup_card_name_zh  # noqa: E402


SYMBOL_MAP = {
    0: ".",
    1: "#",
    2: "*",
}

RARITY_MAP = {
    0: "Common",
    1: "Rare",
    2: "Fresh",
}


def matrix_to_lines(matrix: List[List[int]], link_pos: Tuple[int, int]) -> List[str]:
    """Convert an 8x8 int matrix to printable lines and mark link_pos as 'c'."""
    link_x, link_y = link_pos
    lines: List[str] = []
    for y, row in enumerate(matrix):
        chars = [SYMBOL_MAP.get(cell, "?") for cell in row]
        if 0 <= link_x < 8 and 0 <= link_y < 8 and y == link_y:
            chars[link_x] = "c"
        lines.append("".join(chars))
    return lines


def iter_rotation_views(card) -> Iterable[Tuple[str, List[List[int]], Tuple[int, int], Tuple[int, int]]]:
    """Yield label, matrix, link_pos and edge for each rotation."""
    yield "0deg", card.square_2d_0, card.link_pos_0, card.edge_0
    yield "90deg", card.square_2d_90, card.link_pos_90, card.edge_90
    yield "180deg", card.square_2d_180, card.link_pos_180, card.edge_180
    yield "270deg", card.square_2d_270, card.link_pos_270, card.edge_270


def calc_bounds(matrix: List[List[int]]) -> Tuple[int, int, int, int]:
    """Return (top, bottom, left, right) for non-empty cells."""
    xs: List[int] = []
    ys: List[int] = []
    for y in range(8):
        for x in range(8):
            if matrix[y][x] != 0:
                xs.append(x)
                ys.append(y)
    if not xs:
        return (0, 0, 0, 0)
    return (min(ys), max(ys), min(xs), max(xs))


def print_card_structure(card) -> None:
    """Print key metadata and raw card structure."""
    structure = asdict(card)
    name_zh = lookup_card_name_zh(card.name)
    print("=== Card Structure (summary) ===")
    print(f"name: {card.name}")
    print(f"name_zh: {name_zh or '(not found)'}")
    print(f"number: {card.Number}")
    print(f"rarity: {card.Rarity} ({RARITY_MAP.get(card.Rarity, 'Unknown')})")
    print(f"special_cost: {card.SpecialCost}")
    print(f"card_point: {card.CardPoint}")
    print(f"name_hash: {card.NameHash}")
    print(f"row_id: {card.RowId}")
    print()
    print("=== Card Structure (raw JSON) ===")
    print(json.dumps(structure, ensure_ascii=False, indent=2))
    print()


def print_rotation_previews(card) -> None:
    """Print all 4 rotation 8x8 grids."""
    print("=== Rotation Preview (8x8) ===")
    print("Legend: '.'=Empty, '#'=Fill, '*'=Special, 'c'=centerpos(link_pos)")
    print()
    for label, matrix, link_pos, _edge in iter_rotation_views(card):
        top, bottom, left, right = calc_bounds(matrix)
        print(f"[{label}] link_pos={link_pos}")
        print(f"  上下边界: top={top}, bottom={bottom}")
        print(f"  左右边界: left={left}, right={right}")
        for line in matrix_to_lines(matrix, link_pos):
            print(line)
        print()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Input card id and print structure + 4 rotation previews."
    )
    parser.add_argument("card_id", type=int, help="Card Number id")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        card = create_card_from_id(args.card_id)
    except Exception as exc:
        print(f"Failed to load card {args.card_id}: {exc}", file=sys.stderr)
        return 1

    print_card_structure(card)
    print_rotation_previews(card)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
