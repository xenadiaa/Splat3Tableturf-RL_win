#!/usr/bin/env python3
"""Preview Tableturf map grids by id/name/ename."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import List


PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.view.map_view import render_point_type_grid_lines  # noqa: E402

MAP_JSON_CANDIDATES = [
    PROJECT_ROOT / "data" / "maps" / "MiniGameMapInfo.json",
    PROJECT_ROOT / "src" / "assets" / "MiniGameMapInfo.json",
]


def resolve_map_json() -> Path:
    for candidate in MAP_JSON_CANDIDATES:
        if candidate.exists():
            return candidate
    raise FileNotFoundError("MiniGameMapInfo.json not found in expected locations.")


def load_maps() -> List[dict]:
    map_json = resolve_map_json()
    return json.loads(map_json.read_text(encoding="utf-8"))


def match_map(item: dict, key: str) -> bool:
    k = key.strip().lower()
    return (
        item.get("id", "").lower() == k
        or item.get("name", "").lower() == k
        or item.get("ename", "").lower() == k
    )


def render_grid(point_type: List[List[int]]) -> List[str]:
    return render_point_type_grid_lines(point_type)


def print_map(item: dict) -> None:
    map_id = item.get("id", "")
    name = item.get("name", "")
    ename = item.get("ename", "")
    width = item.get("width")
    height = item.get("height")
    point_type = item.get("point_type", [])

    print(f"=== {name} ({ename}) | id={map_id} ===")
    print(f"size: {width}x{height}")
    print("legend: ' '=NotMap, '.'=Placeable, 'x'=Conflict, 's'=SelfSP, 'e'=EnemySP, 'r'=Self(non-SP), 'b'=Enemy(non-SP)")
    print("grid:")
    for line in render_grid(point_type):
        print(line)
    print()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Preview map matrix by map id/name/ename."
    )
    parser.add_argument(
        "map_keys",
        nargs="*",
        help="Map keys: index(0-14) or id/name/ename. Example: 0 14 or 面具屋 间隔墙",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List all available maps.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    maps = load_maps()

    if args.list:
        print("Available maps:")
        for idx, item in enumerate(maps):
            print(f"- idx={idx} | id={item.get('id')} | name={item.get('name')} | ename={item.get('ename')}")
        return 0

    if not args.map_keys:
        print("Please provide at least one map key, or use --list.", file=sys.stderr)
        return 2

    for key in args.map_keys:
        found = None
        if key.isdigit():
            idx = int(key)
            if 0 <= idx < len(maps):
                found = maps[idx]
            else:
                print(f"Map index out of range: {idx} (expected 0~{len(maps)-1})", file=sys.stderr)
                continue
        else:
            found = next((m for m in maps if match_map(m, key)), None)
        if not found:
            print(f"Map not found: {key}", file=sys.stderr)
            continue
        print_map(found)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
