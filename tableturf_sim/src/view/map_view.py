"""Text rendering helpers for map point_type grids."""

from __future__ import annotations

from typing import Dict, List

# PointType / bitmask preview symbols
MAP_SYMBOL_MAP: Dict[int, str] = {
    0: " ",  # NotMap
    1: ".",  # Placeable / Empty
    2: "x",  # Conflict
    3: "r",  # SelfNormal
    4: "r",  # SelfNormal (alt enum)
    5: "b",  # EnemyNormal
    6: "b",  # EnemyNormal (alt enum)
    7: "x",  # Conflict (bitmask)
    11: "s", # P1Special (bitmask)
    13: "e", # P2Special (bitmask)
    27: "s", # P1 SP active (bitmask)
    29: "e", # P2 SP active (bitmask)
}


def render_point_type_grid_lines(
    point_type: List[List[int]],
    symbol_map: Dict[int, str] | None = None,
) -> List[str]:
    """Convert point_type matrix to printable text lines."""
    lookup = symbol_map or MAP_SYMBOL_MAP
    return ["".join(lookup.get(v, "?") for v in row) for row in point_type]


def render_point_type_grid_text(
    point_type: List[List[int]],
    symbol_map: Dict[int, str] | None = None,
) -> str:
    """Convert point_type matrix to printable text block."""
    return "\n".join(render_point_type_grid_lines(point_type, symbol_map=symbol_map))
