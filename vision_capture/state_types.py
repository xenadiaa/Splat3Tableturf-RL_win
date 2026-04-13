from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple


@dataclass
class ObservedState:
    """State parsed from capture frame (or debug json)."""

    map_id: str
    hand_card_numbers: List[int]
    p1_sp: int
    turn: int
    map_grid: Optional[List[List[int]]] = None
    selected_hand_index: Optional[int] = None
    cursor_xy: Optional[Tuple[int, int]] = None
    rotation: int = 0

