from __future__ import annotations

from typing import Any, Dict, Optional


def provide_state(
    *,
    frame: Any,
    analysis_result: Dict[str, Any],
    turn_index: int,
    map_id: Optional[str],
) -> Dict[str, Any]:
    """
    Example hook for fields that current repos do not expose directly.

    Expected return shape:
    {
      "selected_hand_index": 0,
      "cursor_xy": [5, 5],
      "rotation": 0,
      "p1_sp": 1
    }

    This file is only a placeholder in autocontroller_rebuild_for_RL.
    Replace its body after the vision/UI-state interfaces are implemented.
    """
    del frame, analysis_result, turn_index, map_id
    return {}
