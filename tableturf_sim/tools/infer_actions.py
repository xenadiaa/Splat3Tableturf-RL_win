#!/usr/bin/env python3
"""Infer both players' actions from before/after map snapshots."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.action_infer_utils import infer


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Unified Tableturf inference entry")
    p.add_argument(
        "input_json",
        help=(
            "JSON file for unified inference. Depending on mode, may include "
            "map_id/before_grid/after_grid/known_player/known_action/p1_action/p2_action."
        ),
    )
    p.add_argument(
        "--mode",
        choices=(
            "map_to_both_actions",
            "map_plus_one_action_to_other",
            "both_actions_to_map",
            "played_plus_hand_to_deck",
        ),
        default=None,
    )
    p.add_argument("--max-results", type=int, default=128)
    return p


def main() -> int:
    args = build_parser().parse_args()
    payload = json.loads(Path(args.input_json).read_text(encoding="utf-8"))
    mode = str(args.mode or payload.get("mode", "map_to_both_actions"))
    common_kwargs = {
        "map_id": str(payload["map_id"]),
        "before_grid": payload.get("before_grid"),
        "after_grid": payload.get("after_grid"),
        "p1_deck": [int(x) for x in payload.get("p1_deck", [])] or None,
        "p2_deck": [int(x) for x in payload.get("p2_deck", [])] or None,
        "p1_hand": [int(x) for x in payload.get("p1_hand", [])] or None,
        "p2_hand": [int(x) for x in payload.get("p2_hand", [])] or None,
        "p1_sp": (int(payload["p1_sp"]) if "p1_sp" in payload and payload["p1_sp"] is not None else None),
        "p2_sp": (int(payload["p2_sp"]) if "p2_sp" in payload and payload["p2_sp"] is not None else None),
        "turn": int(payload.get("turn", 1)),
    }
    result = infer(
        mode=mode,
        max_results=int(args.max_results),
        known_player=(str(payload["known_player"]) if "known_player" in payload else None),
        known_action=dict(payload.get("known_action", {})) if "known_action" in payload else None,
        p1_action=dict(payload.get("p1_action", {})) if "p1_action" in payload else None,
        p2_action=dict(payload.get("p2_action", {})) if "p2_action" in payload else None,
        **common_kwargs,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
