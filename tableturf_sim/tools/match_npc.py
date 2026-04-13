#!/usr/bin/env python3
"""Match NPC candidates from map + observed opponent actions."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.npc_match_utils import match_npc_candidates


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Match possible NPCs by map and observed opponent actions")
    p.add_argument("input_json", help="JSON file with map_id and optional observed_actions")
    return p


def main() -> int:
    args = build_parser().parse_args()
    payload = json.loads(Path(args.input_json).read_text(encoding="utf-8"))
    result = match_npc_candidates(
        map_id=str(payload["map_id"]),
        observed_actions=list(payload.get("observed_actions", [])),
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
