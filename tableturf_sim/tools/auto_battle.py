#!/usr/bin/env python3
"""Run automatic P1 vs P2 battle using strategy config."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List, Tuple

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.engine.env_core import init_state, step
from src.strategy import load_strategy_config
from src.strategy.registry import choose_action_from_strategy_id
from src.utils.deck_utils import deck_display_name, load_deck_cards_by_rowid
from src.utils.player_deck_utils import get_player_deck_card_numbers, get_player_deck_name


def _resolve_deck(index: int, npc_rowid: str) -> Tuple[str, List[int], str]:
    rowid = str(npc_rowid or "").strip()
    if rowid:
        cards = load_deck_cards_by_rowid(rowid)
        deck_ids = [int(c["number"]) for c in cards]
        deck_name = deck_display_name(rowid)
        return deck_name, deck_ids, f"npc_rowid:{rowid}"
    deck_name = get_player_deck_name(index)
    deck_ids = get_player_deck_card_numbers(index, require_full_15=True)
    return deck_name, deck_ids, f"player_deck:{index}"


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Run auto battle from strategy config")
    p.add_argument("--map", default="Square")
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--p1-deck", type=int, default=0)
    p.add_argument("--p2-deck", type=int, default=1)
    p.add_argument("--p1-npc-deck-rowid", default="")
    p.add_argument("--p2-npc-deck-rowid", default="")
    p.add_argument("--p1-id", default="AUTO_P1")
    p.add_argument("--p2-id", default="AUTO_P2")
    p.add_argument("--p1-name", default="AutoP1")
    p.add_argument("--p2-name", default="AutoP2")
    return p


def main() -> int:
    args = build_parser().parse_args()
    cfg = load_strategy_config()
    p1_cfg = cfg["auto_battle"]["p1"]
    p2_cfg = cfg["auto_battle"]["p2"]
    if not p1_cfg.get("enabled") or not p2_cfg.get("enabled"):
        raise SystemExit("P1/P2 自动对战都必须在 config 中开启")

    p1_deck_name, p1_deck_ids, p1_deck_src = _resolve_deck(args.p1_deck, args.p1_npc_deck_rowid)
    p2_deck_name, p2_deck_ids, p2_deck_src = _resolve_deck(args.p2_deck, args.p2_npc_deck_rowid)
    state = init_state(
        map_id=args.map,
        p1_deck_ids=p1_deck_ids,
        p2_deck_ids=p2_deck_ids,
        seed=args.seed,
        mode="2P",
        p1_player_id=args.p1_id,
        p2_player_id=args.p2_id,
        p1_player_name=args.p1_name,
        p2_player_name=args.p2_name,
        p1_deck_name=p1_deck_name,
        p2_deck_name=p2_deck_name,
    )
    print(
        f"Auto battle start map={args.map} "
        f"P1={p1_deck_src}({p1_deck_name}) strategy={p1_cfg['strategy_id']} "
        f"P2={p2_deck_src}({p2_deck_name}) strategy={p2_cfg['strategy_id']}"
    )

    while not state.done:
        a1 = choose_action_from_strategy_id(state, "P1", str(p1_cfg["strategy_id"]))
        ok1, r1, _ = step(state, a1)
        if not ok1:
            raise RuntimeError(f"P1 auto action rejected: {r1}")
        if state.done:
            break
        a2 = choose_action_from_strategy_id(state, "P2", str(p2_cfg["strategy_id"]))
        ok2, r2, _ = step(state, a2)
        if not ok2:
            raise RuntimeError(f"P2 auto action rejected: {r2}")

    print(f"winner={state.winner}")
    print(f"log_path={state.log_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
