"""Utilities for matching possible NPCs from map and observed opponent actions."""

from __future__ import annotations

from collections import defaultdict
from typing import Dict, List, Optional

from ..strategy.registry import load_npc_strategy_table
from .deck_utils import deck_display_name, load_deck_cards_by_rowid, npc_name_zh


def match_npc_candidates(
    map_id: str,
    observed_actions: Optional[List[Dict[str, object]]] = None,
) -> Dict[str, object]:
    """
    Match possible NPCs from map and partially observed opponent actions.

    Current implementation:
    - hard filters by map_id
    - scores candidates by whether observed card_number entries are included in the NPC deck
    - ignores rotation/x/y/sp/pass for now
    """
    rows = load_npc_strategy_table()
    observed_actions = list(observed_actions or [])
    observed_cards = [
        int(a["card_number"])
        for a in observed_actions
        if isinstance(a, dict) and a.get("card_number") is not None
    ]

    grouped: Dict[str, dict] = {}
    for row in rows:
        npc_name = str(row.get("npc_name", ""))
        matching = [s for s in row.get("strategies", []) if str(s.get("map_id", "")) == str(map_id)]
        if not matching:
            continue
        deck_rowids = sorted({str(s.get("deck_rowid", "")) for s in matching if s.get("deck_rowid")})
        deck_cards: List[int] = []
        for deck_rowid in deck_rowids:
            deck_cards.extend(int(item["number"]) for item in load_deck_cards_by_rowid(deck_rowid))
        deck_card_set = set(deck_cards)
        matched_cards = [n for n in observed_cards if n in deck_card_set]
        missing_cards = [n for n in observed_cards if n not in deck_card_set]
        grouped[npc_name] = {
            "npc_id": row.get("npc_id"),
            "npc_name": npc_name,
            "npc_name_zh": npc_name_zh(npc_name),
            "order": int(row.get("order", 10**9)),
            "map_id": map_id,
            "deck_rowids": deck_rowids,
            "deck_names": [deck_display_name(r) for r in deck_rowids],
            "styles": sorted({str(s.get("bot_style", "")) for s in matching}),
            "levels": sorted({str(s.get("bot_level", "")) for s in matching}),
            "deck_card_numbers": sorted(deck_card_set),
            "observed_card_numbers": observed_cards,
            "matched_card_numbers": matched_cards,
            "missing_card_numbers": missing_cards,
            "match_count": len(matched_cards),
            "missing_count": len(missing_cards),
            "match_score": len(matched_cards) - len(missing_cards) * 2,
        }

    candidates = list(grouped.values())
    elimination_trace: List[Dict[str, object]] = []
    for observed in observed_cards:
        before_count = len(candidates)
        next_candidates = [c for c in candidates if observed in set(c["deck_card_numbers"])]
        removed = [c["npc_name"] for c in candidates if observed not in set(c["deck_card_numbers"])]
        if next_candidates:
            candidates = next_candidates
        elimination_trace.append(
            {
                "observed_card_number": observed,
                "before_count": before_count,
                "after_count": len(candidates),
                "removed_npc_names": removed,
            }
        )
        if len(candidates) <= 1:
            break

    candidates = sorted(
        candidates,
        key=lambda r: (
            -int(r["match_score"]),
            -int(r["match_count"]),
            int(r["missing_count"]),
            int(r["order"]),
            str(r["npc_name"]),
        ),
    )
    return {
        "ok": True,
        "mode": "match_npc_candidates",
        "implemented": True,
        "map_id": map_id,
        "observed_actions": observed_actions,
        "observed_card_numbers": observed_cards,
        "elimination_trace": elimination_trace,
        "candidate_count": len(candidates),
        "candidates": candidates,
        "message": "filtered by map_id first, then eliminate NPCs whose deck does not contain observed card_number values until one remains or no more elimination is possible",
    }


def npc_card_pool_by_map(map_id: str) -> Dict[str, object]:
    """Collect the union of all NPC deck cards available on a given map."""
    rows = load_npc_strategy_table()
    pool = set()
    npc_names = set()
    deck_rowids = set()
    for row in rows:
        npc_name = str(row.get("npc_name", ""))
        matching = [s for s in row.get("strategies", []) if str(s.get("map_id", "")) == str(map_id)]
        if not matching:
            continue
        npc_names.add(npc_name)
        for s in matching:
            deck_rowid = str(s.get("deck_rowid", "")).strip()
            if not deck_rowid or deck_rowid in deck_rowids:
                continue
            deck_rowids.add(deck_rowid)
            for item in load_deck_cards_by_rowid(deck_rowid):
                pool.add(int(item["number"]))
    return {
        "ok": True,
        "map_id": map_id,
        "npc_count": len(npc_names),
        "deck_count": len(deck_rowids),
        "npc_names": sorted(npc_names),
        "deck_rowids": sorted(deck_rowids),
        "card_numbers": sorted(pool),
    }
