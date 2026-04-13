"""Utilities for loading NPC/deck/card relations from game JSON assets."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from .common_utils import create_card_from_id
from .localization import lookup_card_name_zh, lookup_npc_name_zh


PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

NPC_JSON_CANDIDATES = [
    PROJECT_ROOT / "data" / "MiniGameGameNpcData.json",
    PROJECT_ROOT / "src" / "assets" / "MiniGameGameNpcData.json",
]
DECK_JSON_CANDIDATES = [
    PROJECT_ROOT / "data" / "cards" / "MiniGamePresetDeck.json",
    PROJECT_ROOT / "src" / "assets" / "MiniGamePresetDeck.json",
]
CARD_JSON_CANDIDATES = [
    PROJECT_ROOT / "data" / "cards" / "MiniGameCardInfo.json",
    PROJECT_ROOT / "src" / "assets" / "MiniGameCardInfo.json",
]

AI_STYLE_ZH = {
    "Aggressive": "激进",
    "Balance": "均衡",
    "AccumulateSpecial": "保守(攒SP)",
}


def deck_display_name(deck_rowid: str) -> str:
    """Human-friendly deck name from row id."""
    return deck_rowid.replace("MiniGame_", "", 1)


_npc_cache: Optional[List[dict]] = None
_deck_cache: Optional[List[dict]] = None
_card_cache: Optional[List[dict]] = None
_deck_by_rowid: Optional[Dict[str, dict]] = None
_card_by_rowid: Optional[Dict[str, dict]] = None


def _load_json_from_candidates(paths: List[Path]) -> List[dict]:
    for path in paths:
        if path.exists():
            return json.loads(path.read_text(encoding="utf-8"))
    raise FileNotFoundError(f"JSON not found in candidates: {[str(p) for p in paths]}")


def load_npc_data() -> List[dict]:
    global _npc_cache
    if _npc_cache is None:
        _npc_cache = _load_json_from_candidates(NPC_JSON_CANDIDATES)
    return _npc_cache


def load_preset_decks() -> List[dict]:
    global _deck_cache
    if _deck_cache is None:
        _deck_cache = _load_json_from_candidates(DECK_JSON_CANDIDATES)
    return _deck_cache


def load_card_info() -> List[dict]:
    global _card_cache
    if _card_cache is None:
        _card_cache = _load_json_from_candidates(CARD_JSON_CANDIDATES)
    return _card_cache


def _get_deck_by_rowid() -> Dict[str, dict]:
    global _deck_by_rowid
    if _deck_by_rowid is None:
        _deck_by_rowid = {d["__RowId"]: d for d in load_preset_decks()}
    return _deck_by_rowid


def _get_card_by_rowid() -> Dict[str, dict]:
    global _card_by_rowid
    if _card_by_rowid is None:
        _card_by_rowid = {c["__RowId"]: c for c in load_card_info()}
    return _card_by_rowid


def gyml_to_rowid(path: str) -> str:
    """Extract row id from gyml path."""
    # Work/Gyml/MiniGame_Aori.spl__MiniGamePresetDeck.gyml -> MiniGame_Aori
    name = path.split("/")[-1]
    return name.split(".spl__")[0]


def find_npc(query: str) -> Tuple[int, dict]:
    npcs = load_npc_data()
    q = query.strip()

    if q.isdigit():
        idx = int(q)
        if 0 <= idx < len(npcs):
            return idx, npcs[idx]
        for i, npc in enumerate(npcs):
            if str(npc.get("Order")) == q or str(npc.get("NameHash")) == q:
                return i, npc
        raise IndexError(f"NPC index/order/namehash not found: {q}")

    q_lower = q.lower()
    for i, npc in enumerate(npcs):
        if npc.get("__RowId", "").lower() == q_lower:
            return i, npc
        if npc.get("Name", "").lower() == q_lower:
            return i, npc
    raise KeyError(f"NPC not found by id/name: {query}")


def find_deck(query: str) -> Tuple[int, dict]:
    decks = load_preset_decks()
    q = query.strip()

    if q.isdigit():
        idx = int(q)
        if 0 <= idx < len(decks):
            return idx, decks[idx]
        raise IndexError(f"Deck index out of range: {q}")

    q_lower = q.lower()
    for i, deck in enumerate(decks):
        rowid = deck.get("__RowId", "")
        if rowid.lower() == q_lower:
            return i, deck
        if deck_display_name(rowid).lower() == q_lower:
            return i, deck
    raise KeyError(f"Deck not found by row id: {query}")


def extract_npc_strategy(npc: dict) -> List[dict]:
    levels = npc.get("AILevel", [])
    ai_types = npc.get("AIType", [])
    maps = npc.get("Map", [])
    decks = npc.get("Deck", [])

    size = min(len(levels), len(ai_types), len(maps), len(decks))
    out: List[dict] = []
    for i in range(size):
        ai_type = ai_types[i]
        out.append(
            {
                "level": levels[i],
                "ai_type": ai_type,
                "ai_style_zh": AI_STYLE_ZH.get(ai_type, ai_type),
                "map_id": maps[i],
                "deck_rowid": gyml_to_rowid(decks[i]),
            }
        )
    return out


def load_deck_cards_by_rowid(deck_rowid: str) -> List[dict]:
    deck_by_rowid = _get_deck_by_rowid()
    card_by_rowid = _get_card_by_rowid()
    if deck_rowid not in deck_by_rowid:
        raise KeyError(f"Deck row id not found: {deck_rowid}")

    deck = deck_by_rowid[deck_rowid]
    out: List[dict] = []
    for i, card_path in enumerate(deck.get("Card", []), start=1):
        card_rowid = gyml_to_rowid(card_path)
        card_info = card_by_rowid.get(card_rowid)
        if card_info is None:
            raise KeyError(f"Card row id not found in card info: {card_rowid}")
        number = card_info["Number"]
        card = create_card_from_id(number)
        out.append(
            {
                "slot": i,
                "row_id": card_rowid,
                "number": number,
                "name": card_info.get("Name", ""),
                "name_zh": lookup_card_name_zh(card_info.get("Name", "")),
                "special_cost": card_info.get("SpecialCost", 0),
                "card_point": card.CardPoint,
                "card": card,
            }
        )
    return out


def npc_name_zh(name_en: str) -> Optional[str]:
    return lookup_npc_name_zh(name_en)
