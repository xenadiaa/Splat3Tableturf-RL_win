"""Localization helpers for CN names."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Optional


PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
CN_JSON_CANDIDATES = [
    PROJECT_ROOT / "data" / "CNzh_full_unicode.json",
    PROJECT_ROOT / "src" / "assets" / "CNzh_full_unicode.json",
]

_cn_data_cache: Optional[dict] = None
_mini_game_card_name_cache: Optional[Dict[str, str]] = None
_glossary_cache: Optional[Dict[str, str]] = None


def _load_cn_data() -> dict:
    global _cn_data_cache
    if _cn_data_cache is None:
        for path in CN_JSON_CANDIDATES:
            if path.exists():
                _cn_data_cache = json.loads(path.read_text(encoding="utf-8"))
                break
        if _cn_data_cache is None:
            raise FileNotFoundError("CNzh_full_unicode.json not found.")
    return _cn_data_cache


def _get_section(section_key: str) -> Dict[str, str]:
    data = _load_cn_data()
    section = data.get(section_key, {})
    return section if isinstance(section, dict) else {}


def mini_game_card_name_map() -> Dict[str, str]:
    global _mini_game_card_name_cache
    if _mini_game_card_name_cache is None:
        _mini_game_card_name_cache = _get_section("CommonMsg/MiniGame/MiniGameCardName")
    return _mini_game_card_name_cache


def glossary_name_map() -> Dict[str, str]:
    global _glossary_cache
    if _glossary_cache is None:
        _glossary_cache = _get_section("CommonMsg/Glossary")
    return _glossary_cache


def lookup_card_name_zh(name_en: str) -> Optional[str]:
    if not name_en:
        return None
    d = mini_game_card_name_map()
    if name_en in d:
        return d[name_en]
    d2 = glossary_name_map()
    return d2.get(name_en)


def lookup_npc_name_zh(name_en: str) -> Optional[str]:
    if not name_en:
        return None
    # NPC names usually live in Glossary; fallback to card names for idol/card NPCs.
    d = glossary_name_map()
    if name_en in d:
        return d[name_en]
    # Some NPC ids use underscore variants, while glossary keys are often compact
    # forms like GearShopClothesFsodr / MizutaSdodr.
    compact = name_en.replace("_", "")
    if compact in d:
        return d[compact]
    d2 = mini_game_card_name_map()
    if name_en in d2:
        return d2[name_en]
    return d2.get(compact)
