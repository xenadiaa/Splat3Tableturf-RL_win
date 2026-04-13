"""Compatibility wrapper for strategy package."""

from __future__ import annotations

from typing import Dict, List, Optional

from ..strategy.registry import load_npc_strategy_table, resolve_npc_nn_spec


def get_npc_entry_by_name(npc_name: str) -> Optional[dict]:
    q = str(npc_name).strip().lower()
    if not q:
        return None
    for row in load_npc_strategy_table():
        if str(row.get("npc_name", "")).lower() == q:
            return row
    return None


def get_npc_entry_by_order(order: int) -> Optional[dict]:
    for row in load_npc_strategy_table():
        if int(row.get("order", -1)) == int(order):
            return row
    return None


def get_level_strategy(npc_name: str, level: int) -> Optional[dict]:
    row = get_npc_entry_by_name(npc_name)
    if not row:
        return None
    strategies = row.get("strategies", [])
    for s in strategies:
        if int(s.get("ai_level", -1)) == int(level):
            return s
    return None


def resolve_nn_spec(npc_name: str) -> Optional[Dict[str, object]]:
    return resolve_npc_nn_spec(npc_name)


__all__ = [
    "load_npc_strategy_table",
    "get_npc_entry_by_name",
    "get_npc_entry_by_order",
    "get_level_strategy",
    "resolve_nn_spec",
]
