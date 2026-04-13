"""Unified strategy package for Tableturf.

Contains:
- NPC default strategy table
- NN checkpoint loader
- player/automation strategy registry
- auto-battle config helpers
"""

from .config import (
    ensure_strategy_config,
    load_strategy_config,
    save_strategy_config,
    STRATEGY_CONFIG_PATH,
)
from .registry import (
    load_npc_strategy_table,
    list_available_strategy_ids,
    resolve_npc_nn_spec,
    choose_action_from_strategy_id,
)

__all__ = [
    "STRATEGY_CONFIG_PATH",
    "ensure_strategy_config",
    "load_strategy_config",
    "save_strategy_config",
    "load_npc_strategy_table",
    "list_available_strategy_ids",
    "resolve_npc_nn_spec",
    "choose_action_from_strategy_id",
]
