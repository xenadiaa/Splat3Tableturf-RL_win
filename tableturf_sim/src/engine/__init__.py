# src/engine/__init__.py
from .loaders import load_cards, load_map

# env_core 尚未落地时不阻断 loaders 导入。
try:
    from .env_core import GameState, init_state, legal_actions, step  # type: ignore

    __all__ = [
        "GameState",
        "init_state",
        "step",
        "legal_actions",
        "load_cards",
        "load_map",
    ]
except Exception:
    __all__ = [
        "load_cards",
        "load_map",
    ]
