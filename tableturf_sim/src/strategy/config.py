"""Auto-battle strategy config helpers."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = PROJECT_ROOT / "config"
STRATEGY_CONFIG_PATH = CONFIG_DIR / "strategy_config.json"

DEFAULT_CONFIG: Dict[str, object] = {
    "auto_battle": {
        "p1": {
            "enabled": False,
            "strategy_id": "default:balanced:mid",
        },
        "p2": {
            "enabled": False,
            "strategy_id": "default:balanced:mid",
        },
    }
}


def ensure_strategy_config() -> dict:
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    if not STRATEGY_CONFIG_PATH.exists():
        save_strategy_config(dict(DEFAULT_CONFIG))
    return load_strategy_config()


def load_strategy_config() -> dict:
    ensure_strategy_config_path = False
    if not STRATEGY_CONFIG_PATH.exists():
        ensure_strategy_config_path = True
    if ensure_strategy_config_path:
        return ensure_strategy_config()
    data = json.loads(STRATEGY_CONFIG_PATH.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("strategy config root must be dict")
    return data


def save_strategy_config(data: dict) -> None:
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    STRATEGY_CONFIG_PATH.write_text(
        json.dumps(data, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
