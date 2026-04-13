"""Inference config helpers."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = PROJECT_ROOT / "config"
INFER_CONFIG_PATH = CONFIG_DIR / "infer_config.json"

DEFAULT_INFER_CONFIG: Dict[str, object] = {
    "modes": {
        "map_to_both_actions": {
            "enable_exact": True,
            "enable_fuzzy": False,
        },
        "map_plus_one_action_to_other": {
            "enable_exact": True,
            "enable_fuzzy": False,
        },
        "both_actions_to_map": {
            "enable_exact": True,
            "enable_fuzzy": False,
        },
        "played_plus_hand_to_deck": {
            "enable_exact": True,
            "enable_fuzzy": False,
        },
    }
}


def ensure_infer_config() -> dict:
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    if not INFER_CONFIG_PATH.exists():
        save_infer_config(dict(DEFAULT_INFER_CONFIG))
    return load_infer_config()


def load_infer_config() -> dict:
    if not INFER_CONFIG_PATH.exists():
        return ensure_infer_config()
    data = json.loads(INFER_CONFIG_PATH.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("infer config root must be dict")
    return data


def save_infer_config(data: dict) -> None:
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    INFER_CONFIG_PATH.write_text(
        json.dumps(data, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def infer_mode_flags(mode: str) -> Dict[str, bool]:
    root = load_infer_config()
    modes = root.get("modes", {})
    mode_cfg = modes.get(mode, {}) if isinstance(modes, dict) else {}
    enable_exact = bool(mode_cfg.get("enable_exact", True))
    enable_fuzzy = bool(mode_cfg.get("enable_fuzzy", False))
    return {
        "enable_exact": enable_exact,
        "enable_fuzzy": enable_fuzzy,
    }
