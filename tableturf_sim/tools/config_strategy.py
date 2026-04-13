#!/usr/bin/env python3
"""Interactive strategy config editor for auto-battle."""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.strategy import ensure_strategy_config, load_strategy_config, save_strategy_config
from src.strategy.registry import list_available_strategy_ids


def _choose_yes_no(title: str, current: bool) -> bool:
    while True:
        raw = input(f"{title} [{'Y' if current else 'N'}] (y/n, 回车保持): ").strip().lower()
        if not raw:
            return current
        if raw in ("y", "yes", "1"):
            return True
        if raw in ("n", "no", "0"):
            return False
        print("请输入 y 或 n")


def _choose_strategy(current_id: str) -> str:
    rows = list_available_strategy_ids()
    print("\n可选策略列表:")
    for i, row in enumerate(rows):
        print(f"[{i:02d}] {row['id']} | {row['label']}")
    print(f"当前策略: {current_id}")
    while True:
        raw = input("输入策略编号（回车保持当前）: ").strip()
        if not raw:
            return current_id
        if raw.isdigit():
            idx = int(raw)
            if 0 <= idx < len(rows):
                return str(rows[idx]["id"])
        print("策略编号无效，请重试")


def _edit_side(cfg: dict, side: str) -> None:
    cur = cfg["auto_battle"][side]
    print(f"\n=== 编辑 {side.upper()} 自动策略 ===")
    cur["enabled"] = _choose_yes_no(f"{side.upper()} 自动对战开关", bool(cur.get("enabled", False)))
    cur["strategy_id"] = _choose_strategy(str(cur.get("strategy_id", "default:balanced:mid")))


def main() -> int:
    ensure_strategy_config()
    cfg = load_strategy_config()
    print(f"配置文件: {PROJECT_ROOT / 'config' / 'strategy_config.json'}")
    _edit_side(cfg, "p1")
    _edit_side(cfg, "p2")
    save_strategy_config(cfg)
    print("\n已写入配置:")
    print(cfg)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
