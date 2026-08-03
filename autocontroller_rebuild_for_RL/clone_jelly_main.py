from __future__ import annotations

import argparse
import contextlib
import json
import os
import sys
from dataclasses import asdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from autocontroller_rebuild_for_RL.runtime import (
    AutoControllerRuntime,
    MissingInterfaceError,
    REPO_ROOT,
    apply_clone_jelly_profile,
    load_config,
)
from autocontroller_rebuild_for_RL.main import (
    _ensure_switch_link_ready,
    _ensure_vision_ready,
    _resolved_runtime_config_path,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Clone Jelly Tableturf auto controller")
    parser.add_argument(
        "--config",
        default="autocontroller_rebuild_for_RL/runtime_config.example.json",
        help="config json path",
    )
    parser.add_argument(
        "--once",
        action="store_true",
        help="play only one battle instead of looping forever",
    )
    parser.add_argument(
        "--print-config",
        action="store_true",
        help="print resolved config and exit",
    )
    parser.add_argument(
        "--target-wins",
        type=int,
        default=None,
        help="temporary target wins for this run only; overrides config and stops after reaching it",
    )
    parser.add_argument(
        "--tmp_win_target",
        action="store_true",
        help="prompt for a temporary target win count before starting; does not modify config file",
    )
    return parser.parse_args()


def _prompt_target_wins_if_needed(args: argparse.Namespace) -> int | None:
    if not bool(getattr(args, "tmp_win_target", False)):
        return None
    if args.target_wins is not None:
        return max(0, int(args.target_wins))
    if not (sys.stdin.isatty() and sys.stdout.isatty()):
        return None
    try:
        raw = input("本次临时目标胜场（直接回车表示按配置继续运行）: ").strip()
    except EOFError:
        return None
    if not raw:
        return None
    try:
        return max(0, int(raw))
    except ValueError:
        print(f"无效目标胜场输入：{raw}，将按配置继续运行。")
        return None


def _clear_terminal_for_runtime_ui() -> None:
    with contextlib.suppress(Exception):
        sys.stdout.write("\r\033[3J\033[2J\033[H")
        sys.stdout.flush()


def main() -> int:
    args = _parse_args()
    config = load_config(args.config)
    config = apply_clone_jelly_profile(config)
    resolved_config_path = _resolved_runtime_config_path(args.config)
    if resolved_config_path is not None:
        setattr(config, "_runtime_config_path", str(resolved_config_path))
    setattr(config, "_original_continuous_run", bool(config.continuous_run))
    setattr(config, "_original_target_win_count", int(config.target_win_count))
    if args.print_config:
        print(json.dumps(asdict(config), ensure_ascii=False, indent=2))
        return 0

    try:
        _ensure_switch_link_ready(config, args.config)
        _ensure_vision_ready(config, args.config)
    except Exception as exc:
        print(f"启动失败：{exc}", file=sys.stderr)
        return 1

    override_target_wins = _prompt_target_wins_if_needed(args)
    if override_target_wins is not None:
        if int(override_target_wins) == 0:
            config.continuous_run = True
        else:
            config.continuous_run = False
            config.target_win_count = int(override_target_wins)

    _clear_terminal_for_runtime_ui()

    runtime = AutoControllerRuntime(config)
    try:
        if args.once:
            runtime.play_one_battle()
        else:
            runtime.run_forever()
        return 0
    except KeyboardInterrupt:
        return 130
    except MissingInterfaceError as exc:
        print(
            json.dumps(
                {
                    "ok": False,
                    "error": "MISSING_INTERFACE_FIELDS",
                    "missing_fields": exc.missing_fields,
                    "config": str((REPO_ROOT / args.config) if not Path(args.config).is_absolute() else Path(args.config)),
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return 2
    finally:
        runtime.close()


if __name__ == "__main__":
    raise SystemExit(main())
