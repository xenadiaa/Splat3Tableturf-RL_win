from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Optional

REPO_ROOT = Path(__file__).resolve().parent.parent
ENGINE_ROOT = REPO_ROOT / "tableturf_sim"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(ENGINE_ROOT) not in sys.path:
    sys.path.insert(0, str(ENGINE_ROOT))

from src.engine.loaders import load_map
from switch_connect.policies.router import choose_action
from switch_connect.virtual_gamepad.input_mapper import compile_action_to_remote_steps
from switch_connect.virtual_gamepad.serial_controller import SerialRemoteController
from switch_connect.virtual_gamepad.device_discovery import list_serial_port_labels, parse_device_from_label
from switch_connect.ui.terminal_select import choose_with_arrows
from vision_capture.state_types import ObservedState


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Switch bridge: vision->policy->virtual gamepad")
    p.add_argument("--state-json", help="ObservedState json path")
    p.add_argument("--map-id", default="Square", help="map id (when state-json omitted)")
    p.add_argument("--hand", default="", help="comma-separated hand card numbers, e.g. 1,5,10,20")
    p.add_argument("--sp", type=int, default=0, help="current p1 sp")
    p.add_argument("--turn", type=int, default=1, help="current turn")
    p.add_argument("--selected-index", type=int, default=0, help="selected hand index")
    p.add_argument("--cursor-x", type=int, default=0)
    p.add_argument("--cursor-y", type=int, default=0)
    p.add_argument("--rotation", type=int, default=0, help="current card rotation 0..3")
    p.add_argument("--policy", default="engine", help="engine|nn-module|nn-command")
    p.add_argument("--style", default="", help="engine policy style: balanced/aggressive/conservative")
    p.add_argument("--level", default="high", help="engine policy level: low/mid/high")
    p.add_argument("--nn-module", default="", help="python module callable, e.g. mypkg.policy:infer")
    p.add_argument("--nn-command", default="", help="external command, stdin=state json, stdout=action json")
    p.add_argument("--serial-port", default="", help="e.g. /dev/cu.SLAB_USBtoUART")
    p.add_argument("--pick-serial", action="store_true", help="choose serial port via arrow keys")
    p.add_argument("--print-steps", action="store_true")
    return p.parse_args()


def _build_state_from_args(args: argparse.Namespace) -> ObservedState:
    if args.state_json:
        data = json.loads(Path(args.state_json).read_text(encoding="utf-8"))
        return ObservedState(**data)

    hand = [int(x.strip()) for x in args.hand.split(",") if x.strip()]
    if not hand:
        raise ValueError("hand cannot be empty when --state-json is not provided")
    game_map = load_map(args.map_id)
    return ObservedState(
        map_id=args.map_id,
        hand_card_numbers=hand,
        p1_sp=args.sp,
        turn=args.turn,
        map_grid=[row[:] for row in game_map.grid],
        selected_hand_index=args.selected_index,
        cursor_xy=(args.cursor_x, args.cursor_y),
        rotation=args.rotation,
    )


def _run_serial(port: str, steps) -> None:
    controller = SerialRemoteController(port=port)
    try:
        controller.run_steps(steps)
    finally:
        controller.close()


def main() -> int:
    args = _parse_args()
    obs = _build_state_from_args(args)
    style: Optional[str] = args.style or None
    action = choose_action(
        obs=obs,
        policy=args.policy,
        style=style,
        level=args.level,
        nn_module=args.nn_module,
        nn_command=args.nn_command,
    )
    print(json.dumps({"action": asdict(action)}, ensure_ascii=False))

    if obs.selected_hand_index is not None and obs.cursor_xy is not None:
        steps = compile_action_to_remote_steps(action, obs)
        if args.print_steps:
            print(json.dumps({"steps": [asdict(s) for s in steps]}, ensure_ascii=False))
        selected_port = args.serial_port.strip()
        if args.pick_serial or not selected_port:
            if args.pick_serial:
                labels = list_serial_port_labels()
                if not labels:
                    raise RuntimeError("NO_SERIAL_PORT_FOUND")
                picked = choose_with_arrows(labels, "Select virtual gamepad serial port")
                if not picked:
                    raise RuntimeError("SERIAL_SELECTION_CANCELLED")
                selected_port = parse_device_from_label(picked)
        if selected_port:
            _run_serial(selected_port, steps)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
