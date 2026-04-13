from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from switch_connect.virtual_gamepad.serial_controller import SerialRemoteController
from switch_connect.virtual_gamepad.smart_program_compat import parse_smart_command_csv
from switch_connect.virtual_gamepad.device_discovery import list_serial_port_labels, parse_device_from_label
from switch_connect.ui.terminal_select import choose_with_arrows


def main() -> int:
    p = argparse.ArgumentParser(description="Send AutoController-compatible smart sequence (0xFE mode)")
    p.add_argument("--serial-port", default="", help="e.g. /dev/cu.SLAB_USBtoUART")
    p.add_argument("--pick-serial", action="store_true", help="choose serial port via arrow keys")
    p.add_argument("--baudrate", type=int, default=9600)
    p.add_argument("--commands", required=True, help="CSV: A,1,Nothing,20,DRight,1")
    p.add_argument("--wait-ack-seconds", type=float, default=2.0)
    args = p.parse_args()

    selected_port = args.serial_port.strip()
    if args.pick_serial or not selected_port:
        labels = list_serial_port_labels()
        if not labels:
            print(json.dumps({"ack": False, "error": "NO_SERIAL_PORT_FOUND"}, ensure_ascii=False))
            return 2
        picked = choose_with_arrows(labels, "Select virtual gamepad serial port")
        if not picked:
            print(json.dumps({"ack": False, "error": "SERIAL_SELECTION_CANCELLED"}, ensure_ascii=False))
            return 2
        selected_port = parse_device_from_label(picked)

    parsed = parse_smart_command_csv(args.commands)
    ctl = SerialRemoteController(port=selected_port, baudrate=args.baudrate)
    try:
        ctl.send_smart_sequence(parsed)
        ack = ctl.wait_smart_sequence_start_ack(timeout_seconds=args.wait_ack_seconds)
    finally:
        ctl.close()

    print(
        json.dumps(
            {
                "port": selected_port,
                "commands_count": len(parsed),
                "ack": ack,
            },
            ensure_ascii=False,
        )
    )
    return 0 if ack else 1


if __name__ == "__main__":
    raise SystemExit(main())
