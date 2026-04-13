from __future__ import annotations

from typing import List

try:
    from serial.tools import list_ports
except Exception:  # pragma: no cover
    list_ports = None


def list_serial_port_labels() -> List[str]:
    if list_ports is None:
        return []
    ports = []
    for p in list_ports.comports():
        desc = p.description or ""
        ports.append(f"{p.device}  |  {desc}".strip())
    return ports


def parse_device_from_label(label: str) -> str:
    return label.split("|", 1)[0].strip()

