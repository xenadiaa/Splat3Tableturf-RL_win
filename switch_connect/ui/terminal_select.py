from __future__ import annotations

import shutil
import sys
from typing import List, Optional, Sequence


def _clear_screen() -> None:
    sys.stdout.write("\x1b[2J\x1b[H")
    sys.stdout.flush()


def _read_key() -> str:
    try:
        import msvcrt
    except ImportError:
        return ""

    ch = msvcrt.getwch()
    if ch in ("\r", "\n"):
        return "enter"
    if ch == "\x03":
        return "ctrl_c"
    if ch in ("\x00", "\xe0"):
        nxt = msvcrt.getwch()
        if nxt == "H":
            return "up"
        if nxt == "P":
            return "down"
        return "other"
    if ch == "\x1b":
        return "esc"
    return "other"


def choose_with_arrows(
    options: Sequence[str],
    title: str,
    footer: str = "Use Up/Down then Enter. Ctrl+C to cancel.",
) -> Optional[str]:
    if not options:
        return None
    if not (sys.stdin.isatty() and sys.stdout.isatty()):
        return str(options[0])

    idx = 0
    while True:
        _clear_screen()
        width = shutil.get_terminal_size((80, 24)).columns
        lines: List[str] = []
        for line in str(title).splitlines() or [""]:
            lines.append(line[:width])
        lines.append("-" * min(width, 80))
        for i, opt in enumerate(options):
            prefix = ">  " if i == idx else "   "
            lines.append(f"{prefix}{opt}")
        if str(footer).strip():
            lines.append("")
            for line in str(footer).splitlines() or [""]:
                lines.append(line[:width])
        body = "\r\n".join(lines) + "\r\n"
        sys.stdout.write(body)
        sys.stdout.flush()

        key = _read_key()
        if key == "up":
            idx = (idx - 1) % len(options)
        elif key == "down":
            idx = (idx + 1) % len(options)
        elif key == "enter":
            _clear_screen()
            return str(options[idx])
        elif key in {"ctrl_c", "esc"}:
            _clear_screen()
            return None
