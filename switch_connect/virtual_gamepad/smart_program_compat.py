from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Sequence, Tuple


COMMAND_MAX = 30
SMART_HEX_VERSION = 9
SMART_HEX_ACCEPTED_VERSIONS = {8, 9}

# Token -> firmware command char (from AutoController Others_SmartProgram.c switch-case).
TOKEN_TO_CHAR: Dict[str, str] = {
    "NOTHING": "0",
    "A": "1",
    "B": "2",
    "X": "3",
    "Y": "4",
    "L": "5",
    "R": "6",
    "ZL": "7",
    "ZR": "8",
    "PLUS": "9",
    "MINUS": "q",
    "HOME": "w",
    "CAPTURE": "e",
    "LCLICK": "r",
    "LUP": "t",
    "LDOWN": "y",
    "LLEFT": "u",
    "LRIGHT": "i",
    "RCLICK": "o",
    "RUP": "p",
    "RDOWN": "a",
    "RLEFT": "s",
    "RRIGHT": "d",
    "DUP": "f",
    "DDOWN": "g",
    "DLEFT": "h",
    "DRIGHT": "j",
    "TRIGGERS": "k",
    "LUPLEFT": "l",
    "LUPRIGHT": "z",
    "LDOWNLEFT": "x",
    "LDOWNRIGHT": "c",
    "ASPAM": "v",
    "BSPAM": "b",
    "LOOP": "n",
    "LUPA": "m",
    "LDOWNA": ",",
    "LRIGHTA": ".",
    "DRIGHTR": "/",
    "LUPCLICK": "@",
    "LLEFTB": "#",
    "LRIGHTB": "$",
    "BXDUP": "%",
    "ZRDUP": "^",
    "BY": "&",
    "ZLBX": "*",
    "ZLA": "(",
}

# Friendly aliases used in some scripts.
ALIASES: Dict[str, str] = {
    "NOTHING": "NOTHING",
    "UP": "LUP",
    "DOWN": "LDOWN",
    "LEFT": "LLEFT",
    "RIGHT": "LRIGHT",
    "DUP": "DUP",
    "DDOWN": "DDOWN",
    "DLEFT": "DLEFT",
    "DRIGHT": "DRIGHT",
}


@dataclass
class SmartCommand:
    token: str
    duration: int


def _normalize_token(token: str) -> str:
    t = token.strip().upper()
    t = ALIASES.get(t, t)
    return t


def parse_smart_command_csv(command_csv: str) -> List[SmartCommand]:
    """
    Parse format: "A,1,Nothing,20,DRight,1"
    """
    parts = [p.strip() for p in command_csv.split(",") if p.strip() != ""]
    if len(parts) % 2 != 0:
        raise ValueError("smart command csv must be token,duration pairs")
    out: List[SmartCommand] = []
    for i in range(0, len(parts), 2):
        token = _normalize_token(parts[i])
        if token not in TOKEN_TO_CHAR:
            raise ValueError(f"unknown smart token: {parts[i]}")
        try:
            duration = int(parts[i + 1])
        except ValueError as e:
            raise ValueError(f"invalid duration for token {parts[i]}: {parts[i+1]}") from e
        if duration < 0 or duration > 65535:
            raise ValueError(f"duration out of range (0..65535): {duration}")
        out.append(SmartCommand(token=token, duration=duration))
    return out


def encode_smart_sequence(commands: Sequence[SmartCommand], max_commands: int = COMMAND_MAX) -> bytes:
    """
    Encode to firmware sequence payload:
      [0xFE] + max_commands * ([cmd_char][duration_lo][duration_hi])
    """
    if len(commands) > max_commands:
        raise ValueError(f"too many commands: {len(commands)} > {max_commands}")

    payload = bytearray()
    payload.append(0xFE)
    for i in range(max_commands):
        if i < len(commands):
            cmd = commands[i]
            code = TOKEN_TO_CHAR[cmd.token]
            payload.append(ord(code))
            payload.append(cmd.duration & 0xFF)
            payload.append((cmd.duration >> 8) & 0xFF)
        else:
            payload.append(ord(TOKEN_TO_CHAR["NOTHING"]))
            payload.append(0)
            payload.append(0)
    return bytes(payload)


def encode_smart_sequence_csv(command_csv: str, max_commands: int = COMMAND_MAX) -> bytes:
    return encode_smart_sequence(parse_smart_command_csv(command_csv), max_commands=max_commands)
