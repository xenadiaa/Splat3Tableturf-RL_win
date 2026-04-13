from __future__ import annotations

import contextlib
import struct
import time
from typing import Iterable

from .input_mapper import RemoteStep
from .smart_program_compat import (
    COMMAND_MAX,
    SMART_HEX_ACCEPTED_VERSIONS,
    SMART_HEX_VERSION,
    encode_smart_sequence,
    encode_smart_sequence_csv,
    parse_smart_command_csv,
)

try:
    import serial
except ImportError:  # pragma: no cover
    serial = None


class SerialRemoteController:
    """Talk to AutoController smart-program firmware via 0xFF remote mode."""

    def __init__(self, port: str, baudrate: int = 9600, timeout: float = 0.1):
        if serial is None:
            raise RuntimeError("pyserial is not installed")
        self._ser = serial.Serial(port=port, baudrate=baudrate, timeout=timeout)

    def close(self) -> None:
        self._ser.close()

    def send_bits(self, bits: int) -> None:
        payload = bytes([0xFF]) + struct.pack("<I", bits & 0xFFFFFFFF)
        self._ser.write(payload)

    def release(self) -> None:
        self.send_bits(0)

    def run_steps(self, steps: Iterable[RemoteStep]) -> None:
        for step in steps:
            self.send_bits(step.bits)
            time.sleep(max(0.0, step.hold_ms / 1000.0))
            self.release()
            time.sleep(max(0.0, step.gap_ms / 1000.0))

    def read_bytes(self, max_len: int = 64) -> bytes:
        return self._ser.read(max_len)

    def send_smart_sequence_payload(self, payload: bytes) -> None:
        """
        Send already encoded sequence payload.
        payload must start with 0xFE and include full command table.
        """
        if not payload or payload[0] != 0xFE:
            raise ValueError("smart sequence payload must start with 0xFE")
        self._ser.write(payload)

    def send_smart_sequence(self, commands) -> None:
        payload = encode_smart_sequence(commands)
        self.send_smart_sequence_payload(payload)

    def send_smart_sequence_csv(self, command_csv: str) -> None:
        payload = encode_smart_sequence_csv(command_csv)
        self.send_smart_sequence_payload(payload)

    def _wait_smart_sequence_complete(self, command_count: int, timeout_seconds: float) -> bool:
        remaining = max(0, int(command_count))
        if remaining <= 0:
            return True
        deadline = time.time() + max(0.2, float(timeout_seconds))
        while remaining > 0 and time.time() < deadline:
            b = self._ser.read(1)
            if not b:
                continue
            byte = b[0]
            if remaining == command_count:
                if byte in SMART_HEX_ACCEPTED_VERSIONS or byte == 0xFF:
                    remaining -= 1
                continue
            # Helper-side logic pops one command for every returned byte after the first,
            # even if the byte is unexpected. We mirror that here to avoid hanging.
            remaining -= 1
        return remaining == 0

    def send_smart_sequence_csv_blocking(self, command_csv: str, timeout_seconds: float = 10.0) -> None:
        commands = parse_smart_command_csv(command_csv)
        if not commands:
            return
        deadline = time.time() + max(0.2, float(timeout_seconds))
        chunk_size = max(1, int(COMMAND_MAX) - 2)
        for offset in range(0, len(commands), chunk_size):
            chunk = commands[offset : offset + chunk_size]
            payload = encode_smart_sequence(chunk)
            self.send_smart_sequence_payload(payload)
            remaining_timeout = max(0.2, deadline - time.time())
            if not self._wait_smart_sequence_complete(len(chunk), timeout_seconds=remaining_timeout):
                raise TimeoutError(f"smart sequence did not finish within {timeout_seconds:.1f}s")

    def abort_active_sequence(self) -> None:
        with contextlib.suppress(Exception):
            self.release()
        with contextlib.suppress(Exception):
            self._ser.reset_input_buffer()
        with contextlib.suppress(Exception):
            self._ser.reset_output_buffer()

    def wait_smart_sequence_start_ack(self, timeout_seconds: float = 2.0) -> bool:
        """
        First command completion ack is SMART_HEX_VERSION when accepted.
        """
        deadline = time.time() + max(0.1, timeout_seconds)
        while time.time() < deadline:
            b = self._ser.read(1)
            if not b:
                continue
            if b[0] in SMART_HEX_ACCEPTED_VERSIONS or b[0] == SMART_HEX_VERSION:
                return True
            # Some firmware replies 0xFF for generic command completion.
            if b[0] == 0xFF:
                return True
        return False

    def probe_firmware(self, timeout_seconds: float = 1.0) -> bool:
        try:
            with contextlib.suppress(Exception):
                self._ser.reset_input_buffer()
            with contextlib.suppress(Exception):
                self._ser.reset_output_buffer()
            self.send_smart_sequence_payload(encode_smart_sequence_csv("NOTHING,1"))
            return self.wait_smart_sequence_start_ack(timeout_seconds=timeout_seconds)
        except Exception:
            return False
