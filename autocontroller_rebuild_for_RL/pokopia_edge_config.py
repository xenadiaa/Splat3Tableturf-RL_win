"""Windows DPAPI-backed configuration for the Pokopia edge uploader."""

from __future__ import annotations

import base64
import ctypes
from ctypes import wintypes
from dataclasses import dataclass
import json
import os
from pathlib import Path
from urllib.parse import urlparse


CONFIG_PATH = Path(__file__).resolve().with_name("pokopia_edge_config.json")
CRYPTPROTECT_UI_FORBIDDEN = 0x1


class _DataBlob(ctypes.Structure):
    _fields_ = [("cbData", wintypes.DWORD), ("pbData", ctypes.POINTER(ctypes.c_byte))]


@dataclass(frozen=True)
class EdgeConfig:
    base_url: str
    upload_token: str
    local_url: str = "http://127.0.0.1:8787"
    live_interval_seconds: float = 2.0
    heartbeat_seconds: float = 10.0
    stats_interval_seconds: float = 30.0


def _crypt32():
    if os.name != "nt":
        raise RuntimeError("DPAPI配置只能在Windows上创建和读取。")
    crypt32 = ctypes.WinDLL("crypt32", use_last_error=True)
    crypt32.CryptProtectData.argtypes = [
        ctypes.POINTER(_DataBlob), wintypes.LPCWSTR, ctypes.POINTER(_DataBlob),
        ctypes.c_void_p, ctypes.c_void_p, wintypes.DWORD, ctypes.POINTER(_DataBlob),
    ]
    crypt32.CryptProtectData.restype = wintypes.BOOL
    crypt32.CryptUnprotectData.argtypes = [
        ctypes.POINTER(_DataBlob), ctypes.POINTER(wintypes.LPWSTR), ctypes.POINTER(_DataBlob),
        ctypes.c_void_p, ctypes.c_void_p, wintypes.DWORD, ctypes.POINTER(_DataBlob),
    ]
    crypt32.CryptUnprotectData.restype = wintypes.BOOL
    return crypt32


def _local_free(pointer) -> None:
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel32.LocalFree.argtypes = [ctypes.c_void_p]
    kernel32.LocalFree.restype = ctypes.c_void_p
    kernel32.LocalFree(ctypes.cast(pointer, ctypes.c_void_p))


def _blob(data: bytes):
    buffer = ctypes.create_string_buffer(data)
    blob = _DataBlob(len(data), ctypes.cast(buffer, ctypes.POINTER(ctypes.c_byte)))
    return blob, buffer


def protect_token(token: str) -> str:
    raw = token.encode("utf-8")
    source, source_buffer = _blob(raw)
    output = _DataBlob()
    crypt32 = _crypt32()
    if not crypt32.CryptProtectData(
        ctypes.byref(source), "Pokopia edge upload token", None, None, None,
        CRYPTPROTECT_UI_FORBIDDEN, ctypes.byref(output),
    ):
        raise ctypes.WinError(ctypes.get_last_error())
    try:
        encrypted = ctypes.string_at(output.pbData, output.cbData)
        return base64.b64encode(encrypted).decode("ascii")
    finally:
        _local_free(output.pbData)
        del source_buffer


def unprotect_token(encoded: str) -> str:
    encrypted = base64.b64decode(encoded, validate=True)
    source, source_buffer = _blob(encrypted)
    output = _DataBlob()
    description = wintypes.LPWSTR()
    crypt32 = _crypt32()
    if not crypt32.CryptUnprotectData(
        ctypes.byref(source), ctypes.byref(description), None, None, None,
        CRYPTPROTECT_UI_FORBIDDEN, ctypes.byref(output),
    ):
        raise ctypes.WinError(ctypes.get_last_error())
    try:
        return ctypes.string_at(output.pbData, output.cbData).decode("utf-8")
    finally:
        _local_free(output.pbData)
        if description:
            _local_free(description)
        del source_buffer


def validate_base_url(value: str) -> str:
    normalized = str(value or "").strip().rstrip("/")
    parsed = urlparse(normalized)
    if parsed.scheme != "https" or not parsed.hostname or parsed.username or parsed.password:
        raise ValueError("边缘地址必须是完整HTTPS网址，例如 https://stamp.rabi.date")
    if parsed.path not in {"", "/"} or parsed.query or parsed.fragment:
        raise ValueError("边缘地址不能包含路径、查询参数或片段。")
    return normalized


def save_edge_config(base_url: str, token: str, path: Path = CONFIG_PATH) -> None:
    normalized_url = validate_base_url(base_url)
    secret = str(token or "").strip()
    if len(secret) < 32:
        raise ValueError("上传令牌至少需要32个字符。")
    payload = {
        "schema_version": 1,
        "base_url": normalized_url,
        "token_dpapi": protect_token(secret),
        "local_url": "http://127.0.0.1:8787",
        "live_interval_seconds": 2.0,
        "heartbeat_seconds": 10.0,
        "stats_interval_seconds": 30.0,
    }
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def load_edge_config(path: Path = CONFIG_PATH) -> EdgeConfig:
    payload = json.loads(path.read_text(encoding="utf-8-sig"))
    token = os.environ.get("POKOPIA_EDGE_TOKEN", "").strip()
    if not token:
        token = unprotect_token(str(payload.get("token_dpapi") or ""))
    if len(token) < 32:
        raise ValueError("边缘上传令牌缺失或无效。")
    return EdgeConfig(
        base_url=validate_base_url(str(payload.get("base_url") or "")),
        upload_token=token,
        local_url=str(payload.get("local_url") or "http://127.0.0.1:8787").rstrip("/"),
        live_interval_seconds=max(1.0, float(payload.get("live_interval_seconds") or 2.0)),
        heartbeat_seconds=max(5.0, float(payload.get("heartbeat_seconds") or 10.0)),
        stats_interval_seconds=max(10.0, float(payload.get("stats_interval_seconds") or 30.0)),
    )
