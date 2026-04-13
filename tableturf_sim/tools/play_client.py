#!/usr/bin/env python3
"""Minimal 2P client: receive render, send keys. Works with system Python only."""

from __future__ import annotations

import argparse
from collections import deque
import json
import msvcrt
from pathlib import Path
import re
import socket
import sys
import time
from contextlib import contextmanager
from typing import List, Optional, Tuple

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.net.room_net import BUFFER_SIZE
from src.engine.loaders import load_map
from src.utils.player_deck_utils import (
    DECK_MAX_INDEX,
    DECK_MIN_INDEX,
    check_player_deck,
    get_player_deck_card_numbers,
    get_player_deck_name,
)
from tools.play_service import DISCOVERY_MAGIC, SERVICE_DISCOVERY_PORT, SERVICE_TCP_PORT


CSI_ARROW_RE = re.compile(r"^(?:\[|O)(?:[0-9;]*)([ABCD])")
_KEY_BUFFER: deque[str] = deque()


def _clear() -> None:
    sys.stdout.write("\r\033[2J\033[H")
    sys.stdout.flush()


def _redraw_frame(body: str, *, first_frame: bool) -> None:
    if first_frame:
        sys.stdout.write("\r\033[2J\033[H")
    else:
        sys.stdout.write("\r\033[H\033[J")
    sys.stdout.write(body)
    if not body.endswith("\n"):
        sys.stdout.write("\n")
    sys.stdout.flush()


def _list_valid_player_decks() -> List[Tuple[int, str]]:
    out: List[Tuple[int, str]] = []
    for i in range(DECK_MIN_INDEX, DECK_MAX_INDEX + 1):
        chk = check_player_deck(i, require_full_15=True)
        if chk.get("ok"):
            out.append((i, chk.get("deck_name") or get_player_deck_name(i)))
    return out


def _load_map_rows() -> List[Tuple[str, str]]:
    path = PROJECT_ROOT / "data" / "maps" / "MiniGameMapInfo.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    rows: List[Tuple[str, str]] = []
    for item in data:
        map_id = str(item["id"])
        try:
            map_name_zh = load_map(map_id).name
        except Exception:
            map_name_zh = str(item.get("name") or map_id)
        rows.append((map_id, map_name_zh))
    return rows


def _menu_select_raw(title: str, items: List[str], hint: str, start_idx: int = 0) -> Optional[int]:
    idx = max(0, min(start_idx, len(items) - 1)) if items else 0
    while True:
        _clear()
        print(title)
        print(f"{hint} | x取消返回")
        for i, item in enumerate(items):
            marker = ">" if i == idx else " "
            print(f"{marker} [{i:02d}] {item}")
        k = _read_key()
        ku = k.upper()
        if ku == "UP":
            idx = max(0, idx - 1)
        elif ku == "DOWN":
            idx = min(len(items) - 1, idx + 1)
        elif k.lower() == "z":
            return idx
        elif k.lower() == "x":
            return None


def _pause_notice(message: str, delay_sec: float = 1.2) -> None:
    _clear()
    print(message)
    time.sleep(delay_sec)


@contextmanager
def _raw_mode(fileobj):
    del fileobj
    yield


def _read_char_blocking() -> str:
    if _KEY_BUFFER:
        return _KEY_BUFFER.popleft()
    ch = msvcrt.getwch()
    if ch in ("\x00", "\xe0"):
        nxt = msvcrt.getwch()
        return {"H": "UP", "P": "DOWN", "M": "RIGHT", "K": "LEFT"}.get(nxt, "")
    if ch in ("\r", "\n"):
        return "z"
    return ch


def _read_key() -> str:
    ch = _read_char_blocking()
    if not ch:
        return ""
    return ch


def _read_key_nonblocking(timeout_sec: float = 0.02) -> str:
    deadline = time.time() + timeout_sec
    while time.time() < deadline:
        if msvcrt.kbhit():
            return _read_key()
        time.sleep(0.005)
    return ""


class _JsonLineReader:
    def __init__(self) -> None:
        self.buf = b""

    def feed(self, chunk: bytes) -> List[dict]:
        self.buf += chunk
        out: List[dict] = []
        while b"\n" in self.buf:
            line, self.buf = self.buf.split(b"\n", 1)
            line = line.strip()
            if not line:
                continue
            out.append(json.loads(line.decode("utf-8")))
        return out


def _send_json_line(sock: socket.socket, payload: dict) -> None:
    sock.sendall((json.dumps(payload, ensure_ascii=False) + "\n").encode("utf-8"))


def _service_running_local() -> bool:
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.settimeout(0.2)
    try:
        sock.connect(("127.0.0.1", SERVICE_TCP_PORT))
        _send_json_line(sock, {"type": "ping"})
        data = sock.recv(BUFFER_SIZE)
        return b"pong" in data
    except OSError:
        return False
    finally:
        sock.close()


def _discover_services(timeout_sec: float = 1.0, room_only: bool = True) -> List[dict]:
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    rooms: List[dict] = []
    try:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
        sock.settimeout(0.2)
        payload = DISCOVERY_MAGIC.encode("utf-8")
        for target in ("127.0.0.1", "255.255.255.255"):
            try:
                sock.sendto(payload, (target, SERVICE_DISCOVERY_PORT))
            except OSError:
                continue
        deadline = time.time() + timeout_sec
        while time.time() < deadline:
            try:
                data, _addr = sock.recvfrom(BUFFER_SIZE)
            except socket.timeout:
                continue
            try:
                msg = json.loads(data.decode("utf-8"))
            except Exception:
                continue
            if msg.get("service") and ((not room_only) or msg.get("room_active")):
                rooms.append(msg)
    finally:
        sock.close()
    return rooms


def _recv_one_message(sock: socket.socket, reader: _JsonLineReader, timeout: float = 0.05) -> List[dict]:
    sock.settimeout(timeout)
    try:
        chunk = sock.recv(BUFFER_SIZE)
    except socket.timeout:
        return []
    if not chunk:
        raise RuntimeError("server disconnected")
    return reader.feed(chunk)


def _request_json(sock: socket.socket, payload: dict, expect_type: str, timeout: float = 1.0) -> dict:
    reader = _JsonLineReader()
    _send_json_line(sock, payload)
    deadline = time.time() + timeout
    while time.time() < deadline:
        for msg in _recv_one_message(sock, reader, timeout=0.1):
            if msg.get("type") == "error":
                raise RuntimeError(str(msg.get("message", "unknown error")))
            if msg.get("type") == expect_type:
                return msg
    raise RuntimeError(f"timeout waiting for {expect_type}")


def _choose_service(services: List[dict], title: str) -> dict:
    if not services:
        raise RuntimeError("未搜索到任何服务端")
    labels = []
    for svc in services:
        room = svc.get("room") or {}
        room_text = room.get("room_name") or "(无房间)"
        labels.append(f"{svc.get('host')} | {room_text}")
    idx = _menu_select_raw(title, labels, "方向键上下移动，z确认")
    if idx is None:
        raise RuntimeError("USER_CANCELLED")
    return services[idx]


def _edit_auto_config(services: List[dict], side: str) -> None:
    target = _choose_service(services, "[选择服务端]")
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.connect((str(target["host"]), int(target.get("tcp_port", SERVICE_TCP_PORT))))
    try:
        cfg_msg = _request_json(sock, {"type": "get_strategy_config"}, "strategy_config")
        rows_msg = _request_json(sock, {"type": "list_strategies"}, "strategy_list")
        cfg = cfg_msg.get("config", {})
        rows = rows_msg.get("strategies", [])
        if not isinstance(rows, list) or not rows:
            raise RuntimeError("服务端未返回可用策略")

        side_cfg = ((cfg.get("auto_battle") or {}).get(side) or {})
        enabled_now = bool(side_cfg.get("enabled", False))
        enabled_idx = _menu_select_raw(
            f"[自动对战配置] {side.upper()} 开关",
            [f"开启 ({'当前' if enabled_now else ''})".strip(), f"关闭 ({'当前' if not enabled_now else ''})".strip()],
            "方向键上下移动，z确认",
            start_idx=0 if enabled_now else 1,
        )
        if enabled_idx is None:
            return
        strategy_labels = [f"{row['id']} | {row['label']}" for row in rows]
        current_id = str(side_cfg.get("strategy_id", ""))
        current_idx = next((i for i, row in enumerate(rows) if str(row.get("id")) == current_id), 0)
        strategy_idx = _menu_select_raw(
            f"[自动对战配置] {side.upper()} 策略",
            strategy_labels,
            "方向键上下移动，z确认",
            start_idx=current_idx,
        )
        if strategy_idx is None:
            return
        saved = _request_json(
            sock,
            {
                "type": "set_strategy_config",
                "side": side,
                "enabled": enabled_idx == 0,
                "strategy_id": str(rows[strategy_idx]["id"]),
            },
            "strategy_config_saved",
        )
        _clear()
        print(f"[自动对战配置] 已写入服务端 ({side.upper()})")
        print(json.dumps(saved.get("config", {}), ensure_ascii=False, indent=2))
        time.sleep(1.2)
    finally:
        sock.close()


def _create_room_flow(args) -> Optional[socket.socket]:
    if not _service_running_local():
        _pause_notice("本机服务未启动，当前无法创建房间")
        return None
    map_rows = _load_map_rows()
    map_labels = [f"{map_id} | {map_name_zh}" for map_id, map_name_zh in map_rows]
    map_idx = _menu_select_raw("[选择地图]", map_labels, "方向键上下移动，z确认")
    if map_idx is None:
        return None
    valid_decks = _list_valid_player_decks()
    if not valid_decks:
        _pause_notice("没有可用的完整玩家牌组")
        return None
    deck_labels = [f"deck={idx:02d} | {name}" for idx, name in valid_decks]
    deck_idx = _menu_select_raw("[选择主机牌组]", deck_labels, "方向键上下移动，z确认")
    if deck_idx is None:
        return None
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        sock.connect(("127.0.0.1", SERVICE_TCP_PORT))
        _send_json_line(
            sock,
                {
                    "type": "create_room",
                    "player_name": args.name,
                    "room_name": f"{args.name}-{int(time.time())%10000}",
                    "map_id": map_rows[map_idx][0],
                    "deck_ids": get_player_deck_card_numbers(valid_decks[deck_idx][0], require_full_15=True),
                    "deck_name": get_player_deck_name(valid_decks[deck_idx][0]),
                },
            )
        return sock
    except OSError:
        sock.close()
        _pause_notice("连接本机服务失败，已返回首页")
        return None


def _join_room_flow(args) -> Optional[socket.socket]:
    room_services = _discover_services(room_only=True)
    if not room_services:
        _pause_notice("未搜索到任何房间")
        return None
    labels = [
        f"{s['room'].get('room_name')} | {s['room'].get('map_name_zh') or s['room'].get('map_id')} | {s.get('host')}"
        for s in room_services
    ]
    room_idx = _menu_select_raw("[搜索房间]", labels, "方向键上下移动，z确认")
    if room_idx is None:
        return None
    valid_decks = _list_valid_player_decks()
    if not valid_decks:
        _pause_notice("没有可用的完整玩家牌组")
        return None
    deck_labels = [f"deck={idx:02d} | {name}" for idx, name in valid_decks]
    deck_idx = _menu_select_raw("[选择加入方牌组]", deck_labels, "方向键上下移动，z确认")
    if deck_idx is None:
        return None
    target = room_services[room_idx]
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        sock.connect((str(target["host"]), int(target.get("tcp_port", SERVICE_TCP_PORT))))
        _send_json_line(
            sock,
            {
                "type": "join_room",
                "room_id": target["room"]["room_id"],
                "player_name": args.name,
                "player_id": args.player_id or f"C_{int(time.time())%100000}",
                "deck_ids": get_player_deck_card_numbers(valid_decks[deck_idx][0], require_full_15=True),
                "deck_name": get_player_deck_name(valid_decks[deck_idx][0]),
            },
        )
        return sock
    except OSError:
        sock.close()
        _pause_notice("连接房间失败，已返回首页")
        return None


def _auto_config_flow(local_service_online: bool) -> None:
    if local_service_online:
        try:
            _edit_auto_config([{"host": "127.0.0.1", "tcp_port": SERVICE_TCP_PORT, "room": {}}], "p1")
        except Exception as exc:
            if str(exc) != "USER_CANCELLED":
                _pause_notice(f"配置失败: {exc}")
        return
    services = _discover_services(room_only=False)
    if not services:
        _pause_notice("未搜索到任何服务端")
        return
    try:
        _edit_auto_config(services, "p2")
    except Exception as exc:
        if str(exc) != "USER_CANCELLED":
            _pause_notice(f"配置失败: {exc}")


def _session_loop(sock: socket.socket) -> int:
    reader = _JsonLineReader()
    screen = "连接中..."
    done = False
    result: Optional[dict] = None
    player_role = ""
    last_body = ""
    first_frame = True
    last_draw_ts = 0.0
    while True:
        try:
            for msg in _recv_one_message(sock, reader, timeout=0.02):
                if msg.get("type") == "error":
                    raise RuntimeError(str(msg.get("message")))
                if msg.get("type") == "room_created":
                    room = msg.get("room") or {}
                    screen = (
                        "[房间创建成功]\n"
                        f"房间名: {room.get('room_name', '(无)')}\n"
                        f"地图: {room.get('map_name_zh') or room.get('map_id', '(无)')}\n"
                        "等待 P2 加入..."
                    )
                elif msg.get("type") == "match_started":
                    role = str(msg.get("player_role", ""))
                    player_role = role
                    room = msg.get("room") or {}
                    screen = (
                        "[对局开始]\n"
                        f"你的身份: {role}\n"
                        f"房间名: {room.get('room_name', '(无)')}\n"
                        f"地图: {room.get('map_name_zh') or room.get('map_id', '(无)')}\n"
                        "正在载入对局..."
                    )
                elif msg.get("type") == "render":
                    screen = str(msg.get("screen", ""))
                    done = bool(msg.get("done", False))
                    if done:
                        payload = msg.get("result")
                        result = payload if isinstance(payload, dict) else None
        except Exception as exc:
            _pause_notice(f"连接已断开: {exc}")
            try:
                sock.close()
            except OSError:
                pass
            return 0
        body = screen + "\n"
        now = time.monotonic()
        if body != last_body and (now - last_draw_ts) >= 0.05:
            _redraw_frame(body, first_frame=first_frame)
            first_frame = False
            last_body = body
            last_draw_ts = now
        if done:
            try:
                sock.close()
            except OSError:
                pass
            winner = (result or {}).get("winner")
            p1_score = (result or {}).get("p1_score")
            p2_score = (result or {}).get("p2_score")
            reason = str((result or {}).get("reason") or "")
            if player_role == "P1":
                my_score, opp_score = p1_score, p2_score
            else:
                my_score, opp_score = p2_score, p1_score
            if winner == "draw":
                verdict = "平局"
            elif winner == player_role:
                verdict = "你赢了"
            else:
                verdict = "你输了"
            if reason == "SURRENDER":
                reason_text = "原因: 对方投降" if winner == player_role else "原因: 你投降了"
            elif reason == "PLAYER_DISCONNECTED":
                reason_text = "原因: 对方离开" if winner == player_role else "原因: 你已离开"
            else:
                reason_text = "原因: 正常结算"
            while True:
                body = "\n".join(
                    [
                        "[结算]",
                        verdict,
                        reason_text,
                        f"比分: 我方 {my_score} : 对手 {opp_score}",
                        "按 z 返回主界面",
                    ]
                ) + "\n"
                _redraw_frame(body, first_frame=first_frame)
                first_frame = False
                k = _read_key()
                if k.lower() == "z":
                    return 0
        k = _read_key_nonblocking(0.02)
        if k:
            try:
                _send_json_line(sock, {"type": "key", "key": k})
            except OSError:
                _pause_notice("发送操作失败，已返回首页")
                try:
                    sock.close()
                except OSError:
                    pass
                return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Minimal 2P client")
    p.add_argument("--name", default="Client")
    p.add_argument("--player-id", default="")
    return p


def main() -> int:
    args = build_parser().parse_args()
    with _raw_mode(sys.stdin):
        while True:
            local_service_online = _service_running_local()
            menu_items = ["创建房间(本地服务)", "加入房间", "自动对战配置"]
            mode_idx = _menu_select_raw("[2P最小客户端]", menu_items, "方向键上下移动，z确认")
            if mode_idx is None:
                continue
            if mode_idx == 0:
                sock = _create_room_flow(args)
                if sock is None:
                    continue
                _session_loop(sock)
                continue
            if mode_idx == 1:
                sock = _join_room_flow(args)
                if sock is None:
                    continue
                _session_loop(sock)
                continue
            _auto_config_flow(local_service_online)


if __name__ == "__main__":
    raise SystemExit(main())
