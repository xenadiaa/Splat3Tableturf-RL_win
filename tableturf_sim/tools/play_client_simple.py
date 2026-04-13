#!/usr/bin/env python3
"""Windows minimal 2P client using only the Python standard library."""

from __future__ import annotations

import argparse
from contextlib import contextmanager
import json
import os
from pathlib import Path
import select
import socket
import sys
import threading
import time
from typing import Dict, List, Optional

SERVICE_TCP_PORT = 38568
SERVICE_DISCOVERY_PORT = 38569
DISCOVERY_MAGIC = "TABLETURF_MINCLIENT_DISCOVER_V1"
BUFFER_SIZE = 65535
DECK_CONFIG_PATH = Path(__file__).resolve().with_name("play_client_simple_decks.json")


def _clear() -> None:
    os.system("cls" if os.name == "nt" else "clear")


def _enable_ansi_on_windows() -> None:
    if os.name != "nt":
        return
    try:
        import ctypes

        kernel32 = ctypes.windll.kernel32
        handle = kernel32.GetStdHandle(-11)
        mode = ctypes.c_uint32()
        if kernel32.GetConsoleMode(handle, ctypes.byref(mode)):
            kernel32.SetConsoleMode(handle, mode.value | 0x0004)
    except Exception:
        pass


def _fast_redraw(text: str, first_frame: bool) -> None:
    if first_frame:
        sys.stdout.write("\x1b[2J\x1b[H")
    else:
        sys.stdout.write("\x1b[H\x1b[J")
    sys.stdout.write(text)
    if not text.endswith("\n"):
        sys.stdout.write("\n")
    sys.stdout.flush()


import msvcrt


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


@contextmanager
def _interactive_mode():
    yield


def _read_key_blocking() -> str:
    ch = msvcrt.getwch()
    if ch in ("\x00", "\xe0"):
        nxt = msvcrt.getwch()
        return {"H": "UP", "P": "DOWN", "K": "LEFT", "M": "RIGHT"}.get(nxt, "")
    if ch in ("\r", "\n"):
        return "z"
    return ch


def _read_key_nonblocking(timeout_sec: float = 0.02) -> str:
    deadline = time.time() + timeout_sec
    while time.time() < deadline:
        if msvcrt.kbhit():
            return _read_key_blocking()
        time.sleep(0.005)
    return ""


def _discover_services(timeout_sec: float = 1.0, room_only: bool = False) -> List[dict]:
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    rows: List[dict] = []
    seen = set()
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
            except ConnectionResetError:
                # Windows may surface ICMP port-unreachable from one broadcast target
                # as WSAECONNRESET on recvfrom(). Ignore and keep listening.
                continue
            except OSError as exc:
                if getattr(exc, "winerror", None) == 10054:
                    continue
                raise
            try:
                msg = json.loads(data.decode("utf-8"))
            except Exception:
                continue
            key = (msg.get("host"), msg.get("tcp_port"), (msg.get("room") or {}).get("room_id"))
            if key in seen:
                continue
            if msg.get("service") and ((not room_only) or msg.get("room_active")):
                seen.add(key)
                rows.append(msg)
    finally:
        sock.close()
    return rows


def _recv_one_message(sock: socket.socket, reader: _JsonLineReader, timeout: float = 0.2) -> List[dict]:
    sock.settimeout(timeout)
    try:
        chunk = sock.recv(BUFFER_SIZE)
    except socket.timeout:
        return []
    if not chunk:
        raise RuntimeError("server disconnected")
    return reader.feed(chunk)


def _request_json(sock: socket.socket, payload: dict, expect_type: str, timeout: float = 2.0) -> dict:
    reader = _JsonLineReader()
    _send_json_line(sock, payload)
    deadline = time.time() + timeout
    while time.time() < deadline:
        for msg in _recv_one_message(sock, reader, timeout=0.2):
            if msg.get("type") == "error":
                raise RuntimeError(str(msg.get("message", "unknown error")))
            if msg.get("type") == expect_type:
                return msg
    raise RuntimeError(f"timeout waiting for {expect_type}")


def _input_index(max_exclusive: int, allow_empty: bool = False) -> Optional[int]:
    raw = input("> ").strip()
    if allow_empty and not raw:
        return None
    try:
        idx = int(raw)
    except ValueError:
        return None
    if 0 <= idx < max_exclusive:
        return idx
    return None


def _input_deck_ids() -> List[int]:
    while True:
        print("输入15张卡牌编号，使用逗号分隔，例如: 127,57,132,...")
        raw = input("> ").strip()
        try:
            ids = [int(x.strip()) for x in raw.split(",") if x.strip()]
        except ValueError:
            print("卡牌编号必须是整数。")
            continue
        if len(ids) != 15:
            print(f"需要正好15张，当前是 {len(ids)} 张。")
            continue
        if len(set(ids)) != 15:
            print("卡组不能有重复编号。")
            continue
        return ids


def _load_deck_presets() -> List[dict]:
    if not DECK_CONFIG_PATH.exists():
        return []
    try:
        data = json.loads(DECK_CONFIG_PATH.read_text(encoding="utf-8"))
    except Exception:
        return []
    rows = data.get("decks", data if isinstance(data, list) else [])
    out: List[dict] = []
    for i, row in enumerate(rows):
        if not isinstance(row, dict):
            continue
        cards = row.get("cards")
        if not isinstance(cards, list):
            continue
        try:
            ids = [int(x) for x in cards]
        except Exception:
            continue
        if len(ids) != 15 or len(set(ids)) != 15:
            continue
        out.append(
            {
                "index": i,
                "name": str(row.get("name", f"Deck{i}")),
                "note": str(row.get("note", "")),
                "cards": ids,
            }
        )
    return out


def _choose_deck_from_config() -> Optional[dict]:
    rows = _load_deck_presets()
    if not rows:
        return None
    _clear()
    print("[本地牌组配置]")
    for i, row in enumerate(rows):
        note = f" | {row['note']}" if row.get("note") else ""
        print(f"[{i}] {row['name']}{note}")
    print("选择牌组编号，直接回车则改为手动输入")
    idx = _input_index(len(rows), allow_empty=True)
    if idx is None:
        return None
    return rows[idx]


def _choose_deck_payload() -> tuple[str, List[int]]:
    preset = _choose_deck_from_config()
    if preset is not None:
        return str(preset["name"]), list(preset["cards"])
    print("输入牌组名称，留空则使用 Deck")
    deck_name = input("> ").strip() or "Deck"
    return deck_name, _input_deck_ids()


def _choose_service(room_only: bool) -> Optional[dict]:
    rows = _discover_services(room_only=room_only)
    if not rows:
        print("未搜索到服务端。")
        return None
    _clear()
    print("[服务端列表]")
    for i, row in enumerate(rows):
        room = row.get("room") or {}
        room_text = room.get("room_name") or "(无房间)"
        map_text = room.get("map_name_zh") or room.get("map_id") or "(未知地图)"
        room_id = room.get("room_id") or "-"
        status = room.get("status") or "-"
        host_name = room.get("host_name") or "-"
        print(
            f"[{i}] {row.get('host')}:{row.get('tcp_port')} | 房间:{room_text} | "
            f"ID:{room_id} | 地图:{map_text} | 房主:{host_name} | 状态:{status}"
        )
    print("选择编号，回车取消")
    idx = _input_index(len(rows), allow_empty=True)
    if idx is None:
        return None
    return rows[idx]


def _connect_service_by_ip(host: str) -> tuple[socket.socket, dict]:
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        sock.settimeout(3.0)
        sock.connect((host, SERVICE_TCP_PORT))
        pong = _request_json(sock, {"type": "ping"}, "pong", timeout=2.0)
        return sock, pong
    except Exception:
        sock.close()
        raise


def _manual_join_service() -> Optional[tuple[socket.socket, dict]]:
    _clear()
    print("[手动连接服务器]")
    print("输入服务器IP，回车取消")
    host = input("> ").strip()
    if not host:
        return None
    try:
        sock, pong = _connect_service_by_ip(host)
    except Exception as exc:
        print(f"连接失败: {exc}")
        input("按回车返回主界面")
        return None
    room = pong.get("room") or {}
    if not pong.get("room_active"):
        sock.close()
        print("服务器在线，但当前没有可加入的房间。")
        input("按回车返回主界面")
        return None
    return sock, pong


class SessionState:
    def __init__(self) -> None:
        self.screen: str = "连接中..."
        self.done: bool = False
        self.result: Optional[dict] = None
        self.player_role: str = ""
        self.running: bool = True
        self.last_error: str = ""
        self.lock = threading.Lock()


def _render_result(result: Optional[dict], player_role: str) -> str:
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
    return f"[结算]\n{verdict}\n{reason_text}\n比分: 我方 {my_score} : 对手 {opp_score}\n按回车返回主界面"


def _receiver_loop(sock: socket.socket, session: SessionState) -> None:
    reader = _JsonLineReader()
    try:
        while session.running:
            for msg in _recv_one_message(sock, reader, timeout=0.2):
                with session.lock:
                    if msg.get("type") == "error":
                        session.last_error = str(msg.get("message", "unknown error"))
                        session.running = False
                        return
                    if msg.get("type") == "room_created":
                        room = msg.get("room") or {}
                        session.screen = (
                            "[房间创建成功]\n"
                            f"房间名: {room.get('room_name', '(无)')}\n"
                            f"地图: {room.get('map_name_zh') or room.get('map_id', '(无)')}\n"
                            "等待 P2 加入..."
                        )
                    elif msg.get("type") == "match_started":
                        role = str(msg.get("player_role", ""))
                        room = msg.get("room") or {}
                        session.player_role = role
                        session.screen = (
                            "[对局开始]\n"
                            f"你的身份: {role}\n"
                            f"房间名: {room.get('room_name', '(无)')}\n"
                            f"地图: {room.get('map_name_zh') or room.get('map_id', '(无)')}\n"
                            "输入指令继续操作。"
                        )
                    elif msg.get("type") == "render":
                        session.screen = str(msg.get("screen", ""))
                        session.done = bool(msg.get("done", False))
                        if session.done:
                            session.result = msg.get("result") if isinstance(msg.get("result"), dict) else None
                            session.screen = _render_result(session.result, session.player_role)
    except Exception as exc:
        with session.lock:
            session.last_error = str(exc)
            session.running = False


def _print_commands() -> None:
    print("可用指令:")
    print("  方向键移动")
    print("  z 确认 | x 取消")
    print("  a 逆时针 | s 顺时针")
    print("  q 查看牌组")
    print("  m 激进一步 | n 神经网络一步")
    print("  + 投降 | esc 退出对局")


def _session_loop(sock: socket.socket) -> int:
    session = SessionState()
    thread = threading.Thread(target=_receiver_loop, args=(sock, session), daemon=True)
    thread.start()
    try:
        with _interactive_mode():
            _enable_ansi_on_windows()
            first_frame = True
            last_painted = ""
            last_draw_ts = 0.0
            while True:
                with session.lock:
                    screen = session.screen
                    done = session.done
                    running = session.running
                    last_error = session.last_error
                body = screen + "\n\n"
                if done:
                    body += "按 z 返回主界面\n"
                    if body != last_painted:
                        _fast_redraw(body, first_frame)
                        first_frame = False
                        last_painted = body
                    k = _read_key_blocking().lower()
                    if k == "z":
                        return 0
                    continue
                if not running:
                    body += f"连接已断开: {last_error or 'unknown error'}\n"
                    body += "按任意键返回主界面\n"
                    _fast_redraw(body, first_frame)
                    _read_key_blocking()
                    return 0
                lines = [
                    "可用指令:",
                    "  方向键移动",
                    "  z 确认 | x 取消",
                    "  a 逆时针 | s 顺时针",
                    "  q 查看牌组",
                    "  m 激进一步 | n 神经网络一步",
                    "  + 投降 | esc 退出对局",
                ]
                body += "\n".join(lines) + "\n"
                now = time.monotonic()
                if body != last_painted and (now - last_draw_ts) >= 0.05:
                    _fast_redraw(body, first_frame)
                    first_frame = False
                    last_painted = body
                    last_draw_ts = now
                key = _read_key_nonblocking(0.05)
                if not key:
                    continue
                if key == "ESC":
                    return 0
                if key not in ("UP", "DOWN", "LEFT", "RIGHT", "z", "x", "a", "s", "q", "m", "n", "+"):
                    continue
                try:
                    _send_json_line(sock, {"type": "key", "key": key})
                except OSError:
                    print("发送失败。按任意键返回主界面")
                    _read_key_blocking()
                    return 0
    finally:
        session.running = False
        try:
            sock.close()
        except OSError:
            pass


def _edit_auto_config() -> None:
    svc = _choose_service(room_only=False)
    if svc is None:
        return
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.connect((str(svc["host"]), int(svc.get("tcp_port", SERVICE_TCP_PORT))))
    try:
        cfg_msg = _request_json(sock, {"type": "get_strategy_config"}, "strategy_config")
        rows_msg = _request_json(sock, {"type": "list_strategies"}, "strategy_list")
        cfg = cfg_msg.get("config", {})
        rows = rows_msg.get("strategies", [])
        print("[自动对战配置]")
        print("选择要修改的己方侧：1=P1（本机服务） 2=P2（远端加入） 回车取消")
        side_raw = input("> ").strip()
        if side_raw not in ("1", "2"):
            return
        side = "p1" if side_raw == "1" else "p2"
        print("开启自动对战？ 1=开启 0=关闭 回车取消")
        enabled_raw = input("> ").strip()
        if enabled_raw not in ("0", "1"):
            return
        print("[策略列表]")
        for i, row in enumerate(rows):
            print(f"[{i}] {row['id']} | {row['label']}")
        print("选择策略编号，回车取消")
        idx = _input_index(len(rows), allow_empty=True)
        if idx is None:
            return
        saved = _request_json(
            sock,
            {
                "type": "set_strategy_config",
                "side": side,
                "enabled": enabled_raw == "1",
                "strategy_id": str(rows[idx]["id"]),
            },
            "strategy_config_saved",
        )
        _clear()
        print("已写入：")
        print(json.dumps(saved.get("config", cfg), ensure_ascii=False, indent=2))
        input("按回车返回主界面")
    except Exception as exc:
        print(f"配置失败: {exc}")
        input("按回车返回主界面")
    finally:
        sock.close()


def _create_room_flow(args) -> Optional[socket.socket]:
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        sock.connect(("127.0.0.1", SERVICE_TCP_PORT))
    except OSError:
        print("本机服务未启动，当前无法创建房间。")
        input("按回车返回主界面")
        sock.close()
        return None

    print("输入地图ID（例如 Square / Cross / ManyHole），回车取消")
    map_id = input("> ").strip()
    if not map_id:
        sock.close()
        return None
    print("输入房间名，留空则自动生成")
    room_name = input("> ").strip() or f"{args.name}-{int(time.time())%10000}"
    deck_name, deck_ids = _choose_deck_payload()
    try:
        _send_json_line(
            sock,
            {
                "type": "create_room",
                "player_name": args.name,
                "player_id": args.player_id or f"C_{int(time.time())%100000}",
                "room_name": room_name,
                "map_id": map_id,
                "deck_ids": deck_ids,
                "deck_name": deck_name,
            },
        )
        return sock
    except OSError as exc:
        print(f"创建房间失败: {exc}")
        input("按回车返回主界面")
        sock.close()
        return None


def _join_room_flow(args) -> Optional[socket.socket]:
    print("选择加入方式：1=搜索房间 2=手动输入服务器IP 其他=返回")
    mode = input("> ").strip()
    if mode == "1":
        svc = _choose_service(room_only=True)
        if svc is None:
            return None
        room = svc.get("room") or {}
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        connect_host = str(svc["host"])
        connect_port = int(svc.get("tcp_port", SERVICE_TCP_PORT))
    elif mode == "2":
        manual = _manual_join_service()
        if manual is None:
            return None
        sock, svc = manual
        room = svc.get("room") or {}
        connect_host = ""
        connect_port = 0
    else:
        return None
    try:
        if connect_host:
            sock.connect((connect_host, connect_port))
        _clear()
        print("[加入房间]")
        print(f"房间名: {room.get('room_name', '(无)')}")
        print(f"房间ID: {room.get('room_id', '-')}")
        print(f"地图: {room.get('map_name_zh') or room.get('map_id', '(未知地图)')}")
        print(f"房主: {room.get('host_name', '-')}")
        print()
        deck_name, deck_ids = _choose_deck_payload()
        _send_json_line(
            sock,
            {
                "type": "join_room",
                "room_id": room.get("room_id"),
                "player_name": args.name,
                "player_id": args.player_id or f"C_{int(time.time())%100000}",
                "deck_ids": deck_ids,
                "deck_name": deck_name,
            },
        )
        return sock
    except OSError as exc:
        print(f"加入房间失败: {exc}")
        input("按回车返回主界面")
        sock.close()
        return None


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Cross-platform minimal 2P client")
    p.add_argument("--name", default="Client")
    p.add_argument("--player-id", default="")
    return p


def main() -> int:
    args = build_parser().parse_args()
    while True:
        _clear()
        print("[2P最小客户端 - 跨平台]")
        print("[1] 创建房间(本地服务)")
        print("[2] 加入房间")
        print("[3] 自动对战配置")
        print("[0] 退出")
        choice = input("> ").strip()
        if choice == "0":
            return 0
        if choice == "1":
            sock = _create_room_flow(args)
            if sock is not None:
                _session_loop(sock)
            continue
        if choice == "2":
            sock = _join_room_flow(args)
            if sock is not None:
                _session_loop(sock)
            continue
        if choice == "3":
            _edit_auto_config()
            continue


if __name__ == "__main__":
    raise SystemExit(main())
