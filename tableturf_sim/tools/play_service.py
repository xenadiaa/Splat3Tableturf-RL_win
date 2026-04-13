#!/usr/bin/env python3
"""Minimal 2P room service: owns state/UI, clients only receive render and send keys."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import queue
import socket
import sys
import threading
import time
from typing import Dict, List, Optional, Tuple
from uuid import uuid4

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.engine.env_core import init_state, step
from src.engine.loaders import load_map
from src.assets.tableturf_types import Map_PointBit
from src.strategy import load_strategy_config, save_strategy_config
from src.strategy.registry import choose_action_from_strategy_id, list_available_strategy_ids
from src.utils.player_deck_utils import get_player_deck_card_numbers, get_player_deck_name
from src.view.gamepad_ui import TerminalGamepadUI


SERVICE_TCP_PORT = 38568
SERVICE_DISCOVERY_PORT = 38569
DISCOVERY_MAGIC = "TABLETURF_MINCLIENT_DISCOVER_V1"
BUFFER_SIZE = 65535


def _compute_scores(game_map) -> Tuple[int, int]:
    p1 = 0
    p2 = 0
    for y in range(game_map.height):
        for x in range(game_map.width):
            m = int(game_map.get_point(x, y))
            is_valid = (m & int(Map_PointBit.IsValid)) != 0
            if not is_valid:
                continue
            is_p1 = (m & int(Map_PointBit.IsP1)) != 0
            is_p2 = (m & int(Map_PointBit.IsP2)) != 0
            if is_p1 and not is_p2:
                p1 += 1
            elif is_p2 and not is_p1:
                p2 += 1
    return p1, p2


def _guess_local_ip() -> str:
    candidates: List[str] = []
    try:
        hostnames = [socket.gethostname(), socket.getfqdn(), "localhost"]
        for host in hostnames:
            try:
                infos = socket.getaddrinfo(host, None, socket.AF_INET, socket.SOCK_STREAM)
            except OSError:
                continue
            for info in infos:
                ip = str(info[4][0])
                if ip not in candidates:
                    candidates.append(ip)
    except OSError:
        pass

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        sock.connect(("8.8.8.8", 80))
        ip = str(sock.getsockname()[0])
        if ip not in candidates:
            candidates.append(ip)
    except OSError:
        pass
    finally:
        sock.close()

    def _ip_rank(ip: str) -> Tuple[int, str]:
        if ip.startswith("192."):
            return (0, ip)
        if ip.startswith("10."):
            return (1, ip)
        parts = ip.split(".")
        if len(parts) == 4 and parts[0] == "172":
            try:
                second = int(parts[1])
            except ValueError:
                second = -1
            if 16 <= second <= 31:
                return (2, ip)
        if ip.startswith("127.") or ip.startswith("169.254.") or ip.startswith("198.18."):
            return (9, ip)
        return (3, ip)

    valid = [ip for ip in candidates if "." in ip and not ip.startswith("127.") and not ip.startswith("169.254.") and not ip.startswith("198.18.")]
    if valid:
        valid.sort(key=_ip_rank)
        return valid[0]
    if candidates:
        candidates.sort(key=_ip_rank)
        return candidates[0]
    return "127.0.0.1"


def _send_json_line(sock: socket.socket, payload: dict) -> None:
    sock.sendall((json.dumps(payload, ensure_ascii=False) + "\n").encode("utf-8"))


def _is_valid_deck_ids(deck_ids: List[int]) -> bool:
    return len(deck_ids) == 15 and len(set(deck_ids)) == 15


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


class ClientConn:
    def __init__(self, sock: socket.socket) -> None:
        self.sock = sock
        self.role: Optional[str] = None
        self.name: str = ""
        self.deck_ids: List[int] = []
        self.deck_name: str = ""


class Minimal2PService:
    def __init__(self, bind_host: str) -> None:
        self.bind_host = bind_host
        self.host_ip = _guess_local_ip() if bind_host == "0.0.0.0" else bind_host
        self.clients: List[ClientConn] = []
        self.msg_queue: "queue.Queue[Tuple[ClientConn, dict]]" = queue.Queue()
        self.room: Optional[dict] = None
        self.state = None
        self.uis: Dict[str, TerminalGamepadUI] = {}
        self._stop = threading.Event()

    def _close_room(self) -> None:
        self.room = None
        self.state = None
        self.uis = {}
        for client in self.clients:
            client.role = None
            client.deck_ids = []
            client.deck_name = ""

    def run(self) -> None:
        discovery_thread = threading.Thread(target=self._discovery_loop, daemon=True)
        accept_thread = threading.Thread(target=self._accept_loop, daemon=True)
        discovery_thread.start()
        accept_thread.start()
        print(f"2P service running on {self.bind_host}:{SERVICE_TCP_PORT}")
        print(f"LAN host IP: {self.host_ip}")
        print(f"Windows/simple client direct connect: {self.host_ip}:{SERVICE_TCP_PORT}")
        print("等待客户端连接...")
        while not self._stop.is_set():
            self._process_messages()
            self._tick_auto()
            self._broadcast_render()
            time.sleep(0.05)

    def _discovery_loop(self) -> None:
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        try:
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            sock.bind((self.bind_host, SERVICE_DISCOVERY_PORT))
            sock.settimeout(0.5)
            while not self._stop.is_set():
                try:
                    data, addr = sock.recvfrom(BUFFER_SIZE)
                except socket.timeout:
                    continue
                if data.decode("utf-8", errors="ignore").strip() != DISCOVERY_MAGIC:
                    continue
                payload = {
                    "service": True,
                    "room_active": bool(self.room),
                    "room": self.room or {},
                    "host": self.host_ip,
                    "tcp_port": SERVICE_TCP_PORT,
                }
                sock.sendto(json.dumps(payload, ensure_ascii=False).encode("utf-8"), addr)
        finally:
            sock.close()

    def _accept_loop(self) -> None:
        srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            srv.bind((self.bind_host, SERVICE_TCP_PORT))
            srv.listen(8)
            while not self._stop.is_set():
                conn, _addr = srv.accept()
                client = ClientConn(conn)
                self.clients.append(client)
                threading.Thread(target=self._client_loop, args=(client,), daemon=True).start()
        finally:
            srv.close()

    def _client_loop(self, client: ClientConn) -> None:
        reader = _JsonLineReader()
        try:
            while not self._stop.is_set():
                chunk = client.sock.recv(BUFFER_SIZE)
                if not chunk:
                    break
                for msg in reader.feed(chunk):
                    self.msg_queue.put((client, msg))
        except OSError:
            pass
        finally:
            self.msg_queue.put((client, {"type": "disconnect"}))
            try:
                client.sock.close()
            except OSError:
                pass
            if client in self.clients:
                self.clients.remove(client)

    def _process_messages(self) -> None:
        while True:
            try:
                client, msg = self.msg_queue.get_nowait()
            except queue.Empty:
                break
            msg_type = str(msg.get("type", ""))
            if msg_type == "ping":
                _send_json_line(
                    client.sock,
                    {
                        "type": "pong",
                        "service": True,
                        "room_active": bool(self.room),
                        "room": self.room or {},
                        "host": self.host_ip,
                        "tcp_port": SERVICE_TCP_PORT,
                    },
                )
                continue
            if msg_type == "create_room":
                self._handle_create_room(client, msg)
                continue
            if msg_type == "join_room":
                self._handle_join_room(client, msg)
                continue
            if msg_type == "get_strategy_config":
                self._handle_get_strategy_config(client)
                continue
            if msg_type == "list_strategies":
                self._handle_list_strategies(client)
                continue
            if msg_type == "set_strategy_config":
                self._handle_set_strategy_config(client, msg)
                continue
            if msg_type == "key":
                self._handle_key(client, str(msg.get("key", "")))
                continue
            if msg_type == "disconnect":
                self._handle_disconnect(client)
                continue

    def _handle_create_room(self, client: ClientConn, msg: dict) -> None:
        if self.room is not None:
            _send_json_line(client.sock, {"type": "error", "message": "已有房间，无法重复创建"})
            return
        map_id = str(msg.get("map_id", "Square"))
        try:
            map_name = load_map(map_id).name
        except Exception:
            map_name = map_id
        deck_ids = list(msg.get("deck_ids", []))
        if not _is_valid_deck_ids(deck_ids):
            _send_json_line(client.sock, {"type": "error", "message": "卡组必须为15张且不能有重复编号"})
            return
        client.role = "P1"
        client.name = str(msg.get("player_name", "P1"))
        client.deck_ids = deck_ids
        client.deck_name = str(msg.get("deck_name", ""))
        self.room = {
            "room_id": uuid4().hex[:8],
            "room_name": str(msg.get("room_name", f"{client.name}-{uuid4().hex[:4]}")),
            "map_id": map_id,
            "map_name_zh": map_name,
            "host_name": client.name,
            "status": "waiting_p2",
        }
        _send_json_line(client.sock, {"type": "room_created", "room": self.room})

    def _handle_join_room(self, client: ClientConn, msg: dict) -> None:
        if self.room is None:
            _send_json_line(client.sock, {"type": "error", "message": "当前没有可加入的房间"})
            return
        if self.state is not None:
            _send_json_line(client.sock, {"type": "error", "message": "房间对局已开始"})
            return
        if str(msg.get("room_id", "")) and str(msg.get("room_id")) != str(self.room["room_id"]):
            _send_json_line(client.sock, {"type": "error", "message": "房间ID不匹配"})
            return
        deck_ids = list(msg.get("deck_ids", []))
        if not _is_valid_deck_ids(deck_ids):
            _send_json_line(client.sock, {"type": "error", "message": "卡组必须为15张且不能有重复编号"})
            return
        client.role = "P2"
        client.name = str(msg.get("player_name", "P2"))
        client.deck_ids = deck_ids
        client.deck_name = str(msg.get("deck_name", ""))
        self._start_match()

    def _start_match(self) -> None:
        p1 = next((c for c in self.clients if c.role == "P1"), None)
        p2 = next((c for c in self.clients if c.role == "P2"), None)
        if p1 is None or p2 is None:
            return
        seed = int(time.time() * 1000) & 0x7FFFFFFF
        try:
            self.state = init_state(
                map_id=str(self.room["map_id"]),
                p1_deck_ids=list(p1.deck_ids),
                p2_deck_ids=list(p2.deck_ids),
                seed=seed,
                mode="2P",
                p1_player_id=f"P1_{uuid4().hex[:6]}",
                p2_player_id=f"P2_{uuid4().hex[:6]}",
                p1_player_name=p1.name,
                p2_player_name=p2.name,
                p1_deck_name=p1.deck_name,
                p2_deck_name=p2.deck_name,
            )
        except Exception as exc:
            _send_json_line(p1.sock, {"type": "error", "message": f"开局失败: {exc}"})
            _send_json_line(p2.sock, {"type": "error", "message": f"开局失败: {exc}"})
            p2.role = None
            p2.deck_ids = []
            p2.deck_name = ""
            if self.room is not None:
                self.room["status"] = "waiting_p2"
            self.state = None
            self.uis = {}
            return

        def _submit(state, action):
            ok, reason, payload = step(state, action)
            if ok and reason == "TURN_RESOLVED" and isinstance(payload, dict):
                for ui in self.uis.values():
                    ui._on_turn_resolved(payload)
            return ok, reason, payload

        self.uis = {
            "P1": TerminalGamepadUI(self.state, local_player="P1", submit_action_fn=_submit),
            "P2": TerminalGamepadUI(self.state, local_player="P2", submit_action_fn=_submit),
        }
        self.room["status"] = "playing"
        for c in self.clients:
            _send_json_line(
                c.sock,
                {
                    "type": "match_started",
                    "player_role": c.role,
                    "room": self.room,
                },
            )

    def _handle_get_strategy_config(self, client: ClientConn) -> None:
        _send_json_line(
            client.sock,
            {
                "type": "strategy_config",
                "config": load_strategy_config(),
            },
        )

    def _handle_list_strategies(self, client: ClientConn) -> None:
        _send_json_line(
            client.sock,
            {
                "type": "strategy_list",
                "strategies": list_available_strategy_ids(),
            },
        )

    def _handle_set_strategy_config(self, client: ClientConn, msg: dict) -> None:
        side = str(msg.get("side", "")).lower()
        if side not in ("p1", "p2"):
            _send_json_line(client.sock, {"type": "error", "message": "side 只能是 p1 或 p2"})
            return
        cfg = load_strategy_config()
        side_cfg = cfg.setdefault("auto_battle", {}).setdefault(
            side,
            {"enabled": False, "strategy_id": "default:balanced:mid"},
        )
        if "enabled" in msg:
            side_cfg["enabled"] = bool(msg.get("enabled"))
        if "strategy_id" in msg:
            strategy_id = str(msg.get("strategy_id", ""))
            strategy_ids = {row["id"] for row in list_available_strategy_ids()}
            if strategy_id not in strategy_ids:
                _send_json_line(client.sock, {"type": "error", "message": f"未知策略: {strategy_id}"})
                return
            side_cfg["strategy_id"] = strategy_id
        save_strategy_config(cfg)
        _send_json_line(
            client.sock,
            {
                "type": "strategy_config_saved",
                "config": cfg,
            },
        )

    def _handle_key(self, client: ClientConn, key: str) -> None:
        if self.state is None or client.role not in self.uis:
            return
        ui = self.uis[client.role]
        ui.handle_key(key)

    def _handle_disconnect(self, client: ClientConn) -> None:
        role = client.role
        if self.room is None:
            return
        if self.state is None:
            if role == "P1":
                self._close_room()
            return
        if self.state.done or role not in ("P1", "P2"):
            return
        winner = "P2" if role == "P1" else "P1"
        self.state.done = True
        self.state.winner = winner
        self.state.reason = "PLAYER_DISCONNECTED"
        self.state.pending_actions.clear()
        if self.room is not None:
            self.room["status"] = "finished_disconnect"

    def _tick_auto(self) -> None:
        if self.state is None or self.state.done:
            return
        cfg = load_strategy_config()
        for role in ("P1", "P2"):
            side_cfg = cfg["auto_battle"]["p1" if role == "P1" else "p2"]
            if not side_cfg.get("enabled"):
                continue
            if role in self.state.pending_actions:
                continue
            try:
                action = choose_action_from_strategy_id(self.state, role, str(side_cfg["strategy_id"]))
            except Exception as exc:
                print(f"[auto_battle disabled] role={role} strategy={side_cfg.get('strategy_id')} error={exc}")
                side_cfg["enabled"] = False
                save_strategy_config(cfg)
                continue
            ok, reason, payload = self.uis[role].submit_action_fn(self.state, action)
            if ok and reason == "TURN_RESOLVED" and isinstance(payload, dict):
                for ui in self.uis.values():
                    ui._on_turn_resolved(payload)

    def _broadcast_render(self) -> None:
        for client in list(self.clients):
            try:
                if self.state is None:
                    room_text = (
                        f"[房间等待中]\n"
                        f"房间名: {self.room['room_name'] if self.room else '(无)'}\n"
                        f"地图: {self.room['map_id'] if self.room else '(无)'}\n"
                        f"P1: {next((c.name for c in self.clients if c.role == 'P1'), '(未加入)')}\n"
                        f"P2: {next((c.name for c in self.clients if c.role == 'P2'), '(未加入)')}\n"
                    )
                    _send_json_line(client.sock, {"type": "render", "screen": room_text, "done": False})
                else:
                    role = client.role or "P2"
                    screen = self.uis[role].render() if role in self.uis else "(未分配角色)"
                    payload = {"type": "render", "screen": screen, "done": bool(self.state.done)}
                    if self.state.done:
                        p1_score, p2_score = _compute_scores(self.state.map)
                        payload["result"] = {
                            "winner": self.state.winner,
                            "p1_score": p1_score,
                            "p2_score": p2_score,
                            "reason": self.state.reason,
                        }
                    _send_json_line(client.sock, payload)
            except OSError:
                continue
        if self.state is not None and self.state.done:
            self._close_room()


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Minimal 2P room service")
    p.add_argument("--bind", default="0.0.0.0", help="0.0.0.0 for LAN, 127.0.0.1 for local only")
    return p


def main() -> int:
    args = build_parser().parse_args()
    svc = Minimal2PService(bind_host=args.bind)
    svc.run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
