"""Simple local/LAN room networking for 2P terminal play."""

from __future__ import annotations

from dataclasses import asdict
import json
import queue
import socket
import threading
import time
from typing import Dict, List, Optional

from src.engine.env_core import Action


DISCOVERY_PORT = 38555
ROOM_TCP_PORT = 38556
DISCOVERY_MAGIC = "TABLETURF_DISCOVER_V1"
BUFFER_SIZE = 65535


def _send_json_line(sock: socket.socket, payload: dict) -> None:
    data = (json.dumps(payload, ensure_ascii=False) + "\n").encode("utf-8")
    sock.sendall(data)


def _guess_local_ip() -> str:
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        sock.connect(("8.8.8.8", 80))
        return str(sock.getsockname()[0])
    except OSError:
        try:
            return socket.gethostbyname(socket.gethostname())
        except OSError:
            return "127.0.0.1"
    finally:
        sock.close()


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


class DiscoveryResponder:
    def __init__(self, room_info: dict, bind_host: str) -> None:
        self.room_info = room_info
        self.bind_host = bind_host
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None

    def start(self) -> None:
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()

    def _run(self) -> None:
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        try:
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            sock.bind((self.bind_host, DISCOVERY_PORT))
            sock.settimeout(0.5)
            while not self._stop.is_set():
                try:
                    data, addr = sock.recvfrom(BUFFER_SIZE)
                except socket.timeout:
                    continue
                if data.decode("utf-8", errors="ignore").strip() != DISCOVERY_MAGIC:
                    continue
                reply = dict(self.room_info)
                reply["host"] = _guess_local_ip() if self.bind_host == "0.0.0.0" else self.bind_host
                sock.sendto(json.dumps(reply, ensure_ascii=False).encode("utf-8"), addr)
        finally:
            sock.close()


def discover_rooms(timeout_sec: float = 1.0) -> List[dict]:
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    rooms: Dict[str, dict] = {}
    try:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
        sock.settimeout(0.2)
        payload = DISCOVERY_MAGIC.encode("utf-8")
        for target in ("127.0.0.1", "255.255.255.255"):
            try:
                sock.sendto(payload, (target, DISCOVERY_PORT))
            except OSError:
                continue
        deadline = time.time() + timeout_sec
        while time.time() < deadline:
            try:
                data, _addr = sock.recvfrom(BUFFER_SIZE)
            except socket.timeout:
                continue
            try:
                room = json.loads(data.decode("utf-8"))
            except Exception:
                continue
            room_id = str(room.get("room_id", ""))
            if room_id:
                rooms[room_id] = room
    finally:
        sock.close()
    return list(rooms.values())


class HostRoomServer:
    def __init__(self, room_info: dict, bind_host: str, tcp_port: int = ROOM_TCP_PORT) -> None:
        self.room_info = room_info
        self.bind_host = bind_host
        self.tcp_port = tcp_port
        self.discovery = DiscoveryResponder(room_info=room_info, bind_host=bind_host)
        self.client_sock: Optional[socket.socket] = None
        self.client_queue: "queue.Queue[dict]" = queue.Queue()
        self._reader = _JsonLineReader()
        self._recv_thread: Optional[threading.Thread] = None

    def start(self) -> None:
        self.discovery.start()

    def stop(self) -> None:
        self.discovery.stop()
        if self.client_sock is not None:
            try:
                self.client_sock.close()
            except OSError:
                pass

    def wait_for_client(self) -> dict:
        srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            srv.bind((self.bind_host, self.tcp_port))
            srv.listen(1)
            conn, _addr = srv.accept()
            self.client_sock = conn
            join_msg = self._recv_one_blocking(conn)
            self._recv_thread = threading.Thread(target=self._recv_loop, daemon=True)
            self._recv_thread.start()
            return join_msg
        finally:
            srv.close()

    def send_init(self, payload: dict) -> None:
        if self.client_sock is None:
            raise RuntimeError("client not connected")
        _send_json_line(self.client_sock, {"type": "init", **payload})

    def send_action(self, action: Action) -> None:
        if self.client_sock is None:
            raise RuntimeError("client not connected")
        _send_json_line(self.client_sock, {"type": "action", "action": asdict(action)})

    def poll_messages(self) -> List[dict]:
        out: List[dict] = []
        while True:
            try:
                out.append(self.client_queue.get_nowait())
            except queue.Empty:
                break
        return out

    def _recv_one_blocking(self, sock: socket.socket) -> dict:
        reader = _JsonLineReader()
        while True:
            chunk = sock.recv(BUFFER_SIZE)
            if not chunk:
                raise RuntimeError("client disconnected before join")
            msgs = reader.feed(chunk)
            if msgs:
                return msgs[0]

    def _recv_loop(self) -> None:
        assert self.client_sock is not None
        sock = self.client_sock
        while True:
            try:
                chunk = sock.recv(BUFFER_SIZE)
            except OSError:
                break
            if not chunk:
                break
            try:
                msgs = self._reader.feed(chunk)
            except Exception:
                continue
            for msg in msgs:
                self.client_queue.put(msg)


class RoomClient:
    def __init__(self, host: str, tcp_port: int = ROOM_TCP_PORT) -> None:
        self.host = host
        self.tcp_port = tcp_port
        self.sock: Optional[socket.socket] = None
        self.queue: "queue.Queue[dict]" = queue.Queue()
        self._reader = _JsonLineReader()
        self._recv_thread: Optional[threading.Thread] = None

    def connect(self, join_payload: dict) -> None:
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.sock.connect((self.host, self.tcp_port))
        _send_json_line(self.sock, {"type": "join", **join_payload})
        self._recv_thread = threading.Thread(target=self._recv_loop, daemon=True)
        self._recv_thread.start()

    def close(self) -> None:
        if self.sock is not None:
            try:
                self.sock.close()
            except OSError:
                pass

    def send_action(self, action: Action) -> None:
        if self.sock is None:
            raise RuntimeError("not connected")
        _send_json_line(self.sock, {"type": "action", "action": asdict(action)})

    def poll_messages(self) -> List[dict]:
        out: List[dict] = []
        while True:
            try:
                out.append(self.queue.get_nowait())
            except queue.Empty:
                break
        return out

    def wait_for_init(self, timeout_sec: float = 30.0) -> dict:
        deadline = time.time() + timeout_sec
        while time.time() < deadline:
            for msg in self.poll_messages():
                if msg.get("type") == "init":
                    return msg
            time.sleep(0.05)
        raise TimeoutError("wait for init timed out")

    def _recv_loop(self) -> None:
        assert self.sock is not None
        sock = self.sock
        while True:
            try:
                chunk = sock.recv(BUFFER_SIZE)
            except OSError:
                break
            if not chunk:
                break
            try:
                msgs = self._reader.feed(chunk)
            except Exception:
                continue
            for msg in msgs:
                self.queue.put(msg)
