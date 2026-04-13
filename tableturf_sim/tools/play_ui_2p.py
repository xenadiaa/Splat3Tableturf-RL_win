#!/usr/bin/env python3
"""Play 2P game with local/LAN room discovery and optional auto strategies."""

from __future__ import annotations

import argparse
from collections import deque
import json
import msvcrt
from pathlib import Path
import re
import sys
import time
from contextlib import contextmanager
from typing import Dict, List, Tuple
from uuid import uuid4

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.engine.env_core import Action, init_state, step
from src.net.room_net import HostRoomServer, RoomClient, discover_rooms
from src.strategy import load_strategy_config
from src.strategy.registry import choose_action_from_strategy_id
from src.utils.deck_utils import deck_display_name
from src.utils.player_deck_utils import (
    DECK_MAX_INDEX,
    DECK_MIN_INDEX,
    check_player_deck,
    get_player_deck_card_numbers,
    get_player_deck_name,
)
from src.view.gamepad_ui import TerminalGamepadUI


ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")
CSI_ARROW_RE = re.compile(r"^(?:\[|O)(?:[0-9;]*)([ABCD])")
_KEY_BUFFER: deque[str] = deque()


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="2P UI play mode with local/LAN room support")
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--name", default="Player")
    p.add_argument("--player-id", default="")
    p.add_argument("--line-input", action="store_true")
    return p


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
    if ch in ("i", "I"):
        return "UP"
    if ch in ("k", "K"):
        return "DOWN"
    if ch in ("j", "J"):
        return "LEFT"
    if ch in ("l", "L"):
        return "RIGHT"
    return ch


def _clear() -> None:
    sys.stdout.write("\r\033[2J\033[H")
    sys.stdout.flush()


def _menu_select_raw(title: str, items: List[str], hint: str, start_idx: int = 0) -> int:
    idx = max(0, min(start_idx, len(items) - 1)) if items else 0
    while True:
        _clear()
        print(title)
        print(hint)
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


def _list_valid_player_decks() -> List[Tuple[int, str]]:
    out: List[Tuple[int, str]] = []
    for i in range(DECK_MIN_INDEX, DECK_MAX_INDEX + 1):
        chk = check_player_deck(i, require_full_15=True)
        if chk.get("ok"):
            out.append((i, chk.get("deck_name") or get_player_deck_name(i)))
    return out


def _load_map_ids() -> List[str]:
    path = PROJECT_ROOT / "data" / "maps" / "MiniGameMapInfo.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    return [str(item["id"]) for item in data]


def _build_room_create_config(args: argparse.Namespace) -> dict:
    room_mode_idx = _menu_select_raw("[创建房间]", ["本机房间", "局域网房间"], "方向键上下移动，z确认")
    bind_mode = "local" if room_mode_idx == 0 else "lan"
    map_ids = _load_map_ids()
    map_idx = _menu_select_raw("[选择地图]", map_ids, "方向键上下移动，z确认")
    valid_decks = _list_valid_player_decks()
    deck_labels = [f"deck={idx:02d} | {name}" for idx, name in valid_decks]
    deck_idx = _menu_select_raw("[选择主机牌组]", deck_labels, "方向键上下移动，z确认")
    room_name = f"{args.name}-{uuid4().hex[:6]}"
    return {
        "bind_mode": bind_mode,
        "map_id": map_ids[map_idx],
        "p1_deck_index": valid_decks[deck_idx][0],
        "room_name": room_name,
    }


def _build_join_config(_args: argparse.Namespace) -> dict:
    rooms = discover_rooms()
    if not rooms:
        raise RuntimeError("未搜索到房间")
    labels = [f"{r.get('room_name')} | {r.get('mode')} | {r.get('map_id')} | {r.get('host')}" for r in rooms]
    room_idx = _menu_select_raw("[搜索房间]", labels, "方向键上下移动，z确认")
    valid_decks = _list_valid_player_decks()
    deck_labels = [f"deck={idx:02d} | {name}" for idx, name in valid_decks]
    deck_idx = _menu_select_raw("[选择加入方牌组]", deck_labels, "方向键上下移动，z确认")
    return {
        "room": rooms[room_idx],
        "p2_deck_index": valid_decks[deck_idx][0],
    }


def _local_submit_factory(send_remote, local_player: str):
    def _submit(state, action: Action):
        ok, reason, payload = step(state, action)
        if ok:
            send_remote(action)
        return ok, reason, payload

    return _submit


def _apply_remote_actions(state, msgs: List[dict]) -> None:
    for msg in msgs:
        if msg.get("type") != "action":
            continue
        action = Action(**msg["action"])
        step(state, action)


def _maybe_auto_step(state, local_player: str):
    cfg = load_strategy_config()
    side_cfg = cfg["auto_battle"]["p1" if local_player == "P1" else "p2"]
    if not side_cfg.get("enabled"):
        return None
    if local_player in state.pending_actions:
        return None
    return choose_action_from_strategy_id(state, local_player, str(side_cfg["strategy_id"]))


def _run_ui_loop(state, local_player: str, send_remote, line_input: bool = False) -> int:
    ui = TerminalGamepadUI(state, local_player=local_player, submit_action_fn=_local_submit_factory(send_remote, local_player))
    popup: str | None = None
    while True:
        auto_action = _maybe_auto_step(state, local_player)
        if auto_action is not None and not state.done:
            ok, reason, _payload = ui.submit_action_fn(state, auto_action)
            if ok:
                popup = f"自动出牌: {reason}"

        _clear()
        print("键位: 方向键移动  z确认  x取消  a逆时针  s顺时针  q牌组  +=投降")
        print(ui.render())
        if popup:
            print("\n" + "-" * 32)
            print(popup)
            print("(按任意键继续)")
        if state.done and popup is None:
            popup = f"游戏结束\nwinner={state.winner}\nlog={state.log_path}"

        k = input("输入按键: ").strip() if line_input else _read_key()
        if popup:
            popup = None
            if state.done:
                return 0
            continue
        out = ui.handle_key(k)
        if out:
            popup = ANSI_RE.sub("", out)


def _host_flow(args: argparse.Namespace, cfg: dict) -> int:
    bind_host = "127.0.0.1" if cfg["bind_mode"] == "local" else "0.0.0.0"
    p1_ids = get_player_deck_card_numbers(cfg["p1_deck_index"], require_full_15=True)
    p1_deck_name = get_player_deck_name(cfg["p1_deck_index"])
    room_info = {
        "room_id": uuid4().hex[:8],
        "room_name": cfg["room_name"],
        "mode": cfg["bind_mode"],
        "map_id": cfg["map_id"],
    }
    host = HostRoomServer(room_info=room_info, bind_host=bind_host)
    host.start()
    _clear()
    print(f"房间已创建: {room_info['room_name']}")
    print(f"模式: {cfg['bind_mode']}  地图: {cfg['map_id']}")
    print("等待加入...")
    join_msg = host.wait_for_client()
    p2_ids = list(join_msg["deck_ids"])
    seed = args.seed if args.seed is not None else int(time.time() * 1000) & 0x7FFFFFFF
    init_payload = {
        "room_id": room_info["room_id"],
        "seed": seed,
        "map_id": cfg["map_id"],
        "p1_deck_ids": p1_ids,
        "p2_deck_ids": p2_ids,
        "p1_name": args.name,
        "p2_name": join_msg.get("player_name", "Guest"),
        "p1_player_id": args.player_id or f"P1_{uuid4().hex[:6]}",
        "p2_player_id": join_msg.get("player_id", f"P2_{uuid4().hex[:6]}"),
        "p1_deck_name": p1_deck_name,
        "p2_deck_name": str(join_msg.get("deck_name", "")),
        "client_player": "P2",
    }
    host.send_init(init_payload)
    state = init_state(
        map_id=cfg["map_id"],
        p1_deck_ids=p1_ids,
        p2_deck_ids=p2_ids,
        seed=seed,
        mode="2P",
        p1_player_id=init_payload["p1_player_id"],
        p2_player_id=init_payload["p2_player_id"],
        p1_player_name=init_payload["p1_name"],
        p2_player_name=init_payload["p2_name"],
        p1_deck_name=init_payload["p1_deck_name"],
        p2_deck_name=init_payload["p2_deck_name"],
    )
    original_handle_key = None
    send_remote = host.send_action

    if args.line_input:
        ui = TerminalGamepadUI(state, local_player="P1", submit_action_fn=_local_submit_factory(send_remote, "P1"))
        while not state.done:
            _apply_remote_actions(state, host.poll_messages())
            auto_action = _maybe_auto_step(state, "P1")
            if auto_action is not None and "P1" not in state.pending_actions:
                ui.submit_action_fn(state, auto_action)
                continue
            print("\n" + "=" * 80)
            print(ui.render())
            raw = input("输入按键: ").strip()
            if not raw:
                continue
            out = ui.handle_key(raw)
            if out:
                print("\n" + out)
                _ = input("按回车返回对局页...")
        print(f"winner={state.winner}\nlog_path={state.log_path}")
        return 0

    with _raw_mode(sys.stdin):
        ui = TerminalGamepadUI(state, local_player="P1", submit_action_fn=_local_submit_factory(send_remote, "P1"))
        popup: str | None = None
        while True:
            _apply_remote_actions(state, host.poll_messages())
            auto_action = _maybe_auto_step(state, "P1")
            if auto_action is not None and "P1" not in state.pending_actions and not state.done:
                ok, reason, _ = ui.submit_action_fn(state, auto_action)
                if ok:
                    popup = f"自动出牌: {reason}"
            _clear()
            print("键位: 方向键移动  z确认  x取消  a逆时针  s顺时针  q牌组  +=投降")
            print(ui.render())
            if popup:
                print("\n" + "-" * 32)
                print(popup)
                print("(按任意键继续)")
            if state.done and popup is None:
                popup = f"游戏结束\nwinner={state.winner}\nlog={state.log_path}"
            k = _read_key()
            if popup:
                popup = None
                if state.done:
                    host.stop()
                    return 0
                continue
            out = ui.handle_key(k)
            if out:
                popup = ANSI_RE.sub("", out)


def _join_flow(args: argparse.Namespace, cfg: dict) -> int:
    room = cfg["room"]
    p2_ids = get_player_deck_card_numbers(cfg["p2_deck_index"], require_full_15=True)
    client = RoomClient(str(room["host"]))
    client.connect(
        {
            "player_name": args.name,
            "player_id": args.player_id or f"P2_{uuid4().hex[:6]}",
            "deck_ids": p2_ids,
            "deck_name": get_player_deck_name(cfg["p2_deck_index"]),
        }
    )
    init_msg = client.wait_for_init()
    state = init_state(
        map_id=str(init_msg["map_id"]),
        p1_deck_ids=list(init_msg["p1_deck_ids"]),
        p2_deck_ids=list(init_msg["p2_deck_ids"]),
        seed=int(init_msg["seed"]),
        mode="2P",
        p1_player_id=str(init_msg["p1_player_id"]),
        p2_player_id=str(init_msg["p2_player_id"]),
        p1_player_name=str(init_msg["p1_name"]),
        p2_player_name=str(init_msg["p2_name"]),
        p1_deck_name=str(init_msg["p1_deck_name"]),
        p2_deck_name=str(init_msg["p2_deck_name"]),
    )
    local_player = str(init_msg.get("client_player", "P2"))

    if args.line_input:
        ui = TerminalGamepadUI(state, local_player=local_player, submit_action_fn=_local_submit_factory(client.send_action, local_player))
        while not state.done:
            _apply_remote_actions(state, client.poll_messages())
            auto_action = _maybe_auto_step(state, local_player)
            if auto_action is not None and local_player not in state.pending_actions:
                ui.submit_action_fn(state, auto_action)
                continue
            print("\n" + "=" * 80)
            print(ui.render())
            raw = input("输入按键: ").strip()
            if not raw:
                continue
            out = ui.handle_key(raw)
            if out:
                print("\n" + out)
                _ = input("按回车返回对局页...")
        print(f"winner={state.winner}\nlog_path={state.log_path}")
        client.close()
        return 0

    with _raw_mode(sys.stdin):
        ui = TerminalGamepadUI(state, local_player=local_player, submit_action_fn=_local_submit_factory(client.send_action, local_player))
        popup: str | None = None
        while True:
            _apply_remote_actions(state, client.poll_messages())
            auto_action = _maybe_auto_step(state, local_player)
            if auto_action is not None and local_player not in state.pending_actions and not state.done:
                ok, reason, _ = ui.submit_action_fn(state, auto_action)
                if ok:
                    popup = f"自动出牌: {reason}"
            _clear()
            print("键位: 方向键移动  z确认  x取消  a逆时针  s顺时针  q牌组  +=投降")
            print(ui.render())
            if popup:
                print("\n" + "-" * 32)
                print(popup)
                print("(按任意键继续)")
            if state.done and popup is None:
                popup = f"游戏结束\nwinner={state.winner}\nlog={state.log_path}"
            k = _read_key()
            if popup:
                popup = None
                if state.done:
                    client.close()
                    return 0
                continue
            out = ui.handle_key(k)
            if out:
                popup = ANSI_RE.sub("", out)


def main() -> int:
    args = build_parser().parse_args()
    with _raw_mode(sys.stdin):
        mode_idx = _menu_select_raw("[2P模式]", ["创建房间", "搜索房间"], "方向键上下移动，z确认")
        if mode_idx == 0:
            cfg = _build_room_create_config(args)
        else:
            cfg = _build_join_config(args)

    if mode_idx == 0:
        return _host_flow(args, cfg)
    return _join_flow(args, cfg)


if __name__ == "__main__":
    raise SystemExit(main())
