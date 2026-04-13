#!/usr/bin/env python3
"""Play 1P game with realtime terminal UI (ANSI + raw key input) and pre-game selection flow."""

from __future__ import annotations

import argparse
from collections import deque
import msvcrt
import re
import sys
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Dict, List, Tuple

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.engine.env_core import init_state, step
from src.engine.loaders import load_map
from src.strategy import load_strategy_config
from src.strategy.registry import choose_action_from_strategy_id
from src.utils.deck_utils import (
    deck_display_name,
    load_deck_cards_by_rowid,
    npc_name_zh,
)
from src.utils.npc_strategy_utils import load_npc_strategy_table, resolve_nn_spec
from src.utils.player_deck_utils import (
    DECK_MAX_INDEX,
    DECK_MIN_INDEX,
    check_player_deck,
    get_player_deck_card_numbers,
    get_player_deck_name,
)
from src.view.gamepad_ui import TerminalGamepadUI


AI_TYPE_TO_BOT_STYLE = {
    "Aggressive": "aggressive",
    "Balance": "balanced",
    "AccumulateSpecial": "conservative",
}
LEVEL_TO_BOT_LEVEL = {
    0: "low",
    1: "mid",
    2: "high",
}
MANUAL_AUTO_STRATEGY_ID = "default:aggressive:high"
ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")
CSI_ARROW_RE = re.compile(r"^(?:\[|O)(?:[0-9;]*)([ABCD])")
_KEY_BUFFER: deque[str] = deque()


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="1P UI play mode with keyboard mapping")
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--p1-id", default="U1001")
    p.add_argument("--p1-name", default="Alice")
    p.add_argument("--line-input", action="store_true", help="Use old line-by-line input mode (debug)")
    p.add_argument("--direct", action="store_true", help="Skip selection wizard and use direct args")

    p.add_argument("--map", default="Square")
    p.add_argument("--p1-deck", type=int, default=15)
    p.add_argument(
        "--p1-npc-deck-rowid",
        default="",
        help="Use NPC preset deck rowid for P1 (e.g. MiniGame_Aori). Overrides --p1-deck.",
    )
    p.add_argument(
        "--p1-npc-name",
        default="",
        help="Use highest-level deck of this NPC for P1 (by npc_name). Overrides --p1-deck when rowid is empty.",
    )
    p.add_argument("--p2-deck", type=int, default=1)
    p.add_argument("--bot-style", choices=["balanced", "aggressive", "conservative"], default="aggressive")
    p.add_argument("--bot-level", choices=["low", "mid", "high"], default="high")
    p.add_argument("--p2-id", default="BOT001")
    p.add_argument("--p2-name", default="Bot")
    return p


def _list_valid_player_decks() -> List[Tuple[int, str]]:
    out: List[Tuple[int, str]] = []
    for i in range(DECK_MIN_INDEX, DECK_MAX_INDEX + 1):
        chk = check_player_deck(i, require_full_15=True)
        if chk.get("ok"):
            out.append((i, chk.get("deck_name") or get_player_deck_name(i)))
    return out


def _deck_ids_from_rowid(deck_rowid: str) -> List[int]:
    cards = load_deck_cards_by_rowid(deck_rowid)
    return [c["number"] for c in cards]


def _resolve_npc_deck_rowid_by_name(npc_name: str) -> str:
    q = str(npc_name).strip().lower()
    if not q:
        raise ValueError("empty npc name")
    rows = load_npc_strategy_table()
    for row in rows:
        if str(row.get("npc_name", "")).strip().lower() != q:
            continue
        strategies = sorted(row.get("strategies", []), key=lambda x: int(x["ai_level"]))
        if not strategies:
            break
        return str(strategies[-1]["deck_rowid"])
    raise ValueError(f"npc not found or no strategy: {npc_name}")


def _direct_config(args: argparse.Namespace) -> Dict[str, object]:
    p1_deck_ids: List[int]
    p1_deck_name: str
    p1_deck_source: str
    if args.p1_npc_deck_rowid:
        deck_rowid = str(args.p1_npc_deck_rowid).strip()
        p1_deck_ids = _deck_ids_from_rowid(deck_rowid)
        p1_deck_name = deck_display_name(deck_rowid)
        p1_deck_source = f"npc_rowid:{deck_rowid}"
    elif args.p1_npc_name:
        deck_rowid = _resolve_npc_deck_rowid_by_name(args.p1_npc_name)
        p1_deck_ids = _deck_ids_from_rowid(deck_rowid)
        p1_deck_name = deck_display_name(deck_rowid)
        p1_deck_source = f"npc_name:{args.p1_npc_name}"
    else:
        p1_deck_ids = get_player_deck_card_numbers(args.p1_deck, require_full_15=True)
        p1_deck_name = get_player_deck_name(args.p1_deck)
        p1_deck_source = f"player_deck:{args.p1_deck}"

    return {
        "map_id": args.map,
        "p1_deck_ids": p1_deck_ids,
        "p1_deck_name": p1_deck_name,
        "p1_deck_source": p1_deck_source,
        "p2_deck_ids": get_player_deck_card_numbers(args.p2_deck, require_full_15=True),
        "bot_style": args.bot_style,
        "bot_level": args.bot_level,
        "bot_nn_spec": {},
        "p2_id": args.p2_id,
        "p2_name": args.p2_name,
        "p2_deck_name": get_player_deck_name(args.p2_deck),
    }


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


def _redraw_frame(body: str, *, first_frame: bool) -> None:
    if first_frame:
        sys.stdout.write("\r\033[2J\033[H")
    else:
        # Move to top-left and clear previous drawn region.
        sys.stdout.write("\r\033[H\033[J")
    sys.stdout.write(body)
    if not body.endswith("\n"):
        sys.stdout.write("\n")
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


def _wizard_config_raw() -> Dict[str, object]:
    npcs = sorted(
        load_npc_strategy_table(),
        key=lambda n: int(n.get("order", 10**9)),
    )
    npc_labels: List[str] = []
    for n in npcs:
        en = n.get("npc_name", "")
        zh = npc_name_zh(en) or "未找到中文"
        strategies = n.get("strategies", [])
        map_text = "地图:未知"
        if strategies:
            best = sorted(strategies, key=lambda x: int(x["ai_level"]))[-1]
            map_id = best["map_id"]
            try:
                map_zh = load_map(map_id).name
                map_text = f"地图:{map_zh}"
            except Exception:
                map_text = f"地图:{map_id}"
        npc_labels.append(f"{en} [{zh}] | order={n.get('order')} | {map_text}")

    npc_idx = _menu_select_raw("[选择NPC]", npc_labels, "方向键上下移动，z确认")
    npc = npcs[npc_idx]

    strategies = sorted(npc.get("strategies", []), key=lambda x: int(x["ai_level"]))
    if not strategies:
        raise RuntimeError(f"NPC {npc.get('npc_name')} 没有策略数据")

    diff_labels: List[str] = []
    for s in strategies:
        map_name_zh = load_map(s["map_id"]).name
        diff_labels.append(
            f"level={s['ai_level']} | ai={s['ai_type']} | map={s['map_id']}({map_name_zh}) | deck={deck_display_name(s['deck_rowid'])}"
        )

    diff_idx = _menu_select_raw("[选择难度]", diff_labels, "方向键上下移动，z确认")
    strategy = strategies[diff_idx]

    deck_mode_idx = _menu_select_raw(
        "[选择P1牌组来源]",
        ["玩家牌组(0~32)", "NPC预设牌组(deck_rowid)"],
        "方向键上下移动，z确认",
    )
    if deck_mode_idx == 0:
        valid_decks = _list_valid_player_decks()
        if not valid_decks:
            raise RuntimeError("没有可用的完整玩家牌组（0~32中无15张合法卡组）")
        deck_labels = [f"deck={idx:02d} | {name}" for idx, name in valid_decks]
        deck_pick = _menu_select_raw("[选择玩家牌组]", deck_labels, "方向键上下移动，z确认")
        p1_deck_index = valid_decks[deck_pick][0]
        p1_deck_ids = get_player_deck_card_numbers(p1_deck_index, require_full_15=True)
        p1_deck_name = get_player_deck_name(p1_deck_index)
        p1_deck_source = f"player_deck:{p1_deck_index}"
    else:
        npc_rows = sorted(load_npc_strategy_table(), key=lambda r: int(r.get("order", 10**9)))
        rowid_meta: Dict[str, Tuple[str, str]] = {}
        for row in npc_rows:
            npc_en = str(row.get("npc_name", ""))
            npc_zh = npc_name_zh(npc_en) or "未找到中文"
            for s in row.get("strategies", []):
                deck_rowid = str(s.get("deck_rowid", "")).strip()
                if not deck_rowid or deck_rowid in rowid_meta:
                    continue
                map_id = str(s.get("map_id", "")).strip()
                try:
                    map_zh = load_map(map_id).name
                except Exception:
                    map_zh = map_id or "未知地图"
                rowid_meta[deck_rowid] = (npc_zh, map_zh)

        rowids = sorted(rowid_meta.keys())
        if not rowids:
            raise RuntimeError("找不到 NPC 预设牌组")
        deck_labels = []
        for r in rowids:
            npc_zh, map_zh = rowid_meta.get(r, ("未找到中文", "未知地图"))
            deck_labels.append(f"{deck_display_name(r)} ({r}) [{npc_zh}+{map_zh}]")
        pick = _menu_select_raw("[选择NPC预设牌组]", deck_labels, "方向键上下移动，z确认")
        picked_rowid = rowids[pick]
        p1_deck_ids = _deck_ids_from_rowid(picked_rowid)
        p1_deck_name = deck_display_name(picked_rowid)
        p1_deck_source = f"npc_rowid:{picked_rowid}"

    p2_deck_cards = load_deck_cards_by_rowid(strategy["deck_rowid"])
    p2_deck_ids = [c["number"] for c in p2_deck_cards]

    npc_en = npc.get("npc_name", "NPC")
    level = int(strategy["ai_level"])
    bot_level = str(strategy.get("bot_level", LEVEL_TO_BOT_LEVEL.get(level, "high")))
    bot_style = str(strategy.get("bot_style", AI_TYPE_TO_BOT_STYLE.get(strategy["ai_type"], "balanced")))
    nn_spec = resolve_nn_spec(npc_en) or {}

    return {
        "map_id": strategy["map_id"],
        "p1_deck_ids": p1_deck_ids,
        "p1_deck_name": p1_deck_name,
        "p1_deck_source": p1_deck_source,
        "p2_deck_ids": p2_deck_ids,
        "bot_style": bot_style,
        "bot_level": bot_level,
        "bot_nn_spec": nn_spec,
        "p2_id": f"NPC{npc.get('order', 0)}",
        "p2_name": npc_en,
        "p2_deck_name": deck_display_name(strategy["deck_rowid"]),
    }


def _run_game_raw(args: argparse.Namespace, conf: Dict[str, object]) -> int:
    p1_ids = list(conf["p1_deck_ids"])
    p1_deck_name = str(conf.get("p1_deck_name", ""))
    p2_ids = list(conf["p2_deck_ids"])

    state = init_state(
        map_id=str(conf["map_id"]),
        p1_deck_ids=p1_ids,
        p2_deck_ids=p2_ids,
        seed=args.seed,
        mode="1P",
        bot_style=str(conf["bot_style"]),
        bot_level=str(conf["bot_level"]),
        bot_nn_spec=dict(conf.get("bot_nn_spec", {})),
        p1_player_id=args.p1_id,
        p2_player_id=str(conf["p2_id"]),
        p1_player_name=args.p1_name,
        p2_player_name=str(conf["p2_name"]),
        p1_deck_name=p1_deck_name,
        p2_deck_name=str(conf["p2_deck_name"]),
    )

    ui = TerminalGamepadUI(state)
    popup: str | None = None
    last_body = ""
    first_frame = True
    last_draw_ts = 0.0

    while True:
        cfg = load_strategy_config()
        p1_auto = cfg["auto_battle"]["p1"]
        p1_ready = "P1" not in state.pending_actions and not state.done
        p1_nn_strategy_id = str(p1_auto.get("strategy_id", ""))
        p1_nn_ready = bool(p1_auto.get("enabled")) and p1_ready and p1_nn_strategy_id.startswith("nn:")
        lines = ["键位: 方向键移动  z确认  x取消  a逆时针  s顺时针  q牌组  m激进一步  n神经网络一步  +=投降"]
        if p1_ready:
            lines.append(f"自动对战: 按 m 执行一步策略 ({MANUAL_AUTO_STRATEGY_ID})")
        if p1_nn_ready:
            lines.append(f"神经网络: 按 n 执行一步策略 ({p1_nn_strategy_id})")
        lines.append(ui.render())
        if popup:
            lines.extend(["", "-" * 32, popup, "(按任意键继续)"])
        body = "\n".join(lines) + "\n"
        now = time.monotonic()
        if body != last_body and (now - last_draw_ts) >= 0.05:
            _redraw_frame(body, first_frame=first_frame)
            first_frame = False
            last_body = body
            last_draw_ts = now

        if state.done and popup is None:
            popup = f"游戏结束\nwinner={state.winner}\nlog={state.log_path}"

        k = _read_key()
        if popup:
            popup = None
            if state.done:
                return 0
            continue

        if p1_ready and k == "m":
            try:
                auto_action = choose_action_from_strategy_id(state, "P1", MANUAL_AUTO_STRATEGY_ID)
                ok, reason, _payload = step(state, auto_action)
                popup = f"自动出牌: {reason}" if ok else f"自动出牌失败: {reason}"
            except Exception as exc:
                popup = f"自动策略加载失败: {exc}"
            continue
        if p1_nn_ready and k == "n":
            try:
                auto_action = choose_action_from_strategy_id(state, "P1", p1_nn_strategy_id)
                ok, reason, _payload = step(state, auto_action)
                popup = f"神经网络出牌: {reason}" if ok else f"神经网络出牌失败: {reason}"
            except Exception as exc:
                popup = f"神经网络策略加载失败: {exc}"
            continue

        out = ui.handle_key(k)
        if out:
            popup = ANSI_RE.sub("", out)


def _run_game_line_input(args: argparse.Namespace, conf: Dict[str, object]) -> int:
    p1_ids = list(conf["p1_deck_ids"])
    p1_deck_name = str(conf.get("p1_deck_name", ""))
    p2_ids = list(conf["p2_deck_ids"])

    state = init_state(
        map_id=str(conf["map_id"]),
        p1_deck_ids=p1_ids,
        p2_deck_ids=p2_ids,
        seed=args.seed,
        mode="1P",
        bot_style=str(conf["bot_style"]),
        bot_level=str(conf["bot_level"]),
        bot_nn_spec=dict(conf.get("bot_nn_spec", {})),
        p1_player_id=args.p1_id,
        p2_player_id=str(conf["p2_id"]),
        p1_player_name=args.p1_name,
        p2_player_name=str(conf["p2_name"]),
        p1_deck_name=p1_deck_name,
        p2_deck_name=str(conf["p2_deck_name"]),
    )

    ui = TerminalGamepadUI(state)
    print("键位: 方向键移动  z确认  x取消  a逆时针  s顺时针  q牌组  m自动一步  +=投降")
    while not state.done:
        cfg = load_strategy_config()
        p1_auto = cfg["auto_battle"]["p1"]
        p1_ready = "P1" not in state.pending_actions
        p1_nn_strategy_id = str(p1_auto.get("strategy_id", ""))
        p1_nn_ready = bool(p1_auto.get("enabled")) and p1_ready and p1_nn_strategy_id.startswith("nn:")
        print("\n" + "=" * 80)
        print("键位: 方向键移动  z确认  x取消  a逆时针  s顺时针  q牌组  m激进一步  n神经网络一步  +=投降")
        if p1_ready:
            print(f"自动对战: 按 m 执行一步策略 ({MANUAL_AUTO_STRATEGY_ID})")
        if p1_nn_ready:
            print(f"神经网络: 按 n 执行一步策略 ({p1_nn_strategy_id})")
        print(ui.render())
        raw = input("输入按键: ").strip()
        if not raw:
            continue
        if p1_ready and raw == "m":
            try:
                auto_action = choose_action_from_strategy_id(state, "P1", MANUAL_AUTO_STRATEGY_ID)
                ok, reason, payload = step(state, auto_action)
                print(f"\n自动出牌{'成功' if ok else '失败'}: {reason} {payload}")
            except Exception as exc:
                print(f"\n自动策略加载失败: {exc}")
            continue
        if p1_nn_ready and raw == "n":
            try:
                auto_action = choose_action_from_strategy_id(state, "P1", p1_nn_strategy_id)
                ok, reason, payload = step(state, auto_action)
                print(f"\n神经网络出牌{'成功' if ok else '失败'}: {reason} {payload}")
            except Exception as exc:
                print(f"\n神经网络策略加载失败: {exc}")
            continue
        out = ui.handle_key(raw)
        if out:
            print("\n" + out)
            _ = input("按回车返回对局页...")

    print("\n=== Game Over ===")
    print(ui.render())
    print(f"winner={state.winner}")
    print(f"log_path={state.log_path}")
    return 0


def main() -> int:
    args = build_parser().parse_args()

    if args.line_input:
        conf = _direct_config(args) if args.direct else _wizard_config_raw()
        return _run_game_line_input(args, conf)

    # Realtime mode: keep raw input enabled for both selection flow and in-game loop.
    with _raw_mode(sys.stdin):
        conf = _direct_config(args) if args.direct else _wizard_config_raw()
        return _run_game_raw(args, conf)


if __name__ == "__main__":
    raise SystemExit(main())
