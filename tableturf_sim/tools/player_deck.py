#!/usr/bin/env python3
"""Player deck CRUD and preview tool."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.deck_utils import load_card_info  # noqa: E402
from src.utils.localization import lookup_card_name_zh  # noqa: E402
from src.utils.player_deck_utils import (  # noqa: E402
    check_player_deck,
    clear_player_deck_slot,
    get_player_deck,
    get_player_deck_name,
    get_player_deck_sleeve,
    get_player_deck_slot,
    init_player_deck_file,
    list_player_deck_slots,
    set_player_deck_name,
    set_player_deck_sleeve,
    set_player_deck_slot,
)


def _gyml_to_rowid(gyml_path: str) -> str:
    return gyml_path.split("/")[-1].split(".spl__")[0]


def _card_info_from_path(path_value: Optional[str]):
    if not path_value:
        return None
    row_id = _gyml_to_rowid(path_value)
    cards = load_card_info()
    return next((c for c in cards if c.get("__RowId") == row_id), None)


def cmd_preview(deck_index: int) -> int:
    deck = get_player_deck(deck_index)
    print(f"deck_index: {deck_index}")
    print(f"deck_id: {deck.get('__RowId')}")
    print(f"deck_name: {get_player_deck_name(deck_index)}")
    print(f"card_sleeve: {get_player_deck_sleeve(deck_index)}")
    print("slots:")
    for slot, value in list_player_deck_slots(deck_index):
        if not value:
            print(f"- {slot:02d}: (empty)")
            continue
        info = _card_info_from_path(value)
        if info is None:
            print(f"- {slot:02d}: {value} (unknown card)")
            continue
        name = info.get("Name", "")
        zh = lookup_card_name_zh(name) or "未找到中文"
        print(f"- {slot:02d}: #{info.get('Number'):03d} {name} [{zh}]")
    return 0


def cmd_get(deck_index: int, slot: int) -> int:
    value = get_player_deck_slot(deck_index, slot)
    print(f"deck={deck_index}, slot={slot}")
    if not value:
        print("(empty)")
        return 0
    info = _card_info_from_path(value)
    if info is None:
        print(value)
        return 0
    name = info.get("Name", "")
    zh = lookup_card_name_zh(name) or "未找到中文"
    print(f"#{info.get('Number'):03d} {name} [{zh}]")
    print(value)
    return 0


def cmd_set(deck_index: int, slot: int, card_query: str) -> int:
    info = set_player_deck_slot(deck_index, slot, card_query)
    zh = lookup_card_name_zh(info.get("Name", "")) or "未找到中文"
    print(
        f"updated: deck={deck_index}, slot={slot} -> "
        f"#{info.get('Number'):03d} {info.get('Name')} [{zh}]"
    )
    return 0


def cmd_clear(deck_index: int, slot: int) -> int:
    clear_player_deck_slot(deck_index, slot)
    print(f"updated: deck={deck_index}, slot={slot} -> (empty)")
    return 0


def cmd_check(deck_index: int, strict: bool) -> int:
    result = check_player_deck(deck_index, require_full_15=strict)
    print(f"deck={deck_index}")
    print(f"deck_name={result.get('deck_name')}")
    print(f"card_sleeve={result.get('card_sleeve')}")
    print(f"ok={result['ok']}")
    print(f"filled_count={result['filled_count']}/15")
    print(f"empty_slots={result['empty_slots']}")
    print(f"invalid_slots={result['invalid_slots']}")
    print(f"duplicate_cards={result['duplicate_cards']}")
    print(f"errors={result['errors']}")
    return 0 if result["ok"] else 2


def cmd_set_name(deck_index: int, name: str) -> int:
    set_player_deck_name(deck_index, name)
    print(f"updated: deck={deck_index}, deck_name={get_player_deck_name(deck_index)}")
    return 0


def cmd_set_sleeve(deck_index: int, sleeve: str) -> int:
    sleeve_path = set_player_deck_sleeve(deck_index, sleeve)
    print(f"updated: deck={deck_index}, card_sleeve={sleeve_path}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Player deck preview and CRUD.")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_init = sub.add_parser("init", help="Initialize PlayerPresetDeck.json (0~32).")
    p_init.add_argument("--force", action="store_true", help="Overwrite existing file.")

    p_preview = sub.add_parser("preview", help="Preview a player deck.")
    p_preview.add_argument("deck", type=int, help="Deck index: 0~32")

    p_get = sub.add_parser("get", help="Get one slot.")
    p_get.add_argument("deck", type=int, help="Deck index: 0~32")
    p_get.add_argument("slot", type=int, help="Slot index: 1~15")

    p_set = sub.add_parser("set", help="Set one slot by card id/name.")
    p_set.add_argument("deck", type=int, help="Deck index: 0~32")
    p_set.add_argument("slot", type=int, help="Slot index: 1~15")
    p_set.add_argument("card", help="Card id / english name / chinese name")

    p_clear = sub.add_parser("clear", help="Clear one slot.")
    p_clear.add_argument("deck", type=int, help="Deck index: 0~32")
    p_clear.add_argument("slot", type=int, help="Slot index: 1~15")

    p_check = sub.add_parser("check", help="Check deck validity.")
    p_check.add_argument("deck", type=int, help="Deck index: 0~32")
    p_check.add_argument("--strict", action="store_true", help="Require all 15 slots filled.")

    p_set_name = sub.add_parser("set-name", help="Set custom deck name.")
    p_set_name.add_argument("deck", type=int, help="Deck index: 0~32")
    p_set_name.add_argument("name", help="Custom deck name")

    p_set_sleeve = sub.add_parser("set-sleeve", help="Set card sleeve by id/path.")
    p_set_sleeve.add_argument("deck", type=int, help="Deck index: 0~32")
    p_set_sleeve.add_argument("sleeve", help="Sleeve alias: Default/Aori/... or gyml path")

    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    try:
        if args.cmd == "init":
            path = init_player_deck_file(force=args.force)
            print(f"initialized: {path}")
            return 0
        if args.cmd == "preview":
            return cmd_preview(args.deck)
        if args.cmd == "get":
            return cmd_get(args.deck, args.slot)
        if args.cmd == "set":
            return cmd_set(args.deck, args.slot, args.card)
        if args.cmd == "clear":
            return cmd_clear(args.deck, args.slot)
        if args.cmd == "check":
            return cmd_check(args.deck, args.strict)
        if args.cmd == "set-name":
            return cmd_set_name(args.deck, args.name)
        if args.cmd == "set-sleeve":
            return cmd_set_sleeve(args.deck, args.sleeve)
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
