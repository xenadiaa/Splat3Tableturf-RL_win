"""Player deck persistence and CRUD utilities."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from .deck_utils import load_card_info
from .localization import mini_game_card_name_map


PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
PLAYER_DECK_JSON = PROJECT_ROOT / "data" / "cards" / "PlayerPresetDeck.json"

DECK_MIN_INDEX = 0
DECK_MAX_INDEX = 32
DECK_SLOT_COUNT = 15
DEFAULT_CARD_SLEEVE = "Work/Gyml/MiniGameCardSleeve_Default.spl__MiniGameCardSleeve.gyml"


def _default_deck_name(deck_index: int) -> str:
    return f"PlayerDeck_{deck_index:02d}"


def _empty_deck_row(deck_index: int) -> dict:
    return {
        "__RowId": f"MiniGame_PlayerDeck_{deck_index:02d}",
        "Name": _default_deck_name(deck_index),
        "CardSleeve": DEFAULT_CARD_SLEEVE,
        "Card": [None] * DECK_SLOT_COUNT,
    }


def _normalize_player_decks(data: List[dict]) -> List[dict]:
    """Normalize to fixed 33 decks and fixed 15 slots."""
    rows: List[dict] = []
    for i in range(DECK_MIN_INDEX, DECK_MAX_INDEX + 1):
        row = data[i] if i < len(data) else _empty_deck_row(i)
        cards = row.get("Card", [])
        cards = list(cards[:DECK_SLOT_COUNT]) + [None] * max(0, DECK_SLOT_COUNT - len(cards))
        rows.append(
            {
                "__RowId": row.get("__RowId", f"MiniGame_PlayerDeck_{i:02d}"),
                "Name": row.get("Name", _default_deck_name(i)),
                "CardSleeve": row.get("CardSleeve", DEFAULT_CARD_SLEEVE),
                "Card": cards,
            }
        )
    return rows


def init_player_deck_file(force: bool = False) -> Path:
    if PLAYER_DECK_JSON.exists() and not force:
        return PLAYER_DECK_JSON
    PLAYER_DECK_JSON.parent.mkdir(parents=True, exist_ok=True)
    data = [_empty_deck_row(i) for i in range(DECK_MIN_INDEX, DECK_MAX_INDEX + 1)]
    PLAYER_DECK_JSON.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    return PLAYER_DECK_JSON


def load_player_decks() -> List[dict]:
    if not PLAYER_DECK_JSON.exists():
        init_player_deck_file()
    data = json.loads(PLAYER_DECK_JSON.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError("PlayerPresetDeck.json format invalid: root must be list")
    return _normalize_player_decks(data)


def save_player_decks(decks: List[dict]) -> None:
    rows = _normalize_player_decks(decks)
    _validate_all_player_decks(rows, require_full_15=False)
    PLAYER_DECK_JSON.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")


def _card_indexes() -> Tuple[Dict[int, dict], Dict[str, dict], Dict[str, dict]]:
    cards = load_card_info()
    by_id: Dict[int, dict] = {c["Number"]: c for c in cards}
    by_name: Dict[str, dict] = {str(c["Name"]).lower(): c for c in cards}

    zh_map = mini_game_card_name_map()
    by_name_zh: Dict[str, dict] = {}
    for c in cards:
        zh = zh_map.get(c.get("Name", ""))
        if zh:
            by_name_zh[str(zh).lower()] = c
    return by_id, by_name, by_name_zh


def _load_available_sleeves() -> Dict[str, str]:
    """
    返回可用卡套映射：
    key: lower-case alias
    value: gyml path
    """
    project_root = PROJECT_ROOT
    img_dir = project_root / "data" / "images" / "minigame" / "sleeve"
    result: Dict[str, str] = {}
    if img_dir.exists():
        for png in img_dir.glob("MngCardSleeve_*.png"):
            base = png.stem  # MngCardSleeve_Default
            suffix = base.replace("MngCardSleeve_", "", 1)  # Default
            gyml_name = f"MiniGameCardSleeve_{suffix}"
            gyml_path = f"Work/Gyml/{gyml_name}.spl__MiniGameCardSleeve.gyml"
            result[suffix.lower()] = gyml_path
            result[gyml_name.lower()] = gyml_path
            result[gyml_path.lower()] = gyml_path

    # 兜底确保 Default 存在
    result.setdefault("default", DEFAULT_CARD_SLEEVE)
    result.setdefault(DEFAULT_CARD_SLEEVE.lower(), DEFAULT_CARD_SLEEVE)
    return result


def resolve_sleeve_to_path(sleeve_query: str) -> str:
    q = str(sleeve_query).strip().lower()
    if not q:
        raise ValueError("sleeve query is empty")
    sleeves = _load_available_sleeves()
    if q not in sleeves:
        raise KeyError(f"Sleeve not found: {sleeve_query}")
    return sleeves[q]


def resolve_card_to_path(card_query: str) -> Tuple[str, dict]:
    """
    Resolve card query (id / english name / chinese name) to gyml path.
    """
    by_id, by_name, by_name_zh = _card_indexes()
    q = card_query.strip()
    if not q:
        raise ValueError("card_query is empty")

    card_info = None
    if q.isdigit():
        card_info = by_id.get(int(q))
    if card_info is None:
        card_info = by_name.get(q.lower())
    if card_info is None:
        card_info = by_name_zh.get(q.lower())
    if card_info is None:
        raise KeyError(f"Card not found by id/name: {card_query}")

    row_id = card_info["__RowId"]
    path = f"Work/Gyml/{row_id}.spl__MiniGameCardInfo.gyml"
    return path, card_info


def _validate_deck_index(deck_index: int) -> None:
    if not (DECK_MIN_INDEX <= deck_index <= DECK_MAX_INDEX):
        raise IndexError(f"deck_index out of range: {deck_index}, expected {DECK_MIN_INDEX}~{DECK_MAX_INDEX}")


def _validate_slot(slot: int) -> None:
    if not (1 <= slot <= DECK_SLOT_COUNT):
        raise IndexError(f"slot out of range: {slot}, expected 1~{DECK_SLOT_COUNT}")


def _validate_all_player_decks(decks: List[dict], require_full_15: bool) -> None:
    for i in range(DECK_MIN_INDEX, DECK_MAX_INDEX + 1):
        result = check_player_deck(i, require_full_15=require_full_15, decks=decks)
        if not result["ok"]:
            raise ValueError(
                f"invalid player deck[{i}]: {result['errors']}"
            )


def _validate_deck_or_raise(deck_index: int, require_full_15: bool = False, decks: Optional[List[dict]] = None) -> None:
    result = check_player_deck(deck_index, require_full_15=require_full_15, decks=decks)
    if not result["ok"]:
        raise ValueError(f"player deck[{deck_index}] validation failed: {result['errors']}")


def get_player_deck(deck_index: int) -> dict:
    _validate_deck_index(deck_index)
    decks = load_player_decks()
    _validate_deck_or_raise(deck_index, require_full_15=False, decks=decks)
    return decks[deck_index]


def get_player_deck_slot(deck_index: int, slot: int):
    _validate_deck_index(deck_index)
    _validate_slot(slot)
    return get_player_deck(deck_index)["Card"][slot - 1]


def set_player_deck_slot(deck_index: int, slot: int, card_query: str) -> dict:
    _validate_deck_index(deck_index)
    _validate_slot(slot)
    decks = load_player_decks()
    card_path, card_info = resolve_card_to_path(card_query)
    decks[deck_index]["Card"][slot - 1] = card_path
    _validate_deck_or_raise(deck_index, require_full_15=False, decks=decks)
    save_player_decks(decks)
    return card_info


def clear_player_deck_slot(deck_index: int, slot: int) -> None:
    _validate_deck_index(deck_index)
    _validate_slot(slot)
    decks = load_player_decks()
    decks[deck_index]["Card"][slot - 1] = None
    _validate_deck_or_raise(deck_index, require_full_15=False, decks=decks)
    save_player_decks(decks)


def list_player_deck_slots(deck_index: int) -> List[Tuple[int, Optional[str]]]:
    deck = get_player_deck(deck_index)
    return [(i + 1, deck["Card"][i]) for i in range(DECK_SLOT_COUNT)]


def get_player_deck_card_numbers(deck_index: int, require_full_15: bool = True) -> List[int]:
    """
    返回玩家卡组的卡牌编号列表（按槽位顺序）。
    """
    result = check_player_deck(deck_index, require_full_15=require_full_15)
    if not result["ok"]:
        raise ValueError(f"player deck[{deck_index}] invalid: {result['errors']}")
    deck = get_player_deck(deck_index)
    by_id, _by_name, _by_name_zh = _card_indexes()
    by_rowid = {c["__RowId"]: c for c in by_id.values()}
    numbers: List[int] = []
    for raw in deck["Card"]:
        if raw is None:
            continue
        row_id = raw.split("/")[-1].split(".spl__")[0]
        info = by_rowid[row_id]
        numbers.append(int(info["Number"]))
    return numbers


def set_player_deck_name(deck_index: int, name: str) -> None:
    _validate_deck_index(deck_index)
    normalized = str(name).strip()
    if not normalized:
        raise ValueError("deck name cannot be empty")
    decks = load_player_decks()
    decks[deck_index]["Name"] = normalized
    _validate_deck_or_raise(deck_index, require_full_15=False, decks=decks)
    save_player_decks(decks)


def set_player_deck_sleeve(deck_index: int, sleeve_query: str) -> str:
    _validate_deck_index(deck_index)
    decks = load_player_decks()
    sleeve_path = resolve_sleeve_to_path(sleeve_query)
    decks[deck_index]["CardSleeve"] = sleeve_path
    _validate_deck_or_raise(deck_index, require_full_15=False, decks=decks)
    save_player_decks(decks)
    return sleeve_path


def get_player_deck_sleeve(deck_index: int) -> str:
    deck = get_player_deck(deck_index)
    return str(deck.get("CardSleeve", DEFAULT_CARD_SLEEVE))


def get_player_deck_name(deck_index: int) -> str:
    deck = get_player_deck(deck_index)
    return str(deck.get("Name", _default_deck_name(deck_index)))


def check_player_deck(deck_index: int, require_full_15: bool = False, decks: Optional[List[dict]] = None) -> dict:
    """
    检查玩家卡组有效性。

    Args:
        deck_index: 玩家卡组索引（0~32）
        require_full_15: 是否要求15个槽位都非空

    Returns:
        {
          "ok": bool,
          "deck_index": int,
          "filled_count": int,
          "empty_slots": List[int],
          "invalid_slots": List[{"slot": int, "value": str}],
          "duplicate_cards": List[{"number": int, "name": str, "slots": List[int]}],
          "errors": List[str],
        }
    """
    _validate_deck_index(deck_index)
    all_decks = _normalize_player_decks(decks) if decks is not None else load_player_decks()
    deck = all_decks[deck_index]
    slots = deck["Card"]
    sleeve = str(deck.get("CardSleeve", DEFAULT_CARD_SLEEVE))

    by_id, _by_name, _by_name_zh = _card_indexes()
    by_rowid = {c["__RowId"]: c for c in by_id.values()}

    empty_slots: List[int] = []
    invalid_slots: List[dict] = []
    card_slot_map: Dict[int, List[int]] = {}

    for i, raw in enumerate(slots, start=1):
        if raw is None:
            empty_slots.append(i)
            continue
        if not isinstance(raw, str) or ".spl__MiniGameCardInfo.gyml" not in raw:
            invalid_slots.append({"slot": i, "value": str(raw)})
            continue
        row_id = raw.split("/")[-1].split(".spl__")[0]
        info = by_rowid.get(row_id)
        if info is None:
            invalid_slots.append({"slot": i, "value": raw})
            continue
        number = int(info["Number"])
        card_slot_map.setdefault(number, []).append(i)

    duplicates: List[dict] = []
    for number, pos_list in sorted(card_slot_map.items()):
        if len(pos_list) > 1:
            card = by_id[number]
            duplicates.append(
                {
                    "number": number,
                    "name": card.get("Name", ""),
                    "slots": pos_list,
                }
            )

    errors: List[str] = []
    # check sleeve path
    try:
        resolve_sleeve_to_path(sleeve)
    except Exception:
        errors.append("INVALID_CARD_SLEEVE")

    if require_full_15 and empty_slots:
        errors.append(f"EMPTY_SLOTS:{','.join(map(str, empty_slots))}")
    if invalid_slots:
        errors.append(f"INVALID_SLOTS:{','.join(str(x['slot']) for x in invalid_slots)}")
    if duplicates:
        dup_slots = [str(s) for d in duplicates for s in d["slots"]]
        errors.append(f"DUPLICATE_CARDS_AT_SLOTS:{','.join(dup_slots)}")

    filled_count = DECK_SLOT_COUNT - len(empty_slots)
    ok = len(errors) == 0
    return {
        "ok": ok,
        "deck_index": deck_index,
        "deck_name": deck.get("Name", _default_deck_name(deck_index)),
        "card_sleeve": sleeve,
        "filled_count": filled_count,
        "empty_slots": empty_slots,
        "invalid_slots": invalid_slots,
        "duplicate_cards": duplicates,
        "errors": errors,
    }
