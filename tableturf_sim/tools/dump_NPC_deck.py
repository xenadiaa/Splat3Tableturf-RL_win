import json
from pathlib import Path

ROOT = Path("data")
MUSH_DIR = ROOT / "data" / "1010"
CARD_INFO = ROOT / "cards" / "MiniGameCardInfo.json"
PRESET_DECK = ROOT / "cards" / "MiniGamePresetDeck.json"
NPC_FILE = ROOT / "MiniGameGameNpcData.json"       # <- 改成你的npc表文件名
LANG_FILE = ROOT / "CNzh_full_unicode.json"     # <- 改成你的中文表文件名

def load_json(p: Path):
    return json.loads(p.read_text(encoding="utf-8"))

def square_to_cells(square_64):
    cells, sp = [], []
    for i, v in enumerate(square_64):
        if v == "Empty":
            continue
        x, y = i % 8, i // 8
        cells.append((x, y))
        if v == "Special":
            sp.append((x, y))
    # normalize to min x/y = 0
    minx = min(x for x, _ in cells)
    miny = min(y for _, y in cells)
    cells = sorted([(x-minx, y-miny) for x, y in cells])
    sp = sorted([(x-minx, y-miny) for x, y in sp])
    return cells, sp

def gyml_to_key(path: str) -> str:
    # "Work/Gyml/MiniGame_Atarime.spl__MiniGamePresetDeck.gyml" -> "MiniGame_Atarime"
    name = path.split("/")[-1]
    return name.split(".spl__")[0]

def load_lang_cn(lang_obj):
    """
    你说的 CNzh_unicode 具体结构我没看到，这里先做一个常见的映射方式：
    - 如果是 { "3134251294": "英雄射击枪" } 这种，则按 hash 做 key
    - 如果是更复杂结构，你把示例贴出来我就能精确对齐
    """
    cn_by_hash = {}
    if isinstance(lang_obj, dict):
        # 尝试：key 是字符串数字
        for k, v in lang_obj.items():
            if str(k).isdigit():
                cn_by_hash[int(k)] = v
    return cn_by_hash

def main():
    decks = load_json(PRESET_DECK)
    npcs = load_json(NPC_FILE)
    lang = load_lang_cn(load_json(LANG_FILE))

    decks_by_id = {d["__RowId"]: d for d in decks}
    npcs_by_id = {n["__RowId"]: n for n in npcs}

    npc_id = "MiniGame_Atarime"
    npc = npcs_by_id[npc_id]

    print("NPC:", npc["Name"], "RowId:", npc_id)
    for idx, level in enumerate(npc["AILevel"]):
        ai_type = npc["AIType"][idx]
        map_name = npc["Map"][idx]
        deck_key = gyml_to_key(npc["Deck"][idx])

        print(f"\n== Level {level} | AIType={ai_type} | Map={map_name} | Deck={deck_key} ==")

        deck = decks_by_id[deck_key]
        card_paths = deck["Card"]

        for p in card_paths:
            card_key = gyml_to_key(p)  # MiniGame_HeroShooter...
            card_json_path = MUSH_DIR / f"{card_key}.json"
            card = load_json(card_json_path)

            cells, sp = square_to_cells(card["Square"])
            cost = card["SpecialCost"]
            num = card["Number"]
            name = card["Name"]
            name_cn = lang.get(card.get("NameHash", -1), "")

            print(f"- #{num:03d} {name} {('｜'+name_cn) if name_cn else ''} | cost={cost} | cells={len(cells)} | sp={len(sp)}")

if __name__ == "__main__":
    main()
