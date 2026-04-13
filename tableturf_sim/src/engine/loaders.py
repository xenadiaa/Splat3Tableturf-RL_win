"""加载卡牌、地图等资源的工具函数。"""

import json
from pathlib import Path
from typing import Optional

from ..assets.tableturf_types import Card_Single, GameMap
from ..utils.common_utils import create_card_from_id

# 全局缓存
_map_data_cache: Optional[list] = None
_map_dict_by_id: Optional[dict] = None
_card_data_cache: Optional[list] = None
_cards_cache: Optional[list[Card_Single]] = None
MAP_PADDING = 4


def _load_map_info() -> list:
    """加载地图 JSON 数据（带缓存）。"""
    global _map_data_cache
    if _map_data_cache is None:
        current_file = Path(__file__)
        project_root = current_file.parent.parent.parent
        json_path = project_root / "data" / "maps" / "MiniGameMapInfo.json"

        if not json_path.exists():
            raise FileNotFoundError(f"地图数据文件不存在: {json_path}")

        with open(json_path, "r", encoding="utf-8") as f:
            _map_data_cache = json.load(f)

    return _map_data_cache


def _get_map_dict_by_id() -> dict:
    """获取按 id 索引的地图字典（带缓存）。"""
    global _map_dict_by_id
    if _map_dict_by_id is None:
        maps_data = _load_map_info()
        _map_dict_by_id = {m["id"]: m for m in maps_data}
    return _map_dict_by_id


def load_map(map_id: str) -> GameMap:
    """
    根据地图 ID 从 MiniGameMapInfo.json 加载地图，返回 GameMap 对象。

    Args:
        map_id: 地图 id（如 "WDiamond", "Square", "ManySp" 等）

    Returns:
        GameMap 对象

    Raises:
        KeyError: 找不到指定 id 的地图
    """
    map_dict = _get_map_dict_by_id()

    if map_id not in map_dict:
        raise KeyError(f"找不到 id 为 {map_id!r} 的地图")

    data = map_dict[map_id]
    # 深拷贝 grid，避免外部修改影响缓存
    base_grid = [row[:] for row in data["point_type"]]

    # 在原地图四周补 4 格不可达区域（NotMap=0）作为容错边界。
    # 这样卡牌旋转/锚点在边缘附近时可以临时超出原图范围，但不会落到有效地图格之外。
    base_h = len(base_grid)
    base_w = len(base_grid[0]) if base_h > 0 else 0
    pad = MAP_PADDING
    out_h = base_h + pad * 2
    out_w = base_w + pad * 2
    grid = [[0 for _ in range(out_w)] for _ in range(out_h)]
    for y in range(base_h):
        for x in range(base_w):
            grid[y + pad][x + pad] = base_grid[y][x]

    return GameMap(
        map_id=data["id"],
        name=data["name"],
        ename=data["ename"],
        width=out_w,
        height=out_h,
        grid=grid,
    )


def load_cards():
    """
    从 MiniGameCardInfo.json 加载全部卡牌，并转换为 Card_Single 列表。

    Returns:
        List[Card_Single]，按 Number 升序。
    """
    global _card_data_cache, _cards_cache
    if _cards_cache is not None:
        return _cards_cache

    if _card_data_cache is None:
        current_file = Path(__file__)
        project_root = current_file.parent.parent.parent
        json_path = project_root / "data" / "cards" / "MiniGameCardInfo.json"
        if not json_path.exists():
            raise FileNotFoundError(f"卡牌数据文件不存在: {json_path}")
        with open(json_path, "r", encoding="utf-8") as f:
            _card_data_cache = json.load(f)

    numbers = sorted(card["Number"] for card in _card_data_cache)
    _cards_cache = [create_card_from_id(num) for num in numbers]
    return _cards_cache
