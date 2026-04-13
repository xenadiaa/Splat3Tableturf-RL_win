import json
import os
import math
from typing import List, Tuple, Optional
from pathlib import Path

# 导入类型定义
from ..assets.tableturf_types import (
    Card_Single,
    CardDeck,
    Card_Rotation,
    GameMap,
    Map_PointBit,
    Map_PointMask,
)


# 全局缓存：加载一次 JSON 数据
_card_data_cache: Optional[List[dict]] = None
_card_dict_by_number: Optional[dict] = None


def _load_card_data() -> List[dict]:
    """加载卡牌 JSON 数据（带缓存）。"""
    global _card_data_cache
    if _card_data_cache is None:
        # 获取项目根目录（假设 common_utils.py 在 tableturf_sim/src/utils/）
        current_file = Path(__file__)
        project_root = current_file.parent.parent.parent
        json_path = project_root / "data" / "cards" / "MiniGameCardInfo.json"
        
        if not json_path.exists():
            raise FileNotFoundError(f"卡牌数据文件不存在: {json_path}")
        
        with open(json_path, 'r', encoding='utf-8') as f:
            _card_data_cache = json.load(f)
    
    return _card_data_cache


def _get_card_dict_by_number() -> dict:
    """获取按 Number 索引的卡牌字典（带缓存）。"""
    global _card_dict_by_number
    if _card_dict_by_number is None:
        cards = _load_card_data()
        _card_dict_by_number = {card["Number"]: card for card in cards}
    return _card_dict_by_number


def _square_str_to_int(s: str) -> int:
    """将 Square 字符串转换为整数编码：Empty=0, Fill=1, Special=2"""
    if s == "Empty":
        return 0
    elif s == "Fill":
        return 1
    elif s == "Special":
        return 2
    else:
        raise ValueError(f"未知的 Square 值: {s}")


def _square_list_to_matrix(square_list: List[str]) -> List[List[int]]:
    """将 64 元素的 Square 列表转换为 8x8 矩阵。"""
    if len(square_list) != 64:
        raise ValueError(f"Square 列表长度必须为 64，实际为 {len(square_list)}")
    
    matrix = []
    for row in range(8):
        # 输入顺序按“左下 -> 右，再逐行向上”，因此需要翻转 y 轴
        source_row = 7 - row
        matrix.append([
            _square_str_to_int(square_list[source_row * 8 + col])
            for col in range(8)
        ])
    return matrix


def _rotate_matrix(matrix: List[List[int]], rotation: int) -> List[List[int]]:
    """旋转 8x8 矩阵。rotation: 0=0°, 1=90°CW, 2=180°, 3=270°CW"""
    if rotation == 0:
        return [row[:] for row in matrix]
    
    rotated = [[0] * 8 for _ in range(8)]
    for r in range(8):
        for c in range(8):
            new_r, new_c = rotate_cell(r, c, rotation)
            rotated[new_r][new_c] = matrix[r][c]
    
    return rotated


def _find_first_non_empty(matrix: List[List[int]]) -> Optional[Tuple[int, int]]:
    """找到矩阵中第一个非 Empty（非0）的坐标 (x, y)。"""
    for y in range(8):
        for x in range(8):
            if matrix[y][x] != 0:
                return (x, y)
    return None


def _calculate_edge(matrix: List[List[int]]) -> Tuple[int, int]:
    """计算边界框的最小坐标 (min_x, min_y)。"""
    min_x, min_y = 8, 8
    for y in range(8):
        for x in range(8):
            if matrix[y][x] != 0:
                min_x = min(min_x, x)
                min_y = min(min_y, y)
    
    # 如果没有非空点，返回 (0, 0)
    if min_x == 8:
        return (0, 0)
    return (min_x, min_y)


def _calculate_bounds(matrix: List[List[int]]) -> Tuple[int, int, int, int]:
    """计算边界框 (left, top, right, bottom)。"""
    min_x, min_y = 8, 8
    max_x, max_y = -1, -1
    for y in range(8):
        for x in range(8):
            if matrix[y][x] != 0:
                min_x = min(min_x, x)
                min_y = min(min_y, y)
                max_x = max(max_x, x)
                max_y = max(max_y, y)

    if max_x == -1:
        return (0, 0, 0, 0)
    return (min_x, min_y, max_x, max_y)


def _rotate_link_pos_cw(link_pos: Tuple[int, int], steps: int) -> Tuple[int, int]:
    """将 (x, y) 在 8x8 坐标系中按 90° CW 旋转 steps 次。"""
    x, y = link_pos
    for _ in range(steps % 4):
        x, y = 7 - y, x
    return (x, y)


def _rarity_str_to_int(rarity_str: str) -> int:
    """将稀有度字符串转换为整数：Common=0, Rare=1, Fresh=2"""
    rarity_map = {
        "Common": 0,
        "Rare": 1,
        "Fresh": 2,
    }
    return rarity_map.get(rarity_str, 0)


def create_card_from_id(card_id: int) -> Card_Single:
    """
    根据卡牌编号（Number）从 JSON 数据创建 Card_Single 对象。
    
    Args:
        card_id: 卡牌编号（Number 字段）
    
    Returns:
        Card_Single 对象
    
    Raises:
        KeyError: 如果找不到指定编号的卡牌
    """
    card_dict = _get_card_dict_by_number()
    
    if card_id not in card_dict:
        raise KeyError(f"找不到编号为 {card_id} 的卡牌")
    
    card_data = card_dict[card_id]
    
    # 转换 Square 列表为 8x8 矩阵
    square_2d_0 = _square_list_to_matrix(card_data["Square"])
    
    # 计算 4 个旋转的矩阵
    square_2d_90 = _rotate_matrix(square_2d_0, 1)
    square_2d_180 = _rotate_matrix(square_2d_0, 2)
    square_2d_270 = _rotate_matrix(square_2d_0, 3)
    
    # 0度下 link_pos 按边界中心点计算：
    # x(左右)仅在有小数时向下取整；y(上下)仅在有小数时向上取整
    left, top, right, bottom = _calculate_bounds(square_2d_0)
    center_x = (left + right) / 2.0
    center_y = (top + bottom) / 2.0
    link_pos_0 = (math.floor(center_x), math.ceil(center_y))
    
    width = right - left + 1
    height = bottom - top + 1
    if width % 2 == 0 and height % 2 == 0:
        # 双偶数边长：不做旋转映射，四个角度中心点一致
        link_pos_90 = link_pos_0
        link_pos_180 = link_pos_0
        link_pos_270 = link_pos_0
    else:
        # 存在奇数边长：从 0 度中心点按旋转映射得到其余角度
        link_pos_90 = _rotate_link_pos_cw(link_pos_0, 1)
        link_pos_180 = _rotate_link_pos_cw(link_pos_0, 2)
        link_pos_270 = _rotate_link_pos_cw(link_pos_0, 3)
    
    # 计算边界（min_x, min_y）
    edge_0 = _calculate_edge(square_2d_0)
    edge_90 = _calculate_edge(square_2d_90)
    edge_180 = _calculate_edge(square_2d_180)
    edge_270 = _calculate_edge(square_2d_270)
    
    # 计算点数（非 Empty 格子数）
    card_point = sum(1 for row in square_2d_0 for cell in row if cell != 0)
    
    # 转换 Rarity 和 NameHash
    rarity_int = _rarity_str_to_int(card_data.get("Rarity", "Common"))
    name_hash_str = str(card_data.get("NameHash", ""))
    
    return Card_Single(
        name=card_data.get("Name", ""),
        NameHash=name_hash_str,
        Number=card_data.get("Number", 0),
        Rarity=rarity_int,
        CardPoint=card_point,
        SpecialCost=card_data.get("SpecialCost", 0),
        square_2d_0=square_2d_0,
        square_2d_90=square_2d_90,
        square_2d_180=square_2d_180,
        square_2d_270=square_2d_270,
        link_pos_0=link_pos_0,
        link_pos_90=link_pos_90,
        link_pos_180=link_pos_180,
        link_pos_270=link_pos_270,
        edge_0=edge_0,
        edge_90=edge_90,
        edge_180=edge_180,
        edge_270=edge_270,
        SquareSortOffset=card_data.get("SquareSortOffset", 0),
        RowId=card_data.get("__RowId"),
    )


def create_deck_from_ids(card_ids: List[int]) -> CardDeck:
    """
    根据 15 个卡牌编号创建 CardDeck 对象。
    
    Args:
        card_ids: 15 个卡牌编号的列表
    
    Returns:
        CardDeck 对象
    
    Raises:
        ValueError: 如果 card_ids 长度不是 15
        KeyError: 如果某个卡牌编号不存在
    """
    if len(card_ids) != 15:
        raise ValueError(f"CardDeck 必须包含 15 张卡牌，实际提供了 {len(card_ids)} 张")
    
    cards = [create_card_from_id(card_id) for card_id in card_ids]
    return CardDeck(cards=cards)


def get_collision_mask(m1, x1, y1, m2, x2, y2):
    dx = x2 - x1
    dy = y2 - y1

    # 如果偏移超过 8 格，物理上不可能重叠
    if abs(dx) >= 8 or abs(dy) >= 8:
        return 0
    
    # 将 M2 对齐到 M1 的坐标系
    # 假设 M2 的第 0 位是 (0,0)，第 63 位是 (7,7)
    # 行偏移 dy 相当于移动 8 * dy 位，列偏移 dx 相当于移动 dx 位
    shift_amount = dy * 8 + dx
    
    if shift_amount > 0:
        shifted_m2 = m2 << shift_amount
    else:
        shifted_m2 = m2 >> abs(shift_amount)

    # 关键：位移后可能会有"环绕"污染（比如第1行移到了第2行）
    # 需要根据 dx 使用掩码清空无效位
    col_mask = 0xFFFFFFFFFFFFFFFF
    if dx > 0:
        # 左移，清空左侧 dx 列
        for i in range(dx): col_mask &= ~0x0101010101010101 << i
    elif dx < 0:
        # 右移，清空右侧 dx 列
        for i in range(abs(dx)): col_mask &= ~0x8080808080808080 >> i
        
    shifted_m2 &= col_mask
    
    # 最终冲突位（在 M1 坐标系下）
    return m1 & shifted_m2

#比较单格归属
def compare_single_box_final_result(player, enemy):
    # 假设 Epision 是一个预定义的极小值常数
    epsilon = 0.000001 
    
    # 计算优先级
    player_priority = 30.0 + 0.5*player['is_using_sp'] - player['card_count'] + 30*player['is_sp_box']
    enemy_priority = 30.0 + 0.5*enemy['is_using_sp'] - enemy['card_count'] + 30*player['is_sp_box']
    
    # 逻辑判断
    if abs(player_priority - enemy_priority) < epsilon:
        return "BT_Conflict"
    elif player_priority > enemy_priority:
        return player['box_type']
    else:
        return enemy['box_type']

def _clear_preview(mask: int) -> int:
    """清除预览位，便于做真实格子状态判断。"""
    return int(mask) & ~int(Map_PointBit.IsPreview)


def _is_valid_cell(mask: int) -> bool:
    return (_clear_preview(mask) & int(Map_PointBit.IsValid)) != 0


def _is_empty_cell(mask: int) -> bool:
    return _clear_preview(mask) == int(Map_PointMask.Empty)


def _is_conflict_cell(mask: int) -> bool:
    m = _clear_preview(mask)
    return (m & int(Map_PointBit.IsP1)) != 0 and (m & int(Map_PointBit.IsP2)) != 0


def _is_special_cell(mask: int) -> bool:
    return (_clear_preview(mask) & int(Map_PointBit.IsSp)) != 0


def _is_owner_cell(mask: int, is_p1: bool) -> bool:
    m = _clear_preview(mask)
    owner_bit = int(Map_PointBit.IsP1) if is_p1 else int(Map_PointBit.IsP2)
    return (m & owner_bit) != 0


def _is_owner_normal_cell(mask: int, is_p1: bool) -> bool:
    m = _clear_preview(mask)
    return _is_owner_cell(m, is_p1) and not _is_special_cell(m) and not _is_conflict_cell(m)


def _is_owner_special_cell(mask: int, is_p1: bool) -> bool:
    m = _clear_preview(mask)
    return _is_owner_cell(m, is_p1) and _is_special_cell(m)


def _neighbor_coords(x: int, y: int):
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            if dx == 0 and dy == 0:
                continue
            yield (x + dx, y + dy)


def _card_cells_on_map(
    card: Card_Single,
    anchor_x: int,
    anchor_y: int,
    rotation: int,
) -> List[Tuple[int, int, int]]:
    """
    返回卡牌有效格投影到地图后的坐标：
    (map_x, map_y, cell_type[1=Fill,2=Special])
    """
    matrix = card.get_square_matrix(rotation)
    link_x, link_y = card.get_link_pos(rotation)
    cells: List[Tuple[int, int, int]] = []
    for cy in range(8):
        for cx in range(8):
            cell = matrix[cy][cx]
            if cell == 0:
                continue
            mx = anchor_x + (cx - link_x)
            my = anchor_y + (cy - link_y)
            cells.append((mx, my, cell))
    return cells


def _is_adjacent_to_self(
    game_map: GameMap,
    x: int,
    y: int,
    is_p1: bool,
    need_special_only: bool,
) -> bool:
    for nx, ny in _neighbor_coords(x, y):
        if not (0 <= nx < game_map.width and 0 <= ny < game_map.height):
            continue
        neighbor = game_map.get_point(nx, ny)
        if need_special_only:
            if _is_owner_special_cell(neighbor, is_p1):
                return True
        else:
            if _is_owner_special_cell(neighbor, is_p1) or _is_owner_normal_cell(neighbor, is_p1):
                return True
    return False


def validate_place_card_action(
    card: Card_Single,
    game_map: GameMap,
    anchor_x: int,
    anchor_y: int,
    rotation: int,
    is_p1: bool,
    use_sp_attack: bool = False,
) -> Tuple[bool, str, List[Tuple[int, int, int]]]:
    """
    基础放置合法性检查。

    Returns:
        (is_valid, reason, card_cells_on_map)
    """
    cells = _card_cells_on_map(card, anchor_x, anchor_y, rotation)
    if not cells:
        return False, "CARD_HAS_NO_VALID_CELLS", cells

    has_required_neighbor = False
    for x, y, _cell_type in cells:
        if not (0 <= x < game_map.width and 0 <= y < game_map.height):
            return False, "OUT_OF_MAP", cells

        target = game_map.get_point(x, y)
        if not _is_valid_cell(target):
            return False, "TARGET_NOT_VALID_MAP_CELL", cells

        if use_sp_attack:
            # SP攻击：可落在 Empty / 自己Fill / 敌方Fill；不可落在任何 Special、Conflict
            if _is_special_cell(target) or _is_conflict_cell(target):
                return False, "SP_ATTACK_FORBID_ON_SPECIAL_OR_CONFLICT", cells
            if not (_is_empty_cell(target) or _is_owner_normal_cell(target, True) or _is_owner_normal_cell(target, False)):
                return False, "SP_ATTACK_TARGET_NOT_ALLOWED", cells
            if _is_adjacent_to_self(game_map, x, y, is_p1, need_special_only=True):
                has_required_neighbor = True
        else:
            # 普通放置：所有有效格必须落在 Empty
            if not _is_empty_cell(target):
                return False, "NORMAL_PLACE_REQUIRES_EMPTY", cells
            if _is_adjacent_to_self(game_map, x, y, is_p1, need_special_only=False):
                has_required_neighbor = True

    if not has_required_neighbor:
        if use_sp_attack:
            return False, "NEED_ADJACENT_SELF_SPECIAL", cells
        return False, "NEED_ADJACENT_SELF_FILL_OR_SPECIAL", cells
    return True, "OK", cells


def activate_special_points_and_gain_sp(game_map: GameMap, is_p1: bool) -> int:
    """
    每回合结束调用一次：
    - 对己方 Special 点，若周围8格没有 Empty，则激活并 +1 SP。
    - 激活后会打 IsSupplySp 位，避免重复获得。
    """
    gained = 0
    owner_bit = int(Map_PointBit.IsP1) if is_p1 else int(Map_PointBit.IsP2)
    for y in range(game_map.height):
        for x in range(game_map.width):
            mask = _clear_preview(game_map.get_point(x, y))
            if (mask & owner_bit) == 0:
                continue
            if (mask & int(Map_PointBit.IsSp)) == 0:
                continue
            if (mask & int(Map_PointBit.IsSupplySp)) != 0:
                continue

            has_empty_neighbor = False
            for nx, ny in _neighbor_coords(x, y):
                if not (0 <= nx < game_map.width and 0 <= ny < game_map.height):
                    continue
                neighbor = game_map.get_point(nx, ny)
                if _is_empty_cell(neighbor):
                    has_empty_neighbor = True
                    break

            if not has_empty_neighbor:
                game_map.set_point(x, y, int(mask) | int(Map_PointBit.IsSupplySp))
                gained += 1
    return gained


# ---- 兼容旧占位函数命名 ----
def have_neighbor_with_card(card, pos, game_map, is_p1=True, rotation=0):
    cells = _card_cells_on_map(card, pos[0], pos[1], rotation)
    return any(_is_adjacent_to_self(game_map, x, y, is_p1, need_special_only=False) for x, y, _ in cells)


def have_neighbor_with_special(card, pos, game_map, is_p1=True, rotation=0):
    cells = _card_cells_on_map(card, pos[0], pos[1], rotation)
    return any(_is_adjacent_to_self(game_map, x, y, is_p1, need_special_only=True) for x, y, _ in cells)


def is_able_to_place_card(card, game_map, is_p1=True, rotation=0):
    for y in range(game_map.height):
        for x in range(game_map.width):
            ok, _, _ = validate_place_card_action(card, game_map, x, y, rotation, is_p1, use_sp_attack=False)
            if ok:
                return True
    return False


def is_able_to_place_withsp(card, game_map, is_p1=True, rotation=0):
    for y in range(game_map.height):
        for x in range(game_map.width):
            ok, _, _ = validate_place_card_action(card, game_map, x, y, rotation, is_p1, use_sp_attack=True)
            if ok:
                return True
    return False

def rotate_cell(r: int, c: int, rotation: int) -> tuple[int, int]:
    """(r,c) 旋转后所在的新坐标。rotation: 0,1,2,3 → 0°,90°,180°,270° CW"""
    if rotation == 0:
        return (r, c)
    if rotation == 1:   # 90° CW
        return (c, 7 - r)
    if rotation == 2:   # 180°
        return (7 - r, 7 - c)
    if rotation == 3:   # 270° CW
        return (7 - c, r)
    raise ValueError("rotation must be 0-3")

    return newrc

#锚点选择，json中第一个非empty点
