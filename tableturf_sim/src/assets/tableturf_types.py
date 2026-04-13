
from enum import IntFlag, auto
from dataclasses import dataclass, field, asdict
from typing import List, Tuple, Optional, Any, Dict

"""
位表示定义说明

本文件采用比特位编码方案来精确表达棋盘每个格子的归属、状态以及显示效果，便于高效判断与操作。

bit 定义如下（自低至高）：
bit 0：IsValid（是否有效，1=有效棋盘格，0=无效或地图外）
bit 1：IsP1（1=属于P1/己方）
bit 2：IsP2（1=属于P2/对方）
bit 3：IsSp（1=特殊点/SP点）
bit 4：IsSupplySp（1=已激活特殊点，后续不可再次获取SP）
bit 5：IsPreview（1=为预览/虚影，用于落子预览，放置后归零）

组合含义举例（十进制）：
0b000000 = 0    NotInMap      （无效格，非地图区域）
0b000001 = 1    Empty         （空位，可放置）
0b000011 = 3    NotEmpty-Fill P1      （P1/己方普通点，已放置）
0b000101 = 5    NotEmpty-Fill P2      （P2/对方普通点，已放置）
0b001011 = 11   NotEmpty-Special P1   （P1/己方SP点，已放置，未激活）
0b001101 = 13   NotEmpty-Special P2   （P2/对方SP点，已放置，未激活）
0b000111 = 7    Conflict              （冲突点，普通冲突，灰色）
0b001111 = 15   Conflict              （冲突点，特殊点冲突，灰色）
0b011011 = 27   NotEmpty-Special-Active P1   （P1/己方SP点，已激活）
0b011101 = 29   NotEmpty-Special-Active P2   （P2/对方SP点，已激活）

显示用（预览/虚影）：
0b100011 = 35   NotEmpty-Fill P1 Preview
0b100101 = 37   NotEmpty-Fill P2 Preview
0b101011 = 43   NotEmpty-Special P1 Preview
0b101101 = 45   NotEmpty-Special P2 Preview

位操作逻辑：
- 归属判定、覆盖判定可通过比特位与（&）、或（|）运算实现。
- 判断是否为己方：val & 0b000010 > 0
- 判断是否为SP点：val & 0b001000 > 0
- 判断是否已激活SP：val & 0b010000 > 0
- 判断是否为预览：val & 0b100000 > 0
- 判断是否为可用格：val & 0b000001 > 0

注：
- 本文件仅为位编码说明。具体常量及函数可按上述定义实现。
- 利用比特位高效比较，大大简化归属、冲突与显示等复杂逻辑。
"""

@dataclass
class Card_Rotation(IntFlag):
    DEG_0 = 0b00
    DEG_90 = 0b01
    DEG_180 = 0b10
    DEG_270 = 0b11


class Map_PointBit(IntFlag):
    IsValid = 0b000001       # 有效棋盘格
    IsP1 = 0b000010          # 属于P1/己方
    IsP2 = 0b000100          # 属于P2/对方
    IsSp = 0b001000          # 特殊点/SP点
    IsSupplySp = 0b010000    # 已激活特殊点
    IsPreview = 0b100000     # 预览/虚影

# 典型组合定义
class Map_PointMask:
    NotMap = 0     # 非地图区域（无效格）
    Empty = Map_PointBit.IsValid                      # 空位
    P1Normal = Map_PointBit.IsValid | Map_PointBit.IsP1       # 己方普通点
    P2Normal = Map_PointBit.IsValid | Map_PointBit.IsP2       # 对方普通点
    P1Special = Map_PointBit.IsValid | Map_PointBit.IsP1 | Map_PointBit.IsSp       # 己方SP点，未激活
    P2Special = Map_PointBit.IsValid | Map_PointBit.IsP2 | Map_PointBit.IsSp       # 对方SP点，未激活
    Conflict = Map_PointBit.IsValid | Map_PointBit.IsP1 | Map_PointBit.IsP2        # 普通冲突点
    ConflictSp = Map_PointBit.IsValid | Map_PointBit.IsP1 | Map_PointBit.IsP2 | Map_PointBit.IsSp  # SP冲突点
    P1SpActive = P1Special | Map_PointBit.IsSupplySp      # 激活后的己方SP点
    P2SpActive = P2Special | Map_PointBit.IsSupplySp      # 激活后的对方SP点

    # 预览
    P1Preview = P1Normal | Map_PointBit.IsPreview
    P2Preview = P2Normal | Map_PointBit.IsPreview
    P1SpPreview = P1Special | Map_PointBit.IsPreview
    P2SpPreview = P2Special | Map_PointBit.IsPreview

@dataclass
class Card_Single:
    """单张卡牌数据。加载时通过构造函数赋值，对战时仅读取，无需 setter。"""
    name: str                               # 卡牌名
    NameHash: str                           # 名称哈希
    Number: int                             # 卡牌编号
    Rarity: int                             # 稀有度
    CardPoint: int                          # 点数（计算所有非Empty格子数）
    SpecialCost: int                        # SP需求
    square_2d_0: List[List[int]]            # 0度旋转的8x8矩阵
    square_2d_90: List[List[int]]           # 90度旋转的8x8矩阵
    square_2d_180: List[List[int]]          # 180度旋转的8x8矩阵
    square_2d_270: List[List[int]]          # 270度旋转的8x8矩阵
    link_pos_0: Tuple[int, int]             # 0度旋转的参考点坐标(x, y)
    link_pos_90: Tuple[int, int]            # 90度旋转的参考点坐标(x, y)
    link_pos_180: Tuple[int, int]           # 180度旋转的参考点坐标(x, y)
    link_pos_270: Tuple[int, int]           # 270度旋转的参考点坐标(x, y)
    edge_0: Tuple[int, int, int, int]                 # 0度旋转的参考边界(左, 上, 右, 下)
    edge_90: Tuple[int, int, int, int]                # 90度旋转的参考边界(左, 上, 右, 下)
    edge_180: Tuple[int, int, int, int]               # 180度旋转的参考边界(左, 上, 右, 下)
    edge_270: Tuple[int, int, int, int]               # 270度旋转的参考边界(左, 上, 右, 下)
    SquareSortOffset: int                   # 排序偏移（可根据需要调整类型）
    RowId: Optional[Any] = None             # 行标识符（内部用，可选）

    def get_square_matrix(self, rotation: int) -> List[List[int]]:
        """根据旋转角度返回对应的8x8矩阵."""
        if rotation == Card_Rotation.DEG_0:
            return self.square_2d_0
        elif rotation == Card_Rotation.DEG_90:
            return self.square_2d_90
        elif rotation == Card_Rotation.DEG_180:
            return self.square_2d_180
        elif rotation == Card_Rotation.DEG_270:
            return self.square_2d_270
        else:
            raise ValueError("rotation must be one of (0, 90, 180, 270)")

    def get_link_pos(self, rotation: int) -> Tuple[int, int]:
        """根据旋转角度返回参考点坐标."""
        if rotation == Card_Rotation.DEG_0:
            return self.link_pos_0
        elif rotation == Card_Rotation.DEG_90:
            return self.link_pos_90
        elif rotation == Card_Rotation.DEG_180:
            return self.link_pos_180
        elif rotation == Card_Rotation.DEG_270:
            return self.link_pos_270
        else:
            raise ValueError("rotation must be one of (0, 90, 180, 270)")
    

@dataclass
class CardDeck:
    """包含15张Card_Single卡牌的牌组，实现随机抽取且不放回功能。"""
    cards: List[Card_Single]
    _drawn_flags: List[bool] = None  # 标志哪些卡已被抽取

    def __post_init__(self):
        if len(self.cards) != 15:
            raise ValueError("CardDeck必须包含15张卡牌")
        if self._drawn_flags is None:
            self._drawn_flags = [False] * 15

    def draw_random(self) -> Optional[Card_Single]:
        """随机抽取一张未抽取的卡牌，抽取后不放回，若已全部抽完返回None。"""
        import random
        available_indexes = [i for i, drawn in enumerate(self._drawn_flags) if not drawn]
        if not available_indexes:
            return None
        idx = random.choice(available_indexes)
        self._drawn_flags[idx] = True
        return self.cards[idx]

    def reset(self):
        """重置抽取状态（所有卡都可再次被抽取）。"""
        self._drawn_flags = [False] * 15

    def remaining_cards(self) -> List[Card_Single]:
        """返回还未被抽取的卡牌列表。"""
        return [card for card, drawn in zip(self.cards, self._drawn_flags) if not drawn]

    def all_drawn(self) -> bool:
        """判断是否所有卡都已被抽取。"""
        return all(self._drawn_flags)

@dataclass
class Card_In_Hand:
    """表示手中的四张卡牌结构。"""
    cards: List[Card_Single]

    def __post_init__(self):
        if len(self.cards) != 4:
            raise ValueError("Card_In_Hand必须包含4张Card_Single")

@dataclass
class GameMap:
    """地图数据结构，支持任意尺寸。每个点位为 Map_PointMask 的枚举值（int）。"""
    map_id: str        # 地图 id（对应 JSON 的 id）
    name: str          # 中文名
    ename: str         # 英文名
    width: int         # 地图宽度
    height: int        # 地图高度
    grid: List[List[int]]  # grid[y][x]，每个值为 Map_PointMask 枚举（如 Empty, P1Normal 等）

    def __post_init__(self):
        if len(self.grid) != self.height or any(len(row) != self.width for row in self.grid):
            raise ValueError("grid的尺寸必须与(width, height)一致")

    def get_point(self, x: int, y: int) -> int:
        """获取 (x, y) 位置的点位枚举值（Map_PointMask 常量）。"""
        if not (0 <= x < self.width and 0 <= y < self.height):
            raise IndexError("地图坐标超出范围")
        return self.grid[y][x]

    def set_point(self, x: int, y: int, mask: int):
        """设置 (x, y) 位置为 Map_PointMask 枚举值（如 Map_PointMask.Empty）。"""
        if not (0 <= x < self.width and 0 <= y < self.height):
            raise IndexError("地图坐标超出范围")
        self.grid[y][x] = mask

# ========== 回放结构（方案 A：纯动作日志） ==========

@dataclass
class ReplayMeta:
    """回放元信息：开局配置。"""
    map_id: str                    # 地图 id
    p1_deck: List[int]             # P1 牌组（卡牌 Number 列表，顺序决定抽牌）
    p2_deck: List[int]             # P2 牌组（同上）
    replay_version: int = 2        # 回放结构版本，便于后续兼容升级
    rng_seed: Optional[int] = None # 随机种子（用于严格复现）
    p1_draw_order: Optional[List[int]] = None  # P1 抽牌顺序（可选）
    p2_draw_order: Optional[List[int]] = None  # P2 抽牌顺序（可选）


@dataclass
class ReplayMove:
    #TODO：添加每回合手牌是哪四张
    #TODO: 添加完整地图情况（仅训练）
    """单步动作：出牌或弃权。

    说明：
    - 为了方便 JSON 序列化，这里不直接存放 `Card_Single` / `GameMap` 对象，
      而是存放「可还原」的原始数据：
        * `hand_card_numbers`：当回合 4 张手牌的卡牌编号（Number）
        * `map_grid`：当前整张地图的状态快照（用 `Map_PointMask` 的 int 值）
    - 在训练或回放时，可通过全局卡牌表 + `map_id` 将这些编号还原为完整卡面信息。
    """
    turn: int                      # 回合号
    player: str                    # "P1" 或 "P2"
    phase: Optional[str] = None    # 阶段：如 "select"/"resolve"/"post"
    # 弃权时 pass=True，以下字段可省略/None
    pass_turn: bool = False        # 是否弃权
    card_idx: Optional[int] = None # 手牌下标（0~3）
    card_number: Optional[int] = None   # 卡牌编号
    used_sp: Optional[bool] = None # 是否使用 SP
    x: Optional[int] = None        # 放置位置 x（参考点）
    y: Optional[int] = None        # 放置位置 y（参考点）
    rotation: Optional[int] = None # 旋转角度 0/90/180/270
    valid_action: Optional[bool] = None        # 是否为有效动作
    invalid_reason: Optional[str] = None       # 无效动作原因
    p1_sp_before: Optional[int] = None         # P1 行动前 SP
    p1_sp_after: Optional[int] = None          # P1 行动后 SP
    p2_sp_before: Optional[int] = None         # P2 行动前 SP
    p2_sp_after: Optional[int] = None          # P2 行动后 SP
    # 下面两个字段为「状态快照」，可选，仅在需要做训练 / 分析时写入
    hand_card_numbers: Optional[List[int]] = None   # 当前 4 张手牌对应的卡牌编号列表
    map_grid: Optional[List[List[int]]] = None      # 当前整张地图的 bitmask 网格快照


@dataclass
class ReplayResult:
    """回放结果：终局分数与胜者。"""
    p1_score: int
    p2_score: int
    winner: str                    # "P1" / "P2" / "draw"
    rounds_played: Optional[int] = None  # 实际完成回合数（通常为12）


@dataclass
class Replay:
    """完整回放数据（方案 A）。用于存储、回放、后续转 IL 轨迹。"""

    meta: ReplayMeta
    moves: List[ReplayMove]
    result: ReplayResult

    def to_dict(self) -> Dict[str, Any]:
        """转为可 JSON 序列化的字典。moves 中 pass_turn 导出为 "pass"。"""
        d = asdict(self)
        for m in d["moves"]:
            m["pass"] = m.pop("pass_turn", False)
        return d

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "Replay":
        """从字典（如 json.load）构建。支持 "pass" 或 "pass_turn" 键。"""
        meta_d = d["meta"]
        meta = ReplayMeta(
            replay_version=meta_d.get("replay_version", 1),
            map_id=meta_d["map_id"],
            p1_deck=meta_d["p1_deck"],
            p2_deck=meta_d["p2_deck"],
            rng_seed=meta_d.get("rng_seed"),
            p1_draw_order=meta_d.get("p1_draw_order"),
            p2_draw_order=meta_d.get("p2_draw_order"),
        )
        moves = []
        for m in d["moves"]:
            pass_val = m.get("pass", m.get("pass_turn", False))
            moves.append(ReplayMove(
                turn=m["turn"],
                player=m["player"],
                phase=m.get("phase"),
                pass_turn=pass_val,
                card_idx=m.get("card_idx"),
                card_number=m.get("card_number"),
                used_sp=m.get("used_sp"),
                x=m.get("x"),
                y=m.get("y"),
                rotation=m.get("rotation"),
                valid_action=m.get("valid_action"),
                invalid_reason=m.get("invalid_reason"),
                p1_sp_before=m.get("p1_sp_before"),
                p1_sp_after=m.get("p1_sp_after"),
                p2_sp_before=m.get("p2_sp_before"),
                p2_sp_after=m.get("p2_sp_after"),
                hand_card_numbers=m.get("hand_card_numbers"),
                map_grid=m.get("map_grid"),
            ))
        res_d = d["result"]
        result = ReplayResult(
            p1_score=res_d["p1_score"],
            p2_score=res_d["p2_score"],
            winner=res_d["winner"],
            rounds_played=res_d.get("rounds_played"),
        )
        return cls(meta=meta, moves=moves, result=result)
