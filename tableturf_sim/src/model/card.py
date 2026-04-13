from dataclasses import dataclass
from typing import List, Optional
import json

@dataclass
class Card:
    name: str
    number: int
    rarity: str
    special_cost: int
    # 将 64 个 Square 状态压缩为位掩码，节省空间且方便位运算
    fill_mask: int      # 哪些格子被填充了 (Fill 或 Special)
    special_mask: int   # 哪些格子是特殊点位 (Special)
    row_id: str

    @classmethod
    def from_json(cls, data: dict):
        """解析单条 JSON 数据并进行转换"""
        fill_m = 0
        spec_m = 0
        
        # 遍历 64 个格子，利用位移操作生成 64 位整数
        for i, status in enumerate(data.get("Square", [])):
            if status in ("Fill", "Special"):
                fill_m |= (1 << i)  # 在第 i 位标记为 1
            if status == "Special":
                spec_m |= (1 << i)  # 在第 i 位标记为 1
                
        return cls(
            name=data["Name"],
            number=data["Number"],
            rarity=data["Rarity"],
            special_cost=data["SpecialCost"],
            fill_mask=fill_m,
            special_mask=spec_m,
            row_id=data["__RowId"]
        )

# 使用示例
raw_json = { "Name": "Ajio", "Number": 193, "Rarity": "Rare", "SpecialCost": 4, "Square": ["Empty", "Fill", "Special"], "__RowId": "MiniGame_Ajio" }
card = Card.from_json(raw_json)

print(f"卡牌: {card.name}, 填充位掩码: {bin(card.fill_mask)}")