
from enum import IntFlag, auto

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


class PointBit(IntFlag):
    IsValid = 0b000001       # 有效棋盘格
    IsP1 = 0b000010          # 属于P1/己方
    IsP2 = 0b000100          # 属于P2/对方
    IsSp = 0b001000          # 特殊点/SP点
    IsSupplySp = 0b010000    # 已激活特殊点
    IsPreview = 0b100000     # 预览/虚影

# 典型组合定义
#include <cstdint>

// 基础比特位定义
enum PointBit : uint8_t {
    IsValid    = 1 << 0,  // 000001
    IsP1       = 1 << 1,  // 000010
    IsP2       = 1 << 2,  // 000100
    IsSp       = 1 << 3,  // 001000
    IsSupplySp = 1 << 4,  // 010000
    IsPreview  = 1 << 5   // 100000
};

// 复合掩码定义
namespace PointMask {
    enum Mask : uint8_t {
        NotMap     = 0,
        Empty      = IsValid,
        P1Normal   = IsValid | IsP1,
        P2Normal   = IsValid | IsP2,
        P1Special  = IsValid | IsP1 | IsSp,
        P2Special  = IsValid | IsP2 | IsSp,
        Conflict   = IsValid | IsP1 | IsP2,
        ConflictSp = IsValid | IsP1 | IsP2 | IsSp,
        P1SpActive = P1Special | IsSupplySp,
        P2SpActive = P2Special | IsSupplySp,

        // 预览状态
        P1Preview   = P1Normal | IsPreview,
        P2Preview   = P2Normal | IsPreview,
        P1SpPreview = P1Special | IsPreview,
        P2SpPreview = P2Special | IsPreview
    };
}



