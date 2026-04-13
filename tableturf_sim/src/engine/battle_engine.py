import random

# --- 1. 初始化游戏配置 ---
MAX_ROUNDS = 12

PLAYER_DECK = []
PLAYER_HAND = []
PLAYER_SP = 0

ENEMY_DECK = []
ENEMY_HAND = []
ENEMY_SP = 0

# 生成 3x3 地图: {'A1': '·', 'A2': '·', ..., 'C3': '·'}
game_map = load_map()
def load_map():
    return {f"{r}{c}": "·" for r in "ABC" for c in "123"}

def get_validated_move(player_name, player_hand):
    """
    核心验证逻辑：
    1. 检查传入数据格式是否为 '卡牌 坐标'
    2. 检查所使用卡牌是否在玩家手牌内
    3. 检查放置坐标是否在地图范围内
    4. 检查所使用卡牌是否可以放置在指定坐标（是否有相邻，是否无冲突，是否出图）
    5. 检查玩家sp与卡牌sp值是否够用（sp攻击时需要消耗sp）
    6. 检查sp攻击是否覆盖无法覆盖格子（冲突格、sp格）
    """
    while True:
        try:
            print(f"\n[{player_name}] 手牌: {player_hand}")
            raw_input = input(f"[{player_name}] 请输入指令 (格式: 卡牌 坐标): ").strip().upper()
            
            # 拆分输入，处理可能的空格
            parts = raw_input.split()
            if len(parts) != 2:
                raise ValueError("格式错误")
                
            card, pos = parts[0], parts[1]

            # 验证卡牌（支持输入文字不带emoji的情况，增强容错）
            matched_card = next((c for c in player_hand if card in c), None)
            
            if not matched_card:
                print(f"❌ 无效：你没有 [{card}]，请重新输入！")
            elif pos not in game_map:
                print(f"❌ 无效：坐标 [{pos}] 不在地图内，请重新输入！")
            else:
                print(f"✅ {player_name} 指令确认: {matched_card} 目标 {pos}")
                return matched_card, pos
        except:
            print("❌ 格式错误！请输入类似 '火攻 A1'。")

# --- 2. 初始发牌 ---
hand_a = [random.choice(CARD_POOL) for _ in range(3)]
hand_b = [random.choice(CARD_POOL) for _ in range(3)]

# --- 3. 游戏主循环 ---
for round_num in range(1, MAX_ROUNDS + 1):
    print(f"\n{'='*25} 第 {round_num} / {MAX_ROUNDS} 回合 {'='*25}")
    
    # 获取双方有效输入 (自动处理“无效重发”逻辑)
    card_a, pos_a = get_validated_move("玩家 A", hand_a)
    card_b, pos_b = get_validated_move("玩家 B", hand_b)

    # 执行比较指令并覆写地图
    print("\n--- 正在处理指令 ---")
    if pos_a == pos_b:
        game_map[pos_a] = "💥"
        print(f"💥 冲突！双方都在 {pos_a} 施法，该地变为废墟！")
    else:
        # 覆写地图：取卡牌表情或首字
        game_map[pos_a] = card_a[0]
        game_map[pos_b] = card_b[0]
        print(f"🗺️ 地图更新：{pos_a} 变为 {card_a}，{pos_b} 变为 {card_b}")

    # 告知消耗并各发一张牌
    hand_a.remove(card_a)
    hand_b.remove(card_b)
    
    new_a, new_b = random.choice(CARD_POOL), random.choice(CARD_POOL)
    hand_a.append(new_a)
    hand_b.append(new_b)
    
    print(f"♻️ 系统：旧卡已消耗，新发牌：A -> [{new_a}], B -> [{new_b}]")
    
    # 打印当前地图状态
    print(f"当前地图：{game_map}")

print(f"\n{'='*20} 游戏结束 {'='*20}")
