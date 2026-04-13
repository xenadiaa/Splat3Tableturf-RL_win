# -*- coding: utf-8 -*-
"""临时脚本：将 15 个地图 JSON 合并为一个数组，同 MiniGameCardInfo.json 格式."""
import json
import os

MAP_DIR = os.path.dirname(os.path.abspath(__file__))
FILES = [
    "map_加速高速公路_Pedal_to_the_Metal.json",
    "map_双子岛_Double_Gemini.json",
    "map_小巧竞技场_Box_Seats.json",
    "map_扭转河_River_Drift.json",
    "map_正方广场_Square_Squared.json",
    "map_正直大道_Main_Street.json",
    "map_清脆饼干_Cracker_Snap.json",
    "map_漆黑深林_Sticky_Thicket.json",
    "map_罚分花园_X_Marks_the_Garden.json",
    "map_轻飘湖_Lakefront_Property.json",
    "map_酷似大道_Two_Lane_Splattop.json",
    "map_钢骨建筑_Girder_for_Battle.json",
    "map_间隔墙_Over_the_Line.json",
    "map_雷霆车站_Thunder_Point.json",
    "map_面具屋_Mask_Mansion.json",
]

out = []
for f in FILES:
    path = os.path.join(MAP_DIR, f)
    with open(path, "r", encoding="utf-8") as fp:
        out.append(json.load(fp))

out_path = os.path.join(MAP_DIR, "MiniGameMapInfo.json")
with open(out_path, "w", encoding="utf-8") as fp:
    json.dump(out, fp, ensure_ascii=False, indent=2)
print("Written", out_path, "with", len(out), "maps")
