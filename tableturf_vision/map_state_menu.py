from __future__ import annotations

from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tableturf_vision.map_state_detector import (
    MAP_NAMES,
    _reference_image_path,
    detect_map_state_image_path,
    render_map_state_grid,
)


def _prompt_map_name() -> str:
    print("请选择地图：")
    for i, name in enumerate(MAP_NAMES, start=1):
        print(f"  {i}. {name}")
    while True:
        raw = input("输入地图编号: ").strip()
        if not raw.isdigit():
            print("请输入数字编号。")
            continue
        idx = int(raw)
        if 1 <= idx <= len(MAP_NAMES):
            return MAP_NAMES[idx - 1]
        print("编号超出范围，请重新输入。")


def _prompt_image_path(map_name: str) -> Path:
    default_ref = _reference_image_path(map_name)
    print(f"默认使用参照图: {default_ref}")
    raw = input("直接回车使用参照图，或输入自定义图片路径: ").strip()
    if not raw:
        return default_ref
    p = Path(raw)
    if not p.is_absolute():
        p = REPO_ROOT / p
    return p


def main() -> int:
    map_name = _prompt_map_name()
    image_path = _prompt_image_path(map_name)
    result = detect_map_state_image_path(image_path, map_name)

    print()
    print(f"Image: {result['image']}")
    print(f"Map: {map_name}")
    print(f"Counts: {result['counts']}")
    print("[Grid]")
    for line in render_map_state_grid(map_name, result, colorize=True):
        print(line)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
