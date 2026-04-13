from __future__ import annotations

import argparse
import glob
import json
import sys
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
SIM_ROOT = REPO_ROOT / "tableturf_sim"
if str(SIM_ROOT) not in sys.path:
    sys.path.insert(0, str(SIM_ROOT))

from tableturf_vision.tableturf_mapper import analyze_image, _load_layout
from tableturf_vision.reference_matcher import match_map_by_reference_board_labels
from src.utils.common_utils import create_card_from_id
from src.utils.localization import lookup_card_name_zh


CLR_RESET = "\033[0m"
RGB: Dict[str, Tuple[int, int, int]] = {
    "empty": (0, 0, 0),
    "p1_fill": (255, 255, 0),
    "p1_special": (255, 192, 0),
    "p2_fill": (0, 112, 192),
    "p2_special": (0, 176, 240),
    "conflict": (128, 128, 128),
    "p1_special_activated": (255, 192, 0),
    "p2_special_activated": (0, 176, 240),
}

TOKEN_MAP: Dict[str, str] = {
    "invalid": "  ",
    "empty": "[]",
    "p1_fill": "[]",
    "p1_special": "[]",
    "p2_fill": "[]",
    "p2_special": "[]",
    "conflict": "[]",
    "p1_special_activated": "/\\",
    "p2_special_activated": "/\\",
}


def _rgb(text: str, rgb: Tuple[int, int, int]) -> str:
    r, g, b = rgb
    return f"\033[38;2;{r};{g};{b}m{text}{CLR_RESET}"


def _normalize_board_label(label: str) -> str:
    # Backward compatibility with old mapper labels.
    fallback = {
        "fill": "p1_fill",
        "special": "p1_special",
        "special_activated": "p1_special_activated",
    }
    return fallback.get(label, label)


def _map_board_line(labels: Sequence[str]) -> str:
    out: List[str] = []
    for raw in labels:
        label = _normalize_board_label(raw)
        token = TOKEN_MAP.get(label, "??")
        if token.strip() == "":
            out.append(token)
            continue
        rgb = RGB.get(label, (255, 255, 255))
        out.append(_rgb(token, rgb))
    return "".join(out)


def _render_board_grid(labels_2d: List[List[str]]) -> List[str]:
    return [_map_board_line(row) for row in labels_2d]


def _render_card_grid(labels_2d: List[List[str]]) -> List[str]:
    # Card preview rule: '*' for SP cell, '■' for fill cell, '.' for empty.
    out: List[str] = []
    for row in labels_2d:
        chars: List[str] = []
        for v in row:
            if v in {"special", "special_activated"}:
                chars.append("*")
            elif v == "empty":
                chars.append(".")
            else:
                chars.append("■")
        out.append("".join(chars))
    return out


def _render_cards_2x2(cards: List[Dict]) -> List[str]:
    grids = [_render_card_grid(c["labels"]) for c in cards]
    # guarantee 8 lines display per card block (pad if needed)
    pad_h = 8
    pad_w = 8
    norm: List[List[str]] = []
    for g in grids:
        g2 = [line[:pad_w].ljust(pad_w, ".") for line in g[:pad_h]]
        while len(g2) < pad_h:
            g2.append("." * pad_w)
        norm.append(g2)

    lines: List[str] = []
    for i in range(pad_h):
        lines.append(f"{norm[0][i]}  {norm[1][i]}")
    lines.append("")
    for i in range(pad_h):
        lines.append(f"{norm[2][i]}  {norm[3][i]}")
    return lines


def _label_to_value_for_card(label: str) -> int:
    if label in {"special", "special_activated"}:
        return 2
    if label == "empty":
        return 0
    return 1


def _rotate_matrix_8(mat: List[List[int]], steps: int) -> List[List[int]]:
    out = [row[:] for row in mat]
    for _ in range(steps % 4):
        out = [list(row) for row in zip(*out[::-1])]
    return out


def _extract_card_matrix(card_labels: List[List[str]]) -> List[List[int]]:
    m = [[_label_to_value_for_card(v) for v in row] for row in card_labels]
    if len(m) != 8:
        m = (m + [[0] * 8 for _ in range(8)])[:8]
    m = [((row + [0] * 8)[:8]) for row in m]
    return m


def _card_distance(a: List[List[int]], b: List[List[int]]) -> int:
    dist = 0
    for y in range(8):
        for x in range(8):
            av = a[y][x]
            bv = b[y][x]
            if av != bv:
                # SP mismatch is more important than fill mismatch.
                if av == 2 or bv == 2:
                    dist += 2
                else:
                    dist += 1
    return dist


def _match_card(card_labels: List[List[str]]) -> Dict:
    target = _extract_card_matrix(card_labels)
    best = {
        "number": None,
        "name": "",
        "name_zh": "",
        "distance": 10**9,
        "rotation": 0,
    }
    for number in range(1, 267):
        try:
            c = create_card_from_id(number)
        except Exception:
            continue
        for rot in (0, 1, 2, 3):
            mat = c.get_square_matrix(rot)
            dist = _card_distance(target, mat)
            if dist < best["distance"]:
                best = {
                    "number": int(number),
                    "name": c.name,
                    "name_zh": lookup_card_name_zh(c.name) or "",
                    "distance": int(dist),
                    "rotation": int(rot),
                }
    return best


def _resolve_image_path(input_dir: Path, image_path: str) -> Path:
    if image_path:
        p = Path(image_path)
        if not p.is_absolute():
            p = REPO_ROOT / p
        return p
    cands = sorted(glob.glob(str(input_dir / "capture_*.*")))
    if not cands:
        raise FileNotFoundError(f"no capture_* image in {input_dir}")
    return Path(cands[-1])


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run tableturf mapper and print ui-p1 style text preview in terminal.")
    p.add_argument("--image", default="", help="image path; empty means latest capture image")
    p.add_argument("--input-dir", default="vision_capture/debug")
    p.add_argument("--layout-json", default="tableturf_vision/tableturf_layout.json")
    p.add_argument("--save-json", default="", help="optional output json path")
    return p.parse_args()


def main() -> int:
    args = _parse_args()

    input_dir = Path(args.input_dir)
    if not input_dir.is_absolute():
        input_dir = REPO_ROOT / input_dir
    layout_json = Path(args.layout_json)
    if not layout_json.is_absolute():
        layout_json = REPO_ROOT / layout_json
    layout = _load_layout(layout_json)

    image_path = _resolve_image_path(input_dir=input_dir, image_path=args.image)
    result = analyze_image(
        image_path,
        layout=layout,
        write_overlay=False,
        out_dir=input_dir,
    )
    map_match = match_map_by_reference_board_labels(result["board"]["labels"], layout)
    card_matches = [_match_card(c["labels"]) for c in result["cards"]]

    print(f"Image: {result['image']}")
    print(f"Board: {result['board']['grid_rows']}x{result['board']['grid_cols']}")
    print(
        f"MapMatch: {map_match['map_id']} / {map_match['map_name_zh']} "
        f"(score={map_match['score']:.3f})"
    )
    print("Legend(board):")
    print(f"  {_rgb('[]', RGB['empty'])} empty")
    print(f"  {_rgb('[]', RGB['p1_fill'])} p1_fill")
    print(f"  {_rgb('[]', RGB['p1_special'])} p1_special")
    print(f"  {_rgb('[]', RGB['p2_fill'])} p2_fill")
    print(f"  {_rgb('[]', RGB['p2_special'])} p2_special")
    print(f"  {_rgb('[]', RGB['conflict'])} conflict")
    print(f"  {_rgb('/\\\\', RGB['p1_special_activated'])} p1_special_activated")
    print(f"  {_rgb('/\\\\', RGB['p2_special_activated'])} p2_special_activated")
    print("  '  ' invalid")
    print("")
    print("[Board]")
    for line in _render_board_grid(result["board"]["labels"]):
        print(line)
    print("")
    print("[Hand 2x2 (ui-p1 style)]")
    for line in _render_cards_2x2(result["cards"]):
        print(line)
    print("")
    for idx, c in enumerate(result["cards"], start=1):
        m = card_matches[idx - 1]
        print(
            f"Card{idx}: sig={c['signature']} size={c['grid_rows']}x{c['grid_cols']} "
            f"-> #{m['number']} {m['name_zh']} ({m['name']}) rot={m['rotation']} d={m['distance']}"
        )

    if args.save_json:
        out = Path(args.save_json)
        if not out.is_absolute():
            out = REPO_ROOT / out
        out.parent.mkdir(parents=True, exist_ok=True)
        result["map_match"] = map_match
        result["card_matches"] = card_matches
        out.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\nSaved: {out}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
