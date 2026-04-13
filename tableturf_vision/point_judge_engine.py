from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np

import sys

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tableturf_vision.tableturf_mapper import (
    _load_layout,
    _parse_board,
    _parse_card_with_slot,
)
from tableturf_vision.reference_matcher import (
    load_map_info,
    match_map_by_reference_board_labels,
    valid_mask_from_labels,
)
CARD_INFO_PATH = REPO_ROOT / "tableturf_sim" / "data" / "cards" / "MiniGameCardInfo.json"

CARD_ID_TO_ZH = {
    232: "浮墨幕墙",
    174: "斯普拉旋转枪 联名",
    39: "斯普拉旋转枪",
    201: "S-BLAST92",
}
CALIBRATION_EXPECTED_CARDS_BY_IMAGE = {
    "间隔墙.jpg": [232, 174, 39, 201],
    "间隔墙.png": [232, 174, 39, 201],
}

# User-required exact RGB rules (stored in RGB, converted to BGR on use).
EXACT_RGB_RULES = {
    "empty": [0, 0, 0],
    "p1_fill": [255, 255, 0],
    "p1_special": [255, 192, 0],
    "p2_fill": [0, 112, 192],
    "p2_special": [0, 176, 240],
    "conflict": [128, 128, 128],
}
EXACT_CARD_RGB_RULES = {
    "empty": [0, 0, 0],
    "fill": [255, 255, 0],
    "special": [255, 192, 0],
}


def _rgb_to_bgr(rgb: List[int]) -> List[int]:
    return [int(rgb[2]), int(rgb[1]), int(rgb[0])]


def _map_info_by_id(map_id: str) -> Dict | None:
    for m in load_map_info():
        if str(m.get("id", "")) == str(map_id):
            return m
    return None

def _load_card_info_index() -> Dict[int, Dict]:
    rows = json.loads(CARD_INFO_PATH.read_text(encoding="utf-8"))
    out: Dict[int, Dict] = {}
    for r in rows:
        out[int(r["Number"])] = r
    return out


def _card_square_8x8(card_id: int, card_idx: Dict[int, Dict]) -> List[List[str]]:
    row = card_idx.get(int(card_id))
    if not row:
        return [["empty" for _ in range(8)] for _ in range(8)]
    sq = row.get("Square", [])
    if not isinstance(sq, list) or len(sq) != 64:
        return [["empty" for _ in range(8)] for _ in range(8)]
    # MiniGameCardInfo Square order is bottom-left origin:
    # x increases left->right, then y increases bottom->top.
    # Our display/sample matrix uses top-left origin, so we need vertical flip.
    bottom_up: List[List[str]] = []
    for y in range(8):
        rr: List[str] = []
        for x in range(8):
            v = str(sq[y * 8 + x]).lower()
            if v == "fill":
                rr.append("fill")
            elif v == "special":
                rr.append("special")
            else:
                rr.append("empty")
        bottom_up.append(rr)
    out: List[List[str]] = []
    for r in range(8):
        out.append(bottom_up[7 - r])
    return out


def _resolve_image(image: str, input_dir: str) -> Path:
    if image:
        p = Path(image)
        if not p.is_absolute():
            p = REPO_ROOT / p
        return p
    d = Path(input_dir)
    if not d.is_absolute():
        d = REPO_ROOT / d
    cands = sorted(glob.glob(str(d / "capture_*.*")))
    if not cands:
        raise FileNotFoundError(f"no capture_* image in {d}")
    return Path(cands[-1])

def _sample_bgr_exact(frame: np.ndarray, x: int, y: int) -> List[int]:
    h, w = frame.shape[:2]
    x = int(max(0, min(x, w - 1)))
    y = int(max(0, min(y, h - 1)))
    b, g, r = frame[y, x]
    return [int(b), int(g), int(r)]


def _exact_label_from_bgr(bgr: List[int], rules_rgb: Dict[str, List[int]]) -> str:
    for name, rgb in rules_rgb.items():
        if bgr == _rgb_to_bgr(rgb):
            return name
    return "unknown"


def _ansi_fg(token: str, rgb: List[int]) -> str:
    r, g, b = int(rgb[0]), int(rgb[1]), int(rgb[2])
    return f"\033[38;2;{r};{g};{b}m{token}\033[0m"


def _nearest_label_from_bgr(bgr: List[int], rules_rgb: Dict[str, List[int]]) -> str:
    # Nearest-color fallback for compressed frames (jpg/capture card drift).
    v = np.array([int(bgr[0]), int(bgr[1]), int(bgr[2])], dtype=np.int32)
    best_name = "unknown"
    best_d = 10**18
    for name, rgb in rules_rgb.items():
        tgt = np.array(_rgb_to_bgr(rgb), dtype=np.int32)
        d = int(((v - tgt) ** 2).sum())
        if d < best_d:
            best_d = d
            best_name = name
    return best_name


def _point_type_to_expected_label(pt_code: int) -> str:
    # For map-template calibration:
    # 0: invalid, 1: normal empty cell, 11/13: initial special cells.
    # Other non-zero codes are treated as non-playable/blocked cells in visual map view.
    if pt_code == 0:
        return "invalid"
    if pt_code == 1:
        return "empty"
    if pt_code == 11:
        return "p1_special"
    if pt_code == 13:
        return "p2_special"
    return "conflict"


def _is_blocked_point_type(pt_code: int) -> bool:
    # Non-playable/obstacle cells in map json.
    return int(pt_code) in {7}


def _expected_label_from_point_type(pt_code: int) -> str:
    if pt_code == 0:
        return "invalid"
    if pt_code == 3:
        return "p1_fill"
    if pt_code == 5:
        return "p2_fill"
    if _is_blocked_point_type(pt_code):
        return "conflict"
    if pt_code == 11:
        return "p1_special"
    if pt_code == 13:
        return "p2_special"
    return "empty"


def _board_palette_label_with_constraints(pt_code: int, bgr: List[int]) -> str:
    if pt_code == 0:
        return "invalid"
    if pt_code == 3:
        return "p1_fill"
    if pt_code == 5:
        return "p2_fill"
    if _is_blocked_point_type(pt_code):
        return "conflict"
    if pt_code == 11:
        return "p1_special"
    if pt_code == 13:
        return "p2_special"
    return _nearest_label_from_bgr(bgr, EXACT_RGB_RULES)


def _find_best_uniform_offset(
    frame: np.ndarray,
    board_points: List[Dict],
    sample_radius: int,
    search: int = 6,
) -> Tuple[int, int, float]:
    # Keep spacing fixed; only translate the whole network by (ox, oy).
    best_score = -1.0
    best_ox, best_oy = 0, 0
    for oy in range(-search, search + 1):
        for ox in range(-search, search + 1):
            hit = 0.0
            tot = 0.0
            for p in board_points:
                pt_code = int(p.get("point_type_code", 0))
                x = int(p["x"]) + ox
                y = int(p["y"]) + oy
                bgr = _sample_patch_bgr(frame, x, y, sample_radius) if sample_radius > 0 else _sample_bgr_exact(frame, x, y)
                got = _board_palette_label_with_constraints(pt_code, bgr)
                exp = _expected_label_from_point_type(pt_code)
                w = 1.0
                if pt_code in (0, 7):
                    w = 2.0
                elif pt_code in (3, 5):
                    w = 2.5
                elif pt_code in (11, 13):
                    w = 3.0
                if got == exp:
                    hit += w
                tot += w
            score = float(hit / max(1e-9, tot))
            if score > best_score:
                best_score = score
                best_ox, best_oy = ox, oy
    return best_ox, best_oy, best_score


def _sample_patch_bgr(frame: np.ndarray, x: int, y: int, radius: int) -> List[int]:
    h, w = frame.shape[:2]
    x0 = max(0, int(x - radius))
    x1 = min(w, int(x + radius + 1))
    y0 = max(0, int(y - radius))
    y1 = min(h, int(y + radius + 1))
    patch = frame[y0:y1, x0:x1]
    if patch.size == 0:
        return _sample_bgr_exact(frame, x, y)
    bgr = patch.reshape(-1, 3).mean(axis=0)
    return [int(round(float(bgr[0]))), int(round(float(bgr[1]))), int(round(float(bgr[2])))]


def _build_uniform_centers(
    bx: int,
    by: int,
    bw: int,
    bh: int,
    cols: int,
    rows: int,
    mx: int,
    my: int,
) -> Tuple[List[List[Tuple[int, int]]], float, float]:
    ew = max(1.0, float(bw - 2 * mx))
    eh = max(1.0, float(bh - 2 * my))
    dx = ew / float(cols)
    dy = eh / float(rows)
    centers = [
        [
            (
                int(round(float(bx + mx) + (c + 0.5) * dx)),
                int(round(float(by + my) + (r + 0.5) * dy)),
            )
            for c in range(cols)
        ]
        for r in range(rows)
    ]
    return centers, dx, dy


def _optimize_uniform_grid(
    frame: np.ndarray,
    bx: int,
    by: int,
    bw: int,
    bh: int,
    pt_resized: np.ndarray,
) -> Dict:
    rows, cols = pt_resized.shape[:2]
    best = {"score": -1.0, "mx": 0, "my": 0, "centers": None, "dx": 0.0, "dy": 0.0, "radius": 0}
    # Keep all points on a fixed-spacing uniform network; only solve inner margins.
    max_mx = max(0, int(round(bw * 0.16)))
    max_my = max(0, int(round(bh * 0.16)))
    step_x = max(1, int(round(bw * 0.01)))
    step_y = max(1, int(round(bh * 0.01)))
    for mx in range(0, max_mx + 1, step_x):
        for my in range(0, max_my + 1, step_y):
            centers, dx, dy = _build_uniform_centers(bx, by, bw, bh, cols, rows, mx, my)
            # Smaller patch to avoid bleeding across adjacent rows/cols.
            rad = max(1, min(3, int(round(min(dx, dy) * 0.08))))
            ok = 0
            tot = 0
            for r in range(rows):
                for c in range(cols):
                    exp = _point_type_to_expected_label(int(pt_resized[r, c]))
                    x, y = centers[r][c]
                    bgr = _sample_patch_bgr(frame, x, y, rad)
                    got = _nearest_label_from_bgr(bgr, EXACT_RGB_RULES)
                    tot += 1
                    if got == exp:
                        ok += 1
            score = float(ok / max(1, tot))
            if score > best["score"]:
                best = {"score": score, "mx": mx, "my": my, "centers": centers, "dx": dx, "dy": dy, "radius": rad}
    return best


def _optimize_uniform_card_grid(
    frame: np.ndarray,
    bx: int,
    by: int,
    bw: int,
    bh: int,
    expected_mat: List[List[str]],
) -> Dict:
    rows, cols = 8, 8
    best = {"score": -1.0, "mx": 0, "my": 0, "centers": None, "dx": 0.0, "dy": 0.0, "radius": 0}
    max_mx = max(0, int(round(bw * 0.16)))
    max_my = max(0, int(round(bh * 0.16)))
    step_x = max(1, int(round(bw * 0.01)))
    step_y = max(1, int(round(bh * 0.01)))
    for mx in range(0, max_mx + 1, step_x):
        for my in range(0, max_my + 1, step_y):
            centers, dx, dy = _build_uniform_centers(bx, by, bw, bh, cols, rows, mx, my)
            rad = max(1, int(round(min(dx, dy) * 0.2)))
            ok = 0
            tot = 0
            for r in range(rows):
                for c in range(cols):
                    exp = expected_mat[r][c]
                    x, y = centers[r][c]
                    bgr = _sample_patch_bgr(frame, x, y, rad)
                    got = _nearest_label_from_bgr(bgr, EXACT_CARD_RGB_RULES)
                    tot += 1
                    if got == exp:
                        ok += 1
            score = float(ok / max(1, tot))
            if score > best["score"]:
                best = {"score": score, "mx": mx, "my": my, "centers": centers, "dx": dx, "dy": dy, "radius": rad}
    return best


def init_first_turn_profile(image_path: Path, layout_json: Path | None, profile_out: Path | None) -> Dict:
    frame = cv2.imread(str(image_path))
    if frame is None:
        raise ValueError(f"cannot read image: {image_path}")
    layout = _load_layout(layout_json)

    card_idx = _load_card_info_index()
    board = _parse_board(frame, layout)
    cards = [_parse_card_with_slot(frame, roi, layout, i) for i, roi in enumerate(layout["card_rois_norm"])]

    # Current stage only uses reference base PNGs for map recognition.
    map_match = match_map_by_reference_board_labels(board.labels, layout)
    tmpl_board = map_match.get("board_from_template") or board
    fh, fw = frame.shape[:2]
    map_info = _map_info_by_id(str(map_match.get("map_id", "")))
    if map_info:
        map_w = int(map_info.get("width", 0))
        map_h = int(map_info.get("height", 0))
    else:
        map_w = 0
        map_h = 0

    if map_w > 0 and map_h > 0:
        rows, cols = map_h, map_w
        base = np.array([[int(v) for v in row] for row in map_info.get("point_type", [])], dtype=np.int32)
        if base.shape[:2] != (rows, cols):
            pt_resized = cv2.resize(base, (cols, rows), interpolation=cv2.INTER_NEAREST)
        else:
            pt_resized = base
        # Use JSON map size to generate full board centers, avoiding parser line-miss.
        bx_n, by_n, bw_n, bh_n = layout["board_roi_norm"]
        bx = int(round(float(bx_n) * fw))
        by = int(round(float(by_n) * fh))
        bw = int(round(float(bw_n) * fw))
        bh = int(round(float(bh_n) * fh))
        fit = _optimize_uniform_grid(frame, bx, by, bw, bh, pt_resized)
        centers_abs = fit["centers"]
        grid_meta = {
            "roi_abs": [int(bx), int(by), int(bw), int(bh)],
            "inner_margin_px": [int(fit["mx"]), int(fit["my"])],
            "step_px": [float(fit["dx"]), float(fit["dy"])],
            "sample_radius_px": int(fit["radius"]),
            "calibration_score": float(fit["score"]),
            "network_uniform": True,
        }
    else:
        # Fallback only if map info is unavailable.
        rows = len(tmpl_board.centers_abs)
        cols = len(tmpl_board.centers_abs[0]) if rows else 0
        pt_resized = np.array(valid_mask_from_labels(tmpl_board.labels), dtype=np.int32)
        centers_abs = tmpl_board.centers_abs
        grid_meta = {"network_uniform": False}
    valid_mask = (pt_resized != 0).astype(np.uint8).tolist()

    # Important: expected cards are only used for calibration template images,
    # not permanently bound to a map id during normal recognition.
    expected_ids = CALIBRATION_EXPECTED_CARDS_BY_IMAGE.get(image_path.name, [])
    expected_cards: List[Dict] = []
    for cid in expected_ids:
        card_row = card_idx.get(int(cid), {})
        expected_cards.append(
            {
                "id": int(cid),
                "name_zh": CARD_ID_TO_ZH.get(int(cid), str(card_row.get("Name", f"Card{cid}"))),
                "name_key": str(card_row.get("Name", "")),
            }
        )

    h, w = frame.shape[:2]
    board_points: List[Dict] = []
    for r in range(rows):
        for c in range(cols):
            x, y = centers_abs[r][c]
            sample_radius = int(grid_meta.get("sample_radius_px", 0))
            ref_bgr = _sample_patch_bgr(frame, int(x), int(y), sample_radius) if sample_radius > 0 else _sample_bgr_exact(frame, int(x), int(y))
            pt_code = int(pt_resized[r, c]) if r < pt_resized.shape[0] and c < pt_resized.shape[1] else 0
            if pt_code == 0:
                ref_label = "invalid"
            elif pt_code == 11:
                ref_label = "p1_special"
            elif pt_code == 13:
                ref_label = "p2_special"
            else:
                # For initial map state, default valid cell is empty.
                ref_label = "empty"
            board_points.append(
                {
                    "r": int(r),
                    "c": int(c),
                    "x": int(x),
                    "y": int(y),
                    "x_norm": float(x / max(1, w - 1)),
                    "y_norm": float(y / max(1, h - 1)),
                    "is_valid_map_cell": bool(valid_mask[r][c] == 1),
                    "point_type_code": pt_code,
                    "ref_bgr": ref_bgr,
                    "ref_label": ref_label,
                }
            )

    card_points: Dict[str, List[Dict]] = {}
    card_grids: Dict[str, Dict] = {}
    for i, card in enumerate(cards, start=1):
        expected_mat = _card_square_8x8(expected_ids[i - 1], card_idx) if i - 1 < len(expected_ids) else card.labels
        roi_n = layout["card_rois_norm"][i - 1]
        cx_n, cy_n, cw_n, ch_n = float(roi_n[0]), float(roi_n[1]), float(roi_n[2]), float(roi_n[3])
        bx = int(round(cx_n * fw))
        by = int(round(cy_n * fh))
        bw = int(round(cw_n * fw))
        bh = int(round(ch_n * fh))
        fit = _optimize_uniform_card_grid(frame, bx, by, bw, bh, expected_mat)
        centers_abs = fit["centers"]
        card_grids[f"slot{i}"] = {
            "roi_abs": [int(bx), int(by), int(bw), int(bh)],
            "inner_margin_px": [int(fit["mx"]), int(fit["my"])],
            "step_px": [float(fit["dx"]), float(fit["dy"])],
            "sample_radius_px": int(fit["radius"]),
            "calibration_score": float(fit["score"]),
            "network_uniform": True,
        }
        pts: List[Dict] = []
        for r in range(8):
            for c in range(8):
                x, y = centers_abs[r][c]
                ref_bgr = _sample_patch_bgr(frame, int(x), int(y), int(fit["radius"]))
                ref_label = str(card.labels[r][c]) if r < len(card.labels) and c < len(card.labels[r]) else "empty"
                pts.append(
                    {
                        "r": int(r),
                        "c": int(c),
                        "x": int(x),
                        "y": int(y),
                        "x_norm": float(x / max(1, w - 1)),
                        "y_norm": float(y / max(1, h - 1)),
                        "ref_bgr": ref_bgr,
                        "ref_label": ref_label,
                        "expected_label": (expected_mat[r][c] if expected_mat else ref_label),
                    }
                )
        card_points[f"slot{i}"] = pts

    profile = {
        "version": 1,
        "mode": "point_exact_rgb",
        "source_image": str(image_path),
        "map_match": {
            "enum_index": map_match["enum_index"],
            "map_id": map_match["map_id"],
            "map_name_zh": map_match["map_name_zh"],
            "score": map_match["score"],
            "template_image": map_match.get("template_image", ""),
            "match_mode": str(map_match.get("match_mode", "reference_png_mask_exact_ratio")),
            "expected_cards": expected_cards,
        },
        "board_shape": [rows, cols],
        "board_grid": grid_meta,
        "card_shape": [8, 8],
        "board_points": board_points,
        "card_points": card_points,
        "card_grids": card_grids,
        "exact_rgb_rules": {
            "board": EXACT_RGB_RULES,
            "card": EXACT_CARD_RGB_RULES,
        },
    }
    if profile_out is not None:
        profile_out.parent.mkdir(parents=True, exist_ok=True)
        profile_out.write_text(json.dumps(profile, ensure_ascii=False, indent=2), encoding="utf-8")
    return profile


def eval_turn_by_profile(image_path: Path, profile: Dict) -> Dict:
    frame = cv2.imread(str(image_path))
    if frame is None:
        raise ValueError(f"cannot read image: {image_path}")

    rows, cols = int(profile["board_shape"][0]), int(profile["board_shape"][1])
    sample_radius = int(profile.get("board_grid", {}).get("sample_radius_px", 0))
    off_x, off_y, off_score = _find_best_uniform_offset(frame, profile["board_points"], sample_radius, search=6)

    board_labels = [["invalid" for _ in range(cols)] for _ in range(rows)]
    board_labels_by_rgb = [["invalid" for _ in range(cols)] for _ in range(rows)]
    board_labels_by_palette = [["invalid" for _ in range(cols)] for _ in range(rows)]
    board_colors = [[[0, 0, 0] for _ in range(cols)] for _ in range(rows)]
    board_match = 0
    board_total = 0
    h, w = frame.shape[:2]
    for p in profile["board_points"]:
        r, c = int(p["r"]), int(p["c"])
        x = int(round(float(p.get("x_norm", 0.0)) * max(1, w - 1))) + int(off_x)
        y = int(round(float(p.get("y_norm", 0.0)) * max(1, h - 1))) + int(off_y)
        if not p.get("is_valid_map_cell", True):
            board_labels[r][c] = "invalid"
            continue
        bgr = _sample_patch_bgr(frame, x, y, sample_radius) if sample_radius > 0 else _sample_bgr_exact(frame, x, y)
        board_colors[r][c] = bgr
        board_labels_by_rgb[r][c] = _exact_label_from_bgr(bgr, EXACT_RGB_RULES)
        pt_code = int(p.get("point_type_code", 0))
        board_labels_by_palette[r][c] = _board_palette_label_with_constraints(pt_code, bgr)
        ref_bgr = p.get("ref_bgr", [])
        board_total += 1
        if bgr == ref_bgr:
            board_match += 1
            board_labels[r][c] = str(p.get("ref_label", "empty"))
        else:
            board_labels[r][c] = "changed"

    card_results: Dict[str, Dict] = {}
    slot_summaries: List[Dict] = []
    expected_cards = profile.get("map_match", {}).get("expected_cards", [])
    for slot, points in profile["card_points"].items():
        mat_label = [["empty" for _ in range(8)] for _ in range(8)]
        mat_color = [[[0, 0, 0] for _ in range(8)] for _ in range(8)]
        slot_radius = int(profile.get("card_grids", {}).get(slot, {}).get("sample_radius_px", 0))
        exact_match = 0
        total = 0
        fill_colors = set()
        expected_fill_colors = set()
        for p in points:
            r, c = int(p["r"]), int(p["c"])
            x = int(round(float(p.get("x_norm", 0.0)) * max(1, w - 1)))
            y = int(round(float(p.get("y_norm", 0.0)) * max(1, h - 1)))
            bgr = _sample_patch_bgr(frame, x, y, slot_radius) if slot_radius > 0 else _sample_bgr_exact(frame, x, y)
            mat_color[r][c] = bgr
            total += 1
            ref_bgr = p.get("ref_bgr", [])
            if bgr == ref_bgr:
                exact_match += 1
                mat_label[r][c] = str(p.get("expected_label", p.get("ref_label", "empty")))
            else:
                mat_label[r][c] = "changed"
            if str(p.get("expected_label", "")) == "fill":
                fill_colors.add(tuple(int(v) for v in bgr))
                expected_fill_colors.add(tuple(int(v) for v in ref_bgr))

        confidence = float(exact_match / max(1, total))
        fill_consistent = len(fill_colors) <= 1
        fill_ref_consistent = len(expected_fill_colors) <= 1
        card_results[slot] = {
            "labels": mat_label,
            "colors_bgr": mat_color,
            "confidence": confidence,
            "fill_color_consistent": bool(fill_consistent and fill_ref_consistent),
        }
        slot_idx = int(slot.replace("slot", "")) - 1
        expect = expected_cards[slot_idx] if 0 <= slot_idx < len(expected_cards) else {"id": -1, "name_zh": "", "name_key": ""}
        slot_summaries.append(
            {
                "slot": slot,
                "card_id": int(expect.get("id", -1)),
                "card_name_zh": str(expect.get("name_zh", "")),
                "confidence": confidence,
                "fill_color_consistent": bool(fill_consistent and fill_ref_consistent),
            }
        )

    board_conf = float(board_match / max(1, board_total))
    json_match_total = 0
    json_match_hit = 0
    for p in profile["board_points"]:
        r, c = int(p["r"]), int(p["c"])
        exp = _expected_label_from_point_type(int(p.get("point_type_code", 0)))
        got = board_labels_by_palette[r][c]
        json_match_total += 1
        if got == exp:
            json_match_hit += 1
    json_match_ratio = float(json_match_hit / max(1, json_match_total))
    return {
        "image": str(image_path),
        "map_match": profile["map_match"],
        "mode": "point_exact_rgb",
        "confidence": {
            "map": float(profile.get("map_match", {}).get("score", 0.0)),
            "board": board_conf,
            "cards_min": min((float(v.get("confidence", 0.0)) for v in card_results.values()), default=0.0),
            "overall": min(
                float(profile.get("map_match", {}).get("score", 0.0)),
                board_conf,
                min((float(v.get("confidence", 0.0)) for v in card_results.values()), default=0.0),
            ),
        },
        "board": {
            "labels": board_labels,
            "labels_by_rgb": board_labels_by_rgb,
            "labels_by_palette": board_labels_by_palette,
            "colors_bgr": board_colors,
            "confidence": board_conf,
            "alignment_offset_px": [int(off_x), int(off_y)],
            "alignment_score": float(off_score),
            "json_consistency_ratio": float(json_match_ratio),
            "json_inconsistency_count": int(json_match_total - json_match_hit),
        },
        "cards": card_results,
        "cards_expected": slot_summaries,
    }


def _render_board_tokens(
    board_labels: List[List[str]],
    show_invalid_marker: bool = False,
    colorize: bool = False,
) -> List[str]:
    token = {
        "invalid": ("··" if show_invalid_marker else "  "),
        "empty": "[]",
        "p1_fill": "[]",
        "p1_special": "[]",
        "p2_fill": "[]",
        "p2_special": "[]",
        "conflict": "[]",
        "changed": "##",
        "unknown": "??",
    }
    color = {
        "empty": [0, 0, 0],
        "p1_fill": [255, 255, 0],
        "p1_special": [255, 192, 0],
        "p2_fill": [0, 112, 192],
        "p2_special": [0, 176, 240],
        "conflict": [128, 128, 128],
    }
    if not colorize:
        return ["".join(token.get(v, "??") for v in row) for row in board_labels]

    lines: List[str] = []
    for row in board_labels:
        parts: List[str] = []
        for v in row:
            t = token.get(v, "??")
            if v in color:
                parts.append(_ansi_fg(t, color[v]))
            else:
                parts.append(t)
        lines.append("".join(parts))
    return lines


def _render_card_tokens(card_labels: List[List[str]]) -> List[str]:
    out: List[str] = []
    for row in card_labels:
        chars = []
        for v in row:
            if v == "special":
                chars.append("*")
            elif v == "empty":
                chars.append(".")
            elif v == "changed":
                chars.append("#")
            elif v == "unknown":
                chars.append("?")
            else:
                chars.append("■")
        out.append("".join(chars))
    return out


def _render_cards_2x2(cards: Dict[str, Dict]) -> List[str]:
    g = [_render_card_tokens(cards[f"slot{i}"]["labels"]) for i in range(1, 5)]
    lines: List[str] = []
    for i in range(8):
        lines.append(f"{g[0][i]}  {g[1][i]}")
    lines.append("")
    for i in range(8):
        lines.append(f"{g[2][i]}  {g[3][i]}")
    return lines


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Two-stage point-based exact RGB judge for Tableturf.")
    sub = p.add_subparsers(dest="cmd", required=True)

    p_init = sub.add_parser("init", help="first-turn global map match + fixed point profile build")
    p_init.add_argument("--image", default="")
    p_init.add_argument("--input-dir", default="vision_capture/debug")
    p_init.add_argument("--layout-json", default="tableturf_vision/tableturf_layout.json")
    p_init.add_argument("--profile-out", default="tableturf_vision/point_profile.json")

    p_turn = sub.add_parser("turn", help="per-turn point sampling by profile (exact RGB)")
    p_turn.add_argument("--image", default="")
    p_turn.add_argument("--input-dir", default="vision_capture/debug")
    p_turn.add_argument("--profile", default="tableturf_vision/point_profile.json")
    p_turn.add_argument("--save-json", default="")
    p_turn.add_argument("--print-board", action="store_true", default=True)
    p_turn.add_argument("--show-invalid-marker", action="store_true", default=False)
    p_turn.add_argument("--color-board", action="store_true", default=True)
    return p.parse_args()


def main() -> int:
    args = _parse_args()
    if args.cmd == "init":
        image = _resolve_image(args.image, args.input_dir)
        layout = Path(args.layout_json)
        if not layout.is_absolute():
            layout = REPO_ROOT / layout
        out = Path(args.profile_out)
        if not out.is_absolute():
            out = REPO_ROOT / out
        profile = init_first_turn_profile(image, layout, out)
        print(json.dumps({"ok": True, "profile": str(out), "map_match": profile["map_match"]}, ensure_ascii=False, indent=2))
        return 0

    image = _resolve_image(args.image, args.input_dir)
    profile_path = Path(args.profile)
    if not profile_path.is_absolute():
        profile_path = REPO_ROOT / profile_path
    profile = json.loads(profile_path.read_text(encoding="utf-8"))
    result = eval_turn_by_profile(image, profile)

    print(f"Image: {result['image']}")
    mm = result["map_match"]
    print(f"Map: #{mm.get('enum_index')} {mm.get('map_id')} {mm.get('map_name_zh')} score={mm.get('score')}")
    print("Legend: empty(0,0,0) p1_fill(255,255,0) p1_special(255,192,0) p2_fill(0,112,192) p2_special(0,176,240) conflict(128,128,128)")
    print("[Board]")
    board_for_render = result["board"].get("labels_by_palette", result["board"].get("labels_by_rgb", result["board"]["labels"]))
    for line in _render_board_tokens(
        board_for_render,
        show_invalid_marker=bool(args.show_invalid_marker),
        colorize=bool(args.color_board),
    ):
        print(line)
    print("[Cards]")
    for line in _render_cards_2x2(result["cards"]):
        print(line)

    if args.save_json:
        out = Path(args.save_json)
        if not out.is_absolute():
            out = REPO_ROOT / out
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"saved: {out}")
    else:
        print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
