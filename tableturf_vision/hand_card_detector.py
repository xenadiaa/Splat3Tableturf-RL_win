from __future__ import annotations

import argparse
import json
import sys
from functools import lru_cache
from pathlib import Path
from typing import Dict, List

import cv2
import numpy as np


REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tableturf_vision.tableturf_mapper import DEFAULT_LAYOUT, _card_special_score

CARD_REFERENCE_IMAGES = [
    REPO_ROOT / "tableturf_vision" / "参照基础" / "卡牌识别_可用.png",
    REPO_ROOT / "tableturf_vision" / "参照基础" / "卡牌识别_不可用.png",
]
CARD_COORD_IMAGES = [
    REPO_ROOT / "tableturf_vision" / "参照基础_坐标点确定" / "卡牌识别_可用.png",
    REPO_ROOT / "tableturf_vision" / "参照基础_坐标点确定" / "卡牌识别_不可用.png",
]
SLOT_NAMES = ["left_top", "right_top", "left_bottom", "right_bottom"]


def _extract_pure_green_points(image_path: Path) -> List[Dict]:
    img = cv2.imread(str(image_path), cv2.IMREAD_UNCHANGED)
    if img is None:
        raise ValueError(f"cannot read coordinate image: {image_path}")
    bgr = img[:, :, :3] if img.shape[2] == 4 else img
    mask = (
        (bgr[:, :, 0] == 0)
        & (bgr[:, :, 1] >= 242)
        & (bgr[:, :, 2] <= 20)
    ).astype(np.uint8) * 255
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    h, w = bgr.shape[:2]
    points: List[Dict] = []
    for c in cnts:
        area = cv2.contourArea(c)
        if area < 100:
            continue
        (cx, cy), radius = cv2.minEnclosingCircle(c)
        if radius < 5 or radius > 18:
            continue
        points.append(
            {
                "center": [float(cx), float(cy)],
                "center_norm": [float(cx) / float(w), float(cy) / float(h)],
                "radius": float(radius),
                "radius_norm": float(radius / max(w, h)),
            }
        )
    points.sort(key=lambda p: (p["center"][1], p["center"][0]))
    return points


def _roi_from_norm(img_shape: tuple[int, int, int], roi_norm: List[float]) -> tuple[int, int, int, int]:
    h, w = img_shape[:2]
    x = int(w * float(roi_norm[0]))
    y = int(h * float(roi_norm[1]))
    rw = int(w * float(roi_norm[2]))
    rh = int(h * float(roi_norm[3]))
    return x, y, rw, rh


def _sample_patch_mean(frame_bgr: np.ndarray, cx: float, cy: float, radius: float) -> np.ndarray:
    h, w = frame_bgr.shape[:2]
    r = max(2, int(round(radius * 0.55)))
    ix, iy = int(round(cx)), int(round(cy))
    x0, x1 = max(0, ix - r), min(w, ix + r + 1)
    y0, y1 = max(0, iy - r), min(h, iy + r + 1)
    patch = frame_bgr[y0:y1, x0:x1]
    if patch.size == 0:
        return np.array([0.0, 0.0, 0.0], dtype=np.float32)
    return patch.reshape(-1, 3).mean(axis=0).astype(np.float32)


def _classify_hand_card_point(mean_bgr: np.ndarray) -> str:
    hsv = cv2.cvtColor(np.uint8([[mean_bgr.astype(np.uint8)]]), cv2.COLOR_BGR2HSV)[0, 0]
    h, s, v = int(hsv[0]), int(hsv[1]), int(hsv[2])

    # Tight thresholds learned from the 8 labeled test cards:
    # fill: high-sat yellow/olive, special: high-sat orange.
    if s >= 220 and v >= 120:
        if 23 <= h <= 27:
            return "special"
        if 28 <= h <= 36:
            return "fill"
    return "empty"


@lru_cache(maxsize=1)
def load_hand_card_reference_points() -> Dict[str, List[Dict]]:
    ref_img = cv2.imread(str(CARD_REFERENCE_IMAGES[0]))
    if ref_img is None:
        raise ValueError(f"cannot read reference image: {CARD_REFERENCE_IMAGES[0]}")

    all_point_sets = [_extract_pure_green_points(path) for path in CARD_COORD_IMAGES]
    baseline_points = max(all_point_sets, key=len)
    if len(baseline_points) != 256:
        raise ValueError(f"expected 256 hand-card reference points, got {len(baseline_points)}")

    slot_points: Dict[str, List[Dict]] = {}
    for slot_name, roi_norm in zip(SLOT_NAMES, DEFAULT_LAYOUT["card_rois_norm"]):
        x, y, w, h = _roi_from_norm(ref_img.shape, roi_norm)
        pts = [
            p for p in baseline_points
            if x <= p["center"][0] <= x + w and y <= p["center"][1] <= y + h
        ]
        if len(pts) != 64:
            raise ValueError(f"{slot_name} expected 64 points, got {len(pts)}")

        pts.sort(key=lambda p: (p["center"][1], p["center"][0]))
        mapped: List[Dict] = []
        for row_idx in range(8):
            row_points = sorted(pts[row_idx * 8 : (row_idx + 1) * 8], key=lambda p: p["center"][0])
            for col_idx, point in enumerate(row_points):
                item = dict(point)
                item["grid_row"] = row_idx
                item["grid_col"] = col_idx
                mapped.append(item)
        slot_points[slot_name] = mapped
    return slot_points


def detect_hand_cards(frame_bgr: np.ndarray) -> Dict:
    refs = load_hand_card_reference_points()
    slots_payload = []

    for slot_name in SLOT_NAMES:
        slot_ref_points = refs[slot_name]
        cells: List[Dict] = []
        special_scores: List[tuple[float, int]] = []

        for idx, point in enumerate(slot_ref_points):
            cx = float(point["center_norm"][0]) * float(frame_bgr.shape[1])
            cy = float(point["center_norm"][1]) * float(frame_bgr.shape[0])
            radius = max(4.0, float(point["radius_norm"]) * float(max(frame_bgr.shape[:2])))
            mean_bgr = _sample_patch_mean(frame_bgr, cx, cy, radius)
            label = _classify_hand_card_point(mean_bgr)
            score = _card_special_score(mean_bgr)
            cells.append(
                {
                    "index": idx + 1,
                    "grid_row": int(point["grid_row"]),
                    "grid_col": int(point["grid_col"]),
                    "center": [round(float(cx), 2), round(float(cy), 2)],
                    "radius": round(float(radius), 2),
                    "mean_bgr": [round(float(v), 2) for v in mean_bgr.tolist()],
                    "label": label,
                    "special_score": round(float(score), 4),
                }
            )
            if label == "special":
                special_scores.append((float(score), idx))

        for cell in cells:
            if cell["label"] == "special":
                cell["label"] = "fill"
        if special_scores:
            best_score, best_idx = max(special_scores, key=lambda item: item[0])
            if best_score >= 10.0:
                cells[best_idx]["label"] = "special"

        matrix = [["empty" for _ in range(8)] for _ in range(8)]
        counts = {"fill": 0, "special": 0, "empty": 0}
        for cell in cells:
            label = str(cell["label"])
            matrix[int(cell["grid_row"])][int(cell["grid_col"])] = label
            counts[label] += 1

        slots_payload.append(
            {
                "slot": slot_name,
                "counts": counts,
                "matrix": matrix,
                "cells": cells,
            }
        )

    return {"slots": slots_payload}


def detect_hand_cards_image_path(image_path: Path) -> Dict:
    frame = cv2.imread(str(image_path))
    if frame is None:
        raise ValueError(f"cannot read image: {image_path}")
    result = detect_hand_cards(frame)
    result["image"] = str(image_path)
    return result


def _resolve_image(image: str) -> Path:
    p = Path(image)
    if not p.is_absolute():
        p = REPO_ROOT / p
    return p


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Detect the 4 hand cards as 8x8 matrices.")
    p.add_argument("--image", required=False, default="", help="image path")
    p.add_argument("--json", action="store_true", help="print full JSON")
    p.add_argument("--show-reference-points", action="store_true", help="print the 4x64 pure-green points")
    return p.parse_args()


def main() -> int:
    args = _parse_args()
    if args.show_reference_points:
        refs = load_hand_card_reference_points()
        payload = {}
        for slot_name in SLOT_NAMES:
            payload[slot_name] = {
                "count": len(refs[slot_name]),
                "points": [
                    {
                        "index": idx + 1,
                        "grid_row": int(point["grid_row"]),
                        "grid_col": int(point["grid_col"]),
                        "center": [round(float(point["center"][0]), 2), round(float(point["center"][1]), 2)],
                        "radius": round(float(point["radius"]), 2),
                    }
                    for idx, point in enumerate(refs[slot_name])
                ],
            }
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return 0

    if not args.image:
        raise ValueError("--image is required unless --show-reference-points is used")

    result = detect_hand_cards_image_path(_resolve_image(args.image))
    if args.json:
        print(json.dumps(result, ensure_ascii=False, indent=2))
    else:
        payload = {slot["slot"]: slot["matrix"] for slot in result["slots"]}
        print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
