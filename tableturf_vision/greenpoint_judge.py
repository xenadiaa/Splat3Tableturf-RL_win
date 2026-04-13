from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
MAP_INFO_PATH = REPO_ROOT / "tableturf_sim" / "data" / "maps" / "MiniGameMapInfo.json"

# RGB rules requested by user.
RGB_RULES = {
    "empty": [0, 0, 0],
    "p1_fill": [255, 255, 0],
    "p1_special": [255, 192, 0],
    "p2_fill": [0, 112, 192],
    "p2_special": [0, 176, 240],
    "conflict": [128, 128, 128],
}


def _load_map_info() -> Dict[str, Dict]:
    rows = json.loads(MAP_INFO_PATH.read_text(encoding="utf-8"))
    return {str(r["name"]): r for r in rows}


def _point_type_to_expected(pt: int) -> str:
    return {
        0: "invalid",
        1: "empty",
        3: "p1_fill",
        5: "p2_fill",
        7: "conflict",
        11: "p1_special",
        13: "p2_special",
    }.get(int(pt), "unknown")


def _detect_red_rect(annotated: np.ndarray) -> Tuple[int, int, int, int]:
    hsv = cv2.cvtColor(annotated, cv2.COLOR_BGR2HSV)
    m1 = cv2.inRange(hsv, np.array([0, 80, 80]), np.array([12, 255, 255]))
    m2 = cv2.inRange(hsv, np.array([168, 80, 80]), np.array([180, 255, 255]))
    mask = cv2.bitwise_or(m1, m2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8), iterations=2)
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        raise ValueError("cannot detect red rectangle")
    c = max(cnts, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(c)
    return int(x), int(y), int(w), int(h)


def _detect_green_points(annotated: np.ndarray) -> List[Tuple[float, float, float]]:
    hsv = cv2.cvtColor(annotated, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, np.array([35, 70, 70]), np.array([95, 255, 255]))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8), iterations=1)
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    points: List[Tuple[float, float, float]] = []
    for c in cnts:
        area = cv2.contourArea(c)
        if area < 20:
            continue
        (cx, cy), r = cv2.minEnclosingCircle(c)
        if r < 2 or r > 40:
            continue
        points.append((float(cx), float(cy), float(r)))
    points.sort(key=lambda t: (t[1], t[0]))
    return points


def _nearest_label_rgb(rgb: np.ndarray) -> str:
    names = list(RGB_RULES.keys())
    vals = np.array([RGB_RULES[k] for k in names], dtype=np.float32)
    d = ((vals - rgb[None, :]) ** 2).sum(axis=1)
    return names[int(np.argmin(d))]


def _sample_circle_rgb(image_bgr: np.ndarray, x: float, y: float, radius: float) -> np.ndarray:
    h, w = image_bgr.shape[:2]
    r = max(2, int(round(radius * 0.45)))
    cx, cy = int(round(x)), int(round(y))
    x0, x1 = max(0, cx - r), min(w, cx + r + 1)
    y0, y1 = max(0, cy - r), min(h, cy + r + 1)
    patch = image_bgr[y0:y1, x0:x1]
    if patch.size == 0:
        bgr = image_bgr[max(0, min(cy, h - 1)), max(0, min(cx, w - 1))]
        return np.array([float(bgr[2]), float(bgr[1]), float(bgr[0])], dtype=np.float32)
    yy, xx = np.ogrid[: patch.shape[0], : patch.shape[1]]
    mask = (xx - (cx - x0)) ** 2 + (yy - (cy - y0)) ** 2 <= r * r
    pix = patch[mask]
    if pix.size == 0:
        pix = patch.reshape(-1, 3)
    rgb = pix[:, ::-1].mean(axis=0)
    return rgb.astype(np.float32)


def judge_with_green_points(map_name: str, annotated_path: Path, target_path: Path) -> Dict:
    maps = _load_map_info()
    if map_name not in maps:
        raise ValueError(f"map not found in json: {map_name}")
    m = maps[map_name]
    width, height = int(m["width"]), int(m["height"])
    ptype = np.array(m["point_type"], dtype=np.int32)

    annotated = cv2.imread(str(annotated_path))
    target = cv2.imread(str(target_path))
    if annotated is None:
        raise ValueError(f"cannot read annotated image: {annotated_path}")
    if target is None:
        raise ValueError(f"cannot read target image: {target_path}")

    x, y, w, h = _detect_red_rect(annotated)
    points = _detect_green_points(annotated)

    centers = np.zeros((height, width, 2), dtype=np.float32)
    for r in range(height):
        for c in range(width):
            centers[r, c, 0] = x + (c + 0.5) * w / width
            centers[r, c, 1] = y + (r + 0.5) * h / height

    used = set()
    scored = 0
    hit = 0
    mismatches = []
    for gx, gy, gr in points:
        d = (centers[:, :, 0] - gx) ** 2 + (centers[:, :, 1] - gy) ** 2
        rr, cc = np.unravel_index(np.argmin(d), d.shape)
        key = (int(rr), int(cc))
        if key in used:
            continue
        used.add(key)
        expected = _point_type_to_expected(int(ptype[rr, cc]))
        if expected == "invalid":
            continue
        rgb = _sample_circle_rgb(target, gx, gy, gr)
        got = _nearest_label_rgb(rgb)
        scored += 1
        if got == expected:
            hit += 1
        else:
            mismatches.append(
                {
                    "r": int(rr),
                    "c": int(cc),
                    "expected": expected,
                    "got": got,
                    "sample_rgb": [float(rgb[0]), float(rgb[1]), float(rgb[2])],
                }
            )

    return {
        "map_name_zh": map_name,
        "annotated_path": str(annotated_path),
        "target_path": str(target_path),
        "red_rect": [x, y, w, h],
        "green_points": len(points),
        "mapped_cells": len(used),
        "scored_cells": scored,
        "hit_cells": hit,
        "confidence": float(hit / max(1, scored)),
        "mismatch_count": len(mismatches),
        "mismatches": mismatches[:50],
    }


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Judge map by user green-point annotations.")
    p.add_argument("--map-name", required=True, help="Chinese map name, e.g. 间隔墙")
    p.add_argument("--annotated", default="", help="path to *_标注.jpg")
    p.add_argument("--target", default="", help="path to image to judge; default map image")
    p.add_argument("--debug-dir", default="vision_capture/debug")
    p.add_argument("--save-json", default="")
    return p.parse_args()


def main() -> int:
    args = _parse_args()
    debug_dir = Path(args.debug_dir)
    if not debug_dir.is_absolute():
        debug_dir = REPO_ROOT / debug_dir
    annotated = Path(args.annotated) if args.annotated else debug_dir / f"{args.map_name}_标注.jpg"
    target = Path(args.target) if args.target else debug_dir / f"{args.map_name}.jpg"
    if not annotated.is_absolute():
        annotated = REPO_ROOT / annotated
    if not target.is_absolute():
        target = REPO_ROOT / target

    result = judge_with_green_points(args.map_name, annotated, target)
    if args.save_json:
        out = Path(args.save_json)
        if not out.is_absolute():
            out = REPO_ROOT / out
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"saved: {out}")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
