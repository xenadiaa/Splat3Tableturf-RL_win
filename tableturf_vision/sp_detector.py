from __future__ import annotations

import argparse
import glob
import json
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np


REPO_ROOT = Path(__file__).resolve().parent.parent
SP_COORD_IMAGE = REPO_ROOT / "tableturf_vision" / "参照基础_坐标点确定" / "sp_check.png"
SP_ENEMY_COORD_IMAGE = REPO_ROOT / "tableturf_vision" / "参照基础_坐标点确定" / "sp_check_enemy.png"


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


def _load_sp_reference_points(coord_image: Path, keep: str) -> List[Dict]:
    coord_img = cv2.imread(str(coord_image), cv2.IMREAD_UNCHANGED)
    if coord_img is None:
        raise ValueError(f"cannot read coordinate image: {coord_image}")

    if coord_img.shape[2] == 4:
        coord_bgr = coord_img[:, :, :3]
    else:
        coord_bgr = coord_img

    # The annotation intent is pure green. The saved PNG has antialiased edge pixels,
    # so we keep the rule strict in direction: very high G, very low R/B.
    mask = (
        (coord_bgr[:, :, 0] == 0)
        & (coord_bgr[:, :, 1] >= 242)
        & (coord_bgr[:, :, 2] <= 20)
    ).astype(np.uint8) * 255
    if keep == "bottom":
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8), iterations=1)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8), iterations=1)
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    h, w = coord_bgr.shape[:2]
    points: List[Dict] = []
    for c in cnts:
        area = cv2.contourArea(c)
        if area < 150:
            continue
        (cx, cy), radius = cv2.minEnclosingCircle(c)
        if radius < 6 or radius > 14:
            continue
        if keep == "bottom" and cy < float(h) * 0.9:
            continue
        if keep == "top" and cy > float(h) * 0.2:
            continue
        ax = float(cx)
        ay = float(cy)
        points.append(
            {
                "center": [ax, ay],
                "center_norm": [ax / float(w), ay / float(h)],
                "radius": float(radius),
                "radius_norm": float(radius / max(w, h)),
            }
        )

    points.sort(key=lambda p: p["center"][0])
    return points


@lru_cache(maxsize=1)
def load_sp_reference_points() -> List[Dict]:
    return _load_sp_reference_points(SP_COORD_IMAGE, keep="bottom")


@lru_cache(maxsize=1)
def load_enemy_sp_reference_points() -> List[Dict]:
    return _load_sp_reference_points(SP_ENEMY_COORD_IMAGE, keep="top")


def _scaled_reference_points(img_shape: Tuple[int, int, int], enemy: bool = False) -> List[Dict]:
    h, w = img_shape[:2]
    out: List[Dict] = []
    points = load_enemy_sp_reference_points() if enemy else load_sp_reference_points()
    for p in points:
        cx = float(p["center_norm"][0]) * float(w)
        cy = float(p["center_norm"][1]) * float(h)
        radius = max(4.0, float(p["radius_norm"]) * float(max(w, h)))
        out.append(
            {
                "center": [cx, cy],
                "radius": radius,
            }
        )
    return out


def _sample_patch_hsv(frame_bgr: np.ndarray, cx: float, cy: float, radius: float) -> np.ndarray:
    h, w = frame_bgr.shape[:2]
    r = max(3, int(round(radius * 0.55)))
    ix, iy = int(round(cx)), int(round(cy))
    x0, x1 = max(0, ix - r), min(w, ix + r + 1)
    y0, y1 = max(0, iy - r), min(h, iy + r + 1)
    patch = frame_bgr[y0:y1, x0:x1]
    return cv2.cvtColor(patch, cv2.COLOR_BGR2HSV)


def _is_orange_patch(frame_bgr: np.ndarray, cx: float, cy: float, radius: float) -> Tuple[bool, float]:
    hsv = _sample_patch_hsv(frame_bgr, cx, cy, radius)
    orange_mask = (
        (hsv[:, :, 0] >= 12)
        & (hsv[:, :, 0] <= 35)
        & (hsv[:, :, 1] >= 120)
        & (hsv[:, :, 2] >= 160)
    )
    ratio = float(orange_mask.mean()) if orange_mask.size else 0.0
    return (ratio >= 0.45, ratio)


def _is_enemy_cyan_patch(frame_bgr: np.ndarray, cx: float, cy: float, radius: float) -> Tuple[bool, float]:
    hsv = _sample_patch_hsv(frame_bgr, cx, cy, radius)
    cyan_mask = (
        (hsv[:, :, 0] >= 84)
        & (hsv[:, :, 0] <= 98)
        & (hsv[:, :, 1] >= 150)
        & (hsv[:, :, 2] >= 180)
    )
    ratio = float(cyan_mask.mean()) if cyan_mask.size else 0.0
    mean_bgr = frame_bgr[
        max(0, int(round(cy)) - max(3, int(round(radius * 0.55)))) : int(round(cy)) + max(3, int(round(radius * 0.55))) + 1,
        max(0, int(round(cx)) - max(3, int(round(radius * 0.55)))) : int(round(cx)) + max(3, int(round(radius * 0.55))) + 1,
    ].reshape(-1, 3).mean(axis=0)
    mean_hsv = cv2.cvtColor(np.uint8([[mean_bgr.astype(np.uint8)]]), cv2.COLOR_BGR2HSV)[0, 0]
    mh, ms, mv = int(mean_hsv[0]), int(mean_hsv[1]), int(mean_hsv[2])
    mb, mg, mr = [float(v) for v in mean_bgr.tolist()]
    covered_active = bool(
        84 <= mh <= 95
        and 45 <= ms <= 120
        and mv >= 190
        and mb >= 190
        and mg >= 190
        and 130 <= mr <= 190
        and (mb - mr) >= 25
        and (mg - mr) >= 25
    )
    return (ratio >= 0.45 or covered_active, ratio)


def _detect_sp_points(frame_bgr: np.ndarray, enemy: bool = False) -> Dict:
    ref_points = _scaled_reference_points(frame_bgr.shape, enemy=enemy)
    sampled: List[Dict] = []
    for idx, p in enumerate(ref_points, start=1):
        if enemy:
            active, color_ratio = _is_enemy_cyan_patch(frame_bgr, p["center"][0], p["center"][1], p["radius"])
        else:
            active, color_ratio = _is_orange_patch(frame_bgr, p["center"][0], p["center"][1], p["radius"])
        sampled.append(
            {
                "index": idx,
                "center": [round(float(p["center"][0]), 2), round(float(p["center"][1]), 2)],
                "radius": round(float(p["radius"]), 2),
                ("cyan_ratio" if enemy else "orange_ratio"): color_ratio,
                "active": bool(active),
            }
        )

    sp_count = 0
    for row in sampled:
        if not row["active"]:
            break
        sp_count += 1

    return {
        "sp_count": int(sp_count),
        "active_indices": [row["index"] for row in sampled if row["index"] <= sp_count],
        "reference_point_count": len(sampled),
        "side": ("enemy" if enemy else "self"),
        "points": sampled,
    }


def detect_sp_points(frame_bgr: np.ndarray) -> Dict:
    return _detect_sp_points(frame_bgr, enemy=False)


def detect_enemy_sp_points(frame_bgr: np.ndarray) -> Dict:
    return _detect_sp_points(frame_bgr, enemy=True)


def get_sp_count_frame(frame_bgr: np.ndarray) -> int:
    return int(detect_sp_points(frame_bgr)["sp_count"])


def get_enemy_sp_count_frame(frame_bgr: np.ndarray) -> int:
    return int(detect_enemy_sp_points(frame_bgr)["sp_count"])


def get_sp_count_image_path(image_path: Path) -> int:
    frame = cv2.imread(str(image_path))
    if frame is None:
        raise ValueError(f"cannot read image: {image_path}")
    return get_sp_count_frame(frame)


def get_enemy_sp_count_image_path(image_path: Path) -> int:
    frame = cv2.imread(str(image_path))
    if frame is None:
        raise ValueError(f"cannot read image: {image_path}")
    return get_enemy_sp_count_frame(frame)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Detect current SP count from the bottom-left SP strip.")
    p.add_argument("--image", default="", help="image path; empty means latest capture image")
    p.add_argument("--input-dir", default="vision_capture/debug")
    p.add_argument("--json", action="store_true", help="print full JSON result")
    p.add_argument("--show-reference-points", action="store_true", help="print extracted green-circle reference points")
    p.add_argument("--enemy", action="store_true", help="use enemy SP reference points and cyan detection")
    return p.parse_args()


def main() -> int:
    args = _parse_args()
    if args.show_reference_points:
        points = load_enemy_sp_reference_points() if args.enemy else load_sp_reference_points()
        payload = {
            "count": len(points),
            "side": ("enemy" if args.enemy else "self"),
            "points": [
                {
                    "index": i + 1,
                    "center": [round(float(p["center"][0]), 2), round(float(p["center"][1]), 2)],
                    "radius": round(float(p["radius"]), 2),
                }
                for i, p in enumerate(points)
            ],
        }
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return 0

    image = _resolve_image(args.image, args.input_dir)
    frame = cv2.imread(str(image))
    if frame is None:
        raise ValueError(f"cannot read image: {image}")
    result = detect_enemy_sp_points(frame) if args.enemy else detect_sp_points(frame)
    result["image"] = str(image)
    if args.json:
        print(json.dumps(result, ensure_ascii=False, indent=2))
    else:
        print(result["sp_count"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
