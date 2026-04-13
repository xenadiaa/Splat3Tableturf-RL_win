from __future__ import annotations

import argparse
import glob
import json
import sys
import time
import urllib.parse
import urllib.request
from datetime import datetime
from functools import lru_cache
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

import cv2
import numpy as np


REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

MAP_REFERENCE_DIR = REPO_ROOT / "tableturf_vision" / "参照基础"
MAP_COORD_DIR = REPO_ROOT / "tableturf_vision" / "参照基础_坐标点确定"
MAP_STATE_ERROR_DIR = REPO_ROOT / "tableturf_vision" / "error"
MAP_NAMES = [
    "加速高速公路",
    "双子岛",
    "小巧竞技场",
    "扭转河",
    "正方广场",
    "正直大道",
    "清脆饼干",
    "漆黑深林",
    "罚分花园",
    "轻飘湖",
    "酷似大道",
    "钢骨建筑",
    "间隔墙",
    "雷霆车站",
    "面具屋",
]
MAP_INFO_PATH = REPO_ROOT / "tableturf_sim" / "data" / "maps" / "MiniGameMapInfo.json"

CLR_RESET = "\033[0m"
RGB = {
    "transparent": (0, 0, 0),
    "p1_fill": (255, 255, 0),
    "p1_special": (255, 192, 0),
    "p2_fill": (0, 112, 192),
    "p2_special": (0, 176, 240),
    "conflict": (200, 200, 200),
    "invalid": (80, 80, 80),
}
TOKEN_MAP = {
    "invalid": "  ",
    "transparent": "[]",
    "p1_fill": "[]",
    "p1_special": "[]",
    "p2_fill": "[]",
    "p2_special": "[]",
    "conflict": "[]",
}

CellPos = Tuple[int, int]


_NO_PROXY_OPENER = urllib.request.build_opener(urllib.request.ProxyHandler({}))


def _urlopen_local_direct(req_or_url: Any, timeout: float):
    url = req_or_url.full_url if isinstance(req_or_url, urllib.request.Request) else str(req_or_url)
    parsed = urllib.parse.urlparse(str(url))
    host = str(parsed.hostname or "").strip().lower()
    if host in {"127.0.0.1", "localhost", "::1"}:
        return _NO_PROXY_OPENER.open(req_or_url, timeout=timeout)
    return urllib.request.urlopen(req_or_url, timeout=timeout)


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


def _coord_image_path(map_name: str) -> Path:
    return MAP_COORD_DIR / f"{map_name}.png"


def _reference_image_path(map_name: str) -> Path:
    return MAP_REFERENCE_DIR / f"{map_name}.png"


@lru_cache(maxsize=1)
def _load_map_info() -> Dict[str, Dict]:
    rows = json.loads(MAP_INFO_PATH.read_text(encoding="utf-8"))
    return {str(row["name"]): row for row in rows}


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


def _map_nonzero_columns(map_name: str) -> List[List[int]]:
    map_info = _load_map_info().get(map_name)
    if map_info is None:
        raise ValueError(f"map info not found: {map_name}")
    rows: List[List[int]] = []
    for row in map_info["point_type"]:
        rows.append([col_idx for col_idx, value in enumerate(row) if int(value) != 0])
    return rows


@lru_cache(maxsize=None)
def load_map_reference_points(map_name: str) -> List[Dict]:
    if map_name not in MAP_NAMES:
        raise ValueError(f"unsupported map: {map_name}")
    raw_points = _extract_pure_green_points(_coord_image_path(map_name))
    row_columns = _map_nonzero_columns(map_name)
    expected_count = sum(len(cols) for cols in row_columns)
    if len(raw_points) != expected_count:
        raise ValueError(
            f"reference point count mismatch for {map_name}: "
            f"green_points={len(raw_points)} json_nonzero={expected_count}"
        )

    mapped_points: List[Dict] = []
    offset = 0
    for row_idx, columns in enumerate(row_columns):
        row_count = len(columns)
        row_points = raw_points[offset : offset + row_count]
        if len(row_points) != row_count:
            raise ValueError(
                f"row point count mismatch for {map_name} row {row_idx}: "
                f"got={len(row_points)} expected={row_count}"
            )
        row_points = sorted(row_points, key=lambda p: p["center"][0])
        for col_idx, point in zip(columns, row_points):
            mapped = dict(point)
            mapped["json_row"] = row_idx
            mapped["json_col"] = col_idx
            mapped_points.append(mapped)
        offset += row_count
    return mapped_points


def load_all_map_reference_points() -> Dict[str, List[Dict]]:
    return {name: load_map_reference_points(name) for name in MAP_NAMES}


def _scaled_reference_points(map_name: str, img_shape: tuple[int, int, int]) -> List[Dict]:
    h, w = img_shape[:2]
    out: List[Dict] = []
    for p in load_map_reference_points(map_name):
        cx = float(p["center_norm"][0]) * float(w)
        cy = float(p["center_norm"][1]) * float(h)
        radius = max(4.0, float(p["radius_norm"]) * float(max(w, h)))
        out.append(
            {
                "center": [cx, cy],
                "radius": radius,
                "json_row": int(p["json_row"]),
                "json_col": int(p["json_col"]),
            }
        )
    return out


def _sample_patch_mean_bgr(frame_bgr: np.ndarray, cx: float, cy: float, radius: float) -> np.ndarray:
    h, w = frame_bgr.shape[:2]
    r = max(3, int(round(radius * 0.6)))
    ix, iy = int(round(cx)), int(round(cy))
    x0, x1 = max(0, ix - r), min(w, ix + r + 1)
    y0, y1 = max(0, iy - r), min(h, iy + r + 1)
    patch = frame_bgr[y0:y1, x0:x1]
    if patch.size == 0:
        return np.array([0.0, 0.0, 0.0], dtype=np.float32)
    ph, pw = patch.shape[:2]
    # Use a vertical sliding gradient: strongest at the top edge, linearly
    # down to 0.3 at the midline, then continue decaying toward the bottom.
    # This keeps the tile body dominant while strongly suppressing rising
    # effects from the lower edge.
    row_idx = np.arange(ph, dtype=np.float32)
    if ph == 1:
        weights_1d = np.array([1.0], dtype=np.float32)
    else:
        mid = max(1.0, (ph - 1) / 2.0)
        top_to_mid = np.clip(row_idx / mid, 0.0, 1.0)
        bot_span = max(1.0, (ph - 1) - mid)
        mid_to_bot = np.clip((row_idx - mid) / bot_span, 0.0, 1.0)
        weights_1d = np.where(
            row_idx <= mid,
            1.0 - 0.7 * top_to_mid,
            0.3 - 0.22 * mid_to_bot,
        ).astype(np.float32)
    row_weights = weights_1d.reshape(ph, 1, 1)
    weighted_patch = patch.astype(np.float32) * row_weights
    mean_bgr = weighted_patch.sum(axis=(0, 1)) / (row_weights.sum() * float(pw))
    return mean_bgr.astype(np.float32)


def _classify_cell(mean_bgr: np.ndarray) -> tuple[str, Dict[str, float], bool]:
    hsv = cv2.cvtColor(np.uint8([[mean_bgr.astype(np.uint8)]]), cv2.COLOR_BGR2HSV)[0, 0]
    h, s, v = int(hsv[0]), int(hsv[1]), int(hsv[2])
    b, g, r = [float(x) for x in mean_bgr.tolist()]

    scores = {
        "p1_fill": 0.0,
        "p2_fill": 0.0,
        "p1_special": 0.0,
        "p2_special": 0.0,
        "conflict": 0.0,
        "transparent": 0.0,
    }

    # Transparent board cells are very dark patterned tiles.
    # Recognize them explicitly so that any other uncaptured color state can be
    # treated conservatively as conflict/error instead of a playable empty cell.
    if v <= 52 or (v <= 72 and max(b, g, r) <= 78):
        scores["transparent"] = 2.0 + ((72 - min(v, 72)) / 255.0)

    if 28 <= h <= 42 and s >= 150 and v >= 180:
        scores["p1_fill"] = 1.0 + (v / 255.0)
    if (
        96 <= h <= 135
        and s >= 95
        and 80 <= v <= 255
        and b >= 90
        and b > g + 30
        and g >= 35
        and g >= r - 10
        and (b - r) >= 35
    ):
        # Enemy fill is a deeper blue than enemy special.
        # Keep it bright enough to avoid transparent board cells, but allow
        # the hue/value drift seen in real captures.
        scores["p2_fill"] = 1.0 + (s / 255.0) + min(0.35, (b - g) / 255.0)
    if 12 <= h <= 27 and s >= 140 and v >= 150:
        scores["p1_special"] = 1.0 + (v / 255.0)
    if (
        29 <= h <= 30
        and s >= 235
        and v >= 245
        and b >= 10
        and g >= 245
        and r >= 250
        and abs(g - r) <= 8
    ):
        # Activated own special can shift into the yellow fill hue band.
        # Keep this rule narrow so ordinary fill cells are not broadly affected.
        scores["p1_special"] = max(
            scores["p1_special"],
            1.15 + (v / 255.0) + 0.1,
        )
    if 80 <= h <= 102 and s >= 90 and v >= 180:
        scores["p2_special"] = 1.0 + (v / 255.0)
    if (
        86 <= h <= 94
        and 8 <= s <= 45
        and v >= 245
        and b >= 245
        and g >= 245
        and 210 <= r <= 242
        and (b - r) >= 10
        and (g - r) >= 10
    ):
        # Activated enemy special can wash out into a bright cyan-white.
        scores["p2_special"] = max(
            scores["p2_special"],
            1.12 + (v / 255.0),
        )
    if s <= 45 and v >= 150:
        scores["conflict"] = 1.0 + (v / 255.0)

    is_error = False
    if max(scores.values()) <= 0.0:
        scores["conflict"] = 0.95
        is_error = True

    priority = ["p1_fill", "p2_fill", "p1_special", "p2_special", "conflict", "transparent"]
    label = max(priority, key=lambda name: (scores[name], -priority.index(name)))
    return label, scores, is_error


def detect_map_state(frame_bgr: np.ndarray, map_name: str) -> Dict:
    if map_name not in MAP_NAMES:
        raise ValueError(f"unsupported map: {map_name}")
    ref_points = _scaled_reference_points(map_name, frame_bgr.shape)
    cells: List[Dict] = []
    counts = {
        "p1_fill": 0,
        "p2_fill": 0,
        "p1_special": 0,
        "p2_special": 0,
        "conflict": 0,
        "transparent": 0,
    }
    error_cells: List[Dict] = []

    for idx, p in enumerate(ref_points, start=1):
        mean_bgr = _sample_patch_mean_bgr(frame_bgr, p["center"][0], p["center"][1], p["radius"])
        label, scores, is_error = _classify_cell(mean_bgr)
        counts[label] += 1
        cell = {
            "index": idx,
            "center": [round(float(p["center"][0]), 2), round(float(p["center"][1]), 2)],
            "radius": round(float(p["radius"]), 2),
            "json_row": int(p["json_row"]),
            "json_col": int(p["json_col"]),
            "mean_bgr": [round(float(v), 2) for v in mean_bgr.tolist()],
            "label": label,
            "scores": {k: round(float(v), 4) for k, v in scores.items()},
            "is_error": bool(is_error),
        }
        cells.append(cell)
        if is_error:
            error_cells.append(
                {
                    "index": idx,
                    "json_row": int(p["json_row"]),
                    "json_col": int(p["json_col"]),
                    "center": [round(float(p["center"][0]), 2), round(float(p["center"][1]), 2)],
                    "mean_bgr": [round(float(v), 2) for v in mean_bgr.tolist()],
                }
            )

    return {
        "map_name": map_name,
        "reference_point_count": len(cells),
        "counts": counts,
        "error_count": len(error_cells),
        "error_cells": error_cells,
        "cells": cells,
    }


def _cell_vote_key(cell: Dict) -> CellPos:
    return (int(cell["json_row"]), int(cell["json_col"]))


def _save_unknown_combo_debug_frame(
    frame_bgr: np.ndarray,
    map_name: str,
    unknown_combos: List[Dict],
    resolved_cells: Optional[Dict[CellPos, Dict[str, Any]]] = None,
) -> Optional[str]:
    try:
        MAP_STATE_ERROR_DIR.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        stem = f"map_state_unknown_combo_{map_name}_{stamp}"
        image_path = MAP_STATE_ERROR_DIR / f"{stem}.png"
        meta_path = MAP_STATE_ERROR_DIR / f"{stem}.json"
        doc_path = MAP_STATE_ERROR_DIR / f"{stem}.md"
        cv2.imwrite(str(image_path), frame_bgr)
        resolved_cells = resolved_cells or {}
        meta_path.write_text(
            json.dumps(
                {
                    "timestamp": stamp,
                    "map_name": map_name,
                    "unknown_combo_count": len(unknown_combos),
                    "unknown_combos": unknown_combos,
                    "image": str(image_path),
                    "document": str(doc_path),
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        lines = [
            f"# Unknown Combo Record",
            "",
            f"- timestamp: `{stamp}`",
            f"- map_name: `{map_name}`",
            f"- image: `{image_path}`",
            f"- combo_count: `{len(unknown_combos)}`",
            "",
            "## Cells",
        ]
        for idx, combo in enumerate(unknown_combos, start=1):
            key = (int(combo["json_row"]), int(combo["json_col"]))
            resolved = resolved_cells.get(key, {})
            lines.extend(
                [
                    "",
                    f"### {idx}. cell ({key[0]}, {key[1]})",
                    f"- raw_labels: `{', '.join(combo.get('raw_labels', []))}`",
                    f"- reduced_labels: `{', '.join(combo.get('reduced_labels', []))}`",
                    f"- distribution: `{json.dumps(combo.get('distribution', {}), ensure_ascii=False)}`",
                    f"- below_label: `{combo.get('below_label', '')}`",
                    f"- final_label: `{resolved.get('label', '')}`",
                    f"- vote_summary: `{json.dumps(resolved.get('vote_summary', {}), ensure_ascii=False)}`",
                ]
            )
        doc_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        return str(image_path)
    except Exception:
        return None


def _resolve_downward_combo(
    raw_labels: Set[str],
    below_label: str,
) -> Optional[str]:
    if "p1_special" in raw_labels and "conflict" in raw_labels:
        if below_label == "p1_special":
            return "conflict"
        if below_label == "p2_special":
            return "p1_special"
        if below_label == "conflict":
            return "conflict"
    return None


def _merge_state_results_from_frames(frame_results: List[Dict], frames_bgr: List[np.ndarray], map_name: str) -> Dict:
    if not frame_results:
        raise ValueError("frame_results must not be empty")
    cell_keys = [_cell_vote_key(cell) for cell in frame_results[0]["cells"]]
    initial_cells: List[Dict] = []

    for key in cell_keys:
        samples = [next(cell for cell in result["cells"] if _cell_vote_key(cell) == key) for result in frame_results]
        vote_count: Dict[str, int] = {}
        score_sum: Dict[str, float] = {}
        error_votes = 0
        for sample in samples:
            label = str(sample["label"])
            vote_count[label] = vote_count.get(label, 0) + 1
            score_sum[label] = score_sum.get(label, 0.0) + float(sample["scores"].get(label, 0.0) or 0.0)
            if bool(sample.get("is_error")):
                error_votes += 1

        chosen_label = max(
            vote_count.keys(),
            key=lambda label: (
                int(vote_count[label]),
                float(score_sum.get(label, 0.0)),
                -["p1_fill", "p2_fill", "p1_special", "p2_special", "conflict", "transparent"].index(label),
            ),
        )
        chosen_sample = max(
            samples,
            key=lambda sample: (
                str(sample["label"]) == chosen_label,
                float(sample["scores"].get(chosen_label, 0.0) or 0.0),
            ),
        )
        merged = {
            "index": int(chosen_sample["index"]),
            "center": list(chosen_sample["center"]),
            "radius": float(chosen_sample["radius"]),
            "json_row": int(chosen_sample["json_row"]),
            "json_col": int(chosen_sample["json_col"]),
            "mean_bgr": [
                round(float(sum(float(sample["mean_bgr"][idx]) for sample in samples) / len(samples)), 2)
                for idx in range(3)
            ],
            "label": chosen_label,
            "scores": {k: round(float(sum(float(sample["scores"].get(k, 0.0) or 0.0) for sample in samples)), 4) for k in chosen_sample["scores"].keys()},
            "is_error": bool(error_votes > (len(samples) // 2)),
            "vote_summary": {
                "label_votes": {k: int(v) for k, v in sorted(vote_count.items())},
                "error_votes": int(error_votes),
                "frame_count": len(samples),
            },
            "raw_labels": sorted(vote_count.keys()),
        }
        initial_cells.append(merged)

    base_cell_map = {
        (int(cell["json_row"]), int(cell["json_col"])): cell
        for cell in initial_cells
    }
    working_cells: Dict[CellPos, Dict[str, Any]] = {
        key: dict(value)
        for key, value in base_cell_map.items()
    }
    known_combo_corrections: List[Dict] = []

    # Phase 1: direct reductions that do not depend on lower-neighbor state.
    for key in cell_keys:
        cell = dict(working_cells[key])
        raw_labels = set(str(label) for label in cell.get("raw_labels", []))
        reduced_labels = set(raw_labels)
        reductions: List[str] = []
        if "p2_special" in reduced_labels and "conflict" in reduced_labels:
            reduced_labels.discard("conflict")
            reductions.append("p2_special + conflict => p2_special")
        if "p2_fill" in reduced_labels and "conflict" in reduced_labels:
            reduced_labels.discard("conflict")
            reductions.append("p2_fill + conflict => p2_fill")
        if "p1_fill" in reduced_labels and "conflict" in reduced_labels:
            reduced_labels.discard("conflict")
            reductions.append("p1_fill + conflict => p1_fill")
        if "p1_fill" in reduced_labels and "p1_special" in reduced_labels:
            reduced_labels.discard("p1_fill")
            reductions.append("p1_fill + p1_special => p1_special")
        cell["reduced_labels"] = sorted(reduced_labels)
        if len(reduced_labels) == 1:
            cell["label"] = next(iter(reduced_labels))
            cell["is_error"] = False
        if reductions:
            cell["postprocess_rule"] = " | ".join(reductions)
            known_combo_corrections.append(
                {
                    "json_row": int(cell["json_row"]),
                    "json_col": int(cell["json_col"]),
                    "raw_labels": sorted(raw_labels),
                    "resolved_label": str(cell["label"]),
                    "rule": str(cell["postprocess_rule"]),
                    "phase": "direct",
                }
            )
        working_cells[key] = cell

    # Phase 2: recursive reductions that depend on the lower neighbor being uniquely resolved.
    max_passes = 25
    passes_used = 0
    reached_recursion_limit = False
    for pass_idx in range(max_passes):
        passes_used = pass_idx + 1
        changed = False
        for key in cell_keys:
            cell = dict(working_cells[key])
            reduced_labels = set(str(label) for label in cell.get("reduced_labels", cell.get("raw_labels", [])))
            if not ("p1_special" in reduced_labels and "conflict" in reduced_labels):
                continue
            below = working_cells.get((int(cell["json_row"]) + 1, int(cell["json_col"])))
            if below is None:
                continue
            below_reduced = set(str(label) for label in below.get("reduced_labels", below.get("raw_labels", [])))
            if len(below_reduced) != 1:
                continue
            below_label = next(iter(below_reduced))
            downward = _resolve_downward_combo(reduced_labels, below_label)
            if downward is None:
                continue
            new_reduced = {downward}
            if set(str(x) for x in cell.get("reduced_labels", [])) == new_reduced and str(cell.get("label")) == downward:
                continue
            cell["reduced_labels"] = sorted(new_reduced)
            cell["label"] = downward
            cell["is_error"] = False
            rule = f"p1_special + conflict => {downward} (below={below_label})"
            old_rule = str(cell.get("postprocess_rule", "")).strip()
            cell["postprocess_rule"] = f"{old_rule} | {rule}".strip(" |")
            known_combo_corrections.append(
                {
                    "json_row": int(cell["json_row"]),
                    "json_col": int(cell["json_col"]),
                    "raw_labels": sorted(set(str(label) for label in cell.get("raw_labels", []))),
                    "resolved_label": downward,
                    "rule": rule,
                    "phase": f"downward_pass_{pass_idx + 1}",
                }
            )
            working_cells[key] = cell
            changed = True
        if not changed:
            break
    else:
        reached_recursion_limit = True

    merged_cells: List[Dict] = []
    unknown_combos: List[Dict] = []
    for key in cell_keys:
        cell = dict(working_cells[key])
        raw_labels = set(str(label) for label in cell.get("raw_labels", []))
        reduced_labels = set(str(label) for label in cell.get("reduced_labels", cell.get("raw_labels", [])))
        if len(reduced_labels) >= 2:
            below = working_cells.get((int(cell["json_row"]) + 1, int(cell["json_col"])))
            below_reduced = sorted(set(str(label) for label in below.get("reduced_labels", below.get("raw_labels", [])))) if below is not None else []
            unknown_combos.append(
                {
                    "json_row": int(cell["json_row"]),
                    "json_col": int(cell["json_col"]),
                    "raw_labels": sorted(raw_labels),
                    "reduced_labels": sorted(reduced_labels),
                    "distribution": dict(cell["vote_summary"]["label_votes"]),
                    "below_reduced_labels": below_reduced,
                    "final_label": str(cell.get("label", "")),
                }
            )
        merged_cells.append(cell)

    all_cells_resolved = len(unknown_combos) == 0

    counts = _recount_labels(merged_cells)
    error_cells: List[Dict] = []
    for cell in merged_cells:
        if bool(cell.get("is_error")):
            error_cells.append(
                {
                    "index": int(cell["index"]),
                    "json_row": int(cell["json_row"]),
                    "json_col": int(cell["json_col"]),
                    "center": list(cell["center"]),
                    "mean_bgr": list(cell["mean_bgr"]),
                }
            )

    unknown_combo_debug_image = None
    unknown_combo_debug_document = None
    if unknown_combos:
        resolved_cells = {
            (int(cell["json_row"]), int(cell["json_col"])): cell
            for cell in merged_cells
        }
        unknown_combo_debug_image = _save_unknown_combo_debug_frame(
            frames_bgr[0],
            map_name,
            unknown_combos,
            resolved_cells=resolved_cells,
        )
        if unknown_combo_debug_image:
            unknown_combo_debug_document = str(Path(unknown_combo_debug_image).with_suffix(".md"))

    return {
        "map_name": map_name,
        "reference_point_count": len(merged_cells),
        "counts": counts,
        "error_count": len(error_cells),
        "error_cells": error_cells,
        "cells": merged_cells,
        "frame_count_used": len(frames_bgr),
        "frame_results": frame_results,
        "postprocess_passes_used": passes_used,
        "postprocess_max_passes": max_passes,
        "postprocess_reached_recursion_limit": reached_recursion_limit,
        "postprocess_all_cells_resolved": all_cells_resolved,
        "postprocess_known_combo_corrections": known_combo_corrections,
        "unknown_combo_count": len(unknown_combos),
        "unknown_combos": unknown_combos,
        "unknown_combo_debug_image": unknown_combo_debug_image,
        "unknown_combo_debug_document": unknown_combo_debug_document,
    }


def detect_map_state_from_frames(frames_bgr: List[np.ndarray], map_name: str) -> Dict:
    if not frames_bgr:
        raise ValueError("frames_bgr must not be empty")
    frame_results = [detect_map_state(frame, map_name) for frame in frames_bgr]
    return _merge_state_results_from_frames(frame_results, frames_bgr, map_name)


def _frame_json_url_from_frame_url(frame_url: str) -> str:
    base = str(frame_url).strip()
    if base.endswith("/frame.jpg"):
        return base[:-10] + "/frame.json"
    if base.endswith("/frame.jpeg"):
        return base[:-11] + "/frame.json"
    if base.endswith("/frame.json"):
        return base
    return base.rstrip("/") + "/frame.json"


def _fetch_json(url: str, timeout_seconds: float) -> Dict[str, Any]:
    req = urllib.request.Request(url, headers={"Cache-Control": "no-cache"})
    with _urlopen_local_direct(req, timeout=max(0.3, float(timeout_seconds))) as resp:
        return json.loads(resp.read().decode("utf-8"))


def _fetch_frame(url: str, timeout_seconds: float) -> np.ndarray:
    req = urllib.request.Request(url, headers={"Cache-Control": "no-cache"})
    with _urlopen_local_direct(req, timeout=max(0.3, float(timeout_seconds))) as resp:
        payload = resp.read()
    buf = np.frombuffer(payload, dtype=np.uint8)
    frame = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    if frame is None or frame.size == 0:
        raise ValueError(f"cannot decode frame from {url}")
    return frame


def collect_rotating_frames_from_frame_api(
    frame_url: str = "http://127.0.0.1:8765/frame.jpg",
    frame_json_url: str = "",
    sample_count: int = 5,
    poll_interval_seconds: float = 0.08,
    timeout_seconds: float = 3.0,
) -> Dict[str, Any]:
    if sample_count <= 0:
        raise ValueError("sample_count must be > 0")
    json_url = str(frame_json_url or _frame_json_url_from_frame_url(frame_url))
    deadline = time.monotonic() + max(0.5, float(timeout_seconds))
    seen_keys: Set[Tuple[int, float]] = set()
    frames: List[np.ndarray] = []
    metadata_list: List[Dict[str, Any]] = []
    last_metadata: Optional[Dict[str, Any]] = None

    while len(frames) < sample_count and time.monotonic() < deadline:
        metadata = _fetch_json(json_url, timeout_seconds=min(1.0, timeout_seconds))
        last_metadata = metadata
        key = (
            int(metadata.get("frame_count", 0) or 0),
            float(metadata.get("last_frame_ts", 0.0) or 0.0),
        )
        if key in seen_keys or key == (0, 0.0):
            time.sleep(max(0.01, float(poll_interval_seconds)))
            continue
        frame = _fetch_frame(frame_url, timeout_seconds=min(1.0, timeout_seconds))
        seen_keys.add(key)
        frames.append(frame)
        metadata_list.append(metadata)
        if len(frames) < sample_count:
            time.sleep(max(0.01, float(poll_interval_seconds)))

    if not frames:
        raise ValueError(f"no rotating frames collected from {frame_url}")

    return {
        "frames": frames,
        "metadata": metadata_list,
        "frame_count_collected": len(frames),
        "requested_sample_count": int(sample_count),
        "last_metadata": last_metadata or {},
        "frame_url": frame_url,
        "frame_json_url": json_url,
    }


def detect_map_state_from_frame_api(
    map_name: str,
    frame_url: str = "http://127.0.0.1:8765/frame.jpg",
    frame_json_url: str = "",
    sample_count: int = 5,
    poll_interval_seconds: float = 0.08,
    timeout_seconds: float = 3.0,
) -> Dict:
    collected = collect_rotating_frames_from_frame_api(
        frame_url=frame_url,
        frame_json_url=frame_json_url,
        sample_count=sample_count,
        poll_interval_seconds=poll_interval_seconds,
        timeout_seconds=timeout_seconds,
    )
    result = detect_map_state_from_frames(collected["frames"], map_name)
    result["frame_api"] = {
        "frame_url": collected["frame_url"],
        "frame_json_url": collected["frame_json_url"],
        "frame_count_collected": int(collected["frame_count_collected"]),
        "requested_sample_count": int(collected["requested_sample_count"]),
        "metadata": collected["metadata"],
    }
    return result


def _recount_labels(cells: List[Dict]) -> Dict[str, int]:
    counts = {
        "p1_fill": 0,
        "p2_fill": 0,
        "p1_special": 0,
        "p2_special": 0,
        "conflict": 0,
        "transparent": 0,
    }
    for cell in cells:
        label = str(cell["label"])
        if label in counts:
            counts[label] += 1
    return counts


def apply_p1_special_persistence(result: Dict, persisted_positions: Set[CellPos] | None = None) -> Dict:
    # Deprecated:
    # The older "persisted special positions" workaround is intentionally disabled.
    # Special/fill recognition now relies on direct vision classification plus
    # current-action correction only, so we pass the result through unchanged.
    return result


def _placed_card_special_positions(card_number: int, rotation: int, x: int, y: int) -> Set[CellPos]:
    from tableturf_sim.src.utils.common_utils import _card_cells_on_map, create_card_from_id

    card = create_card_from_id(int(card_number))
    positions: Set[CellPos] = set()
    for map_x, map_y, cell_type in _card_cells_on_map(card, int(x), int(y), int(rotation)):
        if int(cell_type) == 2:
            positions.add((int(map_y), int(map_x)))
    return positions


def apply_current_action_special_correction(
    result: Dict,
    card_number: int,
    rotation: int,
    x: int,
    y: int,
) -> Dict:
    # Deprecated:
    # Current-action special correction is intentionally disabled because the
    # latest vision pipeline can distinguish special vs fill directly.
    return result


class MapStateTracker:
    def __init__(self, map_name: str):
        if map_name not in MAP_NAMES:
            raise ValueError(f"unsupported map: {map_name}")
        self.map_name = map_name

    def reset(self) -> None:
        # Deprecated: no persisted special-position state is kept anymore.
        return None

    def update_frame(self, frame_bgr: np.ndarray) -> Dict:
        result = detect_map_state(frame_bgr, self.map_name)
        return result

    def update_frames(self, frames_bgr: List[np.ndarray]) -> Dict:
        result = detect_map_state_from_frames(frames_bgr, self.map_name)
        return result

    def update_image_path(self, image_path: Path) -> Dict:
        frame = cv2.imread(str(image_path))
        if frame is None:
            raise ValueError(f"cannot read image: {image_path}")
        result = self.update_frame(frame)
        result["image"] = str(image_path)
        return result

    def update_frame_api(
        self,
        frame_url: str = "http://127.0.0.1:8765/frame.jpg",
        frame_json_url: str = "",
        sample_count: int = 5,
        poll_interval_seconds: float = 0.08,
        timeout_seconds: float = 3.0,
    ) -> Dict:
        result = detect_map_state_from_frame_api(
            self.map_name,
            frame_url=frame_url,
            frame_json_url=frame_json_url,
            sample_count=sample_count,
            poll_interval_seconds=poll_interval_seconds,
            timeout_seconds=timeout_seconds,
        )
        return result

    def update_frame_with_action(
        self,
        frame_bgr: np.ndarray,
        card_number: int,
        rotation: int,
        x: int,
        y: int,
    ) -> Dict:
        result = detect_map_state(frame_bgr, self.map_name)
        return result

    def update_image_path_with_action(
        self,
        image_path: Path,
        card_number: int,
        rotation: int,
        x: int,
        y: int,
    ) -> Dict:
        frame = cv2.imread(str(image_path))
        if frame is None:
            raise ValueError(f"cannot read image: {image_path}")
        result = self.update_frame_with_action(frame, card_number, rotation, x, y)
        result["image"] = str(image_path)
        return result


def _build_grid_labels(map_name: str, cells: List[Dict]) -> List[List[str]]:
    map_info = _load_map_info().get(map_name)
    if map_info is None:
        raise ValueError(f"map info not found: {map_name}")
    point_type = map_info["point_type"]
    label_by_pos = {
        (int(cell["json_row"]), int(cell["json_col"])): str(cell["label"])
        for cell in cells
    }
    grid: List[List[str]] = []
    for row_idx, row in enumerate(point_type):
        out_row: List[str] = []
        for col_idx, v in enumerate(row):
            if int(v) == 0:
                out_row.append("invalid")
            else:
                out_row.append(label_by_pos.get((row_idx, col_idx), "transparent"))
        grid.append(out_row)
    return grid


def _rgb(text: str, rgb: tuple[int, int, int]) -> str:
    r, g, b = rgb
    return f"\033[38;2;{r};{g};{b}m{text}{CLR_RESET}"


def render_map_state_grid(map_name: str, result: Dict, colorize: bool = True) -> List[str]:
    grid = _build_grid_labels(map_name, result["cells"])
    lines: List[str] = []
    for row in grid:
        parts: List[str] = []
        for label in row:
            token = TOKEN_MAP.get(label, "??")
            if token.strip() == "":
                parts.append(token)
            elif colorize:
                parts.append(_rgb(token, RGB.get(label, (255, 255, 255))))
            else:
                parts.append(token)
        lines.append("".join(parts))
    return lines


def detect_map_state_image_path(image_path: Path, map_name: str) -> Dict:
    frame = cv2.imread(str(image_path))
    if frame is None:
        raise ValueError(f"cannot read image: {image_path}")
    result = detect_map_state(frame, map_name)
    result["image"] = str(image_path)
    return result


def _make_map_wrapper(map_name: str) -> Callable[[np.ndarray], Dict]:
    def _wrapper(frame_bgr: np.ndarray) -> Dict:
        return detect_map_state(frame_bgr, map_name)

    _wrapper.__name__ = f"detect_map_state__{map_name}"
    _wrapper.__doc__ = f"Detect map-state cells for {map_name}."
    return _wrapper


for _map_name in MAP_NAMES:
    globals()[f"detect_map_state__{_map_name}"] = _make_map_wrapper(_map_name)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Detect current cell-state labels for one of the 15 reference maps.")
    p.add_argument("--map-name", default="", help="Chinese map name")
    p.add_argument("--image", default="", help="image path; empty means latest capture image")
    p.add_argument("--input-dir", default="vision_capture/debug")
    p.add_argument("--frame-url", default="", help="frame API jpg url; when set, use rotating-frame vote mode")
    p.add_argument("--frame-json-url", default="", help="frame API metadata url; defaults to /frame.json")
    p.add_argument("--sample-count", type=int, default=5, help="number of rotating frames to vote")
    p.add_argument("--poll-interval", type=float, default=0.08, help="seconds between frame polling attempts")
    p.add_argument("--timeout-seconds", type=float, default=3.0, help="frame collection timeout")
    p.add_argument("--json", action="store_true", help="print full JSON result")
    p.add_argument("--print-grid", action="store_true", help="print terminal map-structure preview")
    p.add_argument("--no-color", action="store_true", help="disable ANSI color in --print-grid output")
    p.add_argument("--show-reference-points", action="store_true", help="print green-circle reference points for --map-name")
    p.add_argument("--show-all-reference-points", action="store_true", help="print reference points for all 15 maps")
    return p.parse_args()


def main() -> int:
    args = _parse_args()
    if args.show_all_reference_points:
        payload = {}
        for map_name in MAP_NAMES:
            payload[map_name] = {
                "count": len(load_map_reference_points(map_name)),
                "points": [
                    {
                        "index": i + 1,
                        "center": [round(float(p["center"][0]), 2), round(float(p["center"][1]), 2)],
                        "radius": round(float(p["radius"]), 2),
                    }
                    for i, p in enumerate(load_map_reference_points(map_name))
                ],
            }
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return 0

    if not args.map_name:
        raise ValueError("--map-name is required unless --show-all-reference-points is used")

    if args.show_reference_points:
        payload = {
            "map_name": args.map_name,
            "count": len(load_map_reference_points(args.map_name)),
            "points": [
                {
                    "index": i + 1,
                    "center": [round(float(p["center"][0]), 2), round(float(p["center"][1]), 2)],
                    "radius": round(float(p["radius"]), 2),
                }
                for i, p in enumerate(load_map_reference_points(args.map_name))
            ],
        }
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return 0

    if args.frame_url:
        result = detect_map_state_from_frame_api(
            args.map_name,
            frame_url=args.frame_url,
            frame_json_url=args.frame_json_url,
            sample_count=args.sample_count,
            poll_interval_seconds=args.poll_interval,
            timeout_seconds=args.timeout_seconds,
        )
    else:
        image = _resolve_image(args.image, args.input_dir)
        result = detect_map_state_image_path(image, args.map_name)
    if args.json:
        print(json.dumps(result, ensure_ascii=False, indent=2))
    elif args.print_grid:
        if "image" in result:
            print(f"Image: {result['image']}")
        elif "frame_api" in result:
            print(f"Frame API: {result['frame_api']['frame_url']}")
        print(f"Map: {args.map_name}")
        print(f"Counts: {json.dumps(result['counts'], ensure_ascii=False)}")
        print("[Grid]")
        for line in render_map_state_grid(args.map_name, result, colorize=not args.no_color):
            print(line)
    else:
        print(json.dumps(result["counts"], ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
