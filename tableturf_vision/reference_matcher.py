from __future__ import annotations

import json
from collections import defaultdict
from functools import lru_cache
from pathlib import Path
from typing import Dict, List

import cv2
import numpy as np

from tableturf_vision.map_state_detector import (
    MAP_NAMES,
    _classify_cell,
    _load_map_info,
    load_map_reference_points,
)
from tableturf_vision.tableturf_mapper import _load_layout, _parse_board


REPO_ROOT = Path(__file__).resolve().parent.parent
MAP_INFO_PATH = REPO_ROOT / "tableturf_sim" / "data" / "maps" / "MiniGameMapInfo.json"
REFERENCE_TEMPLATE_DIR = REPO_ROOT / "tableturf_vision" / "参照基础"


def load_map_info() -> List[Dict]:
    return json.loads(MAP_INFO_PATH.read_text(encoding="utf-8"))


def map_name_cn_to_id() -> Dict[str, str]:
    out: Dict[str, str] = {}
    for row in load_map_info():
        out[str(row.get("name", ""))] = str(row.get("id", ""))
    return out


def list_reference_template_images() -> List[Path]:
    return [
        REFERENCE_TEMPLATE_DIR / f"{map_name}.png"
        for map_name in MAP_NAMES
        if (REFERENCE_TEMPLATE_DIR / f"{map_name}.png").is_file()
    ]


def _reference_image_path(map_name: str) -> Path:
    return REFERENCE_TEMPLATE_DIR / f"{map_name}.png"


def valid_mask_from_labels(labels: List[List[str]]) -> np.ndarray:
    return np.array([[0 if v == "invalid" else 1 for v in row] for row in labels], dtype=np.uint8)


def compare_masks(obs: np.ndarray, tmpl: np.ndarray) -> Dict:
    oh, ow = obs.shape[:2]
    th, tw = tmpl.shape[:2]
    if (th, tw) != (oh, ow):
        tmpl2 = cv2.resize(tmpl, (ow, oh), interpolation=cv2.INTER_NEAREST)
    else:
        tmpl2 = tmpl
    same = int((obs == tmpl2).sum())
    total = int(obs.size)
    exact_ratio = same / max(1, total)
    return {"exact_ratio": float(exact_ratio), "template_mask_resized": tmpl2}


def _is_board_like(mean_bgr: np.ndarray) -> bool:
    label, _, is_error = _classify_cell(mean_bgr)
    if is_error:
        return False
    if label != "transparent":
        return True
    b, g, r = [float(x) for x in mean_bgr.tolist()]
    return (
        b <= 70
        and g <= 60
        and r <= 60
        and b >= g - 5
        and b >= r - 10
    )


def _sample_patch_mean_bgr_for_map_match(frame_bgr: np.ndarray, cx: float, cy: float, radius: float) -> np.ndarray:
    h, w = frame_bgr.shape[:2]
    r = max(3, int(round(radius * 0.6)))
    ix, iy = int(round(cx)), int(round(cy))
    x0, x1 = max(0, ix - r), min(w, ix + r + 1)
    y0, y1 = max(0, iy - r), min(h, iy + r + 1)
    patch = frame_bgr[y0:y1, x0:x1]
    if patch.size == 0:
        return np.array([0.0, 0.0, 0.0], dtype=np.float32)
    return patch.reshape(-1, 3).mean(axis=0).astype(np.float32)


@lru_cache(maxsize=None)
def _candidate_grid_geometry(map_name: str) -> Dict:
    points = load_map_reference_points(map_name)
    row_y = defaultdict(list)
    col_x = defaultdict(list)
    radius_norms = []
    for point in points:
        row_y[int(point["json_row"])].append(float(point["center_norm"][1]))
        col_x[int(point["json_col"])].append(float(point["center_norm"][0]))
        radius_norms.append(float(point["radius_norm"]))
    row_y_mean = {row: sum(vals) / len(vals) for row, vals in row_y.items()}
    col_x_mean = {col: sum(vals) / len(vals) for col, vals in col_x.items()}
    radius_norm = sum(radius_norms) / max(1, len(radius_norms))
    point_type = _load_map_info()[map_name]["point_type"]
    return {
        "row_y_norm": row_y_mean,
        "col_x_norm": col_x_mean,
        "radius_norm": radius_norm,
        "point_type": point_type,
        "valid_count": int(sum(int(v != 0) for row in point_type for v in row)),
    }


def match_map_by_reference_board_labels(board_labels: List[List[str]], layout: Dict) -> Dict:
    # Map matching must use only the parsed center board grid.
    # Left-hand card area and right-hand play UI are excluded upstream by
    # tableturf_mapper._parse_board via layout["board_roi_norm"].
    obs_mask = valid_mask_from_labels(board_labels)
    best = {
        "enum_index": -1,
        "map_id": "",
        "map_name_zh": "",
        "score": -1.0,
        "obs_shape": [int(obs_mask.shape[0]), int(obs_mask.shape[1])] if obs_mask.size else [0, 0],
        "template_image": "",
        "board_from_template": None,
        "match_mode": "reference_png_topology_score",
    }
    if obs_mask.size == 0:
        return best

    name2id = map_name_cn_to_id()
    maps_order = load_map_info()
    id_to_enum = {str(m.get("id", "")): i + 1 for i, m in enumerate(maps_order)}

    for template_path in list_reference_template_images():
        name_cn = template_path.stem
        map_id = name2id.get(name_cn, "")
        if not map_id:
            continue
        tmpl_frame = cv2.imread(str(template_path))
        if tmpl_frame is None:
            continue
        tmpl_board = _parse_board(tmpl_frame, layout)
        tmpl_mask = valid_mask_from_labels(tmpl_board.labels)
        comp = compare_masks(obs_mask, tmpl_mask)
        score = float(comp["exact_ratio"])
        if score > best["score"]:
            best = {
                "enum_index": int(id_to_enum.get(map_id, -1)),
                "map_id": map_id,
                "map_name_zh": name_cn,
                "score": score,
                "obs_shape": [int(obs_mask.shape[0]), int(obs_mask.shape[1])],
                "template_image": str(template_path),
                "board_from_template": tmpl_board,
                "match_mode": "reference_png_mask_exact_ratio",
            }
    return best


def match_map_from_frame(frame_bgr: np.ndarray) -> Dict:
    name2id = map_name_cn_to_id()
    maps_order = load_map_info()
    id_to_enum = {str(m.get("id", "")): i + 1 for i, m in enumerate(maps_order)}
    h, w = frame_bgr.shape[:2]
    best = {
        "enum_index": -1,
        "map_id": "",
        "map_name_zh": "",
        "score": -1.0,
        "valid_hits": -1,
        "valid_total": 0,
        "template_image": "",
        "board_from_template": None,
        "match_mode": "reference_png_topology_score",
        "candidate_details": [],
    }

    candidate_rows = []
    for map_name in MAP_NAMES:
        geometry = _candidate_grid_geometry(map_name)
        radius = max(4.0, geometry["radius_norm"] * float(max(w, h)))
        point_type = geometry["point_type"]
        valid_hits = 0
        valid_total = 0
        matches = 0
        total = 0
        for row_idx, row in enumerate(point_type):
            if row_idx not in geometry["row_y_norm"]:
                continue
            cy = geometry["row_y_norm"][row_idx] * float(h)
            for col_idx, cell_value in enumerate(row):
                if col_idx not in geometry["col_x_norm"]:
                    continue
                cx = geometry["col_x_norm"][col_idx] * float(w)
                mean_bgr = _sample_patch_mean_bgr_for_map_match(frame_bgr, cx, cy, radius)
                is_board = _is_board_like(mean_bgr)
                expected_valid = int(cell_value) != 0
                total += 1
                matches += int(is_board == expected_valid)
                if expected_valid:
                    valid_total += 1
                    valid_hits += int(is_board)

        score = (matches / total) if total else 0.0
        valid_hit_ratio = (valid_hits / valid_total) if valid_total else 0.0
        template_path = _reference_image_path(map_name)
        row = {
            "map_name_zh": map_name,
            "score": float(score),
            "valid_hit_ratio": float(valid_hit_ratio),
            "valid_hits": int(valid_hits),
            "valid_total": int(valid_total),
            "total_cells_scored": int(total),
            "template_image": str(template_path),
        }
        candidate_rows.append(row)

        better = False
        best_valid_hits = int(best.get("valid_hits", -1))
        best_valid_hit_ratio = float(best.get("valid_hit_ratio", -1.0))
        best_score = float(best.get("score", -1.0))
        if valid_hit_ratio > best_valid_hit_ratio:
            better = True
        elif abs(valid_hit_ratio - best_valid_hit_ratio) <= 1e-9 and score > best_score:
            better = True
        elif (
            abs(valid_hit_ratio - best_valid_hit_ratio) <= 1e-9
            and abs(score - best_score) <= 1e-9
            and valid_hits > best_valid_hits
        ):
            better = True
        if better:
            tmpl_frame = cv2.imread(str(template_path))
            tmpl_board = _parse_board(tmpl_frame, _load_layout(None)) if tmpl_frame is not None else None
            best = {
                "enum_index": int(id_to_enum.get(name2id.get(map_name, ""), -1)),
                "map_id": str(name2id.get(map_name, "")),
                "map_name_zh": map_name,
                "score": float(score),
                "valid_hits": int(valid_hits),
                "valid_total": int(valid_total),
                "valid_hit_ratio": float(valid_hit_ratio),
                "template_image": str(template_path),
                "board_from_template": tmpl_board,
                "match_mode": "reference_png_topology_score",
                "candidate_details": [],
            }

    best["candidate_details"] = sorted(
        candidate_rows,
        key=lambda item: (item["valid_hit_ratio"], item["score"], item["valid_hits"]),
        reverse=True,
    )
    return best


def detect_map_from_frame(frame_bgr: np.ndarray, layout: Dict | None = None) -> Dict:
    layout_use = layout or _load_layout(None)
    board = _parse_board(frame_bgr, layout_use)
    result = match_map_from_frame(frame_bgr)
    result["board_shape"] = [
        len(board.labels),
        len(board.labels[0]) if board.labels else 0,
    ]
    return result


def detect_map_from_image_path(image_path: Path, layout: Dict | None = None) -> Dict:
    path = image_path if image_path.is_absolute() else (REPO_ROOT / image_path)
    frame = cv2.imread(str(path))
    if frame is None:
        raise ValueError(f"cannot read image: {path}")
    result = detect_map_from_frame(frame, layout=layout)
    result["image"] = str(path)
    return result
