from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import cv2
import numpy as np


DEFAULT_LAYOUT = {
    "board_roi_norm": [0.485, 0.12, 0.24, 0.82],
    "card_rois_norm": [
        [0.0125, 0.164, 0.124, 0.313],
        [0.1410, 0.164, 0.124, 0.313],
        [0.0125, 0.482, 0.124, 0.313],
        [0.1410, 0.482, 0.124, 0.313],
    ],
    "card_grid_hint": [8, 8],
    "board_palette_bgr": {
        "invalid": [30, 22, 24],
        "empty": [22, 14, 16],
        "p1_fill": [0, 255, 255],
        "p1_special": [0, 192, 255],
        "p2_fill": [192, 112, 0],
        "p2_special": [240, 176, 0],
        "conflict": [235, 232, 188],
        "p1_special_activated": [0, 192, 255],
        "p2_special_activated": [240, 176, 0]
    },
    "card_palette_bgr": {
        "empty": [22, 14, 16],
        "fill": [24, 240, 232],
        "special": [245, 224, 45],
        "conflict": [235, 232, 188],
        "special_activated": [120, 246, 242]
    },
    "line_merge_px": 4,
    "line_threshold_percentile": 90,
}


@dataclass
class GridParseResult:
    labels: List[List[str]]
    centers_abs: List[List[Tuple[int, int]]]
    line_x: List[int]
    line_y: List[int]
    roi_abs: Tuple[int, int, int, int]


def _load_layout(path: Path | None) -> Dict:
    layout = dict(DEFAULT_LAYOUT)
    if path is None:
        default_path = Path(__file__).resolve().with_name("tableturf_layout.json")
        if default_path.exists():
            data = json.loads(default_path.read_text(encoding="utf-8"))
            if isinstance(data, dict):
                layout.update(data)
        return layout
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, dict):
        layout.update(data)
    return layout


def _roi_from_norm(img_shape: Tuple[int, int, int], roi_norm: Sequence[float]) -> Tuple[int, int, int, int]:
    h, w = img_shape[:2]
    x = int(w * float(roi_norm[0]))
    y = int(h * float(roi_norm[1]))
    rw = int(w * float(roi_norm[2]))
    rh = int(h * float(roi_norm[3]))
    x = max(0, min(x, w - 2))
    y = max(0, min(y, h - 2))
    rw = max(2, min(rw, w - x))
    rh = max(2, min(rh, h - y))
    return x, y, rw, rh


def _merge_close_indices(indices: List[int], merge_px: int) -> List[int]:
    if not indices:
        return []
    out = [indices[0]]
    for idx in indices[1:]:
        if idx - out[-1] <= merge_px:
            continue
        out.append(idx)
    return out


def _peak_indices_1d(arr: np.ndarray, percentile: float, merge_px: int, min_val: float = 0.0) -> List[int]:
    if arr.size < 3:
        return []
    thr = max(float(np.percentile(arr, percentile)), min_val)
    peaks: List[int] = []
    for i in range(1, arr.size - 1):
        if arr[i] >= thr and arr[i] >= arr[i - 1] and arr[i] >= arr[i + 1]:
            peaks.append(i)
    return _merge_close_indices(peaks, merge_px)


def _detect_grid_lines(roi_bgr: np.ndarray, percentile: float, merge_px: int) -> Tuple[List[int], List[int]]:
    gray = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    grad_x = np.mean(np.abs(np.diff(gray.astype(np.float32), axis=1)), axis=0)
    grad_y = np.mean(np.abs(np.diff(gray.astype(np.float32), axis=0)), axis=1)
    line_x = _peak_indices_1d(grad_x, percentile=percentile, merge_px=merge_px, min_val=4.0)
    line_y = _peak_indices_1d(grad_y, percentile=percentile, merge_px=merge_px, min_val=4.0)
    return line_x, line_y


def _regularize_grid_lines(lines: List[int]) -> List[int]:
    if len(lines) < 5:
        return lines
    diffs = np.diff(lines)
    diffs = diffs[(diffs >= 6) & (diffs <= 80)]
    if diffs.size == 0:
        return lines
    step = float(np.median(diffs))
    tol = max(2.0, step * 0.35)

    best: List[int] = []
    for si in range(len(lines)):
        seq = [lines[si]]
        expected = float(lines[si]) + step
        for j in range(si + 1, len(lines)):
            if abs(lines[j] - expected) <= tol:
                seq.append(lines[j])
                expected = float(lines[j]) + step
        if len(seq) > len(best):
            best = seq

    if len(best) >= 4:
        return best
    return lines


def _sample_patch_mean(roi_bgr: np.ndarray, cx: int, cy: int, r: int = 2) -> np.ndarray:
    h, w = roi_bgr.shape[:2]
    x0 = max(0, cx - r)
    x1 = min(w, cx + r + 1)
    y0 = max(0, cy - r)
    y1 = min(h, cy + r + 1)
    patch = roi_bgr[y0:y1, x0:x1]
    return patch.reshape(-1, 3).mean(axis=0)


def _classify_bgr(mean_bgr: np.ndarray, palette_bgr: Dict[str, Sequence[int]]) -> str:
    names = list(palette_bgr.keys())
    colors = np.array([palette_bgr[n] for n in names], dtype=np.float32)
    d = np.linalg.norm(colors - mean_bgr.reshape(1, 3), axis=1)
    return names[int(np.argmin(d))]


def _classify_card_bgr(mean_bgr: np.ndarray) -> str:
    """
    Card mini-board classifier:
    - empty: dark/low-sat
    - special: orange-like (SP point)
    - fill: yellow-like
    """
    bgr_u8 = mean_bgr.astype(np.uint8)
    b, g, r = int(bgr_u8[0]), int(bgr_u8[1]), int(bgr_u8[2])
    bgr = np.uint8([[bgr_u8]])
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)[0, 0]
    h, s, v = int(hsv[0]), int(hsv[1]), int(hsv[2])

    if v < 45 or s < 35:
        return "empty"

    # Bright yellow/orange ink-like pixels.
    if r >= 110 and g >= 90 and b <= 110 and s >= 55 and v >= 85:
        # SP orange tends to have noticeably lower G than yellow fill.
        if (r - g) >= 28 or (10 <= h <= 26):
            return "special"
        return "fill"

    # Fallback: any other bright painted cell still counts as fill.
    if s >= 70 and v >= 80:
        return "fill"

    return "empty"


def _card_special_score(mean_bgr: np.ndarray) -> float:
    b, g, r = float(mean_bgr[0]), float(mean_bgr[1]), float(mean_bgr[2])
    # Favor orange-like high-R, medium-G, low-B cells.
    score = 0.0
    score += max(0.0, r - g) * 1.4
    score += max(0.0, r - b) * 0.6
    score -= max(0.0, g - r) * 0.8
    if r < 80 or g < 60:
        score -= 20.0
    return score


def _parse_grid_from_lines(
    roi_bgr: np.ndarray,
    roi_abs_xy: Tuple[int, int],
    line_x: List[int],
    line_y: List[int],
    palette_bgr: Dict[str, Sequence[int]],
) -> Tuple[List[List[str]], List[List[Tuple[int, int]]]]:
    labels: List[List[str]] = []
    centers_abs: List[List[Tuple[int, int]]] = []
    x_abs, y_abs = roi_abs_xy
    for yi in range(len(line_y) - 1):
        row_l: List[str] = []
        row_c: List[Tuple[int, int]] = []
        cy = int((line_y[yi] + line_y[yi + 1]) / 2)
        for xi in range(len(line_x) - 1):
            cx = int((line_x[xi] + line_x[xi + 1]) / 2)
            mean_bgr = _sample_patch_mean(roi_bgr, cx, cy, r=2)
            row_l.append(_classify_bgr(mean_bgr, palette_bgr))
            row_c.append((x_abs + cx, y_abs + cy))
        labels.append(row_l)
        centers_abs.append(row_c)
    return labels, centers_abs


def _parse_board(frame_bgr: np.ndarray, layout: Dict) -> GridParseResult:
    x, y, w, h = _roi_from_norm(frame_bgr.shape, layout["board_roi_norm"])
    roi = frame_bgr[y : y + h, x : x + w]
    line_x, line_y = _detect_grid_lines(
        roi,
        percentile=float(layout["line_threshold_percentile"]),
        merge_px=int(layout["line_merge_px"]),
    )
    line_x = _regularize_grid_lines(line_x)
    line_y = _regularize_grid_lines(line_y)
    labels, centers_abs = _parse_grid_from_lines(
        roi,
        (x, y),
        line_x,
        line_y,
        palette_bgr=layout.get("board_palette_bgr", layout.get("palette_bgr", {})),
    )
    return GridParseResult(labels=labels, centers_abs=centers_abs, line_x=line_x, line_y=line_y, roi_abs=(x, y, w, h))


def _fallback_card_lines(roi_w: int, roi_h: int, cols: int = 8, rows: int = 8) -> Tuple[List[int], List[int]]:
    line_x = [int(round(i * (roi_w - 1) / cols)) for i in range(cols + 1)]
    line_y = [int(round(i * (roi_h - 1) / rows)) for i in range(rows + 1)]
    return line_x, line_y


def _parse_card(frame_bgr: np.ndarray, card_roi_norm: Sequence[float], layout: Dict) -> GridParseResult:
    x, y, w, h = _roi_from_norm(frame_bgr.shape, card_roi_norm)
    roi = frame_bgr[y : y + h, x : x + w]

    # Card mini-board is a stable sub-area in upper part of card panel.
    gy0 = int(h * 0.07)
    gy1 = int(h * 0.62)
    gx0 = int(w * 0.08)
    gx1 = int(w * 0.92)
    grid_roi = roi[gy0:gy1, gx0:gx1]
    gx, gy = x + gx0, y + gy0
    line_x, line_y = _detect_grid_lines(
        grid_roi,
        percentile=float(layout["line_threshold_percentile"]),
        merge_px=int(layout["line_merge_px"]),
    )
    hint_rows, hint_cols = [int(v) for v in layout["card_grid_hint"]]
    if len(line_x) < hint_cols + 1 or len(line_y) < hint_rows + 1:
        line_x, line_y = _fallback_card_lines(grid_roi.shape[1], grid_roi.shape[0], cols=hint_cols, rows=hint_rows)
    else:
        line_x = line_x[: hint_cols + 1]
        line_y = line_y[: hint_rows + 1]

    labels: List[List[str]] = []
    special_scores: List[List[float]] = []
    centers_abs: List[List[Tuple[int, int]]] = []
    for yi in range(len(line_y) - 1):
        row_l: List[str] = []
        row_s: List[float] = []
        row_c: List[Tuple[int, int]] = []
        cy = int((line_y[yi] + line_y[yi + 1]) / 2)
        for xi in range(len(line_x) - 1):
            cx = int((line_x[xi] + line_x[xi + 1]) / 2)
            mean_bgr = _sample_patch_mean(grid_roi, cx, cy, r=2)
            row_l.append(_classify_card_bgr(mean_bgr))
            row_s.append(_card_special_score(mean_bgr))
            row_c.append((gx + cx, gy + cy))
        labels.append(row_l)
        special_scores.append(row_s)
        centers_abs.append(row_c)

    # Keep one strongest SP candidate per card to suppress orange shading false positives.
    best_pos = (-1, -1)
    best_score = -1e9
    for yi in range(len(labels)):
        for xi in range(len(labels[yi])):
            if labels[yi][xi] == "empty":
                continue
            sc = special_scores[yi][xi]
            if sc > best_score:
                best_score = sc
                best_pos = (yi, xi)

    for yi in range(len(labels)):
        for xi in range(len(labels[yi])):
            if labels[yi][xi] == "empty":
                continue
            labels[yi][xi] = "fill"
    if best_pos[0] >= 0 and best_score >= 10.0:
        labels[best_pos[0]][best_pos[1]] = "special"

    return GridParseResult(labels=labels, centers_abs=centers_abs, line_x=line_x, line_y=line_y, roi_abs=(x, y, w, h))


def _parse_card_with_slot(frame_bgr: np.ndarray, card_roi_norm: Sequence[float], layout: Dict, slot_index: int) -> GridParseResult:
    """
    slot_index=0 (left-top selected card) is often brighter due active highlight.
    Apply a mild V-channel compression to reduce color drift before parsing.
    """
    x, y, w, h = _roi_from_norm(frame_bgr.shape, card_roi_norm)
    roi = frame_bgr[y : y + h, x : x + w]
    if slot_index == 0:
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV).astype(np.float32)
        hsv[:, :, 2] *= 0.88
        hsv[:, :, 2] = np.clip(hsv[:, :, 2], 0, 255)
        roi = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
        frame2 = frame_bgr.copy()
        frame2[y : y + h, x : x + w] = roi
        return _parse_card(frame2, card_roi_norm, layout)
    return _parse_card(frame_bgr, card_roi_norm, layout)


def _normalize_label(label: str) -> str:
    if label in {
        "fill",
        "special",
        "conflict",
        "special_activated",
        "empty",
        "invalid",
        "p1_fill",
        "p1_special",
        "p2_fill",
        "p2_special",
        "p1_special_activated",
        "p2_special_activated",
    }:
        return label
    return "fill"


def _card_signature(labels: List[List[str]]) -> str:
    # 8x8 signature for matching cards across frames.
    code = {"empty": "0", "fill": "1", "special": "2", "conflict": "3", "special_activated": "4"}
    out = []
    for row in labels:
        for c in row:
            n = _normalize_label(c)
            if n in {"p1_fill", "p2_fill"}:
                n = "fill"
            elif n in {"p1_special", "p2_special"}:
                n = "special"
            elif n in {"p1_special_activated", "p2_special_activated"}:
                n = "special_activated"
            out.append(code.get(n, "1"))
    return "".join(out)


def _overlay_grid(img: np.ndarray, parsed: GridParseResult, title: str) -> None:
    x, y, w, h = parsed.roi_abs
    cv2.rectangle(img, (x, y), (x + w, y + h), (255, 255, 255), 1)
    for px in parsed.line_x:
        cv2.line(img, (x + px, y), (x + px, y + h), (180, 180, 180), 1)
    for py in parsed.line_y:
        cv2.line(img, (x, y + py), (x + w, y + py), (180, 180, 180), 1)
    cv2.putText(img, title, (x + 4, max(20, y - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)


def analyze_image(path: Path, layout: Dict, write_overlay: bool, out_dir: Path) -> Dict:
    frame = cv2.imread(str(path))
    if frame is None:
        raise ValueError(f"cannot read image: {path}")

    board = _parse_board(frame, layout)
    cards = [_parse_card_with_slot(frame, roi, layout, i) for i, roi in enumerate(layout["card_rois_norm"])]

    board_labels = [[_normalize_label(c) for c in row] for row in board.labels]
    card_labels = [[[_normalize_label(c) for c in row] for row in g.labels] for g in cards]
    card_signatures = [_card_signature(lbls) for lbls in card_labels]

    payload = {
        "image": str(path),
        "shape": [int(frame.shape[0]), int(frame.shape[1]), int(frame.shape[2])],
        "board": {
            "roi_abs": list(board.roi_abs),
            "grid_rows": len(board_labels),
            "grid_cols": len(board_labels[0]) if board_labels else 0,
            "labels": board_labels,
        },
        "cards": [
            {
                "slot": i + 1,
                "roi_abs": list(cards[i].roi_abs),
                "grid_rows": len(card_labels[i]),
                "grid_cols": len(card_labels[i][0]) if card_labels[i] else 0,
                "labels": card_labels[i],
                "signature": card_signatures[i],
            }
            for i in range(4)
        ],
    }

    if write_overlay:
        vis = frame.copy()
        _overlay_grid(vis, board, "board")
        for i, c in enumerate(cards):
            _overlay_grid(vis, c, f"card{i + 1}")
        overlay_path = out_dir / f"{path.stem}.overlay.jpg"
        cv2.imwrite(str(overlay_path), vis)
        payload["overlay"] = str(overlay_path)
    return payload


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Tableturf grid/keypoint mapper from captured screenshots.")
    p.add_argument("--input-dir", default="vision_capture/debug")
    p.add_argument("--glob", default="capture_*.*")
    p.add_argument("--layout-json", default="tableturf_vision/tableturf_layout.json", help="optional mapping/layout json")
    p.add_argument("--out-dir", default="tableturf_vision/tableturf_out")
    p.add_argument("--limit", type=int, default=0, help="0 means all")
    p.add_argument("--overlay", action="store_true")
    return p.parse_args()


def main() -> int:
    args = _parse_args()
    repo_root = Path(__file__).resolve().parent.parent

    input_dir = Path(args.input_dir)
    if not input_dir.is_absolute():
        input_dir = repo_root / input_dir
    out_dir = Path(args.out_dir)
    if not out_dir.is_absolute():
        out_dir = repo_root / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    layout_json = None
    if args.layout_json:
        layout_json = Path(args.layout_json)
        if not layout_json.is_absolute():
            layout_json = repo_root / layout_json
    layout = _load_layout(layout_json)

    paths = sorted(input_dir.glob(args.glob))
    if args.limit > 0:
        paths = paths[: args.limit]
    if not paths:
        print(json.dumps({"ok": False, "error": "NO_IMAGES_FOUND", "input_dir": str(input_dir), "glob": args.glob}, indent=2))
        return 2

    outputs: List[Dict] = []
    card_sig_counter: Dict[str, int] = {}
    for p in paths:
        row = analyze_image(p, layout=layout, write_overlay=args.overlay, out_dir=out_dir)
        outputs.append(row)
        for c in row["cards"]:
            sig = c["signature"]
            card_sig_counter[sig] = card_sig_counter.get(sig, 0) + 1

        out_json = out_dir / f"{p.stem}.json"
        out_json.write_text(json.dumps(row, ensure_ascii=False, indent=2), encoding="utf-8")

    summary = {
        "ok": True,
        "images": len(outputs),
        "input_dir": str(input_dir),
        "glob": args.glob,
        "out_dir": str(out_dir),
        "layout": layout,
        "card_signature_top": sorted(
            [{"signature": k, "count": v} for k, v in card_sig_counter.items()],
            key=lambda x: x["count"],
            reverse=True,
        )[:20],
    }
    (out_dir / "_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
