# Map State Detector Color Threshold Backup

Date: 2026-03-18
Source: `../map_state_detector.py`
Function: `_classify_cell(mean_bgr)`

This backup records the active color-domain constraints used by map-state recognition.

## Input

- Sample source: `mean_bgr`
- Convert with OpenCV: `cv2.COLOR_BGR2HSV`
- Extracted channels:
  - `h`: HSV hue
  - `s`: HSV saturation
  - `v`: HSV value
  - `b, g, r`: original BGR channel means

## Label Rules

### `p1_fill`

Triggered when all of the following are true:

- `28 <= h <= 42`
- `s >= 150`
- `v >= 180`

Score formula:

- `scores["p1_fill"] = 1.0 + (v / 255.0)`

### `p2_fill`

Triggered when all of the following are true:

- `96 <= h <= 135`
- `s >= 95`
- `80 <= v <= 255`
- `b >= 90`
- `b > g + 30`
- `g >= 35`
- `g >= r - 10`
- `(b - r) >= 35`

Score formula:

- `scores["p2_fill"] = 1.0 + (s / 255.0) + min(0.35, (b - g) / 255.0)`

### `p1_special`

Triggered when all of the following are true:

- `12 <= h <= 27`
- `s >= 140`
- `v >= 150`

Score formula:

- `scores["p1_special"] = 1.0 + (v / 255.0)`

### `p2_special`

Triggered when all of the following are true:

- `80 <= h <= 102`
- `s >= 90`
- `v >= 180`

Score formula:

- `scores["p2_special"] = 1.0 + (v / 255.0)`

### `conflict`

Triggered when all of the following are true:

- `s <= 45`
- `v >= 150`

Score formula:

- `scores["conflict"] = 1.0 + (v / 255.0)`

### `transparent`

Triggered only when no other label receives a score above `0.0`.

Rule:

- `if max(scores.values()) <= 0.0:`
- `scores["transparent"] = 1.0`

## Priority Order

When multiple labels have scores, the final label is selected by this priority order:

1. `p1_fill`
2. `p2_fill`
3. `p1_special`
4. `p2_special`
5. `conflict`
6. `transparent`

Actual code behavior:

- `label = max(priority, key=lambda name: (scores[name], -priority.index(name)))`

## Related Board-Like Rule Used By Map Name Detection

Current map-name recognition also uses a board-like fallback based on the same classifier.

A sampled point is considered board-like if either:

- `_classify_cell(mean_bgr)` is not `transparent`

or:

- `b <= 70`
- `g <= 60`
- `r <= 60`
- `b >= g - 5`
- `b >= r - 10`

## Notes

- This file is a manual backup snapshot of the currently active thresholds.
- If `_classify_cell()` changes later, this file should be updated or a new dated backup should be added.
