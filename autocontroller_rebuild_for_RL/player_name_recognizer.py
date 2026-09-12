"""Standalone player-name text recognition backed by PaddleOCR.

The public interface deliberately has one responsibility: accept a cropped
text-line image and return the text produced by the official recognition
model.  It contains no room-state, banner-status, player-name dictionary,
character-width, or hand-authored glyph correction logic.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping
import os
from pathlib import Path
import threading
from typing import Any

import numpy as np


PLAYER_NAME_MODEL = "PP-OCRv6_medium_rec"

# PaddleOCR downloads official weights on first use.  BOS is the officially
# documented alternative to Hugging Face and is normally more reliable from
# the Windows host's network location in China.
os.environ.setdefault("PADDLE_PDX_MODEL_SOURCE", "BOS")
# Keep the recognition worker from consuming every CPU core and disturbing
# the watchdog preview or desktop input responsiveness.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")


class PlayerNameRecognizerUnavailable(RuntimeError):
    """Raised when the official PaddleOCR recognizer cannot be used."""


_MODEL: Any = None
_MODEL_ERROR: BaseException | None = None
_MODEL_LOCK = threading.Lock()
_INFERENCE_LOCK = threading.Lock()


def _load_model() -> Any:
    global _MODEL
    global _MODEL_ERROR
    with _MODEL_LOCK:
        if _MODEL is not None:
            return _MODEL
        if _MODEL_ERROR is not None:
            raise PlayerNameRecognizerUnavailable(
                f"PaddleOCR初始化失败：{_MODEL_ERROR}"
            ) from _MODEL_ERROR
        try:
            from paddleocr import TextRecognition

            # Do not import PaddlePaddle before PaddleOCR here. PaddleX 3.7
            # imports ModelScope (and therefore PyTorch) while it discovers
            # official model hosts.  Loading Paddle's Windows DLLs first can
            # make PyTorch's shm.dll fail with WinError 127.  PaddleOCR loads
            # the inference backend itself after its host discovery; the
            # cpu_threads argument below still enforces the one-thread limit.
            _MODEL = TextRecognition(
                model_name=PLAYER_NAME_MODEL,
                device="cpu",
                engine="paddle",
                enable_mkldnn=True,
                cpu_threads=1,
            )
        except BaseException as exc:
            _MODEL_ERROR = exc
            raise PlayerNameRecognizerUnavailable(
                f"PaddleOCR初始化失败：{exc}"
            ) from exc
        return _MODEL


def _raw_rec_text(result: Any) -> str:
    """Read PaddleOCR's documented rec_text field without altering it."""
    payload: Any = result
    if not isinstance(payload, Mapping):
        with_json = getattr(payload, "json", None)
        if callable(with_json):
            with_json = with_json()
        if with_json is not None:
            payload = with_json
    if isinstance(payload, Mapping) and "res" in payload:
        payload = payload["res"]
    if isinstance(payload, Mapping):
        value = payload.get("rec_text", "")
        return "" if value is None else str(value)
    try:
        value = result["rec_text"]
    except (KeyError, TypeError, AttributeError):
        return ""
    return "" if value is None else str(value)


def warm_up_player_name_recognizer() -> None:
    """Load the official model before a short-lived notification arrives."""
    _load_model()


def repair_player_name_right_edge(image: np.ndarray) -> np.ndarray:
    """Whiten only neutral dark pixels beyond the last orange name stroke.

    The caller supplies an already cropped player-name image in OpenCV BGR
    order. Orange pixels and every pixel at or left of the final orange
    column are returned byte-for-byte unchanged. Only black/gray remnants in
    the trailing right padding are changed to pure white.
    """
    if image is None:
        raise ValueError("玩家姓名右边缘修复接口不能接收空图像。")
    if image.size == 0 or image.ndim != 3:
        return image.copy()
    repaired = image.copy()
    blue = repaired[:, :, 0]
    green = repaired[:, :, 1]
    red = repaired[:, :, 2]
    red_i = red.astype(np.int16)
    green_i = green.astype(np.int16)
    blue_i = blue.astype(np.int16)
    orange_mask = (
        (red > 145)
        & (green > 45)
        & (green < 225)
        & (blue < 155)
        & (red_i > green_i + 22)
        & (green_i > blue_i + 12)
    )
    orange_columns = np.flatnonzero(orange_mask.any(axis=0))
    if orange_columns.size == 0:
        return repaired

    trailing_start = int(orange_columns[-1]) + 1
    if trailing_start >= repaired.shape[1]:
        return repaired
    trailing = repaired[:, trailing_start:, :3]
    trailing_max = trailing.max(axis=2).astype(np.int16)
    trailing_min = trailing.min(axis=2).astype(np.int16)
    trailing_gray = (
        trailing[:, :, 0].astype(np.float32) * 0.114
        + trailing[:, :, 1].astype(np.float32) * 0.587
        + trailing[:, :, 2].astype(np.float32) * 0.299
    )
    black_or_gray = (
        ((trailing_max - trailing_min) <= 48)
        | (trailing_gray <= 72)
    )
    trailing[black_or_gray] = 255
    repaired[:, trailing_start:, :3] = trailing
    return repaired


def recognize_player_name(
    image: str | Path | np.ndarray,
) -> str:
    """Return only the official model's text for one cropped name image."""
    model_input: str | np.ndarray
    if isinstance(image, Path):
        model_input = str(image)
    else:
        model_input = image
    model = _load_model()
    try:
        with _INFERENCE_LOCK:
            results = model.predict(input=model_input, batch_size=1)
            result = next(iter(results), None)
    except BaseException as exc:
        raise PlayerNameRecognizerUnavailable(
            f"PaddleOCR识别失败：{exc}"
        ) from exc
    return "" if result is None else _raw_rec_text(result)


def main() -> int:
    import cv2

    parser = argparse.ArgumentParser(
        description="输入一张玩家名裁剪图片，输出PaddleOCR识别文字。"
    )
    parser.add_argument("image", type=Path)
    args = parser.parse_args()
    if not args.image.is_file():
        parser.error(f"找不到输入图片：{args.image.resolve()}")
    encoded = np.fromfile(args.image, dtype=np.uint8)
    image = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
    if image is None:
        parser.error(f"无法读取输入图片：{args.image.resolve()}")
    repaired = repair_player_name_right_edge(image)
    print(recognize_player_name(repaired))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
