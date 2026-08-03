from __future__ import annotations

from pathlib import Path
from typing import Union

import cv2
import numpy as np


PathLike = Union[str, Path]


def imread_unicode(path: PathLike, flags: int = cv2.IMREAD_COLOR) -> np.ndarray | None:
    """Read an image without relying on OpenCV's Windows path decoding."""
    try:
        payload = Path(path).read_bytes()
    except OSError:
        return None
    if not payload:
        return None
    return cv2.imdecode(np.frombuffer(payload, dtype=np.uint8), flags)
