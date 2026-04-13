from __future__ import annotations

import contextlib
import json
import os
import re
import select
import subprocess
import threading
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from .state_types import ObservedState


class VisionAdapter:
    """
    Frame -> ObservedState adapter.
    Implement `detect_state` with your OpenCV pipeline.
    """

    def detect_state(self, frame: Any) -> ObservedState:  # pragma: no cover
        raise NotImplementedError("implement OpenCV detection here")

    @staticmethod
    def load_state_json(path: str) -> ObservedState:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        return ObservedState(**data)

    @staticmethod
    def dump_state_json(state: ObservedState, path: str) -> None:
        Path(path).write_text(json.dumps(asdict(state), ensure_ascii=False, indent=2), encoding="utf-8")


def _list_ffmpeg_video_devices_text() -> str:
    cmd = ["ffmpeg", "-f", "dshow", "-list_devices", "true", "-i", "dummy"]
    proc = subprocess.run(cmd, text=True, capture_output=True)
    return (proc.stderr or "") + (proc.stdout or "")


def list_avfoundation_video_devices() -> List[str]:
    """
    Return DirectShow video device names from ffmpeg listing.
    """
    text = _list_ffmpeg_video_devices_text()
    names: List[str] = []
    in_video_section = False
    for line in text.splitlines():
        if "DirectShow video devices" in line:
            in_video_section = True
            continue
        if "DirectShow audio devices" in line:
            in_video_section = False
            continue
        if not in_video_section:
            continue
        m = re.search(r'"(.+)"', line.strip())
        if m:
            names.append(m.group(1).strip())
    return names


def list_avfoundation_video_device_rows() -> List[Dict[str, str]]:
    text = _list_ffmpeg_video_devices_text()
    rows: List[Dict[str, str]] = []
    in_video_section = False
    next_index = 0
    for line in text.splitlines():
        if "DirectShow video devices" in line:
            in_video_section = True
            continue
        if "DirectShow audio devices" in line:
            in_video_section = False
            continue
        if not in_video_section:
            continue
        m = re.search(r'"(.+)"', line.strip())
        if m:
            rows.append({"index": str(next_index), "name": m.group(1).strip()})
            next_index += 1
    return rows


def is_usb_capture_device_name(name: str) -> bool:
    n = (name or "").lower()
    excluded_keywords = (
        "capture screen",
        "screen capture",
        "screen ",
        "display",
        "continuity",
        "iphone",
        "ipad",
        "obs virtual",
        "virtual camera",
        "camera extension",
    )
    if any(k in n for k in excluded_keywords):
        return False

    strong_keywords = (
        "ugreen",
        "uvc",
        "hdmi",
        "cam link",
        "elgato",
        "usb video",
        "video capture",
        "capture card",
    )
    return any(k in n for k in strong_keywords)


def rank_capture_device_name(name: str) -> int:
    n = (name or "").lower()
    if not is_usb_capture_device_name(name):
        return -1000
    score = 0
    if "ugreen" in n:
        score += 100
    if "capture card" in n or "video capture" in n:
        score += 50
    elif "capture" in n:
        score += 40
    if "uvc" in n:
        score += 20
    if "hdmi" in n:
        score += 10
    if "cam link" in n or "elgato" in n:
        score += 30
    return score


def auto_detect_capture_device_name(prefer_usb: bool = True) -> Optional[str]:
    """
    Pick best available capture-card-like video device from ffmpeg device list.
    """
    names = list_avfoundation_video_devices()
    if not names:
        return None

    candidates = names
    if prefer_usb:
        filtered = [n for n in names if is_usb_capture_device_name(n)]
        if not filtered:
            return None
        candidates = filtered

    candidates = sorted(candidates, key=rank_capture_device_name, reverse=True)
    return candidates[0] if candidates else None


class FFmpegCaptureSource:
    """
    Capture frames from DirectShow by device name.
    """

    def __init__(
        self,
        device_name: str = "UGREEN 35287",
        width: int = 1920,
        height: int = 1080,
        fps: int = 30,
        pixel_format: str = "",
        strict_usb_only: bool = True,
    ):
        self.device_name = device_name
        self.width = width
        self.height = height
        self.fps = fps
        self.pixel_format = pixel_format
        self.strict_usb_only = strict_usb_only
        self._proc: Optional[subprocess.Popen] = None
        self._frame_bytes = self.width * self.height * 3
        self._rx_buffer = bytearray()
        self.last_error: Optional[str] = None
        self._reader_thread: Optional[threading.Thread] = None
        self._stop_reader = threading.Event()
        self._frame_lock = threading.Lock()
        self._latest_frame: Optional[np.ndarray] = None
        self._latest_frame_ts: float = 0.0
        self._active_capture_spec: Optional[Dict[str, object]] = None

    def _resolve_video_input_name(self) -> str:
        name = str(self.device_name or "").strip()
        if name.isdigit():
            rows = list_avfoundation_video_device_rows()
            for row in rows:
                if row["index"] == name:
                    return f"video={row['name']}"
            return f"video={name}"
        rows = list_avfoundation_video_device_rows()
        for row in rows:
            if row["name"] == name:
                return f"video={row['name']}"
        return f"video={name}"

    def start(self) -> None:
        if self._proc is not None:
            return
        if self.strict_usb_only and not is_usb_capture_device_name(self.device_name):
            self.last_error = f"DEVICE_REJECTED_NOT_USB_CAPTURE:{self.device_name}"
            raise ValueError(self.last_error)
        inp = self._resolve_video_input_name()
        self._active_capture_spec = {
            "input": inp,
            "width": self.width,
            "height": self.height,
            "fps": self.fps,
            "pixel_format": self.pixel_format,
        }
        cmd = [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-nostdin",
            "-f",
            "dshow",
            "-framerate",
            str(self.fps),
            "-video_size",
            f"{self.width}x{self.height}",
            "-i",
            inp,
            "-pix_fmt",
            "bgr24",
            "-f",
            "rawvideo",
            "pipe:1",
        ]
        if self.pixel_format:
            cmd[7:7] = ["-pixel_format", self.pixel_format]
        self._proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            stdin=subprocess.DEVNULL,
            bufsize=self._frame_bytes * 2,
        )
        if self._proc.stdout is not None:
            with contextlib.suppress(Exception):
                os.set_blocking(self._proc.stdout.fileno(), False)
        if self._proc.stderr is not None:
            with contextlib.suppress(Exception):
                os.set_blocking(self._proc.stderr.fileno(), False)
        self._stop_reader.clear()
        self._reader_thread = threading.Thread(target=self._reader_loop, name="ffmpeg-capture-reader", daemon=True)
        self._reader_thread.start()

    def restart_with(self, width: int, height: int, pixel_format: str) -> None:
        self.stop()
        self.width = int(width)
        self.height = int(height)
        self.pixel_format = str(pixel_format)
        self._frame_bytes = self.width * self.height * 3
        self._rx_buffer = bytearray()
        self.start()

    @property
    def active_capture_spec(self) -> Optional[Dict[str, object]]:
        if self._active_capture_spec is None:
            return None
        return dict(self._active_capture_spec)

    def _stderr_tail(self, max_bytes: int = 4096) -> str:
        if self._proc is None or self._proc.stderr is None:
            return ""
        fd = self._proc.stderr.fileno()
        chunks = bytearray()
        while len(chunks) < max_bytes:
            try:
                part = os.read(fd, min(1024, max_bytes - len(chunks)))
            except BlockingIOError:
                break
            if not part:
                break
            chunks.extend(part)
        return chunks.decode("utf-8", errors="ignore").strip()

    def _read_from_pipe_once(self, timeout_seconds: float) -> bool:
        if self._proc is None or self._proc.stdout is None:
            return False
        out_fd = self._proc.stdout.fileno()
        ready, _, _ = select.select([out_fd], [], [], max(0.0, timeout_seconds))
        if not ready:
            return False
        try:
            chunk = os.read(out_fd, self._frame_bytes * 4)
        except BlockingIOError:
            return False
        if not chunk:
            return False
        self._rx_buffer.extend(chunk)
        return True

    def _pop_frame(self) -> Optional[np.ndarray]:
        if len(self._rx_buffer) < self._frame_bytes:
            return None
        frame_bytes = bytes(self._rx_buffer[: self._frame_bytes])
        del self._rx_buffer[: self._frame_bytes]
        return np.frombuffer(frame_bytes, dtype=np.uint8).reshape((self.height, self.width, 3))

    def _reader_loop(self) -> None:
        while not self._stop_reader.is_set():
            proc = self._proc
            if proc is None:
                return
            got = self._read_from_pipe_once(0.5)
            if not got:
                proc = self._proc
                if proc is None:
                    return
                rc = proc.poll()
                if rc is not None:
                    tail = self._stderr_tail()
                    self.last_error = f"FFMPEG_EXITED({rc}) {tail}".strip()
                    return
                continue
            while True:
                frame = self._pop_frame()
                if frame is None:
                    break
                with self._frame_lock:
                    self._latest_frame = frame.copy()
                    self._latest_frame_ts = time.monotonic()
                    self.last_error = None

    def read(self, timeout_seconds: float = 5.0) -> Optional[np.ndarray]:
        if self._proc is None:
            self.start()
        deadline = time.monotonic() + max(0.1, timeout_seconds)
        while True:
            with self._frame_lock:
                frame = None if self._latest_frame is None else self._latest_frame.copy()
            if frame is not None:
                return frame
            if time.monotonic() >= deadline:
                self.last_error = f"FRAME_TIMEOUT({timeout_seconds}s)"
                return None
            time.sleep(0.02)

    def read_next(self, after_ts: float = 0.0, timeout_seconds: float = 5.0) -> Optional[np.ndarray]:
        if self._proc is None:
            self.start()
        deadline = time.monotonic() + max(0.1, timeout_seconds)
        while True:
            with self._frame_lock:
                frame_ts = self._latest_frame_ts
                frame = None if self._latest_frame is None else self._latest_frame.copy()
            if frame is not None and frame_ts > after_ts:
                self.last_error = None
                return frame
            if time.monotonic() >= deadline:
                self.last_error = f"FRAME_TIMEOUT({timeout_seconds}s)"
                return None
            time.sleep(0.02)

    @property
    def latest_frame_ts(self) -> float:
        with self._frame_lock:
            return float(self._latest_frame_ts)

    def read_with_fallbacks(
        self,
        timeout_seconds: float = 5.0,
        fallback_specs: Optional[List[Dict[str, object]]] = None,
    ) -> Optional[np.ndarray]:
        frame = self.read(timeout_seconds=timeout_seconds)
        if frame is not None:
            return frame
        specs = fallback_specs or []
        for spec in specs:
            self.restart_with(
                width=int(spec.get("width", self.width)),
                height=int(spec.get("height", self.height)),
                pixel_format=str(spec.get("pixel_format", self.pixel_format)),
            )
            frame = self.read(timeout_seconds=timeout_seconds)
            if frame is not None:
                self.last_error = None
                return frame
        return None

    def read_latest(self, timeout_seconds: float = 5.0, drain_ms: int = 50) -> Optional[np.ndarray]:
        del drain_ms
        return self.read(timeout_seconds=timeout_seconds)

    def stop(self) -> None:
        if self._proc is None:
            return
        proc = self._proc
        self._proc = None
        self._stop_reader.set()
        if self._reader_thread is not None:
            self._reader_thread.join(timeout=1.0)
            self._reader_thread = None
        try:
            proc.terminate()
        except Exception:
            pass

        # Avoid blocking forever in wait() when ffmpeg is stuck.
        exited = False
        for _ in range(20):
            rc = proc.poll()
            if rc is not None:
                exited = True
                break
            try:
                time.sleep(0.05)
            except KeyboardInterrupt:
                break

        if not exited:
            try:
                proc.kill()
            except Exception:
                pass
            try:
                proc.wait(timeout=0.5)
            except Exception:
                pass
        if proc.stdout:
            try:
                proc.stdout.close()
            except Exception:
                pass
        if proc.stderr:
            try:
                proc.stderr.close()
            except Exception:
                pass
        with self._frame_lock:
            self._latest_frame = None
            self._latest_frame_ts = 0.0
