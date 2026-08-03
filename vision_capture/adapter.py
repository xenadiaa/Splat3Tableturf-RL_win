from __future__ import annotations

import contextlib
import json
import os
import re
import select
import shutil
import subprocess
import threading
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Optional

import cv2
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


def capture_error_may_indicate_device_busy(error: object) -> bool:
    """Return whether a capture failure commonly means another app owns the device."""
    message = str(error or "").casefold()
    busy_markers = (
        "i/o error",
        "device or resource busy",
        "resource busy",
        "device is busy",
        "device in use",
        "being used by another process",
        "used by another process",
        "cannot start graph",
        "0x800700aa",
    )
    return any(marker in message for marker in busy_markers)


def capture_device_busy_message(error: object) -> str:
    detail = str(error or "").strip()
    message = (
        "视频采集设备可能正被其他程序占用，当前进程无法打开其 I/O。"
        "请关闭 OBS、VLC、相机、浏览器视频页面或其他采集/预览程序后重试。"
    )
    return f"{message} 原始错误：{detail}" if detail else message


def resolve_ffmpeg_tool(tool_name: str = "ffmpeg") -> str:
    """Resolve FFmpeg tools from PATH or the default WinGet install roots."""
    executable_name = f"{str(tool_name).strip()}.exe" if os.name == "nt" else str(tool_name).strip()
    resolved = shutil.which(executable_name) or shutil.which(str(tool_name).strip())
    if resolved:
        return str(resolved)
    if os.name != "nt":
        return ""

    roots: List[Path] = []
    local_app_data = str(os.environ.get("LOCALAPPDATA", "") or "").strip()
    if local_app_data:
        local_root = Path(local_app_data) / "Microsoft" / "WinGet"
        direct_link = local_root / "Links" / executable_name
        if direct_link.is_file():
            return str(direct_link)
        roots.append(local_root / "Packages")

    program_files = str(os.environ.get("ProgramFiles", "") or "").strip()
    if program_files:
        roots.append(Path(program_files) / "WinGet" / "Packages")

    for root in roots:
        if not root.is_dir():
            continue
        with contextlib.suppress(Exception):
            for package_dir in root.glob("Gyan.FFmpeg*"):
                for candidate in package_dir.rglob(executable_name):
                    if candidate.is_file():
                        return str(candidate)
    return ""


def _list_ffmpeg_video_devices_text() -> str:
    ffmpeg_executable = resolve_ffmpeg_tool("ffmpeg")
    if not ffmpeg_executable:
        raise RuntimeError(
            "未找到 ffmpeg.exe。请执行 `winget install -e --id Gyan.FFmpeg`，"
            "安装后重新打开终端或重启 Windows。"
        )
    cmd = [ffmpeg_executable, "-f", "dshow", "-list_devices", "true", "-i", "dummy"]
    # FFmpeg output may contain bytes that are invalid in Windows' active GBK
    # code page. Keep subprocess in binary mode and decode tolerantly here.
    proc = subprocess.run(cmd, capture_output=True)
    stderr = (proc.stderr or b"").decode("utf-8", errors="replace")
    stdout = (proc.stdout or b"").decode("utf-8", errors="replace")
    return stderr + stdout


def _parse_dshow_video_device_rows(text: str) -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []
    in_video_section = False
    pending_video_row: Optional[int] = None
    for line in str(text or "").splitlines():
        lower = line.lower()
        if "directshow video devices" in lower:
            in_video_section = True
            continue
        if "directshow audio devices" in lower:
            in_video_section = False
            pending_video_row = None
            continue
        if "alternative name" in lower:
            match = re.search(r'"([^\"]+)"', line)
            if match and pending_video_row is not None:
                rows[pending_video_row]["device_id"] = match.group(1).strip()
            pending_video_row = None
            continue

        match = re.search(r'"([^\"]+)"', line)
        if not match:
            continue
        suffix = line[match.end() :].lower()
        explicitly_video = bool(re.search(r"\(\s*video(?:\s*[,/)])", suffix))
        if not in_video_section and not explicitly_video:
            continue

        name = match.group(1).strip()
        if not name:
            continue
        rows.append(
            {
                "index": str(len(rows)),
                "name": name,
                "device_id": name,
            }
        )
        pending_video_row = len(rows) - 1
    return rows


def _parse_dshow_video_device_names(text: str) -> List[str]:
    names: List[str] = []
    seen = set()
    for row in _parse_dshow_video_device_rows(text):
        name = row["name"]
        if name in seen:
            continue
        seen.add(name)
        names.append(name)
    return names


def list_avfoundation_video_devices() -> List[str]:
    """Return DirectShow video device names from ffmpeg listing."""
    return _parse_dshow_video_device_names(_list_ffmpeg_video_devices_text())


def list_avfoundation_video_device_rows() -> List[Dict[str, str]]:
    return _parse_dshow_video_device_rows(_list_ffmpeg_video_devices_text())


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
        self._mjpeg_transport = str(pixel_format or "").strip().lower() in {"mjpeg", "mjpg"}
        self.strict_usb_only = strict_usb_only
        self._proc: Optional[subprocess.Popen] = None
        self._frame_bytes = self.width * self.height * 3
        self._rx_buffer = bytearray()
        self.last_error: Optional[str] = None
        self._reader_thread: Optional[threading.Thread] = None
        self._stderr_thread: Optional[threading.Thread] = None
        self._stop_reader = threading.Event()
        self._stderr_lock = threading.Lock()
        self._stderr_buffer = bytearray()
        self._frame_lock = threading.Lock()
        self._latest_frame: Optional[np.ndarray] = None
        self._latest_frame_ts: float = 0.0
        self._active_capture_spec: Optional[Dict[str, object]] = None

    def _resolve_video_input_name(self) -> str:
        name = str(self.device_name or "").strip()
        if name.lower().startswith("@device_"):
            return f"video={name}"
        if name.isdigit():
            rows = list_avfoundation_video_device_rows()
            for row in rows:
                if row["index"] == name:
                    return f"video={row.get('device_id') or row['name']}"
            return f"video={name}"
        rows = list_avfoundation_video_device_rows()
        for row in rows:
            if row.get("device_id") == name:
                return f"video={name}"
        for row in rows:
            if row["name"] == name:
                return f"video={row.get('device_id') or row['name']}"
        return f"video={name}"

    def start(self) -> None:
        if self._proc is not None:
            return
        with self._stderr_lock:
            self._stderr_buffer.clear()
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
            "transport": "mjpeg_copy" if self._mjpeg_transport else "rawvideo",
        }
        ffmpeg_executable = resolve_ffmpeg_tool("ffmpeg")
        if not ffmpeg_executable:
            self.last_error = "FFMPEG_NOT_FOUND"
            raise RuntimeError(
                "未找到 ffmpeg.exe。请执行 `winget install -e --id Gyan.FFmpeg`，"
                "安装后重新打开终端或重启 Windows。"
            )
        cmd = [
            ffmpeg_executable,
            "-hide_banner",
            "-loglevel",
            "error",
            "-nostdin",
            "-f",
            "dshow",
            "-rtbufsize",
            "8M",
        ]
        if self._mjpeg_transport:
            cmd.extend(["-vcodec", "mjpeg"])
        elif self.pixel_format:
            cmd.extend(["-pixel_format", self.pixel_format])
        cmd.extend([
            "-framerate",
            str(self.fps),
            "-video_size",
            f"{self.width}x{self.height}",
            "-i",
            inp,
            "-an",
        ])
        if self._mjpeg_transport:
            cmd.extend(["-c:v", "copy", "-f", "mjpeg", "pipe:1"])
        else:
            cmd.extend(["-pix_fmt", "bgr24", "-f", "rawvideo", "pipe:1"])
        self._proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            stdin=subprocess.DEVNULL,
            bufsize=(1024 * 1024 if self._mjpeg_transport else self._frame_bytes * 2),
        )
        if os.name != "nt":
            if self._proc.stdout is not None:
                with contextlib.suppress(Exception):
                    os.set_blocking(self._proc.stdout.fileno(), False)
            if self._proc.stderr is not None:
                with contextlib.suppress(Exception):
                    os.set_blocking(self._proc.stderr.fileno(), False)
        self._stop_reader.clear()
        self._reader_thread = threading.Thread(target=self._reader_loop, name="ffmpeg-capture-reader", daemon=True)
        self._stderr_thread = threading.Thread(target=self._stderr_loop, name="ffmpeg-stderr-reader", daemon=True)
        self._reader_thread.start()
        self._stderr_thread.start()

    def restart_with(self, width: int, height: int, pixel_format: str) -> None:
        self.stop()
        self.width = int(width)
        self.height = int(height)
        self.pixel_format = str(pixel_format)
        self._mjpeg_transport = self.pixel_format.strip().lower() in {"mjpeg", "mjpg"}
        self._frame_bytes = self.width * self.height * 3
        self._rx_buffer = bytearray()
        self.start()

    @property
    def active_capture_spec(self) -> Optional[Dict[str, object]]:
        if self._active_capture_spec is None:
            return None
        return dict(self._active_capture_spec)

    def _stderr_tail(self, max_bytes: int = 4096) -> str:
        with self._stderr_lock:
            payload = bytes(self._stderr_buffer[-max(1, int(max_bytes)) :])
        return payload.decode("utf-8", errors="replace").strip()

    def _stderr_loop(self) -> None:
        proc = self._proc
        if proc is None or proc.stderr is None:
            return
        stream = proc.stderr
        while not self._stop_reader.is_set():
            try:
                if hasattr(stream, "read1"):
                    chunk = stream.read1(4096)
                else:
                    chunk = stream.read(4096)
            except BlockingIOError:
                self._stop_reader.wait(0.02)
                continue
            except (OSError, ValueError):
                return
            if not chunk:
                if proc.poll() is not None:
                    return
                self._stop_reader.wait(0.02)
                continue
            with self._stderr_lock:
                self._stderr_buffer.extend(chunk)
                if len(self._stderr_buffer) > 16384:
                    del self._stderr_buffer[:-16384]

    def _read_from_pipe_once(self, timeout_seconds: float) -> bool:
        if self._proc is None or self._proc.stdout is None:
            return False
        if os.name == "nt":
            try:
                if self._mjpeg_transport and hasattr(self._proc.stdout, "read1"):
                    chunk = self._proc.stdout.read1(1024 * 1024)
                else:
                    chunk = self._proc.stdout.read(self._frame_bytes)
            except (OSError, ValueError):
                return False
            if not chunk:
                return False
            self._rx_buffer.extend(chunk)
            return True
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
        if self._mjpeg_transport:
            latest_packet: Optional[bytes] = None
            while True:
                start = self._rx_buffer.find(b"\xff\xd8")
                if start < 0:
                    if len(self._rx_buffer) > 1:
                        del self._rx_buffer[:-1]
                    break
                if start > 0:
                    del self._rx_buffer[:start]
                end = self._rx_buffer.find(b"\xff\xd9", 2)
                if end < 0:
                    break
                latest_packet = bytes(self._rx_buffer[: end + 2])
                del self._rx_buffer[: end + 2]
            if latest_packet is None:
                return None
            frame = cv2.imdecode(np.frombuffer(latest_packet, dtype=np.uint8), cv2.IMREAD_COLOR)
            return frame if frame is not None and frame.size > 0 else None
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
                proc = self._proc
                tail = self._stderr_tail()
                if proc is not None and proc.poll() is not None:
                    self.last_error = f"FFMPEG_EXITED({proc.returncode}) {tail}".strip()
                else:
                    self.last_error = f"FRAME_TIMEOUT({timeout_seconds}s) {tail}".strip()
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
                proc = self._proc
                tail = self._stderr_tail()
                if proc is not None and proc.poll() is not None:
                    self.last_error = f"FFMPEG_EXITED({proc.returncode}) {tail}".strip()
                else:
                    self.last_error = f"FRAME_TIMEOUT({timeout_seconds}s) {tail}".strip()
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
        try:
            proc.terminate()
        except Exception:
            pass
        if self._reader_thread is not None:
            self._reader_thread.join(timeout=1.0)
            self._reader_thread = None
        if self._stderr_thread is not None:
            self._stderr_thread.join(timeout=1.0)
            self._stderr_thread = None

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
