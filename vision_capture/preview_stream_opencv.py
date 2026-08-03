from __future__ import annotations

import argparse
import base64
import json
import os
import sys
import threading
import time
from datetime import datetime
from dataclasses import dataclass, field
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from urllib.parse import parse_qs, urlparse

import cv2
import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from switch_connect.ui.terminal_select import choose_with_arrows
from vision_capture.adapter import (
    FFmpegCaptureSource,
    capture_device_busy_message,
    capture_error_may_indicate_device_busy,
    is_usb_capture_device_name,
    list_avfoundation_video_device_rows,
    rank_capture_device_name,
)


CAPTURE_PROFILES: List[Dict[str, object]] = [
    {"width": 1920, "height": 1080, "pixel_format": "mjpeg", "label": "1920x1080 / mjpeg"},
    {"width": 1280, "height": 720, "pixel_format": "mjpeg", "label": "1280x720 / mjpeg"},
    {"width": 1920, "height": 1080, "pixel_format": "uyvy422", "label": "1920x1080 / uyvy422"},
    {"width": 1280, "height": 720, "pixel_format": "uyvy422", "label": "1280x720 / uyvy422"},
    {"width": 1920, "height": 1080, "pixel_format": "nv12", "label": "1920x1080 / nv12"},
    {"width": 1280, "height": 720, "pixel_format": "nv12", "label": "1280x720 / nv12"},
    {"width": 1920, "height": 1080, "pixel_format": "yuyv422", "label": "1920x1080 / yuyv422"},
    {"width": 1280, "height": 720, "pixel_format": "yuyv422", "label": "1280x720 / yuyv422"},
]

DEFAULT_CONFIG: Dict[str, object] = {
    "device_name": "UGREEN 35287",
    "pick_device": False,
    "allow_non_usb": False,
    "preview_spec": "1920x1080 / mjpeg",
    "fps": 30,
    "window_title": "Capture Card Preview",
    "probe_seconds": 2.0,
    "frame_api_host": "127.0.0.1",
    "frame_api_port": 8765,
    "frame_jpeg_quality": 90,
    "out_dir": "vision_capture/debug",
    "prefix": "capture",
}


class _FFmpegPreviewCapture:
    def __init__(self, source: FFmpegCaptureSource):
        self._source = source
        self._last_ts = source.latest_frame_ts

    def read(self) -> Tuple[bool, Optional[np.ndarray]]:
        frame = self._source.read_next(after_ts=self._last_ts, timeout_seconds=0.5)
        if frame is None:
            return False, None
        self._last_ts = self._source.latest_frame_ts
        return True, frame

    def release(self) -> None:
        self._source.stop()

def _load_config(path: Path) -> Dict[str, object]:
    if not path.exists():
        return dict(DEFAULT_CONFIG)
    data = json.loads(path.read_text(encoding="utf-8"))
    out = dict(DEFAULT_CONFIG)
    if isinstance(data, dict):
        out.update(data)
    return out


def _save_config(path: Path, cfg: Dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(cfg, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _fresh_video_device_rows() -> List[Dict[str, str]]:
    return list_avfoundation_video_device_rows()


def _filtered_video_device_rows(prefer_usb_only: bool) -> List[Dict[str, str]]:
    rows = _fresh_video_device_rows()
    if prefer_usb_only:
        usb_rows = [row for row in rows if is_usb_capture_device_name(row["name"])]
        if usb_rows:
            rows = usb_rows
    return rows


def _device_label(device_id: str) -> str:
    value = str(device_id or "").strip()
    if not value:
        return ""
    if value.isdigit():
        for row in _fresh_video_device_rows():
            if row["index"] == value:
                return f"[{row['index']}] {row['name']}"
        return f"[{value}] <unknown>"
    for row in _fresh_video_device_rows():
        if row["name"] == value or row.get("device_id") == value:
            return f"[{row['index']}] {row['name']}"
    return value


def _pick_video_device(prefer_usb_only: bool) -> str:
    rows = _filtered_video_device_rows(prefer_usb_only=prefer_usb_only)
    if not rows:
        return ""
    options = [f"[{row['index']}] {row['name']}" for row in rows]
    picked = choose_with_arrows(options, "Select capture video device")
    if not picked:
        return ""
    for row in rows:
        label = f"[{row['index']}] {row['name']}"
        if picked == label:
            return str(row.get("device_id") or row["name"])
    return ""


def _resolve_device(args: argparse.Namespace) -> str:
    manual = (args.device_name or "").strip()
    if args.pick_device:
        return _pick_video_device(prefer_usb_only=not args.allow_non_usb)
    if manual.lower() == "invalid":
        return _pick_video_device(prefer_usb_only=not args.allow_non_usb)
    if manual:
        return manual
    rows = _filtered_video_device_rows(prefer_usb_only=not args.allow_non_usb)
    if not rows:
        return ""
    rows = sorted(rows, key=lambda row: rank_capture_device_name(row["name"]), reverse=True)
    return str(rows[0].get("device_id") or rows[0]["name"])


def _resolve_video_input_name(device_name: str) -> str:
    name = str(device_name or "").strip()
    if name.isdigit():
        return f"{name}:none"
    for row in _fresh_video_device_rows():
        if row["name"] == name:
            return f"{row['index']}:none"
    return f"{name}:none"


def _profile_from_spec(spec: str) -> Dict[str, object]:
    size, pixel_format = [part.strip() for part in spec.split("/", 1)]
    width_str, height_str = [part.strip() for part in size.lower().split("x", 1)]
    return {
        "width": int(width_str),
        "height": int(height_str),
        "pixel_format": pixel_format,
        "label": f"{int(width_str)}x{int(height_str)} / {pixel_format}",
    }


def _build_profiles(spec: str) -> List[Dict[str, object]]:
    if spec:
        return [_profile_from_spec(spec)]
    return [dict(profile) for profile in CAPTURE_PROFILES]


def _unique_image_path(out_dir: Path, prefix: str, idx: int) -> Path:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    return out_dir / f"{prefix}_{ts}_{idx:05d}.png"


def _resolve_video_index(device_name: str) -> Optional[int]:
    name = str(device_name or "").strip()
    if name.isdigit():
        return int(name)
    for row in _fresh_video_device_rows():
        if row["name"] == name or row.get("device_id") == name:
            return int(row["index"])
    return None


def _open_video_capture(
    device_name: str,
    profiles: List[Dict[str, object]],
    fps: int,
    timeout_seconds: float,
) -> Tuple[Optional[cv2.VideoCapture], Optional[Dict[str, object]], Optional[np.ndarray], str, Optional[str]]:
    last_error = ""
    for profile in profiles:
        source = FFmpegCaptureSource(
            device_name=str(device_name),
            width=int(profile["width"]),
            height=int(profile["height"]),
            fps=int(fps),
            pixel_format=str(profile["pixel_format"]),
            strict_usb_only=False,
        )
        first_frame = source.read(timeout_seconds=max(2.0, timeout_seconds))
        if first_frame is not None:
            return _FFmpegPreviewCapture(source), profile, first_frame, "", str(device_name)
        last_error = source.last_error or f"{profile['label']}: FRAME_TIMEOUT({timeout_seconds}s)"
        source.stop()

    return None, None, None, last_error, None


def _reopen_video_capture(
    current_cap: cv2.VideoCapture,
    device_name: str,
    profiles: List[Dict[str, object]],
    fps: int,
    timeout_seconds: float,
) -> Tuple[Optional[cv2.VideoCapture], Optional[Dict[str, object]], Optional[np.ndarray], str, Optional[str]]:
    current_cap.release()
    return _open_video_capture(
        device_name=device_name,
        profiles=profiles,
        fps=fps,
        timeout_seconds=timeout_seconds,
    )


@dataclass
class FrameState:
    jpeg_quality: int
    frame: Optional[np.ndarray] = None
    last_frame_ts: float = 0.0
    profile_label: str = ""
    device_name: str = ""
    frame_count: int = 0
    last_error: str = ""
    lock: threading.Lock = field(default_factory=threading.Lock)
    condition: threading.Condition = field(init=False)

    def __post_init__(self) -> None:
        self.condition = threading.Condition(self.lock)

    def update_frame(self, frame: np.ndarray) -> None:
        with self.condition:
            self.frame = frame.copy()
            self.last_frame_ts = time.time()
            self.frame_count += 1
            self.last_error = ""
            self.condition.notify_all()

    def snapshot_jpeg(self) -> Optional[bytes]:
        frame = self.snapshot_frame()
        if frame is None:
            return None
        ok, encoded = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), self.jpeg_quality])
        return encoded.tobytes() if ok else None

    def snapshot_frame(self) -> Optional[np.ndarray]:
        with self.lock:
            return None if self.frame is None else self.frame.copy()

    def snapshot_metadata(self) -> Dict[str, object]:
        with self.lock:
            return {
                "device_name": self.device_name,
                "profile_label": self.profile_label,
                "frame_count": self.frame_count,
                "last_frame_ts": self.last_frame_ts,
                "has_frame": self.frame is not None,
                "last_error": self.last_error,
            }

    def collect_future_frames(self, offsets: List[int], timeout_seconds: float) -> List[np.ndarray]:
        targets = sorted(int(v) for v in offsets if int(v) >= 0)
        if not targets:
            return []

        with self.condition:
            if self.frame is None:
                deadline = time.monotonic() + max(0.1, timeout_seconds)
                while self.frame is None:
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        raise TimeoutError(f"FRAME_WAIT_TIMEOUT({timeout_seconds}s)")
                    self.condition.wait(timeout=min(0.2, remaining))

            base_frame_id = self.frame_count
            captured: Dict[int, np.ndarray] = {}
            if 0 in targets and self.frame is not None:
                captured[0] = self.frame.copy()

            deadline = time.monotonic() + max(0.1, timeout_seconds)
            while any(offset not in captured for offset in targets):
                current_frame = None if self.frame is None else self.frame.copy()
                current_frame_id = self.frame_count
                for offset in targets:
                    if offset in captured:
                        continue
                    if current_frame is not None and current_frame_id >= base_frame_id + offset:
                        captured[offset] = current_frame.copy()

                if all(offset in captured for offset in targets):
                    break

                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise TimeoutError(f"FRAME_WAIT_TIMEOUT({timeout_seconds}s)")
                self.condition.wait(timeout=min(0.2, remaining))

        images: List[np.ndarray] = []
        for offset in targets:
            images.append(captured[offset].copy())
        return images

    def collect_future_jpegs(self, offsets: List[int], timeout_seconds: float) -> List[bytes]:
        images = self.collect_future_frames(offsets=offsets, timeout_seconds=timeout_seconds)
        out: List[bytes] = []
        for frame in images:
            ok, encoded = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), self.jpeg_quality])
            if not ok:
                raise RuntimeError("JPEG_ENCODE_FAILED")
            out.append(encoded.tobytes())
        return out


@dataclass
class FpsState:
    value: float = 0.0
    lock: threading.Lock = field(default_factory=threading.Lock)

    def set(self, fps: float) -> None:
        with self.lock:
            self.value = float(fps)

    def get_text(self) -> str:
        with self.lock:
            return f"{self.value:.1f}"


def _fps_worker(state: FrameState, fps_state: FpsState, stop_event: threading.Event) -> None:
    prev_count = 0
    prev_ts = time.monotonic()
    while not stop_event.wait(0.5):
        with state.lock:
            current_count = state.frame_count
        now = time.monotonic()
        elapsed = now - prev_ts
        if elapsed > 0:
            fps_state.set((current_count - prev_count) / elapsed)
        prev_count = current_count
        prev_ts = now


def _capture_worker(cap: cv2.VideoCapture, state: FrameState, stop_event: threading.Event) -> None:
    while not stop_event.is_set():
        ok, frame = cap.read()
        if ok and frame is not None and frame.size > 0:
            state.update_frame(frame)
            continue
        stop_event.wait(0.01)


def _start_capture_worker(
    cap: cv2.VideoCapture,
    state: FrameState,
) -> Tuple[threading.Event, threading.Thread]:
    stop_event = threading.Event()
    worker = threading.Thread(
        target=_capture_worker,
        args=(cap, state, stop_event),
        name="video-capture-worker",
        daemon=True,
    )
    worker.start()
    return stop_event, worker


def _make_handler(state: FrameState):
    def _run_batch_worker(offsets: List[int], timeout_seconds: float, result: Dict[str, object], done: threading.Event) -> None:
        try:
            images = state.collect_future_jpegs(offsets=offsets, timeout_seconds=timeout_seconds)
            result["images"] = [base64.b64encode(encoded).decode("ascii") for encoded in images]
        except Exception as exc:
            result["error"] = str(exc)
        finally:
            done.set()

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, format: str, *args) -> None:
            return

        def do_GET(self) -> None:
            parsed = urlparse(self.path)
            query = parse_qs(parsed.query)

            if parsed.path in {"/health", "/healthz"}:
                body = json.dumps(state.snapshot_metadata(), ensure_ascii=False).encode("utf-8")
                self.send_response(HTTPStatus.OK)
                self.send_header("Content-Type", "application/json; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
                return
            if parsed.path in {"/frame.jpg", "/frame.jpeg"}:
                body = state.snapshot_jpeg()
                if body is None:
                    self.send_error(HTTPStatus.SERVICE_UNAVAILABLE, "No frame available yet")
                    return
                self.send_response(HTTPStatus.OK)
                self.send_header("Content-Type", "image/jpeg")
                self.send_header("Cache-Control", "no-store")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
                return
            if parsed.path == "/frame.json":
                body = json.dumps(state.snapshot_metadata(), ensure_ascii=False).encode("utf-8")
                self.send_response(HTTPStatus.OK)
                self.send_header("Content-Type", "application/json; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
                return
            if parsed.path == "/batch_frames":
                offsets_param = query.get("offsets", [""])[0].strip()
                count_param = query.get("count", [""])[0].strip()
                step_param = query.get("step", ["1"])[0].strip()
                timeout_param = query.get("timeout", ["3.0"])[0].strip()

                try:
                    timeout_seconds = float(timeout_param)
                except ValueError:
                    self.send_error(HTTPStatus.BAD_REQUEST, "timeout must be a number")
                    return

                if offsets_param:
                    try:
                        offsets = [int(part.strip()) for part in offsets_param.split(",") if part.strip()]
                    except ValueError:
                        self.send_error(HTTPStatus.BAD_REQUEST, "offsets must be comma-separated integers")
                        return
                elif count_param:
                    try:
                        count = int(count_param)
                        step = int(step_param)
                    except ValueError:
                        self.send_error(HTTPStatus.BAD_REQUEST, "count and step must be integers")
                        return
                    if count <= 0 or step < 0:
                        self.send_error(HTTPStatus.BAD_REQUEST, "count must be > 0 and step must be >= 0")
                        return
                    offsets = [idx * step for idx in range(count)]
                else:
                    self.send_error(HTTPStatus.BAD_REQUEST, "use offsets=0,3,6 or count=6&step=1")
                    return

                result: Dict[str, object] = {"images": []}
                done = threading.Event()
                worker = threading.Thread(
                    target=_run_batch_worker,
                    args=(offsets, timeout_seconds, result, done),
                    name="batch-frame-worker",
                    daemon=True,
                )
                worker.start()
                done.wait(timeout=max(0.1, timeout_seconds) + 1.0)
                worker.join(timeout=0.1)

                if "error" in result:
                    body = json.dumps({"images": [], "error": str(result["error"])}, ensure_ascii=False).encode("utf-8")
                    self.send_response(HTTPStatus.REQUEST_TIMEOUT)
                    self.send_header("Content-Type", "application/json; charset=utf-8")
                    self.send_header("Content-Length", str(len(body)))
                    self.end_headers()
                    self.wfile.write(body)
                    return

                body = json.dumps(
                    {"images": result["images"]},
                    ensure_ascii=False,
                ).encode("utf-8")
                self.send_response(HTTPStatus.OK)
                self.send_header("Content-Type", "application/json; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
                return
            self.send_error(HTTPStatus.NOT_FOUND, "Supported paths: /health, /frame.jpg, /frame.json, /batch_frames")

    return Handler


def _save_burst_frames_worker(
    state: FrameState,
    out_dir: Path,
    prefix: str,
    start_idx: int,
    count: int,
    timeout_seconds: float,
) -> None:
    try:
        frames = state.collect_future_frames(offsets=list(range(count)), timeout_seconds=timeout_seconds)
    except Exception as exc:
        print(f"[burst-error] {exc}")
        return

    for offset, frame in enumerate(frames):
        path = _unique_image_path(out_dir, prefix, start_idx + offset)
        cv2.imwrite(str(path), frame)
        print(f"[saved] {path}")


def _overlay_status(
    frame: np.ndarray,
    requested_device_label: str,
    saved_count: int,
    fps_text: str,
    profile_label: str,
) -> None:
    left_lines = [
        f"Requested: {requested_device_label}",
        f"Saved: {saved_count}",
        "Keys: Enter save PNG, B burst 30 PNG, R reselect source, Esc/close window/Ctrl+C quit",
    ]
    right_lines = [
        f"Capture: {profile_label}",
        f"FPS: {fps_text}",
    ]
    for idx, text in enumerate(left_lines):
        y = 30 + idx * 28
        cv2.putText(frame, text, (16, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (30, 30, 30), 4, cv2.LINE_AA)
        cv2.putText(frame, text, (16, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 180), 1, cv2.LINE_AA)
    frame_w = frame.shape[1]
    for idx, text in enumerate(right_lines):
        y = 30 + idx * 28
        (text_w, _), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 1)
        x = max(16, frame_w - text_w - 16)
        cv2.putText(frame, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (30, 30, 30), 4, cv2.LINE_AA)
        cv2.putText(frame, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 180), 1, cv2.LINE_AA)


def main() -> int:
    parser = argparse.ArgumentParser(description="Open an OpenCV preview window and expose the latest frame over HTTP.")
    parser.add_argument("--config", default="vision_capture/capture_config.json", help="json config file path")
    parser.add_argument("--device-name", default=None, help="manual AVFoundation device name")
    parser.add_argument("--pick-device", action="store_true", help="choose device via arrow keys")
    parser.add_argument("--allow-non-usb", action="store_true", help="allow non-capture-card cameras")
    parser.add_argument("--spec", default=None, help="single capture spec like '1920x1080 / uyvy422'")
    parser.add_argument("--fps", type=int, default=None, help="capture frame rate")
    parser.add_argument("--window-title", default=None, help="OpenCV window title")
    parser.add_argument("--probe-seconds", type=float, default=None, help="timeout for opening first frame")
    parser.add_argument("--host", default=None, help="HTTP bind host")
    parser.add_argument("--port", type=int, default=None, help="HTTP bind port")
    parser.add_argument("--jpeg-quality", type=int, default=None, help="JPEG quality for /frame.jpg")
    parser.add_argument("--list-profiles", action="store_true", help="print supported capture profiles and exit")
    args = parser.parse_args()

    cfg_path = Path(args.config)
    if not cfg_path.is_absolute():
        cfg_path = REPO_ROOT / cfg_path
    cfg = _load_config(cfg_path)

    if args.list_profiles:
        for profile in CAPTURE_PROFILES:
            print(profile["label"])
        return 0

    if args.device_name is None:
        args.device_name = str(cfg.get("device_name", DEFAULT_CONFIG["device_name"]))
    if not args.pick_device:
        args.pick_device = bool(cfg.get("pick_device", DEFAULT_CONFIG["pick_device"]))
    if not args.allow_non_usb:
        args.allow_non_usb = bool(cfg.get("allow_non_usb", DEFAULT_CONFIG["allow_non_usb"]))
    if args.spec is None:
        args.spec = str(cfg.get("preview_spec", DEFAULT_CONFIG["preview_spec"]))
    if args.fps is None:
        args.fps = int(cfg.get("fps", DEFAULT_CONFIG["fps"]))
    if args.window_title is None:
        args.window_title = str(cfg.get("window_title", DEFAULT_CONFIG["window_title"]))
    if args.probe_seconds is None:
        args.probe_seconds = float(cfg.get("probe_seconds", DEFAULT_CONFIG["probe_seconds"]))
    if args.host is None:
        args.host = str(cfg.get("frame_api_host", DEFAULT_CONFIG["frame_api_host"]))
    if args.port is None:
        args.port = int(cfg.get("frame_api_port", DEFAULT_CONFIG["frame_api_port"]))
    if args.jpeg_quality is None:
        args.jpeg_quality = int(cfg.get("frame_jpeg_quality", DEFAULT_CONFIG["frame_jpeg_quality"]))
    out_dir = Path(str(cfg.get("out_dir", DEFAULT_CONFIG["out_dir"])))
    if not out_dir.is_absolute():
        out_dir = REPO_ROOT / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    prefix = str(cfg.get("prefix", DEFAULT_CONFIG["prefix"]))

    device_name = _resolve_device(args)
    if not device_name:
        print("No capture device selected/detected.")
        return 2

    profiles = _build_profiles(args.spec or "")
    cap, active_profile, first_frame, error, actual_device_name = _open_video_capture(
        device_name=device_name,
        profiles=profiles,
        fps=args.fps,
        timeout_seconds=args.probe_seconds,
    )
    if cap is None or active_profile is None or first_frame is None or not actual_device_name:
        cfg["pick_device"] = False
        _save_config(cfg_path, cfg)
        print(f"Config preserved: device_name={cfg.get('device_name', '')} ({cfg_path})")
        print(f"Unable to open preview stream. last_error={error}")
        if capture_error_may_indicate_device_busy(error):
            print(capture_device_busy_message(error))
        return 1

    state = FrameState(jpeg_quality=max(30, min(100, args.jpeg_quality)))
    state.device_name = actual_device_name
    state.profile_label = str(active_profile["label"])
    state.update_frame(first_frame)
    api_url = f"http://{args.host}:{args.port}"
    requested_device_label = _device_label(device_name)
    actual_device_label = _device_label(actual_device_name)

    server = ThreadingHTTPServer((args.host, args.port), _make_handler(state))
    server_thread = threading.Thread(target=server.serve_forever, name="frame-api-server", daemon=True)
    server_thread.start()
    print(f"Requested device: {requested_device_label}")
    print(f"Actual device: {actual_device_label}")
    print(f"Input index: {_resolve_video_index(device_name)}")
    print(f"Streaming: {active_profile['label']}")
    print(f"Frame API: {api_url}/frame.jpg")
    print(f"Health API: {api_url}/health")
    print(f"Screenshot output: {out_dir}")
    saved_count = 0
    burst_timeout_seconds = max(3.0, 30.0 / max(1, args.fps) + 2.0)
    fps_state = FpsState()
    fps_stop_event = threading.Event()
    fps_thread = threading.Thread(
        target=_fps_worker,
        args=(state, fps_state, fps_stop_event),
        name="fps-worker",
        daemon=True,
    )
    fps_thread.start()
    capture_stop_event, capture_thread = _start_capture_worker(cap, state)

    try:
        cv2.namedWindow(args.window_title, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(
            args.window_title,
            min(int(active_profile["width"]), 1280),
            min(int(active_profile["height"]), 720),
        )

        while True:
            with state.lock:
                latest_frame_ts = state.last_frame_ts
            frame_age = max(0.0, time.time() - latest_frame_ts) if latest_frame_ts > 0 else float("inf")
            if frame_age >= max(1.0, args.probe_seconds):
                state.last_error = f"FRAME_TIMEOUT({args.probe_seconds}s)"
                print(f"No frame received within {args.probe_seconds}s. Check input signal/resolution.")
                break

            display_frame = state.snapshot_frame()
            if display_frame is None:
                continue
            _overlay_status(
                display_frame,
                requested_device_label,
                saved_count,
                fps_state.get_text(),
                str(active_profile["label"]),
            )
            cv2.imshow(args.window_title, display_frame)
            if cv2.getWindowProperty(args.window_title, cv2.WND_PROP_VISIBLE) < 1:
                break
            key_ex = cv2.waitKeyEx(1)
            key = key_ex & 0xFF
            if key == 27:
                break
            if key in (ord("b"), ord("B")):
                start_idx = saved_count + 1
                saved_count += 30
                worker = threading.Thread(
                    target=_save_burst_frames_worker,
                    args=(state, out_dir, prefix, start_idx, 30, burst_timeout_seconds),
                    name="burst-save-worker",
                    daemon=True,
                )
                worker.start()
                print("[burst] saving next 30 frames")
                continue
            if key in (ord("r"), ord("R")):
                cfg["device_name"] = "invalid"
                cfg["pick_device"] = False
                _save_config(cfg_path, cfg)
                print(f"[reselect] config marked invalid: {cfg_path}")
                picked_device = _pick_video_device(prefer_usb_only=not args.allow_non_usb)
                if not picked_device:
                    print("[reselect] cancelled")
                    continue

                capture_stop_event.set()
                capture_thread.join(timeout=2.0)
                next_cap, next_profile, next_frame, reopen_error, next_actual_device_name = _reopen_video_capture(
                    current_cap=cap,
                    device_name=picked_device,
                    profiles=profiles,
                    fps=args.fps,
                    timeout_seconds=args.probe_seconds,
                )
                if (
                    next_cap is None
                    or next_profile is None
                    or next_frame is None
                    or not next_actual_device_name
                ):
                    print(f"[reselect] failed: {reopen_error}")
                    cfg["pick_device"] = False
                    _save_config(cfg_path, cfg)
                    print(f"[reselect] config preserved: device_name={cfg.get('device_name', '')} ({cfg_path})")
                    print("[reselect] exiting preview because selected device could not be opened")
                    break

                cap = next_cap
                active_profile = next_profile
                device_name = picked_device
                actual_device_name = next_actual_device_name
                requested_device_label = _device_label(device_name)
                actual_device_label = _device_label(actual_device_name)
                state.device_name = actual_device_name
                state.profile_label = str(active_profile["label"])
                state.update_frame(next_frame)
                capture_stop_event, capture_thread = _start_capture_worker(cap, state)
                cfg["device_name"] = device_name
                cfg["pick_device"] = False
                _save_config(cfg_path, cfg)
                print(f"[reselect] requested={requested_device_label}")
                print(f"[reselect] actual={actual_device_label}")
                print(f"[reselect] config updated: {cfg_path}")
                continue
            if key in (10, 13):
                saved_count += 1
                path = _unique_image_path(out_dir, prefix, saved_count)
                frame_to_save = state.snapshot_frame()
                if frame_to_save is not None:
                    cv2.imwrite(str(path), frame_to_save)
                print(f"[saved] {path}")
                continue
    finally:
        capture_stop_event.set()
        capture_thread.join(timeout=2.0)
        fps_stop_event.set()
        fps_thread.join(timeout=0.5)
        server.shutdown()
        server.server_close()
        cap.release()
        cv2.destroyAllWindows()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
