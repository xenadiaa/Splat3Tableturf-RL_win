#!/usr/bin/env python3
"""Capture-card preview tool for Windows DirectShow."""

from __future__ import annotations

import argparse
import os
import select
import shutil
import signal
import subprocess
import time
from pathlib import Path

import cv2
import numpy as np


DEFAULT_VIDEO_DEVICE = "0"
DEFAULT_PIXEL_FORMAT = "uyvy422"
DEFAULT_SIZE = "1280x720"
DEFAULT_FPS = 30


def parse_size(size: str) -> tuple[int, int]:
    width_str, height_str = size.lower().split("x", 1)
    return int(width_str), int(height_str)


def run_list_devices() -> int:
    cmd = ["ffmpeg", "-f", "dshow", "-list_devices", "true", "-i", "dummy"]
    # FFmpeg may mix UTF-8 and ANSI bytes on Windows. Avoid strict GBK
    # decoding in subprocess reader threads, which can erase the device list.
    proc = subprocess.run(cmd, capture_output=True)
    stdout = (proc.stdout or b"").decode("utf-8", errors="replace")
    stderr = (proc.stderr or b"").decode("utf-8", errors="replace")
    merged = f"{stdout}\n{stderr}".strip()
    if merged:
        print(merged)
    return proc.returncode


def build_ffmpeg_cmd(
    video_device: str,
    pixel_format: str,
    size: str,
    fps: int,
) -> list[str]:
    input_name = f"video={video_device}"
    return [
        "ffmpeg",
        "-loglevel",
        "error",
        "-f",
        "dshow",
        "-pixel_format",
        pixel_format,
        "-framerate",
        str(fps),
        "-video_size",
        size,
        "-i",
        input_name,
        "-pix_fmt",
        "bgr24",
        "-f",
        "rawvideo",
        "-",
    ]


def main() -> int:
    parser = argparse.ArgumentParser(description="Preview DirectShow capture-card video stream")
    parser.add_argument("--list-devices", action="store_true", help="List DirectShow devices and exit")
    parser.add_argument("--video-device", default=DEFAULT_VIDEO_DEVICE, help="Video device name")
    parser.add_argument("--pixel-format", default=DEFAULT_PIXEL_FORMAT, help="Capture pixel format")
    parser.add_argument("--size", default=DEFAULT_SIZE, help="Capture size such as 1280x720")
    parser.add_argument("--fps", type=int, default=DEFAULT_FPS, help="Capture frame rate")
    parser.add_argument("--save-frame", default="", help="Save latest frame on quit")
    parser.add_argument("--timeout-seconds", type=float, default=8.0, help="Exit if no frame arrives in N seconds")
    args = parser.parse_args()

    if not shutil.which("ffmpeg"):
        raise SystemExit("ffmpeg not found. Install ffmpeg first.")
    if args.list_devices:
        return run_list_devices()

    width, height = parse_size(args.size)
    frame_bytes = width * height * 3
    ffmpeg_cmd = build_ffmpeg_cmd(
        video_device=args.video_device,
        pixel_format=args.pixel_format,
        size=args.size,
        fps=args.fps,
    )
    process = subprocess.Popen(
        ffmpeg_cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        bufsize=10**8,
    )

    latest_frame: np.ndarray | None = None
    last_frame_ts = time.monotonic()
    frame_buffer = bytearray()

    def _stop_ffmpeg() -> None:
        if process.poll() is None:
            process.send_signal(signal.SIGTERM)
            try:
                process.wait(timeout=1)
            except subprocess.TimeoutExpired:
                process.kill()

    try:
        while True:
            if process.stdout is None:
                break

            ready, _, _ = select.select([process.stdout], [], [], 0.2)
            if not ready:
                if process.poll() is not None:
                    break
                if time.monotonic() - last_frame_ts >= args.timeout_seconds:
                    print(f"No frame received within {args.timeout_seconds}s. Check input signal/resolution.")
                    break
                continue

            chunk = os.read(process.stdout.fileno(), frame_bytes - len(frame_buffer))
            if not chunk:
                if process.poll() is not None:
                    break
                continue
            frame_buffer.extend(chunk)
            if len(frame_buffer) < frame_bytes:
                continue

            raw = bytes(frame_buffer[:frame_bytes])
            del frame_buffer[:frame_bytes]

            if len(raw) != frame_bytes:
                break

            frame = np.frombuffer(raw, dtype=np.uint8).reshape((height, width, 3))
            latest_frame = frame
            last_frame_ts = time.monotonic()
            cv2.imshow("Capture Card Preview (Esc/Q to quit)", frame)
            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord("q"), ord("Q")):
                break
    finally:
        if args.save_frame and latest_frame is not None:
            output_path = Path(args.save_frame).expanduser().resolve()
            output_path.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(output_path.as_posix(), latest_frame)
            print(f"Saved frame: {output_path}")

        _stop_ffmpeg()
        cv2.destroyAllWindows()

    if process.returncode not in (0, None):
        err = (process.stderr.read().decode("utf-8", errors="ignore") if process.stderr else "").strip()
        if err:
            print(err)
        return process.returncode
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
