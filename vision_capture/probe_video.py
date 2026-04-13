from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

import cv2

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from vision_capture.adapter import FFmpegCaptureSource, list_avfoundation_video_devices
from vision_capture.adapter import auto_detect_capture_device_name
from vision_capture.adapter import is_usb_capture_device_name
from switch_connect.ui.terminal_select import choose_with_arrows


def _unique_image_path(out_dir: Path, prefix: str, idx: int) -> Path:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    return out_dir / f"{prefix}_{ts}_{idx:05d}.png"


def _frame_md5(frame) -> str:
    return hashlib.md5(frame.tobytes()).hexdigest()


def probe_device_name(
    device_name: str,
    out_dir: Path,
    width: int,
    height: int,
    fps: int,
    shots: int,
    interval_ms: int,
    warmup_seconds: float,
    prefix: str,
    skip_duplicates: bool,
    keep_active: bool,
    keep_active_seconds: float,
    debug: bool,
    read_timeout_seconds: float,
    first_frame_timeout_seconds: float,
    restart_per_shot: bool,
) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    src = FFmpegCaptureSource(device_name=device_name, width=width, height=height, fps=fps)
    first_frame = None
    saved_files = []
    saved_hashes = []
    skipped_duplicate = 0
    last_hash: Optional[str] = None
    interrupted = False
    timeout_exit = False
    error: Optional[str] = None
    first_frame_deadline = time.time() + max(0.1, first_frame_timeout_seconds)
    try:
        try:
            # Warm up capture to pass initial boot/no-signal stage.
            warmup_until = time.time() + max(0.0, warmup_seconds)
            while time.time() < warmup_until:
                frame = src.read_latest(timeout_seconds=read_timeout_seconds)
                if frame is not None and first_frame is None:
                    first_frame = frame
                if first_frame is None and time.time() > first_frame_deadline:
                    timeout_exit = True
                    error = src.last_error or "FIRST_FRAME_TIMEOUT_DURING_WARMUP"
                    break

            if timeout_exit:
                raise TimeoutError(error or "FIRST_FRAME_TIMEOUT")

            shot_index = 0
            while True:
                if not debug and shot_index >= max(0, shots):
                    break
                if restart_per_shot:
                    src.stop()
                    src = FFmpegCaptureSource(device_name=device_name, width=width, height=height, fps=fps)
                    frame = src.read_latest(timeout_seconds=read_timeout_seconds)
                else:
                    frame = src.read_latest(timeout_seconds=read_timeout_seconds)
                if frame is None:
                    if first_frame is None and time.time() > first_frame_deadline:
                        timeout_exit = True
                        error = src.last_error or "FIRST_FRAME_TIMEOUT"
                        break
                    continue
                shot_index += 1
                if first_frame is None:
                    first_frame = frame

                frame_hash = _frame_md5(frame)
                if skip_duplicates and last_hash == frame_hash:
                    skipped_duplicate += 1
                else:
                    path = _unique_image_path(out_dir, prefix, shot_index)
                    cv2.imwrite(str(path), frame)
                    saved_files.append(str(path))
                    saved_hashes.append(frame_hash)
                    last_hash = frame_hash

                if interval_ms > 0:
                    time.sleep(interval_ms / 1000.0)

            if timeout_exit:
                raise TimeoutError(error or "FIRST_FRAME_TIMEOUT")

            # Keep capture card activated after snapshot phase (non-debug mode).
            if keep_active and not debug:
                if keep_active_seconds > 0:
                    end_ts = time.time() + keep_active_seconds
                    while time.time() < end_ts:
                        _ = src.read_latest(timeout_seconds=read_timeout_seconds)
                else:
                    while True:
                        _ = src.read_latest(timeout_seconds=read_timeout_seconds)
        except TimeoutError:
            pass
        except KeyboardInterrupt:
            interrupted = True
    finally:
        src.stop()

    row = {
        "device_name": device_name,
        "opened": bool(first_frame is not None),
        "debug": debug,
        "shots_requested": shots,
        "shots_saved": len(saved_files),
        "skipped_duplicate": skipped_duplicate,
        "keep_active": keep_active,
        "keep_active_seconds": keep_active_seconds,
        "restart_per_shot": restart_per_shot,
        "interrupted": interrupted,
        "timeout_exit": timeout_exit,
        "error": error,
        "saved_files": saved_files,
        "saved_hashes": saved_hashes,
    }
    if error is None and src.last_error and first_frame is None:
        row["error"] = src.last_error
    if len(saved_files) == 0:
        if first_frame is None:
            row["reason"] = "NO_FIRST_FRAME_NO_IMAGE_SAVED"
        elif skipped_duplicate > 0:
            row["reason"] = "ALL_FRAMES_SKIPPED_AS_DUPLICATES"
        else:
            row["reason"] = "NO_IMAGE_SAVED"
    if first_frame is not None:
        h, w = first_frame.shape[:2]
        row["shape"] = [int(h), int(w), int(first_frame.shape[2] if len(first_frame.shape) > 2 else 1)]
        row["mean_bgr"] = [float(first_frame[:, :, c].mean()) for c in range(3)]
    return row


def main() -> int:
    p = argparse.ArgumentParser(description="Probe USB capture video source (DirectShow by device name)")
    p.add_argument("--device-name", default="", help="manual device name; empty means auto-detect")
    p.add_argument("--auto-device", action="store_true", help="force auto-detect capture device")
    p.add_argument("--pick-device", action="store_true", help="choose capture device via arrow keys")
    p.add_argument("--width", type=int, default=1920)
    p.add_argument("--height", type=int, default=1080)
    p.add_argument("--fps", type=int, default=30)
    p.add_argument("--out-dir", default="vision_capture/debug")
    p.add_argument("--shots", type=int, default=10, help="number of snapshots to save")
    p.add_argument("--interval-ms", type=int, default=1000, help="gap between snapshots")
    p.add_argument("--warmup-seconds", type=float, default=5.0, help="warmup before saving snapshots")
    p.add_argument("--prefix", default="capture")
    p.add_argument("--debug", action="store_true", help="continuous screenshot mode until Ctrl+C")
    p.add_argument("--no-skip-duplicates", action="store_true")
    p.add_argument("--read-timeout-seconds", type=float, default=5.0, help="timeout for one frame read")
    p.add_argument(
        "--no-restart-per-shot",
        action="store_true",
        help="use one persistent ffmpeg session (default is restart per shot)",
    )
    p.add_argument(
        "--first-frame-timeout-seconds",
        type=float,
        default=12.0,
        help="exit with error if first frame is not received within this time",
    )
    p.add_argument("--no-keep-active", action="store_true", help="stop capture after snapshot phase")
    p.add_argument(
        "--keep-active-seconds",
        type=float,
        default=0.0,
        help="keep capture alive after snapshots; 0 means until Ctrl+C",
    )
    p.add_argument("--dump-devices", action="store_true")
    args = p.parse_args()

    if args.dump_devices:
        print(json.dumps({"video_devices": list_avfoundation_video_devices()}, ensure_ascii=False, indent=2))

    selected_device = args.device_name.strip()
    if args.pick_device:
        all_devices = list_avfoundation_video_devices()
        usb_devices = [d for d in all_devices if is_usb_capture_device_name(d)]
        pick_pool = usb_devices if usb_devices else all_devices
        if not pick_pool:
            row = {
                "device_name": "",
                "opened": False,
                "error": "NO_VIDEO_DEVICE_FOUND",
                "reason": "PICK_DEVICE_FAILED",
            }
            print(json.dumps(row, ensure_ascii=False, indent=2))
            return 2
        picked = choose_with_arrows(pick_pool, "Select capture video device")
        if not picked:
            row = {
                "device_name": "",
                "opened": False,
                "error": "DEVICE_SELECTION_CANCELLED",
                "reason": "PICK_DEVICE_CANCELLED",
            }
            print(json.dumps(row, ensure_ascii=False, indent=2))
            return 2
        selected_device = picked
    elif args.auto_device or not selected_device:
        selected_device = auto_detect_capture_device_name(prefer_usb=True) or ""
    if not selected_device:
        row = {
            "device_name": "",
            "opened": False,
            "error": "NO_CAPTURE_DEVICE_DETECTED",
            "reason": "AUTO_DETECT_FAILED",
            "video_devices": list_avfoundation_video_devices(),
        }
        print(json.dumps(row, ensure_ascii=False, indent=2))
        return 2

    try:
        row = probe_device_name(
            device_name=selected_device,
            out_dir=Path(args.out_dir),
            width=args.width,
            height=args.height,
            fps=args.fps,
            shots=args.shots,
            interval_ms=args.interval_ms,
            warmup_seconds=args.warmup_seconds,
            prefix=args.prefix,
            skip_duplicates=False if args.debug else (not args.no_skip_duplicates),
            keep_active=not args.no_keep_active,
            keep_active_seconds=args.keep_active_seconds,
            debug=args.debug,
            read_timeout_seconds=args.read_timeout_seconds,
            first_frame_timeout_seconds=args.first_frame_timeout_seconds,
            restart_per_shot=not args.no_restart_per_shot,
        )
    except ValueError as e:
        row = {
            "device_name": selected_device,
            "opened": False,
            "error": str(e),
            "reason": "DEVICE_REJECTED",
        }
    print(json.dumps(row, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
