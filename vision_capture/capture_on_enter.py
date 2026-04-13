from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict

import cv2

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from switch_connect.ui.terminal_select import choose_with_arrows
from vision_capture.adapter import (
    FFmpegCaptureSource,
    auto_detect_capture_device_name,
    is_usb_capture_device_name,
    list_avfoundation_video_devices,
)


DEFAULT_CONFIG: Dict[str, Any] = {
    "device_name": "",
    "auto_device": True,
    "pick_device": True,
    "prefer_usb_only": True,
    "width": 1920,
    "height": 1080,
    "fps": 30,
    "out_dir": "vision_capture/debug",
    "warmup_seconds": 5.0,
    "prefix": "capture",
    "read_timeout_seconds": 5.0,
    "restart_per_shot": True,
    "discard_open_frames": 2,
    "drain_ms": 120,
}


def _unique_image_path(out_dir: Path, prefix: str, idx: int) -> Path:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    return out_dir / f"{prefix}_{ts}_{idx:05d}.png"


def _pick_video_device(prefer_usb_only: bool) -> str:
    devices = list_avfoundation_video_devices()
    if prefer_usb_only:
        usb = [d for d in devices if is_usb_capture_device_name(d)]
        if usb:
            devices = usb
    if not devices:
        return ""
    picked = choose_with_arrows(devices, "Select capture video device")
    return picked or ""


def _load_config(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return dict(DEFAULT_CONFIG)
    data = json.loads(path.read_text(encoding="utf-8"))
    out = dict(DEFAULT_CONFIG)
    out.update(data if isinstance(data, dict) else {})
    return out


def _save_config(path: Path, cfg: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(cfg, ensure_ascii=False, indent=2), encoding="utf-8")


def _resolve_device(cfg: Dict[str, Any], cfg_path: Path) -> str:
    manual = str(cfg.get("device_name", "") or "").strip()
    auto_device = bool(cfg.get("auto_device", True))
    pick_device = bool(cfg.get("pick_device", True))
    prefer_usb_only = bool(cfg.get("prefer_usb_only", True))

    should_pick = pick_device
    if not should_pick and not manual:
        should_pick = True

    if should_pick:
        picked = _pick_video_device(prefer_usb_only=prefer_usb_only)
        if picked:
            cfg["device_name"] = picked
            _save_config(cfg_path, cfg)
        return picked

    if manual:
        return manual

    if auto_device:
        detected = auto_detect_capture_device_name(prefer_usb=prefer_usb_only) or ""
        if detected:
            cfg["device_name"] = detected
            _save_config(cfg_path, cfg)
        return detected
    return ""


def main() -> int:
    p = argparse.ArgumentParser(description="Keep capture active; press Enter to save one screenshot.")
    p.add_argument(
        "--config",
        default="vision_capture/capture_config.json",
        help="json config file path",
    )
    args = p.parse_args()

    cfg_path = Path(args.config)
    if not cfg_path.is_absolute():
        cfg_path = REPO_ROOT / cfg_path
    cfg = _load_config(cfg_path)

    device_name = _resolve_device(cfg, cfg_path)
    if not device_name:
        print("No capture device selected/detected.")
        print("Check capture_config.json and ffmpeg dshow devices.")
        return 2

    out_dir = Path(str(cfg.get("out_dir", DEFAULT_CONFIG["out_dir"])))
    if not out_dir.is_absolute():
        out_dir = REPO_ROOT / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    width = int(cfg.get("width", DEFAULT_CONFIG["width"]))
    height = int(cfg.get("height", DEFAULT_CONFIG["height"]))
    fps = int(cfg.get("fps", DEFAULT_CONFIG["fps"]))
    read_timeout_seconds = float(cfg.get("read_timeout_seconds", DEFAULT_CONFIG["read_timeout_seconds"]))
    warmup_seconds = max(0.1, float(cfg.get("warmup_seconds", DEFAULT_CONFIG["warmup_seconds"])))
    restart_per_shot = bool(cfg.get("restart_per_shot", DEFAULT_CONFIG["restart_per_shot"]))
    discard_open_frames = max(1, int(cfg.get("discard_open_frames", DEFAULT_CONFIG["discard_open_frames"])))
    drain_ms = max(0, int(cfg.get("drain_ms", DEFAULT_CONFIG["drain_ms"])))

    src = FFmpegCaptureSource(
        device_name=device_name,
        width=width,
        height=height,
        fps=fps,
    )

    saved = 0
    try:
        # Warmup once so card/signal has time to stabilize.
        _ = src.read_latest(timeout_seconds=warmup_seconds, drain_ms=drain_ms)
        print(f"Device: {device_name}")
        print(f"Config: {cfg_path}")
        print(f"Output: {out_dir}")
        print(
            "Mode: restart_per_shot="
            f"{restart_per_shot}, discard_open_frames={discard_open_frames}, drain_ms={drain_ms}"
        )
        print("Press Enter to capture one frame; input q then Enter to quit.")
        while True:
            cmd = input()
            if cmd.strip().lower() in {"q", "quit", "exit"}:
                break

            if restart_per_shot:
                src.stop()
                src = FFmpegCaptureSource(
                    device_name=device_name,
                    width=width,
                    height=height,
                    fps=fps,
                )
                frame = None
                for _ in range(discard_open_frames):
                    frame = src.read_latest(timeout_seconds=read_timeout_seconds, drain_ms=drain_ms)
            else:
                frame = src.read_latest(timeout_seconds=read_timeout_seconds, drain_ms=drain_ms)
            if frame is None:
                print(f"[error] no frame ({src.last_error})")
                continue

            saved += 1
            path = _unique_image_path(out_dir, str(cfg.get("prefix", DEFAULT_CONFIG["prefix"])), saved)
            cv2.imwrite(str(path), frame)
            frame_hash = hashlib.md5(frame.tobytes()).hexdigest()
            print(f"[saved] {path} md5={frame_hash}")
    except KeyboardInterrupt:
        pass
    finally:
        src.stop()

    print(f"Done. saved={saved}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
