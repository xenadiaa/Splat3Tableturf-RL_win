from __future__ import annotations

import argparse
import json
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from switch_connect.ui.terminal_select import choose_with_arrows
from vision_capture.adapter import (
    auto_detect_capture_device_name,
    is_usb_capture_device_name,
    list_avfoundation_video_devices,
)


CAPTURE_PROFILES: List[Dict[str, object]] = [
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
    "preview_spec": "1920x1080 / uyvy422",
    "fps": 30,
    "window_title": "Capture Card Preview",
    "probe_seconds": 2.0,
}


def _load_config(path: Path) -> Dict[str, object]:
    if not path.exists():
        return dict(DEFAULT_CONFIG)
    data = json.loads(path.read_text(encoding="utf-8"))
    out = dict(DEFAULT_CONFIG)
    if isinstance(data, dict):
        out.update(data)
    return out


def _pick_video_device(prefer_usb_only: bool) -> str:
    devices = list_avfoundation_video_devices()
    if prefer_usb_only:
        usb_devices = [d for d in devices if is_usb_capture_device_name(d)]
        if usb_devices:
            devices = usb_devices
    if not devices:
        return ""
    picked = choose_with_arrows(devices, "Select capture video device")
    return picked or ""


def _resolve_device(args: argparse.Namespace) -> str:
    manual = (args.device_name or "").strip()
    if args.pick_device:
        return _pick_video_device(prefer_usb_only=not args.allow_non_usb)
    if manual:
        return manual
    return auto_detect_capture_device_name(prefer_usb=not args.allow_non_usb) or ""


def _resolve_video_input_name(device_name: str) -> str:
    name = str(device_name or "").strip()
    return f"video={name}"


def _profile_from_spec(spec: str) -> Dict[str, object]:
    size, pixel_format = [part.strip() for part in spec.split("/", 1)]
    width_str, height_str = [part.strip() for part in size.lower().split("x", 1)]
    return {
        "width": int(width_str),
        "height": int(height_str),
        "pixel_format": pixel_format,
        "label": f"{int(width_str)}x{int(height_str)} / {pixel_format}",
    }


def _build_profiles(args: argparse.Namespace) -> List[Dict[str, object]]:
    if args.spec:
        return [_profile_from_spec(args.spec)]
    return [dict(profile) for profile in CAPTURE_PROFILES]


def _build_ffplay_cmd(
    input_name: str,
    width: int,
    height: int,
    fps: int,
    pixel_format: str,
    window_title: str,
) -> List[str]:
    cmd = [
        "ffplay",
        "-hide_banner",
        "-loglevel",
        "warning",
        "-fflags",
        "nobuffer",
        "-flags",
        "low_delay",
        "-framedrop",
        "-sync",
        "video",
        "-window_title",
        window_title,
        "-f",
        "dshow",
    ]
    cmd.append("-an")
    if pixel_format:
        cmd.extend(["-pixel_format", pixel_format])
    cmd.extend(
        [
            "-framerate",
            str(fps),
            "-video_size",
            f"{width}x{height}",
            "-i",
            input_name,
        ]
    )
    return cmd


def _launch_ffplay_with_fallbacks(
    input_name: str,
    profiles: List[Dict[str, object]],
    fps: int,
    window_title: str,
    probe_seconds: float,
) -> int:
    last_rc = 1
    for profile in profiles:
        title = f"{window_title} [{profile['label']}]"
        cmd = _build_ffplay_cmd(
            input_name=input_name,
            width=int(profile["width"]),
            height=int(profile["height"]),
            fps=fps,
            pixel_format=str(profile["pixel_format"]),
            window_title=title,
        )
        print(f"Trying {profile['label']}")
        proc = subprocess.Popen(cmd)
        try:
            start = time.monotonic()
            while time.monotonic() - start < max(0.5, probe_seconds):
                rc = proc.poll()
                if rc is not None:
                    last_rc = rc
                    break
                time.sleep(0.1)
            else:
                print(f"Streaming with {profile['label']}")
                return proc.wait()
        except KeyboardInterrupt:
            try:
                proc.send_signal(signal.SIGTERM)
            except Exception:
                pass
            return 130

        if proc.poll() is None:
            try:
                proc.send_signal(signal.SIGTERM)
                proc.wait(timeout=1.0)
            except Exception:
                try:
                    proc.kill()
                except Exception:
                    pass
        print(f"Profile failed: {profile['label']}")
    return last_rc


def main() -> int:
    parser = argparse.ArgumentParser(description="Open a live ffplay preview window for the capture card.")
    parser.add_argument("--config", default="vision_capture/capture_config.json", help="json config file path")
    parser.add_argument("--device-name", default=None, help="manual DirectShow device name")
    parser.add_argument("--pick-device", action="store_true", help="choose device via arrow keys")
    parser.add_argument("--allow-non-usb", action="store_true", help="allow non-capture-card cameras")
    parser.add_argument(
        "--spec",
        default=None,
        help="single capture spec like '1920x1080 / uyvy422'; default cycles through built-in profiles",
    )
    parser.add_argument("--fps", type=int, default=None, help="capture frame rate")
    parser.add_argument("--window-title", default=None, help="ffplay window title")
    parser.add_argument(
        "--probe-seconds",
        type=float,
        default=None,
        help="how long to wait before considering a profile successfully opened",
    )
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

    device_name = _resolve_device(args)
    if not device_name:
        print("No capture device selected/detected.")
        return 2

    profiles = _build_profiles(args)
    input_name = _resolve_video_input_name(device_name)
    print(f"Device: {device_name}")
    print(f"Input: {input_name}")
    return _launch_ffplay_with_fallbacks(
        input_name=input_name,
        profiles=profiles,
        fps=args.fps,
        window_title=args.window_title,
        probe_seconds=args.probe_seconds,
    )


if __name__ == "__main__":
    raise SystemExit(main())
