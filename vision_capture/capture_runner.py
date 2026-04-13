from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from switch_connect.ui.terminal_select import choose_with_arrows
from vision_capture.adapter import (
    auto_detect_capture_device_name,
    is_usb_capture_device_name,
    list_avfoundation_video_devices,
)
from vision_capture.probe_video import probe_device_name


DEFAULT_CONFIG: Dict[str, Any] = {
    "device_name": "",
    "auto_device": True,
    "pick_device": True,
    "prefer_usb_only": True,
    "width": 1920,
    "height": 1080,
    "fps": 30,
    "out_dir": "vision_capture/debug",
    "shots": 20,
    "interval_ms": 1000,
    "warmup_seconds": 5.0,
    "prefix": "capture",
    "debug": False,
    "skip_duplicates": False,
    "read_timeout_seconds": 5.0,
    "first_frame_timeout_seconds": 12.0,
    "restart_per_shot": True,
    "keep_active": False,
    "keep_active_seconds": 0.0,
}


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


def _resolve_device(cfg: Dict[str, Any], cfg_path: Path) -> str:
    manual = str(cfg.get("device_name", "") or "").strip()
    auto_device = bool(cfg.get("auto_device", True))
    pick_device = bool(cfg.get("pick_device", True))
    prefer_usb_only = bool(cfg.get("prefer_usb_only", True))

    should_pick = pick_device
    if not should_pick and not manual:
        # No default device configured -> fall back to interactive pick.
        should_pick = True

    if should_pick:
        picked = _pick_video_device(prefer_usb_only=prefer_usb_only)
        if picked:
            cfg["device_name"] = picked
            _save_config(cfg_path, cfg)
        return picked

    # pick_device=false and manual default exists: use it directly.
    if manual:
        return manual

    # Defensive fallback.
    if auto_device:
        detected = auto_detect_capture_device_name(prefer_usb=prefer_usb_only) or ""
        if detected:
            cfg["device_name"] = detected
            _save_config(cfg_path, cfg)
        return detected
    return ""


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Config-driven capture runner")
    p.add_argument(
        "--config",
        default="vision_capture/capture_config.json",
        help="json config file path",
    )
    return p.parse_args()


def main() -> int:
    args = _parse_args()
    cfg_path = Path(args.config)
    if not cfg_path.is_absolute():
        cfg_path = REPO_ROOT / cfg_path
    cfg = _load_config(cfg_path)

    device_name = _resolve_device(cfg, cfg_path)
    if not device_name:
        print(
            json.dumps(
                {
                    "opened": False,
                    "error": "NO_CAPTURE_DEVICE_SELECTED",
                    "reason": "DEVICE_SELECTION_FAILED",
                    "video_devices": list_avfoundation_video_devices(),
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return 2

    out_dir = Path(str(cfg.get("out_dir", DEFAULT_CONFIG["out_dir"])))
    if not out_dir.is_absolute():
        out_dir = REPO_ROOT / out_dir

    result = probe_device_name(
        device_name=device_name,
        out_dir=out_dir,
        width=int(cfg.get("width", DEFAULT_CONFIG["width"])),
        height=int(cfg.get("height", DEFAULT_CONFIG["height"])),
        fps=int(cfg.get("fps", DEFAULT_CONFIG["fps"])),
        shots=int(cfg.get("shots", DEFAULT_CONFIG["shots"])),
        interval_ms=int(cfg.get("interval_ms", DEFAULT_CONFIG["interval_ms"])),
        warmup_seconds=float(cfg.get("warmup_seconds", DEFAULT_CONFIG["warmup_seconds"])),
        prefix=str(cfg.get("prefix", DEFAULT_CONFIG["prefix"])),
        skip_duplicates=bool(cfg.get("skip_duplicates", DEFAULT_CONFIG["skip_duplicates"])),
        keep_active=bool(cfg.get("keep_active", DEFAULT_CONFIG["keep_active"])),
        keep_active_seconds=float(cfg.get("keep_active_seconds", DEFAULT_CONFIG["keep_active_seconds"])),
        debug=bool(cfg.get("debug", DEFAULT_CONFIG["debug"])),
        read_timeout_seconds=float(cfg.get("read_timeout_seconds", DEFAULT_CONFIG["read_timeout_seconds"])),
        first_frame_timeout_seconds=float(
            cfg.get("first_frame_timeout_seconds", DEFAULT_CONFIG["first_frame_timeout_seconds"])
        ),
        restart_per_shot=bool(cfg.get("restart_per_shot", DEFAULT_CONFIG["restart_per_shot"])),
    )
    result["config_path"] = str(cfg_path)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result.get("opened", False) else 1


if __name__ == "__main__":
    raise SystemExit(main())
