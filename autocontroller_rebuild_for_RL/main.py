from __future__ import annotations

import argparse
import contextlib
import json
import os
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from autocontroller_rebuild_for_RL.runtime import (
    AutoControllerRuntime,
    FrameApiAutoLauncher,
    MissingInterfaceError,
    REPO_ROOT,
    load_config,
)
from switch_connect.ui.terminal_select import choose_with_arrows
from switch_connect.virtual_gamepad.device_discovery import list_serial_port_labels, parse_device_from_label
from switch_connect.virtual_gamepad.serial_controller import SerialRemoteController
from vision_capture.adapter import (
    capture_device_busy_message,
    capture_error_may_indicate_device_busy,
    is_usb_capture_device_name,
    list_avfoundation_video_device_rows,
    list_avfoundation_video_devices,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Tableturf auto controller orchestrator")
    parser.add_argument(
        "--config",
        default="autocontroller_rebuild_for_RL/runtime_config.example.json",
        help="config json path",
    )
    parser.add_argument(
        "--once",
        action="store_true",
        help="play only one battle instead of looping forever",
    )
    parser.add_argument(
        "--print-config",
        action="store_true",
        help="print resolved config and exit",
    )
    parser.add_argument(
        "--target-wins",
        type=int,
        default=None,
        help="temporary target wins for this run only; overrides config and stops after reaching it",
    )
    parser.add_argument(
        "--tmp_win_target",
        action="store_true",
        help="prompt for a temporary target win count before starting; does not modify config file",
    )
    return parser.parse_args()


def _prompt_target_wins_if_needed(args: argparse.Namespace) -> int | None:
    if not bool(getattr(args, "tmp_win_target", False)):
        return None
    if args.target_wins is not None:
        return max(0, int(args.target_wins))
    if not (sys.stdin.isatty() and sys.stdout.isatty()):
        return None
    try:
        raw = input("本次临时目标胜场（直接回车表示按配置继续运行）: ").strip()
    except EOFError:
        return None
    if not raw:
        return None
    try:
        return max(0, int(raw))
    except ValueError:
        print(f"无效目标胜场输入：{raw}，将按配置继续运行。")
        return None


def _clear_terminal_for_runtime_ui() -> None:
    with contextlib.suppress(Exception):
        sys.stdout.write("\r\033[3J\033[2J\033[H")
        sys.stdout.flush()


def _load_json_obj(path: Path) -> Dict[str, object]:
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def _write_json_obj(path: Path, payload: Dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _prompt_choice(options: List[str], title: str) -> str:
    if not options:
        return ""
    if sys.stdin.isatty() and sys.stdout.isatty():
        picked = choose_with_arrows(options, title)
        return str(picked or "").strip()
    return str(options[0]).strip()


def _persist_runtime_serial_selection(config_path: Path, serial_port: str) -> None:
    payload = _load_json_obj(config_path)
    payload["serial_port"] = str(serial_port or "").strip()
    payload["pick_serial"] = False
    _write_json_obj(config_path, payload)


def _persist_runtime_capture_selection(config_path: Path, device_name: str) -> None:
    payload = _load_json_obj(config_path)
    payload["capture_device_name"] = str(device_name or "").strip()
    _write_json_obj(config_path, payload)


def _resolved_runtime_config_path(config_path: str | Path) -> Optional[Path]:
    if not config_path:
        return None
    resolved = Path(config_path)
    if not resolved.is_absolute():
        resolved = REPO_ROOT / resolved
    return resolved


def _serial_port_supports_switch_link(serial_port: str) -> bool:
    port = str(serial_port or "").strip()
    if not port:
        return False
    try:
        ctl = SerialRemoteController(port=port, timeout=0.1)
    except Exception:
        return False
    try:
        return bool(ctl.probe_firmware(timeout_seconds=1.2))
    finally:
        with contextlib.suppress(Exception):
            ctl.close()


def _switch_link_probe_priority(label: str, configured_port: str = "") -> tuple[int, str]:
    text = str(label or "").lower()
    port = parse_device_from_label(label)
    if configured_port and port == configured_port:
        return (0, text)
    if "cp210" in text or "usbserial" in text:
        return (1, text)
    if "wch" in text or "ch340" in text:
        return (2, text)
    if port.upper().startswith("COM"):
        return (3, text)
    return (4, text)


def _usable_switch_link_labels(labels: List[str], configured_port: str = "") -> List[str]:
    usable: List[str] = []
    for label in sorted(labels, key=lambda item: _switch_link_probe_priority(item, configured_port)):
        port = parse_device_from_label(label)
        if _serial_port_supports_switch_link(port):
            usable.append(label)
    return usable


def _ensure_switch_link_ready(config, config_path: str | Path = "") -> None:
    labels = list_serial_port_labels()
    configured = str(config.serial_port or "").strip()
    if configured and any(parse_device_from_label(label) == configured for label in labels):
        if _serial_port_supports_switch_link(configured):
            config.serial_port = configured
            config.pick_serial = False
            resolved = _resolved_runtime_config_path(config_path)
            if resolved is not None:
                _persist_runtime_serial_selection(resolved, configured)
            return
    # Re-probe every enumerated port so a transient failure on the configured
    # port does not remove it from the interactive recovery pass.
    usable_labels = _usable_switch_link_labels(labels, configured)
    if not usable_labels:
        if not labels:
            raise RuntimeError(
                "Windows 未枚举到任何串口。请连接运行 AutoController 兼容固件的虚拟手柄，并确认设备管理器中已出现 COM 端口。"
            )
        detected = "; ".join(labels)
        raise RuntimeError(
            "检测到了串口，但没有端口通过 AutoController 固件握手。"
            f"请检查固件、驱动、数据线和端口占用。当前串口：{detected}"
        )
    picked = _prompt_choice(usable_labels, "选择可用的 switch_link 串口")
    if not picked:
        raise RuntimeError("未选择 switch_link 串口，启动已取消。")
    config.serial_port = parse_device_from_label(picked)
    config.pick_serial = False
    if config_path:
        resolved = _resolved_runtime_config_path(config_path)
        if resolved is not None:
            _persist_runtime_serial_selection(resolved, str(config.serial_port or "").strip())


def _usb_capture_device_names() -> List[str]:
    return [name for name in _all_video_device_names() if is_usb_capture_device_name(name)]


def _all_video_device_names() -> List[str]:
    names = [str(name).strip() for name in list_avfoundation_video_devices()]
    seen = set()
    out: List[str] = []
    for name in names:
        if not name or name in seen:
            continue
        seen.add(name)
        out.append(name)
    return out


def _all_video_device_rows() -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []
    seen = set()
    for raw in list_avfoundation_video_device_rows():
        name = str(raw.get("name", "") or "").strip()
        device_id = str(raw.get("device_id", "") or name).strip()
        if not name or not device_id or device_id in seen:
            continue
        seen.add(device_id)
        rows.append(
            {
                "index": str(raw.get("index", len(rows))),
                "name": name,
                "device_id": device_id,
            }
        )
    return rows


def _prompt_video_device(rows: List[Dict[str, str]], title: str) -> str:
    if not rows:
        return ""
    options = [f"[{row['index']}] {row['name']}" for row in rows]
    picked = _prompt_choice(options, title)
    for row, label in zip(rows, options):
        if picked == label:
            return row["device_id"]
    return ""


def _ensure_frame_api_capture_device_selected(config, config_path: str | Path = "") -> None:
    if not str(config.frame_api_url or "").strip():
        return
    launch_config_path = Path(config.frame_api_launch_config)
    if not launch_config_path.is_absolute():
        launch_config_path = REPO_ROOT / launch_config_path
    launch_cfg = _load_json_obj(launch_config_path)
    configured_name = str(launch_cfg.get("device_name", "") or config.capture_device_name or "").strip()
    if configured_name and configured_name.lower() != "invalid":
        rows = _all_video_device_rows()
        exact = [row for row in rows if row["device_id"] == configured_name]
        friendly = [row for row in rows if row["name"] == configured_name]
        selected_name = ""
        if exact:
            selected_name = exact[0]["device_id"]
        elif len(friendly) == 1:
            selected_name = friendly[0]["device_id"]
        elif len(friendly) > 1:
            selected_name = _prompt_video_device(friendly, "检测到同名视频设备，请选择采集卡")
            if not selected_name:
                raise RuntimeError("未选择同名视频设备，启动已取消。")
        if selected_name:
            launch_cfg["device_name"] = selected_name
            launch_cfg["pick_device"] = False
            selected_row = next((row for row in rows if row["device_id"] == selected_name), None)
            launch_cfg["allow_non_usb"] = bool(
                selected_row is not None and not is_usb_capture_device_name(selected_row["name"])
            )
            _write_json_obj(launch_config_path, launch_cfg)
            config.capture_device_name = selected_name
            resolved = _resolved_runtime_config_path(config_path)
            if resolved is not None:
                _persist_runtime_capture_selection(resolved, selected_name)
            return

    all_rows = _all_video_device_rows()
    choices = sorted(all_rows, key=lambda row: 0 if is_usb_capture_device_name(row["name"]) else 1)
    if not choices:
        raise RuntimeError("Windows 未枚举到任何可用视频设备，请检查采集卡连接和驱动。")
    title = (
        f"配置的视频设备不存在（{configured_name}），请选择当前设备"
        if configured_name and configured_name.lower() != "invalid"
        else "请选择视频采集设备（疑似采集卡优先显示）"
    )
    picked = _prompt_video_device(choices, title)
    if not picked:
        raise RuntimeError("未选择视频采集设备，启动已取消。")
    launch_cfg["device_name"] = picked
    launch_cfg["pick_device"] = False
    picked_row = next((row for row in all_rows if row["device_id"] == picked), None)
    launch_cfg["allow_non_usb"] = bool(
        picked_row is not None and not is_usb_capture_device_name(picked_row["name"])
    )
    _write_json_obj(launch_config_path, launch_cfg)
    config.capture_device_name = picked
    resolved = _resolved_runtime_config_path(config_path)
    if resolved is not None:
        _persist_runtime_capture_selection(resolved, picked)


def _ensure_vision_ready(config, config_path: str | Path = "") -> None:
    if not str(config.frame_api_url or "").strip():
        return
    _ensure_frame_api_capture_device_selected(config, config_path)
    launcher = FrameApiAutoLauncher(config)
    try:
        launcher.ensure_started()
        return
    except Exception as exc:
        if capture_error_may_indicate_device_busy(exc):
            raise RuntimeError(capture_device_busy_message(exc)) from exc
        all_rows = _all_video_device_rows()
        if not all_rows:
            raise RuntimeError("视频采集启动失败，且 Windows 当前没有枚举到可重新选择的视频设备。")
        picked = _prompt_video_device(all_rows, "视频采集启动失败，请从全部视频设备中重新选择")
        if not picked:
            raise RuntimeError("未重新选择视频采集设备，启动已取消。")
        launch_config_path = Path(config.frame_api_launch_config)
        if not launch_config_path.is_absolute():
            launch_config_path = REPO_ROOT / launch_config_path
        launch_cfg = _load_json_obj(launch_config_path)
        launch_cfg["device_name"] = picked
        launch_cfg["pick_device"] = False
        picked_row = next((row for row in all_rows if row["device_id"] == picked), None)
        launch_cfg["allow_non_usb"] = bool(
            picked_row is not None and not is_usb_capture_device_name(picked_row["name"])
        )
        _write_json_obj(launch_config_path, launch_cfg)
        config.capture_device_name = picked
        resolved = _resolved_runtime_config_path(config_path)
        if resolved is not None:
            _persist_runtime_capture_selection(resolved, picked)
        launcher = FrameApiAutoLauncher(config)
        launcher.ensure_started()


def main() -> int:
    args = _parse_args()
    config = load_config(args.config)
    resolved_config_path = _resolved_runtime_config_path(args.config)
    if resolved_config_path is not None:
        setattr(config, "_runtime_config_path", str(resolved_config_path))
    setattr(config, "_original_continuous_run", bool(config.continuous_run))
    setattr(config, "_original_target_win_count", int(config.target_win_count))
    if args.print_config:
        print(json.dumps(asdict(config), ensure_ascii=False, indent=2))
        return 0

    try:
        _ensure_switch_link_ready(config, args.config)
        _ensure_vision_ready(config, args.config)
    except Exception as exc:
        print(f"启动失败：{exc}", file=sys.stderr)
        return 1

    override_target_wins = _prompt_target_wins_if_needed(args)
    if override_target_wins is not None:
        if int(override_target_wins) == 0:
            config.continuous_run = True
        else:
            config.continuous_run = False
            config.target_win_count = int(override_target_wins)

    _clear_terminal_for_runtime_ui()

    runtime = AutoControllerRuntime(config)
    try:
        if args.once:
            runtime.play_one_battle()
        else:
            runtime.run_forever()
        return 0
    except KeyboardInterrupt:
        return 130
    except MissingInterfaceError as exc:
        print(
            json.dumps(
                {
                    "ok": False,
                    "error": "MISSING_INTERFACE_FIELDS",
                    "missing_fields": exc.missing_fields,
                    "config": str((REPO_ROOT / args.config) if not Path(args.config).is_absolute() else Path(args.config)),
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return 2
    finally:
        runtime.close()


if __name__ == "__main__":
    raise SystemExit(main())
