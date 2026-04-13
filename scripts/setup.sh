#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_DIR="$ROOT_DIR/.venv"
REQ_FILE="$ROOT_DIR/requirements.txt"
CONFIG_EXAMPLE="$ROOT_DIR/autocontroller_rebuild_for_RL/runtime_config.example.json"
CONFIG_LOCAL="$ROOT_DIR/autocontroller_rebuild_for_RL/runtime_config.local.json"
CAPTURE_CONFIG="$ROOT_DIR/vision_capture/capture_config.json"

echo "[setup] repo root: $ROOT_DIR"

if ! command -v python3 >/dev/null 2>&1; then
  echo "[setup] error: python3 not found"
  exit 1
fi

if [ ! -d "$VENV_DIR" ]; then
  echo "[setup] creating virtual environment at .venv"
  python3 -m venv "$VENV_DIR"
else
  echo "[setup] using existing virtual environment at .venv"
fi

echo "[setup] upgrading pip"
"$VENV_DIR/bin/python" -m pip install --upgrade pip

if [ ! -f "$REQ_FILE" ]; then
  echo "[setup] error: requirements.txt not found"
  exit 1
fi

echo "[setup] installing Python dependencies"
"$VENV_DIR/bin/python" -m pip install -r "$REQ_FILE"

if [ ! -f "$CONFIG_LOCAL" ]; then
  if [ -f "$CONFIG_EXAMPLE" ]; then
    echo "[setup] creating runtime_config.local.json from example"
    cp "$CONFIG_EXAMPLE" "$CONFIG_LOCAL"
  else
    echo "[setup] warning: runtime_config.example.json not found, skip config copy"
  fi
else
  echo "[setup] keeping existing runtime_config.local.json"
fi

echo "[setup] resetting device/serial selection related config"
"$VENV_DIR/bin/python" - <<PY
from pathlib import Path
import json

runtime_path = Path(r"$CONFIG_LOCAL")
capture_path = Path(r"$CAPTURE_CONFIG")

if runtime_path.exists():
    runtime = json.loads(runtime_path.read_text(encoding="utf-8"))
    runtime["serial_port"] = ""
    runtime["pick_serial"] = True
    runtime["capture_device_name"] = ""
    runtime_path.write_text(json.dumps(runtime, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

if capture_path.exists():
    capture = json.loads(capture_path.read_text(encoding="utf-8"))
    capture["device_name"] = ""
    capture["auto_device"] = False
    capture["pick_device"] = True
    capture_path.write_text(json.dumps(capture, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
PY

if command -v ffmpeg >/dev/null 2>&1; then
  echo "[setup] ffmpeg found: $(command -v ffmpeg)"
else
  echo "[setup] warning: ffmpeg not found"
  echo "[setup] capture-card and video preview tools may not work until ffmpeg is installed"
  echo "[setup] install ffmpeg manually, for example:"
  echo "[setup]   macOS (Homebrew): brew install ffmpeg"
  echo "[setup]   Ubuntu/Debian:     sudo apt update && sudo apt install -y ffmpeg"
  echo "[setup]   Conda:             conda install -c conda-forge ffmpeg"
fi

cat <<'EOF'

[setup] done

Next steps:
1. Review autocontroller_rebuild_for_RL/runtime_config.local.json
2. Run your existing commands, for example:

   .venv/bin/python autocontroller_rebuild_for_RL/main.py --config autocontroller_rebuild_for_RL/runtime_config.local.json --tmp_win_target

EOF
