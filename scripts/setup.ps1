Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$RootDir = Split-Path -Parent $PSScriptRoot
$VenvDir = Join-Path $RootDir ".venv"
$ReqFile = Join-Path $RootDir "requirements.txt"
$ConfigExample = Join-Path $RootDir "autocontroller_rebuild_for_RL\runtime_config.example.json"
$ConfigLocal = Join-Path $RootDir "autocontroller_rebuild_for_RL\runtime_config.local.json"
$CaptureConfig = Join-Path $RootDir "vision_capture\capture_config.json"
$VenvPython = Join-Path $VenvDir "Scripts\python.exe"

Write-Host "[setup] repo root: $RootDir"

$PythonCmd = $null
$PythonArgs = @()
if (Get-Command py -ErrorAction SilentlyContinue) {
    $PythonCmd = "py"
    $PythonArgs = @("-3")
} elseif (Get-Command python -ErrorAction SilentlyContinue) {
    $PythonCmd = "python"
} elseif (Get-Command python3 -ErrorAction SilentlyContinue) {
    $PythonCmd = "python3"
} else {
    Write-Host "[setup] error: Python 3 not found"
    Write-Host "[setup] install Python 3.10+ first, then rerun this script"
    exit 1
}

& $PythonCmd @PythonArgs --version
if ($LASTEXITCODE -ne 0) {
    Write-Host "[setup] error: detected Python command cannot run"
    Write-Host "[setup] install Python 3.10+ and enable Add Python to PATH"
    exit 1
}

if ((Test-Path $VenvDir) -and -not (Test-Path $VenvPython)) {
    Write-Host "[setup] existing .venv is incomplete or belongs to another operating system; rebuilding it"
    Remove-Item -Recurse -Force $VenvDir
}

if (-not (Test-Path $VenvPython)) {
    Write-Host "[setup] creating virtual environment at .venv"
    & $PythonCmd @PythonArgs -m venv $VenvDir
    if ($LASTEXITCODE -ne 0) {
        Write-Host "[setup] error: failed to create Windows virtual environment"
        exit 1
    }
} else {
    Write-Host "[setup] using existing virtual environment at .venv"
}

if (-not (Test-Path $VenvPython)) {
    Write-Host "[setup] error: venv python not found at $VenvPython"
    exit 1
}

Write-Host "[setup] upgrading pip"
& $VenvPython -m pip install --upgrade pip

if (-not (Test-Path $ReqFile)) {
    Write-Host "[setup] error: requirements.txt not found"
    exit 1
}

Write-Host "[setup] installing Python dependencies"
& $VenvPython -m pip install -r $ReqFile

if (-not (Test-Path $ConfigLocal)) {
    if (Test-Path $ConfigExample) {
        Write-Host "[setup] creating runtime_config.local.json from example"
        Copy-Item $ConfigExample $ConfigLocal
    } else {
        Write-Host "[setup] warning: runtime_config.example.json not found, skip config copy"
    }
} else {
    Write-Host "[setup] keeping existing runtime_config.local.json"
}

Write-Host "[setup] resetting device/serial selection related config"
& $VenvPython -c @"
from pathlib import Path
import json

runtime_path = Path(r"$ConfigLocal")
capture_path = Path(r"$CaptureConfig")

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
"@

if (Get-Command ffmpeg -ErrorAction SilentlyContinue) {
    $ffmpegPath = (Get-Command ffmpeg).Source
    Write-Host "[setup] ffmpeg found: $ffmpegPath"
} else {
    Write-Host "[setup] warning: ffmpeg not found"
    Write-Host "[setup] capture-card and video preview tools may not work until ffmpeg is installed"
    Write-Host "[setup] install ffmpeg manually, for example:"
    Write-Host "[setup]   winget: winget install Gyan.FFmpeg"
    Write-Host "[setup]   choco:  choco install ffmpeg"
    Write-Host "[setup]   scoop:  scoop install ffmpeg"
    Write-Host "[setup]   conda:  conda install -c conda-forge ffmpeg"
}

Write-Host ""
Write-Host "[setup] done"
Write-Host ""
Write-Host "Next steps:"
Write-Host "1. Review autocontroller_rebuild_for_RL\runtime_config.local.json"
Write-Host "2. Run your existing commands, for example:"
Write-Host ""
Write-Host "   .venv\Scripts\python.exe autocontroller_rebuild_for_RL\main.py --config autocontroller_rebuild_for_RL\runtime_config.local.json --tmp_win_target"
Write-Host ""
