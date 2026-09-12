#!/usr/bin/env python3
"""Windows one-click launcher for the Pokopia site and Macro6 watchdog."""

from __future__ import annotations

import ctypes
from ctypes import wintypes
import os
from pathlib import Path
import subprocess
import sys
import time
from urllib.error import URLError
from urllib.request import urlopen


BASE_DIR = Path(__file__).resolve().parent
ROOT_DIR = BASE_DIR.parent
SERVER_SCRIPT = BASE_DIR / "pokopia_web_server.py"
WATCHDOG_SCRIPT = BASE_DIR / "smart_macro6_watchdog.py"
EDGE_UPLOADER_SCRIPT = BASE_DIR / "pokopia_edge_uploader.py"
EDGE_CONFIG = BASE_DIR / "pokopia_edge_config.json"
WATCHDOG_CONFIG = BASE_DIR / "runtime_config.local.json"
LOCAL_URL = "http://127.0.0.1:8787/"
READY_URL = LOCAL_URL + "api/live"
CREATE_NO_WINDOW = 0x08000000
JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE = 0x00002000
JOB_OBJECT_EXTENDED_LIMIT_INFORMATION = 9


class _IoCounters(ctypes.Structure):
    _fields_ = [
        ("ReadOperationCount", ctypes.c_ulonglong),
        ("WriteOperationCount", ctypes.c_ulonglong),
        ("OtherOperationCount", ctypes.c_ulonglong),
        ("ReadTransferCount", ctypes.c_ulonglong),
        ("WriteTransferCount", ctypes.c_ulonglong),
        ("OtherTransferCount", ctypes.c_ulonglong),
    ]


class _BasicLimitInformation(ctypes.Structure):
    _fields_ = [
        ("PerProcessUserTimeLimit", ctypes.c_longlong),
        ("PerJobUserTimeLimit", ctypes.c_longlong),
        ("LimitFlags", wintypes.DWORD),
        ("MinimumWorkingSetSize", ctypes.c_size_t),
        ("MaximumWorkingSetSize", ctypes.c_size_t),
        ("ActiveProcessLimit", wintypes.DWORD),
        ("Affinity", ctypes.c_size_t),
        ("PriorityClass", wintypes.DWORD),
        ("SchedulingClass", wintypes.DWORD),
    ]


class _ExtendedLimitInformation(ctypes.Structure):
    _fields_ = [
        ("BasicLimitInformation", _BasicLimitInformation),
        ("IoInfo", _IoCounters),
        ("ProcessMemoryLimit", ctypes.c_size_t),
        ("JobMemoryLimit", ctypes.c_size_t),
        ("PeakProcessMemoryUsed", ctypes.c_size_t),
        ("PeakJobMemoryUsed", ctypes.c_size_t),
    ]


class _WindowsProcessGroup:
    """Ensure closing the launcher also closes its server/watchdog children."""

    def __init__(self) -> None:
        self.handle = None
        if os.name != "nt":
            return
        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel32.CreateJobObjectW.argtypes = [ctypes.c_void_p, wintypes.LPCWSTR]
        kernel32.CreateJobObjectW.restype = wintypes.HANDLE
        kernel32.SetInformationJobObject.argtypes = [
            wintypes.HANDLE,
            ctypes.c_int,
            ctypes.c_void_p,
            wintypes.DWORD,
        ]
        kernel32.SetInformationJobObject.restype = wintypes.BOOL
        kernel32.AssignProcessToJobObject.argtypes = [wintypes.HANDLE, wintypes.HANDLE]
        kernel32.AssignProcessToJobObject.restype = wintypes.BOOL
        kernel32.CloseHandle.argtypes = [wintypes.HANDLE]
        kernel32.CloseHandle.restype = wintypes.BOOL
        handle = kernel32.CreateJobObjectW(None, None)
        if not handle:
            return
        limits = _ExtendedLimitInformation()
        limits.BasicLimitInformation.LimitFlags = JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
        ok = kernel32.SetInformationJobObject(
            handle,
            JOB_OBJECT_EXTENDED_LIMIT_INFORMATION,
            ctypes.byref(limits),
            ctypes.sizeof(limits),
        )
        if not ok:
            kernel32.CloseHandle(handle)
            return
        self.handle = handle
        self._kernel32 = kernel32

    def add(self, process: subprocess.Popen) -> bool:
        if self.handle is None:
            return False
        return bool(
            self._kernel32.AssignProcessToJobObject(
                self.handle,
                wintypes.HANDLE(process._handle),  # type: ignore[attr-defined]
            )
        )

    def close(self) -> None:
        if self.handle is not None:
            self._kernel32.CloseHandle(self.handle)
            self.handle = None


def _require_windows() -> None:
    if os.name != "nt":
        raise RuntimeError("该入口专用于 Windows；请在 Windows 上双击根目录 BAT。")
    missing = [
        path
        for path in (SERVER_SCRIPT, WATCHDOG_SCRIPT, WATCHDOG_CONFIG)
        if not path.is_file()
    ]
    if missing:
        raise RuntimeError("缺少启动文件：" + "；".join(str(path) for path in missing))


def _wait_for_server(process: subprocess.Popen, timeout: float = 12.0) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        code = process.poll()
        if code is not None:
            raise RuntimeError(
                f"网页服务启动失败（退出代码 {code}）。请检查上方是否提示8787端口被占用。"
            )
        try:
            with urlopen(READY_URL, timeout=0.8) as response:
                if response.status == 200 and process.poll() is None:
                    return
        except (OSError, URLError):
            pass
        time.sleep(0.2)
    raise RuntimeError("网页服务在12秒内没有就绪，请检查防火墙、Python和8787端口。")


def _open_default_browser() -> None:
    try:
        os.startfile(LOCAL_URL)  # type: ignore[attr-defined]
    except OSError as exc:
        print(f"无法自动打开浏览器：{exc}")
        print(f"请手动访问：{LOCAL_URL}")


def _stop_process(process: subprocess.Popen | None) -> None:
    if process is None or process.poll() is not None:
        return
    if os.name == "nt":
        subprocess.run(
            ["taskkill", "/PID", str(process.pid), "/T", "/F"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            creationflags=CREATE_NO_WINDOW,
            check=False,
        )
        return
    process.terminate()
    try:
        process.wait(timeout=3)
    except subprocess.TimeoutExpired:
        process.kill()


def main() -> int:
    server: subprocess.Popen | None = None
    watchdog: subprocess.Popen | None = None
    edge_uploader: subprocess.Popen | None = None
    process_group = _WindowsProcessGroup()
    try:
        _require_windows()
        environment = os.environ.copy()
        environment.setdefault("PYTHONUTF8", "1")
        environment.setdefault("PYTHONUNBUFFERED", "1")

        print("正在启动 Pokopia 本机只读网页……")
        server = subprocess.Popen(
            [sys.executable, "-u", str(SERVER_SCRIPT)],
            cwd=str(ROOT_DIR),
            env=environment,
        )
        if not process_group.add(server):
            print("提示：未能启用Windows子进程自动回收，退出时仍会主动关闭服务。")
        _wait_for_server(server)
        print(f"网页已就绪：{LOCAL_URL}")
        _open_default_browser()

        if EDGE_CONFIG.is_file():
            print("检测到Cloudflare边缘配置，正在启动独立上传进程……")
            edge_uploader = subprocess.Popen(
                [sys.executable, "-u", str(EDGE_UPLOADER_SCRIPT)],
                cwd=str(ROOT_DIR),
                env=environment,
            )
            process_group.add(edge_uploader)
        else:
            print("Cloudflare边缘上传尚未配置；当前只运行本机网页，Macro6不受影响。")

        print("正在使用原配置启动 Smart Macro6 watchdog……")
        watchdog = subprocess.Popen(
            [
                sys.executable,
                "-u",
                str(WATCHDOG_SCRIPT),
                "--config",
                str(WATCHDOG_CONFIG),
            ],
            cwd=str(ROOT_DIR),
            env=environment,
        )
        process_group.add(watchdog)
        return watchdog.wait()
    except KeyboardInterrupt:
        print("\n收到中断，正在关闭 watchdog 与网页服务……")
        return 130
    except Exception as exc:
        print(f"启动失败：{exc}")
        return 1
    finally:
        _stop_process(watchdog)
        _stop_process(edge_uploader)
        _stop_process(server)
        process_group.close()


if __name__ == "__main__":
    raise SystemExit(main())
