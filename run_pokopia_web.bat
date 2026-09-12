@echo off
chcp 65001 >nul
setlocal
cd /d "%~dp0"
set "PYTHONUTF8=1"
set "PYTHONUNBUFFERED=1"

where python >nul 2>nul
if not errorlevel 1 (
    python -u "%~dp0autocontroller_rebuild_for_RL\pokopia_web_server.py"
    goto finished
)

where py >nul 2>nul
if not errorlevel 1 (
    py -3 -u "%~dp0autocontroller_rebuild_for_RL\pokopia_web_server.py"
    goto finished
)

echo Python was not found. Install Python or add it to PATH.

:finished
echo.
echo Pokopia web service stopped.
pause
endlocal
