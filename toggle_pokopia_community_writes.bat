@echo off
chcp 65001 >nul
setlocal
cd /d "%~dp0"
title Pokopia 玩家开门信息总开关
set "PYTHONUTF8=1"

where python >nul 2>nul
if not errorlevel 1 (
    python ".\autocontroller_rebuild_for_RL\pokopia_community_admin.py" toggle
    goto finished
)

where py >nul 2>nul
if not errorlevel 1 (
    py -3 ".\autocontroller_rebuild_for_RL\pokopia_community_admin.py" toggle
    goto finished
)

echo Python was not found. Use the same Python installation that runs Macro6.

:finished
echo.
echo Press any key to close this window.
pause >nul
endlocal
