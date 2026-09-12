@echo off
setlocal
chcp 65001 >nul
cd /d "%~dp0"

where python >nul 2>nul
if not errorlevel 1 (
    python "%~dp0autocontroller_rebuild_for_RL\macro6_player_stats_cli.py" --week
    goto finished
)

where py >nul 2>nul
if not errorlevel 1 (
    py -3 "%~dp0autocontroller_rebuild_for_RL\macro6_player_stats_cli.py" --week
    goto finished
)

echo Python was not found. Install Python or add it to PATH.

:finished
echo.
pause
endlocal
