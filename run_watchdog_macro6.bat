@echo off
chcp 65001 >nul
setlocal
cd /d "%~dp0"

python "%~dp0autocontroller_rebuild_for_RL\smart_macro6_watchdog.py" --config "%~dp0autocontroller_rebuild_for_RL\runtime_config.local.json"
set "EXIT_CODE=%ERRORLEVEL%"

echo.
if not "%EXIT_CODE%"=="0" echo watchdog macro6 exited with code %EXIT_CODE%.
pause
exit /b %EXIT_CODE%
