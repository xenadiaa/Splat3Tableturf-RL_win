@echo off
chcp 65001 >nul
setlocal
cd /d "%~dp0"

echo [提示] 此入口只运行watchdog，不会启动本地网页或Cloudflare边缘上传。
echo [提示] 需要同步到 rabi.date 时，请改用 run_pokopia_web_and_watchdog.bat。
echo.

python "%~dp0autocontroller_rebuild_for_RL\smart_macro6_watchdog.py" --config "%~dp0autocontroller_rebuild_for_RL\runtime_config.local.json"
set "EXIT_CODE=%ERRORLEVEL%"

echo.
if not "%EXIT_CODE%"=="0" echo watchdog macro6 exited with code %EXIT_CODE%.
pause
exit /b %EXIT_CODE%
