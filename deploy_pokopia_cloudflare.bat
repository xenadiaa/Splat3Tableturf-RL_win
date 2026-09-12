@echo off
setlocal

if /i "%~1"=="--inner" goto inner
start "Pokopia Cloudflare Deploy" "%ComSpec%" /d /k call "%~f0" --inner
exit /b

:inner
cd /d "%~dp0"
title Pokopia Cloudflare Deploy
set "PYTHONUTF8=1"
set "PYTHONUNBUFFERED=1"
chcp 65001 >nul
echo ============================================================
echo Pokopia Cloudflare edge deployment
echo This window will remain open after success or failure.
echo ============================================================
echo.

where node >nul 2>nul
if errorlevel 1 goto no_node

where python >nul 2>nul
if not errorlevel 1 goto use_python

where py >nul 2>nul
if not errorlevel 1 goto use_py
goto no_python

:use_python
python ".\cloudflare\pokopia-edge\setup_cloudflare.py"
set "DEPLOY_EXIT=%errorlevel%"
goto result

:use_py
py -3 ".\cloudflare\pokopia-edge\setup_cloudflare.py"
set "DEPLOY_EXIT=%errorlevel%"
goto result

:no_node
echo Node.js was not found in PATH.
echo Close all CMD windows after installing Node.js LTS, then retry.
echo Test command: node --version
set "DEPLOY_EXIT=1"
goto result

:no_python
echo Python was not found in PATH.
echo Use the same Python installation that runs Macro6.
set "DEPLOY_EXIT=1"
goto result

:result
echo.
if not "%DEPLOY_EXIT%"=="0" goto failed
echo Cloudflare edge deployment completed successfully.
echo You may now start run_pokopia_web_and_watchdog.bat.
goto finished

:failed
echo Deployment did not complete.
echo Keep this window open and send the error shown above.
echo A Python failure is also saved as pokopia_cloudflare_deploy_error.log.

:finished
echo.
echo This CMD window is intentionally kept open. Close it manually when finished.
exit /b %DEPLOY_EXIT%
