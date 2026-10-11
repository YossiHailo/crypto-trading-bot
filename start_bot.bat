@echo off
title Crypto bot (paper)
cd /d "%~dp0"
if not exist .venv\Scripts\python.exe (
  echo Run setup_windows.bat first.
  pause
  exit /b 1
)
rem Open the dashboard a few seconds after start, unless started with --no-browser (autostart).
if /i not "%~1"=="--no-browser" start "" /b cmd /c "timeout /t 8 >nul && start http://127.0.0.1:8765"

:loop
.venv\Scripts\python.exe main.py
set CODE=%errorlevel%
if "%CODE%"=="0" goto stopped
if "%CODE%"=="2" goto config
if "%CODE%"=="3" goto config
echo.
echo The bot stopped unexpectedly (exit code %CODE%). Restarting in 30 seconds...
echo Close this window to cancel.
timeout /t 30 /nobreak >nul
goto loop

:config
echo.
echo The bot did not start because of a settings problem (see the message above). Fix .env and run start_bot.bat again.
pause
exit /b %CODE%

:stopped
echo.
echo The bot was stopped. Double-click start_bot.bat to start it again.
timeout /t 15 >nul
exit /b 0
