@echo off
REM ============================================================
REM  Voice2Anki - launcher
REM
REM  Starts the GUI using the .venv created by setup.bat.
REM  By default it runs on http://127.0.0.1:7860 only, with no
REM  login, and opens your browser once the server is ready.
REM
REM  Any arguments are passed straight through, for example:
REM    run.bat --localnetwork
REM    run.bat --authentication
REM    run.bat --open_browser=False
REM    run.bat --port 8000
REM
REM  Note: to turn a flag OFF, use --name=False (not --no-name).
REM  See all options with:  run.bat --help
REM ============================================================
setlocal
cd /d "%~dp0"

if not exist ".venv\Scripts\python.exe" (
    echo.
    echo ERROR: no .venv found - the environment is not set up yet.
    echo Run setup.bat first, then run this file again.
    echo.
    pause
    exit /b 1
)

echo ============================================================
echo  Voice2Anki
echo ============================================================
echo Start Anki first so AnkiConnect is available.
echo Press Ctrl+C in this window to stop Voice2Anki.
echo.

".venv\Scripts\python.exe" Voice2Anki.py %*
set EXITCODE=%ERRORLEVEL%

if not "%EXITCODE%"=="0" (
    echo.
    echo Voice2Anki exited with code %EXITCODE%.
    pause
)
exit /b %EXITCODE%