@echo off
REM ============================================================
REM  Voice2Anki - one-click setup
REM
REM  Creates .venv (pinned CPython via uv), installs every Python
REM  dependency, and fetches the bundled ffmpeg/ffprobe.
REM  No admin rights and no manual ffmpeg install required.
REM ============================================================
setlocal
cd /d "%~dp0"

echo ============================================================
echo  Voice2Anki setup
echo ============================================================
echo.

REM --- locate a python interpreter -------------------------------------
set "PY="
py -3 -c "import sys" >nul 2>&1 && set "PY=py -3"
if not defined PY (
    python -c "import sys" >nul 2>&1 && set "PY=python"
)
if not defined PY (
    python3 -c "import sys" >nul 2>&1 && set "PY=python3"
)
if not defined PY (
    echo ERROR: no Python interpreter found on this machine.
    echo.
    echo Install Python 3.13 from https://www.python.org/downloads/
    echo and tick "Add python.exe to PATH", then run this file again.
    echo.
    pause
    exit /b 1
)
echo Using interpreter: %PY%
%PY% --version
echo.

REM --- build the environment -------------------------------------------
%PY% setup_env.py
if errorlevel 1 (
    echo.
    echo ============================================================
    echo  SETUP FAILED - see the error above.
    echo ============================================================
    pause
    exit /b 1
)

echo.
%PY% setup_env.py --check
echo.

set /p RUNNOW="Launch Voice2Anki now? [y/N] "
if /i "%RUNNOW%"=="y" (
    echo.
    call run.bat
) else (
    echo.
    echo Setup complete. To run later, just double-click:
    echo    run.bat
)
echo.
pause