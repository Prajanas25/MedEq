@echo off
REM ===================================================================
REM  MedEQ - one-time setup (Windows)
REM  Creates a virtual environment and installs all dependencies.
REM  Run this ONCE, then use run_api.bat and run_dashboard.bat.
REM ===================================================================
cd /d "%~dp0"

echo.
echo [1/3] Creating virtual environment (.venv)...
if exist .venv (
    echo       .venv already exists - reusing it.
) else (
    python -m venv .venv
    if errorlevel 1 (
        echo.
        echo ERROR: could not create the virtual environment.
        echo Make sure Python 3.10+ is installed and on your PATH.
        pause
        exit /b 1
    )
)

echo.
echo [2/3] Upgrading pip...
call .venv\Scripts\python.exe -m pip install --upgrade pip --quiet

echo.
echo [3/3] Installing dependencies (this takes a few minutes)...
call .venv\Scripts\python.exe -m pip install -r requirements.txt
if errorlevel 1 (
    echo.
    echo ERROR: dependency installation failed. See the messages above.
    pause
    exit /b 1
)

echo.
echo ===================================================================
echo  Setup complete.
echo.
echo  Next: open TWO terminals in this folder and run
echo     Terminal 1:  run_api.bat
echo     Terminal 2:  run_dashboard.bat
echo ===================================================================
echo.
pause
