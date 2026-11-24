@echo off
cd /d "%~dp0"
if not exist .venv (
    echo ERROR: .venv not found. Run setup.bat first.
    pause
    exit /b 1
)
call .venv\Scripts\python.exe tests\test_core.py
pause
