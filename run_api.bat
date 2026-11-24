@echo off
REM Starts the MedEQ backend API on http://127.0.0.1:8000
REM Interactive API docs: http://127.0.0.1:8000/docs
cd /d "%~dp0"

if not exist .venv (
    echo ERROR: .venv not found. Run setup.bat first.
    pause
    exit /b 1
)

echo Starting MedEQ API on http://127.0.0.1:8000 ...
echo Leave this window open. Press Ctrl+C to stop.
echo.
call .venv\Scripts\python.exe -m uvicorn src.api:app --reload --port 8000
pause
