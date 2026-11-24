@echo off
REM Starts the MedEQ dashboard on http://localhost:8501
REM The backend (run_api.bat) must already be running in another window.
cd /d "%~dp0"

if not exist .venv (
    echo ERROR: .venv not found. Run setup.bat first.
    pause
    exit /b 1
)

echo Starting MedEQ dashboard on http://localhost:8501 ...
echo Make sure run_api.bat is already running in another window.
echo.
call .venv\Scripts\python.exe -m streamlit run app_streamlit.py
pause
