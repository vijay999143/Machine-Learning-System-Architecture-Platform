@echo off
title OmniArchitect AI Platform
echo ========================================================
echo Starting OmniArchitect AI Platform...
echo Project Folder: %~dp0
echo ========================================================
cd /d "%~dp0"

if not exist "venv\Scripts\python.exe" (
    echo [ERROR] Virtual environment not found! Creating venv...
    python -m venv venv
    call venv\Scripts\activate
    pip install -r requirements.txt
)

echo Activating Python environment and launching server...
echo Server address: http://127.0.0.1:8000
echo Opening browser automatically...
echo Press Ctrl+C to stop the server.
echo ========================================================

start "" "http://127.0.0.1:8000"
venv\Scripts\python.exe -m uvicorn src.app:app --host 0.0.0.0 --port 8000 --reload
pause
