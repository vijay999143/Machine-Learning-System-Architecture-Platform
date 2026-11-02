Write-Host "========================================================" -ForegroundColor Cyan
Write-Host "Starting OmniArchitect AI Platform..." -ForegroundColor Green
Write-Host "Server running at: http://127.0.0.1:8000" -ForegroundColor Yellow
Write-Host "Opening browser automatically..." -ForegroundColor Green
Write-Host "Press Ctrl+C to stop." -ForegroundColor Gray
Write-Host "========================================================" -ForegroundColor Cyan

Set-Location -Path $PSScriptRoot

if (-not (Test-Path ".\venv\Scripts\python.exe")) {
    Write-Host "Creating virtual environment..." -ForegroundColor Yellow
    python -m venv venv
    .\venv\Scripts\Activate.ps1
    pip install -r requirements.txt
}

Start-Process "http://127.0.0.1:8000"
.\venv\Scripts\python.exe -m uvicorn src.app:app --host 0.0.0.0 --port 8000 --reload
