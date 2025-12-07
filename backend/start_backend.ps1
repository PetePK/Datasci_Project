# Start FastAPI Backend Server
Write-Host "Starting backend server..." -ForegroundColor Green
Set-Location $PSScriptRoot
.\venv\Scripts\uvicorn.exe main:app --reload --host 0.0.0.0 --port 8000
