Write-Host "Starting ThreatSentinel IDS Services..." -ForegroundColor Cyan

# 0. Start Database (Docker)
Write-Host "-> Starting PostgreSQL Database (docker compose)..." -ForegroundColor Green
docker compose up -d

# 1. Start FastAPI Backend
Write-Host "-> Starting FastAPI Backend (uvicorn)..." -ForegroundColor Green
Start-Process powershell -ArgumentList @("-NoExit", "-Command", "$host.ui.RawUI.WindowTitle='TS-Backend'; .\myenv\Scripts\activate; uvicorn backend.main:app --reload")

# 2. Start React Frontend
Write-Host "-> Starting React Frontend (npm run dev)..." -ForegroundColor Green
Start-Process powershell -WorkingDirectory ".\frontend\incident-dashboard" -ArgumentList @("-NoExit", "-Command", "$host.ui.RawUI.WindowTitle='TS-Frontend'; npm run dev")

# 3. Start Simulator
Write-Host "-> Starting Simulator (streamlit)..." -ForegroundColor Green
Start-Process powershell -WorkingDirectory ".\simulator" -ArgumentList @("-NoExit", "-Command", "$host.ui.RawUI.WindowTitle='TS-Simulator'; ..\myenv\Scripts\activate; streamlit run .\streamlit_app.py")

Write-Host "All services have been launched in separate windows!" -ForegroundColor Cyan
