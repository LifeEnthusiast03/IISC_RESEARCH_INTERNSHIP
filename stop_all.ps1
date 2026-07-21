Write-Host "Stopping ThreatSentinel IDS Services..." -ForegroundColor Cyan

# 1. Stop Database (Docker)
Write-Host "-> Stopping PostgreSQL Database (docker compose)..." -ForegroundColor Yellow
docker compose down

# 2. Close Terminal Windows & Kill Background Processes
Write-Host "-> Closing service windows and stopping servers..." -ForegroundColor Yellow

# Find all PowerShell processes spawned by start_all.ps1 (they contain TS- in their arguments)
$tsProcesses = Get-CimInstance Win32_Process -Filter "Name = 'powershell.exe'" | 
    Where-Object { $_.CommandLine -match "TS-Backend|TS-Frontend|TS-Simulator" }

foreach ($proc in $tsProcesses) {
    Write-Host "   Killing Process Tree for PID $($proc.ProcessId)..." -ForegroundColor DarkGray
    # Use taskkill /T to kill the process TREE (this ensures node.exe and python.exe die too)
    taskkill /PID $proc.ProcessId /T /F > $null 2>&1
}

Write-Host "All services stopped successfully!" -ForegroundColor Green
