$port = 7272
$proc = Get-NetTCPConnection -LocalPort $port -State Listen -ErrorAction SilentlyContinue | Select-Object -ExpandProperty OwningProcess
if ($proc) {
    Write-Host "Killing PID $proc on port $port..."
    Stop-Process -Id $proc -Force
    Start-Sleep -Seconds 1
}

Write-Host "Starting R2R server..."
Set-Location "$PSScriptRoot\py"
& ".\.venv\Scripts\python.exe" -m r2r.serve
