# Smart Medicine Redistribution System - One-Command Demo (PowerShell)
# Usage: .\run_demo.ps1 [-Profile <dev|demo|full>] [-Seeds <int>] [-SkipInstall]

param(
    [string]$Profile = "dev",
    [int]$Seeds = 1,
    [switch]$SkipInstall
)

$ErrorActionPreference = "Stop"

Write-Host "============================================================" -ForegroundColor Cyan
Write-Host " Smart Medicine Redistribution System - Automated Demo Runner" -ForegroundColor Cyan
Write-Host " Profile: $Profile | Seeds: $Seeds" -ForegroundColor Cyan
Write-Host "============================================================" -ForegroundColor Cyan

# 1. Virtual Environment Setup
$VenvDir = Join-Path $PSScriptRoot ".venv"
$VenvPython = Join-Path $VenvDir "Scripts\python.exe"

if (-not (Test-Path $VenvPython)) {
    Write-Host "`n>> Step 1: Creating Python virtual environment in .venv..." -ForegroundColor Yellow
    python -m venv $VenvDir
    if (-not (Test-Path $VenvPython)) {
        Write-Error "Failed to create virtual environment. Please ensure Python >= 3.12 is installed."
        exit 1
    }
    Write-Host "   [DONE] Virtual environment created." -ForegroundColor Green
} else {
    Write-Host "`n>> Virtual environment detected at $VenvDir." -ForegroundColor Gray
}

# 2. Dependency Installation
if (-not $SkipInstall) {
    Write-Host "`n>> Step 2: Ensuring dependencies are up-to-date..." -ForegroundColor Yellow
    & $VenvPython -m pip install --quiet --upgrade pip
    & $VenvPython -m pip install --quiet -r requirements-pipeline.txt
    Write-Host "   [DONE] Dependencies verified." -ForegroundColor Green
}

# 3. System Doctor Checks
Write-Host "`n>> Step 3: Running System Doctor diagnostic checks..." -ForegroundColor Yellow
& $VenvPython -m pipeline doctor
if ($LASTEXITCODE -ne 0) {
    Write-Host "`n[WARNING] Doctor detected issues. Attempting to proceed..." -ForegroundColor Yellow
}

# 4. Pipeline Demo Execution
Write-Host "`n>> Step 4: Executing end-to-end pipeline demo..." -ForegroundColor Yellow
& $VenvPython -m pipeline demo --profile $Profile --seeds $Seeds
if ($LASTEXITCODE -ne 0) {
    Write-Error "Pipeline demo encountered an error. Check logs above."
    exit $LASTEXITCODE
}

Write-Host "`n============================================================" -ForegroundColor Green
Write-Host " ALL DEMO STAGES COMPLETED SUCCESSFULLY!" -ForegroundColor Green
Write-Host "============================================================" -ForegroundColor Green
Write-Host "`nTo open the interactive executive dashboard, run:" -ForegroundColor Cyan
Write-Host "   & '$VenvPython' -m streamlit run app/Home.py" -ForegroundColor White
Write-Host "or simply:" -ForegroundColor Cyan
Write-Host "   streamlit run app/Home.py`n" -ForegroundColor White
