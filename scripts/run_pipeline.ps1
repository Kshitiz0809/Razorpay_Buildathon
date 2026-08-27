# End-to-end pipeline: EDA -> train -> evaluate -> curated explainability examples.
# Assumes data/raw/creditcard.csv already exists (see scripts/download_data.ps1).
$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
Set-Location $root
$py = ".\.venv\Scripts\python.exe"

Write-Host "== EDA ==" -ForegroundColor Cyan
& $py scripts/eda.py

Write-Host "== Training (baseline + champion + calibration + threshold selection) ==" -ForegroundColor Cyan
& $py -m fraud_risk.models.train

Write-Host "== Evaluation report (TEST, touched once) ==" -ForegroundColor Cyan
& $py -m fraud_risk.evaluation.report

Write-Host "== Curated explainability examples ==" -ForegroundColor Cyan
& $py scripts/build_curated_examples.py

Write-Host "== Test suite ==" -ForegroundColor Cyan
& $py -m pytest -q

Write-Host "Pipeline complete. Start the API with: .venv\Scripts\uvicorn api.main:app --reload" -ForegroundColor Green
Write-Host "Then the dashboard with: .venv\Scripts\streamlit run dashboard/app.py" -ForegroundColor Green
