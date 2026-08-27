# Downloads the Kaggle Credit Card Fraud Detection dataset into data/raw.
# Requires ~/.kaggle/kaggle.json to exist first (see data/README.md).
$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
Set-Location $root

& ".\.venv\Scripts\kaggle.exe" datasets download -d mlg-ulb/creditcardfraud -p data/raw --unzip

Write-Host "Downloaded to data/raw/creditcard.csv"
