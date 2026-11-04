# Run during Loom minute-1 demo
$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot\..

Write-Host "=== LOOM DEMO: image -> millimetres ===" -ForegroundColor Cyan
$img = "dataset\images\raw\both_045.jpeg"
if (-not (Test-Path $img)) {
  $img = (Get-ChildItem dataset\images\raw\both_*.jpeg | Select-Object -First 1).FullName
}
Write-Host "Using image: $img"

New-Item -ItemType Directory -Force -Path outputs | Out-Null
python -m inference.measure --image $img --out outputs\loom_demo.png --json-out outputs\loom_demo.json

Write-Host "`n=== JSON output ===" -ForegroundColor Cyan
Get-Content outputs\loom_demo.json

Write-Host "`nOverlay saved: outputs\loom_demo.png" -ForegroundColor Green
Write-Host "Next on Loom: open inference/measure.py (line ~134) and inference/geometry.py (lines 42, 75)"
