$ErrorActionPreference = "Stop"

$python = "C:\Users\mystery\AppData\Local\Programs\Python\Python312\python.exe"

& $python -m PyInstaller `
    --noconfirm `
    --clean `
    --onefile `
    --windowed `
    --name VisionCubeSolver `
    --paths src `
    src/vision_cube_solver/__main__.py

Write-Host ""
Write-Host "Built: dist\VisionCubeSolver.exe"
