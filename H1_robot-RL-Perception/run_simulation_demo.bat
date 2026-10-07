@echo off
title ANSA OS - Unitree H1 3D Simulation Launcher
cd /d "%~dp0"
echo =====================================================================
echo       ANSA OS - UNITREE H1 ROBOT 3D SIMULATION VIEWER
echo =====================================================================
echo.
echo Starting interactive MuJoCo 3D Window...
echo - Left-click + drag: Rotate camera
echo - Right-click + drag: Zoom in/out
echo - Middle-click + drag: Pan camera
echo - Space: Pause / Resume simulation
echo.

if exist ".venv\Scripts\python.exe" (
    ".venv\Scripts\python.exe" "simulation\rl\demo_obstacle_trials.py"
) else (
    python "simulation\rl\demo_obstacle_trials.py"
)

echo.
echo =====================================================================
echo Simulation finished.
echo =====================================================================
pause
