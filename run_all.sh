#!/bin/bash
# Exit immediately if a command exits with a non-zero status
set -e

echo "=========================================="
echo "  SPACEBORN SIMULATION // ONE-GO SETUP"
echo "=========================================="

# 1. Navigate to project root directory
cd "$(dirname "$0")"

# 2. Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    echo "[INFO] Creating Python virtual environment (venv)..."
    python3 -m venv venv
fi

# 3. Activate virtual environment
echo "[INFO] Activating virtual environment..."
source venv/bin/activate

# 4. Upgrade pip and install dependencies
if [ -f "requirements.txt" ]; then
    echo "[INFO] Installing / verifying dependencies from requirements.txt..."
    pip install --upgrade pip > /dev/null 2>&1
    pip install -r requirements.txt
else
    echo "[WARNING] requirements.txt not found in root directory!"
fi

# 5. Export environment variables for MuJoCo headless rendering
export MUJOCO_GL="egl"

# 6. Launch the FastAPI server
echo "=========================================="
echo "  LAUNCHING SPACEBORN SIMULATION SERVER"
echo "=========================================="
echo "[INFO] Access your dashboard at: http://localhost:8000"
python3 server.py
