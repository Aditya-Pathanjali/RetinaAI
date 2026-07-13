#!/bin/bash
set -e
echo "========================================================"
echo "  Setting up RetinaAI for WSL2 (GPU Acceleration)"
echo "========================================================"

# Create the Python environment
echo "[1/4] Creating Linux Python Virtual Environment (wsl_env)..."
python3 -m venv wsl_env
source wsl_env/bin/activate

# Install PyTorch with CUDA 12.1 (Linux wheels contain PTX for JIT compiling to sm_120)
echo "[2/4] Installing PyTorch with CUDA support (this may take a few minutes)..."
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# Install requirements
echo "[3/4] Installing project requirements..."
# Filter out torch and torchvision from requirements.txt so pip doesn't downgrade them
grep -vE "^(torch|torchvision)" requirements.txt > wsl_requirements.txt
pip install -r wsl_requirements.txt

# Finish
echo "[4/4] Environment successfully configured!"
echo ""
echo "To use your GPU from now on, just open a WSL terminal and run:"
echo "source wsl_env/bin/activate"
echo "python train.py"
echo "========================================================"
