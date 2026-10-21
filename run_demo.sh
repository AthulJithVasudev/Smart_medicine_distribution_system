#!/usr/bin/env bash
# Smart Medicine Redistribution System - One-Command Demo (Bash)
# Usage: ./run_demo.sh [profile] [seeds]

set -euo pipefail

PROFILE="${1:-dev}"
SEEDS="${2:-1}"

echo "============================================================"
echo " Smart Medicine Redistribution System - Automated Demo Runner"
echo " Profile: ${PROFILE} | Seeds: ${SEEDS}"
echo "============================================================"

# 1. Virtual Environment Setup
VENV_DIR=".venv"
if [ ! -d "$VENV_DIR" ]; then
    echo ">> Creating virtual environment in ${VENV_DIR}..."
    python3 -m venv "$VENV_DIR" || python -m venv "$VENV_DIR"
fi

if [ -f "$VENV_DIR/bin/python" ]; then
    VENV_PYTHON="$VENV_DIR/bin/python"
elif [ -f "$VENV_DIR/Scripts/python.exe" ]; then
    VENV_PYTHON="$VENV_DIR/Scripts/python.exe"
else
    VENV_PYTHON="python"
fi

# 2. Dependency Installation
echo ">> Installing/verifying pipeline dependencies..."
"$VENV_PYTHON" -m pip install --quiet --upgrade pip
"$VENV_PYTHON" -m pip install --quiet -r requirements-pipeline.txt

# 3. System Doctor Checks
echo ">> Running System Doctor..."
"$VENV_PYTHON" -m pipeline doctor || true

# 4. Pipeline Demo Execution
echo ">> Executing end-to-end pipeline demo..."
"$VENV_PYTHON" -m pipeline demo --profile "$PROFILE" --seeds "$SEEDS"

echo "============================================================"
echo " ALL DEMO STAGES COMPLETED SUCCESSFULLY!"
echo "============================================================"
echo ""
echo "To open the interactive dashboard, run:"
echo "   $VENV_PYTHON -m streamlit run app/Home.py"
echo "or:"
echo "   streamlit run app/Home.py"
echo ""
