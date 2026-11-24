#!/usr/bin/env bash
# MedEQ - one-time setup (macOS / Linux)
set -e
cd "$(dirname "$0")"

echo "[1/3] Creating virtual environment (.venv)..."
[ -d .venv ] || python3 -m venv .venv

echo "[2/3] Upgrading pip..."
.venv/bin/python -m pip install --upgrade pip --quiet

echo "[3/3] Installing dependencies (this takes a few minutes)..."
.venv/bin/python -m pip install -r requirements.txt

echo
echo "Setup complete. Now open two terminals and run:"
echo "  Terminal 1:  ./run_api.sh"
echo "  Terminal 2:  ./run_dashboard.sh"
