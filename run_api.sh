#!/usr/bin/env bash
# Starts the MedEQ backend API on http://127.0.0.1:8000
cd "$(dirname "$0")"
[ -d .venv ] || { echo "ERROR: .venv not found. Run ./setup.sh first."; exit 1; }
exec .venv/bin/python -m uvicorn src.api:app --reload --port 8000
