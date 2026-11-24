#!/usr/bin/env bash
# Starts the MedEQ dashboard on http://localhost:8501
# The backend (./run_api.sh) must already be running in another terminal.
cd "$(dirname "$0")"
[ -d .venv ] || { echo "ERROR: .venv not found. Run ./setup.sh first."; exit 1; }
exec .venv/bin/python -m streamlit run app_streamlit.py
