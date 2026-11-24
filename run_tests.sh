#!/usr/bin/env bash
cd "$(dirname "$0")"
[ -d .venv ] || { echo "ERROR: .venv not found. Run ./setup.sh first."; exit 1; }
exec .venv/bin/python tests/test_core.py
