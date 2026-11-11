#!/usr/bin/env bash
# One-time setup (macOS / Linux): creates .venv and installs requirements
set -e
cd "$(dirname "$0")"
python3 -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -r requirements.txt
echo "Setup complete. Run: .venv/bin/python run.py"
