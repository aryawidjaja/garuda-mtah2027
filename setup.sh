#!/usr/bin/env bash
# One-time setup: a Python 3.11 virtual environment with the pinned dependencies.
set -euo pipefail
cd "$(dirname "$0")"
if command -v uv >/dev/null 2>&1; then
  uv venv .venv --python 3.11
  uv pip install --python .venv/bin/python -r garuda/requirements.txt
else
  PY=$(command -v python3.11 || true)
  [ -n "$PY" ] || { echo "Python 3.11 not found. Install it (or uv: https://docs.astral.sh/uv/) and rerun."; exit 1; }
  "$PY" -m venv .venv
  .venv/bin/python -m pip install --upgrade pip
  .venv/bin/python -m pip install -r garuda/requirements.txt
fi
echo "Setup done. Next: export MTAH_DATA_DIR=/path/to/Initial_Challenge-Dataset  (the folder that contains tactical/)"
