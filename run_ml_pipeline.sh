#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
if [ -x .venv/bin/python ]; then
  .venv/bin/python scripts/run_demo.py
else
  python3 scripts/run_demo.py
fi
