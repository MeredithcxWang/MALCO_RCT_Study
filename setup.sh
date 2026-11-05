#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.lock.txt
(cd malco_framework && npm ci --no-audit --no-fund)
echo "Setup complete. Optional R setup: Rscript statistical_analysis/setup.R"
