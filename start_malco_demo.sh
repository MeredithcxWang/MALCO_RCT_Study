#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/malco_framework"
export MALCO_MODE="${MALCO_MODE:-demo}"
export NEXT_TELEMETRY_DISABLED=1
npm run dev
