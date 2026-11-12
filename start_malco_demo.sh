#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/malco_framework"
export NEXT_TELEMETRY_DISABLED=1
npm run dev
