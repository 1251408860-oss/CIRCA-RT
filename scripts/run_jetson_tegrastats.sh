#!/usr/bin/env bash
set -euo pipefail

OUT_DIR="${1:-logs}"
INTERVAL_MS="${2:-100}"
mkdir -p "${OUT_DIR}"
OUT_FILE="${OUT_DIR}/tegrastats_$(date +%Y%m%d_%H%M%S).log"

echo "writing ${OUT_FILE}"
tegrastats --interval "${INTERVAL_MS}" | tee "${OUT_FILE}"
