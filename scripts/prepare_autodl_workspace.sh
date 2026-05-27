#!/usr/bin/env bash
set -euo pipefail

mkdir -p results_autodl/{traces,raw,summaries,tables,figures,logs}
mkdir -p autodl_runs/{models,onnx,trt,logs}

echo "Prepared AutoDL directories:"
find results_autodl autodl_runs -maxdepth 2 -type d | sort

echo
echo "Recommended first commands:"
echo "bash scripts/autodl_env_check.sh | tee results_autodl/logs/env_check.log"
echo "python scripts/run_phase5_periodic_pipeline.py"
