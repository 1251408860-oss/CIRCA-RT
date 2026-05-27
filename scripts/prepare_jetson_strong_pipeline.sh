#!/usr/bin/env bash
set -euo pipefail

mkdir -p jetson_runs/raw_logs jetson_runs/traces jetson_runs/scored jetson_runs/tables logs

echo "Directory layout prepared:"
find jetson_runs logs -maxdepth 2 -type d | sort

echo
echo "Next steps:"
echo "1. Run: bash scripts/jetson_env_check.sh | tee jetson_runs/raw_logs/env_check.log"
echo "2. In another shell: bash scripts/run_jetson_tegrastats.sh jetson_runs/raw_logs 100"
echo "3. Run ROS2/TensorRT pipeline and export CIRCA-RT schema CSV into jetson_runs/traces"
