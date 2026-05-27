# AutoDL Reproduction Commands

Updated: 2026-05-19

Remote workspace:

```bash
cd /root/rtss_exp_begin
```

## Full GPU Experiment

```bash
python scripts/run_autodl_full_experiment.py \
  --out-dir results_autodl_full \
  --models resnet18 mobilenet_v2 squeezenet1_1 \
  --runtimes torch_cuda onnx_cuda onnx_tensorrt \
  --seeds 7 8 9 \
  --n 1500 \
  --warmup 120 \
  --deadline-ms 12.0 \
  --interference-size 512 \
  --ort-interference-size 384
```

Expected note:

- `onnx_tensorrt` may fail if TensorRT libraries are not installed.
- This is recorded in `results_autodl_full/tables/provider_status.csv`.

## ROS2 Humble Install

```bash
bash scripts/install_ros2_humble_autodl.sh
```

When sourcing ROS2 in scripts, avoid `set -u` around the setup file:

```bash
set +u
source /opt/ros/humble/setup.bash
set -u
```

## ROS2 Loopback Capture

```bash
set +u
source /opt/ros/humble/setup.bash
set -u

python scripts/run_ros2_dds_latency_capture.py \
  --out-dir results_autodl_ros2 \
  --seeds 7 8 9 \
  --n 900 \
  --period-ms 20 \
  --deadline-ms 50 \
  --node-python /usr/bin/python3

python scripts/run_ros2_dds_eval.py --root results_autodl_ros2
```

## ROS2 Two-Process Capture

```bash
set +u
source /opt/ros/humble/setup.bash
set -u

python scripts/run_ros2_dds_twoprocess_capture.py \
  --out-dir results_autodl_ros2_twoprocess \
  --seeds 7 8 9 \
  --n 900 \
  --period-ms 20 \
  --deadline-ms 50 \
  --node-python /usr/bin/python3

python scripts/run_ros2_dds_eval.py --root results_autodl_ros2_twoprocess
```

## Local Sweep After Download

```powershell
python scripts/run_autodl_config_sweep.py --out-dir results_autodl_sweep
```

## Recent Deep Baseline Adapters

After `results_autodl_final/` and `results_autodl_ros2_twoprocess_final/` are present on AutoDL, run:

```bash
/root/miniconda3/bin/python scripts/run_recent_deep_baselines.py \
  --inputs autodl_full=results_autodl_final autodl_ros2_twoprocess=results_autodl_ros2_twoprocess_final \
  --out-dir results_recent_deep_baselines_final \
  --device cuda \
  --backend torch \
  --epochs 30 \
  --batch-size 128 \
  --max-train-windows 4096 \
  --budgets 0.02 0.05 0.10 0.15
```

If the instance is in no-GPU mode, use `--device cpu --backend sklearn` for a fast correctness check. Use `--device cuda --backend torch` for final numbers.

## Main Output Tables

- `results_autodl_full/tables/latency_full_table.csv`
- `results_autodl_full/tables/monitoring_main_table.csv`
- `results_autodl_full/tables/provider_status.csv`
- `results_autodl_ros2/tables/ros2_main_table.csv`
- `results_autodl_ros2_twoprocess/tables/ros2_main_table.csv`
- `results_autodl_sweep/tables/config_sweep_pareto_table.csv`
- `results_recent_deep_baselines_final/tables/recent_deep_main_table.csv`
- `results_recent_deep_baselines_final/tables/recent_deep_budget_main_table.csv`
