# AutoDL GPU Perception Semantics Experiment

This experiment upgrades the AutoDL evidence from random-tensor GPU timing to real-image perception timing plus semantic residuals.

## What It Measures

- Real CUDA inference latency from torchvision perception models.
- Real image semantics from either `--frame-dir` or the reproducible CIFAR-10 fallback.
- Measured heavy-audit latency from a second model, charged only when a method audits.
- Nominal-only calibration: CIRCA-RT and split baselines fit on `nominal.csv` and are evaluated on held-out scenario traces.
- Token-bucket audit envelope validation for CIRCA-RT using `audit_charge_ms` and measured `audit_cost_ms`.

## Main Command

```bash
cd /root/rtss_exp_begin
/root/miniconda3/bin/python scripts/run_autodl_perception_semantics.py \
  --out-dir results_autodl_perception_semantics \
  --device cuda \
  --dataset cifar10 \
  --models mobilenet_v2 resnet18 \
  --weights default \
  --audit-model resnet18 \
  --audit-weights default \
  --n 900 \
  --seeds 7 8 9 \
  --deadline-ms 25.0 \
  --gpu-stress-size 384 \
  --gpu-stress-repeats 1
```

For a robotics/camera paper claim, replace CIFAR-10 with real collected frames:

```bash
/root/miniconda3/bin/python scripts/run_autodl_perception_semantics.py \
  --frame-dir data/frames \
  --out-dir results_autodl_perception_semantics_realframes \
  --device cuda \
  --models mobilenet_v2 resnet18 \
  --weights default \
  --audit-model resnet18 \
  --audit-weights default \
  --n 900 \
  --seeds 7 8 9
```

## Boundary Check

```bash
/root/miniconda3/bin/python scripts/validate_audit_bound.py \
  --input-root results_autodl_perception_semantics/raw \
  --out results_autodl_perception_semantics/tables/audit_bound_validation.csv \
  --methods CIRCA-RT
```

## Main Outputs

- `results_autodl_perception_semantics/tables/autodl_perception_main_table.csv`
- `results_autodl_perception_semantics/tables/autodl_perception_main_table_ci.csv`
- `results_autodl_perception_semantics/tables/autodl_perception_scenario_table.csv`
- `results_autodl_perception_semantics/tables/semantic_timing_diagnostics.csv`
- `results_autodl_perception_semantics/tables/audit_bound_validation.csv`
- `results_autodl_perception_semantics/figures/recall_vs_audit_rate.png`

## Interpretation Rules

CIFAR-10 is useful as a reproducible real-image AutoDL run, but it is not a Jetson/ROS2/robotics dataset. For an RTSS submission, the strongest claim should use `--frame-dir` with robot/autonomous-system frames and then repeat the same protocol on Jetson/TensorRT.
