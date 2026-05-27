# Local CPU Experiment Completion Report

Date: 2026-05-24

This report records the non-GPU local work completed before additional AutoDL or AGX Orin runs.

## Completed Local Tasks

- Re-ran CIRCA-RT ablations on local synthetic traces and existing AutoDL/ROS2 traces.
- Added `CIRCA-RT-Slack` to the local ablation path and evaluated it against original CIRCA-RT.
- Re-ran matched audit-budget sweeps at 1%, 2%, 5%, 10%, 15%, and 20%.
- Re-ran local attack-strength sensitivity on synthetic coupled attacks.
- Measured CPU monitor overhead breakdown on ROS2 and ONNX MobileNet traces.
- Regenerated real-frame dataset inventory for AV2, nuImages, and DROID manifests.

## Key Local Results

- ROS2 CIRCA-RT: recall 0.417, false alarm 0.075, audit 0.015, p99 2.708 ms, miss 0.0000.
- ROS2 CIRCA-RT-Slack: recall 0.417, false alarm 0.075, audit 0.015, p99 2.708 ms, miss 0.0000.
- ONNX MobileNetV2 at 10% matched audit budget: CIRCA-RT recall 0.400, false alarm 0.069, p99 5.651 ms.
- Synthetic attack-strength sweep: CIRCA-RT recall rises from 0.165 at strength 0.25 to 0.922 at strength 1.50, while audit rises from 0.006 to 0.040.
- CPU full monitor apply overhead is at most 0.0340 ms/frame mean and 0.0452 ms/frame p99 in the two measured traces.
- Token-bucket audit-cost validation checked 483 windows across ROS2/AutoDL scored traces with 0 failures.

## Real-Frame Inventory

- AV2: 2168 frames, 1 sequence, 2168 unique SHA1 values, dominant size 2048x1550.
- nuImages: 217 frames, 67 sequences, 217 unique SHA1 values, dominant size 1600x900.
- DROID: 900 frames, 3 sequences, 900 unique SHA1 values, dominant size 320x180.

## Output Files

- `results_local_cpu_completion_20260524/tables/local_ablation_selected.csv`
- `results_local_cpu_completion_20260524/tables/local_budget_sweep_selected.csv`
- `results_local_cpu_completion_20260524/tables/local_attack_strength_summary.csv`
- `results_local_cpu_completion_20260524/tables/local_overhead_summary.csv`
- `results_local_cpu_completion_20260524/tables/local_dataset_inventory.csv`
- `results_local_cpu_completion_20260524/tables/audit_bound_ros2_validation.csv`
- `results_local_cpu_completion_20260524/tables/audit_bound_autodl_full_validation.csv`
- `results_local_cpu_completion_20260524/figures/`

## Interpretation Boundary

These results strengthen the local algorithmic and analysis evidence. They do not replace the later AGX Orin 64GB real edge-device validation.
