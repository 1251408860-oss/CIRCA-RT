# Local Complete Experiment Report

Updated: 2026-05-19

## Local Work Completed

- Re-ran local split synthetic experiments.
- Re-ran local sensitivity experiments.
- Added and ran CIRCA-RT module ablations on local synthetic and downloaded AutoDL/ROS2 traces.
- Added and ran coupled attack-strength sensitivity.
- Generated paper-ready figures, CSV tables, and LaTeX tables from AutoDL/ROS2/deep-baseline results.

## Key ROS2/DDS Default Results

- `ContextAwareConformal`: recall 0.589, false alarm 0.060, audit 0.123, p99 5.076 ms, overhead 0.541 ms.
- `TranAD`: recall 0.571, false alarm 0.045, audit 0.105, p99 4.127 ms, overhead 0.541 ms.
- `CIRCA-RT`: recall 0.429, false alarm 0.077, audit 0.014, p99 2.638 ms, overhead 0.138 ms.
- `CATCH`: recall 0.382, false alarm 0.045, audit 0.082, p99 4.154 ms, overhead 0.449 ms.
- `DCdetector`: recall 0.339, false alarm 0.039, audit 0.072, p99 4.045 ms, overhead 0.409 ms.

## Main Output Directories

- `results_local_complete/figures/`
- `results_local_complete/tables/`
- `results_local_complete/ablation/`
- `results_local_complete/attack_strength/`
- `results_split/`
- `results_split/sensitivity/`

## Paper Claim Boundary

CIRCA-RT should be presented as a low-audit, low-overhead real-time monitor with competitive detection under strict budget constraints. Do not claim universal recall dominance.
