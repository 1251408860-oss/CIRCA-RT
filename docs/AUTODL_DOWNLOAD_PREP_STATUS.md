# AutoDL Download Preparation Status

Date: 2026-05-20

Remote project path: `/root/rtss_exp_begin`

## Prepared Real Frames

- Dataset: Argoverse 2 Sensor public S3 subset
- Raw directory: `data/raw/av2_sensor_subset`
- Normalized frame directory: `data/frames_av2`
- Active frame path: `data/frames -> frames_av2`
- Download candidates: 2200
- Successful JPG files: 2168
- Failed downloads: 32
- Normalized frame manifest: `data/frames_av2/manifest.csv`
- Download manifest: `data/raw/av2_sensor_subset/download_manifest.csv`

The current `data/frames` tree is ready for:

```bash
cd /root/rtss_exp_begin
/root/miniconda3/bin/python scripts/run_autodl_perception_semantics.py \
  --out-dir results_autodl_perception_semantics_realframes_av2 \
  --device cuda \
  --frame-dir data/frames \
  --models mobilenet_v2 resnet18 \
  --weights default \
  --audit-model resnet18 \
  --audit-weights default \
  --n 900 \
  --seeds 7 8 9 \
  --warmup 30 \
  --deadline-ms 35.0 \
  --gpu-stress-size 2048 \
  --gpu-stress-repeats 2 \
  --coupled-stress-extra-repeats 24 \
  --circa-low-quantile 0.95 \
  --circa-high-quantile 0.99 \
  --token-audit-cost-ms 4.0 \
  --bucket-capacity 12.0 \
  --replenish-rate 0.45 \
  --methods CIRCA-RT ContextAwareConformal ConditionalRFF-HSIC RFF-HSIC SemanticThreshold TimingThreshold AlwaysAudit RandomBudgetAudit \
  --progress-every 150
```

## Cached Model Assets

Already cached in `/root/.cache/torch/hub/checkpoints`:

- `mobilenet_v2-7ebf99e0.pth`
- `resnet18-f37072fd.pth`
- `squeezenet1_1-b8a52dc0.pth`

`resnet50` was intentionally not required for the default real-frame run.

## Baseline Repositories

Official/source repos cloned under `third_party/baselines`:

- `CATCH`
- `MOMENT`
- `TimeMixer`
- `ModernTCN`
- `EasyTSAD`

The main paper-oriented comparison should still use `scripts/run_recent_deep_baselines.py`
for a unified calibration/test split and consistent audit-cost accounting. Official
repos are kept for source-level comparison and optional secondary runs.

## Installed Extra Dependencies

Installed without changing torch:

- `einops`
- `huggingface-hub`
- `transformers`
- `reformer-pytorch`
- `lightgbm`
- `patool`
- `toml`
- `statsmodels`
- `sktime`
- `dash`
- `dash-bootstrap-components`
- `datasets`

Current verified versions:

- `torch 2.8.0+cu128`
- `pandas 2.3.3`
- `scikit-learn 1.7.2`

## Smoke Validation

Completed after dependency changes:

- `py_compile` passed for real-frame and baseline scripts.
- `data/frames` loaded successfully through `list_frame_paths`.
- `scripts/run_recent_deep_baselines.py` ran a CPU/sklearn MOMENT smoke on existing traces and wrote `results_recent_deep_env_smoke`.

Current mode at validation time was no-GPU mode:

- `torch.cuda.is_available() == False`

When GPU mode is enabled again, run the real-frame experiment command above, then run:

```bash
cd /root/rtss_exp_begin
/root/miniconda3/bin/python scripts/validate_audit_bound.py \
  --input-root results_autodl_perception_semantics_realframes_av2/raw \
  --out results_autodl_perception_semantics_realframes_av2/tables/audit_bound_validation.csv \
  --methods CIRCA-RT

/root/miniconda3/bin/python scripts/run_recent_deep_baselines.py \
  --inputs autodl_perception_realframes=results_autodl_perception_semantics_realframes_av2 \
  --out-dir results_recent_deep_baselines_realframes_av2 \
  --device cuda \
  --backend torch \
  --methods CATCH DCdetector TranAD MOMENT TimeMixer ModernTCN \
  --epochs 20 \
  --batch-size 128 \
  --max-train-windows 4096 \
  --budgets 0.02 0.05 0.10 0.15
```
