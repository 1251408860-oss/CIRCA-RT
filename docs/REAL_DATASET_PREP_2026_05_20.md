# Real Dataset Prep Status

Date: 2026-05-20

## What Is Already Ready

- Argoverse 2 Sensor subset has already been downloaded and normalized on AutoDL.
- Recent deep baseline source repositories are already cloned on AutoDL.
- Real-frame normalization now supports video extraction without a system `ffmpeg`
  binary by falling back to `imageio-ffmpeg` or OpenCV.
- CIRCA-RT trace export for official baseline re-runs is now scripted.

## What Was Added In This Round

1. `scripts/download_hf_video_frames.py`
   - Downloads a small HuggingFace-hosted public video/image subset.
   - Supports exact `--repo-file` paths to avoid recursive repository listing on
     large dataset repos.
   - Intended first target: `lerobot/droid_100`.
2. `scripts/prepare_real_dataset_source.py`
   - Imports manually downloaded BDD100K / nuScenes / nuImages archives or
     folders into `data/frames_*`.
   - This avoids blocking the pipeline on dataset-specific folder conventions.
3. `scripts/export_trace_windows_for_official_baselines.py`
   - Exports real-frame trace CSVs into per-model/per-seed train/test `.npy`
     and `.csv` files suitable for official baseline repos.

## Download Constraints

- `BDD100K` generally requires the official site flow and dataset agreement.
- `nuScenes` / `nuImages` require official account-based download links.
- `lerobot/droid_100` is public, but direct AutoDL-to-HuggingFace connectivity
  has been unstable in this environment. The code path is ready, but the actual
  download may need a retried session or local-to-remote sync.

## Recommended Next Commands

### DROID public subset

```bash
cd /root/rtss_exp_begin
bash scripts/run_droid_realframes_experiment.sh
```

The script uses three camera videos from `lerobot/droid_100`, normalizes them
into `data/frames_droid` with per-camera frame balancing, and writes a
dataset-quality report after the real-frame run. In CN-hosted environments it
defaults to `https://hf-mirror.com`; override with
`DROID_HF_ENDPOINT=https://huggingface.co` if direct HuggingFace access is
reliable.

### BDD100K after manual download

```bash
cd /root/rtss_exp_begin
/root/miniconda3/bin/python scripts/prepare_real_dataset_source.py \
  --dataset bdd100k \
  --input /path/to/bdd100k_images_100k.zip \
  --out-dir data/frames_bdd100k \
  --stride 5 \
  --limit 3000
```

### nuScenes / nuImages after manual download

```bash
cd /root/rtss_exp_begin
/root/miniconda3/bin/python scripts/prepare_real_dataset_source.py \
  --dataset nuimages \
  --input /path/to/nuimages-v1.0-mini.tgz \
  --out-dir data/frames_nuimages \
  --stride 3 \
  --limit 3000
```

### Export official baseline inputs from real-frame traces

```bash
cd /root/rtss_exp_begin
/root/miniconda3/bin/python scripts/export_trace_windows_for_official_baselines.py \
  --trace-root results_autodl_perception_semantics_realframes_av2/traces \
  --out-dir data/official_baseline_inputs/av2
```
