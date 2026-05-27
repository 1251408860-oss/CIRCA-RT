from __future__ import annotations

import argparse
import json
import shutil
import zipfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]

CODE_PATHS = [
    "configs",
    "src",
    "requirements.txt",
    "scripts/analyze_deadline_safe_admission.py",
    "scripts/derive_slack_from_circa_scored.py",
    "scripts/run_local_ablation.py",
    "scripts/run_budget_normalized_eval.py",
    "scripts/benchmark_monitor_overhead.py",
    "scripts/validate_audit_bound.py",
    "scripts/make_local_cpu_completion_artifacts.py",
    "scripts/make_t4_ort_trt_integrated_figures.py",
    "scripts/run_agx_orin64_comprehensive_experiment.sh",
    "scripts/run_agx_orin64_ros2_closed_loop.sh",
    "scripts/run_agx_ros2_closed_loop.py",
    "scripts/summarize_agx_orin64_comprehensive.py",
    "scripts/run_jetson_orin_validation.sh",
    "scripts/jetson_env_check.sh",
    "scripts/parse_tegrastats.py",
    "scripts/run_jetson_tensorrt_engine_check.py",
]

DOC_PATHS = [
    "docs/CIRCA_RT_THEORY_AND_STRONG_CLAIM.md",
    "docs/LOCAL_CPU_EXPERIMENTS_2026_05_24.md",
    "docs/AGX_ORIN64_FULL_ARCHIVE_VALIDATION_2026_05_25.md",
    "docs/AGX_ORIN64_ROS2_CLOSED_LOOP_RUNBOOK_2026_05_25.md",
    "paper_draft/DEADLINE_SAFE_ADMISSION_FRAMEWORK_2026_05_24.md",
    "paper_draft/RTX4080_SLACK_REALFRAMES_2026_05_24.md",
    "paper_draft/T4_DEADLINE_SENSITIVITY_AND_FIGURES_2026_05_24.md",
    "paper_draft/T4_ONNX_TENSORRT_FIX_2026_05_24.md",
]

RESULT_PATHS = [
    "results_deadline_safe_admission_20260524/tables",
    "results_deadline_safe_admission_20260524/figures",
    "results_local_cpu_completion_20260524/tables",
    "results_local_cpu_completion_20260524/figures",
    "results_4080_slack_realframes_20260524/summary_tables",
    "results_t4_ort_trt_integrated/tables",
    "results_t4_ort_trt_integrated/figures",
]


def copy_path(src: Path, dst: Path) -> None:
    if not src.exists():
        return
    if src.is_dir():
        if dst.exists():
            shutil.rmtree(dst)
        shutil.copytree(src, dst, ignore=shutil.ignore_patterns("__pycache__", "*.pyc", ".git", "raw", "traces", "data"))
    else:
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)


def write_readme(out: Path) -> None:
    text = """# Anonymous RTSS Artifact Skeleton

This package contains code, proofs, and compact result tables/figures for the
CIRCA-RT deadline-safe audit admission experiments. It intentionally excludes
large raw image datasets, bulky raw scored traces, and the full AGX archive.
The canonical AGX Orin 64GB-class evidence is kept as a sibling external
archive and summarized in docs/AGX_ORIN64_FULL_ARCHIVE_VALIDATION_2026_05_25.md.

## Main Claims Supported

1. Token-bucket admission bounds heavy-audit demand.
2. Slack-admissible admission prevents audit-induced deadline misses.
3. The admission layer can wrap external monitor scores without changing alarm
   recall; it only controls whether the heavy audit is admitted.
4. Existing RTX 4080/T4 measurements and local CPU analyses are summarized in
   compact tables.
5. The AGX ROS2 closed-loop runner is included for device-side execution when
   ROS2/rclpy is available.

## Quick Checks

From the artifact root:

```powershell
python -m compileall src scripts
python scripts/analyze_deadline_safe_admission.py --inputs av2=../results_autodl_perception_semantics_realframes_av2 nuimages=../results_autodl_perception_semantics_realframes_nuimages droid=../results_autodl_perception_semantics_realframes_droid --out-dir ../artifact_recheck_deadline_safe_admission --methods RFF-HSIC ConditionalRFF-HSIC ContextAwareConformal TimingThreshold --quantiles 0.95 0.99 --offsets-ms 0 1 2 4
```

The recheck command expects the full local workspace with raw scored traces
available next to this artifact. Without the raw traces, inspect the included
`results/` tables and `docs/` proof notes.

## Key Files

- `docs/CIRCA_RT_THEORY_AND_STRONG_CLAIM.md`
- `docs/DEADLINE_SAFE_ADMISSION_FRAMEWORK_2026_05_24.md`
- `docs/AGX_ORIN64_FULL_ARCHIVE_VALIDATION_2026_05_25.md`
- `docs/AGX_ORIN64_ROS2_CLOSED_LOOP_RUNBOOK_2026_05_25.md`
- `results/results_deadline_safe_admission_20260524/tables/admission_delta_tight_quantiles.csv`
- `results/results_deadline_safe_admission_20260524/figures/audit_induced_miss_decomposition.png`
- `results/results_4080_slack_realframes_20260524/summary_tables/slack_deadline_sensitivity_delta_selected.csv`
- `results/results_t4_ort_trt_integrated/tables/t4_monitoring_selected.csv`
"""
    (out / "README.md").write_text(text, encoding="utf-8")


def build(out: Path) -> dict[str, object]:
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True, exist_ok=True)
    manifest: dict[str, object] = {"code": [], "docs": [], "results": []}

    for rel in CODE_PATHS:
        src = ROOT / rel
        dst = out / rel
        copy_path(src, dst)
        if src.exists():
            manifest["code"].append(rel)

    for rel in DOC_PATHS:
        src = ROOT / rel
        name = rel.replace("paper_draft/", "docs/")
        dst = out / name
        copy_path(src, dst)
        if src.exists():
            manifest["docs"].append(rel)

    for rel in RESULT_PATHS:
        src = ROOT / rel
        dst = out / "results" / rel
        copy_path(src, dst)
        if src.exists():
            manifest["results"].append(rel)

    write_readme(out)
    (out / "ARTIFACT_MANIFEST.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def make_zip(out: Path) -> Path:
    zip_path = out.with_suffix(".zip")
    if zip_path.exists():
        zip_path.unlink()
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for path in out.rglob("*"):
            if path.is_file():
                zf.write(path, path.relative_to(out.parent))
    return zip_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a compact anonymous RTSS artifact skeleton.")
    parser.add_argument("--out-dir", type=Path, default=ROOT / "artifact_rtss2026_anonymous")
    parser.add_argument("--zip", action="store_true")
    args = parser.parse_args()

    manifest = build(args.out_dir)
    print(json.dumps(manifest, indent=2))
    print(f"wrote {args.out_dir}")
    if args.zip:
        zip_path = make_zip(args.out_dir)
        print(f"wrote {zip_path}")


if __name__ == "__main__":
    main()
