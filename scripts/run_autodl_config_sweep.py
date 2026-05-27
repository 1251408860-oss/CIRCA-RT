from __future__ import annotations

import argparse
import sys
from dataclasses import replace
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from circa_rt.aggregation import bootstrap_ci_table, mean_table
from circa_rt.core import CIRCARuntimeConfig, apply_circa_rt, fit_circa_rt
from circa_rt.metrics import summarize_detection
from circa_rt.schema import read_trace


METRIC_COLS = [
    "attack_recall",
    "false_alarm_rate",
    "detection_delay",
    "audit_rate",
    "mean_audit_cost_ms",
    "monitor_cost_mean_ms",
    "p99_latency_ms",
    "p999_latency_ms",
    "deadline_miss_ratio",
    "latency_inflation_mean_ms",
    "auroc",
    "auprc",
]


CONFIGS = [
    {"window_size": 16, "feature_dim": 32, "low_quantile": 0.85, "high_quantile": 0.97, "replenish_rate": 0.45},
    {"window_size": 16, "feature_dim": 32, "low_quantile": 0.90, "high_quantile": 0.98, "replenish_rate": 0.45},
    {"window_size": 32, "feature_dim": 32, "low_quantile": 0.85, "high_quantile": 0.97, "replenish_rate": 0.45},
    {"window_size": 32, "feature_dim": 32, "low_quantile": 0.90, "high_quantile": 0.98, "replenish_rate": 0.45},
    {"window_size": 32, "feature_dim": 32, "low_quantile": 0.95, "high_quantile": 0.99, "replenish_rate": 0.45},
    {"window_size": 32, "feature_dim": 64, "low_quantile": 0.85, "high_quantile": 0.97, "replenish_rate": 0.45},
    {"window_size": 32, "feature_dim": 64, "low_quantile": 0.90, "high_quantile": 0.98, "replenish_rate": 0.45},
    {"window_size": 64, "feature_dim": 32, "low_quantile": 0.85, "high_quantile": 0.97, "replenish_rate": 0.45},
    {"window_size": 64, "feature_dim": 32, "low_quantile": 0.90, "high_quantile": 0.98, "replenish_rate": 0.45},
]


def config_id(cfg: dict[str, float | int]) -> str:
    return (
        f"w{cfg['window_size']}_d{cfg['feature_dim']}"
        f"_ql{cfg['low_quantile']}_qh{cfg['high_quantile']}_r{cfg['replenish_rate']}"
    )


def base_config(seed: int) -> CIRCARuntimeConfig:
    return CIRCARuntimeConfig(seed=seed)


def autodl_full_groups(root: Path) -> list[dict[str, object]]:
    groups: list[dict[str, object]] = []
    trace_root = root / "results_autodl_full" / "traces"
    for seed_dir in sorted(trace_root.glob("*/*/*/seed*")):
        if not seed_dir.is_dir() or not (seed_dir / "nominal.csv").exists():
            continue
        groups.append(
            {
                "dataset": "autodl_full",
                "runtime": seed_dir.parent.parent.parent.name,
                "provider": seed_dir.parent.parent.name,
                "model": seed_dir.parent.name,
                "seed": int(seed_dir.name.replace("seed", "")),
                "calibration_paths": [seed_dir / "nominal.csv"],
                "test_paths": sorted(seed_dir.glob("*.csv")),
            }
        )
    return groups


def ros2_groups(root: Path) -> list[dict[str, object]]:
    groups: list[dict[str, object]] = []
    trace_root = root / "results_autodl_ros2" / "traces"
    for seed_dir in sorted(trace_root.glob("seed*")):
        if not seed_dir.is_dir() or not (seed_dir / "nominal.csv").exists():
            continue
        cal_paths = [seed_dir / "nominal.csv"]
        if (seed_dir / "mode_shift.csv").exists():
            cal_paths.append(seed_dir / "mode_shift.csv")
        groups.append(
            {
                "dataset": "autodl_ros2",
                "runtime": "ros2_dds_loopback",
                "provider": "rclpy_fastrtps",
                "model": "middleware_trace",
                "seed": int(seed_dir.name.replace("seed", "")),
                "calibration_paths": cal_paths,
                "test_paths": sorted(seed_dir.glob("*.csv")),
            }
        )
    return groups


def evaluate_group(group: dict[str, object], cfg_dict: dict[str, float | int]) -> list[dict[str, object]]:
    seed = int(group["seed"])
    cfg = replace(base_config(seed), **cfg_dict)
    cal = pd.concat([read_trace(p) for p in group["calibration_paths"]], ignore_index=True)
    model = fit_circa_rt(cal, cfg)
    rows: list[dict[str, object]] = []
    cid = config_id(cfg_dict)
    for path in group["test_paths"]:
        raw = read_trace(path)
        scored = apply_circa_rt(model, raw, method=f"CIRCA-RT[{cid}]")
        row = summarize_detection(scored, trace_name=f"{group['runtime']}_{group['model']}_seed{seed}_{path.stem}").to_dict()
        row.update(
            {
                "config_id": cid,
                "dataset": str(group["dataset"]),
                "runtime": str(group["runtime"]),
                "provider": str(group["provider"]),
                "model": str(group["model"]),
                "seed": seed,
                "trace_type": path.stem,
                **cfg_dict,
            }
        )
        rows.append(row)
    return rows


def pareto_table(main: pd.DataFrame) -> pd.DataFrame:
    if main.empty:
        return main
    rows = []
    group_cols = ["dataset", "runtime", "provider", "model"]
    for _, group in main.groupby(group_cols, dropna=False):
        candidates = group.sort_values(
            ["attack_recall", "audit_rate", "p99_latency_ms"],
            ascending=[False, True, True],
        ).copy()
        rows.append(candidates.head(5))
        constrained = candidates[(candidates["false_alarm_rate"] <= 0.12) & (candidates["audit_rate"] <= 0.05)]
        if not constrained.empty:
            rows.append(constrained.head(3))
    return pd.concat(rows, ignore_index=True).drop_duplicates(group_cols + ["config_id"])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="+", default=["autodl_full", "autodl_ros2"])
    parser.add_argument("--out-dir", default=str(ROOT / "results_autodl_sweep"))
    args = parser.parse_args()

    groups: list[dict[str, object]] = []
    if "autodl_full" in args.datasets:
        groups.extend(autodl_full_groups(ROOT))
    if "autodl_ros2" in args.datasets:
        groups.extend(ros2_groups(ROOT))
    if not groups:
        raise SystemExit("no AutoDL trace groups found; download results_autodl_full and/or results_autodl_ros2 first")

    out = Path(args.out_dir)
    table_dir = out / "tables"
    summary_dir = out / "summaries"
    for d in [table_dir, summary_dir]:
        d.mkdir(parents=True, exist_ok=True)

    summaries: list[dict[str, object]] = []
    for group in groups:
        for cfg in CONFIGS:
            summaries.extend(evaluate_group(group, cfg))

    summary = pd.DataFrame(summaries)
    summary.to_csv(summary_dir / "summary_all.csv", index=False)
    summary.to_csv(table_dir / "summary_all.csv", index=False)

    group_cols = ["dataset", "runtime", "provider", "model", "config_id"]
    main = mean_table(summary, group_cols, METRIC_COLS).sort_values(
        ["dataset", "runtime", "model", "attack_recall", "audit_rate"],
        ascending=[True, True, True, False, True],
    )
    main.to_csv(table_dir / "config_sweep_main_table.csv", index=False)
    bootstrap_ci_table(summary, group_cols, METRIC_COLS).to_csv(table_dir / "config_sweep_main_table_ci.csv", index=False)
    scenario = mean_table(summary, group_cols + ["trace_type"], METRIC_COLS)
    scenario.to_csv(table_dir / "config_sweep_scenario_table.csv", index=False)
    pareto = pareto_table(main)
    pareto.to_csv(table_dir / "config_sweep_pareto_table.csv", index=False)

    print(pareto.head(40).to_string(index=False))


if __name__ == "__main__":
    main()
