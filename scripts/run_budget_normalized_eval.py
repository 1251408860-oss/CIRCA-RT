from __future__ import annotations

import argparse
import json
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from circa_rt.aggregation import bootstrap_ci_table, mean_table
from circa_rt.baselines import RECENT_DEEP_BASELINE_FUNCS, SPLIT_BASELINE_FUNCS, BaselineConfig
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

METHODS = [
    "CIRCA-RT",
    "RFF-HSIC",
    "ConditionalRFF-HSIC",
    "ContextAwareConformal",
    "EWMATiming",
    "CUSUMTiming",
    "TimingThreshold",
    "OneClassSVMConcat",
    "IsolationForestConcat",
    "LearnedFourierIndependence",
    "OnlineConformalAnomaly",
]


def load_configs() -> tuple[CIRCARuntimeConfig, BaselineConfig]:
    cfg = json.loads((ROOT / "configs" / "experiment_config.json").read_text(encoding="utf-8"))
    c = cfg["circa_rt"]
    b = cfg["baselines"]
    return (
        CIRCARuntimeConfig(
            window_size=int(c["window_size"]),
            feature_dim=int(c["feature_dim"]),
            low_quantile=float(c["low_quantile"]),
            high_quantile=float(c["high_quantile"]),
            monitor_cost_ms=float(c["monitor_cost_ms"]),
            heavy_audit_cost_ms=float(c["heavy_audit_cost_ms"]),
            bucket_capacity=float(c["bucket_capacity"]),
            replenish_rate=float(c["replenish_rate"]),
            seed=int(cfg["random_seed"]),
        ),
        BaselineConfig(
            window_size=int(b["window_size"]),
            monitor_cost_ms=float(b["monitor_cost_ms"]),
            audit_cost_ms=float(b["audit_cost_ms"]),
            periodic_audit_interval=int(b["periodic_audit_interval"]),
            random_audit_rate=float(b["random_audit_rate"]),
            seed=int(cfg["random_seed"]),
        ),
    )


def apply_budget(df: pd.DataFrame, *, method: str, budget: float, audit_cost_ms: float, monitor_cost_ms: float) -> pd.DataFrame:
    out = df.copy()
    score = out["score"].to_numpy(dtype=float) if "score" in out.columns else out["alarm"].to_numpy(dtype=float)
    n_audit = int(round(float(budget) * len(out)))
    audit = np.zeros(len(out), dtype=np.int64)
    if n_audit > 0:
        order = np.argsort(score)[::-1]
        audit[order[:n_audit]] = 1
    out["method"] = f"{method}@budget{budget:.3f}"
    out["alarm"] = audit.copy()
    out["audit"] = audit
    out["audit_cost_ms"] = audit.astype(float) * audit_cost_ms
    out["monitor_cost_ms"] = monitor_cost_ms
    out["e2e_latency_ms"] = out["baseline_latency_ms"].to_numpy(dtype=float) + out["monitor_cost_ms"] + out["audit_cost_ms"]
    out["deadline_miss"] = (out["e2e_latency_ms"] > out["deadline_ms"]).astype(int)
    return out


def trace_groups(result_dir: Path, dataset: str) -> list[dict[str, object]]:
    groups: list[dict[str, object]] = []
    if dataset == "autodl_full":
        for seed_dir in sorted((result_dir / "traces").glob("*/*/*/seed*")):
            if not seed_dir.is_dir() or not (seed_dir / "nominal.csv").exists():
                continue
            groups.append(
                {
                    "dataset": dataset,
                    "runtime": seed_dir.parent.parent.parent.name,
                    "provider": seed_dir.parent.parent.name,
                    "model": seed_dir.parent.name,
                    "seed": int(seed_dir.name.replace("seed", "")),
                    "calibration": [seed_dir / "nominal.csv"],
                    "tests": sorted(seed_dir.glob("*.csv")),
                }
            )
    else:
        for seed_dir in sorted((result_dir / "traces").glob("seed*")):
            if not seed_dir.is_dir() or not (seed_dir / "nominal.csv").exists():
                continue
            calibration = [seed_dir / "nominal.csv"]
            if (seed_dir / "mode_shift.csv").exists():
                calibration.append(seed_dir / "mode_shift.csv")
            groups.append(
                {
                    "dataset": dataset,
                    "runtime": dataset,
                    "provider": "rclpy_fastrtps",
                    "model": "middleware_trace",
                    "seed": int(seed_dir.name.replace("seed", "")),
                    "calibration": calibration,
                    "tests": sorted(seed_dir.glob("*.csv")),
                }
            )
    return groups


def scores_for_group(group: dict[str, object], circa_cfg: CIRCARuntimeConfig, baseline_cfg: BaselineConfig) -> list[pd.DataFrame]:
    seed = int(group["seed"])
    calibration = pd.concat([read_trace(p) for p in group["calibration"]], ignore_index=True)
    circa_model = fit_circa_rt(calibration, replace(circa_cfg, seed=seed))
    scored: list[pd.DataFrame] = []
    for path in group["tests"]:
        raw = read_trace(path)
        circa = apply_circa_rt(circa_model, raw, method="CIRCA-RT")
        circa["base_method"] = "CIRCA-RT"
        scored.append(circa)
        for method, func in SPLIT_BASELINE_FUNCS.items():
            if method in {"AlwaysAudit", "PeriodicAudit", "RandomBudgetAudit"} or method in RECENT_DEEP_BASELINE_FUNCS:
                continue
            result = func(calibration, raw, replace(baseline_cfg, seed=seed))
            result["base_method"] = method
            scored.append(result)
    return scored


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs", nargs="+", required=True, help="dataset=path entries")
    parser.add_argument("--budgets", nargs="+", type=float, default=[0.02, 0.05, 0.10, 0.15])
    parser.add_argument("--out-dir", default=str(ROOT / "results_budget_normalized"))
    args = parser.parse_args()

    circa_cfg, baseline_cfg = load_configs()
    out = Path(args.out_dir)
    table_dir = out / "tables"
    summary_dir = out / "summaries"
    for d in [table_dir, summary_dir]:
        d.mkdir(parents=True, exist_ok=True)

    summaries: list[dict[str, object]] = []
    for entry in args.inputs:
        dataset, raw_path = entry.split("=", 1)
        result_dir = Path(raw_path)
        for group in trace_groups(result_dir, dataset):
            for scored in scores_for_group(group, circa_cfg, baseline_cfg):
                base_method = str(scored["base_method"].iloc[0])
                if base_method not in METHODS:
                    continue
                for budget in args.budgets:
                    adjusted = apply_budget(
                        scored,
                        method=base_method,
                        budget=budget,
                        audit_cost_ms=baseline_cfg.audit_cost_ms,
                        monitor_cost_ms=circa_cfg.monitor_cost_ms if base_method == "CIRCA-RT" else baseline_cfg.monitor_cost_ms,
                    )
                    row = summarize_detection(adjusted, trace_name=f"{group['dataset']}_{group['model']}_{group['seed']}").to_dict()
                    row.update(
                        {
                            "dataset": group["dataset"],
                            "runtime": group["runtime"],
                            "provider": group["provider"],
                            "model": group["model"],
                            "seed": group["seed"],
                            "trace_type": Path(str(scored["run_id"].iloc[0])).stem,
                            "base_method": base_method,
                            "budget": budget,
                        }
                    )
                    summaries.append(row)

    summary = pd.DataFrame(summaries)
    summary.to_csv(summary_dir / "summary_all.csv", index=False)
    summary.to_csv(table_dir / "summary_all.csv", index=False)

    group_cols = ["dataset", "runtime", "provider", "model", "budget", "base_method"]
    main = mean_table(summary, group_cols, METRIC_COLS).sort_values(
        ["dataset", "runtime", "model", "budget", "attack_recall"],
        ascending=[True, True, True, True, False],
    )
    main.to_csv(table_dir / "budget_normalized_main_table.csv", index=False)
    bootstrap_ci_table(summary, group_cols, METRIC_COLS).to_csv(table_dir / "budget_normalized_main_table_ci.csv", index=False)
    best = main.sort_values(["dataset", "runtime", "model", "budget", "attack_recall"], ascending=[True, True, True, True, False])
    best = best.groupby(["dataset", "runtime", "provider", "model", "budget"], as_index=False).head(5)
    best.to_csv(table_dir / "budget_normalized_top5_table.csv", index=False)
    print(best.head(80).to_string(index=False))


if __name__ == "__main__":
    main()
