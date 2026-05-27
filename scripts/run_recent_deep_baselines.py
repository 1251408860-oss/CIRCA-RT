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
from circa_rt.deep_baselines import (
    DEEP_BASELINE_METHODS,
    DeepBaselineConfig,
    apply_recent_deep_baseline,
    fit_recent_deep_baseline,
)
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


def _safe_path_part(value: object) -> str:
    text = str(value)
    for ch in ["/", "\\", ":", "*", "?", '"', "<", ">", "|", " "]:
        text = text.replace(ch, "_")
    return text


def load_deep_config(args: argparse.Namespace) -> DeepBaselineConfig:
    cfg_path = ROOT / "configs" / "experiment_config.json"
    if cfg_path.exists():
        raw = json.loads(cfg_path.read_text(encoding="utf-8"))
        base = raw.get("baselines", {})
        seed = int(raw.get("random_seed", 7))
        audit_cost_ms = float(base.get("audit_cost_ms", 4.0))
        window_size = int(args.window_size or base.get("window_size", 32))
    else:
        seed = 7
        audit_cost_ms = 4.0
        window_size = int(args.window_size or 32)
    return DeepBaselineConfig(
        window_size=window_size,
        epochs=int(args.epochs),
        batch_size=int(args.batch_size),
        lr=float(args.lr),
        hidden_dim=int(args.hidden_dim),
        latent_dim=int(args.latent_dim),
        threshold_quantile=float(args.threshold_quantile),
        monitor_cost_floor_ms=float(args.monitor_cost_floor_ms),
        audit_cost_ms=audit_cost_ms,
        seed=seed,
        device=str(args.device),
        backend=str(args.backend),
        max_train_windows=int(args.max_train_windows),
    )


def trace_groups(result_dir: Path, dataset: str) -> list[dict[str, object]]:
    groups: list[dict[str, object]] = []
    if dataset in {"autodl_perception", "autodl_perception_realframes", "perception"}:
        for seed_dir in sorted((result_dir / "traces").glob("*/seed*")):
            if not seed_dir.is_dir() or not (seed_dir / "nominal.csv").exists():
                continue
            model_name = seed_dir.parent.name
            seed_text = seed_dir.name.replace("seed", "")
            groups.append(
                {
                    "dataset": dataset,
                    "runtime": "autodl_perception_semantics",
                    "provider": "torchvision_cuda",
                    "model": model_name,
                    "seed": int(seed_text),
                    "calibration": [seed_dir / "nominal.csv"],
                    "tests": sorted(seed_dir.glob("*.csv")),
                }
            )
    elif dataset == "autodl_full":
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
                    "provider": "ros2_dds_twoprocess",
                    "model": "middleware_trace",
                    "seed": int(seed_dir.name.replace("seed", "")),
                    "calibration": calibration,
                    "tests": sorted(seed_dir.glob("*.csv")),
                }
            )
    return groups


def apply_budget(df: pd.DataFrame, *, base_method: str, budget: float, audit_cost_ms: float) -> pd.DataFrame:
    out = df.copy()
    score = out["score"].to_numpy(dtype=float)
    n_audit = int(round(float(budget) * len(out)))
    audit = np.zeros(len(out), dtype=np.int64)
    if n_audit > 0:
        order = np.argsort(score)[::-1]
        audit[order[:n_audit]] = 1
    out["method"] = f"{base_method}@budget{budget:.3f}"
    out["alarm"] = audit.copy()
    out["audit"] = audit
    out["audit_cost_ms"] = audit.astype(float) * float(audit_cost_ms)
    out["e2e_latency_ms"] = out["baseline_latency_ms"].to_numpy(dtype=float) + out["monitor_cost_ms"].to_numpy(dtype=float) + out["audit_cost_ms"]
    out["deadline_miss"] = (out["e2e_latency_ms"] > out["deadline_ms"]).astype(int)
    return out


def summarize_one(
    scored: pd.DataFrame,
    *,
    group: dict[str, object],
    method: str,
    trace_type: str,
    budget: float | None = None,
) -> dict[str, object]:
    trace_name = f"{group['dataset']}_{group['runtime']}_{group['model']}_seed{group['seed']}_{trace_type}"
    row = summarize_detection(scored, trace_name=trace_name).to_dict()
    row.update(
        {
            "dataset": group["dataset"],
            "runtime": group["runtime"],
            "provider": group["provider"],
            "model": group["model"],
            "seed": group["seed"],
            "trace_type": trace_type,
            "base_method": method,
            "budget": budget if budget is not None else "",
            "deep_backend": str(scored["deep_backend"].iloc[0]) if "deep_backend" in scored.columns else "",
            "deep_train_seconds": float(scored["deep_train_seconds"].iloc[0]) if "deep_train_seconds" in scored.columns else float("nan"),
        }
    )
    return row


def write_tables(summary_rows: list[dict[str, object]], budget_rows: list[dict[str, object]], out: Path) -> None:
    table_dir = out / "tables"
    summary_dir = out / "summaries"
    for d in [table_dir, summary_dir]:
        d.mkdir(parents=True, exist_ok=True)

    summary = pd.DataFrame(summary_rows)
    summary.to_csv(summary_dir / "recent_deep_summary_all.csv", index=False)
    summary.to_csv(table_dir / "recent_deep_summary_all.csv", index=False)
    if not summary.empty:
        group_cols = ["dataset", "runtime", "provider", "model", "method"]
        main = mean_table(summary, group_cols, METRIC_COLS).sort_values(
            ["dataset", "runtime", "model", "attack_recall", "p99_latency_ms"],
            ascending=[True, True, True, False, True],
        )
        main.to_csv(table_dir / "recent_deep_main_table.csv", index=False)
        bootstrap_ci_table(summary, group_cols, METRIC_COLS).to_csv(table_dir / "recent_deep_main_table_ci.csv", index=False)
        scenario = mean_table(summary, ["dataset", "trace_type", "method"], METRIC_COLS).sort_values(
            ["dataset", "trace_type", "attack_recall"],
            ascending=[True, True, False],
        )
        scenario.to_csv(table_dir / "recent_deep_scenario_table.csv", index=False)

    budget = pd.DataFrame(budget_rows)
    budget.to_csv(summary_dir / "recent_deep_budget_summary_all.csv", index=False)
    budget.to_csv(table_dir / "recent_deep_budget_summary_all.csv", index=False)
    if not budget.empty:
        budget_group_cols = ["dataset", "runtime", "provider", "model", "budget", "base_method"]
        budget_main = mean_table(budget, budget_group_cols, METRIC_COLS).sort_values(
            ["dataset", "runtime", "model", "budget", "attack_recall"],
            ascending=[True, True, True, True, False],
        )
        budget_main.to_csv(table_dir / "recent_deep_budget_main_table.csv", index=False)
        bootstrap_ci_table(budget, budget_group_cols, METRIC_COLS).to_csv(table_dir / "recent_deep_budget_main_table_ci.csv", index=False)
        top = budget_main.groupby(["dataset", "runtime", "provider", "model", "budget"], as_index=False).head(5)
        top.to_csv(table_dir / "recent_deep_budget_top5_table.csv", index=False)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs", nargs="+", required=True, help="dataset=path entries, e.g. autodl_full=results_autodl_final")
    parser.add_argument("--methods", nargs="+", default=list(DEEP_BASELINE_METHODS), choices=list(DEEP_BASELINE_METHODS))
    parser.add_argument("--out-dir", default=str(ROOT / "results_recent_deep_baselines"))
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--backend", default="auto", choices=["auto", "torch", "sklearn"])
    parser.add_argument("--window-size", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--latent-dim", type=int, default=16)
    parser.add_argument("--threshold-quantile", type=float, default=0.99)
    parser.add_argument("--monitor-cost-floor-ms", type=float, default=0.12)
    parser.add_argument("--max-train-windows", type=int, default=4096)
    parser.add_argument("--budgets", nargs="+", type=float, default=[0.02, 0.05, 0.10, 0.15])
    parser.add_argument("--max-groups", type=int, default=0, help="debug only; 0 means all groups")
    args = parser.parse_args()

    base_cfg = load_deep_config(args)
    out = Path(args.out_dir)
    raw_dir = out / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)

    groups: list[dict[str, object]] = []
    for entry in args.inputs:
        dataset, raw_path = entry.split("=", 1)
        result_dir = Path(raw_path)
        if not result_dir.is_absolute():
            result_dir = ROOT / result_dir
        groups.extend(trace_groups(result_dir, dataset))
    if args.max_groups:
        groups = groups[: int(args.max_groups)]

    summary_rows: list[dict[str, object]] = []
    budget_rows: list[dict[str, object]] = []

    for group in groups:
        seed = int(group["seed"])
        calibration = pd.concat([read_trace(p) for p in group["calibration"]], ignore_index=True)
        for method in args.methods:
            cfg = replace(base_cfg, seed=seed)
            print(
                f"[recent-deep] fitting {method} dataset={group['dataset']} runtime={group['runtime']} "
                f"model={group['model']} seed={seed} device={cfg.device}",
                flush=True,
            )
            model = fit_recent_deep_baseline(calibration, method, cfg)
            for test_path in group["tests"]:
                trace_type = Path(test_path).stem
                raw = read_trace(test_path)
                scored = apply_recent_deep_baseline(model, raw)
                out_dir = (
                    raw_dir
                    / _safe_path_part(group["dataset"])
                    / _safe_path_part(group["runtime"])
                    / _safe_path_part(group["provider"])
                    / _safe_path_part(group["model"])
                    / f"seed{seed}"
                    / method
                    / _safe_path_part(trace_type)
                )
                out_dir.mkdir(parents=True, exist_ok=True)
                scored.to_csv(out_dir / "scored.csv", index=False)
                summary_rows.append(summarize_one(scored, group=group, method=method, trace_type=trace_type))
                for budget in args.budgets:
                    adjusted = apply_budget(scored, base_method=method, budget=budget, audit_cost_ms=cfg.audit_cost_ms)
                    budget_rows.append(summarize_one(adjusted, group=group, method=method, trace_type=trace_type, budget=budget))

    write_tables(summary_rows, budget_rows, out)
    main_path = out / "tables" / "recent_deep_main_table.csv"
    budget_path = out / "tables" / "recent_deep_budget_main_table.csv"
    print(f"wrote {main_path}")
    print(f"wrote {budget_path}")
    if main_path.exists():
        print(pd.read_csv(main_path).head(30).to_string(index=False))


if __name__ == "__main__":
    main()
