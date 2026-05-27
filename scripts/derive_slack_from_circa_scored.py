from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from circa_rt.aggregation import bootstrap_ci_table, mean_table
from circa_rt.metrics import summarize_detection


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


def first_float(df: pd.DataFrame, col: str, default: float) -> float:
    if col not in df.columns or df[col].dropna().empty:
        return float(default)
    return float(df[col].dropna().iloc[0])


def derive_slack(df: pd.DataFrame, *, method: str, slack_margin_ms: float) -> pd.DataFrame:
    out = df.copy()
    q_low = first_float(out, "q_low", float(np.quantile(out["score"].to_numpy(dtype=float), 0.95)))
    q_high = first_float(out, "q_high", float(np.quantile(out["score"].to_numpy(dtype=float), 0.99)))
    token_cost = first_float(out, "token_audit_cost_ms", first_float(out, "bucket_capacity", 12.0) / 3.0)
    bucket_capacity = first_float(out, "bucket_capacity", 12.0)
    replenish_rate = first_float(out, "replenish_rate", 0.45)
    audit_bound = first_float(out, "audit_latency_bound_ms", first_float(out, "token_audit_cost_ms", token_cost))
    monitor_cost = first_float(out, "monitor_cost_ms", 0.08)

    score = out["score"].to_numpy(dtype=float)
    baseline = out["baseline_latency_ms"].to_numpy(dtype=float)
    deadline = out["deadline_ms"].to_numpy(dtype=float)
    slack_before_audit = deadline - baseline - monitor_cost

    alarm = np.zeros(len(out), dtype=np.int64)
    audit = np.zeros(len(out), dtype=np.int64)
    bucket_trace = np.zeros(len(out), dtype=np.float64)
    bucket = float(bucket_capacity)
    for i, value in enumerate(score):
        bucket = min(float(bucket_capacity), bucket + float(replenish_rate))
        if value >= q_low:
            alarm[i] = 1
        if value >= q_high and bucket >= token_cost and slack_before_audit[i] >= audit_bound + slack_margin_ms:
            audit[i] = 1
            bucket -= token_cost
        bucket_trace[i] = bucket

    out["method"] = method
    out["alarm"] = alarm
    out["audit"] = audit
    out["monitor_cost_ms"] = monitor_cost
    if "audit_latency_ms" in out.columns:
        out["audit_cost_ms"] = audit.astype(float) * out["audit_latency_ms"].to_numpy(dtype=float)
    else:
        out["audit_cost_ms"] = audit.astype(float) * audit_bound
    out["audit_token_charge_ms"] = audit.astype(float) * token_cost
    out["audit_charge_ms"] = out["audit_token_charge_ms"]
    out["bucket_level"] = bucket_trace
    out["deadline_slack_before_audit_ms"] = slack_before_audit
    out["slack_admissible"] = (slack_before_audit >= audit_bound + slack_margin_ms).astype(int)
    out["e2e_latency_ms"] = baseline + out["monitor_cost_ms"].to_numpy(dtype=float) + out["audit_cost_ms"].to_numpy(dtype=float)
    out["deadline_miss"] = (out["e2e_latency_ms"].to_numpy(dtype=float) > deadline).astype(int)
    return out


def metadata_from_path(source_root: Path, path: Path) -> dict[str, Any]:
    rel = path.relative_to(source_root / "raw")
    parts = rel.parts
    return {
        "primary_model": parts[0] if len(parts) > 0 else "",
        "seed": int(parts[1].replace("seed", "")) if len(parts) > 1 and parts[1].startswith("seed") else -1,
        "scenario": parts[2] if len(parts) > 2 else "",
    }


def copy_context_files(source_root: Path, out: Path) -> None:
    for name in ["logs", "figures"]:
        src = source_root / name
        dst = out / name
        if src.exists() and not dst.exists():
            shutil.copytree(src, dst)
    table_dir = out / "tables"
    table_dir.mkdir(parents=True, exist_ok=True)
    for name in ["real_dataset_quality.json", "semantic_timing_diagnostics.csv"]:
        src = source_root / "tables" / name
        if src.exists():
            shutil.copy2(src, table_dir / name)


def write_tables_and_figure(out: Path, rows: list[dict[str, Any]]) -> None:
    summary_dir = out / "summaries"
    table_dir = out / "tables"
    figure_dir = out / "figures"
    for d in [summary_dir, table_dir, figure_dir]:
        d.mkdir(parents=True, exist_ok=True)

    summary = pd.DataFrame(rows)
    summary.to_csv(summary_dir / "summary_all.csv", index=False)
    summary.to_csv(table_dir / "summary_all.csv", index=False)

    attack_summary = summary[summary["scenario"] != "nominal"].copy()
    group_cols = ["primary_model", "method"]
    main = mean_table(attack_summary, group_cols, METRIC_COLS).sort_values(
        ["primary_model", "attack_recall", "audit_rate", "p99_latency_ms"],
        ascending=[True, False, True, True],
    )
    main.to_csv(table_dir / "autodl_perception_main_table.csv", index=False)
    bootstrap_ci_table(attack_summary, group_cols, METRIC_COLS).to_csv(table_dir / "autodl_perception_main_table_ci.csv", index=False)

    scenario = mean_table(summary, ["primary_model", "scenario", "method"], METRIC_COLS).sort_values(
        ["primary_model", "scenario", "attack_recall", "audit_rate"],
        ascending=[True, True, False, True],
    )
    scenario.to_csv(table_dir / "autodl_perception_scenario_table.csv", index=False)

    try:
        import matplotlib.pyplot as plt

        key = main[main["method"].isin(["CIRCA-RT", "CIRCA-RT-Slack"])].copy()
        fig, ax = plt.subplots(figsize=(7.0, 4.5))
        for method, group in key.groupby("method"):
            ax.scatter(group["audit_rate"], group["attack_recall"], label=method, s=54)
            for _, row in group.iterrows():
                ax.annotate(str(row["primary_model"]), (row["audit_rate"], row["attack_recall"]), fontsize=8, xytext=(3, 3), textcoords="offset points")
        ax.set_xlabel("Audit rate")
        ax.set_ylabel("Attack recall")
        ax.set_title("CIRCA-RT vs slack-admissible audit")
        ax.grid(True, alpha=0.25)
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(figure_dir / "circa_vs_slack_recall_audit.png", dpi=180)
        fig.savefig(figure_dir / "circa_vs_slack_recall_audit.pdf")
        plt.close(fig)
    except Exception as exc:
        (out / "logs").mkdir(parents=True, exist_ok=True)
        (out / "logs" / "figure_error.txt").write_text(repr(exc), encoding="utf-8")


def write_deadline_sensitivity(source_root: Path, out: Path, args: argparse.Namespace) -> None:
    if not args.deadlines_ms:
        return
    rows: list[dict[str, Any]] = []
    for path in sorted((source_root / "raw").rglob("CIRCA-RT/scored.csv")):
        meta = metadata_from_path(source_root, path)
        original = pd.read_csv(path)
        for deadline in args.deadlines_ms:
            base = original.copy()
            base["deadline_ms"] = float(deadline)
            base["deadline_miss"] = (base["e2e_latency_ms"].to_numpy(dtype=float) > float(deadline)).astype(int)
            slack = original.copy()
            slack["deadline_ms"] = float(deadline)
            slack = derive_slack(slack, method=args.method, slack_margin_ms=args.slack_margin_ms)
            for method, scored in [("CIRCA-RT", base), (args.method, slack)]:
                summary = summarize_detection(scored, trace_name=f"{meta['primary_model']}_seed{meta['seed']}_{meta['scenario']}_deadline{deadline:g}").to_dict()
                summary.update(
                    {
                        "primary_model": meta["primary_model"],
                        "seed": meta["seed"],
                        "scenario": meta["scenario"],
                        "deadline_ms_eval": float(deadline),
                        "source_root": str(source_root),
                        "protocol": "derived_deadline_sensitivity_from_existing_4080_circa_scores",
                        "slack_margin_ms": args.slack_margin_ms,
                    }
                )
                rows.append(summary)
    if not rows:
        return

    table_dir = out / "tables"
    figure_dir = out / "figures"
    table_dir.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)
    summary = pd.DataFrame(rows)
    summary.to_csv(table_dir / "slack_deadline_sensitivity_summary_all.csv", index=False)
    attack_summary = summary[summary["scenario"] != "nominal"].copy()
    group_cols = ["primary_model", "deadline_ms_eval", "method"]
    main = mean_table(attack_summary, group_cols, METRIC_COLS).sort_values(["primary_model", "deadline_ms_eval", "method"])
    main.to_csv(table_dir / "slack_deadline_sensitivity_main_table.csv", index=False)
    try:
        import matplotlib.pyplot as plt

        for metric, ylabel, name in [
            ("deadline_miss_ratio", "deadline miss ratio", "slack_deadline_miss_sensitivity"),
            ("audit_rate", "audit rate", "slack_deadline_audit_sensitivity"),
        ]:
            fig, axes = plt.subplots(1, max(1, main["primary_model"].nunique()), figsize=(11.0, 4.3), sharey=True)
            if not isinstance(axes, np.ndarray):
                axes = np.array([axes])
            for ax, (model, group) in zip(axes, main.groupby("primary_model")):
                for method, mgroup in group.groupby("method"):
                    mgroup = mgroup.sort_values("deadline_ms_eval")
                    ax.plot(mgroup["deadline_ms_eval"], mgroup[metric], marker="o", label=method)
                ax.set_title(str(model))
                ax.set_xlabel("deadline (ms)")
                ax.grid(True, alpha=0.25)
            axes[0].set_ylabel(ylabel)
            axes[-1].legend(fontsize=8)
            fig.tight_layout()
            fig.savefig(figure_dir / f"{name}.png", dpi=180)
            fig.savefig(figure_dir / f"{name}.pdf")
            plt.close(fig)
    except Exception as exc:
        (out / "logs").mkdir(parents=True, exist_ok=True)
        (out / "logs" / "deadline_sensitivity_figure_error.txt").write_text(repr(exc), encoding="utf-8")


def run_one_source(source_root: Path, out: Path, args: argparse.Namespace) -> None:
    rows: list[dict[str, Any]] = []
    out.mkdir(parents=True, exist_ok=True)
    copy_context_files(source_root, out)
    for path in sorted((source_root / "raw").rglob("CIRCA-RT/scored.csv")):
        meta = metadata_from_path(source_root, path)
        original = pd.read_csv(path)
        for method, scored in [
            ("CIRCA-RT", original),
            (args.method, derive_slack(original, method=args.method, slack_margin_ms=args.slack_margin_ms)),
        ]:
            raw_dir = out / "raw" / meta["primary_model"] / f"seed{meta['seed']}" / meta["scenario"] / method
            raw_dir.mkdir(parents=True, exist_ok=True)
            scored.to_csv(raw_dir / "scored.csv", index=False)
            summary = summarize_detection(scored, trace_name=f"{meta['primary_model']}_seed{meta['seed']}_{meta['scenario']}").to_dict()
            summary.update(
                {
                    "primary_model": meta["primary_model"],
                    "seed": meta["seed"],
                    "scenario": meta["scenario"],
                    "source_root": str(source_root),
                    "protocol": "derived_slack_from_existing_4080_circa_scores",
                    "slack_margin_ms": args.slack_margin_ms,
                }
            )
            for col in ["frame_source", "audit_latency_bound_ms", "token_audit_cost_ms", "bucket_capacity", "replenish_rate"]:
                if col in scored.columns and not scored.empty:
                    summary[col] = scored[col].iloc[0]
            rows.append(summary)
    if not rows:
        raise SystemExit(f"no CIRCA-RT scored files found under {source_root}")
    write_tables_and_figure(out, rows)
    write_deadline_sensitivity(source_root, out, args)


def main() -> None:
    parser = argparse.ArgumentParser(description="Derive slack-admissible CIRCA-RT results from existing CIRCA-RT scored traces.")
    parser.add_argument("--inputs", nargs="+", required=True, help="name=path entries")
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--method", default="CIRCA-RT-Slack")
    parser.add_argument("--slack-margin-ms", type=float, default=0.0)
    parser.add_argument("--deadlines-ms", nargs="*", type=float, default=[])
    args = parser.parse_args()

    manifest: dict[str, Any] = {"method": args.method, "slack_margin_ms": args.slack_margin_ms, "sources": []}
    for entry in args.inputs:
        name, raw_path = entry.split("=", 1)
        source_root = Path(raw_path)
        out = args.out_dir / name
        run_one_source(source_root, out, args)
        manifest["sources"].append({"name": name, "source_root": str(source_root), "out": str(out)})
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "derive_slack_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
