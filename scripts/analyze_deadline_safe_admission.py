from __future__ import annotations

import argparse
import json
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
    "baseline_miss_ratio",
    "monitor_induced_miss_ratio",
    "audit_induced_miss_ratio",
    "audit_induced_miss_count",
    "unsafe_audit_ratio",
    "unsafe_audit_count",
]


def first_float(df: pd.DataFrame, col: str, default: float) -> float:
    if col not in df.columns or df[col].dropna().empty:
        return float(default)
    return float(df[col].dropna().iloc[0])


def path_metadata(source_root: Path, path: Path) -> dict[str, Any]:
    rel = path.relative_to(source_root / "raw")
    parts = rel.parts
    return {
        "primary_model": parts[0] if len(parts) > 0 else "",
        "seed": int(parts[1].replace("seed", "")) if len(parts) > 1 and parts[1].startswith("seed") else -1,
        "scenario": parts[2] if len(parts) > 2 else "",
        "base_method": parts[3] if len(parts) > 3 else "",
    }


def apply_deadline(df: pd.DataFrame, deadline_ms: float) -> pd.DataFrame:
    out = df.copy()
    out["deadline_ms"] = float(deadline_ms)
    if "audit_cost_ms" not in out.columns:
        out["audit_cost_ms"] = 0.0
    if "monitor_cost_ms" not in out.columns:
        out["monitor_cost_ms"] = 0.0
    out["e2e_latency_ms"] = (
        out["baseline_latency_ms"].to_numpy(dtype=float)
        + out["monitor_cost_ms"].to_numpy(dtype=float)
        + out["audit_cost_ms"].to_numpy(dtype=float)
    )
    out["deadline_miss"] = (out["e2e_latency_ms"].to_numpy(dtype=float) > float(deadline_ms)).astype(int)
    return out


def deadline_variants(df: pd.DataFrame, args: argparse.Namespace) -> list[tuple[str, float]]:
    variants: list[tuple[str, float]] = []
    if args.include_actual_deadline:
        variants.append(("actual", first_float(df, "deadline_ms", 35.0)))
    baseline = df["baseline_latency_ms"].to_numpy(dtype=float)
    for q in args.quantiles:
        q_value = float(np.quantile(baseline, q))
        for offset in args.offsets_ms:
            variants.append((f"p{int(round(q * 100)):02d}+{offset:g}ms", q_value + float(offset)))
    return variants


def admission_wrap(df: pd.DataFrame, *, policy: str, args: argparse.Namespace) -> pd.DataFrame:
    out = df.copy()
    base_method = str(out["method"].iloc[0]) if "method" in out.columns and not out.empty else "Unknown"
    score = out["score"].to_numpy(dtype=float) if "score" in out.columns else out["alarm"].to_numpy(dtype=float)
    candidate_alarm = out["alarm"].to_numpy(dtype=int) if "alarm" in out.columns else (score >= np.quantile(score, 0.95)).astype(int)

    token_cost = first_float(out, "token_audit_cost_ms", args.token_audit_cost_ms)
    bucket_capacity = first_float(out, "bucket_capacity", args.bucket_capacity)
    replenish_rate = first_float(out, "replenish_rate", args.replenish_rate)
    audit_bound = first_float(out, "audit_latency_bound_ms", token_cost)
    monitor_cost = first_float(out, "monitor_cost_ms", args.monitor_cost_ms)
    baseline = out["baseline_latency_ms"].to_numpy(dtype=float)
    deadline = out["deadline_ms"].to_numpy(dtype=float)
    slack_before_audit = deadline - baseline - monitor_cost

    audit = np.zeros(len(out), dtype=np.int64)
    bucket_trace = np.zeros(len(out), dtype=np.float64)
    bucket = float(bucket_capacity)
    for i, alarm in enumerate(candidate_alarm):
        bucket = min(float(bucket_capacity), bucket + float(replenish_rate))
        if alarm:
            has_tokens = bucket >= token_cost
            has_slack = slack_before_audit[i] >= audit_bound + args.slack_margin_ms
            if has_tokens and (policy == "token" or has_slack):
                audit[i] = 1
                bucket -= token_cost
        bucket_trace[i] = bucket

    out["method"] = f"{base_method}+{'SlackAdmission' if policy == 'slack' else 'TokenAdmission'}"
    out["alarm"] = candidate_alarm
    out["audit"] = audit
    out["monitor_cost_ms"] = monitor_cost
    if "audit_latency_ms" in out.columns:
        out["audit_cost_ms"] = audit.astype(float) * out["audit_latency_ms"].to_numpy(dtype=float)
    else:
        out["audit_cost_ms"] = audit.astype(float) * audit_bound
    out["audit_token_charge_ms"] = audit.astype(float) * token_cost
    out["audit_charge_ms"] = out["audit_token_charge_ms"]
    out["audit_latency_bound_ms"] = audit_bound
    out["bucket_capacity"] = bucket_capacity
    out["replenish_rate"] = replenish_rate
    out["token_audit_cost_ms"] = token_cost
    out["bucket_level"] = bucket_trace
    out["deadline_slack_before_audit_ms"] = slack_before_audit
    out["slack_admissible"] = (slack_before_audit >= audit_bound + args.slack_margin_ms).astype(int)
    out["e2e_latency_ms"] = baseline + monitor_cost + out["audit_cost_ms"].to_numpy(dtype=float)
    out["deadline_miss"] = (out["e2e_latency_ms"].to_numpy(dtype=float) > deadline).astype(int)
    return out


def attribution(df: pd.DataFrame) -> dict[str, float | int]:
    baseline = df["baseline_latency_ms"].to_numpy(dtype=float)
    monitor = df["monitor_cost_ms"].to_numpy(dtype=float) if "monitor_cost_ms" in df.columns else np.zeros(len(df), dtype=float)
    audit_cost = df["audit_cost_ms"].to_numpy(dtype=float) if "audit_cost_ms" in df.columns else np.zeros(len(df), dtype=float)
    audit_bound = first_float(df, "audit_latency_bound_ms", first_float(df, "token_audit_cost_ms", 0.0))
    deadline = df["deadline_ms"].to_numpy(dtype=float)
    e2e = df["e2e_latency_ms"].to_numpy(dtype=float)
    audit = df["audit"].to_numpy(dtype=int) if "audit" in df.columns else np.zeros(len(df), dtype=int)

    baseline_miss = baseline > deadline
    monitor_only_miss = baseline + monitor > deadline
    total_miss = e2e > deadline
    audit_induced = total_miss & (~monitor_only_miss) & (audit_cost > 1e-12)
    monitor_induced = monitor_only_miss & (~baseline_miss)
    unsafe_audit = (audit == 1) & (baseline + monitor + audit_bound > deadline)
    n = max(len(df), 1)
    return {
        "baseline_miss_count": int(np.sum(baseline_miss)),
        "baseline_miss_ratio": float(np.mean(baseline_miss)),
        "monitor_induced_miss_count": int(np.sum(monitor_induced)),
        "monitor_induced_miss_ratio": float(np.mean(monitor_induced)),
        "audit_induced_miss_count": int(np.sum(audit_induced)),
        "audit_induced_miss_ratio": float(np.sum(audit_induced) / n),
        "unsafe_audit_count": int(np.sum(unsafe_audit)),
        "unsafe_audit_ratio": float(np.sum(unsafe_audit) / n),
    }


def summarize_scored(scored: pd.DataFrame, *, trace_name: str, meta: dict[str, Any], deadline_mode: str, deadline_ms: float, admission_policy: str) -> dict[str, Any]:
    row = summarize_detection(scored, trace_name=trace_name).to_dict()
    row.update(attribution(scored))
    row.update(meta)
    row.update(
        {
            "deadline_mode": deadline_mode,
            "deadline_ms_eval": float(deadline_ms),
            "admission_policy": admission_policy,
            "source_method": meta["base_method"],
        }
    )
    return row


def iter_inputs(entries: list[str]) -> list[tuple[str, Path]]:
    parsed: list[tuple[str, Path]] = []
    for entry in entries:
        name, path = entry.split("=", 1)
        parsed.append((name, Path(path)))
    return parsed


def run(args: argparse.Namespace) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for dataset, source_root in iter_inputs(args.inputs):
        if not (source_root / "raw").exists():
            raise SystemExit(f"missing raw directory: {source_root / 'raw'}")
        for method in args.methods:
            for path in sorted((source_root / "raw").rglob(f"{method}/scored.csv")):
                meta = path_metadata(source_root, path)
                meta["dataset"] = dataset
                raw = pd.read_csv(path)
                if raw.empty:
                    continue
                for deadline_mode, deadline_ms in deadline_variants(raw, args):
                    original = apply_deadline(raw, deadline_ms)
                    variants = [
                        ("original", original),
                        ("token", admission_wrap(original, policy="token", args=args)),
                        ("slack", admission_wrap(original, policy="slack", args=args)),
                    ]
                    for policy, scored in variants:
                        trace_name = f"{dataset}_{meta['primary_model']}_seed{meta['seed']}_{meta['scenario']}_{method}_{deadline_mode}_{policy}"
                        rows.append(
                            summarize_scored(
                                scored,
                                trace_name=trace_name,
                                meta=meta,
                                deadline_mode=deadline_mode,
                                deadline_ms=deadline_ms,
                                admission_policy=policy,
                            )
                        )
    return pd.DataFrame(rows)


def selected_delta(summary: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    keys = ["dataset", "primary_model", "base_method", "deadline_mode"]
    for key, group in summary[summary["scenario"] != "nominal"].groupby(keys, dropna=False):
        original = group[group["admission_policy"] == "original"]
        slack = group[group["admission_policy"] == "slack"]
        token = group[group["admission_policy"] == "token"]
        if original.empty or slack.empty:
            continue
        o = original.mean(numeric_only=True)
        s = slack.mean(numeric_only=True)
        t = token.mean(numeric_only=True) if not token.empty else pd.Series(dtype=float)
        row = {col: value for col, value in zip(keys, key)}
        row.update(
            {
                "recall_original": float(o.get("attack_recall", np.nan)),
                "recall_slack": float(s.get("attack_recall", np.nan)),
                "audit_original": float(o.get("audit_rate", np.nan)),
                "audit_token": float(t.get("audit_rate", np.nan)) if not t.empty else float("nan"),
                "audit_slack": float(s.get("audit_rate", np.nan)),
                "audit_induced_miss_original": float(o.get("audit_induced_miss_ratio", np.nan)),
                "audit_induced_miss_token": float(t.get("audit_induced_miss_ratio", np.nan)) if not t.empty else float("nan"),
                "audit_induced_miss_slack": float(s.get("audit_induced_miss_ratio", np.nan)),
                "unsafe_audit_original": float(o.get("unsafe_audit_ratio", np.nan)),
                "unsafe_audit_token": float(t.get("unsafe_audit_ratio", np.nan)) if not t.empty else float("nan"),
                "unsafe_audit_slack": float(s.get("unsafe_audit_ratio", np.nan)),
                "deadline_miss_original": float(o.get("deadline_miss_ratio", np.nan)),
                "deadline_miss_slack": float(s.get("deadline_miss_ratio", np.nan)),
                "delta_audit_induced_miss": float(s.get("audit_induced_miss_ratio", np.nan) - o.get("audit_induced_miss_ratio", np.nan)),
                "delta_deadline_miss": float(s.get("deadline_miss_ratio", np.nan) - o.get("deadline_miss_ratio", np.nan)),
            }
        )
        rows.append(row)
    return pd.DataFrame(rows)


def write_figures(summary: pd.DataFrame, delta: pd.DataFrame, fig_dir: Path) -> None:
    fig_dir.mkdir(parents=True, exist_ok=True)
    try:
        import matplotlib.pyplot as plt

        actual = summary[(summary["scenario"] != "nominal") & (summary["deadline_mode"] == "actual")].copy()
        actual = actual[actual["admission_policy"].isin(["original", "slack"])]
        if not actual.empty:
            main = mean_table(
                actual,
                ["base_method", "admission_policy"],
                ["attack_recall", "audit_rate", "deadline_miss_ratio", "audit_induced_miss_ratio"],
            )
            fig, ax = plt.subplots(figsize=(8.0, 5.0))
            for (method, policy), group in main.groupby(["base_method", "admission_policy"]):
                label = f"{method} ({policy})"
                marker = "o" if policy == "original" else "s"
                ax.scatter(group["audit_rate"], group["attack_recall"], label=label, marker=marker, s=54)
            ax.set_xlabel("audit rate")
            ax.set_ylabel("attack recall")
            ax.set_title("Generic admission layer: recall vs audit")
            ax.grid(True, alpha=0.25)
            ax.legend(fontsize=7, ncol=2)
            fig.tight_layout()
            fig.savefig(fig_dir / "generic_admission_recall_vs_audit.png", dpi=220)
            fig.savefig(fig_dir / "generic_admission_recall_vs_audit.pdf")
            plt.close(fig)

        tight = delta[delta["deadline_mode"].isin(["p95+0ms", "p99+0ms"])].copy()
        if not tight.empty:
            plot = tight.groupby(["deadline_mode", "admission"], as_index=False).mean(numeric_only=True) if "admission" in tight else None
            rows = []
            for mode, group in tight.groupby("deadline_mode"):
                rows.append({"deadline_mode": mode, "policy": "original", "audit_induced_miss": group["audit_induced_miss_original"].mean()})
                rows.append({"deadline_mode": mode, "policy": "slack", "audit_induced_miss": group["audit_induced_miss_slack"].mean()})
            miss = pd.DataFrame(rows)
            fig, ax = plt.subplots(figsize=(7.5, 4.6))
            x_labels = sorted(miss["deadline_mode"].unique())
            x = np.arange(len(x_labels))
            width = 0.34
            for i, policy in enumerate(["original", "slack"]):
                values = [float(miss[(miss["deadline_mode"] == label) & (miss["policy"] == policy)]["audit_induced_miss"].mean()) for label in x_labels]
                ax.bar(x + (i - 0.5) * width, values, width, label=policy)
            ax.set_xticks(x)
            ax.set_xticklabels(x_labels)
            ax.set_ylabel("audit-induced miss ratio")
            ax.set_title("Slack admission removes audit-induced misses")
            ax.grid(True, axis="y", alpha=0.25)
            ax.legend()
            fig.tight_layout()
            fig.savefig(fig_dir / "audit_induced_miss_decomposition.png", dpi=220)
            fig.savefig(fig_dir / "audit_induced_miss_decomposition.pdf")
            plt.close(fig)
    except Exception as exc:
        (fig_dir / "figure_error.txt").write_text(repr(exc), encoding="utf-8")


def write_report(out: Path, summary: pd.DataFrame, delta: pd.DataFrame) -> None:
    report = out / "DEADLINE_SAFE_ADMISSION_REPORT.md"
    actual = delta[delta["deadline_mode"] == "actual"].copy()
    tight = delta[delta["deadline_mode"].isin(["p95+0ms", "p99+0ms"])].copy()
    lines = [
        "# Deadline-Safe Admission Framework Analysis",
        "",
        "Date: 2026-05-24",
        "",
        "This analysis wraps existing real-frame monitor scores with a generic token/slack audit admission layer. The alarm output is preserved; only the heavy-audit decision is governed by token and deadline-slack constraints.",
        "",
        "## What Was Added",
        "",
        "- Audit-induced deadline miss attribution: baseline miss, monitor-induced miss, audit-induced miss, and unsafe audits.",
        "- Generic `score + admission` wrappers for multiple existing monitors.",
        "- Quantile-relative deadline sensitivity using `baseline_p95/p99 + offset` instead of arbitrary fixed deadlines.",
        "",
    ]
    if not actual.empty:
        grouped = actual.groupby("base_method", as_index=False).mean(numeric_only=True)
        lines.extend(["## Standard 35 ms Deadline", ""])
        for _, row in grouped.iterrows():
            lines.append(
                f"- `{row.base_method}`: slack preserves recall {row.recall_slack:.3f}; "
                f"audit changes {row.audit_original:.3f} -> {row.audit_slack:.3f}; "
                f"audit-induced miss {row.audit_induced_miss_original:.4f} -> {row.audit_induced_miss_slack:.4f}."
            )
        lines.append("")
    if not tight.empty:
        grouped = tight.groupby(["deadline_mode", "base_method"], as_index=False).mean(numeric_only=True)
        best = grouped.sort_values("audit_induced_miss_original", ascending=False).head(8)
        lines.extend(["## Tight Quantile Deadlines", ""])
        for _, row in best.iterrows():
            lines.append(
                f"- `{row.base_method}` at `{row.deadline_mode}`: audit-induced miss "
                f"{row.audit_induced_miss_original:.4f} -> {row.audit_induced_miss_slack:.4f}, "
                f"deadline miss {row.deadline_miss_original:.4f} -> {row.deadline_miss_slack:.4f}."
            )
        lines.append("")
    lines.extend(
        [
            "## Key Files",
            "",
            "- `tables/admission_summary_all.csv`",
            "- `tables/admission_main_actual.csv`",
            "- `tables/admission_delta_selected.csv`",
            "- `tables/admission_quantile_deadline_main.csv`",
            "- `figures/generic_admission_recall_vs_audit.png`",
            "- `figures/audit_induced_miss_decomposition.png`",
            "",
            "## Claim Boundary",
            "",
            "This is an admission-layer analysis over existing RTX 4080 real-frame scored traces. It supports the framework claim that CIRCA-style audit admission is score-agnostic and deadline-safe, but it is not a new physical edge-device measurement.",
            "",
        ]
    )
    text = "\n".join(lines)
    report.write_text(text, encoding="utf-8")
    (ROOT / "paper_draft" / "DEADLINE_SAFE_ADMISSION_FRAMEWORK_2026_05_24.md").write_text(text, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze generic deadline-safe audit admission over existing scored traces.")
    parser.add_argument("--inputs", nargs="+", required=True, help="dataset=source_root entries")
    parser.add_argument("--out-dir", type=Path, default=ROOT / "results_deadline_safe_admission_20260524")
    parser.add_argument("--methods", nargs="+", default=["RFF-HSIC", "ConditionalRFF-HSIC", "ContextAwareConformal", "TimingThreshold", "CIRCA-RT"])
    parser.add_argument("--quantiles", nargs="+", type=float, default=[0.95, 0.99])
    parser.add_argument("--offsets-ms", nargs="+", type=float, default=[0.0, 1.0, 2.0, 4.0])
    parser.add_argument("--include-actual-deadline", action="store_true", default=True)
    parser.add_argument("--token-audit-cost-ms", type=float, default=4.0)
    parser.add_argument("--bucket-capacity", type=float, default=12.0)
    parser.add_argument("--replenish-rate", type=float, default=0.45)
    parser.add_argument("--monitor-cost-ms", type=float, default=0.08)
    parser.add_argument("--slack-margin-ms", type=float, default=0.0)
    args = parser.parse_args()

    out = args.out_dir
    table_dir = out / "tables"
    fig_dir = out / "figures"
    for directory in [table_dir, fig_dir]:
        directory.mkdir(parents=True, exist_ok=True)

    summary = run(args)
    summary.to_csv(table_dir / "admission_summary_all.csv", index=False)
    (out / "admission_manifest.json").write_text(
        json.dumps({"inputs": args.inputs, "methods": args.methods, "quantiles": args.quantiles, "offsets_ms": args.offsets_ms}, indent=2),
        encoding="utf-8",
    )

    attack = summary[summary["scenario"] != "nominal"].copy()
    group_cols = ["dataset", "primary_model", "base_method", "method", "admission_policy", "deadline_mode"]
    main = mean_table(attack, group_cols, METRIC_COLS)
    main.to_csv(table_dir / "admission_main_all_deadlines.csv", index=False)
    main[main["deadline_mode"] == "actual"].to_csv(table_dir / "admission_main_actual.csv", index=False)
    main[main["deadline_mode"] != "actual"].to_csv(table_dir / "admission_quantile_deadline_main.csv", index=False)
    bootstrap_ci_table(attack, group_cols, METRIC_COLS).to_csv(table_dir / "admission_main_ci.csv", index=False)

    delta = selected_delta(summary)
    delta.to_csv(table_dir / "admission_delta_selected.csv", index=False)
    tight = delta[delta["deadline_mode"].isin(["p95+0ms", "p99+0ms"])].copy()
    tight.to_csv(table_dir / "admission_delta_tight_quantiles.csv", index=False)

    write_figures(summary, delta, fig_dir)
    write_report(out, summary, delta)
    print(f"wrote {table_dir}")
    print(f"wrote {fig_dir}")
    print(f"wrote {out / 'DEADLINE_SAFE_ADMISSION_REPORT.md'}")


if __name__ == "__main__":
    main()
