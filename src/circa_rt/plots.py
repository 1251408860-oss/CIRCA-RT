from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def plot_metric_bars(summary: pd.DataFrame, out_dir: str | Path) -> None:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    key_methods = [
        "SemanticThreshold",
        "TimingThreshold",
        "RFF-HSIC",
        "OnlineConformalAnomaly",
        "AlwaysAudit",
        "CIRCA-RT",
    ]
    df = summary[summary["method"].isin(key_methods)].copy()
    if df.empty:
        return
    agg = df.groupby("method", as_index=False)[
        ["attack_recall", "false_alarm_rate", "audit_rate", "p99_latency_ms", "deadline_miss_ratio"]
    ].mean()
    for metric in ["attack_recall", "false_alarm_rate", "audit_rate", "p99_latency_ms", "deadline_miss_ratio"]:
        plt.figure(figsize=(9, 4))
        plt.bar(agg["method"], agg[metric])
        plt.xticks(rotation=35, ha="right")
        plt.ylabel(metric)
        plt.tight_layout()
        plt.savefig(out / f"{metric}.png", dpi=180)
        plt.close()


def plot_trace_scores(raw: pd.DataFrame, scored: pd.DataFrame, out_path: str | Path) -> None:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(10, 4))
    plt.plot(scored["seq"], scored["score"], label="CIRCA-RT score")
    attack = raw["label"].to_numpy() == 1
    if attack.any():
        plt.fill_between(raw["seq"], 0, max(scored["score"].max(), 1e-6), where=attack, alpha=0.2, label="attack")
    if "q_low" in scored.columns:
        plt.axhline(float(scored["q_low"].iloc[0]), linestyle="--", linewidth=1, label="q_low")
    if "q_high" in scored.columns:
        plt.axhline(float(scored["q_high"].iloc[0]), linestyle=":", linewidth=1, label="q_high")
    plt.xlabel("sequence")
    plt.ylabel("score")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close()
