from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


METHOD_ORDER = [
    "CIRCA-RT",
    "ContextAwareConformal",
    "ConditionalRFF-HSIC",
    "RandomBudgetAudit",
    "AlwaysAudit",
]


def q(series: pd.Series, quantile: float) -> float:
    return float(series.quantile(quantile))


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize Jetson deadline slack by scenario and method.")
    parser.add_argument("--trace", type=Path, required=True)
    parser.add_argument("--power", type=Path, default=None)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--note", type=str, default="")
    args = parser.parse_args()

    df = pd.read_csv(args.trace)
    rows: list[dict[str, object]] = []
    for (scenario, method), group in df.groupby(["scenario", "method"], dropna=False):
        deadline = float(group["deadline_ms"].dropna().iloc[0])
        p95 = q(group["e2e_latency_ms"], 0.95)
        p99 = q(group["e2e_latency_ms"], 0.99)
        p999 = q(group["e2e_latency_ms"], 0.999)
        rows.append(
            {
                "scenario": scenario,
                "method": method,
                "deadline_ms": deadline,
                "audit_rate": float(group["audit"].mean()),
                "p95_latency_ms": p95,
                "p99_latency_ms": p99,
                "p999_latency_ms": p999,
                "p99_slack_ms": deadline - p99,
                "p999_slack_ms": deadline - p999,
                "min_slack_ms": float((group["deadline_ms"] - group["e2e_latency_ms"]).min()),
                "deadline_miss_ratio": float(group["deadline_miss"].mean()),
                "gpu_util_p99": q(group["context_gpu_util"], 0.99),
                "net_delay_p99_ms": q(group["context_net_delay_ms"], 0.99),
                "message_age_p99_ms": q(group["context_message_age_ms"], 0.99),
            }
        )

    out = pd.DataFrame(rows)
    out["method"] = pd.Categorical(out["method"], METHOD_ORDER, ordered=True)
    out.sort_values(["scenario", "method"], inplace=True)
    out["method"] = out["method"].astype(str)

    if args.power and args.power.exists():
        power = pd.read_csv(args.power)
        keep = [
            "scenario",
            "mean_power_w",
            "peak_temp_c",
            "mean_gpu_util_proxy_pct",
        ]
        out = out.merge(power[[c for c in keep if c in power.columns]], on="scenario", how="left")

    if args.note:
        out["note"] = args.note

    args.out.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.out, index=False)

    selected = out[
        (out["scenario"].isin(["gpu_interference", "coupled_semantic_timing_attack"]))
        & (out["method"].isin(["CIRCA-RT", "ContextAwareConformal", "AlwaysAudit"]))
    ].copy()
    print(selected.to_string(index=False))
    print(args.out)


if __name__ == "__main__":
    main()
