from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


METHOD_ORDER = [
    "CIRCA-RT",
    "ContextAwareConformal",
    "ConditionalRFF-HSIC",
    "RandomBudgetAudit",
    "AlwaysAudit",
]


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute deadline-miss sensitivity from measured latency traces.")
    parser.add_argument("--trace", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--deadlines-ms", type=float, nargs="+", default=[33.333, 34.0, 35.0, 36.0])
    parser.add_argument("--note", type=str, default="")
    args = parser.parse_args()

    df = pd.read_csv(args.trace)
    rows: list[dict[str, object]] = []
    for deadline in args.deadlines_ms:
        for (scenario, method), group in df.groupby(["scenario", "method"], dropna=False):
            latency = group["e2e_latency_ms"].astype(float)
            rows.append(
                {
                    "scenario": scenario,
                    "method": method,
                    "deadline_ms": float(deadline),
                    "audit_rate": float(group["audit"].mean()),
                    "miss_ratio_at_deadline": float((latency > deadline).mean()),
                    "p95_latency_ms": float(latency.quantile(0.95)),
                    "p99_latency_ms": float(latency.quantile(0.99)),
                    "p999_latency_ms": float(latency.quantile(0.999)),
                    "p99_slack_ms": float(deadline - latency.quantile(0.99)),
                    "p999_slack_ms": float(deadline - latency.quantile(0.999)),
                    "min_slack_ms": float((deadline - latency).min()),
                }
            )

    out = pd.DataFrame(rows)
    out["method"] = pd.Categorical(out["method"], METHOD_ORDER, ordered=True)
    out.sort_values(["scenario", "deadline_ms", "method"], inplace=True)
    out["method"] = out["method"].astype(str)
    if args.note:
        out["note"] = args.note

    args.out.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.out, index=False)

    selected = out[
        (out["scenario"] == "coupled_semantic_timing_attack")
        & (out["method"].isin(["CIRCA-RT", "ContextAwareConformal", "AlwaysAudit"]))
    ].copy()
    print(selected.to_string(index=False))
    print(args.out)


if __name__ == "__main__":
    main()
