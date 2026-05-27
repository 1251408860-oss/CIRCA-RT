from __future__ import annotations

import argparse
import math
from pathlib import Path

import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize token-bucket audit demand by window size.")
    parser.add_argument("--trace", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--windows", type=int, nargs="+", default=[1, 8, 16, 32, 64, 128, 180])
    parser.add_argument("--method", type=str, default="CIRCA-RT")
    parser.add_argument("--note", type=str, default="")
    args = parser.parse_args()

    df = pd.read_csv(args.trace)
    method_df = df[df["method"].astype(str) == args.method].copy()
    if method_df.empty:
        raise SystemExit(f"no rows for method={args.method!r} in {args.trace}")

    capacity = float(method_df["bucket_capacity"].dropna().iloc[0])
    replenish = float(method_df["replenish_rate"].dropna().iloc[0])
    token_cost = float(method_df["token_audit_cost_ms"].dropna().iloc[0])
    audit_latency_bound = float(method_df["audit_latency_bound_ms"].dropna().iloc[0])

    rows: list[dict[str, object]] = []
    for window in args.windows:
        audit_bound = min(window, math.floor((capacity + replenish * window) / token_cost))
        rows.append(
            {
                "window_frames": int(window),
                "circa_audit_bound": int(audit_bound),
                "circa_audit_density_bound": audit_bound / window,
                "always_audit_demand": int(window),
                "always_audit_density": 1.0,
                "audit_reduction_vs_always": 1.0 - audit_bound / window,
                "audit_work_bound_ms": audit_bound * audit_latency_bound,
                "always_audit_work_ms": window * audit_latency_bound,
                "bucket_capacity": capacity,
                "replenish_rate": replenish,
                "token_audit_cost_ms": token_cost,
                "audit_latency_bound_ms": audit_latency_bound,
            }
        )

    out = pd.DataFrame(rows)
    if args.note:
        out["note"] = args.note
    args.out.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.out, index=False)
    print(out.to_string(index=False))
    print(args.out)


if __name__ == "__main__":
    main()
