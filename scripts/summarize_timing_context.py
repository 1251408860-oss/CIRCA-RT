from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize timing/context p99 values by scenario.")
    parser.add_argument("--trace", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--method", type=str, default="CIRCA-RT")
    parser.add_argument("--note", type=str, default="")
    args = parser.parse_args()

    df = pd.read_csv(args.trace)
    if args.method:
        df = df[df["method"].astype(str) == args.method].copy()
    if df.empty:
        raise SystemExit(f"no rows for method={args.method!r} in {args.trace}")

    rows: list[dict[str, object]] = []
    for scenario, group in df.groupby("scenario", dropna=False):
        row = {
            "scenario": scenario,
            "method": args.method,
            "net_delay_p99_ms": float(group["context_net_delay_ms"].quantile(0.99)),
            "jitter_p99_ms": float(group["context_jitter_ms"].quantile(0.99)),
            "queue_depth_p99": float(group["context_queue_depth"].quantile(0.99)),
            "message_age_p99_ms": float(group["context_message_age_ms"].quantile(0.99)),
            "gpu_util_p99": float(group["context_gpu_util"].quantile(0.99)),
            "cpu_util_p99": float(group["context_cpu_util"].quantile(0.99)),
        }
        if args.note:
            row["note"] = args.note
        rows.append(row)

    out = pd.DataFrame(rows).sort_values("scenario")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.out, index=False)
    print(out.to_string(index=False))
    print(args.out)


if __name__ == "__main__":
    main()
