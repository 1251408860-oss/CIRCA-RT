from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from circa_rt.core import CIRCARuntimeConfig, apply_circa_rt, fit_circa_rt
from circa_rt.features import rolling_cross_scores_by_group
from circa_rt.schema import CONTEXT_COLUMNS, read_trace


def build_config(args: argparse.Namespace) -> CIRCARuntimeConfig:
    if args.config and args.config.exists():
        cfg = json.loads(args.config.read_text(encoding="utf-8"))["circa_rt"]
        return CIRCARuntimeConfig(
            window_size=int(cfg["window_size"]),
            feature_dim=int(cfg["feature_dim"]),
            low_quantile=float(cfg["low_quantile"]),
            high_quantile=float(cfg["high_quantile"]),
            monitor_cost_ms=float(cfg["monitor_cost_ms"]),
            heavy_audit_cost_ms=float(cfg["heavy_audit_cost_ms"]),
            bucket_capacity=float(cfg["bucket_capacity"]),
            replenish_rate=float(cfg["replenish_rate"]),
            seed=int(cfg.get("seed", 7)),
        )
    return CIRCARuntimeConfig(
        window_size=args.window_size,
        feature_dim=args.feature_dim,
        low_quantile=args.low_quantile,
        high_quantile=args.high_quantile,
        monitor_cost_ms=0.0,
        heavy_audit_cost_ms=args.token_audit_cost_ms,
        bucket_capacity=args.bucket_capacity,
        replenish_rate=args.replenish_rate,
        seed=args.seed,
    )


def summarize(component: str, samples: list[float], n: int, repeat: int) -> dict[str, object]:
    arr = np.asarray(samples, dtype=np.float64)
    return {
        "component": component,
        "n_frames": int(n),
        "repeat": int(repeat),
        "mean_ms_per_frame": float(np.mean(arr)),
        "p50_ms_per_frame": float(np.quantile(arr, 0.50)),
        "p95_ms_per_frame": float(np.quantile(arr, 0.95)),
        "p99_ms_per_frame": float(np.quantile(arr, 0.99)),
        "max_ms_per_frame": float(np.max(arr)),
    }


def timed_per_frame(fn, n: int, repeat: int, warmup: int) -> list[float]:
    for _ in range(warmup):
        fn()
    samples: list[float] = []
    for _ in range(repeat):
        start = time.perf_counter()
        fn()
        samples.append((time.perf_counter() - start) * 1000.0 / max(n, 1))
    return samples


def main() -> None:
    parser = argparse.ArgumentParser(description="Microbenchmark CIRCA-RT monitor scoring overhead on a scored trace.")
    parser.add_argument("--trace", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=ROOT / "configs" / "experiment_config.json")
    parser.add_argument("--repeat", type=int, default=200)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--window-size", type=int, default=32)
    parser.add_argument("--feature-dim", type=int, default=32)
    parser.add_argument("--low-quantile", type=float, default=0.95)
    parser.add_argument("--high-quantile", type=float, default=0.99)
    parser.add_argument("--token-audit-cost-ms", type=float, default=4.0)
    parser.add_argument("--bucket-capacity", type=float, default=12.0)
    parser.add_argument("--replenish-rate", type=float, default=0.45)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--note", type=str, default="")
    args = parser.parse_args()

    df = read_trace(args.trace)
    if df.empty:
        raise SystemExit(f"empty trace: {args.trace}")

    cfg = build_config(args)
    fit_df = df.copy()
    if int((fit_df["label"].astype(int) == 0).sum()) < max(10, cfg.window_size):
        raise SystemExit("trace does not contain enough benign rows for calibration")
    model = fit_circa_rt(fit_df, cfg)
    n = len(df)

    context = df[CONTEXT_COLUMNS].to_numpy(dtype=np.float64)
    mode = df["mode"].astype(str).to_numpy()
    semantic = df["semantic_residual"].to_numpy(dtype=np.float64)
    timing = df["timing_residual_ms"].to_numpy(dtype=np.float64)
    groups = df["run_id"].astype(str).to_numpy()

    def residualization():
        return model.residualizer.transform(semantic, timing, context, mode)

    u0, v0 = residualization()

    def rff_transform():
        return model.rff_u.transform(u0), model.rff_v.transform(v0)

    phi_u0, phi_v0 = rff_transform()

    def rolling_score():
        return rolling_cross_scores_by_group(phi_u0, phi_v0, groups, cfg.window_size)

    def full_apply():
        return apply_circa_rt(model, df)

    rows = [
        summarize("residualization", timed_per_frame(residualization, n, args.repeat, args.warmup), n, args.repeat),
        summarize("rff_transform", timed_per_frame(rff_transform, n, args.repeat, args.warmup), n, args.repeat),
        summarize("rolling_score", timed_per_frame(rolling_score, n, args.repeat, args.warmup), n, args.repeat),
        summarize("full_apply_per_frame", timed_per_frame(full_apply, n, args.repeat, args.warmup), n, args.repeat),
    ]
    out = pd.DataFrame(rows)
    out["source_trace"] = str(args.trace)
    if args.note:
        out["note"] = args.note
    args.out.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.out, index=False)
    print(out.to_string(index=False))
    print(args.out)


if __name__ == "__main__":
    main()
