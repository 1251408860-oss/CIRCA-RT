from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from circa_rt.perception_semantics import max_rolling_sum


def load_default_cfg() -> tuple[float, float]:
    cfg_path = ROOT / "configs" / "experiment_config.json"
    if not cfg_path.exists():
        return 12.0, 0.45
    cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
    c = cfg.get("circa_rt", {})
    return float(c.get("bucket_capacity", 12.0)), float(c.get("replenish_rate", 0.45))


def infer_method(path: Path, df: pd.DataFrame) -> str:
    if "method" in df.columns and not df.empty:
        return str(df["method"].iloc[0])
    return path.parent.name


def validate_file(path: Path, args: argparse.Namespace) -> list[dict[str, object]]:
    df = pd.read_csv(path)
    if df.empty:
        return []
    method = infer_method(path, df)
    if args.methods and method not in set(args.methods):
        return []
    rows: list[dict[str, object]] = []
    groups = df.groupby("run_id", dropna=False) if "run_id" in df.columns else [("", df)]
    for run_id, group in groups:
        capacity = float(group["bucket_capacity"].dropna().iloc[0]) if "bucket_capacity" in group and group["bucket_capacity"].notna().any() else args.bucket_capacity
        replenish = float(group["replenish_rate"].dropna().iloc[0]) if "replenish_rate" in group and group["replenish_rate"].notna().any() else args.replenish_rate
        for cost_col in args.cost_cols:
            if cost_col not in group.columns:
                continue
            costs = group[cost_col].to_numpy(dtype=np.float64)
            n = len(costs)
            windows = sorted(set([w for w in args.windows if w > 0] + [n]))
            for window in windows:
                w = min(window, n)
                observed = max_rolling_sum(costs, w)
                bound = capacity + replenish * w
                rows.append(
                    {
                        "source": str(path.relative_to(args.input_root)),
                        "method": method,
                        "run_id": str(run_id),
                        "n": int(n),
                        "window": int(w),
                        "cost_col": cost_col,
                        "max_window_cost_ms": observed,
                        "bound_ms": float(bound),
                        "margin_ms": float(bound - observed),
                        "pass": bool(observed <= bound + args.tolerance_ms),
                        "bucket_capacity": capacity,
                        "replenish_rate": replenish,
                    }
                )
    return rows


def main() -> None:
    default_capacity, default_replenish = load_default_cfg()
    parser = argparse.ArgumentParser(description="Validate CIRCA-RT token-bucket audit-cost envelope over scored CSV files.")
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--methods", nargs="+", default=["CIRCA-RT"])
    parser.add_argument("--windows", nargs="+", type=int, default=[32, 64, 128, 256, 512, 1024])
    parser.add_argument("--cost-cols", nargs="+", default=["audit_token_charge_ms", "audit_charge_ms"])
    parser.add_argument("--bucket-capacity", type=float, default=default_capacity)
    parser.add_argument("--replenish-rate", type=float, default=default_replenish)
    parser.add_argument("--tolerance-ms", type=float, default=1e-6)
    args = parser.parse_args()

    if not args.input_root.exists():
        raise SystemExit(f"input root does not exist: {args.input_root}")
    rows: list[dict[str, object]] = []
    for path in sorted(args.input_root.rglob("scored.csv")):
        rows.extend(validate_file(path, args))
    out = pd.DataFrame(rows)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.out, index=False)
    if out.empty:
        raise SystemExit("no matching scored.csv files were found")
    failed = out[~out["pass"].astype(bool)]
    print(f"wrote {args.out}")
    print(f"checked_rows={len(out)} failed_rows={len(failed)}")
    if not failed.empty:
        print(failed.head(20).to_string(index=False))
        raise SystemExit(2)


if __name__ == "__main__":
    main()
