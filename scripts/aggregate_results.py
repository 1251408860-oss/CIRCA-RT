from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary-dir", default=str(ROOT / "results" / "summaries"))
    parser.add_argument("--out-dir", default=str(ROOT / "results" / "tables"))
    args = parser.parse_args()
    summary_dir = Path(args.summary_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    frames = [pd.read_csv(p) for p in summary_dir.glob("*.csv")]
    if not frames:
        raise SystemExit(f"no summary csv files found in {summary_dir}")
    all_df = pd.concat(frames, ignore_index=True)
    all_df.to_csv(out_dir / "summary_all.csv", index=False)
    main = (
        all_df.groupby("method", as_index=False)[
            [
                "attack_recall",
                "false_alarm_rate",
                "detection_delay",
                "audit_rate",
                "mean_audit_cost_ms",
                "p99_latency_ms",
                "p999_latency_ms",
                "deadline_miss_ratio",
                "latency_inflation_mean_ms",
                "auroc",
                "auprc",
            ]
        ]
        .mean()
        .sort_values(["attack_recall", "p99_latency_ms"], ascending=[False, True])
    )
    main.to_csv(out_dir / "main_table.csv", index=False)
    ablation_methods = [
        "CIRCA-RT",
        "ConditionalRFF-HSIC",
        "RFF-HSIC",
        "SemanticThreshold",
        "TimingThreshold",
        "RandomBudgetAudit",
        "AlwaysAudit",
    ]
    all_df[all_df["method"].isin(ablation_methods)].to_csv(out_dir / "ablation_table.csv", index=False)
    print(f"wrote {out_dir / 'main_table.csv'}")


if __name__ == "__main__":
    main()
