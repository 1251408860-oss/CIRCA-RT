from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from circa_rt.baselines import BASELINE_FUNCS, BaselineConfig
from circa_rt.metrics import summarize_detection
from circa_rt.schema import read_trace


def build_config(config_path: Path) -> BaselineConfig:
    cfg = json.loads(config_path.read_text(encoding="utf-8"))["baselines"]
    return BaselineConfig(
        window_size=int(cfg["window_size"]),
        monitor_cost_ms=float(cfg["monitor_cost_ms"]),
        audit_cost_ms=float(cfg["audit_cost_ms"]),
        periodic_audit_interval=int(cfg["periodic_audit_interval"]),
        random_audit_rate=float(cfg["random_audit_rate"]),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--config", default=str(ROOT / "configs" / "experiment_config.json"))
    parser.add_argument("--methods", nargs="*", default=list(BASELINE_FUNCS.keys()))
    args = parser.parse_args()

    input_path = Path(args.input)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    df = read_trace(input_path)
    cfg = build_config(Path(args.config))
    summaries = []
    for method in args.methods:
        if method not in BASELINE_FUNCS:
            raise ValueError(f"unknown method: {method}")
        result = BASELINE_FUNCS[method](df, cfg)
        method_dir = out_dir / method
        method_dir.mkdir(parents=True, exist_ok=True)
        result.to_csv(method_dir / "scored.csv", index=False)
        summaries.append(summarize_detection(result, trace_name=input_path.stem).to_dict())
    summary_df = pd.DataFrame(summaries)
    summary_df.to_csv(out_dir / "summary.csv", index=False)
    print(summary_df.to_string(index=False))


if __name__ == "__main__":
    main()
