from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from circa_rt.core import CIRCARuntimeConfig, run_circa_rt
from circa_rt.metrics import summarize_detection
from circa_rt.schema import read_trace


def build_config(config_path: Path) -> CIRCARuntimeConfig:
    cfg = json.loads(config_path.read_text(encoding="utf-8"))["circa_rt"]
    return CIRCARuntimeConfig(
        window_size=int(cfg["window_size"]),
        feature_dim=int(cfg["feature_dim"]),
        low_quantile=float(cfg["low_quantile"]),
        high_quantile=float(cfg["high_quantile"]),
        monitor_cost_ms=float(cfg["monitor_cost_ms"]),
        heavy_audit_cost_ms=float(cfg["heavy_audit_cost_ms"]),
        bucket_capacity=float(cfg["bucket_capacity"]),
        replenish_rate=float(cfg["replenish_rate"]),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--config", default=str(ROOT / "configs" / "experiment_config.json"))
    args = parser.parse_args()

    input_path = Path(args.input)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    df = read_trace(input_path)
    result = run_circa_rt(df, build_config(Path(args.config)))
    result.to_csv(out_dir / "scored.csv", index=False)
    summary = summarize_detection(result, trace_name=input_path.stem).to_dict()
    Path(out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
