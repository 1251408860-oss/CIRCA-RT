from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from circa_rt.schema import write_trace
from circa_rt.synthetic import generate_trace


TRACE_TYPES = [
    "nominal",
    "mode_shift",
    "stealthy_coupled_attack",
    "timing_only_attack",
    "semantic_only_attack",
    "mixed_attack",
]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(ROOT / "configs" / "experiment_config.json"))
    parser.add_argument("--out-dir", default=str(ROOT / "traces" / "synthetic"))
    parser.add_argument("--n", type=int, default=None)
    parser.add_argument("--n-seeds", type=int, default=None)
    args = parser.parse_args()

    cfg = json.loads(Path(args.config).read_text(encoding="utf-8"))
    n = args.n or int(cfg["n_per_trace"])
    n_seeds = args.n_seeds or int(cfg["n_seeds"])
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for seed in range(n_seeds):
        for trace_type in TRACE_TYPES:
            df = generate_trace(
                trace_type=trace_type,
                seed=int(cfg["random_seed"]) + seed,
                n=n,
                deadline_ms=float(cfg["deadline_ms"]),
                baseline_latency_ms=float(cfg["baseline_latency_ms"]),
            )
            write_trace(df, out_dir / f"{trace_type}_seed{seed}.csv")
    print(f"generated {n_seeds * len(TRACE_TYPES)} traces in {out_dir}")


if __name__ == "__main__":
    main()
