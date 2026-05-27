from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_FEATURES = (
    "semantic_score",
    "semantic_residual",
    "timing_residual_ms",
    "latency_ms",
    "context_brightness",
    "context_edge_density",
    "context_motion",
)


@dataclass(frozen=True)
class TraceFile:
    path: Path
    model: str
    seed: str
    scenario: str


def discover_trace_files(trace_root: Path) -> list[TraceFile]:
    files: list[TraceFile] = []
    for path in sorted(trace_root.rglob("*.csv")):
        rel = path.relative_to(trace_root)
        if len(rel.parts) < 3:
            continue
        model = rel.parts[0]
        seed = rel.parts[1].replace("seed", "")
        scenario = path.stem
        files.append(TraceFile(path=path, model=model, seed=seed, scenario=scenario))
    return files


def label_column(df: pd.DataFrame) -> pd.Series:
    for col in ("is_fault", "fault", "label", "y"):
        if col in df.columns:
            return df[col].fillna(0).astype(int)
    if "scenario" in df.columns:
        return (df["scenario"].astype(str).str.lower() != "nominal").astype(int)
    return pd.Series(np.zeros(len(df), dtype=np.int64), index=df.index)


def feature_frame(df: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    available = [c for c in features if c in df.columns]
    if not available:
        numeric = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
        blocked = {"is_fault", "fault", "label", "y", "frame_idx", "idx", "seed"}
        available = [c for c in numeric if c not in blocked]
    if not available:
        raise ValueError("no usable numeric feature columns found")
    out = df[available].apply(pd.to_numeric, errors="coerce")
    return out.replace([np.inf, -np.inf], np.nan).ffill().bfill().fillna(0.0)


def write_split(
    *,
    out_dir: Path,
    trace: TraceFile,
    scenario: str,
    features: pd.DataFrame,
    labels: pd.Series,
    train: bool,
) -> dict[str, str | int]:
    split_dir = out_dir / trace.model / f"seed{trace.seed}"
    split_dir.mkdir(parents=True, exist_ok=True)
    stem = "train_nominal" if train else f"test_{scenario}"
    csv_path = split_dir / f"{stem}.csv"
    npy_path = split_dir / f"{stem}.npy"
    label_path = split_dir / f"{stem}_labels.npy"
    features.to_csv(csv_path, index=False)
    np.save(npy_path, features.to_numpy(dtype=np.float32))
    np.save(label_path, labels.to_numpy(dtype=np.int64))
    return {
        "model": trace.model,
        "seed": trace.seed,
        "scenario": scenario,
        "split": "train" if train else "test",
        "rows": len(features),
        "csv": str(csv_path),
        "npy": str(npy_path),
        "labels": str(label_path),
        "source": str(trace.path),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export CIRCA-RT trace CSVs into train/test arrays consumable by official baseline repositories."
    )
    parser.add_argument(
        "--trace-root",
        type=Path,
        required=True,
        help="Trace root, e.g. results_autodl_perception_semantics_realframes_av2/traces.",
    )
    parser.add_argument("--out-dir", type=Path, default=ROOT / "data" / "official_baseline_inputs")
    parser.add_argument("--features", nargs="+", default=list(DEFAULT_FEATURES))
    parser.add_argument("--nominal-scenario", default="nominal")
    parser.add_argument("--include-nominal-test", action="store_true")
    args = parser.parse_args()

    trace_root = args.trace_root.resolve()
    traces = discover_trace_files(trace_root)
    if not traces:
        raise SystemExit(f"no trace CSV files found under {trace_root}")

    grouped: dict[tuple[str, str], dict[str, TraceFile]] = {}
    for trace in traces:
        grouped.setdefault((trace.model, trace.seed), {})[trace.scenario] = trace

    manifest_rows: list[dict[str, str | int]] = []
    feature_meta: dict[str, list[str]] = {}
    for (model, seed), by_scenario in sorted(grouped.items()):
        nominal = by_scenario.get(args.nominal_scenario)
        if nominal is None:
            continue
        train_df = pd.read_csv(nominal.path)
        train_x = feature_frame(train_df, args.features)
        feature_meta[f"{model}/seed{seed}"] = list(train_x.columns)
        trace_stub = TraceFile(path=nominal.path, model=model, seed=seed, scenario=args.nominal_scenario)
        manifest_rows.append(
            write_split(
                out_dir=args.out_dir.resolve(),
                trace=trace_stub,
                scenario=args.nominal_scenario,
                features=train_x,
                labels=pd.Series(np.zeros(len(train_x), dtype=np.int64)),
                train=True,
            )
        )

        for scenario, trace in sorted(by_scenario.items()):
            if scenario == args.nominal_scenario and not args.include_nominal_test:
                continue
            test_df = pd.read_csv(trace.path)
            test_x = feature_frame(test_df, list(train_x.columns))
            test_y = label_column(test_df)
            manifest_rows.append(
                write_split(
                    out_dir=args.out_dir.resolve(),
                    trace=trace,
                    scenario=scenario,
                    features=test_x,
                    labels=test_y,
                    train=False,
                )
            )

    if not manifest_rows:
        raise SystemExit(f"no nominal scenario '{args.nominal_scenario}' found under {trace_root}")

    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = out_dir / "manifest.csv"
    with manifest.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(manifest_rows[0].keys()))
        writer.writeheader()
        writer.writerows(manifest_rows)
    (out_dir / "feature_columns.json").write_text(json.dumps(feature_meta, indent=2), encoding="utf-8")
    print(f"exported_splits={len(manifest_rows)} out_dir={out_dir} manifest={manifest}")


if __name__ == "__main__":
    main()
