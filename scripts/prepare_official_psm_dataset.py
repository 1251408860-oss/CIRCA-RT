from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class SplitRow:
    model: str
    seed: str
    scenario: str
    split: str
    rows: int
    npy: Path
    labels: Path


def _load_manifest(manifest: Path) -> list[SplitRow]:
    df = pd.read_csv(manifest)
    rows: list[SplitRow] = []
    for _, row in df.iterrows():
        rows.append(
            SplitRow(
                model=str(row["model"]),
                seed=str(row["seed"]),
                scenario=str(row["scenario"]),
                split=str(row["split"]),
                rows=int(row["rows"]),
                npy=Path(str(row["npy"])).resolve(),
                labels=Path(str(row["labels"])).resolve(),
            )
        )
    return rows


def _to_frame(values: np.ndarray) -> pd.DataFrame:
    if values.ndim == 1:
        values = values[:, None]
    cols = {"timestamp": np.arange(values.shape[0], dtype=np.int64)}
    for idx in range(values.shape[1]):
        cols[f"f{idx:02d}"] = values[:, idx]
    return pd.DataFrame(cols)


def _to_label_frame(labels: np.ndarray) -> pd.DataFrame:
    labels = np.asarray(labels).reshape(-1)
    return pd.DataFrame({"timestamp": np.arange(labels.shape[0], dtype=np.int64), "label": labels.astype(np.int64)})


def _crop_rows(values: np.ndarray, *, max_rows: int) -> np.ndarray:
    if max_rows <= 0 or values.shape[0] <= max_rows:
        return values
    return values[:max_rows]


def _crop_test_rows(values: np.ndarray, labels: np.ndarray, *, max_rows: int) -> tuple[np.ndarray, np.ndarray]:
    labels = np.asarray(labels).reshape(-1)
    if max_rows <= 0 or values.shape[0] <= max_rows:
        return values, labels
    positives = np.flatnonzero(labels > 0)
    if positives.size == 0:
        start = 0
    else:
        start = max(0, int(positives[0]) - max_rows // 4)
        start = min(start, values.shape[0] - max_rows)
    end = start + max_rows
    return values[start:end], labels[start:end]


def export_case(
    *,
    train_row: SplitRow,
    test_row: SplitRow,
    out_dir: Path,
    max_train_rows: int,
    max_test_rows: int,
) -> dict[str, object]:
    train = np.load(train_row.npy)
    test = np.load(test_row.npy)
    labels = np.load(test_row.labels)
    if train.ndim != 2 or test.ndim != 2:
        raise ValueError(f"expected 2D train/test arrays, got {train.shape} and {test.shape}")
    if test.shape[0] != labels.reshape(-1).shape[0]:
        raise ValueError(f"test rows and label rows mismatch for {test_row.npy}")
    train = _crop_rows(train, max_rows=max_train_rows)
    test, labels = _crop_test_rows(test, labels, max_rows=max_test_rows)

    out_dir.mkdir(parents=True, exist_ok=True)
    _to_frame(train).to_csv(out_dir / "train.csv", index=False)
    _to_frame(test).to_csv(out_dir / "test.csv", index=False)
    _to_label_frame(labels).to_csv(out_dir / "test_label.csv", index=False)

    anomaly_ratio_pct = float(labels.reshape(-1).mean() * 100.0)
    meta = {
        "model": test_row.model,
        "seed": test_row.seed,
        "scenario": test_row.scenario,
        "train_rows": int(train.shape[0]),
        "test_rows": int(test.shape[0]),
        "feature_dim": int(train.shape[1]),
        "anomaly_ratio_pct": anomaly_ratio_pct,
        "crop": {
            "max_train_rows": int(max_train_rows),
            "max_test_rows": int(max_test_rows),
        },
        "sources": {
            "train_npy": str(train_row.npy),
            "test_npy": str(test_row.npy),
            "test_labels": str(test_row.labels),
        },
    }
    (out_dir / "metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return meta


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert exported official-baseline inputs into the official PSM-style directory format."
    )
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--include-nominal-test", action="store_true")
    parser.add_argument("--max-train-rows", type=int, default=0, help="Optional truncation for smoke tests; 0 keeps all rows.")
    parser.add_argument("--max-test-rows", type=int, default=0, help="Optional truncation for smoke tests; 0 keeps all rows.")
    args = parser.parse_args()

    rows = _load_manifest(args.manifest.resolve())
    grouped: dict[tuple[str, str], dict[str, SplitRow]] = {}
    for row in rows:
        grouped.setdefault((row.model, row.seed), {})[f"{row.split}:{row.scenario}"] = row

    exported: list[dict[str, object]] = []
    for (model, seed), by_key in sorted(grouped.items()):
        train_row = by_key.get("train:nominal")
        if train_row is None:
            continue
        for key, test_row in sorted(by_key.items()):
            if test_row.split != "test":
                continue
            if test_row.scenario == "nominal" and not args.include_nominal_test:
                continue
            case_dir = args.out_dir.resolve() / model / f"seed{seed}" / test_row.scenario
            meta = export_case(
                train_row=train_row,
                test_row=test_row,
                out_dir=case_dir,
                max_train_rows=max(0, int(args.max_train_rows)),
                max_test_rows=max(0, int(args.max_test_rows)),
            )
            exported.append({"case_dir": str(case_dir), **meta})

    if not exported:
        raise SystemExit("no PSM-style datasets were exported")
    summary = pd.DataFrame(exported)
    summary.to_csv(args.out_dir.resolve() / "manifest.csv", index=False)
    print(f"exported_cases={len(exported)} out_dir={args.out_dir.resolve()}")


if __name__ == "__main__":
    main()
