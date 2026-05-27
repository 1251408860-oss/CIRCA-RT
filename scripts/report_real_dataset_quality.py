from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]


def summarize(manifest: Path) -> dict[str, object]:
    df = pd.read_csv(manifest)
    summary = {
        "manifest": str(manifest.resolve()),
        "rows": int(len(df)),
        "datasets": df["dataset_hint"].value_counts().to_dict(),
        "sequence_count": int(df["sequence_hint"].nunique()),
        "top_sequences": df["sequence_hint"].value_counts().head(15).to_dict(),
        "width_min": int(df["width"].min()),
        "width_max": int(df["width"].max()),
        "height_min": int(df["height"].min()),
        "height_max": int(df["height"].max()),
    }
    sizes = df[["width", "height"]].value_counts().head(10)
    summary["top_sizes"] = {f"{int(w)}x{int(h)}": int(count) for (w, h), count in sizes.items()}
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Report quality statistics for a normalized real-frame dataset.")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=None, help="Optional JSON report output path.")
    args = parser.parse_args()

    summary = summarize(args.manifest.resolve())
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
