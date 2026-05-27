from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean, stdev


METHOD_NAMES = {
    "modern_tcn": "ModernTCN_official",
    "time_mixer": "TimeMixer_official",
}


def _parse_case_name(path: Path) -> tuple[str, int | None, str]:
    parts = path.stem.split("__")
    if len(parts) < 3:
        return path.stem, None, "unknown"
    primary_model = parts[0]
    seed_text = parts[1]
    seed = int(seed_text.replace("seed", "")) if seed_text.startswith("seed") else None
    scenario = "__".join(parts[2:])
    return primary_model, seed, scenario


def _metric(value: object) -> float | None:
    if isinstance(value, int | float):
        return float(value)
    return None


def collect(input_dir: Path, dataset: str) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for method_dir in sorted(p for p in input_dir.iterdir() if p.is_dir()):
        method = METHOD_NAMES.get(method_dir.name, method_dir.name)
        for path in sorted(method_dir.glob("*.json")):
            payload = json.loads(path.read_text(encoding="utf-8"))
            primary_model, seed, scenario = _parse_case_name(path)
            metrics = payload.get("metrics") or {}
            rows.append(
                {
                    "dataset": dataset,
                    "method": method,
                    "primary_model": primary_model,
                    "seed": "" if seed is None else seed,
                    "scenario": scenario,
                    "returncode": payload.get("returncode"),
                    "timed_out": bool(payload.get("timed_out")),
                    "smoke": bool(payload.get("smoke")),
                    "accuracy": _metric(metrics.get("accuracy")),
                    "precision": _metric(metrics.get("precision")),
                    "recall": _metric(metrics.get("recall")),
                    "f1": _metric(metrics.get("f1")),
                    "case_dir": payload.get("case_dir"),
                    "json_path": str(path),
                }
            )
    return rows


def write_case_rows(rows: list[dict[str, object]], out_csv: Path) -> None:
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "dataset",
        "method",
        "primary_model",
        "seed",
        "scenario",
        "returncode",
        "timed_out",
        "smoke",
        "accuracy",
        "precision",
        "recall",
        "f1",
        "case_dir",
        "json_path",
    ]
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _summarize_group(rows: list[dict[str, object]], keys: tuple[str, ...]) -> list[dict[str, object]]:
    groups: dict[tuple[object, ...], list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        groups[tuple(row[k] for k in keys)].append(row)

    summary: list[dict[str, object]] = []
    for group_key, group_rows in sorted(groups.items()):
        item = {key: value for key, value in zip(keys, group_key)}
        item["n_cases"] = len(group_rows)
        item["n_success"] = sum(1 for r in group_rows if r["returncode"] == 0 and not r["timed_out"])
        item["n_timeout"] = sum(1 for r in group_rows if r["timed_out"])
        for metric in ("accuracy", "precision", "recall", "f1"):
            values = [float(r[metric]) for r in group_rows if isinstance(r.get(metric), float)]
            item[f"{metric}_mean"] = mean(values) if values else ""
            item[f"{metric}_std"] = stdev(values) if len(values) > 1 else 0.0 if values else ""
        summary.append(item)
    return summary


def write_summary(rows: list[dict[str, object]], out_csv: Path, keys: tuple[str, ...]) -> None:
    summary = _summarize_group(rows, keys)
    fields = list(keys) + [
        "n_cases",
        "n_success",
        "n_timeout",
        "accuracy_mean",
        "accuracy_std",
        "precision_mean",
        "precision_std",
        "recall_mean",
        "recall_std",
        "f1_mean",
        "f1_std",
    ]
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(summary)


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize official PSM baseline JSON outputs into CSV tables.")
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--dataset", default="av2_official_psm")
    args = parser.parse_args()

    rows = collect(args.input_dir.resolve(), args.dataset)
    if not rows:
        raise SystemExit(f"no official baseline JSON files found under {args.input_dir}")

    write_case_rows(rows, args.out_dir / "official_psm_case_metrics.csv")
    write_summary(rows, args.out_dir / "official_psm_summary_by_method_model.csv", ("dataset", "method", "primary_model"))
    write_summary(rows, args.out_dir / "official_psm_summary_by_method.csv", ("dataset", "method"))
    print(f"wrote official summary rows={len(rows)} out_dir={args.out_dir.resolve()}")


if __name__ == "__main__":
    main()
