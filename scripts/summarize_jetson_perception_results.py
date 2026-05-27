from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]


def copy_table(src: Path, dst: Path) -> bool:
    if not src.exists():
        return False
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_bytes(src.read_bytes())
    return True


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize Jetson perception results and device logs.")
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--raw-logs-dir", type=Path, default=None)
    parser.add_argument("--platform-label", type=str, default="jetson_orin_validation")
    args = parser.parse_args()

    out = args.out_dir
    table_dir = out / "tables"
    log_dir = out / "logs"
    raw_logs_dir = args.raw_logs_dir or (out / "raw_logs")
    table_dir.mkdir(parents=True, exist_ok=True)

    copies = [
        ("summary_all.csv", "jetson_summary_all.csv"),
        ("autodl_perception_main_table.csv", "jetson_main_table.csv"),
        ("autodl_perception_main_table_ci.csv", "jetson_main_table_ci.csv"),
        ("autodl_perception_scenario_table.csv", "jetson_scenario_table.csv"),
    ]
    copied: list[str] = []
    for src_name, dst_name in copies:
        src = table_dir / src_name
        dst = table_dir / dst_name
        if copy_table(src, dst):
            copied.append(dst_name)

    summary_file = table_dir / "summary_all.csv"
    summary_rows: dict[str, object] = {
        "out_dir": str(out),
        "platform_label": args.platform_label,
        "copied_tables": copied,
    }
    if summary_file.exists():
        df = pd.read_csv(summary_file)
        summary_rows["summary_rows"] = int(len(df))
        if "platform" in df.columns:
            jetson_df = df[df["platform"].astype(str).str.contains("jetson", case=False, na=False)].copy()
            if not jetson_df.empty:
                jetson_df.to_csv(table_dir / "jetson_summary_rows.csv", index=False)
                summary_rows["jetson_summary_rows"] = int(len(jetson_df))
                if {"method", "attack_recall", "audit_rate", "p99_latency_ms"}.issubset(jetson_df.columns):
                    best = (
                        jetson_df.groupby(["primary_model", "method"], dropna=False)[["attack_recall", "audit_rate", "p99_latency_ms"]]
                        .mean(numeric_only=True)
                        .reset_index()
                        .sort_values(["primary_model", "attack_recall", "audit_rate"], ascending=[True, False, True])
                    )
                    best.to_csv(table_dir / "jetson_method_overview.csv", index=False)
                    summary_rows["jetson_methods"] = sorted(best["method"].dropna().astype(str).unique().tolist())

    tegra_logs = sorted(raw_logs_dir.glob("tegrastats_*.log")) if raw_logs_dir.exists() else []
    if tegra_logs:
        parse_out = table_dir / "tegrastats_samples.csv"
        summary_out = table_dir / "tegrastats_summary.csv"
        subprocess.run(
            [
                sys.executable,
                str(ROOT / "scripts" / "parse_tegrastats.py"),
                "--out",
                str(parse_out),
                "--summary-out",
                str(summary_out),
                *[str(p) for p in tegra_logs],
            ],
            check=True,
        )
        summary_rows["tegrastats_logs"] = [str(p) for p in tegra_logs]

    log_dir.mkdir(parents=True, exist_ok=True)
    (table_dir / "jetson_result_manifest.json").write_text(json.dumps(summary_rows, indent=2, ensure_ascii=False), encoding="utf-8")
    print(table_dir / "jetson_result_manifest.json")


if __name__ == "__main__":
    main()
