from __future__ import annotations

import argparse
import re
from pathlib import Path

import pandas as pd


RAM_RE = re.compile(r"RAM\s+(?P<used>\d+)/(?:\s*)?(?P<total>\d+)MB")
SWAP_RE = re.compile(r"SWAP\s+(?P<used>\d+)/(?:\s*)?(?P<total>\d+)MB")
CPU_RE = re.compile(r"CPU\s+\[(?P<body>[^\]]+)\]")
FREQ_RE = re.compile(r"(?P<name>EMC_FREQ|GR3D_FREQ)\s+(?P<pct>\d+)%")
TEMP_RE = re.compile(r"(?P<name>AO|CPU|GPU|PLL)@(?P<temp>\d+)C")
VDD_RE = re.compile(r"VDD_IN\s+(?P<mw>\d+)mW")
TS_RE = re.compile(r"^(?:(?P<ts>\d{2}-\d{2}-\d{4}\s+\d{2}:\d{2}:\d{2})\s+)?(?P<body>.*)$")


def parse_cpu_body(body: str) -> tuple[float | None, float | None, int, str]:
    active_utils: list[float] = []
    active_freqs: list[float] = []
    tokens = [tok.strip() for tok in body.split(",") if tok.strip()]
    for tok in tokens:
        if tok.lower() == "off":
            continue
        m = re.match(r"(?P<util>\d+)%@(?P<freq>\d+)", tok)
        if m:
            active_utils.append(float(m.group("util")))
            active_freqs.append(float(m.group("freq")))
    mean_util = float(sum(active_utils) / len(active_utils)) if active_utils else None
    mean_freq = float(sum(active_freqs) / len(active_freqs)) if active_freqs else None
    return mean_util, mean_freq, len(active_utils), body


def parse_line(line: str) -> dict[str, object] | None:
    raw = line.rstrip("\n")
    if not raw.strip() or "RAM" not in raw:
        return None
    m = TS_RE.match(raw)
    body = m.group("body") if m else raw
    row: dict[str, object] = {"raw_line": raw, "timestamp_text": m.group("ts") if m else ""}
    if "RAM" in body:
        mm = RAM_RE.search(body)
        if mm:
            row["ram_used_mb"] = float(mm.group("used"))
            row["ram_total_mb"] = float(mm.group("total"))
    if "SWAP" in body:
        mm = SWAP_RE.search(body)
        if mm:
            row["swap_used_mb"] = float(mm.group("used"))
            row["swap_total_mb"] = float(mm.group("total"))
    mcpu = CPU_RE.search(body)
    if mcpu:
        mean_util, mean_freq, active_count, raw_cpu = parse_cpu_body(mcpu.group("body"))
        row["cpu_active_cores"] = float(active_count)
        row["cpu_mean_util_pct"] = mean_util
        row["cpu_mean_freq_mhz"] = mean_freq
        row["cpu_raw"] = raw_cpu
    for mm in FREQ_RE.finditer(body):
        key = mm.group("name").lower()
        row[key] = float(mm.group("pct"))
    for mm in TEMP_RE.finditer(body):
        row[f"{mm.group('name').lower()}_temp_c"] = float(mm.group("temp"))
    mv = VDD_RE.search(body)
    if mv:
        row["vdd_in_mw"] = float(mv.group("mw"))
    return row if len(row) > 1 else None


def parse_file(path: Path) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    with path.open("r", encoding="utf-8", errors="ignore") as fh:
        for idx, line in enumerate(fh, start=1):
            row = parse_line(line)
            if row is None:
                continue
            row["source_file"] = str(path)
            row["line_no"] = idx
            rows.append(row)
    return pd.DataFrame(rows)


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()
    numeric_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    rows: list[dict[str, object]] = []
    for source_file, group in df.groupby("source_file", dropna=False):
        row: dict[str, object] = {"source_file": source_file, "samples": int(len(group))}
        for col in numeric_cols:
            if col in {"line_no"}:
                continue
            series = group[col].dropna()
            if series.empty:
                continue
            row[f"{col}_mean"] = float(series.mean())
            row[f"{col}_max"] = float(series.max())
        rows.append(row)
    total: dict[str, object] = {"source_file": "__all__", "samples": int(len(df))}
    for col in numeric_cols:
        if col in {"line_no"}:
            continue
        series = df[col].dropna()
        if series.empty:
            continue
        total[f"{col}_mean"] = float(series.mean())
        total[f"{col}_max"] = float(series.max())
    rows.append(total)
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Parse tegrastats logs into tabular summaries.")
    parser.add_argument("inputs", nargs="+", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--summary-out", type=Path, default=None)
    args = parser.parse_args()

    frames = [parse_file(path) for path in args.inputs]
    df = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False)
    if args.summary_out is not None:
        summarize(df).to_csv(args.summary_out, index=False)
    print(args.out)


if __name__ == "__main__":
    main()
