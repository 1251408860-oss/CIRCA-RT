from __future__ import annotations

import numpy as np
import pandas as pd


def mean_table(df: pd.DataFrame, group_cols: list[str], metric_cols: list[str]) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=group_cols + metric_cols)
    return df.groupby(group_cols, as_index=False, dropna=False)[metric_cols].mean()


def bootstrap_ci_table(
    df: pd.DataFrame,
    group_cols: list[str],
    metric_cols: list[str],
    *,
    n_boot: int = 1000,
    seed: int = 7,
) -> pd.DataFrame:
    rows: list[dict[str, float | int | str]] = []
    rng = np.random.default_rng(seed)
    if df.empty:
        return pd.DataFrame(columns=group_cols + [f"{m}_{suffix}" for m in metric_cols for suffix in ("mean", "ci_low", "ci_high")])
    for group_key, group in df.groupby(group_cols, dropna=False):
        if not isinstance(group_key, tuple):
            group_key = (group_key,)
        row: dict[str, float | int | str] = {col: group_key[i] for i, col in enumerate(group_cols)}
        row["n_rows"] = int(len(group))
        for metric in metric_cols:
            values = group[metric].dropna().to_numpy(dtype=float)
            if values.size == 0:
                row[f"{metric}_mean"] = float("nan")
                row[f"{metric}_ci_low"] = float("nan")
                row[f"{metric}_ci_high"] = float("nan")
                continue
            mean = float(np.mean(values))
            if values.size == 1:
                ci_low = ci_high = mean
            else:
                boot = rng.choice(values, size=(n_boot, values.size), replace=True).mean(axis=1)
                ci_low = float(np.quantile(boot, 0.025))
                ci_high = float(np.quantile(boot, 0.975))
            row[f"{metric}_mean"] = mean
            row[f"{metric}_ci_low"] = ci_low
            row[f"{metric}_ci_high"] = ci_high
        rows.append(row)
    return pd.DataFrame(rows)
