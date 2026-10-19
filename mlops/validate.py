"""Data validation gate: runs before training and on every new batch.

Bad data should stop the pipeline loudly, not train a quietly broken model.
Checks are explicit and cheap (the same idea as Great Expectations / pandera,
without the dependency): schema, nulls, value ranges, duplicate and missing
timestamps, and a minimum row count.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

SCHEMA = {"ts": "datetime64", "holiday": "int", "workingday": "int", "temp": "float",
          "atemp": "float", "hum": "float", "windspeed": "float", "weathersit": "int", "cnt": "int"}
RANGES = {"temp": (0, 1), "atemp": (0, 1), "hum": (0, 1), "windspeed": (0, 1),
          "weathersit": (1, 4), "holiday": (0, 1), "workingday": (0, 1), "cnt": (0, 2000)}


@dataclass
class ValidationResult:
    ok: bool
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


def validate(df: pd.DataFrame, min_rows: int = 24 * 7, max_missing_hours_frac: float = 0.02,
             require_target: bool = True) -> ValidationResult:
    errs, warns = [], []
    cols = [c for c in SCHEMA if require_target or c != "cnt"]
    missing = [c for c in cols if c not in df.columns]
    if missing:
        return ValidationResult(False, [f"missing columns: {missing}"])
    if len(df) < min_rows:
        errs.append(f"only {len(df)} rows (< {min_rows})")
    if not pd.api.types.is_datetime64_any_dtype(df["ts"]):
        errs.append("ts is not a datetime column")
    for c in cols:
        n = int(df[c].isna().sum())
        if n:
            errs.append(f"{c}: {n} nulls")
    for c, (lo, hi) in RANGES.items():
        if c in df and c in cols:
            bad = int(((df[c] < lo) | (df[c] > hi)).sum())
            if bad:
                errs.append(f"{c}: {bad} values outside [{lo}, {hi}]")
    dups = int(df["ts"].duplicated().sum())
    if dups:
        errs.append(f"{dups} duplicate timestamps")
    if len(df) > 1 and pd.api.types.is_datetime64_any_dtype(df["ts"]):
        expected = int((df.ts.max() - df.ts.min()) / pd.Timedelta(hours=1)) + 1
        gap = 1 - df.ts.nunique() / expected
        # The real UCI data has a few missing hours (system outages): tolerate a little.
        (errs if gap > max_missing_hours_frac else warns if gap > 0 else []).append(
            f"{gap:.2%} of hours missing")
    return ValidationResult(not errs, errs, warns)
