"""Data: UCI Bike Sharing (hourly rentals, Washington D.C., 2011-2012) + features.

Why this dataset for an MLOps project: demand roughly doubles from 2011 to 2012,
so a model trained on early data drifts in production. That gives the monitoring
and retraining loop something real to catch, not a staged example.

Forecasting setup: predict hourly rentals one day ahead. Every feature must be
known 24h before the target hour, so lags start at 24h (no lag-1 leakage, which
is the most common bug in demand-forecasting portfolios).
"""
from __future__ import annotations

import io
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

UCI_URL = "https://archive.ics.uci.edu/static/public/275/bike+sharing+dataset.zip"
TARGET = "cnt"
LAGS = [24, 48, 168]          # same hour yesterday, 2 days ago, last week
ROLL = [24, 168]              # trailing means, shifted by 24h
WEATHER = ["temp", "atemp", "hum", "windspeed", "weathersit"]


def download(dest: Path = Path("data/hour.csv")) -> Path:
    import httpx
    dest.parent.mkdir(parents=True, exist_ok=True)
    r = httpx.get(UCI_URL, timeout=60, follow_redirects=True)
    r.raise_for_status()
    with zipfile.ZipFile(io.BytesIO(r.content)) as z:
        dest.write_bytes(z.read("hour.csv"))
    return dest


def load(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["ts"] = pd.to_datetime(df["dteday"]) + pd.to_timedelta(df["hr"], unit="h")
    return df[["ts", "holiday", "workingday", *WEATHER, TARGET]].sort_values("ts").reset_index(drop=True)


def synthetic(days: int = 540, seed: int = 0, growth: float = 0.8) -> pd.DataFrame:
    """Same schema as UCI hourly data: daily + weekly seasonality, weather effects,
    and a growth trend (so later data drifts from earlier data)."""
    rng = np.random.default_rng(seed)
    ts = pd.date_range("2011-01-01", periods=days * 24, freq="h")
    hr, dow, doy = ts.hour.to_numpy(), ts.dayofweek.to_numpy(), ts.dayofyear.to_numpy()
    working = (dow < 5).astype(int)
    temp = np.clip(0.5 + 0.3 * np.sin(2 * np.pi * (doy - 100) / 365) + rng.normal(0, 0.08, len(ts)), 0, 1)
    hum = np.clip(rng.normal(0.6, 0.15, len(ts)), 0, 1)
    wind = np.clip(rng.gamma(2, 0.1, len(ts)), 0, 0.9)
    weather = rng.choice([1, 2, 3], len(ts), p=[0.65, 0.28, 0.07])
    commute = working * (np.exp(-((hr - 8) ** 2) / 2) * 1.6 + np.exp(-((hr - 17.5) ** 2) / 3) * 1.8)
    leisure = (1 - working) * np.exp(-((hr - 14) ** 2) / 12) * 1.2
    base = 30 + 220 * (commute + leisure + 0.25 * np.exp(-((hr - 13) ** 2) / 30))
    trend = 1 + growth * np.arange(len(ts)) / len(ts)
    mu = base * (0.5 + temp) * (1.25 - 0.25 * weather) * (1 - 0.3 * hum) * trend
    cnt = rng.poisson(np.clip(mu, 1, None))
    return pd.DataFrame({"ts": ts, "holiday": 0, "workingday": working, "temp": temp.round(3),
                         "atemp": (temp * 0.95).round(3), "hum": hum.round(3),
                         "windspeed": wind.round(3), "weathersit": weather, TARGET: cnt})


def features(df: pd.DataFrame) -> pd.DataFrame:
    """Calendar + weather + target lags. Lags/rolls are shifted >= 24h so every
    feature is available at prediction time (one day ahead)."""
    df = df.sort_values("ts").set_index("ts").asfreq("h")   # make missing hours explicit
    out = pd.DataFrame(index=df.index)
    out["hour"] = df.index.hour
    out["dow"] = df.index.dayofweek
    out["month"] = df.index.month
    out["hour_sin"], out["hour_cos"] = np.sin(2 * np.pi * out.hour / 24), np.cos(2 * np.pi * out.hour / 24)
    out["holiday"] = df["holiday"].ffill()
    out["workingday"] = df["workingday"].ffill()
    for c in WEATHER:  # weather = forecast at serving time; observed values stand in here
        out[c] = df[c].ffill()
    y = df[TARGET]
    for lag in LAGS:
        out[f"lag_{lag}"] = y.shift(lag)
    for w in ROLL:
        out[f"roll_mean_{w}"] = y.shift(24).rolling(w, min_periods=w // 2).mean()
    out[TARGET] = y
    out = out.dropna().reset_index()
    # One dtype for every model input. Calendar fields come out as int32 here but
    # arrive as int64 from JSON at serving time, and MLflow's signature check rejects
    # that mismatch. float64 everywhere is consistent and tolerates missing values.
    out[FEATURES] = out[FEATURES].astype("float64")
    return out


FEATURES = ["hour", "dow", "month", "hour_sin", "hour_cos", "holiday", "workingday", *WEATHER,
            *[f"lag_{l}" for l in LAGS], *[f"roll_mean_{w}" for w in ROLL]]


def time_split(df: pd.DataFrame, valid_days: int = 28, test_days: int = 28):
    """Train | valid | test, strictly in time order. No shuffling, ever."""
    end = df.ts.max()
    test_start = end - pd.Timedelta(days=test_days)
    valid_start = test_start - pd.Timedelta(days=valid_days)
    return (df[df.ts <= valid_start], df[(df.ts > valid_start) & (df.ts <= test_start)],
            df[df.ts > test_start])
