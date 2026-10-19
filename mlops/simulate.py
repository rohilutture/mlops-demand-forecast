"""Replay a year of "production" week by week to show the monitoring loop working.

    python -m mlops.simulate --data data/hour.csv --start 2012-01-01

  * Train the first champion on data before --start.
  * Each week: the champion forecasts the week; actuals arrive; compute live MAE
    and drift vs. the champion's training reference; apply the retrain rule.
  * On retrain: train on everything seen so far; the promotion rule decides whether
    the new version replaces the champion.
  * Counterfactual: the same weeks scored by the frozen first model, so the
    chart shows what the loop is worth.
A retrain cooldown stops a burst of drift from triggering a retrain every week.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import mlflow
import numpy as np
import pandas as pd
from mlflow import MlflowClient

from . import monitor
from .data import FEATURES, LAGS, ROLL, TARGET, WEATHER, features, load, synthetic
from .train import MODEL_NAME, metrics, run

DRIFT_COLS = [c for c in WEATHER if c != "weathersit"] + [f"lag_{l}" for l in LAGS] + \
             [f"roll_mean_{w}" for w in ROLL]  # calendar features drift by construction: excluded


def champion_context(client: MlflowClient):
    mv = client.get_model_version_by_alias(MODEL_NAME, "champion")
    r = client.get_run(mv.run_id)
    ref = pd.DataFrame(mlflow.artifacts.load_dict(f"runs:/{mv.run_id}/reference_features.json"))
    model = mlflow.pyfunc.load_model(f"models:/{MODEL_NAME}@champion")
    return mv.version, model, r.data.metrics["test_wape"], ref


def simulate(raw: pd.DataFrame, start: str, weeks: int | None = None, cooldown_weeks: int = 4,
             experiment: str = "bike-demand-sim", out_dir: Path = Path("reports"),
             tolerance: float = 0.15, drift_share_alert: float = 0.5) -> pd.DataFrame:
    start = pd.Timestamp(start)
    client = MlflowClient()
    run(raw[raw.ts < start], experiment, do_backtest=False)
    version, model, base_wape, ref = champion_context(client)
    frozen_model = model
    feat = features(raw)
    rows, last_retrain = [], -10**9
    week_starts = pd.date_range(start, feat.ts.max(), freq="7D")
    if weeks:
        week_starts = week_starts[:weeks]
    for w, ws in enumerate(week_starts):
        week = feat[(feat.ts >= ws) & (feat.ts < ws + pd.Timedelta(days=7))]
        if len(week) < 24:
            continue
        pred = np.clip(model.predict(week[FEATURES]), 0, None)
        live = metrics(week[TARGET], pred)
        frozen = metrics(week[TARGET], np.clip(frozen_model.predict(week[FEATURES]), 0, None))
        drift = monitor.feature_drift(ref, week, DRIFT_COLS)
        dec = monitor.decide(drift, base_wape, live["wape"], tolerance, drift_share_alert)
        row = {"week": ws.date(), "champion_version": version,
               "live_mae": round(live["mae"], 2), "frozen_mae": round(frozen["mae"], 2),
               "live_wape": round(live["wape"], 4), "frozen_wape": round(frozen["wape"], 4),
               "baseline_wape": round(base_wape, 4),
               "drift_share": drift.drift_share, "top_drifted": ",".join(dec.top_drifted[:3]),
               "retrain_signal": dec.retrain, "retrained": False, "promoted": False,
               "reasons": " | ".join(dec.reasons)}
        if dec.retrain and w - last_retrain >= cooldown_weeks:
            seen = raw[raw.ts < ws + pd.Timedelta(days=7)]
            res = run(seen, experiment, do_backtest=False)
            last_retrain = w
            row["retrained"], row["promoted"] = True, res["decision"]["promoted"]
            if res["decision"]["promoted"]:
                version, model, base_wape, ref = champion_context(client)
        rows.append(row)
        print(json.dumps(row, default=str), flush=True)
    tl = pd.DataFrame(rows)
    out_dir.mkdir(parents=True, exist_ok=True)
    tl.to_csv(out_dir / "simulation_timeline.csv", index=False)
    plot(tl, out_dir / "simulation.png")
    return tl


def plot(tl: pd.DataFrame, path: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(10, 4))
    x = pd.to_datetime(tl.week)
    ax.plot(x, tl.frozen_wape, color="#9aa5b1", label="frozen first model")
    ax.plot(x, tl.live_wape, color="#2a6fdb", label="with monitoring + retraining")
    for xi in x[tl.promoted]:
        ax.axvline(xi, color="#1a9e5f", alpha=0.5, lw=1)
    for xi in x[tl.retrained & ~tl.promoted]:
        ax.axvline(xi, color="#d08a00", alpha=0.5, lw=1, ls="--")
    ax.set_ylabel("Weekly WAPE")
    ax.set_title("Production replay: green = retrain promoted, orange dashed = retrain rejected")
    ax.legend(loc="upper left")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--data")
    p.add_argument("--synthetic", action="store_true")
    p.add_argument("--start", default="2012-01-01")
    p.add_argument("--weeks", type=int)
    p.add_argument("--cooldown-weeks", type=int, default=4)
    a = p.parse_args(argv)
    raw = synthetic() if a.synthetic else load(Path(a.data))
    tl = simulate(raw, a.start, a.weeks, a.cooldown_weeks)
    print(f"\nmean weekly WAPE: loop {tl.live_wape.mean():.3f} vs frozen {tl.frozen_wape.mean():.3f}; "
          f"MAE: loop {tl.live_mae.mean():.1f} vs frozen {tl.frozen_mae.mean():.1f}; "
          f"retrains {int(tl.retrained.sum())}, promotions {int(tl.promoted.sum())}")


if __name__ == "__main__":
    main()
