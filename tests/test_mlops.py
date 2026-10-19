import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")

from mlops import monitor
from mlops.data import FEATURES, TARGET, features, synthetic, time_split
from mlops.train import metrics
from mlops.validate import validate


@pytest.fixture(scope="module")
def raw():
    return synthetic(days=300, seed=1)


# ------------------------------------------------------------------ data
def test_lags_never_look_less_than_24h_back(raw):
    f = features(raw).set_index("ts")
    y = raw.set_index("ts")[TARGET]
    t = f.index[500]
    assert f.loc[t, "lag_24"] == y.loc[t - pd.Timedelta(hours=24)]
    assert f.loc[t, "lag_168"] == y.loc[t - pd.Timedelta(hours=168)]
    # rolling mean ends exactly 24h before the target hour
    window = y.loc[t - pd.Timedelta(hours=24 + 23): t - pd.Timedelta(hours=24)]
    assert np.isclose(f.loc[t, "roll_mean_24"], window.mean())


def test_changing_the_target_hour_does_not_change_its_features(raw):
    """Leakage test: perturb y at time t; features at t must be identical."""
    f1 = features(raw)
    raw2 = raw.copy()
    i = 3000
    raw2.loc[i, TARGET] += 999
    f2 = features(raw2)
    t = raw.loc[i, "ts"]
    row1, row2 = f1[f1.ts == t][FEATURES], f2[f2.ts == t][FEATURES]
    pd.testing.assert_frame_equal(row1.reset_index(drop=True), row2.reset_index(drop=True))


def test_time_split_is_ordered(raw):
    tr, va, te = time_split(features(raw))
    assert tr.ts.max() < va.ts.min() and va.ts.max() < te.ts.min()


# ------------------------------------------------------------------ validation
def test_validation_catches_bad_batches(raw):
    assert validate(raw).ok
    bad = raw.copy()
    bad.loc[5, "hum"] = 1.7
    bad.loc[9, "temp"] = None
    r = validate(bad)
    assert not r.ok and any("hum" in e for e in r.errors) and any("temp" in e for e in r.errors)
    assert not validate(pd.concat([raw, raw.head(3)])).ok                 # duplicate timestamps
    assert not validate(raw.drop(columns=["windspeed"])).ok
    gappy = raw.drop(index=range(100, 400))
    assert not validate(gappy).ok and validate(raw.drop(index=[100])).warnings


# ------------------------------------------------------------------ metrics + drift
def test_metrics():
    m = metrics([10, 0, 20], [12, 0, 18])
    assert m["mae"] == pytest.approx(4 / 3) and m["wape"] == pytest.approx(4 / 30)


def test_psi_stable_vs_shifted():
    rng = np.random.default_rng(0)
    ref = rng.normal(0, 1, 5000)
    assert monitor.psi(ref, rng.normal(0, 1, 5000)) < 0.05
    assert monitor.psi(ref, rng.normal(1, 1, 5000)) > 0.25
    cat = rng.choice([1, 2, 3], 5000, p=[.6, .3, .1])
    assert monitor.psi(cat, rng.choice([1, 2, 3], 5000, p=[.2, .3, .5])) > 0.25


def test_retrain_rule():
    rng = np.random.default_rng(0)
    ref = pd.DataFrame({"a": rng.normal(0, 1, 2000), "b": rng.normal(0, 1, 2000)})
    same = pd.DataFrame({"a": rng.normal(0, 1, 500), "b": rng.normal(0, 1, 500)})
    moved = pd.DataFrame({"a": rng.normal(2, 1, 500), "b": rng.normal(2, 1, 500)})
    ok = monitor.decide(monitor.feature_drift(ref, same, ["a", "b"]), 0.10, 0.105)
    assert not ok.retrain
    perf = monitor.decide(monitor.feature_drift(ref, same, ["a", "b"]), 0.10, 0.13)
    assert perf.retrain and "WAPE" in perf.reasons[0]
    drift = monitor.decide(monitor.feature_drift(ref, moved, ["a", "b"]), 0.10, None)
    assert drift.retrain and drift.top_drifted[0] in {"a", "b"}


# ------------------------------------------------------------------ MLflow: train, promote, serve, simulate
@pytest.fixture
def mlflow_tmp(tmp_path, monkeypatch):
    import mlflow
    uri = f"sqlite:///{tmp_path / 'mlflow.db'}"
    monkeypatch.setenv("MLFLOW_TRACKING_URI", uri)
    mlflow.set_tracking_uri(uri)
    monkeypatch.chdir(tmp_path)
    return tmp_path


FAST = {"n_estimators": 300, "learning_rate": 0.08, "max_depth": 6, "subsample": 0.8,
        "colsample_bytree": 0.8, "objective": "reg:squarederror", "early_stopping_rounds": 30,
        "random_state": 0}


def test_train_beats_naive_and_promotion_rule(raw, mlflow_tmp):
    from mlflow import MlflowClient
    from mlops.train import MODEL_NAME, run
    r1 = run(raw, "t", do_backtest=False, params=FAST)
    assert r1["test"]["mae"] < r1["naive"]["mae"]           # skill over seasonal naive
    assert r1["decision"]["promoted"] and r1["decision"]["reason"] == "no champion yet"
    # Same data + same seed -> same model: not better by the 2% margin -> stays challenger.
    r2 = run(raw, "t", do_backtest=False, params=FAST)
    assert not r2["decision"]["promoted"]
    c = MlflowClient()
    assert c.get_model_version_by_alias(MODEL_NAME, "champion").version == r1["decision"]["new_version"]
    assert c.get_model_version_by_alias(MODEL_NAME, "challenger").version == r2["decision"]["new_version"]


def test_bad_data_blocks_training(raw, mlflow_tmp):
    from mlops.train import run
    bad = raw.copy()
    bad.loc[0, "cnt"] = -5
    with pytest.raises(ValueError, match="validation failed"):
        run(bad, "t", do_backtest=False, params=FAST)


def test_backtest_folds(raw):
    from mlops.train import backtest
    folds = backtest(features(raw), folds=3, horizon_days=7, params=FAST)
    assert len(folds) == 3 and all(f["mae"] > 0 for f in folds)


def test_api_predict_and_log(raw, mlflow_tmp, monkeypatch):
    from fastapi.testclient import TestClient
    from mlops.train import run
    run(raw, "t", do_backtest=False, params=FAST)
    import mlops.serve as serve
    monkeypatch.setattr(serve, "PRED_LOG", str(mlflow_tmp / "logs" / "p.jsonl"))
    serve._state.update(model=None, version=None)
    monkeypatch.setattr(serve, "MODEL_URI", "models:/bike-demand@champion")
    c = TestClient(serve.app)
    row = features(raw).iloc[-1]
    payload = {"rows": [{"ts": str(row.ts), **{k: float(row[k]) if isinstance(row[k], float) else int(row[k])
                                               for k in FEATURES}}]}
    r = c.post("/predict", json=payload)
    assert r.status_code == 200 and r.json()["predictions"][0] >= 0
    logged = [json.loads(l) for l in open(mlflow_tmp / "logs" / "p.jsonl")]
    assert logged[0]["prediction"] == r.json()["predictions"][0] and "lag_24" in logged[0]
    payload["rows"][0]["hum"] = 3.0
    assert c.post("/predict", json=payload).status_code == 422  # schema guards bad inputs


def test_simulation_loop_runs(mlflow_tmp):
    import mlops.simulate as sim
    import mlops.train as tr
    old = tr.PARAMS
    tr.PARAMS = FAST
    try:
        raw = synthetic(days=330, seed=2, growth=1.5)
        tl = sim.simulate(raw, "2011-10-01", weeks=8, cooldown_weeks=3, experiment="sim",
                          out_dir=mlflow_tmp / "reports")
    finally:
        tr.PARAMS = old
    assert len(tl) == 8 and (mlflow_tmp / "reports" / "simulation.png").exists()
    assert {"live_wape", "frozen_wape", "drift_share", "retrained"} <= set(tl.columns)
    # cooldown respected: no two retrains closer than 3 weeks
    idx = list(np.where(tl.retrained)[0])
    assert all(b - a >= 3 for a, b in zip(idx, idx[1:]))
