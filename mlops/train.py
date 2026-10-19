"""Train -> evaluate -> log to MLflow -> champion/challenger promotion.

    python -m mlops.train --data data/hour.csv            (MLFLOW_TRACKING_URI for a server)

Steps:
  1. validate raw data (fails fast)
  2. features + time split (train | valid | test)
  3. XGBoost with early stopping on valid; seasonal-naive baseline (lag 168h)
  4. rolling-origin backtest for stability (one number can be lucky)
  5. log params, metrics, feature importance, and the model with its signature
  6. register a new version; promote alias "champion" ONLY if it beats the
     current champion on the SAME test window by a margin (otherwise it stays
     "challenger"). Promotion is a decision with a rule, not "latest wins".
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import mlflow
import numpy as np
import pandas as pd
import xgboost as xgb
from mlflow import MlflowClient
from mlflow.models import infer_signature

from .data import FEATURES, TARGET, features, load, synthetic, time_split
from .validate import validate

MODEL_NAME = os.environ.get("MODEL_NAME", "bike-demand")
PARAMS = {"n_estimators": 2000, "learning_rate": 0.03, "max_depth": 7, "subsample": 0.8,
          "colsample_bytree": 0.8, "min_child_weight": 5, "reg_lambda": 1.0,
          "objective": "reg:squarederror", "early_stopping_rounds": 100, "random_state": 0}


def metrics(y, p) -> dict:
    y, p = np.asarray(y, float), np.asarray(p, float)
    return {"mae": float(np.mean(np.abs(y - p))),
            "rmse": float(np.sqrt(np.mean((y - p) ** 2))),
            # sMAPE is bounded and copes with near-zero night-time demand, unlike MAPE
            "smape": float(np.mean(2 * np.abs(y - p) / (np.abs(y) + np.abs(p) + 1e-9))),
            # WAPE = total abs error / total demand. Scale-free, so it stays comparable as
            # volume grows; monitoring uses it for that reason.
            "wape": float(np.sum(np.abs(y - p)) / max(np.sum(np.abs(y)), 1e-9))}


def fit(train: pd.DataFrame, valid: pd.DataFrame, params=PARAMS) -> xgb.XGBRegressor:
    m = xgb.XGBRegressor(**params)
    m.fit(train[FEATURES], train[TARGET], eval_set=[(valid[FEATURES], valid[TARGET])], verbose=False)
    return m


def backtest(feat: pd.DataFrame, folds: int = 4, horizon_days: int = 14, params=PARAMS) -> list[dict]:
    """Expanding-window, rolling-origin evaluation: train on everything before each
    cutoff, test on the next `horizon_days`. Reports spread across folds."""
    out = []
    end = feat.ts.max()
    for k in range(folds, 0, -1):
        cut = end - pd.Timedelta(days=horizon_days * k)
        tr = feat[feat.ts <= cut - pd.Timedelta(days=horizon_days)]
        va = feat[(feat.ts > cut - pd.Timedelta(days=horizon_days)) & (feat.ts <= cut)]
        te = feat[(feat.ts > cut) & (feat.ts <= cut + pd.Timedelta(days=horizon_days))]
        if len(tr) < 24 * 30 or len(te) == 0:
            continue
        m = fit(tr, va, {**params, "n_estimators": 600})
        out.append({"cutoff": str(cut.date()), **metrics(te[TARGET], m.predict(te[FEATURES]))})
    return out


def champion_mae(client: MlflowClient, test: pd.DataFrame) -> float | None:
    """Re-score the current champion on today's test window (apples to apples)."""
    try:
        mv = client.get_model_version_by_alias(MODEL_NAME, "champion")
    except Exception:
        return None
    model = mlflow.pyfunc.load_model(f"models:/{MODEL_NAME}@champion")
    return metrics(test[TARGET], model.predict(test[FEATURES]))["mae"], mv.version


def run(raw: pd.DataFrame, experiment: str = "bike-demand", promote_margin: float = 0.02,
        do_backtest: bool = True, params=PARAMS) -> dict:
    v = validate(raw)
    if not v.ok:
        raise ValueError(f"data validation failed: {v.errors}")
    feat = features(raw)
    train, valid, test = time_split(feat)
    mlflow.set_experiment(experiment)
    client = MlflowClient()
    with mlflow.start_run() as run:
        mlflow.log_params({k: v for k, v in params.items()})
        mlflow.log_params({"n_train": len(train), "n_valid": len(valid), "n_test": len(test),
                           "train_end": str(train.ts.max()), "test_start": str(test.ts.min()),
                           "features": ",".join(FEATURES)})
        model = fit(train, valid, params)
        mlflow.log_param("best_iteration", int(model.best_iteration))
        m_model = metrics(test[TARGET], model.predict(test[FEATURES]))
        m_naive = metrics(test[TARGET], test["lag_168"])
        mlflow.log_metrics({f"test_{k}": v for k, v in m_model.items()})
        mlflow.log_metrics({f"naive_{k}": v for k, v in m_naive.items()})
        mlflow.log_metric("skill_vs_naive", 1 - m_model["mae"] / m_naive["mae"])
        bt = backtest(feat, params=params) if do_backtest else []
        if bt:
            maes = [b["mae"] for b in bt]
            mlflow.log_metrics({"backtest_mae_mean": float(np.mean(maes)),
                                "backtest_mae_std": float(np.std(maes))})
            mlflow.log_dict({"folds": bt}, "backtest.json")
        imp = dict(sorted(zip(FEATURES, model.feature_importances_.round(4).tolist()),
                          key=lambda kv: -kv[1]))
        mlflow.log_dict(imp, "feature_importance.json")
        # Reference sample for drift monitoring, stored WITH the model it describes.
        mlflow.log_dict(train[FEATURES].sample(min(5000, len(train)), random_state=0)
                        .to_dict(orient="list"), "reference_features.json")
        sig = infer_signature(test[FEATURES].head(50), model.predict(test[FEATURES].head(50)))
        info = mlflow.xgboost.log_model(model, name="model", signature=sig,
                                        input_example=test[FEATURES].head(3),
                                        registered_model_name=MODEL_NAME)
        version = info.registered_model_version

        champ = champion_mae(client, test)
        decision = {"new_version": version, "new_mae": m_model["mae"]}
        if champ is None:
            promote, decision["reason"] = True, "no champion yet"
        else:
            champ_mae, champ_ver = champ
            decision.update(champion_version=champ_ver, champion_mae=champ_mae)
            promote = m_model["mae"] < champ_mae * (1 - promote_margin)
            decision["reason"] = (f"beats champion by >{promote_margin:.0%}" if promote
                                  else f"not better than champion by {promote_margin:.0%}")
        client.set_registered_model_alias(MODEL_NAME, "champion" if promote else "challenger", version)
        decision["promoted"] = promote
        mlflow.log_dict(decision, "promotion_decision.json")
        mlflow.set_tag("promoted", str(promote))
        return {"run_id": run.info.run_id, "test": m_model, "naive": m_naive, "backtest": bt,
                "decision": decision}


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--data", help="UCI hour.csv (omit with --synthetic)")
    p.add_argument("--synthetic", action="store_true")
    p.add_argument("--until", help="train only on data up to this date (simulate the past)")
    p.add_argument("--experiment", default="bike-demand")
    p.add_argument("--no-backtest", action="store_true")
    a = p.parse_args(argv)
    raw = synthetic() if a.synthetic else load(Path(a.data))
    if a.until:
        raw = raw[raw.ts <= pd.Timestamp(a.until)]
    res = run(raw, a.experiment, do_backtest=not a.no_backtest)
    print(json.dumps({k: res[k] for k in ("test", "naive", "decision")}, indent=2, default=str))


if __name__ == "__main__":
    main()
