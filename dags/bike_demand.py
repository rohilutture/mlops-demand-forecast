"""Airflow DAGs: weekly training pipeline + daily monitoring that can trigger it.

  bike_demand_train   ingest -> validate -> train/evaluate/register -> reload API if promoted
  bike_demand_monitor read prediction log -> drift + live WAPE -> branch: trigger train | ok

Tasks call the same functions as the CLI, so the pipeline logic is tested in
pytest, not only inside Airflow. (Airflow 2.x TaskFlow API; compose pins 2.10.)
"""
from __future__ import annotations

import os
from datetime import datetime, timedelta

from airflow.decorators import dag, task
from airflow.operators.empty import EmptyOperator
from airflow.operators.trigger_dagrun import TriggerDagRunOperator

DATA = os.environ.get("BIKE_DATA", "/opt/project/data/hour.csv")
API = os.environ.get("BIKE_API", "http://api:8000")
DEFAULTS = {"retries": 2, "retry_delay": timedelta(minutes=5)}


@dag(schedule="@weekly", start_date=datetime(2026, 1, 1), catchup=False, default_args=DEFAULTS,
     tags=["bike-demand"])
def bike_demand_train():
    @task
    def ingest() -> str:
        from pathlib import Path
        from mlops.data import download
        p = Path(DATA)
        if not p.exists():
            download(p)  # in production: the day's batch lands in S3; this pulls it
        return str(p)

    @task
    def validate_data(path: str) -> str:
        from mlops.data import load
        from mlops.validate import validate
        res = validate(load(path))
        if not res.ok:
            raise ValueError(f"validation failed, not training: {res.errors}")
        for w in res.warnings:
            print("WARNING:", w)
        return path

    @task
    def train(path: str) -> dict:
        from mlops.data import load
        from mlops.train import run
        res = run(load(path), "bike-demand")
        return {"promoted": res["decision"]["promoted"], "test": res["test"],
                "decision": res["decision"]}

    @task
    def reload_api(result: dict):
        if not result["promoted"]:
            print("challenger not promoted; API keeps current champion:", result["decision"]["reason"])
            return
        import httpx
        httpx.post(f"{API}/reload", timeout=60).raise_for_status()

    reload_api(train(validate_data(ingest())))


@dag(schedule="@daily", start_date=datetime(2026, 1, 1), catchup=False, default_args=DEFAULTS,
     tags=["bike-demand"])
def bike_demand_monitor():
    @task.branch
    def check() -> str:
        import json
        import mlflow
        import pandas as pd
        from mlflow import MlflowClient
        from mlops import monitor
        from mlops.data import load, features
        from mlops.simulate import DRIFT_COLS
        from mlops.train import MODEL_NAME, metrics

        log = pd.read_json(os.environ.get("PRED_LOG", "/opt/project/logs/predictions.jsonl"), lines=True)
        log["ts"] = pd.to_datetime(log["ts"])
        recent = log[log.ts >= log.ts.max() - pd.Timedelta(days=7)]
        client = MlflowClient()
        mv = client.get_model_version_by_alias(MODEL_NAME, "champion")
        ref = pd.DataFrame(mlflow.artifacts.load_dict(f"runs:/{mv.run_id}/reference_features.json"))
        base = client.get_run(mv.run_id).data.metrics["test_wape"]
        # Join actuals (which arrive a day later) onto logged predictions.
        actual = features(load(DATA))[["ts", "cnt"]]
        joined = recent.merge(actual, on="ts", how="inner")
        live = metrics(joined.cnt, joined.prediction)["wape"] if len(joined) >= 24 else None
        dec = monitor.decide(monitor.feature_drift(ref, recent, DRIFT_COLS), base, live)
        monitor.evidently_html(ref[DRIFT_COLS], recent[DRIFT_COLS], "/opt/project/reports/drift.html")
        print(json.dumps(monitor.as_dict(dec), default=str))
        return "trigger_retrain" if dec.retrain else "healthy"

    trigger = TriggerDagRunOperator(task_id="trigger_retrain", trigger_dag_id="bike_demand_train")
    check() >> [trigger, EmptyOperator(task_id="healthy")]


bike_demand_train()
bike_demand_monitor()
