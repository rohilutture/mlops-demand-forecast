"""Prediction API. Loads whatever model version holds the `champion` alias.

    MLFLOW_TRACKING_URI=http://localhost:5000 uvicorn mlops.serve:app --port 8000

Every prediction is logged (inputs, output, model version, request id) to a JSONL
prediction log (local path or s3://bucket/prefix via PRED_LOG). That log is what
monitoring reads: no log, no monitoring.
"""
from __future__ import annotations

import json
import os
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from .data import FEATURES

_BAKED = "/app/model" if Path("/app/model/MLmodel").exists() else None
MODEL_URI = os.environ.get("MODEL_URI") or _BAKED or \
    f"models:/{os.environ.get('MODEL_NAME', 'bike-demand')}@champion"
PRED_LOG = os.environ.get("PRED_LOG", "logs/predictions.jsonl")


class Row(BaseModel):
    ts: datetime
    hour: int = Field(ge=0, le=23)
    dow: int = Field(ge=0, le=6)
    month: int = Field(ge=1, le=12)
    hour_sin: float
    hour_cos: float
    holiday: int = Field(ge=0, le=1)
    workingday: int = Field(ge=0, le=1)
    temp: float = Field(ge=0, le=1)
    atemp: float = Field(ge=0, le=1)
    hum: float = Field(ge=0, le=1)
    windspeed: float = Field(ge=0, le=1)
    weathersit: int = Field(ge=1, le=4)
    lag_24: float = Field(ge=0)
    lag_48: float = Field(ge=0)
    lag_168: float = Field(ge=0)
    roll_mean_24: float = Field(ge=0)
    roll_mean_168: float = Field(ge=0)


class Batch(BaseModel):
    rows: list[Row] = Field(min_length=1, max_length=1000)


app = FastAPI(title="bike-demand forecaster")
_state = {"model": None, "version": None, "loaded_at": None}


def load_model(uri: str = MODEL_URI):
    import mlflow
    model = mlflow.pyfunc.load_model(uri)
    _state.update(model=model, version=model.metadata.run_id, loaded_at=time.time())
    return model


def _log(records: list[dict]):
    line = "".join(json.dumps(r, default=str) + "\n" for r in records)
    if PRED_LOG.startswith("s3://"):
        import boto3  # one object per request batch: simple, append-free
        bucket, _, prefix = PRED_LOG[5:].partition("/")
        key = f"{prefix.rstrip('/')}/{datetime.now(timezone.utc):%Y/%m/%d}/{uuid.uuid4().hex}.jsonl"
        boto3.client("s3").put_object(Bucket=bucket, Key=key, Body=line.encode())
    else:
        Path(PRED_LOG).parent.mkdir(parents=True, exist_ok=True)
        with open(PRED_LOG, "a", encoding="utf-8") as f:
            f.write(line)


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": _state["model"] is not None, "model_run": _state["version"]}


@app.post("/predict")
def predict(batch: Batch):
    if _state["model"] is None:
        try:
            load_model()
        except Exception as e:
            raise HTTPException(503, f"model unavailable: {e}")
    df = pd.DataFrame([r.model_dump() for r in batch.rows])
    preds = [max(0.0, float(p)) for p in _state["model"].predict(df[FEATURES].astype("float64"))]  # demand >= 0
    rid = uuid.uuid4().hex
    _log([{"request_id": rid, "ts": r.ts, "logged_at": datetime.now(timezone.utc),
           "model_run": _state["version"], "prediction": p, **r.model_dump(exclude={"ts"})}
          for r, p in zip(batch.rows, preds)])
    return {"request_id": rid, "model_run": _state["version"], "predictions": preds}


@app.post("/reload")
def reload():
    """Called by the pipeline after a promotion so the new champion goes live."""
    load_model()
    return {"model_run": _state["version"]}
