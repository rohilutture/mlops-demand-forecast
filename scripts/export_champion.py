"""Download the current champion from the MLflow registry into ./model_export,
so the CD pipeline can bake it into the API image (image tag <-> model version).

    MLFLOW_TRACKING_URI=http://localhost:5000 python scripts/export_champion.py
"""
import json
import os
import shutil
from pathlib import Path

import mlflow
from mlflow import MlflowClient

name = os.environ.get("MODEL_NAME", "bike-demand")
out = Path("model_export")
shutil.rmtree(out, ignore_errors=True)
mv = MlflowClient().get_model_version_by_alias(name, "champion")
mlflow.artifacts.download_artifacts(f"models:/{name}@champion", dst_path=str(out))
(out / "model_version.json").write_text(json.dumps({"name": name, "version": mv.version,
                                                    "run_id": mv.run_id}))
print(f"exported {name} v{mv.version} -> {out}")
