FROM python:3.11-slim
WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 curl && rm -rf /var/lib/apt/lists/*
COPY requirements-serve.txt .
RUN pip install --no-cache-dir -r requirements-serve.txt
COPY mlops ./mlops
# Optional: a champion model exported by scripts/export_champion.py is baked into the image,
# so the image tag pins the exact model (no runtime dependency on the MLflow server).
# (The real file in the COPY keeps the glob legal when model_export/ does not exist.)
COPY requirements-serve.txt model_expor[t] /app/model/
ENV PRED_LOG=/app/logs/predictions.jsonl
EXPOSE 8000
HEALTHCHECK CMD curl -fs localhost:8000/health || exit 1
CMD ["uvicorn", "mlops.serve:app", "--host", "0.0.0.0", "--port", "8000"]
