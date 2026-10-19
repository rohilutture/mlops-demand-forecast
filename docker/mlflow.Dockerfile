# Same MLflow major version as the clients (registry aliases + MLflow 3 log_model API).
FROM python:3.11-slim
RUN pip install --no-cache-dir "mlflow>=3.1,<4"
EXPOSE 5000
