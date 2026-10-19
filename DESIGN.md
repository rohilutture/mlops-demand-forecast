# Design notes

## Forecasting setup and leakage
- **Task:** forecast hourly rentals **one day ahead**. So every feature must be known 24 hours before the target hour:
  - lags of 24h, 48h, and 168h, plus rolling means shifted by 24h
  - **no lag-1.** Using the previous hour's demand to predict this hour is the most common leakage in demand-forecasting projects. It inflates accuracy, and the model can't run in production.
- **Two tests enforce this.** One checks the lag arithmetic directly. The other perturbs y(t) and asserts that features(t) don't change.
- **Weather is the observed value,** standing in for the forecast you'd have at serving time. That is a stated limitation: real weather forecasts have error, so production accuracy would be somewhat worse.
- **Splits are strictly in time order.** Train | valid (early stopping) | test (last 28 days).

## Baseline and backtest
- **Seasonal naive** (same hour last week) is the baseline every forecast must beat. `skill_vs_naive = 1 − MAE_model / MAE_naive` is logged for every run.
- **Rolling-origin backtest.** Several cutoffs, each trained on everything before it and tested on the next two weeks. Its mean and standard deviation show whether one good test window was luck.

## Metrics
- **MAE** is reported in rentals per hour, which people can interpret.
- **WAPE** (total absolute error / total demand) drives monitoring. Demand doubles over the dataset, so absolute error grows even when the model is fine. In a first version, MAE-based monitoring flagged degradation almost every week for exactly that reason. WAPE is scale-free.
- **sMAPE** is logged for comparability. Plain MAPE is avoided because night-time demand near zero makes it explode.

## Validation gate
Explicit checks run before training and on new batches: schema, nulls, ranges, duplicate timestamps, and missing hours. The real UCI data has a few missing hours (system outages), so a small gap is a warning and a large one is an error. Bad data fails the DAG loudly instead of training a quietly broken model.

## Registry and promotion
- Every run registers a new model version. **That does not make it live.**
- A new version becomes `champion` only if it beats the current champion's MAE **on the same test window** by at least 2%. Otherwise it gets the `challenger` alias. "Latest wins" is how regressions ship.
- Each run stores its **reference feature sample** as an artifact. Monitoring always compares production with the data *that champion* was trained on.
- Promotion decisions are logged as JSON artifacts (who beat whom, by how much), giving an audit trail.

## Monitoring and the retrain rule
- **Drift (no labels needed, so it's immediate).** PSI with quantile bins from the reference, plus a KS test, on weather and lag features.
  - Calendar features are excluded: a one-week window always "drifts" from a year of hours and days by construction.
  - PSI thresholds are the industry rule of thumb: <0.1 stable, 0.1–0.25 moderate, >0.25 major.
- **Performance (needs actuals, so it lags a day).** Live WAPE vs. the champion's training WAPE.
- **Retrain if:**
  - live WAPE is more than 15% worse than training WAPE, **or**
  - at least 50% of monitored features show major drift.

  Drift alone is a weak trigger, since the model may cope fine. Performance is the ground truth, but it arrives late. The rule combines them, and every decision records its reasons.
- **Cooldown** (4 weeks): a burst of drift shouldn't trigger a retrain storm.
- **Evidently** renders the HTML drift report for humans. The decision logic is plain, tested code, so a library upgrade can't silently change when retraining fires.

## Serving
- Pydantic bounds on every input (e.g. humidity in [0, 1]) reject garbage with a 422 before it reaches the model.
- **Every prediction is logged** with its inputs, the model run ID, and a request ID. That's the raw material for monitoring. Logs go to S3 in production, one object per request batch, partitioned by date.
- `/reload` lets the pipeline swap in a newly promoted champion without restarting the container.
- **The model is baked into the image** by the CD pipeline. The image tag then identifies the exact model, rollback is redeploying an older tag, and the API doesn't depend on the tracking server at runtime.

## Infrastructure
- **App Runner rather than ECS + ALB.** One resource gives HTTPS, autoscaling, health checks, and auto-deploy on ECR push. That's the right amount of infrastructure for a single stateless API, and it's cheap to tear down. ECS/EKS would be the next step for multi-service or GPU workloads.
- **GitHub → AWS auth uses OIDC.** CI assumes a role scoped to this repo's `main` branch. There are no long-lived access keys anywhere.
- **S3:** versioned, encrypted, public access blocked, with a lifecycle rule that expires prediction logs after 90 days.
- **ECR:** scans on push and keeps only the last 10 images.

## Known limitations
- Observed weather stands in for weather forecasts.
- The Airflow compose service runs in `standalone` mode, which is for development. Production would use MWAA or a proper deployment.
- The MLflow server in compose uses SQLite. Production would use Postgres plus S3 artifacts, or a managed MLflow.
