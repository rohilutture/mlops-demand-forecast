"""Production monitoring: feature drift, prediction drift, and live accuracy -> retrain decision.

Two signals, because each fails alone:
  * Drift (no labels needed, available immediately): PSI per feature and a KS
    test, comparing recent serving inputs with the training reference sample.
    Drift alone can be harmless (a feature moved but the model copes).
  * Performance (needs actuals, which for demand arrive the next day): WAPE on
    recent predictions vs. WAPE at training time. WAPE, not MAE: absolute error
    grows with volume, so in a growing business MAE "degrades" even when the model
    is fine. This is the ground truth, but it lags.
Retrain when performance degraded beyond tolerance, OR when drift is broad AND
severe (many features with PSI > 0.25). Each decision is written with its
reasons so an on-call engineer can read why it fired.

Evidently is used for the human-readable HTML drift report; the decision logic
is plain, tested code so it can't change under a library upgrade.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field

import numpy as np
import pandas as pd
from scipy import stats

PSI_WARN, PSI_ALERT = 0.10, 0.25


def psi(ref: np.ndarray, cur: np.ndarray, bins: int = 10) -> float:
    """Population Stability Index with quantile bins taken from the reference.
    < 0.1 stable · 0.1-0.25 moderate shift · > 0.25 major shift (industry rule of thumb)."""
    ref, cur = np.asarray(ref, float), np.asarray(cur, float)
    edges = np.unique(np.quantile(ref, np.linspace(0, 1, bins + 1)))
    if len(edges) < 3:  # (near-)constant or low-cardinality feature: bin by value
        vals = np.unique(np.r_[ref, cur])
        r = np.array([(ref == v).mean() for v in vals])
        c = np.array([(cur == v).mean() for v in vals])
    else:
        edges[0], edges[-1] = -np.inf, np.inf
        r = np.histogram(ref, edges)[0] / len(ref)
        c = np.histogram(cur, edges)[0] / len(cur)
    r, c = np.clip(r, 1e-4, None), np.clip(c, 1e-4, None)
    return float(np.sum((c - r) * np.log(c / r)))


@dataclass
class DriftReport:
    per_feature: dict = field(default_factory=dict)  # name -> {psi, ks_p, status}
    drift_share: float = 0.0                           # share of features with PSI > alert
    prediction_psi: float | None = None


def feature_drift(reference: pd.DataFrame, current: pd.DataFrame, cols: list[str],
                  ref_pred=None, cur_pred=None) -> DriftReport:
    rep = DriftReport()
    for c in cols:
        p = psi(reference[c], current[c])
        ks_p = float(stats.ks_2samp(reference[c], current[c]).pvalue)
        rep.per_feature[c] = {"psi": round(p, 4), "ks_p": round(ks_p, 5),
                              "status": "alert" if p > PSI_ALERT else "warn" if p > PSI_WARN else "ok"}
    rep.drift_share = round(np.mean([v["status"] == "alert" for v in rep.per_feature.values()]), 4)
    if ref_pred is not None and cur_pred is not None:
        rep.prediction_psi = round(psi(ref_pred, cur_pred), 4)
    return rep


@dataclass
class RetrainDecision:
    retrain: bool
    reasons: list[str]
    live_wape: float | None
    baseline_wape: float
    drift_share: float
    top_drifted: list[str]


def decide(drift: DriftReport, baseline_wape: float, live_wape: float | None,
           tolerance: float = 0.15, drift_share_alert: float = 0.3) -> RetrainDecision:
    reasons = []
    if live_wape is not None and live_wape > baseline_wape * (1 + tolerance):
        reasons.append(f"live WAPE {live_wape:.3f} > {1 + tolerance:.2f} x training WAPE {baseline_wape:.3f}")
    if drift.drift_share >= drift_share_alert:
        reasons.append(f"{drift.drift_share:.0%} of features have PSI > {PSI_ALERT}")
    top = sorted(drift.per_feature, key=lambda c: -drift.per_feature[c]["psi"])[:5]
    return RetrainDecision(bool(reasons), reasons, live_wape, baseline_wape, drift.drift_share, top)


def evidently_html(reference: pd.DataFrame, current: pd.DataFrame, path: str) -> bool:
    """Optional pretty report (Evidently >= 0.7 API). Returns False if unavailable."""
    try:
        from evidently import Report
        from evidently.presets import DataDriftPreset
    except ImportError:
        return False
    Report([DataDriftPreset()]).run(current_data=current, reference_data=reference).save_html(path)
    return True


def as_dict(x):
    return asdict(x)
