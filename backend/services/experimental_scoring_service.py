"""Read-only scoring of newly trained experiments; never changes alert quota."""
from __future__ import annotations

import uuid

import joblib
import pandas as pd

from backend.features.feature_engineering import FEATURE_COLUMNS
from backend.services.experiment_service import EXPERIMENT_DIR, get_experiment
from backend.services.scoring_service import get_scoring_service


def score_experimental(transaction: dict, job_id: str) -> dict:
    try:
        uuid.UUID(job_id)
    except (TypeError, ValueError) as exc:
        raise ValueError("Identifiant expérimental invalide.") from exc
    job = get_experiment(job_id)
    if job.get("status") != "completed":
        raise ValueError("L'expérience sélectionnée n'est pas terminée.")
    artifact = EXPERIMENT_DIR / job_id / "model.joblib"
    if not artifact.is_file():
        raise ValueError("Artefact de cette expérience introuvable.")
    # Run exactly the same causal feature engineering as the production model.
    baseline = get_scoring_service().score(transaction, mode="preview")
    model = joblib.load(artifact)  # Only server-generated trusted experiment artifacts.
    config_path = EXPERIMENT_DIR / job_id / "config.json"
    if not config_path.is_file():
        raise ValueError("Configuration des features de l'expérience introuvable.")
    import json
    used = json.loads(config_path.read_text(encoding="utf-8")).get("feature_columns")
    if not used or any(c not in FEATURE_COLUMNS for c in used):
        raise ValueError("Schéma de l'expérience invalide.")
    row = pd.DataFrame([baseline["features"]], columns=used)
    if hasattr(model, "predict_risk"):
        risk_score = float(model.predict_risk(row)[0])
        kind = "normal_train_anomaly_percentile"
    else:
        classes = list(model.classes_)
        risk_score = float(model.predict_proba(row)[0, classes.index(1)])
        kind = "uncalibrated_model_score"
    threshold = float(job["metrics"]["threshold"])
    return {
        **baseline,
        "fraud_score": risk_score,
        "threshold": threshold,
        "decision": "fraud" if risk_score >= threshold else "non_fraud",
        "model_name": f"{job['algorithm']} · {job_id[:8]}",
        "score_kind": kind,
        "baseline_fraud_score": baseline["fraud_score"],
        "mode": "preview",
    }
