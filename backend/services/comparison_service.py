"""Read-only registry for genuine model artifact metrics and experiments."""

from __future__ import annotations

import json
from pathlib import Path

from backend.services.model_service import CONFIG_PATH, PROJECT_ROOT
from backend.services.experiment_service import list_experiments

V1_CONFIG = PROJECT_ROOT / "models" / "reference" / "xgboost_v1_config.json"
METRICS = ("pr_auc", "roc_auc", "precision", "recall", "precision_at_k", "f2",
           "mcc", "total_cost", "threshold", "daily_alert_budget")


def model_comparison_history() -> dict:
    rows = []
    if V1_CONFIG.is_file():
        v1 = json.loads(V1_CONFIG.read_text(encoding="utf-8"))
        result = v1.get("test_results", {})
        rows.append({
            "model": "XGBoost V1",
            "family": "supervised",
            "split": "Test",
            "source": "models/reference/xgboost_v1_config.json",
            "protocol_note": "Résultats historiques de l'artefact V1; comparabilité de partition non vérifiée.",
            "pr_auc": result.get("PR_AUC"),
            "roc_auc": result.get("ROC_AUC"),
            "precision": result.get("Precision"),
            "recall": result.get("Recall"),
            "precision_at_k": None,
            "f2": result.get("F2"),
            "mcc": result.get("MCC"),
            "total_cost": result.get("Total_Cost"),
            "threshold": v1.get("threshold"),
            "daily_alert_budget": v1.get("daily_alert_budget"),
        })
    if CONFIG_PATH.is_file():
        v2 = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
        rows.append({
            "model": "Random Forest V2 (opérationnel)",
            "family": "supervised",
            "split": "Test",
            "source": "models/config.json",
            "protocol_note": "Résultats finaux V2 fournis avec le pipeline opérationnel.",
            "pr_auc": v2.get("pr_auc_test"),
            "roc_auc": v2.get("roc_auc_test"),
            "precision": v2.get("precision_test"),
            "recall": v2.get("recall_test"),
            "precision_at_k": None,
            "f2": v2.get("f2_test"),
            "mcc": v2.get("mcc_test"),
            "total_cost": None,
            "threshold": v2.get("threshold"),
            "daily_alert_budget": v2.get("daily_alert_budget"),
        })
    for experiment in list_experiments():
        metrics = experiment.get("metrics", {})
        if experiment.get("status") != "completed":
            continue
        rows.append({
            "model": f"Expérience {experiment['algorithm']} · {experiment['id'][:8]}",
            "family": "unsupervised" if experiment["algorithm"] in {"isolation_forest", "autoencoder"} else "supervised",
            "split": "Validation",
            "source": experiment.get("artifact"),
            "protocol_note": "Expérience locale; entraînée sur Train et évaluée sur Validation.",
            "pr_curve": metrics.get("pr_curve"), "confusion_matrix": metrics.get("confusion_matrix"),
            "pr_auc": metrics.get("pr_auc"), "roc_auc": metrics.get("roc_auc"),
            "precision": metrics.get("precision"), "recall": metrics.get("recall"),
            "precision_at_k": metrics.get("precision_at_k"), "f2": metrics.get("f2"), "mcc": metrics.get("mcc"),
            "total_cost": metrics.get("total_cost"), "threshold": metrics.get("threshold"),
            "daily_alert_budget": None,
        })
    return {
        "items": rows,
        "unavailable_metrics": [
            "Precision@k, matrices de confusion et courbes ne sont pas incluses dans les configs historiques.",
            "Isolation Forest et Autoencoder n'ont pas de résultats historiques fournis.",
        ],
    }
