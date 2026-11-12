"""Offline threshold/budget simulation using imported Validation scores only."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    fbeta_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)

from backend.services.import_service import CHUNK_SIZE, ImportError, _import_paths


def _load_validation(import_id: str) -> pd.DataFrame:
    _, path = _import_paths(import_id)
    parts = []
    for chunk in pd.read_csv(path, chunksize=CHUNK_SIZE):
        cols = {str(c).strip().lower(): c for c in chunk.columns}
        split_col = cols.get("split", cols.get("_split"))
        if split_col is None:
            raise ImportError("Le fichier doit identifier sa partition via split ou _split.")
        validation = chunk.loc[
            chunk[split_col].astype("string").str.lower().isin(["validation", "val"])
        ]
        if validation.empty:
            continue
        def col(*names):
            source = next((cols[n] for n in names if n in cols), None)
            if source is None:
                raise ImportError(f"Colonne requise pour la simulation : {names[0]}.")
            return source
        parts.append(pd.DataFrame({
            "timestamp": pd.to_datetime(validation[col("timestamp", "trans_date_trans_time", "date")], errors="coerce"),
            "score": pd.to_numeric(validation[col("score", "fraud_score")], errors="coerce"),
            "y_true": pd.to_numeric(validation[col("is_fraud", "y_true", "label")], errors="coerce"),
            "amount": pd.to_numeric(validation[col("amt", "amount")], errors="coerce"),
        }))
    if not parts:
        raise ImportError("Aucune ligne de Validation (val/validation) n'est présente.")
    frame = pd.concat(parts, ignore_index=True)
    if frame.isna().any().any():
        raise ImportError("La Validation contient des labels, scores, dates ou montants manquants/invalides.")
    if not frame["y_true"].isin([0, 1]).all():
        raise ImportError("Les labels Validation doivent être binaires (0/1).")
    if frame["y_true"].nunique() < 2:
        raise ImportError("La Validation doit contenir les deux classes pour calculer les AUC.")
    if not frame["score"].between(0, 1).all():
        raise ImportError("Les scores doivent être compris entre 0 et 1.")
    return frame


def _evaluate(frame: pd.DataFrame, threshold: float, budget: int, cost: float, method: str) -> dict:
    ordered = frame.sort_values("timestamp", kind="stable").reset_index(drop=True).copy()
    ordered["day"] = ordered["timestamp"].dt.strftime("%Y-%m-%d")
    eligible = ordered["score"] >= threshold
    if method == "chronological":
        ranks = eligible.groupby(ordered["day"]).cumsum()
        selected = eligible & (ranks <= budget)
    else:
        ranks = ordered.assign(_eligible=eligible).groupby("day")["score"].rank(
            method="first", ascending=False
        )
        selected = eligible & (ranks <= budget)
    y = ordered["y_true"].to_numpy(dtype=np.int8)
    predictions = selected.to_numpy(dtype=np.int8)
    scores = ordered["score"].to_numpy(dtype=np.float64)
    amounts = ordered["amount"].to_numpy(dtype=np.float64)
    tp = int(((y == 1) & (predictions == 1)).sum())
    fp = int(((y == 0) & (predictions == 1)).sum())
    fn = int(((y == 1) & (predictions == 0)).sum())
    alert_count = int(predictions.sum())
    missed_fraud_amount = float(amounts[(y == 1) & (predictions == 0)].sum())
    alert_cost = float(alert_count * cost)
    daily_max = int(pd.Series(predictions).groupby(ordered["day"]).sum().max()) if len(predictions) else 0
    return {
        "rows": len(ordered),
        "fraud_count": int(y.sum()),
        "alerts": alert_count,
        "precision": float(precision_score(y, predictions, zero_division=0)),
        "recall": float(recall_score(y, predictions, zero_division=0)),
        "f2": float(fbeta_score(y, predictions, beta=2, zero_division=0)),
        "mcc": float(matthews_corrcoef(y, predictions)),
        "pr_auc": float(average_precision_score(y, scores)),
        "roc_auc": float(roc_auc_score(y, scores)),
        "confusion_matrix": confusion_matrix(y, predictions, labels=[0, 1]).tolist(),
        "missed_fraud_amount": missed_fraud_amount,
        "alert_cost": alert_cost,
        "total_cost": missed_fraud_amount + alert_cost,
        "cost_without_model": float(amounts[y == 1].sum()),
        "max_alerts_per_day": daily_max,
        "budget": budget,
    }


def run_simulation(
    import_id: str, *, threshold: float, daily_budget: int,
    cost_per_alert: float, method: str,
) -> dict:
    if not 0 <= threshold <= 1:
        raise ImportError("Le seuil doit être compris entre 0 et 1.")
    if daily_budget < 0 or daily_budget > 100_000:
        raise ImportError("Le budget doit être positif et raisonnable.")
    if not np.isfinite(cost_per_alert) or cost_per_alert < 0:
        raise ImportError("Le coût par alerte doit être un nombre positif.")
    if method not in {"topk_batch", "chronological"}:
        raise ImportError("Mode attendu : topk_batch ou chronological.")
    frame = _load_validation(import_id)
    result = _evaluate(frame, threshold, daily_budget, cost_per_alert, method)
    # A labeled Validation threshold sweep supports the precision/recall/cost trade-off chart.
    candidates = np.unique(np.quantile(frame["score"], np.linspace(0, 1, 31)))
    curve = []
    for candidate in candidates:
        point = _evaluate(frame, float(candidate), daily_budget, cost_per_alert, method)
        curve.append({
            "threshold": float(candidate), "precision": point["precision"],
            "recall": point["recall"], "alerts": point["alerts"],
            "total_cost": point["total_cost"],
        })
    return {
        **result,
        "method": method,
        "source_split": "Validation",
        "source_import_id": import_id,
        "threshold": threshold,
        "cost_per_alert": cost_per_alert,
        "threshold_curve": curve,
    }
