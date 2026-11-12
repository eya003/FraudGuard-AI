"""Validated, read-only inference on already engineered, source-supplied features.

No historical features are reconstructed from a CSV row; the caller must
provide all 15 features computed causally on the source dataset.
"""
from __future__ import annotations
import math
from typing import Any

import pandas as pd

from backend.features.feature_engineering import FEATURE_COLUMNS


def normalize_features(values: dict[str, Any]) -> dict:
    if set(values) != set(FEATURE_COLUMNS):
        missing = sorted(set(FEATURE_COLUMNS) - set(values))
        extra = sorted(set(values) - set(FEATURE_COLUMNS))
        raise ValueError(f"15 features exactes requises; absentes={missing}, inattendues={extra}.")
    normalized = {}
    for key in FEATURE_COLUMNS:
        value = values[key]
        if key == "category":
            if not isinstance(value, str) or not value.strip():
                raise ValueError("La catégorie doit être une chaîne non vide.")
            normalized[key] = value.strip()
            continue
        if value is None or (isinstance(value, float) and math.isnan(value)):
            normalized[key] = float("nan")
            continue
        try:
            numeric = float(value)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError(f"Feature numérique invalide : {key}") from exc
        if not math.isfinite(numeric):
            raise ValueError(f"Feature non finie : {key}")
        normalized[key] = numeric
    return normalized


def score_precomputed_features(model: Any, config: dict, values: dict, trans_num: str) -> dict:
    if not isinstance(trans_num, str) or not trans_num.strip():
        raise ValueError("Identifiant trans_num non vide obligatoire.")
    normalized = normalize_features(values)
    features = list(FEATURE_COLUMNS)
    if config.get("features") != features:
        raise ValueError("Le schéma des features ne correspond pas à celui du modèle.")
    proba = model.predict_proba(pd.DataFrame([normalized], columns=features))
    classes = list(model.classes_)
    if 1 not in classes:
        raise ValueError("Le modèle ne contient pas la classe de fraude 1.")
    score = float(proba[0, classes.index(1)])
    threshold = float(config["threshold"])
    return {
        "trans_num": trans_num, "fraud_score": score, "threshold": threshold,
        "decision": "fraud" if score >= threshold else "non_fraud",
        "mode": "preview", "features": {
            key: (None if isinstance(value, float) and math.isnan(value) else value)
            for key, value in normalized.items()
        },
        "feature_origin": "provided_by_csv",
        "feature_context_note": (
            "Features fournies par le CSV, non recalculées. "
            "La causalité de leur historique dépend de la provenance du fichier."
        ),
    }
