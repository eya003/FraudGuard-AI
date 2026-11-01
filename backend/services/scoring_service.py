"""Scoring RF V2 avec séparation preview/record et émission d'alertes en record."""

from __future__ import annotations

from typing import Any, Literal

import pandas as pd

from backend.features.feature_engineering import FEATURE_COLUMNS
from backend.services.feature_service import FeatureService
from backend.services.model_service import load_model_artifacts
from backend.services.alert_service import AlertService
from functools import lru_cache


class ScoringService:
    def __init__(
        self,
        feature_service: FeatureService | None = None,
        *,
        model: Any | None = None,
        config: dict | None = None,
    ) -> None:
        if model is None or config is None:
            loaded_model, loaded_config = load_model_artifacts()
            model = loaded_model if model is None else model
            config = loaded_config if config is None else config
        features = config.get("features")
        if features != list(FEATURE_COLUMNS):
            raise ValueError("La configuration du modèle ne correspond pas au schéma de features V2.")
        model_features = getattr(model, "feature_names_in_", None)
        if model_features is not None and list(model_features) != features:
            raise ValueError("L'ordre des features du modèle ne correspond pas à config.json.")
        classes = list(model.classes_)
        if 1 not in classes:
            raise ValueError("Le pipeline ne contient pas la classe positive 1.")
        self.model = model
        self.config = config
        self.feature_service = feature_service or FeatureService()
        self.alert_service = AlertService(self.feature_service)
        self.feature_columns = features
        self.fraud_class_index = classes.index(1)

    def score(
        self, transaction: dict, *, mode: Literal["preview", "record"] = "preview"
    ) -> dict:
        raw = pd.DataFrame([transaction])
        result = (
            self.feature_service.preview(raw)
            if mode == "preview"
            else self.feature_service.transform_and_persist(raw)
        )
        score = float(
            self.model.predict_proba(result.features.loc[:, self.feature_columns])[
                0, self.fraud_class_index
            ]
        )
        threshold = float(self.config["threshold"])
        feature_values = {}
        for column, value in result.features.loc[0, self.feature_columns].items():
            if pd.isna(value):
                feature_values[column] = None
            elif hasattr(value, "item"):
                feature_values[column] = value.item()
            else:
                feature_values[column] = value
        response = {
            "trans_num": result.transaction_ids[0],
            "fraud_score": score,
            "threshold": threshold,
            "decision": "fraud" if score >= threshold else "non_fraud",
            "mode": mode,
            "features": feature_values,
        }
        if mode == "record":
            alert_result = self.alert_service.evaluate_recorded_score(
                result.transaction_ids[0], score, self.config
            )
            response["alert_decision"] = alert_result["outcome"]
            response["alert"] = alert_result["alert"]
        return response


@lru_cache(maxsize=1)
def get_scoring_service() -> ScoringService:
    return ScoringService()
