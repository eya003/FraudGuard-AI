"""Chargement explicite des artefacts FraudGuard V2.

Ce module ne charge rien à l’import. L’application pourra appeler
``load_model_artifacts`` au démarrage dans l’étape d’intégration du modèle.
"""

import json
from functools import lru_cache
from pathlib import Path
from typing import Any

import joblib

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODEL_PATH = PROJECT_ROOT / "models" / "random_forest_pipeline_v2.joblib"
CONFIG_PATH = PROJECT_ROOT / "models" / "config.json"


@lru_cache(maxsize=1)
def load_model_artifacts() -> tuple[Any, dict]:
    """Charge et contrôle le pipeline et sa configuration, sans effectuer de prédiction."""
    if not MODEL_PATH.is_file():
        raise FileNotFoundError(f"Pipeline FraudGuard V2 introuvable : {MODEL_PATH}")
    if not CONFIG_PATH.is_file():
        raise FileNotFoundError(f"Configuration FraudGuard V2 introuvable : {CONFIG_PATH}")

    with CONFIG_PATH.open(encoding="utf-8") as config_file:
        config = json.load(config_file)

    features = config.get("features")
    threshold = config.get("threshold")
    if not isinstance(features, list) or not features:
        raise ValueError("La configuration V2 doit définir la liste ordonnée 'features'.")
    if not isinstance(threshold, (int, float)) or not 0 <= threshold <= 1:
        raise ValueError("Le seuil V2 doit être un nombre compris entre 0 et 1.")

    # joblib repose sur pickle : charge uniquement l’artefact de confiance du projet.
    model = joblib.load(MODEL_PATH)
    model_features = getattr(model, "feature_names_in_", None)
    if model_features is not None and list(model_features) != features:
        raise ValueError(
            "L’ordre des features du pipeline ne correspond pas à models/config.json."
        )
    return model, config


def load_model_config() -> dict:
    """Read operational configuration without deserializing the model."""
    if not CONFIG_PATH.is_file():
        raise FileNotFoundError(f"Configuration FraudGuard V2 introuvable : {CONFIG_PATH}")
    return json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
