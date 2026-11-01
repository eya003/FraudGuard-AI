"""Unsupervised experiments, fit only on historical normal Train transactions.

Output is a percentile of normal-Train anomaly scores, NOT a calibrated
probability of fraud. The transformation is frozen after fitting.
"""
from __future__ import annotations

import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import IsolationForest
from sklearn.impute import SimpleImputer
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from backend.features.feature_engineering import FEATURE_COLUMNS


class AnomalyExperiment:
    def __init__(self, algorithm: str, params: dict | None = None, seed: int = 42):
        if algorithm not in {"isolation_forest", "autoencoder"}:
            raise ValueError("Algorithme non supervisé inconnu.")
        self.algorithm = algorithm
        self.params = params or {}
        self.seed = seed
        numeric = [c for c in FEATURE_COLUMNS if c != "category"]
        self.preprocess = ColumnTransformer([
            ("num", Pipeline([("imputer", SimpleImputer(strategy="median")),
                              ("scale", StandardScaler())]), numeric),
            ("cat", Pipeline([("imputer", SimpleImputer(strategy="most_frequent")),
                              ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))]), ["category"]),
        ], sparse_threshold=0)

    def fit(self, frame):
        data = np.asarray(self.preprocess.fit_transform(frame), dtype=np.float64)
        if self.algorithm == "isolation_forest":
            self.estimator = IsolationForest(
                n_estimators=int(self.params.get("n_estimators", 150)),
                contamination="auto", max_samples="auto", random_state=self.seed, n_jobs=2)
            self.estimator.fit(data)
        else:
            # Reconstruction model, trained only on the normal-class Train records.
            self.estimator = MLPRegressor(
                hidden_layer_sizes=(32, 12, 32), activation="relu",
                max_iter=int(self.params.get("max_iter", 35)),
                early_stopping=True, validation_fraction=0.1, n_iter_no_change=5,
                random_state=self.seed)
            self.estimator.fit(data, data)
        self.reference_scores_ = np.sort(self._raw(data))
        return self

    def _raw(self, values):
        if self.algorithm == "isolation_forest":
            return -np.asarray(self.estimator.score_samples(values), dtype=float)
        predicted = np.asarray(self.estimator.predict(values), dtype=float)
        return np.mean((values - predicted) ** 2, axis=1)

    def predict_risk(self, frame):
        data = np.asarray(self.preprocess.transform(frame), dtype=np.float64)
        raw = self._raw(data)
        return np.searchsorted(self.reference_scores_, raw, side="right") / len(self.reference_scores_)
