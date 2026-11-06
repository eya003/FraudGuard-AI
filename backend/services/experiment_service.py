"""Asynchronous, isolated experiments trained on Train and evaluated on Validation."""
from __future__ import annotations

import json
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (average_precision_score, precision_recall_curve, confusion_matrix, fbeta_score,
                             matthews_corrcoef, precision_score, recall_score,
                             roc_auc_score)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit

from backend.services.import_service import CHUNK_SIZE, ImportError, _import_paths
from backend.services.model_service import PROJECT_ROOT
from backend.features.feature_engineering import FEATURE_COLUMNS
from backend.services.anomaly_models import AnomalyExperiment

EXPERIMENT_DIR = PROJECT_ROOT / "data" / "experiments"
_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="fraudguard-ml")
_lock = threading.Lock()
_enqueue_lock = threading.Lock()
MAX_TRAIN_SAMPLE = 100_000
_numeric = [c for c in FEATURE_COLUMNS if c != "category"]


def _read_splits(import_id: str, sample_limit: int | None, *, chronological: bool = False):
    """Stream Train and Validation; sample Train without materializing 1M+ rows.

    Uniform-priority per-class reservoir is deterministic for the same CSV and seed.
    Full Validation is kept for unbiased evaluation; Test is never loaded.
    """
    import numpy as np
    _, path = _import_paths(import_id)
    rng = np.random.default_rng(42)
    val_parts = []
    train_parts = []
    pools = {0: pd.DataFrame(), 1: pd.DataFrame()}
    cap = max(1, sample_limit // 2) if sample_limit else None
    for chunk in pd.read_csv(path, chunksize=CHUNK_SIZE):
        cols = {str(c).strip().lower(): c for c in chunk.columns}
        split_col = cols.get("_split", cols.get("split"))
        label_col = cols.get("is_fraud", cols.get("label", cols.get("y_true")))
        time_col = next((cols[k] for k in ("trans_date_trans_time", "timestamp", "date") if k in cols), None)
        if split_col is None or label_col is None:
            raise ImportError("Le CSV doit contenir split/_split et is_fraud/label.")
        missing = [c for c in FEATURE_COLUMNS if c.lower() not in cols]
        if missing:
            raise ImportError(f"Features requises absentes : {missing}")
        splits = chunk[split_col].astype("string").str.lower().replace({"val": "validation"})
        selected = splits.isin(["train", "validation"])
        if not selected.any():
            continue
        base = chunk.loc[selected, [cols[c.lower()] for c in FEATURE_COLUMNS]].copy()
        base.columns = FEATURE_COLUMNS
        base["_label"] = pd.to_numeric(chunk.loc[selected, label_col], errors="coerce")
        base["_split"] = splits.loc[selected].to_numpy()
        if time_col:
            base["_time"] = pd.to_datetime(chunk.loc[selected, time_col], errors="coerce")
        validation = base.loc[base["_split"].eq("validation")]
        if not validation.empty:
            val_parts.append(validation)
        train_chunk = base.loc[base["_split"].eq("train")]
        if train_chunk.empty:
            continue
        if cap is None:
            train_parts.append(train_chunk)
        else:
            if train_chunk["_label"].isna().any():
                raise ImportError("Train contient des labels invalides.")
            for label in [0, 1]:
                items = train_chunk.loc[train_chunk["_label"].eq(label)].copy()
                if items.empty:
                    continue
                items["_priority"] = rng.random(len(items))
                pools[label] = pd.concat([pools[label], items], ignore_index=True)
                if len(pools[label]) > cap:
                    pools[label] = pools[label].nsmallest(cap, "_priority")
    train = pd.concat(list(pools.values()), ignore_index=True).drop(columns="_priority", errors="ignore") if cap else (
        pd.concat(train_parts, ignore_index=True) if train_parts else pd.DataFrame())
    validation = pd.concat(val_parts, ignore_index=True) if val_parts else pd.DataFrame()
    if train.empty or validation.empty:
        raise ImportError("Il faut des partitions Train et Validation non vides.")
    for part, name in ((train, "Train"), (validation, "Validation")):
        if part["_label"].isna().any() or not part["_label"].isin([0, 1]).all() or part["_label"].nunique() < 2:
            raise ImportError(f"La partition {name} doit avoir des labels binaires et deux classes.")
    if chronological:
        if "_time" not in train or train["_time"].isna().any():
            raise ImportError("TimeSeriesSplit exige une colonne temporelle valide (timestamp/date).")
        train = train.sort_values("_time", kind="stable")
    return train, validation

def _model(algorithm: str, params: dict, class_weight: str, seed: int, feature_columns=None):
    feature_columns = list(feature_columns or FEATURE_COLUMNS)
    n_estimators = int(params.get("n_estimators", 100))
    max_depth = params.get("max_depth", 12)
    if algorithm in {"random_forest", "xgboost", "isolation_forest"} and not 10 <= n_estimators <= 300:
        raise ValueError("Limite publique : n_estimators doit être compris entre 10 et 300.")
    if algorithm in {"random_forest", "xgboost"} and max_depth is not None and not 1 <= int(max_depth) <= 24:
        raise ValueError("Limite publique : max_depth doit être compris entre 1 et 24.")
    if algorithm == "autoencoder" and not 5 <= int(params.get("max_iter", 35)) <= 50:
        raise ValueError("Limite publique : l'Autoencoder accepte 5 à 50 itérations.")
    if algorithm == "logistic_regression":
        estimator = LogisticRegression(
            C=float(params.get("C", 1.0)), max_iter=int(params.get("max_iter", 300)),
            class_weight=None if class_weight == "none" else "balanced",
            random_state=seed, n_jobs=1,
        )
    elif algorithm == "random_forest":
        estimator = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_leaf=int(params.get("min_samples_leaf", 2)),
            max_features=float(params.get("max_features", 1.0)),
            max_samples=float(params["max_samples"]) if params.get("max_samples") else None,
            class_weight=None if class_weight == "none" else class_weight,
            n_jobs=-1, random_state=seed,
        )
    elif algorithm == "xgboost":
        try:
            from xgboost import XGBClassifier
        except ImportError as exc:
            raise ValueError("XGBoost n'est pas installé dans l'environnement Python.") from exc
        estimator = XGBClassifier(
            n_estimators=n_estimators,
            max_depth=int(max_depth or 6),
            learning_rate=float(params.get("learning_rate", 0.1)),
            scale_pos_weight=float(params.get("scale_pos_weight", 1.0)),
            min_child_weight=int(params.get("min_child_weight", 1)),
            subsample=float(params.get("subsample", 1.0)),
            colsample_bytree=float(params.get("colsample_bytree", 1.0)),
            reg_lambda=float(params.get("reg_lambda", 1.0)),
            eval_metric="logloss", n_jobs=-1, random_state=seed,
        )
    elif algorithm in {"isolation_forest", "autoencoder"}:
        return AnomalyExperiment(algorithm, params=params, seed=seed)
    else:
        raise ValueError("Algorithme non disponible.")
    steps = []
    numeric = [column for column in feature_columns if column != "category"]
    if numeric:
        steps.append(("num", Pipeline([("imputer", SimpleImputer(strategy="median"))]), numeric))
    if "category" in feature_columns:
        steps.append(("cat", Pipeline([("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore"))]), ["category"]))
    preprocess = ColumnTransformer(steps)
    return Pipeline([("preprocessing", preprocess), ("model", estimator)])


def _save_job(job: dict):
    EXPERIMENT_DIR.mkdir(parents=True, exist_ok=True)
    destination = EXPERIMENT_DIR / f"{job['id']}.json"
    temporary = destination.with_name(f"{destination.stem}.{uuid.uuid4().hex}.tmp")
    with _lock:
        temporary.write_text(json.dumps(job, indent=2), encoding="utf-8")
        try:
            for attempt in range(10):
                try:
                    temporary.replace(destination)
                    break
                except PermissionError:
                    if attempt == 9:
                        raise
                    time.sleep(0.02 * (attempt + 1))
        finally:
            temporary.unlink(missing_ok=True)


def _execute(job: dict, import_id: str, algorithm: str, params: dict, class_weight: str,
             sample_limit: int | None, threshold: float, smotenc: bool, cv_folds: int,
             feature_columns: list[str]):
    try:
        job["status"] = "running"; _save_job(job)
        train, val = _read_splits(import_id, sample_limit, chronological=cv_folds > 0)
        model = _model(algorithm, params, class_weight, seed=42, feature_columns=feature_columns)
        X_train = train[feature_columns]
        y_train = train["_label"].astype(int)
        if algorithm in {"isolation_forest", "autoencoder"} and (smotenc or cv_folds or class_weight != "none"):
            raise ValueError("Pour Isolation Forest et Autoencoder, désactivez SMOTENC, CV et pondération.")
        if algorithm == "xgboost" and class_weight != "none":
            raise ValueError("XGBoost utilise scale_pos_weight, pas class_weight.")
        if cv_folds:
            if smotenc:
                raise ValueError("La recherche TimeSeriesSplit et SMOTENC ne peuvent pas être combinées dans cette version.")
            folds = min(cv_folds, max(2, len(train) // 20))
            search = GridSearchCV(model, {"model__n_estimators": [100, 200],
                "model__max_depth": [8, 16]}, cv=TimeSeriesSplit(n_splits=folds),
                scoring="average_precision", n_jobs=1, refit=True)
            search.fit(X_train, y_train)
            model = search.best_estimator_
            job["best_params"] = search.best_params_
            job["cv_score"] = float(search.best_score_)
        elif smotenc:
            try:
                from imblearn.over_sampling import SMOTENC
            except ImportError as exc:
                raise ValueError("Installez imbalanced-learn pour activer SMOTENC.") from exc
            resampled = X_train.copy()
            if "category" not in feature_columns:
                raise ValueError("SMOTENC exige une feature catégorielle. Désactivez-le pour cette ablation.")
            for col in [c for c in feature_columns if c != "category"]:
                resampled[col] = resampled[col].fillna(resampled[col].median())
            resampled["category"] = resampled["category"].fillna("(missing)").astype(str)
            sampler = SMOTENC(categorical_features=[feature_columns.index("category")], random_state=42)
            X_train, y_train = sampler.fit_resample(resampled, y_train)
        from threadpoolctl import threadpool_limits
        with threadpool_limits(limits=2):
            if algorithm in {"isolation_forest", "autoencoder"}:
                model.fit(X_train.loc[y_train.eq(0)])
                scores = model.predict_risk(val[feature_columns])
            else:
                model.fit(X_train, y_train)
                scores = model.predict_proba(val[feature_columns])[:, list(model.classes_).index(1)]
        y = val["_label"].astype(int).to_numpy()
        pred = (scores >= threshold).astype(int)
        topk_precision = None
        topk_recall = None
        if "_time" in val and val["_time"].notna().all():
            ranking = val[["_time", "_label"]].copy()
            ranking["_score"] = scores
            ranking["_day"] = ranking["_time"].dt.strftime("%Y-%m-%d")
            topk = ranking.sort_values("_score", ascending=False, kind="stable").groupby("_day", sort=False).head(30)
            topk_precision = float(topk["_label"].mean()) if len(topk) else None
            topk_recall = float(topk["_label"].sum() / max(int(y.sum()), 1))
        missed_amount = float(val.loc[(val["_label"].to_numpy() == 1) & (pred == 0), "amt"].sum())
        alert_count = int(pred.sum())
        precision_points, recall_points, _ = precision_recall_curve(y, scores)
        indices = sorted(set([*range(0, len(precision_points), max(1, len(precision_points) // 150)), len(precision_points)-1]))
        pr_points = [{"recall": float(recall_points[i]), "precision": float(precision_points[i])} for i in indices]
        monthly = []
        if "_time" in val and val["_time"].notna().all():
            val_months = val["_time"].dt.to_period("M").astype(str)
            for month in sorted(val_months.unique()):
                mask = val_months.eq(month).to_numpy()
                labels = y[mask]
                monthly.append({"month":month, "transactions":int(mask.sum()),
                                "fraud_rate":float(labels.mean()),
                                "pr_auc":float(average_precision_score(labels, scores[mask])) if len(set(labels)) == 2 else None})
        metrics = {"rows_train": len(train), "rows_validation": len(val),
                   "monthly_pr_auc": monthly, "features_used": feature_columns,
                   "pr_curve": pr_points,
                   "score_kind": "normal_train_anomaly_percentile" if algorithm in {"isolation_forest", "autoencoder"} else "uncalibrated_model_score",
                   "pr_auc": float(average_precision_score(y, scores)),
                   "roc_auc": float(roc_auc_score(y, scores)),
                   "precision": float(precision_score(y, pred, zero_division=0)),
                   "recall": float(recall_score(y, pred, zero_division=0)),
                   "f2": float(fbeta_score(y, pred, beta=2, zero_division=0)),
                   "mcc": float(matthews_corrcoef(y, pred)),
                   "precision_at_k": topk_precision, "recall_at_k": topk_recall,
                   "total_cost": missed_amount + alert_count * 5.0,
                   "confusion_matrix": confusion_matrix(y, pred, labels=[0, 1]).tolist(),
                   "threshold": threshold, "evaluation_split": "Validation"}
        dest = EXPERIMENT_DIR / job["id"]
        dest.mkdir(parents=True, exist_ok=True)
        joblib.dump(model, dest / "model.joblib")
        (dest / "config.json").write_text(json.dumps({"algorithm": algorithm, "params": params,
            "class_weight": class_weight, "threshold": threshold, "feature_columns": feature_columns,
            "trained_at": datetime.now(timezone.utc).isoformat()}, indent=2), encoding="utf-8")
        job.update(status="completed", metrics=metrics, artifact=str(dest.relative_to(PROJECT_ROOT)))
    except Exception as exc:
        job.update(status="failed", error=f"{type(exc).__name__}: {exc}")
    finally:
        _save_job(job)


def start_experiment(import_id: str, *, algorithm: str, params: dict,
                     class_weight: str, sample_limit: int | None, threshold: float,
                     smotenc: bool = False, cv_folds: int = 0,
                     feature_columns: list[str] | None = None) -> dict:
    if class_weight not in {"none", "balanced", "balanced_subsample"}:
        raise ValueError("Pondération de classes invalide.")
    if sample_limit is not None and not 100 <= sample_limit <= MAX_TRAIN_SAMPLE:
        raise ValueError(f"L'échantillon Train doit être entre 100 et {MAX_TRAIN_SAMPLE:,} lignes.")
    feature_columns = list(FEATURE_COLUMNS) if feature_columns is None else list(feature_columns)
    if (not feature_columns or len(set(feature_columns)) != len(feature_columns)
            or any(c not in FEATURE_COLUMNS for c in feature_columns)):
        raise ValueError("La sélection de features est invalide.")
    if algorithm in {"isolation_forest", "autoencoder"} and feature_columns != list(FEATURE_COLUMNS):
        raise ValueError("L'ablation de variables est disponible pour les modèles supervisés uniquement.")
    _model(algorithm, params, class_weight, 42, feature_columns)  # Validate before enqueueing.
    if algorithm in {"isolation_forest", "autoencoder"} and (class_weight != "none" or smotenc or cv_folds):
        raise ValueError("Pas de pondération, SMOTENC ni recherche CV pour les modèles non supervisés.")
    if algorithm == "xgboost" and class_weight != "none":
        raise ValueError("Choisissez none et réglez scale_pos_weight dans les hyperparamètres XGBoost.")
    if cv_folds and algorithm != "random_forest":
        raise ValueError("TimeSeriesSplit paramétré est disponible uniquement pour Random Forest.")
    if not 0 <= cv_folds <= 5:
        raise ValueError("TimeSeriesSplit accepte de 0 à 5 folds.")
    if not 0 <= threshold <= 1:
        raise ValueError("Le seuil doit être dans [0, 1].")
    with _enqueue_lock:
        active = [job for job in list_experiments() if job.get("status") in {"queued", "running"}]
        if active:
            raise ValueError("Un entraînement est déjà en cours; réessayez après sa fin.")
        job = {"id": str(uuid.uuid4()), "status": "queued", "created_at": datetime.now(timezone.utc).isoformat(),
               "algorithm": algorithm, "import_id": import_id}
        _save_job(job)
        _executor.submit(_execute, job, import_id, algorithm, params, class_weight,
                         sample_limit, threshold, smotenc, cv_folds, feature_columns)
    return job


def get_experiment(job_id: str) -> dict:
    try: uuid.UUID(job_id)
    except (ValueError, AttributeError) as exc: raise FileNotFoundError("Identifiant expérimental invalide.") from exc
    path = EXPERIMENT_DIR / f"{job_id}.json"
    if not path.is_file(): raise FileNotFoundError("Expérience introuvable.")
    return json.loads(path.read_text(encoding="utf-8"))


def list_experiments() -> list[dict]:
    if not EXPERIMENT_DIR.exists(): return []
    return sorted((json.loads(p.read_text(encoding="utf-8")) for p in EXPERIMENT_DIR.glob("*.json")),
                  key=lambda x: x.get("created_at", ""), reverse=True)
