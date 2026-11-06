"""Export real model scores on notebook Validation for Comparison/Risk pages.

Run in Kaggle after cells 27-50. The file excludes Test and contains no
customer names, addresses, card numbers, or transaction identifiers.
"""
from pathlib import Path

import numpy as np
import pandas as pd


required = {
    "val_fe", "y_val", "scores_normal", "scores_balanced", "scores_smote",
    "scores_rf", "scores_rf_balanced", "scores_rf_smote", "scores_rf_best",
    "scores_xgb_initial", "scores_xgb_weighted", "scores_xgb_smote",
    "scores_xgb_best", "scores_if", "scores_ae",
}
missing = sorted(required - set(globals()))
if missing:
    raise RuntimeError(f"Exécutez les cellules V2 jusqu'à 50; variables absentes : {missing}")
for column in ("trans_date_trans_time", "amt", "category"):
    if column not in val_fe:
        raise RuntimeError(f"val_fe ne contient pas {column}.")

label = np.asarray(y_val, dtype=np.int8)
if not np.isin(label, [0, 1]).all() or len(label) != len(val_fe):
    raise RuntimeError("y_val doit contenir un label binaire par ligne Validation.")

scores = {
    "score_logistic": scores_normal,
    "score_logistic_balanced": scores_balanced,
    "score_logistic_smote": scores_smote,
    "score_rf_initial": scores_rf,
    "score_rf_balanced": scores_rf_balanced,
    "score_rf_smote": scores_rf_smote,
    "score_rf_optimized": scores_rf_best,
    "score_xgb_initial": scores_xgb_initial,
    "score_xgb_weighted": scores_xgb_weighted,
    "score_xgb_smote": scores_xgb_smote,
    "score_xgb_optimized": scores_xgb_best,
    "score_isolation_forest": scores_if,
    "score_autoencoder": scores_ae,
}
frame = pd.DataFrame({
    "timestamp": pd.to_datetime(val_fe["trans_date_trans_time"]).to_numpy(),
    "amt": pd.to_numeric(val_fe["amt"], errors="raise").to_numpy(),
    "category": val_fe["category"].astype(str).to_numpy(),
    "is_fraud": label,
    "split": "validation",
})
for name, values in scores.items():
    values = np.asarray(values, dtype=np.float64)
    if len(values) != len(frame) or not np.isfinite(values).all():
        raise RuntimeError(f"Scores invalides ou longueur incorrecte : {name}")
    frame[name] = values

output = Path("/kaggle/working/fraudguard_v2_validation_model_scores.csv.gz")
frame.to_csv(output, index=False, compression="gzip", float_format="%.17g")
print("Export Validation complet :", output, "| lignes :", len(frame))
print("Scores :", list(scores))
print("Aucun score Test ni identifiant de transaction/carte n'est exporté.")
