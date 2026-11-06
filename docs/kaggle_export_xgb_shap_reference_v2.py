"""Export grouped XGBoost Validation SHAP rows for Explainable AI.

Run after cells 58-66. Contains the notebook's validation SHAP sample only,
with no card or transaction IDs and no Test rows.
"""
from pathlib import Path

import numpy as np
import pandas as pd


required = {"X_shap", "phi_grouped", "base_values", "grouped_features", "indices",
            "scores_xgb_best", "y_val", "val_fe", "optimize_threshold"}
missing = sorted(required - set(globals()))
if missing:
    raise RuntimeError(f"Exécutez les cellules SHAP V2 58-66; variables absentes : {missing}")
features = list(grouped_features)
phi = np.asarray(phi_grouped, dtype=np.float64)
indices = np.asarray(indices, dtype=np.int64)
base = np.asarray(base_values, dtype=np.float64).reshape(-1)
if phi.shape != (len(indices), len(features)) or len(base) != len(indices):
    raise RuntimeError("Dimensions SHAP incompatibles avec l'échantillon V2.")
if (indices < 0).any() or (indices >= len(val_fe)).any():
    raise RuntimeError("Un index de l'échantillon sort de la partition Validation.")

threshold = optimize_threshold(
    np.asarray(scores_xgb_best), np.asarray(y_val),
    pd.to_numeric(val_fe["amt"], errors="raise").to_numpy(), alert_cost=5,
)["Threshold"]
frame = X_shap.loc[:, features].reset_index(drop=True).copy()
frame.insert(0, "validation_row", indices)
frame["timestamp"] = pd.to_datetime(val_fe.iloc[indices]["trans_date_trans_time"]).to_numpy()
frame["is_fraud"] = np.asarray(y_val)[indices]
frame["score_xgb_optimized"] = np.asarray(scores_xgb_best)[indices]
frame["base_value"] = base
frame["threshold_validation"] = float(threshold)
frame["error_type"] = np.where(
    (frame["score_xgb_optimized"] >= threshold) & (frame["is_fraud"] == 0), "false_positive",
    np.where((frame["score_xgb_optimized"] < threshold) & (frame["is_fraud"] == 1),
             "false_negative", "correct"),
)
for position, feature in enumerate(features):
    frame[f"shap_{feature}"] = phi[:, position]

output = Path("/kaggle/working/fraudguard_v2_xgb_shap_validation_sample.csv.gz")
frame.to_csv(output, index=False, compression="gzip", float_format="%.17g")
print("Échantillon SHAP Validation exporté :", output, "| lignes :", len(frame))
print("Répartition des erreurs :", frame["error_type"].value_counts().to_dict())
print("Seuil utilisé : coût métier optimisé sur Validation; aucune ligne Test.")
