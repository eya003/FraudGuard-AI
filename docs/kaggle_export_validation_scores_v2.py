"""Kaggle cell: export real V2 Validation scores for Risk Simulator.

Run after the final V2 pipeline and Validation arrays have been created.
This does not train the model and does not access Test predictions.
"""

from pathlib import Path

import numpy as np
import pandas as pd

required = {"final_model", "X_val", "y_val", "val_fe", "final_threshold"}
missing = sorted(required - set(globals()))
if missing:
    raise RuntimeError(f"Variables V2 absentes : {missing}")
if len(X_val) != len(y_val) or len(X_val) != len(val_fe):
    raise RuntimeError("X_val, y_val et val_fe doivent avoir le même nombre de lignes.")
if "trans_date_trans_time" not in val_fe or "amt" not in val_fe:
    raise RuntimeError("val_fe doit contenir trans_date_trans_time et amt.")

fraud_index = list(final_model.classes_).index(1)
scores = final_model.predict_proba(X_val)[:, fraud_index]
if not np.isfinite(scores).all():
    raise RuntimeError("Scores de Validation non finis; export interrompu.")

validation = pd.DataFrame(
    {
        "timestamp": pd.to_datetime(val_fe["trans_date_trans_time"]).to_numpy(),
        "score": scores,
        "is_fraud": np.asarray(y_val),
        "amt": val_fe["amt"].to_numpy(),
        "split": "validation",
        "model": "Random Forest V2",
        "threshold_reference": float(final_threshold),
        "daily_budget_reference": int(final_daily_budget) if "final_daily_budget" in globals() else 30,
        "cost_per_alert_reference": float(final_alert_cost) if "final_alert_cost" in globals() else 5.0,
    }
)
if not validation["is_fraud"].isin([0, 1]).all():
    raise RuntimeError("Les labels de Validation doivent être binaires.")

output = Path("/kaggle/working/fraudguard_v2_validation_scores.csv")
validation.to_csv(output, index=False, float_format="%.17g")
print("Scores V2 réels de Validation exportés :", output, "— lignes :", len(validation))
print("Aucune prédiction Test n'est incluse.")
