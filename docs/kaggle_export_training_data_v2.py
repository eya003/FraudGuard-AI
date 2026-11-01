"""Kaggle cell: export Train and Validation features for the FraudGuard ML Laboratory.

Run after notebook cell 25. This export intentionally excludes Test rows.
"""

from pathlib import Path

import pandas as pd

required = {"history_df", "features"}
missing = sorted(required - set(globals()))
if missing:
    raise RuntimeError(f"Exécutez les cellules V2 jusqu'à 25; variables absentes : {missing}")

feature_columns = list(features)
required_columns = [*feature_columns, "is_fraud", "_split", "trans_date_trans_time"]
missing_columns = [name for name in required_columns if name not in history_df]
if missing_columns:
    raise RuntimeError(f"Colonnes absentes de history_df : {missing_columns}")

train_validation = history_df.loc[
    history_df["_split"].astype(str).str.lower().isin(["train", "val"]),
    required_columns,
].copy()
if set(train_validation["_split"].astype(str).str.lower()) != {"train", "val"}:
    raise RuntimeError("Les partitions Train et Validation doivent toutes deux être présentes.")
if not train_validation["is_fraud"].isin([0, 1]).all():
    raise RuntimeError("is_fraud doit contenir uniquement 0 et 1.")

train_validation["_split"] = train_validation["_split"].astype(str).str.lower()
train_validation = train_validation.sort_values(
    "trans_date_trans_time", kind="stable"
).reset_index(drop=True)

output = Path("/kaggle/working/fraudguard_train_val_features_v2.csv")
train_validation.to_csv(output, index=False, float_format="%.17g")
print("Export :", output, "| lignes :", len(train_validation))
print("Partitions :", train_validation["_split"].value_counts().to_dict())
print("La partition Test n'est pas exportée.")
