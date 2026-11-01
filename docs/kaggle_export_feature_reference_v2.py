"""Cellule à copier dans Kaggle après la cellule 73 du notebook fraude-v2."""

from pathlib import Path

import numpy as np
import pandas as pd


# Variables déjà créées dans le notebook : history_df (cellules 12–24),
# features (cellule 25) et loaded_model (cellule 73).
required_variables = {"history_df", "features", "loaded_model"}
missing_variables = [name for name in required_variables if name not in globals()]
if missing_variables:
    raise RuntimeError(f"Exécutez d'abord les cellules V2; variables absentes : {missing_variables}")

raw_columns = [
    "trans_date_trans_time",
    "cc_num",
    "amt",
    "city_pop",
    "category",
    "lat",
    "long",
    "merch_lat",
    "merch_long",
]
raw_export_columns = ["trans_num", *raw_columns]
feature_columns = list(features)  # Liste du notebook créée à la cellule 25.

assert len(feature_columns) == 15
assert list(loaded_model.feature_names_in_) == feature_columns
required_columns = [*raw_export_columns, "_split", *feature_columns]
missing_columns = [column for column in required_columns if column not in history_df]
if missing_columns:
    raise RuntimeError(f"Colonnes absentes de history_df : {missing_columns}")
if history_df["trans_num"].duplicated().any():
    raise RuntimeError("trans_num n'est pas unique dans history_df; export interrompu.")

ordered = history_df.sort_values(
    "trans_date_trans_time", kind="stable"
).reset_index(drop=True)
scenario_by_id = {}


def add_scenario(rows, scenario):
    for transaction_id in rows["trans_num"].astype(str):
        scenario_by_id.setdefault(transaction_id, set()).add(scenario)


# Plusieurs premières transactions par carte, pour couvrir les historiques vides.
first_per_card = ordered.groupby("cc_num", sort=False).head(1).head(5)
add_scenario(first_per_card, "first_purchase")

# Échantillon déterministe du Test final, afin d'inclure un éventail de cartes et dates.
test_rows = ordered.loc[ordered["_split"] == "test"]
random_test = test_rows.sample(n=min(50, len(test_rows)), random_state=42)
add_scenario(random_test, "random_test")

# Groupes complets ayant le même cc_num et le même horodatage.
same_time_mask = ordered.duplicated(
    subset=["cc_num", "trans_date_trans_time"], keep=False
)
same_time_rows = ordered.loc[same_time_mask]
same_time_groups = same_time_rows[
    ["cc_num", "trans_date_trans_time"]
].drop_duplicates().head(10)
if not same_time_groups.empty:
    simultaneous = same_time_rows.merge(
        same_time_groups,
        on=["cc_num", "trans_date_trans_time"],
        how="inner",
        sort=False,
    )
    add_scenario(simultaneous, "simultaneous")

# Chercher des exemples réels aux bornes exactes; les tests locaux couvrent
# aussi ces bornes synthétiquement si le jeu Kaggle n'en contient pas.
boundary_durations = {
    "boundary_1h": np.timedelta64(1, "h"),
    "boundary_24h": np.timedelta64(24, "h"),
    "boundary_30d": np.timedelta64(30, "D"),
}
boundary_rows = []
for card, card_rows in ordered.groupby("cc_num", sort=False):
    times = card_rows["trans_date_trans_time"].to_numpy(dtype="datetime64[ns]")
    is_test = card_rows["_split"].eq("test").to_numpy()
    for scenario, duration in boundary_durations.items():
        left = np.searchsorted(times, times - duration, side="left")
        valid = left < len(times)
        exact_boundary = np.zeros(len(times), dtype=bool)
        exact_boundary[valid] = times[left[valid]] == times[valid] - duration
        candidates = np.flatnonzero(valid & exact_boundary & is_test)
        if len(candidates):
            row = card_rows.iloc[[int(candidates[0])]]
            boundary_rows.append((scenario, row))
            add_scenario(row, scenario)

selected_ids = set(scenario_by_id)
targets = ordered.loc[ordered["trans_num"].astype(str).isin(selected_ids)].copy()
targets["reference_case"] = targets["trans_num"].astype(str).map(
    lambda transaction_id: "|".join(sorted(scenario_by_id[transaction_id]))
)
targets = targets.sort_values("trans_date_trans_time", kind="stable").reset_index(drop=True)

if targets.empty:
    raise RuntimeError("Aucune transaction de référence n'a été sélectionnée.")

# Garder le contexte exact nécessaire pour chaque cible : les transactions
# des 30 jours précédents et une transaction antérieure supplémentaire pour
# time_since_last_tx. Tout autre contexte est exclu.
context_ids = set()
rows_by_card = {card: rows for card, rows in ordered.groupby("cc_num", sort=False)}
for card, card_targets in targets.groupby("cc_num", sort=False):
    card_history = rows_by_card[card]
    times = card_history["trans_date_trans_time"].to_numpy(dtype="datetime64[ns]")
    ids = card_history["trans_num"].astype(str).to_numpy()
    for target_time in card_targets["trans_date_trans_time"].to_numpy(dtype="datetime64[ns]"):
        right = np.searchsorted(times, target_time, side="left")
        left_30d = np.searchsorted(
            times, target_time - np.timedelta64(30, "D"), side="left"
        )
        context_ids.update(ids[left_30d:right])
        if left_30d > 0:
            context_ids.add(ids[left_30d - 1])

context_ids.difference_update(selected_ids)
context = ordered.loc[ordered["trans_num"].astype(str).isin(context_ids)]
context = context.sort_values("trans_date_trans_time", kind="stable")

fraud_class_index = list(loaded_model.classes_).index(1)
expected_scores = loaded_model.predict_proba(targets[feature_columns])[:, fraud_class_index]

export_dir = Path("/kaggle/working/FraudGuard_V2_Feature_Reference")
export_dir.mkdir(parents=True, exist_ok=True)
history_path = export_dir / "feature_history_v2.csv"
reference_path = export_dir / "feature_reference_v2.csv"

context.loc[:, raw_export_columns].to_csv(
    history_path, index=False, float_format="%.17g"
)
reference = targets.loc[:, [*raw_export_columns, *feature_columns]].copy()
reference["expected_score"] = expected_scores
reference["reference_case"] = targets["reference_case"].to_numpy()
reference.to_csv(reference_path, index=False, float_format="%.17g")

print("Historique brut exporté :", history_path, "— lignes :", len(context))
print("Références brutes/features/scores exportées :", reference_path, "— lignes :", len(reference))
print("Cas inclus :", sorted({case for cases in scenario_by_id.values() for case in cases}))
print("Cibles aux bornes exactes trouvées :", [name for name, _ in boundary_rows])
print("Features :", feature_columns)
print("Aucune étiquette is_fraud n'est exportée.")
