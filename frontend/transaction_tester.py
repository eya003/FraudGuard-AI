"""Pure helpers shared by the Streamlit Transaction Tester and its tests."""

from __future__ import annotations

from typing import Any

import pandas as pd

RAW_TRANSACTION_FIELDS = (
    "trans_num",
    "trans_date_trans_time",
    "cc_num",
    "amt",
    "city_pop",
    "category",
    "lat",
    "long",
    "merch_lat",
    "merch_long",
)


def prepare_csv_transaction(row: dict[str, Any], import_id: str, row_index: int) -> dict:
    """Keep only the API raw schema and generate a stable ID if absent."""
    values = dict(row)
    if "amt" not in values and "amount" in values:
        values["amt"] = values["amount"]
    if pd.isna(values.get("trans_num")) or not str(values.get("trans_num", "")).strip():
        values["trans_num"] = f"{import_id}-{row_index}"
    missing = [
        field for field in RAW_TRANSACTION_FIELDS
        if field != "trans_num" and (field not in values or pd.isna(values[field]))
    ]
    if missing:
        raise ValueError(f"Champs bruts absents ou vides : {', '.join(missing)}")

    transaction = {}
    for field in RAW_TRANSACTION_FIELDS:
        value = values[field]
        transaction[field] = value.item() if hasattr(value, "item") else value
    return transaction


def response_view(response: dict) -> dict:
    """Expose exactly the API result fields consumed by the result panel."""
    required = ("trans_num", "fraud_score", "threshold", "decision", "mode", "features")
    missing = [key for key in required if key not in response]
    if missing:
        raise ValueError(f"Réponse API incomplète : {', '.join(missing)}")
    shown = {key: response[key] for key in required}
    for extra in ("model_name", "score_kind", "baseline_fraud_score", "alert_decision", "alert"):
        if extra in response:
            shown[extra] = response[extra]
    return shown
