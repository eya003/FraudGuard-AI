"""Transformation des transactions brutes selon le notebook FraudGuard V2.

Les historiques sont calculés à la date de l'événement, jamais à l'ordre
d'arrivée des lignes. Pour un même compte et un même horodatage, les lignes
sont simultanées et ne s'incluent pas les unes les autres dans l'historique.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd


RAW_COLUMNS: tuple[str, ...] = (
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

FEATURE_COLUMNS: tuple[str, ...] = (
    "amt",
    "city_pop",
    "hour_sin",
    "hour_cos",
    "day_of_week",
    "is_weekend",
    "time_since_last_tx",
    "has_previous_tx",
    "tx_count_1h",
    "tx_count_24h",
    "tx_count_30d",
    "avg_amount_30d",
    "amount_ratio_30d",
    "distance_km",
    "category",
)

NUMERIC_RAW_COLUMNS: tuple[str, ...] = (
    "amt",
    "city_pop",
    "lat",
    "long",
    "merch_lat",
    "merch_long",
)

EARTH_RADIUS_KM = 6371.0


def validate_raw_transactions(data: pd.DataFrame, *, name: str = "transactions") -> pd.DataFrame:
    missing_columns = [column for column in RAW_COLUMNS if column not in data.columns]
    if missing_columns:
        raise ValueError(f"{name}: colonnes brutes manquantes : {missing_columns}")

    frame = data.loc[:, list(RAW_COLUMNS)].copy().reset_index(drop=True)
    frame["trans_date_trans_time"] = pd.to_datetime(
        frame["trans_date_trans_time"], errors="coerce"
    )
    if frame["trans_date_trans_time"].isna().any():
        rows = frame.index[frame["trans_date_trans_time"].isna()].tolist()[:10]
        raise ValueError(f"{name}: horodatage absent ou illisible aux lignes {rows}.")
    if isinstance(frame["trans_date_trans_time"].dtype, pd.DatetimeTZDtype):
        raise ValueError(
            f"{name}: les horodatages avec fuseau ne sont pas définis dans le notebook V2."
        )

    frame["cc_num"] = pd.to_numeric(frame["cc_num"], errors="coerce")
    invalid_card = frame["cc_num"].isna()
    if invalid_card.any():
        rows = frame.index[invalid_card].tolist()[:10]
        raise ValueError(f"{name}: cc_num absent aux lignes {rows}.")
    card_numbers = frame["cc_num"].to_numpy(dtype=np.float64)
    if not np.equal(card_numbers, np.floor(card_numbers)).all():
        raise ValueError(f"{name}: cc_num doit contenir des identifiants entiers comme dans le notebook V2.")
    frame["cc_num"] = frame["cc_num"].astype(np.int64)
    if frame["category"].isna().any():
        rows = frame.index[frame["category"].isna()].tolist()[:10]
        raise ValueError(f"{name}: category absente aux lignes {rows}.")
    non_text_categories = ~frame["category"].map(lambda value: isinstance(value, str))
    if non_text_categories.any():
        rows = frame.index[non_text_categories].tolist()[:10]
        raise ValueError(f"{name}: category doit contenir les chaînes du notebook V2; lignes {rows}.")

    for column in NUMERIC_RAW_COLUMNS:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
        invalid = frame[column].isna() | ~np.isfinite(frame[column].to_numpy(dtype=float))
        if invalid.any():
            rows = frame.index[invalid].tolist()[:10]
            raise ValueError(f"{name}: {column} absent ou invalide aux lignes {rows}.")

    city_pop = frame["city_pop"].to_numpy(dtype=float)
    if not np.equal(city_pop, np.floor(city_pop)).all():
        raise ValueError(f"{name}: city_pop doit contenir des valeurs entières.")
    frame["amt"] = frame["amt"].astype(np.float64)
    frame["city_pop"] = frame["city_pop"].astype(np.int64)
    frame["category"] = frame["category"].astype(object)
    return frame


def haversine_distance_km(
    lat1: Sequence[float] | np.ndarray,
    lon1: Sequence[float] | np.ndarray,
    lat2: Sequence[float] | np.ndarray,
    lon2: Sequence[float] | np.ndarray,
) -> np.ndarray:
    """Distance en km, avec la formule Haversine du notebook (R = 6371 km)."""
    lat1_rad, lon1_rad, lat2_rad, lon2_rad = np.radians(
        [np.asarray(value, dtype=np.float64) for value in (lat1, lon1, lat2, lon2)]
    )
    delta_lat = lat2_rad - lat1_rad
    delta_lon = lon2_rad - lon1_rad
    a = (
        np.sin(delta_lat / 2) ** 2
        + np.cos(lat1_rad)
        * np.cos(lat2_rad)
        * np.sin(delta_lon / 2) ** 2
    )
    a = np.clip(a, 0, 1)
    return EARTH_RADIUS_KM * 2 * np.arcsin(np.sqrt(a))


def _historical_features(
    targets: pd.DataFrame, history: pd.DataFrame
) -> pd.DataFrame:
    """Calcule les sept features historiques avec les conventions V2."""
    result = pd.DataFrame(index=targets.index)
    result["time_since_last_tx"] = np.nan
    result["has_previous_tx"] = np.int64(0)
    result["tx_count_1h"] = np.int32(0)
    result["tx_count_24h"] = np.int32(0)
    result["tx_count_30d"] = np.int32(0)
    result["avg_amount_30d"] = np.nan
    result["amount_ratio_30d"] = np.nan

    target_groups = targets.groupby("cc_num", sort=False).indices
    history_groups = history.groupby("cc_num", sort=False).indices
    target_times_all = targets["trans_date_trans_time"].to_numpy(dtype="datetime64[ns]")
    history_times_all = history["trans_date_trans_time"].to_numpy(dtype="datetime64[ns]")
    history_amounts_all = history["amt"].to_numpy(dtype=np.float64)

    for card, target_positions in target_groups.items():
        history_positions = history_groups.get(card)
        if history_positions is None or len(history_positions) == 0:
            continue

        # Stable sort mirrors the notebook. searchsorted(side="left") excludes
        # all events at t, including each transaction in a simultaneous group.
        order = np.argsort(history_times_all[history_positions], kind="stable")
        positions = history_positions[order]
        times = history_times_all[positions]
        amounts = history_amounts_all[positions]
        cumulative_amounts = np.concatenate(([0.0], np.cumsum(amounts)))

        target_times = target_times_all[target_positions]
        right = np.searchsorted(times, target_times, side="left")
        left_1h = np.searchsorted(
            times, target_times - np.timedelta64(1, "h"), side="left"
        )
        left_24h = np.searchsorted(
            times, target_times - np.timedelta64(24, "h"), side="left"
        )
        left_30d = np.searchsorted(
            times, target_times - np.timedelta64(30, "D"), side="left"
        )

        previous_exists = right > 0
        previous_time = np.full(len(target_positions), np.datetime64("NaT"), dtype="datetime64[ns]")
        previous_time[previous_exists] = times[right[previous_exists] - 1]
        elapsed_minutes = (
            (target_times - previous_time) / np.timedelta64(1, "m")
        ).astype(np.float64)

        count_30d = right - left_30d
        amount_sum_30d = cumulative_amounts[right] - cumulative_amounts[left_30d]
        avg_30d = np.divide(
            amount_sum_30d,
            count_30d,
            out=np.full(len(target_positions), np.nan, dtype=np.float64),
            where=count_30d > 0,
        )
        current_amounts = targets.iloc[target_positions]["amt"].to_numpy(dtype=np.float64)
        with np.errstate(divide="ignore", invalid="ignore"):
            amount_ratio = current_amounts / avg_30d

        result.loc[target_positions, "time_since_last_tx"] = elapsed_minutes
        result.loc[target_positions, "has_previous_tx"] = previous_exists.astype(np.int64)
        result.loc[target_positions, "tx_count_1h"] = (right - left_1h).astype(np.int32)
        result.loc[target_positions, "tx_count_24h"] = (right - left_24h).astype(np.int32)
        result.loc[target_positions, "tx_count_30d"] = count_30d.astype(np.int32)
        result.loc[target_positions, "avg_amount_30d"] = avg_30d
        result.loc[target_positions, "amount_ratio_30d"] = amount_ratio

    result["has_previous_tx"] = result["has_previous_tx"].astype(np.int64)
    for column in ("tx_count_1h", "tx_count_24h", "tx_count_30d"):
        result[column] = result[column].astype(np.int32)
    return result


def build_features(
    transactions: pd.DataFrame,
    *,
    history: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Retourne exactement les 15 features V2, alignées sur l'ordre d'entrée.

    ``transactions`` et ``history`` utilisent les colonnes brutes du notebook.
    Les événements de ``history`` et du lot sont sélectionnés par horodatage
    strictement antérieur à chaque cible; les lignes simultanées sont exclues.
    Sans historique, délai, moyenne et ratio valent NaN; indicateur vaut 0 et
    compteurs valent 0, conformément au notebook.

    Le notebook ne définit pas de stratégie de déduplication : cette fonction
    conserve donc les lignes telles quelles et ne supprime aucun événement.
    """
    targets = validate_raw_transactions(transactions, name="transactions")
    if history is None:
        past = targets
    else:
        past = validate_raw_transactions(history, name="history")
        past = pd.concat([past, targets], ignore_index=True)

    times = targets["trans_date_trans_time"]
    hour = times.dt.hour
    day_of_week = times.dt.dayofweek

    time_features = pd.DataFrame(
        {
            "hour_sin": np.sin(2 * np.pi * hour / 24).astype(np.float64),
            "hour_cos": np.cos(2 * np.pi * hour / 24).astype(np.float64),
            "day_of_week": day_of_week.astype(np.int64),
            "is_weekend": (day_of_week >= 5).astype(np.int64),
        }
    )
    historical = _historical_features(targets, past)
    distance = haversine_distance_km(
        targets["lat"], targets["long"], targets["merch_lat"], targets["merch_long"]
    ).astype(np.float64)

    features = pd.DataFrame(
        {
            "amt": targets["amt"].astype(np.float64),
            "city_pop": targets["city_pop"].astype(np.int64),
            "hour_sin": time_features["hour_sin"],
            "hour_cos": time_features["hour_cos"],
            "day_of_week": time_features["day_of_week"],
            "is_weekend": time_features["is_weekend"],
            "time_since_last_tx": historical["time_since_last_tx"].astype(np.float64),
            "has_previous_tx": historical["has_previous_tx"].astype(np.int64),
            "tx_count_1h": historical["tx_count_1h"].astype(np.int32),
            "tx_count_24h": historical["tx_count_24h"].astype(np.int32),
            "tx_count_30d": historical["tx_count_30d"].astype(np.int32),
            "avg_amount_30d": historical["avg_amount_30d"].astype(np.float64),
            "amount_ratio_30d": historical["amount_ratio_30d"].astype(np.float64),
            "distance_km": distance,
            "category": targets["category"].astype(object),
        }
    )
    return features.loc[:, list(FEATURE_COLUMNS)].reset_index(drop=True)
