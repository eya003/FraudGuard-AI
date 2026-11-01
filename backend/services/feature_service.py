"""Orchestration des features V2 avec stockage durable et événements désordonnés."""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

from backend.features.feature_engineering import (
    FEATURE_COLUMNS,
    RAW_COLUMNS,
    build_features,
    validate_raw_transactions,
)
from backend.repositories.history_repository import (
    DuplicateTransactionError,
    SQLiteHistoryRepository,
)

load_dotenv()
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_HISTORY_DB = PROJECT_ROOT / "data" / "fraudguard_history.sqlite3"


@dataclass(frozen=True)
class FeatureBatch:
    transaction_ids: list[str]
    features: pd.DataFrame


class FeatureService:
    def __init__(self, repository: SQLiteHistoryRepository | None = None) -> None:
        if repository is None:
            configured_path = Path(os.getenv("HISTORY_DB_PATH", str(DEFAULT_HISTORY_DB)))
            if not configured_path.is_absolute():
                configured_path = PROJECT_ROOT / configured_path
            repository = SQLiteHistoryRepository(configured_path)
        self.repository = repository

    @staticmethod
    def _canonical_record(transaction_id: str, row: pd.Series) -> dict:
        timestamp = pd.Timestamp(row["trans_date_trans_time"])
        record = {
            "trans_num": transaction_id,
            "trans_date_trans_time": timestamp.isoformat(sep=" ", timespec="nanoseconds"),
            "event_time_ns": int(timestamp.value),
            "cc_num": int(row["cc_num"]),
            "amt": float(row["amt"]),
            "city_pop": int(row["city_pop"]),
            "category": str(row["category"]),
            "lat": float(row["lat"]),
            "long": float(row["long"]),
            "merch_lat": float(row["merch_lat"]),
            "merch_long": float(row["merch_long"]),
        }
        payload = {key: value for key, value in record.items() if key != "trans_num"}
        record["payload_hash"] = hashlib.sha256(
            json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
        ).hexdigest()
        return record

    def transform_and_persist(self, transactions: pd.DataFrame) -> FeatureBatch:
        """Idempotently persist raw events and return their 15 V2 features.

        Same ``trans_num`` + same normalized raw payload is ignored by storage;
        reusing an ID for another payload rejects and rolls back the entire batch.
        Results are calculated from event-time history, not arrival order.
        """
        if transactions.empty:
            raise ValueError("Le lot de transactions est vide.")
        if "trans_num" not in transactions.columns:
            raise ValueError("Colonne requise pour l’idempotence : trans_num.")

        transaction_ids = transactions["trans_num"].reset_index(drop=True)
        if transaction_ids.isna().any():
            raise ValueError("trans_num ne peut pas être absent.")
        transaction_ids = transaction_ids.map(lambda value: value if isinstance(value, str) else None)
        if transaction_ids.isna().any() or transaction_ids.str.strip().eq("").any():
            raise ValueError("trans_num doit être un identifiant texte non vide.")

        normalized = validate_raw_transactions(transactions, name="transactions")
        first_positions: list[int] = []
        unique_ids: list[str] = []
        records_by_id: dict[str, dict] = {}
        for position, transaction_id in enumerate(transaction_ids.tolist()):
            record = self._canonical_record(transaction_id, normalized.iloc[position])
            previous = records_by_id.get(transaction_id)
            if previous is not None:
                if previous["payload_hash"] != record["payload_hash"]:
                    raise DuplicateTransactionError(
                        f"trans_num {transaction_id!r} apparaît avec plusieurs contenus dans le lot."
                    )
                continue
            records_by_id[transaction_id] = record
            first_positions.append(position)
            unique_ids.append(transaction_id)

        targets = normalized.iloc[first_positions].reset_index(drop=True)
        targets.insert(0, "trans_num", unique_ids)
        records = [records_by_id[transaction_id] for transaction_id in unique_ids]
        cards = sorted(set(targets["cc_num"].astype(int).tolist()))

        # Insert, read the persistent history, and calculate within one write
        # transaction so a failed calculation cannot leave a partial batch.
        with self.repository.transaction() as session:
            session.record_batch(records)
            history = session.history_for_cards(
                cards,
                exclude_transaction_ids=unique_ids,
                before_event_time_ns=int(targets["trans_date_trans_time"].max().value),
            )
            history = history.drop(columns=["trans_num"])
            features = build_features(targets, history=history)

        if tuple(features.columns) != FEATURE_COLUMNS:
            raise RuntimeError("Le Feature Engineering n’a pas retourné le schéma V2 attendu.")
        return FeatureBatch(transaction_ids=unique_ids, features=features)

    def preview(self, transactions: pd.DataFrame) -> FeatureBatch:
        """Calculate one transaction's V2 features without inserting it.

        The lookup is strictly by event time. An existing transaction ID is
        permitted only when its normalized raw payload is identical.
        """
        if len(transactions) != 1:
            raise ValueError("La prévisualisation accepte exactement une transaction.")
        if "trans_num" not in transactions.columns:
            raise ValueError("Colonne requise : trans_num.")
        transaction_id = transactions.iloc[0]["trans_num"]
        if not isinstance(transaction_id, str) or not transaction_id.strip():
            raise ValueError("trans_num doit être un identifiant texte non vide.")

        normalized = validate_raw_transactions(transactions, name="transactions")
        record = self._canonical_record(transaction_id, normalized.iloc[0])
        with self.repository.read_session() as session:
            existing_hash = session.payload_hash_for_id(transaction_id)
            if existing_hash is not None and existing_hash != record["payload_hash"]:
                raise DuplicateTransactionError(
                    f"trans_num {transaction_id!r} existe déjà avec un contenu différent."
                )
            history = session.history_before(
                int(record["cc_num"]), int(record["event_time_ns"])
            )

        targets = normalized.copy()
        targets.insert(0, "trans_num", [transaction_id])
        features = build_features(targets, history=history.drop(columns=["trans_num"]))
        if tuple(features.columns) != FEATURE_COLUMNS:
            raise RuntimeError("Le Feature Engineering n’a pas retourné le schéma V2 attendu.")
        return FeatureBatch(transaction_ids=[transaction_id], features=features)
