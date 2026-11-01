"""Historique SQLite persistant et idempotent par identifiant de transaction."""

from __future__ import annotations

import sqlite3
from contextlib import closing, contextmanager
from pathlib import Path
from typing import Iterator

import pandas as pd

from backend.features.feature_engineering import RAW_COLUMNS


class DuplicateTransactionError(ValueError):
    """Un identifiant déjà utilisé est présenté avec un autre contenu."""


class SQLiteHistorySession:
    def __init__(self, connection: sqlite3.Connection) -> None:
        self.connection = connection

    def record_batch(self, records: list[dict]) -> None:
        """Insère les nouveaux IDs et refuse les IDs existants au contenu différent."""
        seen: dict[str, str] = {}
        for record in records:
            transaction_id = record["trans_num"]
            payload_hash = record["payload_hash"]
            previous_hash = seen.get(transaction_id)
            if previous_hash is not None:
                if previous_hash != payload_hash:
                    raise DuplicateTransactionError(
                        f"trans_num {transaction_id!r} apparaît avec plusieurs contenus dans le lot."
                    )
                continue
            seen[transaction_id] = payload_hash

            existing = self.connection.execute(
                "SELECT payload_hash FROM transactions WHERE trans_num = ?",
                (transaction_id,),
            ).fetchone()
            if existing is not None:
                if existing[0] != payload_hash:
                    raise DuplicateTransactionError(
                        f"trans_num {transaction_id!r} existe déjà avec un contenu différent."
                    )
                continue

            self.connection.execute(
                """INSERT INTO transactions (
                    trans_num, trans_date_trans_time, event_time_ns, cc_num,
                    amt, city_pop, category, lat, long, merch_lat, merch_long,
                    payload_hash
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    transaction_id,
                    record["trans_date_trans_time"],
                    record["event_time_ns"],
                    record["cc_num"],
                    record["amt"],
                    record["city_pop"],
                    record["category"],
                    record["lat"],
                    record["long"],
                    record["merch_lat"],
                    record["merch_long"],
                    payload_hash,
                ),
            )

    def history_for_cards(
        self,
        card_numbers: list[int],
        *,
        exclude_transaction_ids: list[str],
        before_event_time_ns: int,
    ) -> pd.DataFrame:
        columns = ["trans_num", *RAW_COLUMNS]
        if not card_numbers:
            return pd.DataFrame(columns=columns)
        self.connection.execute(
            "CREATE TEMP TABLE IF NOT EXISTS batch_target_ids "
            "(trans_num TEXT PRIMARY KEY)"
        )
        self.connection.execute("DELETE FROM batch_target_ids")
        self.connection.executemany(
            "INSERT OR IGNORE INTO batch_target_ids (trans_num) VALUES (?)",
            [(transaction_id,) for transaction_id in exclude_transaction_ids],
        )
        rows: list[tuple] = []
        # Stay below SQLite's traditional 999 bind-variable limit.
        for start in range(0, len(card_numbers), 800):
            group = card_numbers[start : start + 800]
            placeholders = ",".join("?" for _ in group)
            rows.extend(
                self.connection.execute(
                    f"""SELECT trans_num, trans_date_trans_time, cc_num, amt, city_pop,
                               category, lat, long, merch_lat, merch_long
                        FROM transactions
                        WHERE cc_num IN ({placeholders})
                          AND event_time_ns < ?
                          AND NOT EXISTS (
                              SELECT 1 FROM batch_target_ids AS target
                              WHERE target.trans_num = transactions.trans_num
                          )
                        ORDER BY event_time_ns""",
                    [*group, before_event_time_ns],
                ).fetchall()
            )
        history = pd.DataFrame(rows, columns=columns)
        if not history.empty:
            history["trans_date_trans_time"] = pd.to_datetime(
                history["trans_date_trans_time"], errors="raise"
            )
        return history

    def payload_hash_for_id(self, transaction_id: str) -> str | None:
        row = self.connection.execute(
            "SELECT payload_hash FROM transactions WHERE trans_num = ?", (transaction_id,)
        ).fetchone()
        return None if row is None else str(row[0])

    def history_before(self, card_number: int, event_time_ns: int) -> pd.DataFrame:
        columns = ["trans_num", *RAW_COLUMNS]
        rows = self.connection.execute(
            """SELECT trans_num, trans_date_trans_time, cc_num, amt, city_pop,
                      category, lat, long, merch_lat, merch_long
               FROM transactions WHERE cc_num = ? AND event_time_ns < ?
               ORDER BY event_time_ns""",
            (card_number, event_time_ns),
        ).fetchall()
        history = pd.DataFrame(rows, columns=columns)
        if not history.empty:
            history["trans_date_trans_time"] = pd.to_datetime(
                history["trans_date_trans_time"], errors="raise"
            )
        return history

class SQLiteHistoryRepository:
    def __init__(self, database_path: str | Path) -> None:
        self.database_path = Path(database_path)
        if str(self.database_path) != ":memory:":
            self.database_path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        return sqlite3.connect(str(self.database_path), timeout=30)

    def _initialize(self) -> None:
        with closing(self._connect()) as connection:
            connection.execute(
                """CREATE TABLE IF NOT EXISTS transactions (
                    trans_num TEXT PRIMARY KEY,
                    trans_date_trans_time TEXT NOT NULL,
                    event_time_ns INTEGER NOT NULL,
                    cc_num INTEGER NOT NULL,
                    amt REAL NOT NULL,
                    city_pop INTEGER NOT NULL,
                    category TEXT NOT NULL,
                    lat REAL NOT NULL,
                    long REAL NOT NULL,
                    merch_lat REAL NOT NULL,
                    merch_long REAL NOT NULL,
                    payload_hash TEXT NOT NULL
                )"""
            )
            connection.execute(
                "CREATE INDEX IF NOT EXISTS idx_transactions_card_time "
                "ON transactions (cc_num, event_time_ns)"
            )
            connection.commit()

    @contextmanager
    def transaction(self) -> Iterator[SQLiteHistorySession]:
        connection = self._connect()
        try:
            connection.execute("BEGIN IMMEDIATE")
            yield SQLiteHistorySession(connection)
            connection.commit()
        except Exception:
            connection.rollback()
            raise
        finally:
            connection.close()

    @contextmanager
    def read_session(self) -> Iterator[SQLiteHistorySession]:
        """Expose SELECT-only helpers without opening a write transaction."""
        connection = self._connect()
        try:
            yield SQLiteHistorySession(connection)
        finally:
            connection.close()

    def count(self) -> int:
        with closing(self._connect()) as connection:
            return int(connection.execute("SELECT COUNT(*) FROM transactions").fetchone()[0])
