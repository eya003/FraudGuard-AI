"""SQLite alert decisions with atomic daily-budget enforcement and deduplication."""

from __future__ import annotations

import sqlite3
import uuid
from contextlib import closing
from datetime import datetime, timezone
from pathlib import Path

ALERT_STATUSES = ("new", "in_progress", "confirmed", "rejected")


class SQLiteAlertRepository:
    def __init__(self, database_path: str | Path) -> None:
        self.database_path = Path(database_path)
        with closing(self._connect()) as connection:
            connection.executescript(
                """CREATE TABLE IF NOT EXISTS alerts (
                    alert_id TEXT PRIMARY KEY,
                    trans_num TEXT NOT NULL UNIQUE,
                    score REAL NOT NULL,
                    threshold REAL NOT NULL,
                    alert_date TEXT NOT NULL,
                    cost REAL NOT NULL,
                    status TEXT NOT NULL CHECK(status IN
                        ('new','in_progress','confirmed','rejected')),
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_alerts_date ON alerts(alert_date);
                CREATE TABLE IF NOT EXISTS alert_decisions (
                    trans_num TEXT PRIMARY KEY,
                    outcome TEXT NOT NULL,
                    alert_id TEXT,
                    score REAL NOT NULL,
                    threshold REAL NOT NULL,
                    evaluated_at TEXT NOT NULL,
                    alert_date TEXT NOT NULL
                );"""
            )
            connection.commit()

    def _connect(self) -> sqlite3.Connection:
        return sqlite3.connect(str(self.database_path), timeout=30)

    def evaluate(
        self, *, trans_num: str, score: float, threshold: float,
        daily_budget: int, cost_per_alert: float, now: datetime | None = None,
    ) -> dict:
        instant = (now or datetime.now(timezone.utc)).astimezone(timezone.utc)
        today = instant.date().isoformat()
        timestamp = instant.isoformat()
        connection = self._connect()
        try:
            connection.execute("BEGIN IMMEDIATE")
            previous = connection.execute(
                "SELECT outcome, alert_id FROM alert_decisions WHERE trans_num = ?",
                (trans_num,),
            ).fetchone()
            if previous:
                outcome, alert_id = previous
            else:
                alert_id = None
                if score < threshold:
                    outcome = "below_threshold"
                else:
                    used = int(connection.execute(
                        "SELECT COUNT(*) FROM alerts WHERE alert_date = ?", (today,)
                    ).fetchone()[0])
                    if used >= daily_budget:
                        outcome = "budget_exhausted"
                    else:
                        outcome = "alert_created"
                        alert_id = str(uuid.uuid4())
                        connection.execute(
                            """INSERT INTO alerts (
                                alert_id, trans_num, score, threshold, alert_date, cost,
                                status, created_at, updated_at
                            ) VALUES (?, ?, ?, ?, ?, ?, 'new', ?, ?)""",
                            (alert_id, trans_num, score, threshold, today,
                             cost_per_alert, timestamp, timestamp),
                        )
                connection.execute(
                    """INSERT INTO alert_decisions (
                        trans_num, outcome, alert_id, score, threshold, evaluated_at, alert_date
                    ) VALUES (?, ?, ?, ?, ?, ?, ?)""",
                    (trans_num, outcome, alert_id, score, threshold, timestamp, today),
                )
            row = None
            if alert_id:
                row = connection.execute(
                    """SELECT alert_id, trans_num, score, threshold, alert_date,
                              cost, status, created_at, updated_at
                       FROM alerts WHERE alert_id = ?""",
                    (alert_id,),
                ).fetchone()
            connection.commit()
            return {"outcome": outcome, "alert": self._alert_dict(row) if row else None}
        except Exception:
            connection.rollback()
            raise
        finally:
            connection.close()

    @staticmethod
    def _alert_dict(row) -> dict | None:
        if row is None:
            return None
        keys = ("alert_id", "trans_num", "score", "threshold", "alert_date",
                "cost", "status", "created_at", "updated_at")
        return dict(zip(keys, row, strict=True))

    def list_alerts(self, *, status: str = "", alert_date: str = "",
                    offset: int = 0, limit: int = 100) -> dict:
        conditions, params = [], []
        if status:
            if status not in ALERT_STATUSES:
                raise ValueError(f"Statut invalide; valeurs : {', '.join(ALERT_STATUSES)}.")
            conditions.append("status = ?")
            params.append(status)
        if alert_date:
            conditions.append("alert_date = ?")
            params.append(alert_date)
        where = " WHERE " + " AND ".join(conditions) if conditions else ""
        with closing(self._connect()) as connection:
            total = int(connection.execute(f"SELECT COUNT(*) FROM alerts{where}", params).fetchone()[0])
            rows = connection.execute(
                f"""SELECT alert_id, trans_num, score, threshold, alert_date,
                           cost, status, created_at, updated_at
                    FROM alerts{where} ORDER BY created_at DESC LIMIT ? OFFSET ?""",
                [*params, limit, offset],
            ).fetchall()
        return {"total": total, "items": [self._alert_dict(row) for row in rows]}

    def update_status(self, alert_id: str, status: str) -> dict:
        if status not in ALERT_STATUSES:
            raise ValueError(f"Statut invalide; valeurs : {', '.join(ALERT_STATUSES)}.")
        now = datetime.now(timezone.utc).isoformat()
        with closing(self._connect()) as connection:
            connection.execute("BEGIN IMMEDIATE")
            cursor = connection.execute(
                "UPDATE alerts SET status = ?, updated_at = ? WHERE alert_id = ?",
                (status, now, alert_id),
            )
            if cursor.rowcount != 1:
                connection.rollback()
                raise FileNotFoundError("Alerte introuvable.")
            row = connection.execute(
                """SELECT alert_id, trans_num, score, threshold, alert_date,
                          cost, status, created_at, updated_at
                   FROM alerts WHERE alert_id = ?""",
                (alert_id,),
            ).fetchone()
            connection.commit()
        return self._alert_dict(row)

    def summary(self, alert_date: str = "") -> dict:
        day = alert_date or datetime.now(timezone.utc).date().isoformat()
        with closing(self._connect()) as connection:
            used = int(connection.execute(
                "SELECT COUNT(*) FROM alerts WHERE alert_date = ?", (day,)
            ).fetchone()[0])
            total = int(connection.execute("SELECT COUNT(*) FROM alerts").fetchone()[0])
            outcomes = {
                row[0]: int(row[1]) for row in connection.execute(
                    "SELECT outcome, COUNT(*) FROM alert_decisions GROUP BY outcome"
                ).fetchall()
            }
        return {"date_utc": day, "alerts_today": used, "total_alerts": total,
                "outcomes": outcomes}
