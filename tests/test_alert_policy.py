import tempfile
import unittest
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

from backend.repositories.alert_repository import SQLiteAlertRepository


class AlertPolicyRepositoryTest(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.repository = SQLiteAlertRepository(Path(self.temp_dir.name) / "alerts.sqlite3")
        self.now = datetime(2025, 3, 4, 23, 59, tzinfo=timezone.utc)

    def tearDown(self):
        self.temp_dir.cleanup()

    def evaluate(self, trans_num, score=0.9, budget=30):
        return self.repository.evaluate(
            trans_num=trans_num, score=score, threshold=0.22266507475921693,
            daily_budget=budget, cost_per_alert=5.0, now=self.now,
        )

    def test_threshold_deduplication_and_status_history(self):
        low = self.evaluate("below", score=0.1)
        self.assertEqual(low["outcome"], "below_threshold")
        self.assertIsNone(low["alert"])

        first = self.evaluate("eligible")
        replay = self.evaluate("eligible")
        self.assertEqual(first["outcome"], "alert_created")
        self.assertEqual(first["alert"]["alert_id"], replay["alert"]["alert_id"])
        self.assertEqual(self.repository.summary(self.now.date().isoformat())["alerts_today"], 1)
        updated = self.repository.update_status(first["alert"]["alert_id"], "confirmed")
        self.assertEqual(updated["status"], "confirmed")

    def test_daily_budget_is_enforced_atomically_under_concurrent_scoring(self):
        ids = [f"tx-{i}" for i in range(12)]
        with ThreadPoolExecutor(max_workers=12) as pool:
            results = list(pool.map(lambda tx_id: self.evaluate(tx_id, budget=2), ids))
        outcomes = [item["outcome"] for item in results]
        self.assertEqual(outcomes.count("alert_created"), 2)
        self.assertEqual(outcomes.count("budget_exhausted"), 10)
        self.assertEqual(self.repository.summary(self.now.date().isoformat())["alerts_today"], 2)

    def test_late_event_consumes_budget_on_processing_utc_day(self):
        outcome = self.repository.evaluate(
            trans_num="late-tx", score=0.9, threshold=0.2, daily_budget=30,
            cost_per_alert=5, now=self.now,
        )
        self.assertEqual(outcome["alert"]["alert_date"], "2025-03-04")

    def test_invalid_status_and_unknown_alert_are_rejected(self):
        with self.assertRaises(ValueError):
            self.repository.update_status("missing", "closed")
        with self.assertRaises(FileNotFoundError):
            self.repository.update_status("missing", "rejected")


if __name__ == "__main__":
    unittest.main()
