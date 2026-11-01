import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from backend.repositories.history_repository import (
    DuplicateTransactionError,
    SQLiteHistoryRepository,
)
from backend.services.feature_service import FeatureService


def transaction(transaction_id, timestamp, amount, card=1001):
    return {
        "trans_num": transaction_id,
        "trans_date_trans_time": timestamp,
        "cc_num": card,
        "amt": amount,
        "city_pop": 1000,
        "category": "shopping_pos",
        "lat": 36.0,
        "long": -80.0,
        "merch_lat": 36.1,
        "merch_long": -80.1,
    }


class FeatureServicePersistenceTest(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.database_path = Path(self.temp_dir.name) / "history.sqlite3"
        self.repository = SQLiteHistoryRepository(self.database_path)
        self.service = FeatureService(self.repository)

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_out_of_order_events_use_event_time_and_history_survives_restart(self):
        later = transaction("tx-later", "2020-01-01 00:20:00", 20)
        later_result = self.service.transform_and_persist(pd.DataFrame([later]))
        self.assertTrue(np.isnan(later_result.features.loc[0, "time_since_last_tx"]))

        earlier = transaction("tx-earlier", "2020-01-01 00:00:00", 10)
        earlier_result = self.service.transform_and_persist(pd.DataFrame([earlier]))
        self.assertTrue(np.isnan(earlier_result.features.loc[0, "time_since_last_tx"]))

        restarted = FeatureService(SQLiteHistoryRepository(self.database_path))
        current = transaction("tx-current", "2020-01-01 00:30:00", 30)
        result = restarted.transform_and_persist(pd.DataFrame([current]))
        row = result.features.iloc[0]
        self.assertEqual(row["time_since_last_tx"], 10.0)
        self.assertEqual(row["tx_count_1h"], 2)
        self.assertEqual(row["avg_amount_30d"], 15.0)
        self.assertEqual(restarted.repository.count(), 3)

    def test_identical_replay_is_idempotent_but_changed_payload_rolls_back(self):
        first = transaction("stable-id", "2020-01-01 00:00:00", 10)
        duplicate_batch = pd.DataFrame([first, first])
        result = self.service.transform_and_persist(duplicate_batch)
        self.assertEqual(result.transaction_ids, ["stable-id"])
        self.assertEqual(len(result.features), 1)
        self.assertEqual(self.repository.count(), 1)

        self.service.transform_and_persist(pd.DataFrame([first]))
        self.assertEqual(self.repository.count(), 1)

        changed = transaction("stable-id", "2020-01-01 00:00:00", 11)
        another = transaction("new-id", "2020-01-01 00:05:00", 12)
        with self.assertRaises(DuplicateTransactionError):
            self.service.transform_and_persist(pd.DataFrame([another, changed]))
        self.assertEqual(self.repository.count(), 1)

    def test_distinct_simultaneous_transactions_are_both_stored_but_not_each_others_history(self):
        same_time = pd.DataFrame(
            [
                transaction("sim-a", "2020-01-01 00:00:00", 10),
                transaction("sim-b", "2020-01-01 00:00:00", 20),
            ]
        )
        result = self.service.transform_and_persist(same_time)
        self.assertEqual(len(result.features), 2)
        self.assertEqual(result.features["has_previous_tx"].tolist(), [0, 0])
        later = transaction("sim-later", "2020-01-01 00:10:00", 30)
        later_result = self.service.transform_and_persist(pd.DataFrame([later]))
        self.assertEqual(later_result.features.loc[0, "tx_count_1h"], 2)

    def test_transaction_id_is_required(self):
        with self.assertRaisesRegex(ValueError, "trans_num"):
            self.service.transform_and_persist(
                pd.DataFrame([transaction("id", "2020-01-01", 10)]).drop(columns="trans_num")
            )


if __name__ == "__main__":
    unittest.main()
