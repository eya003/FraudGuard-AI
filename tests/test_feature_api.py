import tempfile
import unittest
from pathlib import Path

from fastapi import HTTPException

from backend.repositories.history_repository import SQLiteHistoryRepository
from backend.routers.features import FeatureEngineeringRequest, engineer_features
from backend.services.feature_service import FeatureService


class FeatureApiTest(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        repository = SQLiteHistoryRepository(Path(self.temp_dir.name) / "api.sqlite3")
        self.service = FeatureService(repository)

    def tearDown(self):
        self.temp_dir.cleanup()

    @staticmethod
    def transaction(transaction_id, timestamp, amount):
        return {
            "trans_num": transaction_id,
            "trans_date_trans_time": timestamp,
            "cc_num": 1001,
            "amt": amount,
            "city_pop": 1000,
            "category": "shopping_pos",
            "lat": 36.0,
            "long": -80.0,
            "merch_lat": 36.1,
            "merch_long": -80.1,
        }

    def test_engineer_route_returns_15_features_and_persists_history(self):
        first = self.transaction("api-1", "2020-01-01 00:00:00", 10)
        payload = engineer_features(
            FeatureEngineeringRequest(transactions=[first]), self.service
        )
        self.assertEqual(payload["count"], 1)
        self.assertEqual(len(payload["feature_columns"]), 15)
        self.assertIsNone(payload["items"][0]["time_since_last_tx"])

        second = self.transaction("api-2", "2020-01-01 00:10:00", 20)
        payload = engineer_features(
            FeatureEngineeringRequest(transactions=[second]), self.service
        )
        item = payload["items"][0]
        self.assertEqual(item["time_since_last_tx"], 10.0)
        self.assertEqual(item["tx_count_1h"], 1)

    def test_api_rejects_reused_id_with_changed_payload(self):
        original = self.transaction("api-id", "2020-01-01 00:00:00", 10)
        changed = self.transaction("api-id", "2020-01-01 00:00:00", 11)
        engineer_features(FeatureEngineeringRequest(transactions=[original]), self.service)
        with self.assertRaises(HTTPException) as context:
            engineer_features(
                FeatureEngineeringRequest(transactions=[changed]), self.service
            )
        self.assertEqual(context.exception.status_code, 400)
        self.assertIn("contenu différent", context.exception.detail)


if __name__ == "__main__":
    unittest.main()
