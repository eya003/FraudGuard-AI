import tempfile
import unittest
from pathlib import Path

from fastapi import HTTPException
import pandas as pd

from backend.main import app
from backend.repositories.history_repository import SQLiteHistoryRepository
from backend.routers.scoring import TransactionScoreRequest, score_transaction
from backend.routers.explanations import global_importance
from backend.services.feature_service import FeatureService
from backend.services.model_service import load_model_artifacts
from backend.services.scoring_service import ScoringService


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


class ScoringApiTest(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.repository = SQLiteHistoryRepository(Path(self.temp_dir.name) / "score.sqlite3")
        self.features = FeatureService(self.repository)
        self.model, self.config = load_model_artifacts()
        self.scorer = ScoringService(self.features, model=self.model, config=self.config)

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_preview_score_matches_direct_pipeline_and_does_not_persist(self):
        raw = transaction("preview-1", "2020-01-01 00:00:00", 42.5)
        before = self.repository.count()
        response = score_transaction(TransactionScoreRequest(transaction=raw), self.scorer)

        feature_batch = self.features.preview(pd.DataFrame([raw]))
        expected = self.model.predict_proba(
            feature_batch.features.loc[:, self.config["features"]]
        )[0, list(self.model.classes_).index(1)]
        self.assertEqual(response["trans_num"], "preview-1")
        self.assertAlmostEqual(response["fraud_score"], float(expected), places=15)
        self.assertEqual(response["threshold"], self.config["threshold"])
        self.assertEqual(list(response["features"]), self.config["features"])
        self.assertEqual(len(response["features"]), 15)
        self.assertEqual(
            response["decision"],
            "fraud" if expected >= self.config["threshold"] else "non_fraud",
        )
        self.assertEqual(response["mode"], "preview")
        self.assertEqual(response["mode"], "preview")
        self.assertEqual(self.repository.count(), before)

    def test_record_mode_persists_and_duplicate_replay_is_idempotent(self):
        raw = transaction("record-1", "2020-01-01 00:00:00", 42.5)
        request = TransactionScoreRequest(transaction=raw, mode="record")
        first = score_transaction(request, self.scorer)
        second = score_transaction(request, self.scorer)
        self.assertEqual(first["mode"], "record")
        self.assertAlmostEqual(first["fraud_score"], second["fraud_score"], places=15)
        self.assertEqual(self.repository.count(), 1)

    def test_changed_duplicate_is_rejected_without_persisting(self):
        first = transaction("duplicate-1", "2020-01-01 00:00:00", 10)
        score_transaction(
            TransactionScoreRequest(transaction=first, mode="record"), self.scorer
        )
        changed = transaction("duplicate-1", "2020-01-01 00:00:00", 11)
        with self.assertRaises(HTTPException) as context:
            score_transaction(TransactionScoreRequest(transaction=changed), self.scorer)
        self.assertEqual(context.exception.status_code, 400)
        self.assertEqual(self.repository.count(), 1)

    def test_preview_of_out_of_order_event_uses_only_earlier_event_time(self):
        later = transaction("later", "2020-01-01 00:20:00", 20)
        score_transaction(
            TransactionScoreRequest(transaction=later, mode="record"), self.scorer
        )
        earlier = transaction("earlier", "2020-01-01 00:10:00", 10)
        response = score_transaction(TransactionScoreRequest(transaction=earlier), self.scorer)
        self.assertEqual(response["mode"], "preview")
        self.assertEqual(self.repository.count(), 1)

    def test_route_is_registered_and_preview_is_the_default(self):
        self.assertIn("/api/v1/transactions/score", app.openapi()["paths"])
        self.assertEqual(TransactionScoreRequest(transaction={}).mode, "preview")

    def test_global_feature_importance_comes_from_operational_random_forest(self):
        result = global_importance()
        self.assertEqual(result["method"].startswith("Importance impurity Random Forest"), True)
        self.assertEqual(sum(item["feature"] == "category" for item in result["items"]), 1)
        self.assertAlmostEqual(sum(item["importance"] for item in result["items"]), 1.0, places=6)


if __name__ == "__main__":
    unittest.main()
