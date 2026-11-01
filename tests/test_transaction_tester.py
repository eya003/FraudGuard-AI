import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from backend.main import app
from backend.services import import_service
from frontend import api_client
from frontend.transaction_tester import prepare_csv_transaction, response_view


class TransactionTesterHelpersTest(unittest.TestCase):
    def test_csv_row_is_reduced_to_api_raw_schema_and_never_uses_labels(self):
        source = {
            "trans_date_trans_time": "2020-01-01 00:00:00",
            "cc_num": 123,
            "amount": 12.5,
            "city_pop": 100,
            "category": "shopping_pos",
            "lat": 1.0,
            "long": 2.0,
            "merch_lat": 3.0,
            "merch_long": 4.0,
            "is_fraud": 1,
            "irrelevant": "ignored",
        }
        prepared = prepare_csv_transaction(source, "some-import", 8)
        self.assertEqual(prepared["trans_num"], "some-import-8")
        self.assertEqual(prepared["amt"], 12.5)
        self.assertNotIn("is_fraud", prepared)
        self.assertNotIn("amount", prepared)
        self.assertEqual(len(prepared), 10)

    def test_csv_row_requires_all_raw_features_inputs(self):
        with self.assertRaisesRegex(ValueError, "trans_date_trans_time"):
            prepare_csv_transaction({"amt": 5}, "import", 0)

    def test_response_view_preserves_api_score_threshold_decision_and_features(self):
        api_response = {
            "trans_num": "real-transaction",
            "fraud_score": 0.3141592653589793,
            "threshold": 0.22266507475921693,
            "decision": "fraud",
            "mode": "preview",
            "features": {"amt": 125.5, "category": "shopping_pos"},
            "unexpected": "not displayed",
        }
        self.assertEqual(
            response_view(api_response),
            {key: api_response[key] for key in (
                "trans_num", "fraud_score", "threshold", "decision", "mode", "features"
            )},
        )


class TransactionTesterApiClientTest(unittest.TestCase):
    def test_client_posts_selected_mode_and_returns_actual_api_json(self):
        transaction = {"trans_num": "manual-1", "amt": 10}
        actual_response = {
            "trans_num": "manual-1", "fraud_score": 0.123456789,
            "threshold": 0.22266507475921693, "decision": "non_fraud",
            "mode": "preview", "features": {"amt": 10},
        }
        mocked = Mock(status_code=200, is_error=False)
        mocked.json.return_value = actual_response
        with patch.object(api_client.requests, "post", return_value=mocked) as post:
            shown_source = api_client.score_transaction(transaction, mode="preview")
        self.assertIs(shown_source, actual_response)
        self.assertEqual(post.call_args.kwargs["json"], {
            "transaction": transaction, "mode": "preview"
        })

    def test_client_surfaces_validation_detail_from_fastapi(self):
        mocked = Mock(status_code=400, is_error=True)
        mocked.json.return_value = {"detail": "champs invalides"}
        with patch.object(api_client.requests, "post", return_value=mocked):
            with self.assertRaisesRegex(ValueError, "HTTP 400.*champs invalides"):
                api_client.score_transaction({"trans_num": "bad"})


class ImportedCsvRowApiTest(unittest.TestCase):
    def test_read_imported_csv_row_by_zero_based_index(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            directory = Path(temp_dir)
            import_id = "12345678-1234-5678-1234-567812345678"
            (directory / f"{import_id}.json").write_text(
                json.dumps({"row_count": 2}), encoding="utf-8"
            )
            (directory / f"{import_id}_transactions.csv").write_text(
                "trans_num,amt,is_fraud\na,10,0\nb,20,1\n", encoding="utf-8"
            )
            with patch.object(import_service, "UPLOAD_DIR", directory):
                result = import_service.read_import_row(import_id, 1)
            self.assertEqual(result["trans_num"], "b")
            self.assertEqual(result["amt"], 20)
            self.assertEqual(result["is_fraud"], 1)

    def test_row_endpoint_and_transaction_tester_route_are_registered(self):
        paths = app.openapi()["paths"]
        self.assertIn("/api/v1/imports/{import_id}/rows/{row_index}", paths)
        self.assertIn("/api/v1/transactions/score", paths)


if __name__ == "__main__":
    unittest.main()
