import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from backend.services import import_service
from backend.services.risk_simulator_service import run_simulation


class MultiModelRiskSimulationTest(unittest.TestCase):
    def test_user_can_select_real_validation_score_column(self):
        with tempfile.TemporaryDirectory() as tmp:
            with patch.object(import_service, "UPLOAD_DIR", Path(tmp)):
                item = import_service.save_csv("scores.csv", (
                    "timestamp,amt,is_fraud,split,score_rf,score_if\n"
                    "2020-01-01 00:00:00,100,1,validation,0.9,0.1\n"
                    "2020-01-01 00:01:00,10,0,validation,0.8,0.9\n"
                    "2020-01-01 00:02:00,80,1,validation,0.2,0.8\n"
                    "2020-01-01 00:03:00,20,0,validation,0.1,0.2\n"
                ).encode())
                result = run_simulation(item["id"], threshold=0.5, daily_budget=1,
                                        cost_per_alert=5, method="topk_batch",
                                        score_column="score_rf")
        self.assertEqual(result["score_column"], "score_rf")
        self.assertEqual(result["alerts"], 1)
        self.assertEqual(result["precision"], 1.0)


if __name__ == "__main__":
    unittest.main()
