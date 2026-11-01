"""Experimental models are read-only and never issue operational alerts."""
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import joblib
import json
import numpy as np

from backend.features.feature_engineering import FEATURE_COLUMNS
from backend.services import experimental_scoring_service as scoring


class DummyPipeline:
    classes_ = [0, 1]
    def predict_proba(self, frame):
        assert list(frame.columns) == list(FEATURE_COLUMNS)
        return np.array([[0.13, 0.87]])


class ExperimentPreviewTest(unittest.TestCase):
    def test_experimental_score_is_preview_and_keeps_operational_baseline(self):
        with tempfile.TemporaryDirectory() as tmp:
            job_id = "20315b51-28cf-4e12-bac8-253b94851488"
            dest = Path(tmp) / job_id
            dest.mkdir()
            joblib.dump(DummyPipeline(), dest / "model.joblib")
            (dest / "config.json").write_text(json.dumps({"feature_columns":FEATURE_COLUMNS}), encoding="utf-8")
            baseline = {"features": {c: "food" if c == "category" else 1.0 for c in FEATURE_COLUMNS},
                        "fraud_score": 0.33, "decision": "fraud", "threshold": 0.22,
                        "mode": "preview", "trans_num": "check-1"}
            operations = SimpleNamespace(score=lambda transaction, mode: baseline)
            metadata = {"algorithm": "random_forest", "status": "completed", "metrics": {"threshold": 0.75}}
            with patch.object(scoring, "EXPERIMENT_DIR", Path(tmp)), \
                 patch.object(scoring, "get_scoring_service", return_value=operations), \
                 patch.object(scoring, "get_experiment", return_value=metadata):
                result = scoring.score_experimental({"trans_num":"check-1"}, job_id)
            self.assertEqual(result["mode"], "preview")
            self.assertEqual(result["decision"], "fraud")
            self.assertEqual(result["baseline_fraud_score"], 0.33)
            self.assertAlmostEqual(result["fraud_score"], 0.87)
            self.assertEqual(result["threshold"], 0.75)

    def test_path_traversal_experiment_id_rejected(self):
        with self.assertRaises(ValueError):
            scoring.score_experimental({}, "../../models/anything")


if __name__ == "__main__":
    unittest.main()
