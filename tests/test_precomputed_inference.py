import math
import os
import numpy as np
import unittest
from unittest.mock import patch

from fastapi import HTTPException

from backend.services.demo_policy import require_local_write_mode
from backend.services.precomputed_scoring import normalize_features, score_precomputed_features
from backend.features.feature_engineering import FEATURE_COLUMNS
from frontend.feature_inputs import prepare_precomputed_features


class FakeModel:
    classes_ = [0, 1]
    def predict_proba(self, data):
        return np.asarray([[0.8, 0.2]])


class PrecomputedScoringTest(unittest.TestCase):
    def setUp(self):
        self.values = {name: 1.0 for name in FEATURE_COLUMNS}
        self.values["category"] = "shopping_net"
        self.config = {"features": list(FEATURE_COLUMNS), "threshold": 0.15}

    def test_only_complete_15_feature_rows_are_accepted(self):
        self.assertIsNone(prepare_precomputed_features({"amt": 12.0}))
        row = {**self.values, "is_fraud": 1, "cc_num": "sensitive"}
        self.assertEqual(prepare_precomputed_features(row), self.values)
        self.assertEqual(normalize_features(self.values)["category"], "shopping_net")

    def test_preview_has_no_labels_or_record_mode(self):
        result = score_precomputed_features(FakeModel(), self.config, self.values, "row-1")
        self.assertEqual(result["decision"], "fraud")
        self.assertEqual(result["mode"], "preview")
        self.assertEqual(result["feature_origin"], "provided_by_csv")
        self.assertNotIn("is_fraud", result)

    def test_missing_and_unknown_values_rejected(self):
        with self.assertRaises(ValueError):
            normalize_features({k: v for k, v in self.values.items() if k != "amt"})
        with self.assertRaises(ValueError):
            normalize_features({**self.values, "is_fraud": 1})
        with self.assertRaises(ValueError):
            normalize_features({**self.values, "amt": float("inf")})

    def test_nan_missing_numerics_allowed(self):
        row = normalize_features({**self.values, "avg_amount_30d": None})
        self.assertTrue(math.isnan(row["avg_amount_30d"]))

    def test_public_demo_denies_mutations(self):
        with patch.dict(os.environ, {"PUBLIC_DEMO_MODE": "true"}):
            with self.assertRaises(HTTPException) as context:
                require_local_write_mode()
        self.assertEqual(context.exception.status_code, 403)


if __name__ == "__main__":
    unittest.main()
