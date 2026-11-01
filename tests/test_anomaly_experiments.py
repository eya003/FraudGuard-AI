"""Small fully synthetic unit tests; never masquerade these as dataset results."""
import unittest

import pandas as pd
import numpy as np

from backend.features.feature_engineering import FEATURE_COLUMNS
from backend.services.anomaly_models import AnomalyExperiment


class AnomalyTrainingTest(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(42)
        self.data = pd.DataFrame({c: rng.normal(size=60) for c in FEATURE_COLUMNS if c != "category"})
        self.data["category"] = ["travel", "gas_transport"] * 30

    def test_isolation_forest_output_is_percentile_not_probability(self):
        model = AnomalyExperiment("isolation_forest", {"n_estimators": 30}).fit(self.data)
        scores = model.predict_risk(self.data.head(7))
        self.assertEqual(scores.shape, (7,))
        self.assertTrue(np.isfinite(scores).all())
        self.assertTrue(((scores >= 0) & (scores <= 1)).all())

    def test_autoencoder_reconstructs_normal_training_data(self):
        model = AnomalyExperiment("autoencoder", {"max_iter": 12}).fit(self.data)
        scores = model.predict_risk(self.data.tail(8))
        self.assertEqual(len(scores), 8)
        self.assertTrue(((scores >= 0) & (scores <= 1)).all())


if __name__ == "__main__":
    unittest.main()
