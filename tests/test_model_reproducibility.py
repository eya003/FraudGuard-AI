import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from backend.services.model_service import load_model_artifacts


PROJECT_ROOT = Path(__file__).resolve().parents[1]
REFERENCE_PATH = PROJECT_ROOT / "data" / "reference_predictions_v2.csv"
EXPECTED_SCORE_COLUMN = "expected_score"
ABSOLUTE_TOLERANCE = 1e-10
RELATIVE_TOLERANCE = 1e-7


@unittest.skipUnless(
    REFERENCE_PATH.is_file(),
    "Les 100 scores Kaggle de référence sont un export local non publié ; test ignoré en CI publique.",
)
class ModelReproducibilityTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.model, cls.config = load_model_artifacts()
        cls.features = cls.config["features"]
        cls.reference = pd.read_csv(REFERENCE_PATH)

    def test_reference_schema_matches_pipeline_feature_order(self) -> None:
        self.assertEqual(len(self.features), 15)
        self.assertEqual(list(self.model.feature_names_in_), self.features)
        self.assertEqual(
            self.reference.columns.tolist(),
            self.features + [EXPECTED_SCORE_COLUMN],
        )
        self.assertEqual(len(self.reference), 100)

    def test_local_fraud_scores_match_kaggle_reference(self) -> None:
        fraud_class_index = list(self.model.classes_).index(1)
        actual_scores = self.model.predict_proba(self.reference[self.features])[
            :, fraud_class_index
        ]
        expected_scores = self.reference[EXPECTED_SCORE_COLUMN].to_numpy(dtype=float)
        max_absolute_error = float(np.max(np.abs(actual_scores - expected_scores)))

        self.assertTrue(
            np.isfinite(expected_scores).all(),
            "expected_score contient une valeur non finie.",
        )
        np.testing.assert_allclose(
            actual_scores,
            expected_scores,
            rtol=RELATIVE_TOLERANCE,
            atol=ABSOLUTE_TOLERANCE,
            err_msg=f"Écart absolu maximal constaté : {max_absolute_error:.17g}",
        )


if __name__ == "__main__":
    unittest.main()
