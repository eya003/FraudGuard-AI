"""End-to-end Kaggle/backend parity test; enabled after Kaggle exports are copied."""

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from backend.features.feature_engineering import FEATURE_COLUMNS, RAW_COLUMNS
from backend.repositories.history_repository import SQLiteHistoryRepository
from backend.services.feature_service import FeatureService
from backend.services.model_service import CONFIG_PATH, load_model_artifacts


PROJECT_ROOT = Path(__file__).resolve().parents[1]
REFERENCE_DIR = PROJECT_ROOT / "data" / "kaggle_reference_v2"
HISTORY_PATH = REFERENCE_DIR / "feature_history_v2.csv"
REFERENCE_PATH = REFERENCE_DIR / "feature_reference_v2.csv"
EXPORTS_AVAILABLE = HISTORY_PATH.is_file() and REFERENCE_PATH.is_file()
FEATURE_ATOL = 1e-8
FEATURE_RTOL = 1e-7
SCORE_ATOL = 1e-12
SCORE_RTOL = 1e-10


@unittest.skipUnless(
    EXPORTS_AVAILABLE,
    "Copiez les deux exports Kaggle dans data/kaggle_reference_v2 pour activer ce test.",
)
class KaggleFeatureParityTest(unittest.TestCase):
    def test_features_and_random_forest_scores_match_kaggle(self):
        config = json.loads(Path(CONFIG_PATH).read_text(encoding="utf-8"))
        feature_columns = list(config["features"])
        self.assertEqual(feature_columns, list(FEATURE_COLUMNS))

        history = pd.read_csv(HISTORY_PATH, dtype={"trans_num": str})
        reference = pd.read_csv(REFERENCE_PATH, dtype={"trans_num": str})
        raw_columns = ["trans_num", *RAW_COLUMNS]
        self.assertFalse(reference.empty)
        self.assertTrue(set(raw_columns + feature_columns + ["expected_score", "reference_case"]).issubset(reference.columns))

        with tempfile.TemporaryDirectory(prefix="fraudguard-kaggle-parity-") as temp_dir:
            repository = SQLiteHistoryRepository(Path(temp_dir) / "parity.sqlite3")
            service = FeatureService(repository)

            if not history.empty:
                history = history.sort_values("trans_date_trans_time", kind="stable")
                service.transform_and_persist(history.loc[:, raw_columns])

            reference = reference.sort_values("trans_date_trans_time", kind="stable").reset_index(drop=True)
            result = service.transform_and_persist(reference.loc[:, raw_columns])
            self.assertEqual(result.transaction_ids, reference["trans_num"].tolist())
            actual = result.features

            self.assertEqual(len(actual), len(reference))
            self.assertEqual(list(actual.columns), feature_columns)
            self.assertEqual(actual["category"].tolist(), reference["category"].tolist())

            max_numeric_error = 0.0
            for feature in feature_columns:
                if feature == "category":
                    continue
                expected = reference[feature].to_numpy(dtype=np.float64)
                recalculated = actual[feature].to_numpy(dtype=np.float64)
                finite = np.isfinite(expected) & np.isfinite(recalculated)
                if finite.any():
                    max_numeric_error = max(
                        max_numeric_error,
                        float(np.max(np.abs(expected[finite] - recalculated[finite]))),
                    )
                np.testing.assert_allclose(
                    recalculated,
                    expected,
                    rtol=FEATURE_RTOL,
                    atol=FEATURE_ATOL,
                    equal_nan=True,
                    err_msg=f"Feature Kaggle/backend différente : {feature}",
                )
            print(f"Écart absolu maximal des features numériques : {max_numeric_error:.17g}")

            model, loaded_config = load_model_artifacts()
            self.assertEqual(feature_columns, loaded_config["features"])
            self.assertEqual(list(model.feature_names_in_), feature_columns)
            fraud_index = list(model.classes_).index(1)
            scores = model.predict_proba(actual[feature_columns])[:, fraud_index]
            expected_scores = reference["expected_score"].to_numpy(dtype=np.float64)
            max_score_error = float(np.max(np.abs(scores - expected_scores)))
            print(f"Écart absolu maximal des scores : {max_score_error:.17g}")
            np.testing.assert_allclose(
                scores,
                expected_scores,
                rtol=SCORE_RTOL,
                atol=SCORE_ATOL,
                err_msg="Les scores Random Forest locaux diffèrent des scores Kaggle.",
            )


if __name__ == "__main__":
    unittest.main()
