import json
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from backend.features.feature_engineering import FEATURE_COLUMNS, build_features
from backend.services.model_service import CONFIG_PATH


def raw_row(timestamp, card=1001, amount=10.0, **overrides):
    row = {
        "trans_date_trans_time": timestamp,
        "cc_num": card,
        "amt": amount,
        "city_pop": 1000,
        "category": "shopping_pos",
        "lat": 36.0,
        "long": -80.0,
        "merch_lat": 36.0,
        "merch_long": -80.0,
    }
    row.update(overrides)
    return row


class FeatureEngineeringTest(unittest.TestCase):
    def test_exact_feature_schema_and_notebook_types(self):
        data = pd.DataFrame([raw_row("2020-01-04 13:30:00")])
        features = build_features(data)
        self.assertEqual(tuple(features.columns), FEATURE_COLUMNS)
        self.assertEqual(len(features.columns), 15)
        config = json.loads(Path(CONFIG_PATH).read_text(encoding="utf-8"))
        self.assertEqual(list(FEATURE_COLUMNS), config["features"])
        expected_dtypes = {
            "amt": np.dtype("float64"),
            "city_pop": np.dtype("int64"),
            "hour_sin": np.dtype("float64"),
            "hour_cos": np.dtype("float64"),
            "day_of_week": np.dtype("int64"),
            "is_weekend": np.dtype("int64"),
            "time_since_last_tx": np.dtype("float64"),
            "has_previous_tx": np.dtype("int64"),
            "tx_count_1h": np.dtype("int32"),
            "tx_count_24h": np.dtype("int32"),
            "tx_count_30d": np.dtype("int32"),
            "avg_amount_30d": np.dtype("float64"),
            "amount_ratio_30d": np.dtype("float64"),
            "distance_km": np.dtype("float64"),
            "category": np.dtype("object"),
        }
        self.assertEqual(features.dtypes.to_dict(), expected_dtypes)
        self.assertEqual(features.loc[0, "day_of_week"], 5)
        self.assertEqual(features.loc[0, "is_weekend"], 1)
        self.assertAlmostEqual(features.loc[0, "hour_sin"], np.sin(2 * np.pi * 13 / 24))
        self.assertAlmostEqual(features.loc[0, "hour_cos"], np.cos(2 * np.pi * 13 / 24))

    def test_first_purchase_has_notebook_missing_history_values(self):
        features = build_features(pd.DataFrame([raw_row("2020-01-01 00:00:00")]))
        row = features.iloc[0]
        self.assertEqual(row["has_previous_tx"], 0)
        self.assertEqual(row["tx_count_1h"], 0)
        self.assertEqual(row["tx_count_24h"], 0)
        self.assertEqual(row["tx_count_30d"], 0)
        self.assertTrue(np.isnan(row["time_since_last_tx"]))
        self.assertTrue(np.isnan(row["avg_amount_30d"]))
        self.assertTrue(np.isnan(row["amount_ratio_30d"]))

    def test_history_is_card_specific_and_windows_include_lower_bound(self):
        rows = [
            raw_row("2020-01-01 00:00:00", amount=10),
            raw_row("2020-01-01 00:30:00", amount=20),  # exactly 30 days before target
            raw_row("2020-01-02 01:30:00", amount=30),
            raw_row("2020-01-30 00:30:00", amount=40),  # exactly 24 hours before target
            raw_row("2020-01-30 23:30:00", amount=50),  # exactly 1 hour before target
            raw_row("2020-01-31 00:30:00", amount=60),
            raw_row("2020-01-31 00:30:00", card=2002, amount=7),
        ]
        # Intentionally unordered: features remain tied to their input rows.
        order = [5, 1, 6, 3, 0, 4, 2]
        data = pd.DataFrame([rows[i] for i in order])
        features = build_features(data)
        last_card_a = features.iloc[0]
        isolated_card_b = features.iloc[2]

        self.assertEqual(last_card_a["tx_count_1h"], 1)
        self.assertEqual(last_card_a["tx_count_24h"], 2)
        self.assertEqual(last_card_a["tx_count_30d"], 4)
        self.assertAlmostEqual(last_card_a["avg_amount_30d"], 35.0)
        self.assertAlmostEqual(last_card_a["amount_ratio_30d"], 60.0 / 35.0)
        self.assertEqual(isolated_card_b["has_previous_tx"], 0)

    def test_simultaneous_transactions_do_not_see_each_other(self):
        data = pd.DataFrame(
            [
                raw_row("2020-01-01 00:00:00", amount=10),
                raw_row("2020-01-01 00:00:00", amount=20),
                raw_row("2020-01-01 00:10:00", amount=30),
            ]
        )
        features = build_features(data)
        for index in (0, 1):
            self.assertEqual(features.loc[index, "has_previous_tx"], 0)
            self.assertEqual(features.loc[index, "tx_count_1h"], 0)
        self.assertEqual(features.loc[2, "tx_count_1h"], 2)
        self.assertAlmostEqual(features.loc[2, "avg_amount_30d"], 15.0)

    def test_external_history_is_used_only_when_strictly_earlier(self):
        targets = pd.DataFrame(
            [raw_row("2020-01-01 01:00:00", amount=20)]
        )
        history = pd.DataFrame(
            [
                raw_row("2020-01-01 00:00:00", amount=10),
                raw_row("2020-01-01 01:00:00", amount=99),
                raw_row("2020-01-01 02:00:00", amount=88),
            ]
        )
        features = build_features(targets, history=history)
        row = features.iloc[0]
        self.assertEqual(row["time_since_last_tx"], 60.0)
        self.assertEqual(row["tx_count_1h"], 1)
        self.assertEqual(row["tx_count_24h"], 1)
        self.assertEqual(row["tx_count_30d"], 1)
        self.assertEqual(row["avg_amount_30d"], 10.0)
        self.assertEqual(row["amount_ratio_30d"], 2.0)

    def test_missing_raw_values_and_timezones_are_rejected(self):
        missing_amount = pd.DataFrame([raw_row("2020-01-01", amount=np.nan)])
        with self.assertRaisesRegex(ValueError, "amt absent ou invalide"):
            build_features(missing_amount)
        aware_time = pd.DataFrame([raw_row("2020-01-01T00:00:00Z")])
        with self.assertRaisesRegex(ValueError, "fuseau"):
            build_features(aware_time)

    def test_haversine_distance_matches_notebook_example(self):
        data = pd.DataFrame(
            [
                raw_row(
                    "2020-01-01",
                    lat=36.08,
                    long=-81.18,
                    merch_lat=36.01,
                    merch_long=-82.05,
                )
            ]
        )
        distance = build_features(data).loc[0, "distance_km"]
        self.assertAlmostEqual(distance, 78.60, places=1)


if __name__ == "__main__":
    unittest.main()
