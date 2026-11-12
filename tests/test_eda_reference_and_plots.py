import copy
import unittest

from frontend.eda_reference import load_reference, validate_reference
from frontend.eda_visualizations import classes_chart, amount_chart, records_chart


class EdaReferenceTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data = load_reference()

    def test_global_and_train_partition_counts(self):
        meta = self.data["metadata"]
        self.assertEqual(meta["global_statistics"]["transactions"], 1_852_394)
        self.assertEqual(meta["global_statistics"]["frauds"], 9_651)
        self.assertEqual(meta["train_statistics"]["fraud"], 6_401)
        self.assertEqual(meta["train_statistics"]["normal"], 1_105_035)

    def test_partition_dates(self):
        periods = self.data["metadata"]["partition_dates"]
        self.assertEqual(periods["train"]["end"], "2020-04-07 07:00:14")
        self.assertEqual(periods["test"]["end"], "2020-12-31 23:59:34")

    def test_all_group_totals_and_fraud_rates(self):
        for key in ("monthly", "hourly", "weekday", "category", "channel"):
            rows = self.data["eda_train"][key]
            self.assertEqual(sum(r["transactions"] for r in rows), 1_111_436)
            self.assertEqual(sum(r["frauds"] for r in rows), 6_401)
            for r in rows:
                self.assertAlmostEqual(r["fraud_rate_percent"], 100 * r["frauds"] / r["transactions"], places=9)

    def test_malformed_group_refused(self):
        invalid = copy.deepcopy(self.data)
        invalid["eda_train"]["hourly"][0]["frauds"] += 1
        with self.assertRaisesRegex(ValueError, "incohérent"):
            validate_reference(invalid)

    def test_histogram_density_and_bins(self):
        chart = amount_chart(self.data)
        self.assertEqual(len(chart.data), 2)
        self.assertEqual(len(chart.data[0].y), 99)
        self.assertEqual(len(chart.data[1].y), 99)
        edges = self.data["eda_train"]["amount_histogram"]["bin_edges"]
        for kind in ("fraud", "normal"):
            densities = self.data["eda_train"]["amount_histogram"]["classes"][kind]["density"]
            self.assertAlmostEqual(
                sum((edges[i + 1] - edges[i]) * v for i, v in enumerate(densities)),
                1.0, places=5,
            )

    def test_class_chart_uses_log_axis(self):
        fig = classes_chart(self.data)
        self.assertEqual(list(fig.data[0].y), [1_105_035, 6_401])
        self.assertEqual(fig.layout.yaxis.type, "log")

    def test_seven_figure_data_series(self):
        for section, size in (("monthly", 16), ("hourly", 24), ("weekday", 7),
                              ("category", 14), ("channel", 3)):
            chart = records_chart(self.data, section)
            self.assertEqual(len(chart.data[0].x), size)
            self.assertEqual(len(chart.data[0].customdata), size)

    def test_counts_are_not_confused_with_rates(self):
        rate = records_chart(self.data, "hourly")
        count = records_chart(self.data, "hourly", metric="frauds")
        self.assertNotEqual(rate.data[0].y[0], count.data[0].y[0])
        self.assertEqual(count.data[0].y[0], self.data["eda_train"]["hourly"][0]["frauds"])

    def test_private_rows_are_not_in_eda_file(self):
        self.assertEqual(self.data["privacy"]["raw_rows_included"], False)
        self.assertEqual(self.data["privacy"]["card_numbers_included"], False)
        self.assertEqual(self.data["privacy"]["transaction_ids_included"], False)


if __name__ == "__main__":
    unittest.main()
