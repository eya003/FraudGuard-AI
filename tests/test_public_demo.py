import unittest

from backend.services.public_demo_service import public_demo_available, public_sample, public_summary


class PublicDemoAssetsTest(unittest.TestCase):
    def test_full_dataset_aggregates_are_exact_and_present(self):
        self.assertTrue(public_demo_available())
        summary = public_summary()
        self.assertEqual(summary["rows"], 1_852_394)
        self.assertEqual(summary["fraud_count"], 9_651)
        self.assertEqual(summary["labelled_rows"], 1_852_394)
        self.assertEqual(sum(item["transactions"] for item in summary["categories"]), summary["rows"])
        self.assertEqual(sum(item["transactions"] for item in summary["months"]), summary["rows"])
        self.assertEqual(len(summary["amount_histogram"]["normal"]), 100)

    def test_sample_is_real_labelled_and_filterable(self):
        all_rows = public_sample(limit=100)
        self.assertTrue(all_rows["items"])
        category = all_rows["items"][0]["category"]
        filtered = public_sample(limit=100, category=category, class_filter="fraud")
        self.assertTrue(all(row["category"] == category for row in filtered["items"]))
        self.assertTrue(all(int(row["is_fraud"]) == 1 for row in filtered["items"]))
        self.assertTrue(all("street" not in row and "dob" not in row for row in all_rows["items"]))

    def test_public_demo_router_is_registered_for_public_frontend(self):
        from backend.main import app

        # FastAPI 0.12x defers included routers in app.routes; OpenAPI resolves
        # those routers and is the stable public contract to verify here.
        paths = set(app.openapi()["paths"])
        self.assertIn("/api/v1/public-demo/summary", paths)
        self.assertIn("/api/v1/public-demo/sample", paths)


if __name__ == "__main__":
    unittest.main()
