import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd

from backend.services import experiment_service as exp
from backend.services import import_service
from backend.features.feature_engineering import FEATURE_COLUMNS


class ExperimentServiceTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.path = Path(self.temp.name)
        self.patches = [patch.object(import_service, "UPLOAD_DIR", self.path / "uploads"),
                        patch.object(exp, "EXPERIMENT_DIR", self.path / "experiments"),
                        patch.object(exp, "PROJECT_ROOT", self.path)]
        for p in self.patches: p.start()
        rows=[]
        for i in range(120):
            row={c: float((i % 13) + 1) for c in FEATURE_COLUMNS}
            row["category"] = "food" if i % 2 else "travel"
            row["is_fraud"] = i % 2
            row["split"] = "train" if i < 80 else "val"
            rows.append(row)
        csv=pd.DataFrame(rows).to_csv(index=False).encode()
        self.imported=import_service.save_csv("experiment.csv", csv)

    def tearDown(self):
        for p in reversed(self.patches): p.stop()
        self.temp.cleanup()

    def test_splits_are_disjoint_and_test_partition_is_not_used(self):
        train, val = exp._read_splits(self.imported["id"], 50)
        self.assertEqual(len(train), 50)
        self.assertEqual(len(val), 40)
        self.assertEqual(set(train["_split"]), {"train"})
        self.assertEqual(set(val["_split"]), {"validation"})

    def test_background_training_saves_separate_experiment_and_validation_metrics(self):
        job = exp.start_experiment(self.imported["id"], algorithm="logistic_regression",
            params={"max_iter": 100}, class_weight="balanced", sample_limit=100, threshold=0.5)
        for _ in range(1200):
            current=exp.get_experiment(job["id"])
            if current["status"] in {"completed", "failed"}: break
            time.sleep(0.05)
        self.assertEqual(current["status"], "completed", current.get("error"))
        self.assertEqual(current["metrics"]["evaluation_split"], "Validation")
        self.assertTrue((self.path / "experiments" / job["id"] / "model.joblib").is_file())

    def test_unknown_algorithm_rejected_before_enqueue(self):
        with self.assertRaises(ValueError):
            exp.start_experiment(self.imported["id"], algorithm="invalid", params={},
                class_weight="none", sample_limit=100, threshold=0.5)


if __name__ == "__main__": unittest.main()
