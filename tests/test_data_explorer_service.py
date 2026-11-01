import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from backend.services import import_service


CSV = (
    "trans_num,trans_date_trans_time,cc_num,amt,city_pop,category,is_fraud,_split\n"
    "tx-1,2020-01-01 00:00:00,111,10,100,A,0,train\n"
    "tx-2,2020-01-01 01:00:00,222,25,200,B,1,val\n"
    "tx-3,2020-01-02 02:00:00,111,15,100,A,0,test\n"
)


class StreamingImportExplorerTest(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.directory = Path(self.temp_dir.name)
        self.patches = [
            patch.object(import_service, "UPLOAD_DIR", self.directory),
            patch.object(import_service, "CHUNK_SIZE", 2),
        ]
        for item in self.patches:
            item.start()
        self.metadata = import_service.save_csv("transactions.csv", CSV.encode())
        self.import_id = self.metadata["id"]

    def tearDown(self):
        for item in reversed(self.patches):
            item.stop()
        self.temp_dir.cleanup()

    def test_import_reads_in_chunks_and_records_exact_counts(self):
        self.assertEqual(self.metadata["row_count"], 3)
        self.assertEqual(self.metadata["missing_cells"], 0)
        self.assertEqual(self.metadata["columns"][-1], "_split")

    def test_paged_server_side_search_split_and_time_filters(self):
        result = import_service.read_import_table(
            self.import_id, offset=0, limit=1, search="tx-", split="val",
            start="2020-01-01", end="2020-01-01",
        )
        self.assertEqual(result["total"], 1)
        self.assertEqual(result["items"][0]["trans_num"], "tx-2")
        second = import_service.read_import_table(self.import_id, offset=1, limit=1)
        self.assertEqual(second["items"][0]["trans_num"], "tx-2")

    def test_summary_uses_source_labels_and_partition_totals(self):
        summary = import_service.summarize_import(self.import_id)
        self.assertEqual(summary["rows"], 3)
        self.assertEqual(summary["fraud_labels"]["fraud_count"], 1)
        self.assertAlmostEqual(summary["fraud_labels"]["fraud_rate"], 1 / 3)
        self.assertEqual(summary["amount"]["sum"], 50)
        self.assertAlmostEqual(summary["numeric_stats"]["amt"]["mean"], 50 / 3)
        self.assertEqual(summary["split_counts"], {"train": 1, "val": 1, "test": 1})
        self.assertEqual(summary["categories"][0]["category"], "A")
        validation = import_service.summarize_import(self.import_id, split="val")
        self.assertEqual(validation["rows"], 1)
        self.assertEqual(validation["fraud_count"] if "fraud_count" in validation else validation["fraud_labels"]["fraud_count"], 1)

    def test_large_offset_row_lookup_scans_chunks_and_serializes_missing_as_null(self):
        row = import_service.read_import_row(self.import_id, 2)
        self.assertEqual(row["trans_num"], "tx-3")
        self.assertEqual(row["amt"], 15)

    def test_import_rejects_csv_without_a_supported_amount_column(self):
        with self.assertRaisesRegex(import_service.ImportError, "Colonne de montant"):
            import_service.save_csv("bad.csv", b"foo,bar\n1,2\n")


if __name__ == "__main__":
    unittest.main()
