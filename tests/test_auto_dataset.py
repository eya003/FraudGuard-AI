import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from backend.services import import_service


class AutoImportTest(unittest.TestCase):
    def test_local_dataset_auto_registered_once_with_real_counts(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            source=root/'sparkov.csv'
            source.write_text('amt,category,is_fraud\n10,food,0\n20,travel,1\n',encoding='utf-8')
            with patch.object(import_service,'UPLOAD_DIR',root/'imports'), \
                 patch.dict('os.environ',{'DATASET_PATH':str(source)}):
                first=import_service.list_imports()
                second=import_service.list_imports()
            self.assertEqual(len(first),1)
            self.assertEqual(len(second),1)
            self.assertEqual(first[0]['row_count'],2)
            self.assertEqual(first[0]['missing_cells'],0)
            self.assertTrue(first[0]['auto_imported'])


if __name__=='__main__':unittest.main()
