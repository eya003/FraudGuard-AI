import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from backend.services import import_service


class ProfileTest(unittest.TestCase):
    def test_feature_profile_uses_actual_values_and_respects_split(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory=Path(tmp)
            with patch.object(import_service,"UPLOAD_DIR",directory):
                item=import_service.save_csv("training.csv",(
                    "amt,category,split,is_fraud\n"
                    "10,food,train,0\n20,travel,train,1\n"
                    "50,food,val,1\n60,food,val,0\n"
                ).encode())
                profile=import_service.profile_feature(item["id"],"amt",split="val",bins=4)
                self.assertEqual(profile["rows"],2)
                self.assertEqual(profile["mean"],55)
                self.assertEqual(sum(x["count"] for x in profile["values"]),2)
                cats=import_service.profile_feature(item["id"],"category",split="train")
                self.assertEqual(cats["rows"],2)
                self.assertEqual({x["label"] for x in cats["values"]},{"food","travel"})

if __name__=="__main__":unittest.main()
