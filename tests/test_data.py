import json
import tempfile
import unittest
from pathlib import Path

from echoclip.data import load_manifest, split_manifest, validate_manifest


class TestManifest(unittest.TestCase):
    def test_load_json_pairs(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "m.json"
            p.write_text(json.dumps({"pairs": [{"image": "a.png", "text": "HELLO"}]}))
            pairs = load_manifest(p)
            self.assertEqual(len(pairs), 1)

    def test_split_keeps_train_nonempty(self):
        pairs = [{"image": f"{i}.png", "text": "T"} for i in range(10)]
        train, val = split_manifest(pairs, val_ratio=0.2, seed=0)
        self.assertGreater(len(train), 0)
        self.assertGreater(len(val), 0)

    def test_validate_missing_file(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            pairs = [{"image": "missing.png", "text": "X"}]
            errs = validate_manifest(pairs, root)
            self.assertTrue(any("not found" in e for e in errs))


if __name__ == "__main__":
    unittest.main()
