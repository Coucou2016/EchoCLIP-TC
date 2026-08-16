import csv
import json
import tempfile
import unittest
from pathlib import Path

from echoclip.structured_text import pair_record


class TestEchoNetBuilderHelpers(unittest.TestCase):
    def test_pair_record_manifest_roundtrip(self):
        rec = pair_record(
            "Videos/demo.avi",
            ef=55.2,
            edv=130.0,
            extra={"split": "TEST", "ed_frame": 12, "es_frame": 28},
        )
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "manifest.json"
            path.write_text(json.dumps({"pairs": [rec]}), encoding="utf-8")
            from echoclip.data import load_manifest

            pairs = load_manifest(path)
            self.assertEqual(pairs[0]["ef"], 55.2)
            self.assertEqual(pairs[0]["ed_frame"], 12)
            self.assertIn("LV EJECTION FRACTION", pairs[0]["text"].upper())

    def test_filelist_parse_via_script(self):
        import importlib.util

        script = Path(__file__).resolve().parents[1] / "scripts" / "build_echonet_manifest.py"
        spec = importlib.util.spec_from_file_location("build_echonet_manifest", script)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            videos = root / "Videos"
            videos.mkdir()
            filelist = root / "FileList.csv"
            with filelist.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(
                    handle, fieldnames=["FileName", "EF", "EDV", "ESV", "Split", "NumberOfFrames"]
                )
                writer.writeheader()
                writer.writerow(
                    {
                        "FileName": "clip0.avi",
                        "EF": "41.5",
                        "EDV": "180",
                        "ESV": "90",
                        "Split": "TEST",
                        "NumberOfFrames": "50",
                    }
                )
            rows = mod.load_filelist(filelist)
            pairs, skipped = mod.build_pairs(
                rows, videos, require_video=False, include_dilation=True
            )
            self.assertEqual(len(pairs), 1)
            self.assertEqual(pairs[0]["split"], "TEST")
            self.assertEqual(pairs[0]["ef"], 41.5)
            self.assertTrue(any("MILD DILATED LEFT VENTRICLE" in c for c in pairs[0]["captions"]))
            self.assertEqual(skipped, [])
            self.assertIsNone(mod._find_filelist(root / "missing"))
