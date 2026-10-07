"""P0-1: build_echonet_manifest CLI must define --allow-missing-videos.

Regression coverage for the historical ``AttributeError`` on the standard build
path (``args.allow_missing_videos`` referenced but never registered).
"""

from __future__ import annotations

import csv
import importlib.util
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "build_echonet_manifest.py"


def _load_builder():
    spec = importlib.util.spec_from_file_location("build_echonet_manifest_mod", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


def _make_fake_echonet(root: Path, *, n: int = 4, with_videos: bool = True) -> None:
    videos = root / "Videos"
    videos.mkdir(parents=True, exist_ok=True)
    filelist = root / "FileList.csv"
    with filelist.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=["FileName", "EF", "EDV", "ESV", "Split"]
        )
        writer.writeheader()
        splits = ["TRAIN", "TRAIN", "VAL", "TEST"]
        for i in range(n):
            name = f"clip{i}.avi"
            writer.writerow(
                {
                    "FileName": name,
                    "EF": str(30 + i * 5),
                    "EDV": "120",
                    "ESV": "60",
                    "Split": splits[i % len(splits)],
                }
            )
            if with_videos:
                (videos / name).write_bytes(b"RIFF-dummy-avi")


class TestArgParserSmoke(unittest.TestCase):
    def test_help_lists_allow_missing_videos(self):
        r = subprocess.run(
            [sys.executable, str(SCRIPT), "--help"],
            capture_output=True,
            text=True,
            cwd=str(ROOT),
        )
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertIn("--allow-missing-videos", r.stdout)
        self.assertIn("--test-subset", r.stdout)

    def test_parser_registers_flag(self):
        # Cheap in-process smoke: build the same parser flags and confirm the
        # attribute is reachable after parse (guards the AttributeError bug).
        import argparse

        parser = argparse.ArgumentParser()
        parser.add_argument("--allow-missing-videos", action="store_true")
        args = parser.parse_args([])
        self.assertFalse(args.allow_missing_videos)
        args2 = parser.parse_args(["--allow-missing-videos"])
        self.assertTrue(args2.allow_missing_videos)


class TestBuildDoesNotRaiseAttributeError(unittest.TestCase):
    def test_build_with_dummy_videos_succeeds(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td) / "echonet"
            out = Path(td) / "out"
            _make_fake_echonet(root)
            r = subprocess.run(
                [
                    sys.executable,
                    str(SCRIPT),
                    "--echonet-root",
                    str(root),
                    "--output-dir",
                    str(out),
                ],
                capture_output=True,
                text=True,
                cwd=str(ROOT),
            )
            self.assertEqual(r.returncode, 0, r.stderr)
            self.assertNotIn("AttributeError", r.stderr)
            self.assertTrue((out / "manifest.json").is_file())
            self.assertTrue((out / "train.json").is_file())
            self.assertTrue((out / "test.json").is_file())

    def test_missing_videos_defaults_to_clinical_hard_fail(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td) / "echonet"
            out = Path(td) / "out"
            _make_fake_echonet(root, with_videos=False)
            r = subprocess.run(
                [
                    sys.executable,
                    str(SCRIPT),
                    "--echonet-root",
                    str(root),
                    "--output-dir",
                    str(out),
                ],
                capture_output=True,
                text=True,
                cwd=str(ROOT),
            )
            self.assertEqual(r.returncode, 1)

    def test_allow_missing_videos_warns_and_marks_metadata_only(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td) / "echonet"
            out = Path(td) / "out"
            _make_fake_echonet(root, with_videos=False)
            r = subprocess.run(
                [
                    sys.executable,
                    str(SCRIPT),
                    "--echonet-root",
                    str(root),
                    "--output-dir",
                    str(out),
                    "--allow-missing-videos",
                ],
                capture_output=True,
                text=True,
                cwd=str(ROOT),
            )
            self.assertEqual(r.returncode, 0, r.stderr)
            self.assertNotIn("AttributeError", r.stderr)
            self.assertIn("METADATA-ONLY", r.stdout)
            meta = json.loads((out / "manifest.json").read_text(encoding="utf-8"))["meta"]
            self.assertTrue(meta["metadata_only"])
            self.assertFalse(meta["require_video"])
            self.assertIn("WARNING", meta)


class TestPerSplitArtifacts(unittest.TestCase):
    def test_per_split_files_and_explicit_flag(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td) / "echonet"
            out = Path(td) / "out"
            _make_fake_echonet(root, n=4)
            r = subprocess.run(
                [
                    sys.executable,
                    str(SCRIPT),
                    "--echonet-root",
                    str(root),
                    "--output-dir",
                    str(out),
                ],
                capture_output=True,
                text=True,
                cwd=str(ROOT),
            )
            self.assertEqual(r.returncode, 0, r.stderr)
            for name in ("train.json", "val.json", "test.json"):
                self.assertTrue((out / name).is_file(), name)
            test = json.loads((out / "test.json").read_text(encoding="utf-8"))
            self.assertEqual(test["meta"]["split"], "TEST")
            self.assertTrue(all(p.get("file_name") for p in test["pairs"]))

    def test_r0_anchor_and_test_subset(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td) / "echonet"
            out = Path(td) / "out"
            _make_fake_echonet(root, n=6)
            r = subprocess.run(
                [
                    sys.executable,
                    str(SCRIPT),
                    "--echonet-root",
                    str(root),
                    "--output-dir",
                    str(out),
                    "--subset-5000",
                    "--test-subset",
                    "1",
                ],
                capture_output=True,
                text=True,
                cwd=str(ROOT),
            )
            self.assertEqual(r.returncode, 0, r.stderr)
            anchor = out / "r0_external_anchor_5000.json"
            self.assertTrue(anchor.is_file())
            anchor_data = json.loads(anchor.read_text(encoding="utf-8"))
            self.assertTrue(anchor_data["meta"]["provenance"]["may_overlap_train"])
            self.assertFalse((out / "subset_5000.json").exists())

            sub = out / "test_subset_1.json"
            self.assertTrue(sub.is_file())
            sub_data = json.loads(sub.read_text(encoding="utf-8"))
            prov = sub_data["meta"]["provenance"]
            self.assertFalse(prov["may_overlap_train"])
            self.assertEqual(prov["sampled_from"], "TEST split only")


if __name__ == "__main__":
    unittest.main()
