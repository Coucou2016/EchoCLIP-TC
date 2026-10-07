"""P0-2: split integrity — official split names, TRAIN/TEST disjointness,
and TEST-only subsets for adapted-model evaluation (R2–R6).
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

from echoclip.protocol import (
    assert_disjoint,
    manifest_ids,
    split_provenance,
)

ROOT = Path(__file__).resolve().parents[1]
BUILDER = ROOT / "scripts" / "build_echonet_manifest.py"


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


def _fake_echonet(root: Path, n: int = 6) -> None:
    videos = root / "Videos"
    videos.mkdir(parents=True, exist_ok=True)
    splits = ["TRAIN", "TRAIN", "TRAIN", "VAL", "TEST", "TEST"]
    with (root / "FileList.csv").open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(
            fh, fieldnames=["FileName", "EF", "EDV", "ESV", "Split"]
        )
        writer.writeheader()
        for i in range(n):
            name = f"study{i}.avi"
            writer.writerow(
                {
                    "FileName": name,
                    "EF": str(30 + i),
                    "EDV": "120",
                    "ESV": "60",
                    "Split": splits[i % len(splits)],
                }
            )
            (videos / name).write_bytes(b"RIFF-dummy")


class TestAssertDisjoint(unittest.TestCase):
    def test_manifest_ids_prefers_file_name(self):
        pairs = [
            {"file_name": "a.avi", "image": "Videos/a.avi"},
            {"image": "Videos/b.avi"},
            {"file_name": "a.avi"},  # duplicate dropped
        ]
        self.assertEqual(manifest_ids(pairs), ["a.avi", "b.avi"])

    def test_disjoint_passes(self):
        train = [{"file_name": "t1.avi"}, {"file_name": "t2.avi"}]
        test = [{"file_name": "e1.avi"}, {"file_name": "e2.avi"}]
        # Should not raise
        assert_disjoint(train, test)

    def test_overlap_raises_with_ids_listed(self):
        train = [{"file_name": "a.avi"}, {"file_name": "b.avi"}]
        test = [{"file_name": "b.avi"}, {"file_name": "c.avi"}]
        with self.assertRaises(RuntimeError) as ctx:
            assert_disjoint(train, test, label_train="TRAIN", label_eval="TEST")
        msg = str(ctx.exception)
        self.assertIn("b.avi", msg)
        self.assertIn("TRAIN", msg)
        self.assertIn("TEST", msg)

    def test_split_provenance_keys_and_overlap(self):
        train = [{"file_name": "a.avi"}, {"file_name": "b.avi"}]
        test = [{"file_name": "b.avi"}]
        prov = split_provenance(None, None, train, test)
        for key in (
            "train_manifest_sha256",
            "test_manifest_sha256",
            "train_n",
            "test_n",
            "train_test_overlap_n",
        ):
            self.assertIn(key, prov)
        self.assertEqual(prov["train_n"], 2)
        self.assertEqual(prov["test_n"], 1)
        self.assertEqual(prov["train_test_overlap_n"], 1)

    def test_split_provenance_hashes_files(self):
        with tempfile.TemporaryDirectory() as td:
            train = Path(td) / "train.json"
            test = Path(td) / "test.json"
            train.write_text('{"pairs":[{"file_name":"t.avi"}]}', encoding="utf-8")
            test.write_text('{"pairs":[{"file_name":"e.avi"}]}', encoding="utf-8")
            prov = split_provenance(train, test)
            self.assertEqual(len(str(prov["train_manifest_sha256"])), 64)
            self.assertEqual(len(str(prov["test_manifest_sha256"])), 64)
            self.assertNotEqual(
                prov["train_manifest_sha256"], prov["test_manifest_sha256"]
            )


class TestBuilderSplitNames(unittest.TestCase):
    def _build(self, extra=()):
        td = tempfile.TemporaryDirectory()
        root = Path(td.name) / "echonet"
        out = Path(td.name) / "out"
        _fake_echonet(root)
        r = subprocess.run(
            [
                sys.executable,
                str(BUILDER),
                "--echonet-root",
                str(root),
                "--output-dir",
                str(out),
                *extra,
            ],
            capture_output=True,
            text=True,
            cwd=str(ROOT),
        )
        self.assertEqual(r.returncode, 0, r.stderr)
        return td, out

    def test_official_split_names(self):
        td, out = self._build()
        try:
            for name in ("train.json", "val.json", "test.json"):
                self.assertTrue((out / name).is_file(), name)
            train = json.loads((out / "train.json").read_text(encoding="utf-8"))
            test = json.loads((out / "test.json").read_text(encoding="utf-8"))
            self.assertEqual(train["meta"]["split"], "TRAIN")
            self.assertEqual(test["meta"]["split"], "TEST")
        finally:
            td.cleanup()

    def test_train_test_disjoint_after_build(self):
        td, out = self._build()
        try:
            train = json.loads((out / "train.json").read_text(encoding="utf-8"))["pairs"]
            test = json.loads((out / "test.json").read_text(encoding="utf-8"))["pairs"]
            assert_disjoint(train, test)  # no RuntimeError
            prov = split_provenance(out / "train.json", out / "test.json", train, test)
            self.assertEqual(prov["train_test_overlap_n"], 0)
        finally:
            td.cleanup()

    def test_learned_model_subset_is_test_only(self):
        td, out = self._build(extra=("--test-subset", "2", "--subset-5000"))
        try:
            train = json.loads((out / "train.json").read_text(encoding="utf-8"))["pairs"]
            sub = json.loads((out / "test_subset_2.json").read_text(encoding="utf-8"))
            self.assertEqual(sub["meta"]["split"], "TEST")
            self.assertFalse(sub["meta"]["provenance"]["may_overlap_train"])
            assert_disjoint(train, sub["pairs"])

            # Historical mixed-pool subset is explicitly an R0-only anchor.
            anchor = json.loads(
                (out / "r0_external_anchor_5000.json").read_text(encoding="utf-8")
            )
            self.assertTrue(anchor["meta"]["provenance"]["may_overlap_train"])
            self.assertIn("R0", anchor["meta"]["provenance"]["role"])
        finally:
            td.cleanup()


class TestProtocolGuard(unittest.TestCase):
    def test_adapted_ids_include_r2_through_r6(self):
        mod = _load("run_protocol_split_guard", ROOT / "scripts" / "run_protocol.py")
        for eid in ("R2", "R3", "R4", "R5", "R6"):
            self.assertIn(eid, mod.ADAPTED_EXPERIMENT_IDS)
        self.assertNotIn("R0", mod.ADAPTED_EXPERIMENT_IDS)
        self.assertNotIn("R1", mod.ADAPTED_EXPERIMENT_IDS)

    def test_guard_flags_overlap_for_adapted_only(self):
        mod = _load("run_protocol_split_guard2", ROOT / "scripts" / "run_protocol.py")
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            train = root / "train.json"
            test = root / "test.json"
            shared = [{"image": "v.avi", "file_name": "v.avi", "text": "x"}]
            train.write_text(json.dumps({"pairs": shared}), encoding="utf-8")
            test.write_text(json.dumps({"pairs": shared}), encoding="utf-8")

            err = mod._guard_adapted_split_integrity(
                mod.get_experiment("R5"), train, test
            )
            self.assertIsNotNone(err)
            self.assertIn("overlap", err.lower())

            # Zero-shot R0 is exempt (may legitimately reuse the external anchor).
            err0 = mod._guard_adapted_split_integrity(
                mod.get_experiment("R0"), train, test
            )
            self.assertIsNone(err0)


if __name__ == "__main__":
    unittest.main()
