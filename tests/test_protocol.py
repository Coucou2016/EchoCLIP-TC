"""Tests for protocol helpers and public echo manifest builders."""

from __future__ import annotations

import csv
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

from echoclip.protocol import (
    EXPERIMENT_IDS,
    get_experiment,
    list_experiments,
    merge_metrics_meta,
    write_subset_ids,
)


ROOT = Path(__file__).resolve().parents[1]


def _load_script(name: str):
    path = ROOT / "scripts" / name
    spec = importlib.util.spec_from_file_location(name.replace(".py", ""), path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


class TestProtocolSpecs(unittest.TestCase):
    def test_catalog_complete(self):
        specs = list_experiments()
        self.assertEqual([s.id for s in specs], list(EXPERIMENT_IDS))
        b0 = get_experiment("b0")
        self.assertEqual(b0.pool, "frames")
        self.assertFalse(b0.train)
        m1 = get_experiment("M1")
        self.assertEqual(m1.pool, "mean")
        m2 = get_experiment("M2")
        self.assertTrue(m2.train)
        self.assertEqual(m2.pool, "temporal")
        m4 = get_experiment("M4")
        self.assertTrue(m4.calibrate)

    def test_unknown_raises(self):
        with self.assertRaises(KeyError):
            get_experiment("M99")

    def test_subset_ids_writer(self):
        pairs = [
            {"file_name": "a.avi", "image": "Videos/a.avi"},
            {"file_name": "b.avi", "image": "Videos/b.avi"},
        ]
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "subset_5000_ids.json"
            write_subset_ids(pairs, path, seed=42, n=5000)
            data = json.loads(path.read_text(encoding="utf-8"))
            self.assertEqual(data["seed"], 42)
            self.assertEqual(data["ids"], ["a.avi", "b.avi"])
            self.assertTrue(path.with_suffix(".txt").exists())

    def test_subset_ids_samples_when_oversized(self):
        pairs = [{"file_name": f"v{i}.avi"} for i in range(20)]
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "subset_ids.json"
            write_subset_ids(pairs, path, seed=42, n=5)
            data = json.loads(path.read_text(encoding="utf-8"))
            self.assertEqual(data["n_written"], 5)
            self.assertEqual(len(data["ids"]), 5)
            # Deterministic with seed
            write_subset_ids(pairs, path, seed=42, n=5)
            data2 = json.loads(path.read_text(encoding="utf-8"))
            self.assertEqual(data["ids"], data2["ids"])
            # already_sampled keeps all rows
            write_subset_ids(pairs, path, seed=42, n=5, already_sampled=True)
            data3 = json.loads(path.read_text(encoding="utf-8"))
            self.assertEqual(data3["n_written"], 20)

    def test_merge_metrics_demo_flag(self):
        spec = get_experiment("B0")
        out = merge_metrics_meta({"mae": 1.0}, experiment=spec, demo=True)
        self.assertTrue(out["demo_is_not_clinical"])
        self.assertEqual(out["experiment_id"], "B0")

    def test_protocol_comparison_table(self):
        from echoclip.protocol import (
            build_comparison_rows,
            comparison_to_markdown,
            write_protocol_comparison,
        )

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for exp_id, mae, demo in (("B0", 7.1, False), ("M1", 8.0, False), ("M2", 6.5, True)):
                d = root / exp_id
                d.mkdir()
                (d / "metrics.json").write_text(
                    json.dumps(
                        {
                            "experiment_id": exp_id,
                            "mae": mae,
                            "rmse": mae + 1,
                            "auc_ef_lt_50": 0.8,
                            "load_source": "scratch_fallback" if demo else "hf-hub:demo",
                            "demo_is_not_clinical": demo,
                            "pool": "frames" if exp_id == "B0" else "mean",
                            "n_eval": 10,
                        }
                    ),
                    encoding="utf-8",
                )
            paths = write_protocol_comparison(root)
            self.assertTrue(paths["json"].exists())
            self.assertTrue(paths["md"].exists())
            payload = json.loads(paths["json"].read_text(encoding="utf-8"))
            self.assertEqual(payload["n_experiments"], 3)
            self.assertTrue(payload["any_demo"])
            rows = build_comparison_rows(
                {r["experiment_id"]: r for r in payload["rows"]}
            )
            self.assertEqual([r["experiment_id"] for r in rows], ["B0", "M1", "M2"])
            md = comparison_to_markdown(payload["rows"])
            self.assertIn("B0", md)
            self.assertIn("Honesty", md)

    def test_write_protocol_table_script(self):
        mod = _load_script("write_protocol_table.py")
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            d = root / "B0"
            d.mkdir()
            (d / "metrics.json").write_text(
                json.dumps({"mae": 1.0, "demo_is_not_clinical": True, "load_source": "x"}),
                encoding="utf-8",
            )
            # Simulate argv
            import sys

            old = sys.argv
            try:
                sys.argv = ["write_protocol_table.py", "--protocol-root", str(root)]
                code = mod.main()
            finally:
                sys.argv = old
            self.assertEqual(code, 0)
            self.assertTrue((root / "comparison.md").exists())


class TestPublicEchoBuilders(unittest.TestCase):
    def test_missing_root_fails_clearly(self):
        mod = _load_script("build_public_echo_manifest.py")
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(FileNotFoundError) as ctx:
                mod.build_camus(Path(tmp))
            self.assertIn("CAMUS", str(ctx.exception))
            with self.assertRaises(FileNotFoundError) as ctx2:
                mod.build_echonet_family(Path(tmp), dataset_key="echonet_pediatric")
            self.assertIn("Pediatric", str(ctx2.exception))

    def test_camus_mock_layout(self):
        mod = _load_script("build_public_echo_manifest.py")
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            patient = root / "training" / "patient0001"
            patient.mkdir(parents=True)
            (patient / "Info_2CH.cfg").write_text(
                "EF: 55.0\nSex: F\nAge: 40\n", encoding="utf-8"
            )
            (patient / "Info_4CH.cfg").write_text(
                "EF=48.5\n", encoding="utf-8"
            )
            # placeholder media
            (patient / "patient0001_2CH_sequence.avi").write_bytes(b"RIFF")
            (patient / "patient0001_4CH_ED.png").write_bytes(b"\x89PNG")
            pairs, meta, skipped = mod.build_camus(root, require_media=True)
            self.assertEqual(len(pairs), 2)
            views = {p["view"] for p in pairs}
            self.assertEqual(views, {"A2C", "A4C"})
            self.assertTrue(all(p["split"] == "TRAIN" for p in pairs))
            self.assertEqual(meta["source"], "CAMUS")
            efs = sorted(p["ef"] for p in pairs)
            self.assertEqual(efs, [48.5, 55.0])

    def test_pediatric_filelist_via_adapter(self):
        mod = _load_script("build_public_echo_manifest.py")
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "Videos").mkdir()
            filelist = root / "FileList.csv"
            with filelist.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(
                    handle,
                    fieldnames=["FileName", "EF", "EDV", "ESV", "Split"],
                )
                writer.writeheader()
                writer.writerow(
                    {
                        "FileName": "ped0.avi",
                        "EF": "62",
                        "EDV": "40",
                        "ESV": "15",
                        "Split": "TEST",
                    }
                )
            pairs, meta, skipped = mod.build_echonet_family(
                root,
                dataset_key="echonet_pediatric",
                require_video=False,
            )
            self.assertEqual(len(pairs), 1)
            self.assertEqual(pairs[0]["ef"], 62.0)
            self.assertEqual(meta["source"], "echonet_pediatric")
            self.assertEqual(skipped, [])

    def test_echonet_subset_ids_in_builder(self):
        mod = _load_script("build_echonet_manifest.py")
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            out = root / "out"
            videos = root / "Videos"
            videos.mkdir()
            filelist = root / "FileList.csv"
            with filelist.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(
                    handle,
                    fieldnames=["FileName", "EF", "EDV", "ESV", "Split"],
                )
                writer.writeheader()
                for i in range(3):
                    writer.writerow(
                        {
                            "FileName": f"c{i}.avi",
                            "EF": str(40 + i),
                            "EDV": "100",
                            "ESV": "50",
                            "Split": "TEST",
                        }
                    )
            # Invoke main pieces
            rows = mod.load_filelist(filelist)
            pairs, _ = mod.build_pairs(rows, videos, require_video=False)
            from echoclip.protocol import write_subset_ids

            ids_path = write_subset_ids(pairs, out / "subset_5000_ids.json", seed=42, n=5000)
            self.assertTrue(ids_path.exists())
            data = json.loads(ids_path.read_text(encoding="utf-8"))
            self.assertEqual(len(data["ids"]), 3)


if __name__ == "__main__":
    unittest.main()

