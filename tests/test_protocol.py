"""Tests for protocol matrix, paper-mode hard-fail, and EF grid."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from echoclip.protocol import (
    DEFAULT_EF_VALUES,
    EXPERIMENT_IDS,
    OFFICIAL_EF_VALUES,
    assert_primary_eval_sampling,
    get_experiment,
    list_experiments,
    merge_metrics_meta,
    resolve_eval_sample_strategy,
    resolve_experiment_id,
    write_protocol_comparison,
    write_subset_ids,
)


ROOT = Path(__file__).resolve().parents[1]


class TestProtocolSpecs(unittest.TestCase):
    def test_catalog_complete(self):
        specs = list_experiments()
        self.assertEqual([s.id for s in specs], list(EXPERIMENT_IDS))
        r0 = get_experiment("R0")
        self.assertEqual(r0.pool, "frames")
        self.assertIn("EchoCLIP-based", r0.title)
        self.assertEqual(r0.eval_sample_strategy, "uniform")
        self.assertEqual(r0.paper_eval_sample_strategy, "official_stride")
        r0u = get_experiment("R0U16")
        self.assertEqual(r0u.eval_sample_strategy, "uniform")
        r1 = get_experiment("M1")  # alias
        self.assertEqual(r1.id, "R1")
        self.assertEqual(r1.pool, "mean")
        r2 = get_experiment("R2")
        self.assertEqual(r2.prediction_mode, "direct_regression")
        r5 = get_experiment("M2")
        self.assertEqual(r5.id, "R5")
        self.assertTrue(r5.train)
        self.assertIn("NOT a zero-shot", r5.description)
        self.assertEqual(r5.prediction_mode, "zeroshot")
        r6 = get_experiment("M4")
        self.assertEqual(r6.id, "R6")
        self.assertTrue(r6.calibrate)
        oracle = get_experiment("Oracle-EDES")
        self.assertTrue(oracle.annotation_assisted)
        self.assertEqual(oracle.eval_sample_strategy, "ed_es")

    def test_legacy_aliases(self):
        self.assertEqual(resolve_experiment_id("B0"), "R0")
        self.assertEqual(resolve_experiment_id("S0"), "R2")
        self.assertEqual(resolve_experiment_id("S2"), "R4")

    def test_unknown_raises(self):
        with self.assertRaises(KeyError):
            get_experiment("M99")

    def test_primary_eval_rejects_mixed_ed_es(self):
        with self.assertRaises(ValueError) as ctx:
            assert_primary_eval_sampling(split="test", strategy="mixed", experiment_id="R5")
        self.assertIn("uniform", str(ctx.exception).lower())
        with self.assertRaises(ValueError):
            assert_primary_eval_sampling(split="val", strategy="ed_es", experiment_id="M1")
        # Oracle allowed
        assert_primary_eval_sampling(
            split="test", strategy="ed_es", experiment_id="ORACLE_EDES"
        )

    def test_resolve_eval_honors_val_sample_strategy(self):
        spec = get_experiment("R5")
        ss = resolve_eval_sample_strategy(
            spec=spec,
            cli_strategy=None,
            cfg={"val_sample_strategy": "uniform", "sample_strategy": "mixed"},
            split="test",
        )
        self.assertEqual(ss, "uniform")
        # CLI override that is illegal still fails
        with self.assertRaises(ValueError):
            resolve_eval_sample_strategy(
                spec=spec, cli_strategy="mixed", cfg={}, split="test"
            )

    def test_ef_grids(self):
        self.assertEqual(OFFICIAL_EF_VALUES, list(range(0, 101)))
        self.assertEqual(len(OFFICIAL_EF_VALUES), 101)
        self.assertEqual(DEFAULT_EF_VALUES[0], 15)
        self.assertEqual(DEFAULT_EF_VALUES[-1], 80)

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

    def test_subset_ids_samples_when_oversized(self):
        pairs = [{"file_name": f"v{i}.avi"} for i in range(20)]
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "subset_ids.json"
            write_subset_ids(pairs, path, seed=42, n=5)
            data = json.loads(path.read_text(encoding="utf-8"))
            self.assertEqual(data["n_written"], 5)
            write_subset_ids(pairs, path, seed=42, n=5, already_sampled=True)
            data3 = json.loads(path.read_text(encoding="utf-8"))
            self.assertEqual(data3["n_written"], 20)

    def test_merge_metrics_demo_and_paper(self):
        spec = get_experiment("B0")
        out = merge_metrics_meta({"mae": 1.0}, experiment=spec, demo=True)
        self.assertTrue(out["demo_is_not_clinical"])
        self.assertEqual(out["experiment_id"], "R0")
        out2 = merge_metrics_meta({"mae": 1.0}, experiment=spec, paper=True)
        self.assertTrue(out2["official_reproduction"])

    def test_protocol_comparison_table(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for exp_id, mae, demo in (("R0", 7.1, False), ("R1", 8.0, False), ("R5", 6.5, True)):
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
                            "pool": "frames" if exp_id == "R0" else "mean",
                            "n_eval": 10,
                        }
                    ),
                    encoding="utf-8",
                )
            paths = write_protocol_comparison(root)
            payload = json.loads(paths["json"].read_text(encoding="utf-8"))
            self.assertEqual(payload["n_experiments"], 3)
            self.assertIn("legacy_aliases", payload)
            md = paths["md"].read_text(encoding="utf-8")
            self.assertIn("R0", md)


class TestPaperModeHardFail(unittest.TestCase):
    def test_from_official_paper_mode_raises_without_weights(self):
        import os

        from echoclip.config import EchoCLIPConfig
        from echoclip.model import EchoCLIP

        cfg = EchoCLIPConfig(vision_backbone="simple_cnn", pretrained_vision=False)
        env = {**os.environ, "ECHOCLIP_SKIP_HUB": "1"}
        with mock.patch.dict(os.environ, env, clear=False):
            with self.assertRaises(RuntimeError) as ctx:
                EchoCLIP.from_official_echo_clip(
                    cfg, allow_scratch_fallback=False
                )
            self.assertIn("Paper", str(ctx.exception))

    def test_inference_official_ef_grid(self):
        from echoclip.config import EchoCLIPConfig
        from echoclip.model import EchoCLIP
        from echoclip.zeroshot import EchoCLIPInference

        cfg = EchoCLIPConfig(
            vision_backbone="simple_cnn",
            pretrained_vision=False,
            embed_dim=64,
            text_width=64,
            text_layers=2,
            text_heads=4,
        )
        model = EchoCLIP(cfg)
        eng = EchoCLIPInference(model, device="cpu", official_reproduction=True)
        self.assertEqual(eng.default_ef_values, list(range(0, 101)))
        eng2 = EchoCLIPInference(model, device="cpu", official_reproduction=False)
        self.assertEqual(eng2.default_ef_values[0], 15)


class TestPublicEchoBuilders(unittest.TestCase):
    def _load_script(self, name: str):
        import importlib.util

        path = ROOT / "scripts" / name
        spec = importlib.util.spec_from_file_location(name.replace(".py", ""), path)
        mod = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        spec.loader.exec_module(mod)
        return mod

    def test_missing_root_fails_clearly(self):
        mod = self._load_script("build_public_echo_manifest.py")
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(FileNotFoundError) as ctx:
                mod.build_camus(Path(tmp))
            self.assertIn("CAMUS", str(ctx.exception))
            with self.assertRaises(FileNotFoundError) as ctx2:
                mod.build_echonet_family(Path(tmp), dataset_key="echonet_pediatric")
            self.assertIn("Pediatric", str(ctx2.exception))

    def test_camus_mock_layout(self):
        mod = self._load_script("build_public_echo_manifest.py")
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            patient = root / "training" / "patient0001"
            patient.mkdir(parents=True)
            (patient / "Info_2CH.cfg").write_text(
                "EF: 55.0\nSex: F\nAge: 40\n", encoding="utf-8"
            )
            (patient / "Info_4CH.cfg").write_text("EF=48.5\n", encoding="utf-8")
            (patient / "patient0001_2CH_sequence.avi").write_bytes(b"RIFF")
            (patient / "patient0001_4CH_ED.png").write_bytes(b"\x89PNG")
            pairs, meta, skipped = mod.build_camus(root, require_media=True)
            self.assertEqual(len(pairs), 2)
            views = {p["view"] for p in pairs}
            self.assertEqual(views, {"A2C", "A4C"})

    def test_pediatric_filelist_via_adapter(self):
        import csv

        mod = self._load_script("build_public_echo_manifest.py")
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

    def test_echonet_subset_ids_in_builder(self):
        import csv

        mod = self._load_script("build_echonet_manifest.py")
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
            rows = mod.load_filelist(filelist)
            pairs, _ = mod.build_pairs(rows, videos, require_video=False)
            ids_path = write_subset_ids(pairs, out / "subset_5000_ids.json", seed=42, n=5000)
            self.assertTrue(ids_path.exists())


if __name__ == "__main__":
    unittest.main()
