"""Tests for EF-only captions, CardiacCLIP stub, attention skeleton, seeds helper."""

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]


def _load_script(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


class TestEFOnlyCaptions(unittest.TestCase):
    def test_filters_dilation_by_default(self):
        from echoclip.data import EchoCLIPDataset
        from echoclip.text import EchoTokenizer

        with tempfile.TemporaryDirectory() as td:
            td = Path(td)
            # Minimal 1x1 png via torch save is awkward; use empty image path
            # and mock by writing a tiny PPM-less approach — use make_demo style.
            from PIL import Image

            img = td / "f0.png"
            Image.new("RGB", (32, 32), color=(10, 10, 10)).save(img)
            manifest = {
                "pairs": [
                    {
                        "image": "f0.png",
                        "text": "THE EJECTION FRACTION IS 55%. THE LEFT VENTRICLE IS SEVERELY DILATED. ",
                        "captions": [
                            "THE EJECTION FRACTION IS ESTIMATED TO BE 55%. ",
                            "THE LEFT VENTRICLE IS SEVERELY DILATED. ",
                        ],
                        "ef": 55.0,
                        "edv": 260.0,
                    }
                ]
            }
            # Need ≥1 sample; dataset doesn't split here
            mp = td / "m.json"
            mp.write_text(json.dumps(manifest), encoding="utf-8")
            tok = EchoTokenizer(context_length=77)
            ds = EchoCLIPDataset(
                mp,
                manifest_dir=td,
                tokenizer=tok,
                caption_mode="random",
                use_edv_captions=False,
                seed=0,
            )
            text = ds._choose_text(ds.pairs[0], 0)
            self.assertIn("EJECTION", text.upper())
            self.assertNotIn("DILATED", text.upper())

            ds_edv = EchoCLIPDataset(
                mp,
                manifest_dir=td,
                tokenizer=tok,
                caption_mode="join",
                use_edv_captions=True,
            )
            text2 = ds_edv._choose_text(ds_edv.pairs[0], 0)
            self.assertIn("DILATED", text2.upper())


class TestCardiacCLIPStub(unittest.TestCase):
    def test_unavailable_without_weights(self):
        from echoclip.cardiacclip_stub import (
            comparison_table_row,
            document_gaps,
            load_cardiacclip_interface,
        )

        r = load_cardiacclip_interface(weights_path=Path("/no/such/weights.pt"))
        self.assertFalse(r.available)
        self.assertIsNone(r.mae)
        row = comparison_table_row(r)
        self.assertEqual(row["clinical_numbers"], "待补充")
        self.assertTrue(len(document_gaps()) >= 2)


class TestAttentionSkeleton(unittest.TestCase):
    def test_toy_runs(self):
        mod = _load_script(
            "analyze_attention_edes",
            ROOT / "scripts" / "analyze_attention_edes.py",
        )
        out = mod.run_toy_analysis(seed=1, n_frames=12)
        self.assertEqual(out["n_frames"], 12)
        self.assertIn("weight_ed", out)
        self.assertIn("待补充", out["note"])


class TestSeedsAggregate(unittest.TestCase):
    def test_mean_sd(self):
        mod = _load_script("run_seeds", ROOT / "scripts" / "run_seeds.py")
        agg = mod.aggregate_metrics(
            {
                "0": {"mae": 8.0, "rmse": 10.0},
                "1": {"mae": 10.0, "rmse": 12.0},
                "2": {"mae": 9.0, "rmse": 11.0},
            }
        )
        self.assertAlmostEqual(agg["mae"]["mean"], 9.0)
        self.assertGreater(agg["mae"]["sd"], 0.0)
        self.assertEqual(agg["mae"]["n"], 3)


class TestTrainPaperGuard(unittest.TestCase):
    def test_paper_flag_in_help(self):
        import subprocess
        import sys

        r = subprocess.run(
            [sys.executable, str(ROOT / "scripts" / "train.py"), "--help"],
            capture_output=True,
            text=True,
            cwd=str(ROOT),
        )
        self.assertEqual(r.returncode, 0)
        self.assertIn("--paper", r.stdout)
        self.assertIn("--use-edv-captions", r.stdout)
        self.assertIn("--no-ef-soft-contrastive", r.stdout)


if __name__ == "__main__":
    unittest.main()
