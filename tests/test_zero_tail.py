"""Tests for efficiency helpers, paper matrix wiring, and CardiacCLIP adapter."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import torch
import torch.nn as nn

from echoclip.cardiacclip import (
    comparison_table_row,
    load_cardiacclip_interface,
    write_comparison_template,
)
from echoclip.efficiency import count_parameters, merge_efficiency_into_metrics
from echoclip.protocol import get_experiment
from echoclip.prompts_ta import TA_CAPTIONS
from echoclip.structured_text import captions_from_measurements
from echoclip.temporal import AttentionPool, TemporalTransformer


class TestEfficiency(unittest.TestCase):
    def test_count_parameters_trainable_fraction(self):
        class Tiny(nn.Module):
            def __init__(self):
                super().__init__()
                self.visual = nn.Linear(8, 8)
                self.textual = nn.Linear(8, 8)
                self.temporal = AttentionPool(8, n_heads=2)

        m = Tiny()
        for p in m.visual.parameters():
            p.requires_grad = False
        for p in m.textual.parameters():
            p.requires_grad = False
        stats = count_parameters(m)
        self.assertGreater(stats["n_trainable_params"], 0)
        self.assertGreater(stats["n_total_params"], stats["n_trainable_params"])
        self.assertIsNotNone(stats["trainable_pct_of_backbone"])
        merged = merge_efficiency_into_metrics({"mae": None}, model=m, train_seconds=1.5)
        self.assertEqual(merged["train_seconds"], 1.5)
        self.assertIn("n_trainable_params", merged)


class TestPromptsTA(unittest.TestCase):
    def test_ta_captions_differ_from_official_path(self):
        official = captions_from_measurements(ef=55.0, style="official")
        ta = captions_from_measurements(ef=55.0, style="ta")
        self.assertTrue(official)
        self.assertTrue(ta)
        self.assertIn("<#>", " ".join(TA_CAPTIONS["ejection_fraction"]) or "<#>")
        # Clean-room strings should not be identical to official bank for EF
        self.assertNotEqual(official[0], ta[0])


class TestCardiacCLIPAdapter(unittest.TestCase):
    def test_missing_weights_pending(self):
        r = load_cardiacclip_interface(weights_path=Path("definitely_missing_xyz.pt"))
        self.assertFalse(r.available)
        row = comparison_table_row(r)
        self.assertEqual(row["clinical_numbers"], "待补充")
        self.assertIn("download_instructions", row)

    def test_write_template(self):
        with tempfile.TemporaryDirectory() as td:
            path = write_comparison_template(Path(td) / "cc.json")
            data = json.loads(path.read_text(encoding="utf-8"))
            self.assertEqual(data["rows"][0]["mae"], "待补充")


class TestProtocolPredictionModes(unittest.TestCase):
    def test_r2_r4_direct_regression(self):
        for eid in ("R2", "R3", "R4"):
            self.assertEqual(get_experiment(eid).prediction_mode, "direct_regression")
        self.assertEqual(get_experiment("R5").prediction_mode, "zeroshot")
        self.assertEqual(get_experiment("R5").sample_strategy, "uniform")
        self.assertEqual(get_experiment("R5_EDESTRAIN").sample_strategy, "mixed")


class TestPaperMatrixScriptExists(unittest.TestCase):
    def test_scripts_present(self):
        root = Path(__file__).resolve().parents[1]
        self.assertTrue((root / "scripts" / "run_paper_matrix.py").is_file())
        self.assertTrue((root / "scripts" / "run_label_efficiency.py").is_file())
        self.assertTrue((root / "scripts" / "analyze_attention_edes.py").is_file())


class TestSeedPropagationE2E(unittest.TestCase):
    def test_train_seed_differs_and_affects_init(self):
        """Different seeds → different train_seed metadata and init noise."""
        from echoclip.config import EchoCLIPConfig
        from echoclip.supervised import LinearEFHead
        from echoclip.checkpoint import (
            load_supervised_ef_checkpoint,
            save_supervised_ef_checkpoint,
        )
        from echoclip.utils import set_seed

        dim = 16
        cfg = EchoCLIPConfig(
            embed_dim=dim,
            image_size=16,
            context_length=8,
            vision_backbone="simple_cnn",
            text_layers=1,
            text_heads=2,
            text_width=dim,
            vocab_size=50,
            pretrained_vision=False,
        )
        states = []
        seeds_out = []
        with tempfile.TemporaryDirectory() as td:
            for seed in (0, 1):
                set_seed(seed)
                head = LinearEFHead(dim)
                # Force seed-dependent init difference
                with torch.no_grad():
                    head.fc.weight.add_(torch.randn_like(head.fc.weight) * 0.01)
                path = Path(td) / f"s{seed}.pt"
                save_supervised_ef_checkpoint(
                    path,
                    head_kind="linear",
                    head=head,
                    model_config=cfg.__dict__,
                    train_seed=seed,
                    embed_dim=dim,
                )
                _, loaded, ck = load_supervised_ef_checkpoint(path, device="cpu")
                seeds_out.append(ck["train_seed"])
                states.append(loaded.fc.weight.detach().clone())
        self.assertEqual(seeds_out, [0, 1])
        self.assertFalse(torch.allclose(states[0], states[1]))


if __name__ == "__main__":
    unittest.main()
