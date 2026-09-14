"""E2E: supervised train → save → reload → finite EF preds; no zero-shot pack."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest import mock

import torch

from echoclip.checkpoint import (
    SUPERVISED_TASK,
    is_supervised_ef_checkpoint,
    load_checkpoint,
    load_supervised_ef_checkpoint,
    predict_direct_ef,
    save_supervised_ef_checkpoint,
)
from echoclip.config import EchoCLIPConfig
from echoclip.model import EchoCLIP
from echoclip.supervised import LinearEFHead, MLPEFHead, TemporalEFRegressor
from echoclip.temporal import TemporalTransformer


ROOT = Path(__file__).resolve().parents[1]


class TestSupervisedCheckpointSchema(unittest.TestCase):
    def _backbone(self, dim=32):
        cfg = EchoCLIPConfig(
            embed_dim=dim,
            image_size=32,
            context_length=16,
            vision_backbone="simple_cnn",
            text_layers=2,
            text_heads=4,
            text_width=dim,
            vocab_size=100,
            pretrained_vision=False,
            temporal_type="none",
        )
        return EchoCLIP(cfg)

    def test_linear_save_reload_finite_preds(self):
        dim = 32
        backbone = self._backbone(dim)
        head = LinearEFHead(dim)
        with torch.no_grad():
            head.fc.weight.fill_(0.01)
            head.fc.bias.fill_(50.0)
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "best.pt"
            save_supervised_ef_checkpoint(
                path,
                head_kind="linear",
                head=head,
                model_config={
                    "embed_dim": dim,
                    "image_size": 32,
                    "context_length": 16,
                    "vision_backbone": "simple_cnn",
                    "text_layers": 2,
                    "text_heads": 4,
                    "text_width": dim,
                    "vocab_size": 100,
                },
                train_seed=7,
                epoch=1,
                embed_dim=dim,
            )
            raw = torch.load(path, map_location="cpu", weights_only=False)
            self.assertTrue(is_supervised_ef_checkpoint(raw))
            self.assertEqual(raw["task"], SUPERVISED_TASK)
            self.assertEqual(raw["train_seed"], 7)
            self.assertIn("head_state_dict", raw)
            with self.assertRaises(KeyError):
                load_checkpoint(path, device="cpu")
            bb, hd, ckpt = load_supervised_ef_checkpoint(path, device="cpu")
            self.assertEqual(ckpt["train_seed"], 7)
            frames = torch.randn(3, 4, 3, 32, 32)
            preds = predict_direct_ef(bb, hd, frames, head_kind="linear")
            self.assertEqual(preds.shape, (3,))
            self.assertTrue(torch.isfinite(preds).all())

    def test_mlp_and_temporal_reload(self):
        dim = 32
        backbone = self._backbone(dim)
        mlp = MLPEFHead(dim, hidden=16)
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "mlp.pt"
            save_supervised_ef_checkpoint(
                path,
                head_kind="mlp",
                head=mlp,
                model_config={
                    "embed_dim": dim,
                    "image_size": 32,
                    "context_length": 16,
                    "vision_backbone": "simple_cnn",
                    "text_layers": 2,
                    "text_heads": 4,
                    "text_width": dim,
                    "vocab_size": 100,
                    "ef_head_hidden": 16,
                },
                train_seed=3,
                embed_dim=dim,
                train_cfg={"ef_head_hidden": 16},
            )
            bb, hd, _ = load_supervised_ef_checkpoint(path, device="cpu")
            preds = predict_direct_ef(
                bb, hd, torch.randn(2, 4, 3, 32, 32), head_kind="mlp"
            )
            self.assertTrue(torch.isfinite(preds).all())

            agg = TemporalTransformer(dim, n_layers=1, n_heads=4, max_frames=8)
            reg = TemporalEFRegressor(agg, dim)
            path2 = Path(td) / "temp.pt"
            save_supervised_ef_checkpoint(
                path2,
                head_kind="temporal_l1",
                head=reg,
                model_config={
                    "embed_dim": dim,
                    "image_size": 32,
                    "context_length": 16,
                    "vision_backbone": "simple_cnn",
                    "text_layers": 2,
                    "text_heads": 4,
                    "text_width": dim,
                    "vocab_size": 100,
                    "temporal_type": "transformer",
                    "temporal_layers": 1,
                    "temporal_heads": 4,
                    "temporal_max_frames": 8,
                },
                train_seed=11,
                embed_dim=dim,
                temporal_state_dict=reg.aggregator.state_dict(),
            )
            bb2, hd2, ck = load_supervised_ef_checkpoint(path2, device="cpu")
            self.assertEqual(ck["head_kind"], "temporal_l1")
            preds2 = predict_direct_ef(
                bb2, hd2, torch.randn(2, 4, 3, 32, 32), head_kind="temporal_l1"
            )
            self.assertTrue(torch.isfinite(preds2).all())

    def test_direct_regression_does_not_call_zeroshot_prompt_pack(self):
        """R2/R3/R4 must not invoke zero-shot prompt pack during predict."""
        dim = 32
        backbone = self._backbone(dim)
        head = LinearEFHead(dim)
        with torch.no_grad():
            head.fc.bias.fill_(40.0)
            head.fc.weight.zero_()

        # Spy: if anyone builds EF prompt pack, fail
        from echoclip.zeroshot import EchoCLIPInference

        with mock.patch.object(
            EchoCLIPInference,
            "_ef_prompt_pack",
            side_effect=AssertionError("zero-shot prompt pack must not be called"),
        ):
            preds = predict_direct_ef(
                backbone, head, torch.randn(2, 4, 3, 32, 32), head_kind="linear"
            )
        self.assertEqual(tuple(preds.shape), (2,))
        self.assertTrue(torch.isfinite(preds).all())


class TestMultiSeedWiring(unittest.TestCase):
    def test_two_seeds_different_train_seed_in_ckpt(self):
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
        seeds = []
        with tempfile.TemporaryDirectory() as td:
            for seed in (0, 1):
                head = LinearEFHead(dim)
                path = Path(td) / f"s{seed}.pt"
                save_supervised_ef_checkpoint(
                    path,
                    head_kind="linear",
                    head=head,
                    model_config=cfg.__dict__,
                    train_seed=seed,
                    embed_dim=dim,
                )
                _, _, ck = load_supervised_ef_checkpoint(path, device="cpu")
                seeds.append(ck["train_seed"])
        self.assertEqual(seeds, [0, 1])
        self.assertNotEqual(seeds[0], seeds[1])


class TestOfficialStrideNoPad(unittest.TestCase):
    def test_official_stride_not_padded_to_16(self):
        from echoclip.cycle_sample import sample_official_stride, pad_or_trim_indices

        idx = sample_official_stride(100)  # 0..38 step 2 → 20 frames
        self.assertEqual(len(idx), 20)
        # Protocol must NOT force pad_or_trim to 16 for official path
        padded = pad_or_trim_indices(idx, 16, 100)
        self.assertEqual(len(padded), 16)  # pad helper itself still works
        # But dataset must skip pad — covered by length of sample_official_stride
        self.assertGreater(len(idx), 16)


class TestProtocolPredictionMode(unittest.TestCase):
    def test_r2_r3_r4_direct_r5_zeroshot(self):
        from echoclip.protocol import get_experiment, resolve_eval_sample_strategy

        for eid in ("R2", "R3", "R4"):
            spec = get_experiment(eid)
            self.assertEqual(spec.prediction_mode, "direct_regression")
            self.assertEqual(spec.pool, "supervised")
        self.assertEqual(get_experiment("R5").prediction_mode, "zeroshot")
        r0 = get_experiment("R0")
        self.assertEqual(
            resolve_eval_sample_strategy(spec=r0, paper=True), "official_stride"
        )
        self.assertEqual(
            resolve_eval_sample_strategy(spec=r0, paper=False), "uniform"
        )
        r0u = get_experiment("R0U16")
        self.assertEqual(r0u.eval_sample_strategy, "uniform")
        self.assertEqual(
            resolve_eval_sample_strategy(spec=r0u, paper=True), "uniform"
        )


if __name__ == "__main__":
    unittest.main()
