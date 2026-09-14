"""Tests for supervised EF baselines and soft contrastive loss."""

from __future__ import annotations

import unittest

import torch

from echoclip.loss import EFSoftContrastiveLoss
from echoclip.supervised import supervised_api_smoke


class TestSupervisedAPI(unittest.TestCase):
    def test_toy_tensor_hooks(self):
        out = supervised_api_smoke(dim=32, t=4, batch=3)
        self.assertIn("ridge_mae", out)
        self.assertIn("mlp_loss", out)
        self.assertIn("temporal_l1_loss", out)
        self.assertTrue(all(isinstance(v, float) for v in out.values()))


class TestEFSoftContrastive(unittest.TestCase):
    def test_soft_loss_finite(self):
        b, d = 4, 16
        v = torch.nn.functional.normalize(torch.randn(b, d), dim=-1)
        t = torch.nn.functional.normalize(torch.randn(b, d), dim=-1)
        scale = torch.tensor(2.0)
        ef = torch.tensor([30.0, 32.0, 60.0, 62.0])
        loss = EFSoftContrastiveLoss(ef_temperature=5.0)(v, t, scale, ef=ef)
        self.assertTrue(torch.isfinite(loss))
        self.assertGreater(float(loss), 0.0)

    def test_fallback_without_ef(self):
        b, d = 3, 8
        v = torch.randn(b, d)
        t = torch.randn(b, d)
        scale = torch.tensor(1.0)
        loss = EFSoftContrastiveLoss()(v, t, scale, ef=None)
        self.assertTrue(torch.isfinite(loss))

    def test_partial_nan_ef_keeps_batch(self):
        b, d = 4, 16
        v = torch.nn.functional.normalize(torch.randn(b, d), dim=-1)
        t = torch.nn.functional.normalize(torch.randn(b, d), dim=-1)
        scale = torch.tensor(2.0)
        ef = torch.tensor([30.0, float("nan"), 60.0, 62.0])
        loss = EFSoftContrastiveLoss(ef_temperature=5.0)(v, t, scale, ef=ef)
        self.assertTrue(torch.isfinite(loss))
        self.assertGreater(float(loss), 0.0)


if __name__ == "__main__":
    unittest.main()
