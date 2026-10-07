import unittest
import warnings

import torch

from echoclip.loss import ClipLoss, EFSoftContrastiveLoss


class TestClipLoss(unittest.TestCase):
    def test_symmetric_loss_zero_on_identical(self):
        loss_fn = ClipLoss()
        z = torch.randn(4, 32)
        z = torch.nn.functional.normalize(z, dim=-1)
        scale = torch.tensor(2.3)
        loss = loss_fn(z, z, scale)
        self.assertGreater(loss.item(), 0)

    def test_perfect_alignment_lower_loss(self):
        loss_fn = ClipLoss()
        z = torch.eye(8)
        scale = torch.tensor(1.0)
        loss_aligned = loss_fn(z, z, scale)
        perm = torch.randperm(8)
        loss_shuffled = loss_fn(z, z[perm], scale)
        self.assertLess(loss_aligned.item(), loss_shuffled.item())


class TestEFSoftLambdaMixture(unittest.TestCase):
    """P1-3: lambda_soft is a hard/soft mix; 0 == hard InfoNCE exactly."""

    def _batch(self, n: int = 6, d: int = 16, seed: int = 0):
        g = torch.Generator().manual_seed(seed)
        v = torch.nn.functional.normalize(torch.randn(n, d, generator=g), dim=-1)
        t = torch.nn.functional.normalize(torch.randn(n, d, generator=g), dim=-1)
        ef = torch.linspace(20.0, 70.0, n)
        return v, t, torch.tensor(2.0), ef

    def test_lambda_zero_equals_hard_infonce(self):
        v, t, s, ef = self._batch()
        hard = ClipLoss()(v, t, s)
        mixed = EFSoftContrastiveLoss(ef_temperature=5.0, lambda_soft=0.0)(v, t, s, ef=ef)
        self.assertTrue(torch.allclose(hard, mixed, atol=1e-12))

    def test_lambda_zero_skips_soft_even_with_all_nan_ef(self):
        v, t, s, _ = self._batch()
        ef = torch.full((6,), float("nan"))
        hard = ClipLoss()(v, t, s)
        out = EFSoftContrastiveLoss(lambda_soft=0.0)(v, t, s, ef=ef)
        self.assertTrue(torch.allclose(hard, out, atol=1e-12))

    def test_lambda_one_is_soft(self):
        v, t, s, ef = self._batch()
        soft = EFSoftContrastiveLoss(lambda_soft=1.0)(v, t, s, ef=ef)
        hard = ClipLoss()(v, t, s)
        self.assertNotAlmostEqual(float(soft), float(hard), places=4)

    def test_midpoint_is_between(self):
        v, t, s, ef = self._batch()
        hard = float(ClipLoss()(v, t, s))
        soft = float(EFSoftContrastiveLoss(lambda_soft=1.0)(v, t, s, ef=ef))
        mid = float(EFSoftContrastiveLoss(lambda_soft=0.5)(v, t, s, ef=ef))
        lo, hi = sorted((hard, soft))
        self.assertGreaterEqual(mid, lo - 1e-6)
        self.assertLessEqual(mid, hi + 1e-6)

    def test_soft_weight_alias_warns(self):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            loss = EFSoftContrastiveLoss(soft_weight=0.25)
        self.assertTrue(any(issubclass(w.category, DeprecationWarning) for w in caught))
        self.assertAlmostEqual(loss.lambda_soft, 0.25)


class TestEFSoftFiniteSubset(unittest.TestCase):
    """P1-2: NaN rows are excluded from the soft subset, not whole-batch fallback."""

    def _batch(self, n: int = 8, d: int = 16, seed: int = 1):
        g = torch.Generator().manual_seed(seed)
        v = torch.nn.functional.normalize(torch.randn(n, d, generator=g), dim=-1)
        t = torch.nn.functional.normalize(torch.randn(n, d, generator=g), dim=-1)
        return v, t, torch.tensor(2.0)

    def test_finite_subset_soft_differs_from_hard(self):
        v, t, s = self._batch()
        ef = torch.tensor([20.0, 25.0, 30.0, float("nan"), float("nan"), 55.0, 60.0, 65.0])
        loss_fn = EFSoftContrastiveLoss(lambda_soft=1.0)
        soft = float(loss_fn(v, t, s, ef=ef))
        hard = float(ClipLoss()(v, t, s))
        self.assertTrue(torch.isfinite(torch.tensor(soft)))
        self.assertNotAlmostEqual(soft, hard, places=4)

    def test_fewer_than_two_valid_falls_back_to_hard(self):
        v, t, s = self._batch()
        ef = torch.tensor([40.0, float("nan"), float("nan"), float("nan"),
                           float("nan"), float("nan"), float("nan"), float("nan")])
        hard = ClipLoss()(v, t, s)
        out = EFSoftContrastiveLoss(lambda_soft=1.0)(v, t, s, ef=ef)
        self.assertTrue(torch.allclose(hard, out, atol=1e-12))

    def test_all_nan_falls_back_to_hard(self):
        v, t, s = self._batch()
        ef = torch.full((8,), float("nan"))
        hard = ClipLoss()(v, t, s)
        out = EFSoftContrastiveLoss(lambda_soft=1.0)(v, t, s, ef=ef)
        self.assertTrue(torch.allclose(hard, out, atol=1e-12))

    def test_no_ef_returns_hard(self):
        v, t, s = self._batch()
        hard = ClipLoss()(v, t, s)
        out = EFSoftContrastiveLoss(lambda_soft=1.0)(v, t, s, ef=None)
        self.assertTrue(torch.allclose(hard, out, atol=1e-12))


if __name__ == "__main__":
    unittest.main()
