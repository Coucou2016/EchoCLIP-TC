"""Golden tests: aggregation matches echonet/echo_CLIP utils (fixed tensors)."""

from __future__ import annotations

import unittest

import numpy as np
import torch
import torch.nn.functional as F

from echoclip.cycle_sample import sample_official_stride
from echoclip.official_parity import (
    PARITY_GAPS,
    compute_regression_metric_official,
    official_stride_indices,
)
from echoclip.protocol import OFFICIAL_EF_VALUES
from echoclip.zeroshot import compute_binary_score, compute_regression_score


class TestOfficialB0Aggregation(unittest.TestCase):
    def test_parity_gaps_documented(self):
        self.assertGreaterEqual(len(PARITY_GAPS), 3)

    def test_official_ef_grid_0_100(self):
        self.assertEqual(OFFICIAL_EF_VALUES[0], 0)
        self.assertEqual(OFFICIAL_EF_VALUES[-1], 100)
        self.assertEqual(len(OFFICIAL_EF_VALUES), 101)

    def test_stride_matches_zero_shot_example(self):
        for t in (5, 20, 40, 41, 100):
            a = official_stride_indices(t)
            b = sample_official_stride(t)
            np.testing.assert_array_equal(a, b)
            np.testing.assert_array_equal(a, np.arange(0, min(40, t), 2))

    def test_aggregation_matches_official_utils_fixed_seed(self):
        torch.manual_seed(0)
        n, t, d, c = 2, 8, 16, 101  # one template × EF 0–100
        video = F.normalize(torch.randn(n, t, d), dim=-1)
        prompts = F.normalize(torch.randn(c, d), dim=-1)
        values = list(range(c))
        ref = compute_regression_metric_official(video, prompts, values)
        ours = compute_regression_score(video, prompts, values)
        self.assertTrue(torch.allclose(ours, ref, atol=1e-6, rtol=1e-5), (ours, ref))

    def test_aggregation_two_templates_like_official(self):
        """Official builds 2 templates × 0..100 = 202 candidates."""
        torch.manual_seed(7)
        n, t, d = 1, 20, 32
        c = 202
        video = F.normalize(torch.randn(n, t, d), dim=-1)
        prompts = F.normalize(torch.randn(c, d), dim=-1)
        values = [i % 101 for i in range(c)]
        ref = compute_regression_metric_official(video, prompts, values)
        ours = compute_regression_score(video, prompts, values)
        self.assertTrue(torch.allclose(ours, ref, atol=1e-6, rtol=1e-5))
        # top 20% of 202 → 40
        self.assertEqual(int(202 * 0.2), 40)

    def test_binary_mean_matches_official_shape(self):
        torch.manual_seed(1)
        video = F.normalize(torch.randn(1, 4, 8), dim=-1)
        prompts = F.normalize(torch.randn(3, 8), dim=-1)
        # Official: (video @ prompts.T).mean(-1).mean(-1)
        ref = (video @ prompts.T).mean(dim=-1).mean(dim=-1)
        ours = compute_binary_score(video, prompts)
        self.assertTrue(torch.allclose(ours, ref, atol=1e-6))


if __name__ == "__main__":
    unittest.main()
