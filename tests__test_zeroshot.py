import unittest

import torch
import torch.nn.functional as F

from echoclip.zeroshot import (
    _as_frame_batch,
    _as_prompt_batch,
    compute_binary_score,
    compute_regression_score,
)


class TestZeroshotShapes(unittest.TestCase):
    def test_1d_frame_embedding(self):
        frame = torch.randn(64)
        prompts = torch.randn(5, 64)
        score = compute_binary_score(frame, prompts)
        self.assertEqual(score.shape, torch.Size([1]))

    def test_2d_frame_embeddings(self):
        frames = torch.randn(3, 64)
        prompts = torch.randn(4, 64)
        score = compute_binary_score(frames, prompts)
        self.assertEqual(score.shape, torch.Size([1]))

    def test_batch_helpers(self):
        f = _as_frame_batch(torch.randn(64))
        self.assertEqual(f.shape, (1, 1, 64))
        p = _as_prompt_batch(torch.randn(3, 64))
        self.assertEqual(p.shape, (1, 3, 64))


class TestB0M1PoolingSemantics(unittest.TestCase):
    """Protocol fairness: B0 ranks per frame; M1 mean-pools embeddings first.

    API note: 2D tensors to ``compute_regression_score`` mean ``(T, D)`` for a
    *single* video (batch dim added as 1). Batched video vectors ``(B, D)``
    must be passed as ``(B, 1, D)`` (see ``EchoCLIPInference.zero_shot_ef_batch``).
    """

    def _prompts_and_values(self, dim: int = 8, n: int = 20):
        prompts = F.normalize(torch.randn(n, dim), dim=-1)
        values = list(range(15, 15 + n))
        return prompts, values

    def test_bd_batch_requires_explicit_t_axis(self):
        prompts, values = self._prompts_and_values()
        z = F.normalize(torch.randn(2, 8), dim=-1)
        # Correct batched video-vector path (M1/M2):
        a = compute_regression_score(z.unsqueeze(1), prompts, values)
        self.assertEqual(tuple(a.shape), (2,))
        # 2D without unsqueeze is interpreted as one video with T=B frames
        single = compute_regression_score(z, prompts, values)
        self.assertEqual(tuple(single.shape), (1,))

    def test_t1_b0_equals_m1(self):
        prompts, values = self._prompts_and_values()
        frame = F.normalize(torch.randn(1, 1, 8), dim=-1)
        b0 = compute_regression_score(frame, prompts, values)
        m1 = compute_regression_score(frame.mean(dim=1).unsqueeze(1), prompts, values)
        self.assertTrue(torch.allclose(b0, m1))

    def test_identical_frames_b0_equals_m1(self):
        prompts, values = self._prompts_and_values()
        base = F.normalize(torch.randn(1, 8), dim=-1)
        frames = base.unsqueeze(1).expand(1, 5, 8).contiguous()
        b0 = compute_regression_score(frames, prompts, values)
        m1 = compute_regression_score(frames.mean(dim=1).unsqueeze(1), prompts, values)
        self.assertTrue(torch.allclose(b0, m1, atol=1e-5))

    def test_rank_crossing_b0_differs_from_m1(self):
        """Construct frames whose per-frame rankings disagree → B0 != M1."""
        prompts = torch.tensor([[1.0, 0.0], [0.0, 1.0]], dtype=torch.float32)
        values = [30.0, 70.0]
        frames = torch.tensor(
            [[[1.0, 0.0], [0.0, 1.0]]],
            dtype=torch.float32,
        )
        b0 = compute_regression_score(frames, prompts, values)
        m1 = compute_regression_score(frames.mean(dim=1).unsqueeze(1), prompts, values)
        self.assertFalse(torch.allclose(b0, m1))


if __name__ == "__main__":
    unittest.main()
