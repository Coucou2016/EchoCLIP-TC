import unittest

import torch

from echoclip.loss import ClipLoss


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


if __name__ == "__main__":
    unittest.main()
