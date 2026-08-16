import unittest

import torch

from echoclip.config import EchoCLIPConfig
from echoclip.model import EchoCLIP
from echoclip.temporal import AttentionPool, TemporalTransformer, build_temporal, pool_frame_features


class TestTemporalShapes(unittest.TestCase):
    def test_attention_pool_shape(self):
        pool = AttentionPool(dim=32, n_heads=4)
        x = torch.randn(3, 16, 32)
        y = pool(x)
        self.assertEqual(tuple(y.shape), (3, 32))

    def test_transformer_shape_and_grad(self):
        agg = TemporalTransformer(dim=32, n_layers=2, n_heads=4, max_frames=16)
        x = torch.randn(2, 8, 32, requires_grad=True)
        y = agg(x)
        self.assertEqual(tuple(y.shape), (2, 32))
        y.sum().backward()
        self.assertIsNotNone(x.grad)

    def test_build_mean_is_none(self):
        self.assertIsNone(build_temporal("mean", 32))
        self.assertIsNone(build_temporal("none", 32))

    def test_pool_frame_features_normalize(self):
        x = torch.randn(4, 6, 16)
        y = pool_frame_features(x, aggregator=None, normalize=True)
        self.assertEqual(tuple(y.shape), (4, 16))
        norms = y.norm(dim=-1)
        self.assertTrue(torch.allclose(norms, torch.ones_like(norms), atol=1e-5))

    def test_echo_clip_encode_video(self):
        cfg = EchoCLIPConfig(
            vision_backbone="simple_cnn",
            pretrained_vision=False,
            embed_dim=64,
            text_width=64,
            text_layers=2,
            text_heads=4,
            temporal_type="transformer",
            temporal_layers=1,
            temporal_heads=4,
            temporal_max_frames=8,
        )
        model = EchoCLIP(cfg)
        frames = torch.randn(2, 4, 3, 32, 32)
        z = model.encode_video(frames)
        self.assertEqual(tuple(z.shape), (2, 64))
        self.assertTrue(torch.allclose(z.norm(dim=-1), torch.ones(2), atol=1e-4))
        tokens = torch.randint(0, 100, (2, 16))
        img_f, txt_f, scale = model(frames, tokens)
        self.assertEqual(tuple(img_f.shape), (2, 64))
        self.assertEqual(tuple(txt_f.shape), (2, 64))
        self.assertEqual(scale.ndim, 0)


if __name__ == "__main__":
    unittest.main()
