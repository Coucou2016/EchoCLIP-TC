import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import torch

from echoclip.checkpoint import load_checkpoint
from echoclip.config import EchoCLIPConfig
from echoclip.model import EchoCLIP
from echoclip.utils import config_from_dict


class TestCheckpointConfig(unittest.TestCase):
    def test_config_from_dict_ignores_unknown(self):
        cfg = config_from_dict({"embed_dim": 256, "unknown_key": 99})
        self.assertEqual(cfg.embed_dim, 256)
        self.assertIsInstance(cfg, EchoCLIPConfig)

    def test_external_clip_missing_raises(self):
        """Official ckpt must not soft-load as scratch_fallback."""
        cfg = EchoCLIPConfig(
            embed_dim=32,
            image_size=32,
            context_length=16,
            vision_backbone="simple_cnn",
            text_layers=2,
            text_heads=4,
            text_width=32,
            vocab_size=100,
            pretrained_vision=False,
            temporal_type="none",
        )
        model = EchoCLIP(cfg)
        # Pretend an official tower was saved.
        state = model.state_dict()
        state["external_clip.dummy_weight"] = torch.zeros(1)
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "fake_official.pt"
            torch.save(
                {"epoch": 0, "model_state": state, "model_config": {
                    "embed_dim": 32,
                    "image_size": 32,
                    "context_length": 16,
                    "vision_backbone": "simple_cnn",
                    "text_layers": 2,
                    "text_heads": 4,
                    "text_width": 32,
                    "vocab_size": 100,
                    "pretrained_vision": False,
                    "temporal_type": "none",
                }},
                path,
            )
            env = {**os.environ, "ECHOCLIP_SKIP_HUB": "1"}
            with mock.patch.dict(os.environ, env, clear=False):
                with self.assertRaises(RuntimeError) as ctx:
                    load_checkpoint(path, device="cpu")
            self.assertIn("external_clip", str(ctx.exception).lower())


if __name__ == "__main__":
    unittest.main()
