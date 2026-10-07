"""P0-3: ``--prediction-mode direct_ef`` must never touch the prompt path.

A spy asserts the zero-shot prompt pack is never built on the direct-EF path,
and that the CLI module exposes a stable {zeroshot, direct_ef} contract.
"""

from __future__ import annotations

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import torch

ROOT = Path(__file__).resolve().parents[1]


def _load_eval_clinical():
    path = ROOT / "scripts" / "eval_clinical.py"
    spec = importlib.util.spec_from_file_location("eval_clinical_predmode", path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


def _tiny_backbone():
    from echoclip.config import EchoCLIPConfig
    from echoclip.model import EchoCLIP

    cfg = EchoCLIPConfig(
        embed_dim=16,
        image_size=32,
        context_length=8,
        vision_backbone="simple_cnn",
        text_layers=1,
        text_heads=2,
        text_width=16,
        vocab_size=50,
        pretrained_vision=False,
    )
    return EchoCLIP(cfg)


class TestModeAliases(unittest.TestCase):
    def test_normalize(self):
        mod = _load_eval_clinical()
        self.assertEqual(mod._normalize_prediction_mode("direct_ef"), "direct_ef")
        self.assertEqual(
            mod._normalize_prediction_mode("direct_regression"), "direct_ef"
        )
        self.assertEqual(mod._normalize_prediction_mode("zeroshot"), "zeroshot")
        self.assertEqual(mod._normalize_prediction_mode("auto"), "zeroshot")

    def test_cli_accepts_direct_ef_and_legacy(self):
        import subprocess
        import sys

        r = subprocess.run(
            [sys.executable, str(ROOT / "scripts" / "eval_clinical.py"), "--help"],
            capture_output=True,
            text=True,
            cwd=str(ROOT),
        )
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertIn("direct_ef", r.stdout)
        self.assertIn("direct_regression", r.stdout)
        self.assertIn("zeroshot", r.stdout)


class TestDirectEfNeverCallsPromptPath(unittest.TestCase):
    def test_predict_ef_direct_uses_no_prompt_pack(self):
        """Monkeypatch the zero-shot prompt pack to explode if invoked."""
        from echoclip.supervised import LinearEFHead
        from echoclip.zeroshot import EchoCLIPInference

        mod = _load_eval_clinical()
        backbone = _tiny_backbone()
        head = LinearEFHead(16)
        with torch.no_grad():
            head.fc.bias.fill_(45.0)

        batch = {
            "image": torch.randn(3, 4, 3, 32, 32),
            "text": torch.zeros(3, 8, dtype=torch.long),
        }

        class _Loader:
            def __iter__(self):
                yield batch

        with mock.patch.object(
            EchoCLIPInference,
            "_ef_prompt_pack",
            side_effect=AssertionError("zero-shot prompt pack must not be called"),
        ):
            preds = mod.predict_ef_direct(
                backbone, head, _Loader(), head_kind="linear", device="cpu"
            )
        self.assertEqual(preds.shape, (3,))
        self.assertTrue(bool(torch.isfinite(torch.as_tensor(preds)).all()))

    def test_run_split_direct_ef_path(self):
        """_run_split with direct_ef routes to predict_ef_direct (no engine)."""
        from PIL import Image

        from echoclip.supervised import LinearEFHead

        mod = _load_eval_clinical()
        backbone = _tiny_backbone()
        head = LinearEFHead(16)

        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            img_dir = root / "images"
            img_dir.mkdir()
            pairs = []
            for i in range(4):
                Image.new("RGB", (48, 48), color=(i * 30, 10, 10)).save(
                    img_dir / f"a{i}.png"
                )
                pairs.append(
                    {
                        "image": f"images/a{i}.png",
                        "text": f"LV EJECTION FRACTION IS {40 + i}%.",
                        "ef": float(40 + i),
                    }
                )
            manifest = root / "test.json"
            manifest.write_text(json.dumps({"pairs": pairs}), encoding="utf-8")

            args = mock.Mock()
            args.experiment_id = None
            args.allow_annotation_assisted = False
            args.video_frames = 2
            args.seed = 0
            args.batch_size = 2
            cfg = {"image_size": 32, "context_length": 8}

            y_true, y_pred, info = mod._run_split(
                None,  # no engine ⇒ must use the direct path
                manifest,
                root,
                cfg,
                args,
                "supervised_direct",
                sample_strategy="uniform",
                split="test",
                prediction_mode="direct_ef",
                backbone=backbone,
                head=head,
                head_kind="linear",
                device="cpu",
            )
            self.assertEqual(info["prediction_mode"], "direct_ef")
            self.assertEqual(len(y_true), 4)
            self.assertTrue(bool(torch.isfinite(torch.as_tensor(y_pred)).all()))


class TestProtocolRoutesDirectEf(unittest.TestCase):
    def test_protocol_uses_direct_ef_for_r2_r4(self):
        from echoclip.protocol import get_experiment

        src = (ROOT / "scripts" / "run_protocol.py").read_text(encoding="utf-8")
        self.assertIn('"direct_ef"', src)
        # R2–R4 are supervised; R1/R5/R0 remain zero-shot.
        for eid in ("R2", "R3", "R4"):
            self.assertEqual(get_experiment(eid).prediction_mode, "direct_regression")
            self.assertEqual(get_experiment(eid).pool, "supervised")
        self.assertEqual(get_experiment("R5").prediction_mode, "zeroshot")
        self.assertEqual(get_experiment("R0").prediction_mode, "zeroshot")


if __name__ == "__main__":
    unittest.main()
