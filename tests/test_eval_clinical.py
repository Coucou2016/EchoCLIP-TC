"""Tests for eval_clinical guards (temporal pool without aggregator)."""

from __future__ import annotations

import importlib.util
import io
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]


def _load_eval_clinical():
    path = ROOT / "scripts" / "eval_clinical.py"
    spec = importlib.util.spec_from_file_location("eval_clinical_mod", path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


class TestEvalClinicalGuards(unittest.TestCase):
    def test_temporal_pool_without_aggregator_fails(self):
        from echoclip.config import EchoCLIPConfig
        from echoclip.model import EchoCLIP

        mod = _load_eval_clinical()
        cfg = EchoCLIPConfig(vision_backbone="simple_cnn", pretrained_vision=False)
        model = EchoCLIP(cfg)
        self.assertIsNone(model.temporal)

        with tempfile.TemporaryDirectory() as tmp:
            man = Path(tmp) / "manifest.json"
            man.write_text(
                '{"pairs":[{"image":"a.png","text":"EF 50%","ef":50},'
                '{"image":"b.png","text":"EF 40%","ef":40}]}',
                encoding="utf-8",
            )
            import sys

            old = sys.argv
            try:
                sys.argv = [
                    "eval_clinical.py",
                    "--init-official",
                    "--manifest",
                    str(man),
                    "--manifest-dir",
                    tmp,
                    "--pool",
                    "temporal",
                    "--config",
                    str(ROOT / "configs" / "default.yaml"),
                ]
                with mock.patch.object(mod, "validate_manifest", return_value=[]):
                    with mock.patch.object(
                        mod.EchoCLIP,
                        "from_official_echo_clip",
                        return_value=model,
                    ):
                        buf = io.StringIO()
                        with redirect_stdout(buf):
                            code = mod.main()
                self.assertEqual(code, 1)
                self.assertIn("temporal aggregator", buf.getvalue())
            finally:
                sys.argv = old

    def test_resolve_pool_auto_without_temporal(self):
        from echoclip.config import EchoCLIPConfig
        from echoclip.model import EchoCLIP

        mod = _load_eval_clinical()
        model = EchoCLIP(
            EchoCLIPConfig(vision_backbone="simple_cnn", pretrained_vision=False)
        )
        self.assertEqual(mod.resolve_pool("auto", model), "frames")
        model.attach_temporal("transformer", n_layers=1, n_heads=4, max_frames=8)
        self.assertEqual(mod.resolve_pool("auto", model), "temporal")


class TestRunProtocolCalLeak(unittest.TestCase):
    def test_m4_same_cal_test_hard_fails_non_demo(self):
        path = ROOT / "scripts" / "run_protocol.py"
        spec = importlib.util.spec_from_file_location("run_protocol_mod", path)
        mod = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        spec.loader.exec_module(mod)

        with tempfile.TemporaryDirectory() as tmp:
            man = Path(tmp) / "test.json"
            man.write_text('{"pairs":[]}', encoding="utf-8")
            args = mock.Mock()
            args.config = ROOT / "configs" / "echonet_dynamic.yaml"
            args.seed = 42
            args.video_frames = 4
            args.sample_strategy = None
            args.batch_size = None
            args.device = None
            args.official_checkpoint = None
            args.no_official = True
            args.vision_backbone = "simple_cnn"
            args.demo = False
            spec_m4 = mod.get_experiment("M4")
            code = mod._eval_experiment(
                args,
                spec_m4,
                test_manifest=man,
                cal_manifest=man,
                manifest_dir=Path(tmp),
                checkpoint=None,
                out_metrics=Path(tmp) / "metrics.json",
                demo=False,
            )
            self.assertEqual(code, 1)

    def test_dry_run_preserves_summary(self):
        path = ROOT / "scripts" / "run_protocol.py"
        spec = importlib.util.spec_from_file_location("run_protocol_dry", path)
        mod = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        spec.loader.exec_module(mod)

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            demo_man = root / "manifest.json"
            demo_man.write_text(
                '{"pairs":[{"image":"x.png","text":"EF 50%"}]}',
                encoding="utf-8",
            )
            protocol = root / "checkpoints" / "protocol"
            protocol.mkdir(parents=True)
            summary = protocol / "summary.json"
            summary.write_text(
                '{"experiments":{"B0":"ok"},"demo":true,"note":"keep-me"}',
                encoding="utf-8",
            )
            import sys

            old = sys.argv
            try:
                sys.argv = [
                    "run_protocol.py",
                    "--demo",
                    "--dry-run",
                    "--experiments",
                    "B0",
                    "--output-root",
                    str(root),
                    "--test-manifest",
                    str(demo_man),
                    "--config",
                    str(ROOT / "configs" / "default.yaml"),
                ]
                # Force demo manifests via --demo; patch resolve to our temp demo
                with mock.patch.object(
                    mod,
                    "_resolve_manifests",
                    return_value=(demo_man, demo_man, demo_man.parent, demo_man),
                ):
                    code = mod.main()
            finally:
                sys.argv = old
            self.assertEqual(code, 0)
            kept = summary.read_text(encoding="utf-8")
            self.assertIn("keep-me", kept)


if __name__ == "__main__":
    unittest.main()
