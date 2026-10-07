"""P0-3: supervised R2/R3/R4 save → load → direct EF inference roundtrip.

Acceptance: train → best.pt → **fresh Python process** → load → TEST inference →
finite EF values → metrics.json, with no in-process head object.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import textwrap
import unittest
from pathlib import Path

import torch

from echoclip.checkpoint import (
    is_supervised_ef_checkpoint,
    load_checkpoint,
    load_ef_regression_checkpoint,
    predict_direct_ef,
    save_ef_regression_checkpoint,
)
from echoclip.config import EchoCLIPConfig
from echoclip.model import EchoCLIP
from echoclip.supervised import LinearEFHead, MLPEFHead, TemporalEFRegressor
from echoclip.supervised_checkpoint import SCHEMA_VERSION, TASK
from echoclip.temporal import TemporalTransformer

ROOT = Path(__file__).resolve().parents[1]


def _backbone_cfg(dim: int = 32) -> dict:
    return {
        "embed_dim": dim,
        "image_size": 32,
        "context_length": 16,
        "vision_backbone": "simple_cnn",
        "text_layers": 2,
        "text_heads": 4,
        "text_width": dim,
        "vocab_size": 100,
        "pretrained_vision": False,
        "temporal_type": "none",
    }


def _save_head(path: Path, kind: str, dim: int = 32) -> Path:
    cfg = _backbone_cfg(dim)
    if kind in ("linear", "ridge"):
        head = LinearEFHead(dim)
        with torch.no_grad():
            head.fc.weight.fill_(0.01)
            head.fc.bias.fill_(50.0)
        save_ef_regression_checkpoint(
            path,
            head_kind="linear",
            head=head,
            backbone_config=cfg,
            train_seed=7,
            epoch=1,
            embed_dim=dim,
            split_seed=7,
            sampler_seed=7,
        )
    elif kind == "mlp":
        head = MLPEFHead(dim, hidden=16)
        cfg = {**cfg, "ef_head_hidden": 16}
        save_ef_regression_checkpoint(
            path,
            head_kind="mlp",
            head=head,
            backbone_config=cfg,
            train_seed=7,
            embed_dim=dim,
            train_cfg={"ef_head_hidden": 16},
        )
    elif kind == "temporal_l1":
        agg = TemporalTransformer(dim, n_layers=1, n_heads=4, max_frames=8)
        reg = TemporalEFRegressor(agg, dim)
        cfg = {
            **cfg,
            "temporal_type": "transformer",
            "temporal_layers": 1,
            "temporal_heads": 4,
            "temporal_max_frames": 8,
        }
        save_ef_regression_checkpoint(
            path,
            head_kind="temporal_l1",
            head=reg,
            backbone_config=cfg,
            train_seed=7,
            embed_dim=dim,
            temporal_state_dict=reg.aggregator.state_dict(),
        )
    else:
        raise ValueError(kind)
    return path


class TestSchemaFields(unittest.TestCase):
    def test_schema_version_and_task(self):
        with tempfile.TemporaryDirectory() as td:
            path = _save_head(Path(td) / "best.pt", "linear")
            raw = torch.load(path, map_location="cpu", weights_only=False)
            self.assertEqual(raw["schema_version"], SCHEMA_VERSION)
            self.assertEqual(raw["format_version"], SCHEMA_VERSION)  # legacy alias
            self.assertEqual(raw["task"], TASK)
            self.assertIn("head_state_dict", raw)
            self.assertIn("backbone_config", raw)
            self.assertIn("backbone_load_source", raw)
            self.assertEqual(raw["train_seed"], 7)
            self.assertEqual(raw["split_seed"], 7)
            self.assertEqual(raw["sampler_seed"], 7)
            self.assertTrue(is_supervised_ef_checkpoint(raw))

    def test_load_checkpoint_refuses_supervised(self):
        with tempfile.TemporaryDirectory() as td:
            path = _save_head(Path(td) / "best.pt", "linear")
            with self.assertRaises(KeyError):
                load_checkpoint(path, device="cpu")


class TestRoundtripAllHeads(unittest.TestCase):
    def _check(self, kind: str):
        with tempfile.TemporaryDirectory() as td:
            path = _save_head(Path(td) / "best.pt", kind)
            bb, head, ckpt = load_ef_regression_checkpoint(path, device="cpu")
            self.assertIsInstance(bb, EchoCLIP)
            frames = torch.randn(3, 4, 3, 32, 32)
            preds = predict_direct_ef(bb, head, frames, head_kind=ckpt["head_kind"])
            self.assertEqual(tuple(preds.shape), (3,))
            self.assertTrue(torch.isfinite(preds).all(), preds)
            return preds

    def test_r2_linear(self):
        self._check("linear")

    def test_r3_mlp(self):
        self._check("mlp")

    def test_r4_temporal(self):
        self._check("temporal_l1")


FRESH_PROCESS_SNIPPET = textwrap.dedent(
    """
    import json, sys
    import torch
    from echoclip.checkpoint import load_ef_regression_checkpoint, predict_direct_ef

    ckpt_path, out_path = sys.argv[1], sys.argv[2]
    bb, head, ck = load_ef_regression_checkpoint(ckpt_path, device="cpu")
    frames = torch.randn(4, 4, 3, 32, 32)
    preds = predict_direct_ef(bb, head, frames, head_kind=ck["head_kind"])
    payload = {
        "finite": bool(torch.isfinite(preds).all()),
        "n": int(preds.numel()),
        "preds": [float(x) for x in preds.tolist()],
        "head_kind": ck["head_kind"],
        "schema_version": ck.get("schema_version"),
        "train_seed": ck.get("train_seed"),
    }
    open(out_path, "w", encoding="utf-8").write(json.dumps(payload, indent=2))
    """
)


class TestFreshProcessLoad(unittest.TestCase):
    def _run_fresh(self, kind: str):
        with tempfile.TemporaryDirectory() as td:
            ckpt = _save_head(Path(td) / "best.pt", kind)
            out = Path(td) / "preds.json"
            script = Path(td) / "run_infer.py"
            script.write_text(FRESH_PROCESS_SNIPPET, encoding="utf-8")
            env = {**os.environ, "PYTHONPATH": str(ROOT)}
            r = subprocess.run(
                [sys.executable, str(script), str(ckpt), str(out)],
                capture_output=True,
                text=True,
                cwd=str(ROOT),
                env=env,
            )
            self.assertEqual(r.returncode, 0, r.stderr)
            data = json.loads(out.read_text(encoding="utf-8"))
            self.assertTrue(data["finite"], data)
            self.assertEqual(data["n"], 4)
            self.assertEqual(data["schema_version"], SCHEMA_VERSION)

    def test_fresh_process_linear(self):
        self._run_fresh("linear")

    def test_fresh_process_temporal(self):
        self._run_fresh("temporal_l1")


def _write_demo_ef_manifest(root: Path, n: int = 8) -> Path:
    """Create tiny PNGs + an EF-labeled manifest (schema/wiring test only)."""
    from PIL import Image

    img_dir = root / "images"
    img_dir.mkdir(parents=True, exist_ok=True)
    pairs = []
    for i in range(n):
        name = f"e{i}.png"
        Image.new("RGB", (48, 48), color=(i * 20 % 255, 40, 90)).save(img_dir / name)
        pairs.append(
            {
                "image": f"images/{name}",
                "text": f"LV EJECTION FRACTION IS {30 + i * 5}%.",
                "ef": float(30 + i * 5),
            }
        )
    manifest = root / "train.json"
    manifest.write_text(json.dumps({"pairs": pairs}), encoding="utf-8")
    return manifest


class TestTrainCLIEndToEnd(unittest.TestCase):
    """Full acceptance for R2: train_supervised → best.pt → eval_clinical direct_ef."""

    def test_r2_train_then_eval_direct_ef(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            manifest = _write_demo_ef_manifest(root, n=8)
            out_dir = root / "ckpt"
            train = subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "scripts" / "train_supervised.py"),
                    "--config",
                    str(ROOT / "configs" / "default.yaml"),
                    "--manifest",
                    str(manifest),
                    "--manifest-dir",
                    str(root),
                    "--output-dir",
                    str(out_dir),
                    "--head",
                    "linear",
                    "--no-official",
                    "--vision-backbone",
                    "simple_cnn",
                    "--video-frames",
                    "2",
                    "--epochs",
                    "1",
                    "--demo",
                    "--seed",
                    "3",
                ],
                capture_output=True,
                text=True,
                cwd=str(ROOT),
            )
            self.assertEqual(train.returncode, 0, train.stderr)
            ckpt = out_dir / "best.pt"
            self.assertTrue(ckpt.is_file())
            raw = torch.load(ckpt, map_location="cpu", weights_only=False)
            self.assertEqual(raw["train_seed"], 3)

            metrics_out = root / "metrics.json"
            ev = subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "scripts" / "eval_clinical.py"),
                    "--config",
                    str(ROOT / "configs" / "default.yaml"),
                    "--checkpoint",
                    str(ckpt),
                    "--manifest",
                    str(manifest),
                    "--manifest-dir",
                    str(root),
                    "--prediction-mode",
                    "direct_ef",
                    "--video-frames",
                    "2",
                    "--batch-size",
                    "4",
                    "--output",
                    str(metrics_out),
                ],
                capture_output=True,
                text=True,
                cwd=str(ROOT),
            )
            self.assertEqual(ev.returncode, 0, ev.stderr)
            self.assertTrue(metrics_out.is_file())
            metrics = json.loads(metrics_out.read_text(encoding="utf-8"))
            self.assertEqual(metrics["prediction_mode"], "direct_ef")
            self.assertIn("mae", metrics)
            self.assertTrue(metrics["n_eval"] >= 2)
            self.assertTrue(float(metrics["mae"]) >= 0.0)
            # Metadata honesty: no --paper ⇒ not an official reproduction.
            self.assertFalse(metrics["official_reproduction_requested"])
            self.assertFalse(metrics["official_reproduction_verified"])


if __name__ == "__main__":
    unittest.main()
