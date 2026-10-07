"""P0-4: real multi-seed training — seeds must reach both trainers.

Acceptance:
  (a) ``run_seeds.py`` -> ``run_protocol.py`` -> ``train.py`` /
      ``train_supervised.py`` all accept and forward ``--seed``.
  (b) Checkpoints record ``train_seed`` (+ ``split_seed`` / ``sampler_seed``).
  (c) Two different seeds produce non-identical checkpoint hashes.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import subprocess
import sys
import tempfile
import unittest
import unittest.mock
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    h.update(Path(path).read_bytes())
    return h.hexdigest()


def _run_train(script: str, manifest: Path, manifest_dir: Path, out_dir: Path, seed: int):
    return subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / script),
            "--config",
            str(ROOT / "configs" / "default.yaml"),
            "--manifest",
            str(manifest),
            "--manifest-dir",
            str(manifest_dir),
            "--output-dir",
            str(out_dir),
            "--no-official",
            "--vision-backbone",
            "simple_cnn",
            "--video-frames",
            "2",
            "--epochs",
            "1",
            "--device",
            "cpu",
            "--seed",
            str(seed),
        ],
        capture_output=True,
        text=True,
        cwd=str(ROOT),
    )


def _write_demo_manifest(root: Path, n: int = 6) -> Path:
    from PIL import Image

    img_dir = root / "images"
    img_dir.mkdir(parents=True, exist_ok=True)
    pairs = []
    for i in range(n):
        name = f"s{i}.png"
        Image.new("RGB", (48, 48), color=(i * 25 % 255, 30, 120)).save(img_dir / name)
        pairs.append(
            {
                "image": f"images/{name}",
                "text": f"LV EJECTION FRACTION IS {35 + i * 6}%.",
                "ef": float(35 + i * 6),
            }
        )
    manifest = root / "train.json"
    manifest.write_text(json.dumps({"pairs": pairs}), encoding="utf-8")
    return manifest


def _load_module(name: str, relpath: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / relpath)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


class TestSeedFlagDocumented(unittest.TestCase):
    def test_train_help_exposes_seed(self):
        for script in ("train.py", "train_supervised.py", "run_seeds.py"):
            r = subprocess.run(
                [sys.executable, str(ROOT / "scripts" / script), "--help"],
                capture_output=True,
                text=True,
                cwd=str(ROOT),
            )
            self.assertEqual(r.returncode, 0, r.stderr)
            self.assertIn("--seed", r.stdout, script)


class TestSetSeedProvenance(unittest.TestCase):
    def test_set_seed_seeds_all_sources_and_records_mode(self):
        import numpy as np
        import random

        from echoclip.utils import set_seed

        info = set_seed(123, determinism="fast")
        self.assertEqual(info["seed"], 123)
        self.assertEqual(info["determinism_mode"], "fast")
        self.assertIn("cudnn_deterministic", info)
        self.assertIn("cudnn_benchmark", info)

        # random / numpy / torch must be reproducible after re-seeding.
        random.seed(123)
        a_rand = random.random()
        np.random.seed(123)
        a_np = float(np.random.rand())
        torch.manual_seed(123)
        a_torch = torch.rand(1).item()

        set_seed(123, determinism="fast")
        self.assertEqual(random.random(), a_rand)
        self.assertEqual(float(np.random.rand()), a_np)
        self.assertEqual(torch.rand(1).item(), a_torch)

    def test_strict_determinism_sets_cudnn_flags(self):
        from echoclip.utils import set_seed

        info = set_seed(0, determinism="strict")
        self.assertEqual(info["determinism_mode"], "strict")
        self.assertTrue(torch.backends.cudnn.deterministic)
        self.assertFalse(torch.backends.cudnn.benchmark)
        # Restore a neutral default so later tests are unaffected.
        set_seed(0, determinism="fast")

    def test_invalid_mode_falls_back_to_default(self):
        from echoclip.utils import set_seed

        info = set_seed(1, determinism="not-a-mode")
        self.assertIn(info["determinism_mode"], ("fast", "strict"))
        set_seed(0, determinism="fast")


class TestSeedReachesTrainers(unittest.TestCase):
    def test_train_py_records_train_seed(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            manifest = _write_demo_manifest(root)
            out_dir = root / "ckpt_train"
            r = _run_train("train.py", manifest, root, out_dir, seed=11)
            self.assertEqual(r.returncode, 0, r.stderr)
            ckpt = torch.load(
                out_dir / "best.pt", map_location="cpu", weights_only=False
            )
            self.assertEqual(ckpt["train_seed"], 11)
            self.assertEqual(ckpt["split_seed"], 11)
            self.assertEqual(ckpt["sampler_seed"], 11)
            self.assertIn("determinism_mode", ckpt)
            meta = json.loads((out_dir / "train_meta.json").read_text(encoding="utf-8"))
            self.assertEqual(meta["train_seed"], 11)

    def test_train_supervised_records_train_seed(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            manifest = _write_demo_manifest(root)
            out_dir = root / "ckpt_sup"
            r = subprocess.run(
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
                    "--seed",
                    "13",
                    "--demo",
                ],
                capture_output=True,
                text=True,
                cwd=str(ROOT),
            )
            self.assertEqual(r.returncode, 0, r.stderr)
            ckpt = torch.load(
                out_dir / "best.pt", map_location="cpu", weights_only=False
            )
            self.assertEqual(ckpt["train_seed"], 13)
            self.assertEqual(ckpt["split_seed"], 13)
            self.assertEqual(ckpt["sampler_seed"], 13)


class TestDistinctSeedsDiffer(unittest.TestCase):
    def test_two_seeds_give_different_checkpoints(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            manifest = _write_demo_manifest(root)
            a = root / "a"
            b = root / "b"
            r1 = _run_train("train.py", manifest, root, a, seed=1)
            self.assertEqual(r1.returncode, 0, r1.stderr)
            r2 = _run_train("train.py", manifest, root, b, seed=2)
            self.assertEqual(r2.returncode, 0, r2.stderr)
            self.assertNotEqual(_sha256(a / "best.pt"), _sha256(b / "best.pt"))
            ca = torch.load(a / "best.pt", map_location="cpu", weights_only=False)
            cb = torch.load(b / "best.pt", map_location="cpu", weights_only=False)
            self.assertNotEqual(ca["train_seed"], cb["train_seed"])


class TestRunnerForwardsSeed(unittest.TestCase):
    def test_run_seeds_parser_forwards_seed_to_protocol(self):
        run_seeds = _load_module("run_seeds_seedprop", "scripts/run_seeds.py")
        captured: list[list[str]] = []

        class _FakeCompleted:
            returncode = 0

        def _fake_run(cmd, *a, **kw):
            captured.append([str(c) for c in cmd])
            return _FakeCompleted()

        td = tempfile.mkdtemp(prefix="seedprop_")
        self.addCleanup(lambda: __import__("shutil").rmtree(td, ignore_errors=True))
        argv = [
            "run_seeds.py",
            "--seeds",
            "4,5",
            "--experiments",
            "R1",
            "--output-root",
            td,
            "--dry-run",
        ]
        with unittest.mock.patch.object(sys, "argv", argv), unittest.mock.patch.object(
            run_seeds.subprocess, "run", side_effect=_fake_run
        ):
            code = run_seeds.main()
        self.assertEqual(code, 0)
        self.assertTrue(captured, "run_seeds did not invoke run_protocol.py")
        joined = [" ".join(c) for c in captured]
        self.assertTrue(all("run_protocol.py" in s for s in joined), joined)
        for seed, cmd in zip(("4", "5"), captured):
            self.assertIn("--seed", cmd)
            self.assertEqual(cmd[cmd.index("--seed") + 1], seed)

    def test_run_protocol_forwards_seed_to_train_and_supervised(self):
        src = (ROOT / "scripts" / "run_protocol.py").read_text(encoding="utf-8")
        # Both trainer invocations must include a --seed forward.
        self.assertIn('"--seed",\n        str(args.seed),', src)
        for trainer in ("train.py", "train_supervised.py"):
            self.assertIn(trainer, src)


if __name__ == "__main__":
    unittest.main()
