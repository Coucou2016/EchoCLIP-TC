"""P0-7: repo integrity audit — real tree passes, broken refs fail loudly."""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _load_audit():
    path = ROOT / "scripts" / "audit_repo_integrity.py"
    spec = importlib.util.spec_from_file_location("audit_repo_integrity_mod", path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


class TestRealTreePasses(unittest.TestCase):
    def test_audit_ok_on_tree_layout(self):
        mod = _load_audit()
        verdict = mod.audit(ROOT)
        self.assertTrue(verdict["ok"], verdict)
        self.assertEqual(verdict["missing_required"], [])
        self.assertEqual(verdict["missing_referenced"], {})

    def test_cli_exit_zero(self):
        r = subprocess.run(
            [sys.executable, str(ROOT / "scripts" / "audit_repo_integrity.py"), "--json"],
            capture_output=True,
            text=True,
            cwd=str(ROOT),
        )
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
        data = json.loads(r.stdout)
        self.assertTrue(data["ok"])


class TestDetectsMissingReferencedPath(unittest.TestCase):
    def _fake_root(self, td: Path) -> Path:
        root = Path(td)
        (root / "README.md").write_text(
            "# test\n\nSee `scripts/does_not_exist.py` for details.\n",
            encoding="utf-8",
        )
        (root / "DATA.md").write_text("# data\n", encoding="utf-8")
        (root / "scripts").mkdir()
        (root / "scripts" / "exists.py").write_text("x = 1\n", encoding="utf-8")
        return root

    def test_broken_backticked_ref_fails(self):
        mod = _load_audit()
        with tempfile.TemporaryDirectory() as td:
            root = self._fake_root(Path(td))
            verdict = mod.audit(root)
            self.assertFalse(verdict["ok"])
            self.assertIn("README.md", verdict["missing_referenced"])
            self.assertIn(
                "scripts/does_not_exist.py",
                verdict["missing_referenced"]["README.md"],
            )

    def test_missing_required_path_fails(self):
        mod = _load_audit()
        with tempfile.TemporaryDirectory() as td:
            root = self._fake_root(Path(td))
            verdict = mod.audit(root)
            self.assertTrue(verdict["missing_required"])

    def test_planned_heading_exempts(self):
        mod = _load_audit()
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            (root / "README.md").write_text(
                "# test\n\n"
                "## Planned\n\n"
                "```\n"
                "scripts/not_yet_built/  # planned module\n"
                "  planned_thing.py\n"
                "```\n",
                encoding="utf-8",
            )
            (root / "DATA.md").write_text("# data\n", encoding="utf-8")
            verdict = mod.audit(root)
            self.assertEqual(verdict["missing_referenced"], {}, verdict)


class TestExtractors(unittest.TestCase):
    def test_ignores_urls_and_commands(self):
        mod = _load_audit()
        text = (
            "Run `python scripts/train.py` then `pip install -r requirements.txt`.\n"
            "See https://github.com/x/y and `Coucou2016/EchoCLIP-TC`.\n"
        )
        found = mod.extract_inline_paths(text)
        self.assertEqual(found, set(), found)

    def test_directory_context_in_layout(self):
        mod = _load_audit()
        text = "```\nechoclip/\n\n  model.py\n\n  data.py\n```\n"
        layout, planned = mod.extract_layout_paths(text)
        self.assertIn("echoclip/model.py", layout)
        self.assertIn("echoclip/data.py", layout)
        self.assertEqual(planned, set())


if __name__ == "__main__":
    unittest.main()
