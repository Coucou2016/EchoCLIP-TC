"""P0-8: config loading expands ``${ENV_VAR}`` and ``~`` recursively."""

from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path

from echoclip.config_io import (
    expand_config,
    expand_path_str,
    load_yaml_config,
)

ROOT = Path(__file__).resolve().parents[1]


class TestEnvironmentVariableExpansion(unittest.TestCase):
    def test_environment_variable_expansion(self):
        with tempfile.TemporaryDirectory() as td:
            cfg_path = Path(td) / "cfg.yaml"
            cfg_path.write_text(
                "echonet_root: ${ECHOCLIP_TEST_ROOT}\n"
                "manifest_dir: ${ECHOCLIP_TEST_ROOT}/data\n"
                "nested:\n"
                "  paths:\n"
                "    - ${ECHOCLIP_TEST_ROOT}/a.json\n"
                "    - plain/rel.json\n"
                "num: 7\n",
                encoding="utf-8",
            )
            os.environ["ECHOCLIP_TEST_ROOT"] = str(Path(td) / "echonet")
            try:
                cfg = load_yaml_config(cfg_path)
            finally:
                os.environ.pop("ECHOCLIP_TEST_ROOT", None)

            self.assertEqual(cfg["echonet_root"], str(Path(td) / "echonet"))
            self.assertEqual(
                cfg["manifest_dir"], f"{Path(td) / 'echonet'}/data"
            )
            self.assertEqual(
                cfg["nested"]["paths"][0], f"{Path(td) / 'echonet'}/a.json"
            )
            self.assertEqual(cfg["nested"]["paths"][1], "plain/rel.json")
            self.assertEqual(cfg["num"], 7)
            self.assertNotIn("${", cfg["echonet_root"])

    def test_echonet_dynamic_config_expands(self):
        os.environ["ECHONET_ROOT"] = "/tmp/echonet_root_probe"
        try:
            cfg = load_yaml_config(ROOT / "configs" / "echonet_dynamic.yaml")
        finally:
            os.environ.pop("ECHONET_ROOT", None)
        self.assertEqual(cfg["echonet_root"], "/tmp/echonet_root_probe")
        self.assertEqual(cfg["manifest_dir"], "/tmp/echonet_root_probe")

    def test_expanduser(self):
        cfg = expand_config({"p": "~/echo/data"})
        self.assertFalse(cfg["p"].startswith("~"))
        self.assertTrue(cfg["p"].startswith(str(Path.home())))

    def test_strict_mode_raises_on_unset(self):
        os.environ.pop("ECHOCLIP_DEFINITELY_UNSET", None)
        with tempfile.TemporaryDirectory() as td:
            cfg_path = Path(td) / "cfg.yaml"
            cfg_path.write_text(
                "p: ${ECHOCLIP_DEFINITELY_UNSET}/x\n", encoding="utf-8"
            )
            with self.assertRaises(KeyError):
                load_yaml_config(cfg_path, strict=True)

    def test_non_strict_leaves_unset_placeholder(self):
        os.environ.pop("ECHOCLIP_DEFINITELY_UNSET", None)
        out = expand_config({"p": "${ECHOCLIP_DEFINITELY_UNSET}/x"})
        self.assertIn("ECHOCLIP_DEFINITELY_UNSET", out["p"])

    def test_literal_dollar_via_double_dollar(self):
        out = expand_config({"s": "cost $$5"})
        self.assertEqual(out["s"], "cost $5")

    def test_record_provenance(self):
        with tempfile.TemporaryDirectory() as td:
            cfg_path = Path(td) / "cfg.yaml"
            cfg_path.write_text("a: 1\n", encoding="utf-8")
            cfg = load_yaml_config(cfg_path, record_provenance=True)
            self.assertTrue(cfg["_config_env_expanded"])
            self.assertEqual(cfg["_config_env_source"], str(cfg_path))

    def test_empty_config_returns_empty_dict(self):
        with tempfile.TemporaryDirectory() as td:
            cfg_path = Path(td) / "empty.yaml"
            cfg_path.write_text("", encoding="utf-8")
            self.assertEqual(load_yaml_config(cfg_path), {})

    def test_expand_path_str_optional(self):
        self.assertIsNone(expand_path_str(None))
        self.assertEqual(expand_path_str("a/b"), "a/b")


class TestLoadersRouteThroughConfigIo(unittest.TestCase):
    def test_train_scripts_use_load_yaml_config(self):
        for rel in (
            "scripts/train.py",
            "scripts/train_supervised.py",
            "scripts/run_protocol.py",
            "scripts/eval_clinical.py",
        ):
            src = (ROOT / rel).read_text(encoding="utf-8")
            self.assertIn("load_yaml_config", src, rel)


if __name__ == "__main__":
    unittest.main()
