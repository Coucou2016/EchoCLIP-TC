"""Single source of truth for the paper-build directory layout.

Earlier runs hard-coded ``F:\\EchoCLIP_paper_build``. That volume was
remapped by Windows and every artefact underneath became unreachable, so the
location now comes from one place and can be overridden with the
``ECHOCLIP_BUILD`` environment variable. Nothing outside this module should
contain an absolute build path.
"""

from __future__ import annotations

import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BUILD = REPO_ROOT / "_paper_build_zh"


def build_root() -> Path:
    """Root of all generated artefacts (caches, results, figures, logs)."""
    env = os.environ.get("ECHOCLIP_BUILD")
    root = Path(env).expanduser().resolve() if env else DEFAULT_BUILD
    root.mkdir(parents=True, exist_ok=True)
    return root


def subdir(name: str) -> Path:
    p = build_root() / name
    p.mkdir(parents=True, exist_ok=True)
    return p


def cache_dir() -> Path:
    return subdir("cache")


def results_dir() -> Path:
    return subdir("results_zh")


def figures_dir() -> Path:
    return subdir("figures")


def logs_dir() -> Path:
    return subdir("logs")


def exports_dir() -> Path:
    return subdir("exports")


def sources_dir() -> Path:
    return subdir("sources_zh")


def data_root() -> Path:
    env = os.environ.get("ECHONET_ROOT")
    if env:
        return Path(env).expanduser().resolve()
    return Path(r"E:\Datasets\EchoNet-Dynamic")


def manifest_dir() -> Path:
    return REPO_ROOT / "data" / "echonet_dynamic"


def weights_dir() -> Path:
    return REPO_ROOT / "models" / "echo-clip"


def papers_dir() -> Path:
    p = REPO_ROOT / "papers"
    p.mkdir(parents=True, exist_ok=True)
    return p


if __name__ == "__main__":
    for name in (
        "build_root",
        "cache_dir",
        "results_dir",
        "figures_dir",
        "logs_dir",
        "exports_dir",
        "sources_dir",
        "data_root",
        "manifest_dir",
        "weights_dir",
        "papers_dir",
    ):
        print(f"{name:<14} {globals()[name]()}")
