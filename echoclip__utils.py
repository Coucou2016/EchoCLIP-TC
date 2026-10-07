"""Reproducibility and config helpers."""

import os
import random
from typing import Any, Dict, Optional

import numpy as np
import torch

from echoclip.config import EchoCLIPConfig

# Determinism modes recorded in checkpoints / metrics so a run can state
# exactly how reproducible it is. ``strict`` sets cuDNN deterministic+disabled
# benchmark (slower); ``fast`` leaves cuDNN benchmarks on (faster, possibly
# nondeterministic kernels). CPU paths are unaffected by the cuDNN flags.
DETERMINISM_MODES = ("fast", "strict")
DEFAULT_DETERMINISM_MODE = "fast"


def set_seed(seed: int, *, determinism: Optional[str] = None) -> Dict[str, Any]:
    """Seed ``random``, ``numpy``, ``torch`` and all CUDA devices.

    Returns a provenance dict (``seed``, ``determinism_mode``, the cuDNN flags
    actually applied, and whether CUDA is present) so callers can persist
    exactly which mode produced a checkpoint.
    """
    seed = int(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    mode = str(
        determinism or os.environ.get("ECHOCLIP_DETERMINISM", DEFAULT_DETERMINISM_MODE)
    )
    mode = mode.strip().lower()
    if mode not in DETERMINISM_MODES:
        mode = DEFAULT_DETERMINISM_MODE

    cudnn = torch.backends.cudnn
    if mode == "strict":
        cudnn.deterministic = True
        cudnn.benchmark = False
        # Required for deterministic cuBLAS GEMMs on CUDA (PyTorch >= 1.8).
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    else:
        cudnn.deterministic = False
        cudnn.benchmark = True

    return {
        "seed": seed,
        "determinism_mode": mode,
        "cudnn_deterministic": bool(cudnn.deterministic),
        "cudnn_benchmark": bool(cudnn.benchmark),
        "cuda": bool(torch.cuda.is_available()),
    }


def config_from_dict(cfg: Dict[str, Any]) -> EchoCLIPConfig:
    """Build EchoCLIPConfig from checkpoint or YAML dict (unknown keys ignored)."""
    fields = EchoCLIPConfig.__dataclass_fields__
    kwargs = {k: cfg[k] for k in fields if k in cfg}
    return EchoCLIPConfig(**kwargs)
