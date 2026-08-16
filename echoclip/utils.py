"""Reproducibility and config helpers."""

import random
from typing import Any, Dict, Optional

import numpy as np
import torch

from echoclip.config import EchoCLIPConfig


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def config_from_dict(cfg: Dict[str, Any]) -> EchoCLIPConfig:
    """Build EchoCLIPConfig from checkpoint or YAML dict (unknown keys ignored)."""
    fields = EchoCLIPConfig.__dataclass_fields__
    kwargs = {k: cfg[k] for k in fields if k in cfg}
    return EchoCLIPConfig(**kwargs)
