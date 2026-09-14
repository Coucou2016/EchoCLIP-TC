"""Parameter-count and optional timing helpers for EchoCLIP-TA metrics.json."""

from __future__ import annotations

import time
from contextlib import contextmanager
from typing import Any, Dict, Iterator, Optional

import torch
import torch.nn as nn


def count_parameters(module: nn.Module) -> Dict[str, Any]:
    """Trainable / total parameter counts and backbone share."""
    n_trainable = 0
    n_total = 0
    for p in module.parameters():
        n = int(p.numel())
        n_total += n
        if p.requires_grad:
            n_trainable += n

    backbone_total = 0
    for name in ("visual", "textual", "external_clip"):
        sub = getattr(module, name, None)
        if sub is None:
            continue
        backbone_total += sum(int(p.numel()) for p in sub.parameters())

    temporal_total = 0
    temporal = getattr(module, "temporal", None)
    if temporal is not None:
        temporal_total = sum(int(p.numel()) for p in temporal.parameters())

    pct_of_backbone = None
    if backbone_total > 0:
        pct_of_backbone = 100.0 * float(n_trainable) / float(backbone_total)

    return {
        "n_trainable_params": n_trainable,
        "n_total_params": n_total,
        "n_backbone_params": backbone_total,
        "n_temporal_params": temporal_total,
        "trainable_pct_of_backbone": pct_of_backbone,
        "trainable_pct_of_total": (
            100.0 * float(n_trainable) / float(n_total) if n_total else None
        ),
    }


def count_module_params(module: Optional[nn.Module]) -> int:
    if module is None:
        return 0
    return sum(int(p.numel()) for p in module.parameters())


@contextmanager
def timed_section(store: Optional[Dict[str, Any]] = None, key: str = "seconds") -> Iterator[Dict[str, float]]:
    """Context manager that records wall-clock seconds into ``store[key]``."""
    t0 = time.perf_counter()
    box: Dict[str, float] = {}
    try:
        yield box
    finally:
        elapsed = time.perf_counter() - t0
        box["seconds"] = float(elapsed)
        if store is not None:
            store[key] = float(elapsed)


def merge_efficiency_into_metrics(
    metrics: Dict[str, Any],
    *,
    model: Optional[nn.Module] = None,
    extra: Optional[Dict[str, Any]] = None,
    train_seconds: Optional[float] = None,
    eval_seconds: Optional[float] = None,
) -> Dict[str, Any]:
    """Attach efficiency fields without inventing clinical MAE."""
    out = dict(metrics)
    if model is not None:
        out.update(count_parameters(model))
    if train_seconds is not None:
        out["train_seconds"] = float(train_seconds)
    if eval_seconds is not None:
        out["eval_seconds"] = float(eval_seconds)
    if extra:
        out.update(extra)
    return out
