"""Checkpoint load/save with consistent model_config metadata.

The unified supervised EF-regression schema (R2–R4, ``task=ef_regression``)
lives in :mod:`echoclip.supervised_checkpoint`; this module re-exports it so
existing call sites keep working while there is exactly one on-disk schema.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional, Union

import torch
import torch.nn as nn

from echoclip.config import EchoCLIPConfig
from echoclip.model import EchoCLIP
from echoclip.utils import config_from_dict

# Re-export the canonical supervised schema (single source of truth).
from echoclip.supervised_checkpoint import (  # noqa: F401
    SCHEMA_VERSION,
    SUPERVISED_FORMAT_VERSION,
    SUPERVISED_TASK,
    TASK as SUPERVISED_TASK_NAME,
    backbone_config_dict,
    is_supervised_ef_checkpoint,
    load_ef_regression_checkpoint,
    predict_direct_ef,
    predict_direct_ef_loader,
    save_ef_regression_checkpoint,
)

PathLike = Union[str, Path]


def model_config_dict(model: EchoCLIP) -> Dict[str, Any]:
    """Serializable config for the generic (contrastive / TC) checkpoint path."""
    c = model.config
    return {
        "embed_dim": c.embed_dim,
        "image_size": c.image_size,
        "context_length": c.context_length,
        "vision_backbone": c.vision_backbone,
        "text_layers": c.text_layers,
        "text_heads": c.text_heads,
        "text_width": c.text_width,
        "vocab_size": c.vocab_size,
        "pretrained_vision": c.pretrained_vision,
        "temporal_type": getattr(c, "temporal_type", "none"),
        "temporal_layers": getattr(c, "temporal_layers", 2),
        "temporal_heads": getattr(c, "temporal_heads", 8),
        "temporal_max_frames": getattr(c, "temporal_max_frames", 64),
        "load_source": getattr(model, "load_source", "scratch"),
    }


def save_checkpoint(
    path: PathLike,
    model: EchoCLIP,
    epoch: int,
    train_cfg: Optional[Dict[str, Any]] = None,
    extra: Optional[Dict[str, Any]] = None,
) -> None:
    ckpt = {
        "epoch": epoch,
        "model_state": model.state_dict(),
        "model_config": model_config_dict(model),
        "config": train_cfg or {},
    }
    if train_cfg and "seed" in train_cfg:
        ckpt.setdefault("train_seed", int(train_cfg["seed"]))
    # P0-4: persist the seed provenance explicitly when provided.
    if train_cfg:
        for key in ("split_seed", "sampler_seed", "determinism_mode"):
            if key in train_cfg:
                ckpt.setdefault(key, train_cfg[key])
    if extra:
        ckpt.update(extra)
    torch.save(ckpt, path)


def load_checkpoint(
    path: PathLike,
    device: str = "cpu",
    strict: bool = False,
) -> tuple[EchoCLIP, Dict[str, Any]]:
    """Load contrastive / TC model and return (model, full checkpoint dict).

    For supervised EF heads (R2–R4), use ``load_supervised_ef_checkpoint`` (or
    the canonical ``load_ef_regression_checkpoint``) instead.
    """
    path = Path(path)
    try:
        ckpt = torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        ckpt = torch.load(path, map_location=device)

    if is_supervised_ef_checkpoint(ckpt):
        raise KeyError(
            f"Checkpoint {path} is a supervised EF regression checkpoint "
            f"(task={ckpt.get('task')!r}). Use load_ef_regression_checkpoint() "
            "and --prediction-mode direct_ef, not load_checkpoint()."
        )

    cfg_dict = (
        ckpt.get("model_config")
        or ckpt.get("backbone_config")
        or ckpt.get("config", {})
    )
    model_cfg = config_from_dict(cfg_dict)
    model_cfg.pretrained_vision = False
    if "model_state" not in ckpt:
        raise KeyError(
            f"Checkpoint {path} missing 'model_state'. "
            "If this is an R2/R3 linear/MLP head, use load_ef_regression_checkpoint()."
        )
    state = ckpt["model_state"]
    has_external = any(str(k).startswith("external_clip.") for k in state)
    if has_external:
        from echoclip.model import EchoCLIP as _Echo

        model = _Echo.from_official_echo_clip(model_cfg)
        if getattr(model, "external_clip", None) is None or getattr(
            model, "load_source", ""
        ) == "scratch_fallback":
            raise RuntimeError(
                f"Checkpoint {path} contains external_clip weights but official "
                "EchoCLIP towers could not be rematerialized (hub skipped/unavailable "
                "or load failed). Install/open hub access, pass a local official "
                "checkpoint, or retrain with --no-official so the save has no "
                "external_clip.* keys."
            )
    else:
        model = EchoCLIP(model_cfg)
        if getattr(model, "temporal", None) is None and any(
            str(k).startswith("temporal.") for k in state
        ):
            model.attach_temporal(
                getattr(model_cfg, "temporal_type", "transformer") or "transformer",
                n_layers=getattr(model_cfg, "temporal_layers", 2),
                n_heads=getattr(model_cfg, "temporal_heads", 8),
                max_frames=getattr(model_cfg, "temporal_max_frames", 64),
            )
    model.load_state_dict(state, strict=strict)
    model.to(device)
    return model, ckpt


# ---------------------------------------------------------------------------
# Backwards-compatible aliases for the v1 supervised API names.
# ---------------------------------------------------------------------------

def save_supervised_ef_checkpoint(
    path: PathLike,
    *,
    head_kind: str,
    head: nn.Module,
    model_config: Optional[Dict[str, Any]] = None,
    backbone_config: Optional[Dict[str, Any]] = None,
    train_seed: int,
    epoch: int = 0,
    embed_dim: Optional[int] = None,
    temporal_state_dict: Optional[Dict[str, Any]] = None,
    backbone_state: Optional[Dict[str, Any]] = None,
    train_cfg: Optional[Dict[str, Any]] = None,
    meta: Optional[Dict[str, Any]] = None,
    backbone_load_source: Optional[str] = None,
    split_seed: Optional[int] = None,
    sampler_seed: Optional[int] = None,
) -> Dict[str, Any]:
    """Deprecated alias → :func:`save_ef_regression_checkpoint`."""
    return save_ef_regression_checkpoint(
        path,
        head_kind=head_kind,
        head=head,
        backbone_config=backbone_config or model_config or {},
        train_seed=train_seed,
        epoch=epoch,
        embed_dim=embed_dim,
        temporal_state_dict=temporal_state_dict,
        backbone_state=backbone_state,
        train_cfg=train_cfg,
        meta=meta,
        backbone_load_source=backbone_load_source,
        split_seed=split_seed,
        sampler_seed=sampler_seed,
    )


def load_supervised_ef_checkpoint(
    path: PathLike,
    device: str = "cpu",
    *,
    backbone: Optional[EchoCLIP] = None,
    allow_scratch_fallback: bool = True,
):
    """Deprecated alias → :func:`load_ef_regression_checkpoint`."""
    return load_ef_regression_checkpoint(
        path,
        device,
        backbone=backbone,
        allow_scratch_fallback=allow_scratch_fallback,
    )


__all__ = [
    "SCHEMA_VERSION",
    "SUPERVISED_FORMAT_VERSION",
    "SUPERVISED_TASK",
    "backbone_config_dict",
    "is_supervised_ef_checkpoint",
    "load_checkpoint",
    "load_ef_regression_checkpoint",
    "load_supervised_ef_checkpoint",
    "model_config_dict",
    "predict_direct_ef",
    "predict_direct_ef_loader",
    "save_checkpoint",
    "save_ef_regression_checkpoint",
    "save_supervised_ef_checkpoint",
]
