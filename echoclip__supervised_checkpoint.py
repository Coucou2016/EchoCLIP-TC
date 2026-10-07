"""Unified supervised EF-regression checkpoint schema (R2/R3/R4).

Why this module exists
----------------------
R2 (linear/ridge), R3 (MLP) and R4 (temporal + direct L1/Huber) previously saved
``head_state_dict`` in shapes that ``load_checkpoint`` (which needs
``model_state``) could not read, and R4's EF head lived only in
``train_cfg["head_state"]`` and was never restored. This module defines **one**
on-disk schema so a *fresh Python process* can train → ``best.pt`` → load →
TEST inference → finite EF values → ``metrics.json`` without any in-process head
object.

Schema (``SCHEMA_VERSION`` 2)
-----------------------------
Required keys::

    schema_version      int    # this module's schema revision
    task                str    # "ef_regression"
    head_kind           str    # linear|mlp|temporal_l1 (aliases ridge/s0/…)
    head_state_dict     dict   # head (or TemporalEFRegressor) state_dict
    backbone_config     dict   # EchoCLIPConfig fields + load_source
    backbone_load_source str   # "hf-hub:…", "official_checkpoint", "scratch", …
    embed_dim           int
    train_seed          int
    split_seed          int|None   # TRAIN/VAL split seed (P0-4)
    sampler_seed        int|None   # per-epoch frame-sampler seed base (P0-4)
    train_cfg           dict
    epoch               int

Optional keys::

    temporal_state_dict dict   # R4 aggregator state (also mirrored from head)
    model_state         dict   # full backbone state when the caller saved it
    meta                dict   # free-form provenance (determinism mode, …)

Legacy aliases kept for backwards compatibility::

    format_version  == schema_version
    model_config    == backbone_config
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn

from echoclip.config import EchoCLIPConfig
from echoclip.model import EchoCLIP
from echoclip.utils import config_from_dict

PathLike = Union[str, Path]

# Bump when the payload shape changes in a way old loaders cannot read.
SCHEMA_VERSION = 2

# Historical alias (v1 name) — kept so older code/checks keep working.
SUPERVISED_FORMAT_VERSION = SCHEMA_VERSION

TASK = "ef_regression"
# Historical alias used by echoclip.checkpoint / tests.
SUPERVISED_TASK = TASK


def _read_ckpt(path: PathLike, device: str) -> Dict[str, Any]:
    try:
        return torch.load(Path(path), map_location=device, weights_only=False)
    except TypeError:  # torch < 2.0 has no weights_only kwarg
        return torch.load(Path(path), map_location=device)


def is_supervised_ef_checkpoint(ckpt: Dict[str, Any]) -> bool:
    """True if the payload is (or looks like) a supervised EF checkpoint."""
    if not isinstance(ckpt, dict):
        return False
    if str(ckpt.get("task") or "").strip().lower() == TASK:
        return True
    if ckpt.get("head_kind") and (
        ckpt.get("head_state_dict") is not None or ckpt.get("head_state") is not None
    ):
        return True
    return False


def backbone_config_dict(model: EchoCLIP) -> Dict[str, Any]:
    """Serializable backbone config + load provenance."""
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


def save_ef_regression_checkpoint(
    path: PathLike,
    *,
    head_kind: str,
    head: nn.Module,
    backbone_config: Dict[str, Any],
    train_seed: int,
    epoch: int = 0,
    embed_dim: Optional[int] = None,
    temporal_state_dict: Optional[Dict[str, Any]] = None,
    backbone_state: Optional[Dict[str, Any]] = None,
    backbone_load_source: Optional[str] = None,
    train_cfg: Optional[Dict[str, Any]] = None,
    meta: Optional[Dict[str, Any]] = None,
    split_seed: Optional[int] = None,
    sampler_seed: Optional[int] = None,
) -> Dict[str, Any]:
    """Persist the unified R2–R4 supervised EF schema."""
    kind = str(head_kind).strip().lower()
    cfg = dict(backbone_config)
    if backbone_load_source is not None:
        cfg.setdefault("load_source", backbone_load_source)
    load_source = backbone_load_source or cfg.get("load_source") or "scratch"

    payload: Dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "format_version": SCHEMA_VERSION,  # legacy alias
        "task": TASK,
        "head_kind": kind,
        "head_state_dict": head.state_dict(),
        "backbone_config": cfg,
        "model_config": cfg,  # legacy alias
        "backbone_load_source": load_source,
        "embed_dim": int(embed_dim or cfg.get("embed_dim") or 0),
        "train_seed": int(train_seed),
        "split_seed": None if split_seed is None else int(split_seed),
        "sampler_seed": None if sampler_seed is None else int(sampler_seed),
        "epoch": int(epoch),
        "train_cfg": train_cfg or {},
        "meta": meta or {},
    }
    if temporal_state_dict is not None:
        payload["temporal_state_dict"] = temporal_state_dict
    if backbone_state is not None:
        payload["model_state"] = backbone_state
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)
    return payload


def _rebuild_head(kind: str, dim: int, ckpt: Dict[str, Any]) -> nn.Module:
    from echoclip.supervised import LinearEFHead, MLPEFHead, TemporalEFRegressor
    from echoclip.temporal import build_temporal

    key = str(kind).strip().lower()
    cfg = ckpt.get("backbone_config") or ckpt.get("model_config") or {}
    train_cfg = ckpt.get("train_cfg") or {}
    hidden = int(
        train_cfg.get("ef_head_hidden")
        or cfg.get("ef_head_hidden")
        or (ckpt.get("meta") or {}).get("hidden")
        or 256
    )
    if key in ("linear", "ridge", "s0"):
        return LinearEFHead(dim)
    if key in ("mlp", "s1"):
        return MLPEFHead(dim, hidden=hidden)
    if key in ("temporal_l1", "temporal_huber", "s2"):
        n_layers = int(cfg.get("temporal_layers", train_cfg.get("temporal_layers", 2)))
        n_heads = int(cfg.get("temporal_heads", train_cfg.get("temporal_heads", 8)))
        max_frames = int(
            cfg.get("temporal_max_frames", train_cfg.get("temporal_max_frames", 64))
        )
        agg = build_temporal(
            "transformer",
            dim,
            n_layers=n_layers,
            n_heads=n_heads,
            max_frames=max_frames,
        )
        assert agg is not None
        reg = TemporalEFRegressor(agg, dim, head="linear", hidden=hidden)
        tsd = ckpt.get("temporal_state_dict")
        if tsd is not None:
            reg.aggregator.load_state_dict(tsd, strict=False)
        return reg
    raise ValueError(f"Unknown supervised head_kind {kind!r}")


def load_ef_regression_checkpoint(
    path: PathLike,
    device: str = "cpu",
    *,
    backbone: Optional[EchoCLIP] = None,
    allow_scratch_fallback: bool = True,
) -> Tuple[EchoCLIP, nn.Module, Dict[str, Any]]:
    """Load ``(backbone, head_module, ckpt)`` from a unified supervised ckpt.

    Works in a fresh process: no in-process head object is required. For
    ``temporal_l1`` the returned ``head_module`` is a ``TemporalEFRegressor``
    (aggregator + linear); for linear/MLP it is the head alone (the caller
    mean-pools the backbone features).
    """
    path = Path(path)
    ckpt = _read_ckpt(path, device)

    # Legacy R4 path: save_checkpoint + train_cfg["head_state"] / supervised_head
    if not is_supervised_ef_checkpoint(ckpt):
        train_cfg = ckpt.get("config") or ckpt.get("train_cfg") or {}
        if train_cfg.get("supervised_head") or train_cfg.get("head_state"):
            ckpt = dict(ckpt)
            ckpt["task"] = TASK
            ckpt["head_kind"] = str(
                train_cfg.get("supervised_head") or ckpt.get("head_kind") or "temporal_l1"
            )
            if "head_state_dict" not in ckpt and train_cfg.get("head_state"):
                ckpt["head_state_dict"] = train_cfg["head_state"]
            if "train_seed" not in ckpt and "seed" in train_cfg:
                ckpt["train_seed"] = int(train_cfg["seed"])
        else:
            raise KeyError(
                f"Checkpoint {path} is not a supervised EF checkpoint "
                f"(missing task={TASK!r} / head_state_dict)."
            )

    kind = str(ckpt.get("head_kind") or "linear").strip().lower()
    cfg_dict = dict(ckpt.get("backbone_config") or ckpt.get("model_config") or {})
    dim = int(ckpt.get("embed_dim") or cfg_dict.get("embed_dim") or 512)

    if backbone is None:
        model_cfg = config_from_dict(cfg_dict)
        model_cfg.pretrained_vision = False
        if ckpt.get("model_state") and any(
            str(k).startswith("external_clip.") for k in ckpt["model_state"]
        ):
            backbone = EchoCLIP.from_official_echo_clip(
                model_cfg, allow_scratch_fallback=allow_scratch_fallback
            )
            backbone.load_state_dict(ckpt["model_state"], strict=False)
        else:
            backbone = EchoCLIP(model_cfg)
            if ckpt.get("model_state"):
                backbone.load_state_dict(ckpt["model_state"], strict=False)
        backbone.to(device)
    else:
        backbone.to(device)

    for p in backbone.parameters():
        p.requires_grad = False
    backbone.eval()

    head = _rebuild_head(kind, dim, ckpt)
    hsd = ckpt.get("head_state_dict") or ckpt.get("head_state")
    train_cfg = ckpt.get("train_cfg") or ckpt.get("config") or {}
    if hsd is None and train_cfg.get("head_state"):
        hsd = train_cfg["head_state"]

    if kind in ("temporal_l1", "temporal_huber", "s2"):
        if hsd is not None:
            keys = set(hsd.keys())
            if any(k.startswith("head.") or k.startswith("aggregator.") for k in keys):
                head.load_state_dict(hsd, strict=False)
            elif any(k.startswith("fc.") for k in keys):
                head.head.load_state_dict(hsd, strict=False)
            else:
                head.load_state_dict(hsd, strict=False)
        tsd = ckpt.get("temporal_state_dict")
        if tsd is not None:
            head.aggregator.load_state_dict(tsd, strict=False)
        elif ckpt.get("model_state"):
            t_keys = {
                k[len("temporal.") :]: v
                for k, v in ckpt["model_state"].items()
                if str(k).startswith("temporal.")
            }
            if t_keys:
                head.aggregator.load_state_dict(t_keys, strict=False)
    else:
        if hsd is None:
            raise KeyError(f"Supervised checkpoint {path} missing head_state_dict")
        head.load_state_dict(hsd, strict=False)

    head.to(device)
    head.eval()
    return backbone, head, ckpt


@torch.no_grad()
def predict_direct_ef(
    backbone: EchoCLIP,
    head: nn.Module,
    frames: torch.Tensor,
    *,
    head_kind: str,
    mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Direct EF regression (no zero-shot prompt pack).

    Flow: frame features → optional temporal pooling → EF head.

    ``frames``: ``(B,T,C,H,W)`` or ``(B,C,H,W)``.
    Returns ``(B,)`` EF predictions in percent.
    """
    from echoclip.supervised import TemporalEFRegressor, extract_mean_pooled_features

    kind = str(head_kind).strip().lower()
    backbone.eval()
    head.eval()

    if frames.dim() == 4:
        frames = frames.unsqueeze(1)

    if kind in ("temporal_l1", "temporal_huber", "s2"):
        if not isinstance(head, TemporalEFRegressor):
            raise TypeError("temporal_l1 requires TemporalEFRegressor head")
        feats = backbone.encode_frame_features(frames)
        return head(feats, mask=mask)

    # linear / mlp: mean-pool then head
    if frames.dim() == 5:
        z = extract_mean_pooled_features(backbone, frames, mask=mask)
    else:
        z = backbone.encode_image(frames)
    return head(z)


@torch.no_grad()
def predict_direct_ef_loader(
    backbone: EchoCLIP,
    head: nn.Module,
    loader,
    *,
    head_kind: str,
    device: str = "cpu",
) -> List[float]:
    """Run :func:`predict_direct_ef` over a DataLoader; return flat preds."""
    preds: List[float] = []
    for batch in loader:
        images = batch["image"].to(device)
        out = predict_direct_ef(backbone, head, images, head_kind=head_kind)
        preds.extend(float(x) for x in out.detach().cpu().view(-1).tolist())
    return preds
