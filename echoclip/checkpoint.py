"""Checkpoint load/save with consistent model_config metadata.

Also defines the unified supervised EF-regression checkpoint schema used by
R2–R4 (``task=ef_regression``) so eval can reload heads without zero-shot prompts.
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

# Unified supervised EF checkpoint schema (R2/R3/R4).
SUPERVISED_FORMAT_VERSION = 1
SUPERVISED_TASK = "ef_regression"


def model_config_dict(model: EchoCLIP) -> Dict[str, Any]:
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
    if extra:
        ckpt.update(extra)
    torch.save(ckpt, path)


def load_checkpoint(
    path: PathLike,
    device: str = "cpu",
    strict: bool = False,
) -> tuple[EchoCLIP, Dict[str, Any]]:
    """Load contrastive / TC model and return (model, full checkpoint dict).

    For supervised EF heads (R2–R4), use ``load_supervised_ef_checkpoint`` instead.
    """
    path = Path(path)
    try:
        ckpt = torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        ckpt = torch.load(path, map_location=device)

    if is_supervised_ef_checkpoint(ckpt):
        raise KeyError(
            f"Checkpoint {path} is a supervised EF regression checkpoint "
            f"(task={ckpt.get('task')!r}). Use load_supervised_ef_checkpoint() "
            "and --prediction-mode direct_regression, not load_checkpoint()."
        )

    cfg_dict = ckpt.get("model_config") or ckpt.get("config", {})
    model_cfg = config_from_dict(cfg_dict)
    model_cfg.pretrained_vision = False
    if "model_state" not in ckpt:
        raise KeyError(
            f"Checkpoint {path} missing 'model_state'. "
            "If this is an R2/R3 linear/MLP head, use load_supervised_ef_checkpoint()."
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


def is_supervised_ef_checkpoint(ckpt: Dict[str, Any]) -> bool:
    """True if payload is (or looks like) a supervised EF regression checkpoint."""
    if not isinstance(ckpt, dict):
        return False
    task = str(ckpt.get("task") or "").strip().lower()
    if task == SUPERVISED_TASK:
        return True
    if ckpt.get("head_kind") and (
        ckpt.get("head_state_dict") is not None or ckpt.get("head_state") is not None
    ):
        return True
    return False


def save_supervised_ef_checkpoint(
    path: PathLike,
    *,
    head_kind: str,
    head: nn.Module,
    model_config: Dict[str, Any],
    train_seed: int,
    epoch: int = 0,
    embed_dim: Optional[int] = None,
    temporal_state_dict: Optional[Dict[str, Any]] = None,
    backbone_state: Optional[Dict[str, Any]] = None,
    train_cfg: Optional[Dict[str, Any]] = None,
    meta: Optional[Dict[str, Any]] = None,
    backbone_load_source: Optional[str] = None,
) -> Dict[str, Any]:
    """Persist the unified R2–R4 supervised EF schema."""
    kind = str(head_kind).strip().lower()
    payload: Dict[str, Any] = {
        "format_version": SUPERVISED_FORMAT_VERSION,
        "task": SUPERVISED_TASK,
        "head_kind": kind,
        "head_state_dict": head.state_dict(),
        "model_config": dict(model_config),
        "train_seed": int(train_seed),
        "epoch": int(epoch),
        "embed_dim": int(embed_dim or model_config.get("embed_dim") or 0),
        "train_cfg": train_cfg or {},
        "meta": meta or {},
    }
    if temporal_state_dict is not None:
        payload["temporal_state_dict"] = temporal_state_dict
    if backbone_state is not None:
        # Optional: only for cases that also want load_checkpoint-style reload.
        payload["model_state"] = backbone_state
    if backbone_load_source is not None:
        payload["backbone_load_source"] = backbone_load_source
        payload["model_config"].setdefault("load_source", backbone_load_source)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)
    return payload


def _rebuild_head(kind: str, dim: int, ckpt: Dict[str, Any]) -> nn.Module:
    from echoclip.supervised import LinearEFHead, MLPEFHead, TemporalEFRegressor
    from echoclip.temporal import build_temporal

    key = str(kind).strip().lower()
    cfg = ckpt.get("model_config") or {}
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
            cfg.get(
                "temporal_max_frames",
                train_cfg.get("temporal_max_frames", 64),
            )
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
        # Prefer dedicated temporal_state_dict; fall back to aggregator keys in head.
        tsd = ckpt.get("temporal_state_dict")
        if tsd is not None:
            reg.aggregator.load_state_dict(tsd, strict=False)
        # Head weights may live under head_state_dict (full TemporalEFRegressor)
        # or legacy train_cfg["head_state"] (LinearEFHead only).
        return reg
    raise ValueError(f"Unknown supervised head_kind {kind!r}")


def load_supervised_ef_checkpoint(
    path: PathLike,
    device: str = "cpu",
    *,
    backbone: Optional[EchoCLIP] = None,
    allow_scratch_fallback: bool = True,
) -> Tuple[EchoCLIP, nn.Module, Dict[str, Any]]:
    """Load backbone + EF regression head from a unified supervised checkpoint.

    Returns ``(backbone, head_module, ckpt)``. For ``temporal_l1``, ``head_module``
    is a ``TemporalEFRegressor`` (aggregator + linear). For linear/MLP it is the
    head alone (caller mean-pools backbone features).
    """
    path = Path(path)
    try:
        ckpt = torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        ckpt = torch.load(path, map_location=device)

    # Legacy R4: save_checkpoint + train_cfg["head_state"] / supervised_head
    if not is_supervised_ef_checkpoint(ckpt):
        train_cfg = ckpt.get("config") or ckpt.get("train_cfg") or {}
        if train_cfg.get("supervised_head") or train_cfg.get("head_state"):
            ckpt = dict(ckpt)
            ckpt["task"] = SUPERVISED_TASK
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
                f"(missing task={SUPERVISED_TASK!r} / head_state_dict)."
            )

    kind = str(ckpt.get("head_kind") or "linear").strip().lower()
    cfg_dict = dict(ckpt.get("model_config") or {})
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
    # Load weights
    hsd = ckpt.get("head_state_dict") or ckpt.get("head_state")
    train_cfg = ckpt.get("train_cfg") or ckpt.get("config") or {}
    if hsd is None and train_cfg.get("head_state"):
        hsd = train_cfg["head_state"]

    if kind in ("temporal_l1", "temporal_huber", "s2"):
        # May be full TemporalEFRegressor state or only LinearEFHead state.
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
            # Pull temporal.* from full backbone state if present
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
    """Run ``predict_direct_ef`` over a DataLoader; return flat list of preds."""
    preds: List[float] = []
    for batch in loader:
        images = batch["image"].to(device)
        out = predict_direct_ef(backbone, head, images, head_kind=head_kind)
        preds.extend(float(x) for x in out.detach().cpu().view(-1).tolist())
    return preds
