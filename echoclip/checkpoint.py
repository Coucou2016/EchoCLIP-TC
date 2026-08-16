"""Checkpoint load/save with consistent model_config metadata."""

from pathlib import Path
from typing import Any, Dict, Optional, Union

import torch

from echoclip.config import EchoCLIPConfig
from echoclip.model import EchoCLIP
from echoclip.utils import config_from_dict

PathLike = Union[str, Path]


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
    if extra:
        ckpt.update(extra)
    torch.save(ckpt, path)


def load_checkpoint(
    path: PathLike,
    device: str = "cpu",
    strict: bool = False,
) -> tuple[EchoCLIP, Dict[str, Any]]:
    """Load model and return (model, full checkpoint dict)."""
    path = Path(path)
    try:
        ckpt = torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        ckpt = torch.load(path, map_location=device)

    cfg_dict = ckpt.get("model_config") or ckpt.get("config", {})
    model_cfg = config_from_dict(cfg_dict)
    model_cfg.pretrained_vision = False
    state = ckpt["model_state"]
    has_external = any(str(k).startswith("external_clip.") for k in state)
    if has_external:
        from echoclip.model import EchoCLIP as _Echo

        model = _Echo.from_official_echo_clip(model_cfg)
        # Checkpoint embeds official towers; do not silently continue on
        # scratch_fallback / missing external_clip (would eval random towers).
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
