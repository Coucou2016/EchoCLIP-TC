"""Dual-encoder CLIP architecture for echocardiogram vision-language learning."""

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from echoclip.config import EchoCLIPConfig

try:
    import timm
except ImportError:
    timm = None


class LayerNorm(nn.LayerNorm):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return super().forward(x.float()).type(x.dtype)


class QuickGELU(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.sigmoid(1.702 * x)


class ResidualAttentionBlock(nn.Module):
    def __init__(self, d_model: int, n_head: int):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, n_head, batch_first=True)
        self.ln_1 = LayerNorm(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            QuickGELU(),
            nn.Linear(d_model * 4, d_model),
        )
        self.ln_2 = LayerNorm(d_model)

    def forward(self, x: torch.Tensor, attn_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        x_norm = self.ln_1(x)
        attn_out, _ = self.attn(
            x_norm, x_norm, x_norm, attn_mask=attn_mask, need_weights=False
        )
        x = x + attn_out
        x = x + self.mlp(self.ln_2(x))
        return x


class TextTransformer(nn.Module):
    """Decoder-only text encoder (CLIP architecture, 77-token context)."""

    def __init__(self, config: EchoCLIPConfig):
        super().__init__()
        self.context_length = config.context_length
        self.token_embedding = nn.Embedding(config.vocab_size, config.text_width)
        self.positional_embedding = nn.Parameter(
            torch.empty(config.context_length, config.text_width)
        )
        self.transformer = nn.ModuleList(
            [ResidualAttentionBlock(config.text_width, config.text_heads) for _ in range(config.text_layers)]
        )
        self.ln_final = LayerNorm(config.text_width)
        self.text_projection = nn.Parameter(torch.empty(config.text_width, config.embed_dim))
        self.register_buffer(
            "attn_mask",
            self._build_causal_mask(config.context_length),
            persistent=False,
        )
        self._init_parameters()

    @staticmethod
    def _build_causal_mask(context_length: int) -> torch.Tensor:
        mask = torch.empty(context_length, context_length)
        mask.fill_(float("-inf"))
        mask.triu_(1)
        return mask

    def _init_parameters(self) -> None:
        nn.init.normal_(self.token_embedding.weight, std=0.02)
        nn.init.normal_(self.positional_embedding, std=0.01)
        nn.init.normal_(self.text_projection, std=self.text_projection.shape[0] ** -0.5)

    def forward(self, text: torch.LongTensor) -> torch.Tensor:
        seq_len = text.shape[1]
        x = self.token_embedding(text) + self.positional_embedding[:seq_len]
        mask = self.attn_mask[:seq_len, :seq_len]
        for block in self.transformer:
            x = block(x, attn_mask=mask)
        # Pool at EOT: last non-padding token (CLIP pads with zeros).
        eot_idx = text.to(torch.int64).argmax(dim=-1)
        x = self.ln_final(x[torch.arange(x.shape[0], device=x.device), eot_idx])
        return x @ self.text_projection


class SimpleVisionEncoder(nn.Module):
    """Lightweight CNN when timm/torchvision is unavailable (smoke tests)."""

    def __init__(self, embed_dim: int):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
        )
        self.projection = nn.Linear(128, embed_dim)
        self.num_features = 128

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        x = self.features(images).flatten(1)
        return self.projection(x)


class VisionEncoder(nn.Module):
    """ConvNeXt (or timm backbone) with projection to shared embedding space."""

    def __init__(self, config: EchoCLIPConfig):
        super().__init__()
        if config.vision_backbone == "simple_cnn":
            self.model = SimpleVisionEncoder(config.embed_dim)
            self.projection = nn.Identity()
        elif timm is None:
            raise ImportError(
                "timm/torchvision could not be imported (often a broken _lzma DLL on Windows). "
                "Set vision_backbone: simple_cnn in config for local smoke tests."
            )
        else:
            self.model = timm.create_model(
                config.vision_backbone,
                pretrained=config.pretrained_vision,
                num_classes=0,
            )
            feat_dim = self.model.num_features
            self.projection = nn.Linear(feat_dim, config.embed_dim)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        feats = self.model(images)
        return self.projection(feats)


OFFICIAL_ECHOCLIP_HUB = "hf-hub:mkaichristensen/echo-clip"


class EchoCLIP(nn.Module):
    """
    EchoCLIP: contrastive vision-language model for echocardiograms.

    Image encoder: ConvNeXt-Base (default). Text encoder: CLIP-style transformer.
    Optional ``temporal`` module (EchoCLIP-TC) pools frame embeddings (B, T, D)
    into a video vector without changing the dual-encoder CLIP objective.
    """

    def __init__(self, config: Optional[EchoCLIPConfig] = None):
        super().__init__()
        self.config = config or EchoCLIPConfig()
        self.visual = VisionEncoder(self.config)
        self.textual = TextTransformer(self.config)
        self.logit_scale = nn.Parameter(torch.ones([]) * torch.log(torch.tensor(1 / 0.07)))
        self.temporal = None
        self.external_clip = None
        self.load_source = "scratch"
        kind = getattr(self.config, "temporal_type", "none")
        if kind and str(kind).lower() not in ("none", "mean", ""):
            self.attach_temporal(
                kind,
                n_layers=getattr(self.config, "temporal_layers", 2),
                n_heads=getattr(self.config, "temporal_heads", 8),
                max_frames=getattr(self.config, "temporal_max_frames", 64),
            )

    def attach_temporal(
        self,
        kind: str = "transformer",
        n_layers: int = 2,
        n_heads: int = 8,
        max_frames: int = 64,
    ) -> None:
        from echoclip.temporal import build_temporal

        dim = self.config.embed_dim
        module = build_temporal(
            kind, dim, n_layers=n_layers, n_heads=n_heads, max_frames=max_frames
        )
        self.temporal = module
        self.config.temporal_type = kind if module is not None else "none"
        self.config.temporal_layers = n_layers
        self.config.temporal_heads = n_heads
        self.config.temporal_max_frames = max_frames

    def _visual_feats(self, images: torch.Tensor) -> torch.Tensor:
        if self.external_clip is not None:
            return self.external_clip.encode_image(images)
        return self.visual(images)

    def encode_image(self, images: torch.Tensor) -> torch.Tensor:
        if images.dim() == 5:
            return self.encode_video(images)
        return F.normalize(self._visual_feats(images), dim=-1)

    def encode_text(self, text: torch.LongTensor) -> torch.Tensor:
        if self.external_clip is not None:
            return F.normalize(self.external_clip.encode_text(text), dim=-1)
        return F.normalize(self.textual(text), dim=-1)

    def encode_frame_features(self, frames: torch.Tensor) -> torch.Tensor:
        """Encode (B, T, C, H, W) to unnormalized (B, T, D) frame features."""
        if frames.dim() == 4:
            frames = frames.unsqueeze(0)
        bsz, n_frames, channels, height, width = frames.shape
        flat = frames.reshape(bsz * n_frames, channels, height, width)
        feats = self._visual_feats(flat)
        return feats.view(bsz, n_frames, -1)

    def encode_video(
        self,
        frames: torch.Tensor,
        frame_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Pool a clip to one L2-normalized embedding. frames: (B, T, C, H, W)."""
        from echoclip.temporal import pool_frame_features

        feats = self.encode_frame_features(frames)
        return pool_frame_features(
            feats, aggregator=self.temporal, mask=frame_mask, normalize=True
        )

    def forward(
        self, images: torch.Tensor, text: torch.LongTensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if images.dim() == 5:
            image_features = self.encode_video(images)
        else:
            image_features = self.encode_image(images)
        text_features = self.encode_text(text)
        return image_features, text_features, self.logit_scale

    @classmethod
    def from_open_clip(cls, config: Optional[EchoCLIPConfig] = None) -> "EchoCLIP":
        """Initialize vision/text weights from OpenCLIP LAION checkpoint (paper finetune start)."""
        cfg = config or EchoCLIPConfig()
        model = cls(cfg)
        try:
            import open_clip
        except ImportError as e:
            raise ImportError("Install open-clip-torch to use from_open_clip()") from e

        oc_model, _, _ = open_clip.create_model_and_transforms(
            cfg.open_clip_model,
            pretrained=cfg.open_clip_tag,
        )
        model.visual.model.load_state_dict(oc_model.visual.state_dict(), strict=False)
        te = model.textual
        te.token_embedding.load_state_dict(oc_model.token_embedding.state_dict())
        te.positional_embedding.data.copy_(oc_model.positional_embedding.data)
        te.ln_final.load_state_dict(oc_model.ln_final.state_dict())
        te.text_projection.data.copy_(oc_model.text_projection.data)
        te.transformer.load_state_dict(oc_model.transformer.state_dict(), strict=False)
        model.logit_scale.data.copy_(oc_model.logit_scale.data)
        model.load_source = f"open_clip:{cfg.open_clip_model}:{cfg.open_clip_tag}"
        return model

    @classmethod
    def from_official_echo_clip(
        cls,
        config: Optional[EchoCLIPConfig] = None,
        checkpoint_path: Optional[str] = None,
        hub: str = OFFICIAL_ECHOCLIP_HUB,
    ) -> "EchoCLIP":
        """
        Load official EchoCLIP when possible; never raise on missing hub weights.

        Order: local ``checkpoint_path`` (this repo's format) → open_clip hub →
        ``EchoCLIP(config)`` with ``simple_cnn`` if timm is unavailable.
        """
        import os
        import warnings
        from pathlib import Path

        cfg = config or EchoCLIPConfig()
        if checkpoint_path:
            path = Path(checkpoint_path)
            if path.exists():
                from echoclip.checkpoint import load_checkpoint

                model, _ = load_checkpoint(path, strict=False)
                model.load_source = f"local_checkpoint:{path}"
                return model
            warnings.warn(f"Official/local checkpoint not found: {path}")

        # Local towers must construct even if the hub model will take over encoding.
        if timm is None and cfg.vision_backbone != "simple_cnn":
            warnings.warn(
                "timm unavailable; using vision_backbone=simple_cnn for local towers"
            )
            cfg.vision_backbone = "simple_cnn"
            cfg.pretrained_vision = False

        skip_hub = os.environ.get("ECHOCLIP_SKIP_HUB", "").strip().lower() in (
            "1",
            "true",
            "yes",
        )
        if not skip_hub:
            try:
                import open_clip

                oc_model, _, _ = open_clip.create_model_and_transforms(hub)
                # Align local embed_dim with the hub tower before attaching temporal.
                probe = getattr(oc_model, "visual", None)
                hub_dim = None
                for attr in ("output_dim", "embed_dim", "proj"):
                    if probe is None:
                        break
                    val = getattr(probe, attr, None)
                    if isinstance(val, int) and val > 0:
                        hub_dim = val
                        break
                    if attr == "proj" and val is not None and hasattr(val, "shape"):
                        # Linear proj: out features; Parameter: last dim
                        shape = tuple(val.shape)
                        hub_dim = int(shape[-1]) if shape else None
                if hub_dim is not None and hub_dim != cfg.embed_dim:
                    warnings.warn(
                        f"Official EchoCLIP embed_dim={hub_dim} differs from config "
                        f"{cfg.embed_dim}; syncing config + temporal aggregator."
                    )
                    cfg.embed_dim = hub_dim
                model = cls(cfg)
                model.external_clip = _OpenCLIPHolder(oc_model)
                if hasattr(oc_model, "logit_scale"):
                    model.logit_scale.data.copy_(oc_model.logit_scale.data)
                # Re-attach temporal if dim changed after initial construction.
                kind = getattr(cfg, "temporal_type", "none")
                if kind and str(kind).lower() not in ("none", "mean", ""):
                    if model.temporal is None or (
                        hub_dim is not None
                        and getattr(model.temporal, "cls", None) is not None
                        and model.temporal.cls.shape[-1] != hub_dim
                    ):
                        model.attach_temporal(
                            kind,
                            n_layers=getattr(cfg, "temporal_layers", 2),
                            n_heads=getattr(cfg, "temporal_heads", 8),
                            max_frames=getattr(cfg, "temporal_max_frames", 64),
                        )
                model.load_source = hub
                return model
            except Exception as exc:
                warnings.warn(
                    f"Could not load official EchoCLIP ({hub}): {exc}. "
                    "Falling back to a local randomly initialized model."
                )

        model = cls(cfg)
        model.load_source = "scratch_fallback"
        return model

    def encode_video_frames(
        self, frame_batch: torch.Tensor, pool: str = "auto"
    ) -> torch.Tensor:
        """Encode (N_frames, C, H, W) and pool to one embedding.

        pool='auto' uses the temporal aggregator when attached, else mean.
        """
        if pool == "auto":
            pool = "temporal" if self.temporal is not None else "mean"
        if pool in ("temporal", "attn", "transformer") and frame_batch.dim() == 4:
            return self.encode_video(frame_batch.unsqueeze(0)).squeeze(0)
        feats = self.encode_image(frame_batch)
        if pool == "mean":
            return F.normalize(feats.mean(dim=0), dim=-1)
        if pool == "max":
            return F.normalize(feats.max(dim=0).values, dim=-1)
        raise ValueError(f"Unknown pool: {pool}")


class _OpenCLIPHolder(nn.Module):
    """Wrap an open_clip model so encode_image / encode_text stay in one module."""

    def __init__(self, oc_model: nn.Module):
        super().__init__()
        self.model = oc_model

    def encode_image(self, images: torch.Tensor) -> torch.Tensor:
        return self.model.encode_image(images)

    def encode_text(self, text: torch.LongTensor) -> torch.Tensor:
        return self.model.encode_text(text)
