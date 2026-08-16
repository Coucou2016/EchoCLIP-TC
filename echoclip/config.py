from dataclasses import dataclass, field
from typing import Optional


@dataclass
class EchoCLIPConfig:
    """Model and training defaults aligned with Christensen et al., Nature Medicine 2024."""

    embed_dim: int = 512
    image_size: int = 224
    context_length: int = 77
    vision_backbone: str = "convnext_base"  # paper: ConvNeXt-Base
    text_layers: int = 12
    text_heads: int = 8
    text_width: int = 512
    vocab_size: int = 49408  # GPT-2 BPE (CLIP)
    pretrained_vision: bool = True
    open_clip_tag: Optional[str] = "laion2b_s34b_b88k"  # LAION CLIP init (paper)
    open_clip_model: str = "convnext_base_w_320"
    # EchoCLIP-TC temporal aggregator (none|mean|attn_pool|transformer)
    temporal_type: str = "none"
    temporal_layers: int = 2
    temporal_heads: int = 8
    temporal_max_frames: int = 64
