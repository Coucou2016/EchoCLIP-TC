"""Cycle-aware temporal aggregation on frozen frame embeddings.

Maps a sequence of CLIP frame features (B, T, D) to a single video vector (B, D).
Mean pooling (no extra parameters) is handled by EchoCLIP.encode_video when
``temporal`` is None; this module provides Attention Pooling and a small
Temporal Transformer for EchoCLIP-TC.
"""

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


def divisible_heads(dim: int, n_heads: int) -> int:
    """Largest head count <= n_heads that divides dim."""
    h = min(max(1, n_heads), dim)
    while h > 1 and dim % h != 0:
        h -= 1
    return h


class AttentionPool(nn.Module):
    """Single learnable query attending over frame tokens: (B, T, D) -> (B, D)."""

    def __init__(self, dim: int, n_heads: int = 8):
        super().__init__()
        n_heads = divisible_heads(dim, n_heads)
        self.query = nn.Parameter(torch.zeros(1, 1, dim))
        nn.init.normal_(self.query, std=0.02)
        self.ln = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, n_heads, batch_first=True)

    def forward(
        self, x: torch.Tensor, mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        # mask: True = padding (key_padding_mask convention)
        q = self.query.expand(x.size(0), -1, -1)
        kv = self.ln(x)
        out, _ = self.attn(q, kv, kv, key_padding_mask=mask, need_weights=False)
        return out.squeeze(1)


class TemporalTransformer(nn.Module):
    """CLS-token Temporal Transformer over frame embeddings: (B, T, D) -> (B, D)."""

    def __init__(
        self,
        dim: int,
        n_layers: int = 2,
        n_heads: int = 8,
        max_frames: int = 64,
    ):
        super().__init__()
        from echoclip.model import LayerNorm, ResidualAttentionBlock

        n_heads = divisible_heads(dim, n_heads)
        self.max_frames = max_frames
        self.cls = nn.Parameter(torch.zeros(1, 1, dim))
        nn.init.normal_(self.cls, std=0.02)
        self.positional_embedding = nn.Parameter(torch.empty(1, max_frames + 1, dim))
        nn.init.normal_(self.positional_embedding, std=0.01)
        self.blocks = nn.ModuleList(
            [ResidualAttentionBlock(dim, n_heads) for _ in range(n_layers)]
        )
        self.ln = LayerNorm(dim)

    def forward(
        self, x: torch.Tensor, mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        del mask  # ResidualAttentionBlock uses attn_mask, not key padding
        bsz, seq_len, _ = x.shape
        if seq_len > self.max_frames:
            x = x[:, : self.max_frames]
            seq_len = self.max_frames
        cls = self.cls.expand(bsz, -1, -1)
        tokens = torch.cat([cls, x], dim=1)
        tokens = tokens + self.positional_embedding[:, : seq_len + 1]
        for block in self.blocks:
            tokens = block(tokens)
        return self.ln(tokens[:, 0])


def mean_pool(x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
    """Average frame embeddings; ``mask`` True = padding."""
    if mask is None:
        return x.mean(dim=1)
    valid = (~mask).unsqueeze(-1).to(dtype=x.dtype)
    denom = valid.sum(dim=1).clamp(min=1.0)
    return (x * valid).sum(dim=1) / denom


def build_temporal(
    kind: Optional[str],
    dim: int,
    n_layers: int = 2,
    n_heads: int = 8,
    max_frames: int = 64,
) -> Optional[nn.Module]:
    """Factory. ``none`` / ``mean`` / None → no module (caller mean-pools)."""
    if kind is None:
        return None
    key = str(kind).strip().lower()
    if key in ("", "none", "mean", "avg", "average"):
        return None
    if key in ("attn", "attn_pool", "attention", "attention_pool"):
        return AttentionPool(dim, n_heads=n_heads)
    if key in ("transformer", "temporal_transformer", "temporal"):
        return TemporalTransformer(
            dim, n_layers=n_layers, n_heads=n_heads, max_frames=max_frames
        )
    raise ValueError(
        f"Unknown temporal aggregator '{kind}'. "
        "Use none|mean|attn_pool|transformer."
    )


def pool_frame_features(
    frame_features: torch.Tensor,
    aggregator: Optional[nn.Module] = None,
    mask: Optional[torch.Tensor] = None,
    normalize: bool = True,
) -> torch.Tensor:
    """(B, T, D) -> (B, D)."""
    if aggregator is None:
        pooled = mean_pool(frame_features, mask=mask)
    else:
        pooled = aggregator(frame_features, mask=mask)
    if normalize:
        pooled = F.normalize(pooled, dim=-1)
    return pooled
