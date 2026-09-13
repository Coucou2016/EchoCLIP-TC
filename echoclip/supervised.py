"""Supervised EF heads on frozen EchoCLIP features (S0/S1/S2 protocol baselines).

These are label-supervised baselines for the EchoCLIP-TA comparison matrix.
They work with toy tensors when EchoNet / hub weights are absent.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class LinearEFHead(nn.Module):
    """Single linear map from video embedding → EF (%)."""

    def __init__(self, dim: int, bias: bool = True):
        super().__init__()
        self.fc = nn.Linear(dim, 1, bias=bias)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.fc(z).squeeze(-1)


class MLPEFHead(nn.Module):
    """Two-layer MLP EF head."""

    def __init__(self, dim: int, hidden: int = 256, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden, 1),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z).squeeze(-1)


class TemporalEFRegressor(nn.Module):
    """Temporal aggregator + linear EF head (direct L1/Huber, not contrastive)."""

    def __init__(
        self,
        aggregator: nn.Module,
        dim: int,
        head: str = "linear",
        hidden: int = 256,
    ):
        super().__init__()
        self.aggregator = aggregator
        if head == "mlp":
            self.head = MLPEFHead(dim, hidden=hidden)
        else:
            self.head = LinearEFHead(dim)

    def forward(
        self, frame_features: torch.Tensor, mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        from echoclip.temporal import pool_frame_features

        z = pool_frame_features(
            frame_features, aggregator=self.aggregator, mask=mask, normalize=True
        )
        return self.head(z)


def build_supervised_head(
    kind: str,
    dim: int,
    *,
    aggregator: Optional[nn.Module] = None,
    hidden: int = 256,
) -> nn.Module:
    key = str(kind).strip().lower()
    if key in ("linear", "ridge", "s0"):
        return LinearEFHead(dim)
    if key in ("mlp", "s1"):
        return MLPEFHead(dim, hidden=hidden)
    if key in ("temporal_l1", "temporal_huber", "s2"):
        if aggregator is None:
            raise ValueError("temporal_l1 head requires an aggregator module")
        return TemporalEFRegressor(aggregator, dim, head="linear", hidden=hidden)
    raise ValueError(f"Unknown supervised head {kind!r}. Use linear|mlp|temporal_l1.")


def ef_regression_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    *,
    kind: str = "huber",
    delta: float = 1.0,
) -> torch.Tensor:
    """L1 or Huber loss; ignores NaN targets."""
    mask = torch.isfinite(target)
    if not bool(mask.any()):
        return pred.sum() * 0.0
    p = pred[mask]
    y = target[mask]
    key = str(kind).strip().lower()
    if key in ("l1", "mae", "absolute"):
        return F.l1_loss(p, y)
    return F.huber_loss(p, y, delta=float(delta))


@dataclass
class RidgeEFSolution:
    """Closed-form ridge regression on CPU numpy/torch features."""

    weight: torch.Tensor  # (D,)
    bias: float
    alpha: float

    def predict(self, z: torch.Tensor) -> torch.Tensor:
        return z @ self.weight.to(dtype=z.dtype, device=z.device) + float(self.bias)


def fit_ridge_ef(
    features: torch.Tensor,
    targets: torch.Tensor,
    *,
    alpha: float = 1.0,
) -> RidgeEFSolution:
    """Fit ridge EF head: y ≈ W z + b. Features (N, D), targets (N,)."""
    z = features.detach().float()
    y = targets.detach().float()
    mask = torch.isfinite(y)
    z, y = z[mask], y[mask]
    if z.shape[0] < 2:
        raise ValueError("Need at least 2 finite EF labels to fit ridge head")
    # Center
    z_mean = z.mean(dim=0, keepdim=True)
    y_mean = y.mean()
    zc = z - z_mean
    yc = y - y_mean
    d = zc.shape[1]
    # (Z^T Z + α I) w = Z^T y
    xtx = zc.T @ zc + float(alpha) * torch.eye(d, dtype=zc.dtype, device=zc.device)
    xty = zc.T @ yc
    w = torch.linalg.solve(xtx, xty)
    bias = float(y_mean - (z_mean @ w).squeeze())
    return RidgeEFSolution(weight=w.cpu(), bias=bias, alpha=float(alpha))


@torch.no_grad()
def extract_mean_pooled_features(
    model: nn.Module,
    frames: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Encode (B,T,C,H,W) → L2-normalized mean-pooled (B,D)."""
    from echoclip.temporal import pool_frame_features

    if not hasattr(model, "encode_frame_features"):
        raise TypeError("model must provide encode_frame_features")
    feats = model.encode_frame_features(frames)
    return pool_frame_features(feats, aggregator=None, mask=mask, normalize=True)


def train_mlp_ef_step(
    head: nn.Module,
    features: torch.Tensor,
    targets: torch.Tensor,
    optimizer: torch.optim.Optimizer,
    *,
    loss_kind: str = "huber",
) -> float:
    """One optimizer step for an MLP/linear head on precomputed features."""
    head.train()
    optimizer.zero_grad(set_to_none=True)
    pred = head(features)
    loss = ef_regression_loss(pred, targets, kind=loss_kind)
    loss.backward()
    optimizer.step()
    return float(loss.item())


def train_temporal_ef_step(
    model: nn.Module,
    frames: torch.Tensor,
    targets: torch.Tensor,
    optimizer: torch.optim.Optimizer,
    *,
    loss_kind: str = "huber",
    mask: Optional[torch.Tensor] = None,
) -> float:
    """One step: encode frames → TemporalEFRegressor → L1/Huber."""
    model.train()
    optimizer.zero_grad(set_to_none=True)
    if not isinstance(model, TemporalEFRegressor):
        raise TypeError("Expected TemporalEFRegressor")
    # Caller should pass frame features or we need a backbone — accept features
    # shaped (B,T,D) when frames.dim()==3, else expect a separate path.
    if frames.dim() == 3:
        pred = model(frames, mask=mask)
    else:
        raise ValueError(
            "train_temporal_ef_step expects precomputed frame features (B,T,D). "
            "Encode with a frozen backbone first."
        )
    loss = ef_regression_loss(pred, targets, kind=loss_kind)
    loss.backward()
    optimizer.step()
    return float(loss.item())


def supervised_api_smoke(
    dim: int = 32, t: int = 4, batch: int = 3
) -> Dict[str, float]:
    """Minimal API check with toy tensors (no EchoNet / hub required)."""
    feats = F.normalize(torch.randn(batch, t, dim), dim=-1)
    y = torch.tensor([45.0, 55.0, 35.0][:batch])
    mean_z = feats.mean(dim=1)
    ridge = fit_ridge_ef(mean_z, y, alpha=0.1)
    pred_ridge = ridge.predict(mean_z)
    linear = LinearEFHead(dim)
    opt = torch.optim.SGD(linear.parameters(), lr=0.05)
    loss_lin = train_mlp_ef_step(linear, mean_z, y, opt, loss_kind="l1")
    mlp = MLPEFHead(dim, hidden=16)
    opt2 = torch.optim.Adam(mlp.parameters(), lr=1e-2)
    loss_mlp = train_mlp_ef_step(mlp, mean_z.detach(), y, opt2)
    from echoclip.temporal import TemporalTransformer

    agg = TemporalTransformer(dim, n_layers=1, n_heads=4, max_frames=t)
    reg = TemporalEFRegressor(agg, dim)
    opt3 = torch.optim.Adam(reg.parameters(), lr=1e-2)
    loss_t = train_temporal_ef_step(reg, feats, y, opt3)
    return {
        "ridge_mae": float(torch.mean(torch.abs(pred_ridge - y)).item()),
        "linear_loss": loss_lin,
        "mlp_loss": loss_mlp,
        "temporal_l1_loss": loss_t,
    }
