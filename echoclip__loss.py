from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class ClipLoss(nn.Module):
    """Symmetric InfoNCE loss used by CLIP / EchoCLIP."""

    def __init__(self, local_loss: bool = False):
        super().__init__()
        self.local_loss = local_loss

    def forward(
        self,
        image_features: torch.Tensor,
        text_features: torch.Tensor,
        logit_scale: torch.Tensor,
    ) -> torch.Tensor:
        image_features = F.normalize(image_features, dim=-1)
        text_features = F.normalize(text_features, dim=-1)
        logit_scale = logit_scale.exp()

        logits_per_image = logit_scale * image_features @ text_features.T
        logits_per_text = logits_per_image.T
        batch_size = image_features.shape[0]
        labels = torch.arange(batch_size, device=image_features.device)

        loss_i = F.cross_entropy(logits_per_image, labels)
        loss_t = F.cross_entropy(logits_per_text, labels)
        return (loss_i + loss_t) / 2.0


class TemporalClipLoss(nn.Module):
    """Video-level InfoNCE, optionally mixed with a second-view consistency term.

    ``clip_weight`` applies to video–text CLIP. If ``video_features_2`` is given,
    ``view_weight`` adds InfoNCE between two cycle samples of the same batch
    (same-index pairs are positives; other patients are negatives).
    """

    def __init__(self, clip_weight: float = 1.0, view_weight: float = 0.0):
        super().__init__()
        self.clip_loss = ClipLoss()
        self.clip_weight = float(clip_weight)
        self.view_weight = float(view_weight)

    def forward(
        self,
        video_features: torch.Tensor,
        text_features: torch.Tensor,
        logit_scale: torch.Tensor,
        video_features_2: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        loss = self.clip_weight * self.clip_loss(
            video_features, text_features, logit_scale
        )
        if video_features_2 is not None and self.view_weight:
            loss = loss + self.view_weight * self.clip_loss(
                video_features, video_features_2, logit_scale
            )
        return loss


class EFSoftContrastiveLoss(nn.Module):
    """EF-aware soft / multi-positive contrastive loss (optional train flag).

    Videos with similar EF are soft positives. Target distribution for video i
    over texts j is proportional to ``exp(-|EF_i - EF_j| / temperature_ef)``
    (self always included).

    Hard / soft mixture (P1-3)
    --------------------------
    ``lambda_soft`` is a *mix ratio* in ``[0, 1]``::

        loss = lambda_soft * soft_EF_loss + (1 - lambda_soft) * hard_InfoNCE

    ``lambda_soft=0`` is **exactly** hard InfoNCE (soft branch not evaluated),
    ``lambda_soft=1`` is pure soft (legacy behavior). ``lambda_soft`` defaults
    to 1.0 so existing configs keep their numbers.

    Finite-EF subset (P1-2)
    -----------------------
    Rows with NaN/missing EF are **excluded** from the soft target matrix rather
    than polluting it, and the soft term is computed on the finite subset only
    (``n_valid``). When fewer than 2 finite EF labels remain, the soft term is
    undefined and the loss falls back to hard InfoNCE for the whole batch.
    """

    def __init__(
        self,
        ef_temperature: float = 5.0,
        soft_weight: Optional[float] = None,
        lambda_soft: float = 1.0,
    ):
        super().__init__()
        self.ef_temperature = float(ef_temperature)
        # ``soft_weight`` was misleadingly named (it scaled the KL term rather
        # than mixing). Kept as a deprecated alias for ``lambda_soft``.
        if soft_weight is not None:
            import warnings

            warnings.warn(
                "EFSoftContrastiveLoss(soft_weight=...) is deprecated; use "
                "lambda_soft=... (a hard/soft mix ratio, 0 == hard InfoNCE).",
                DeprecationWarning,
                stacklevel=2,
            )
            lambda_soft = float(soft_weight)
        self.lambda_soft = float(min(max(lambda_soft, 0.0), 1.0))
        # Backwards-compatible attribute (documented as the mix ratio).
        self.soft_weight = self.lambda_soft
        self.hard = ClipLoss()

    def forward(
        self,
        video_features: torch.Tensor,
        text_features: torch.Tensor,
        logit_scale: torch.Tensor,
        ef: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        hard = self.hard(video_features, text_features, logit_scale)
        # lambda_soft == 0 must be *exactly* hard InfoNCE (no soft computation).
        if self.lambda_soft <= 0.0:
            return hard
        if ef is None:
            return hard

        finite = torch.isfinite(ef)
        if int(finite.sum()) < 2:
            # Soft targets are undefined with <2 valid labels: hard fallback.
            return hard

        video_features = F.normalize(video_features, dim=-1)
        text_features = F.normalize(text_features, dim=-1)

        # Restrict to the finite-EF subset (P1-2): no NaN rows/cols in the
        # target matrix, and no cross-talk from dropped samples.
        ef_valid = ef.float()[finite]
        vid_valid = video_features[finite]
        txt_valid = text_features[finite]

        scale = logit_scale.exp()
        logits = scale * vid_valid @ txt_valid.T  # (n_valid, n_valid)
        logits_t = logits.T

        dist = torch.abs(ef_valid.view(-1, 1) - ef_valid.view(1, -1))
        soft = torch.exp(-dist / max(self.ef_temperature, 1e-3))
        soft = soft / soft.sum(dim=1, keepdim=True).clamp(min=1e-8)

        log_prob = F.log_softmax(logits, dim=1)
        soft_loss_i = -(soft * log_prob).sum(dim=1)
        log_prob_t = F.log_softmax(logits_t, dim=1)
        soft_loss_t = -(soft * log_prob_t).sum(dim=1)
        soft_loss = (soft_loss_i.mean() + soft_loss_t.mean()) / 2.0

        return self.lambda_soft * soft_loss + (1.0 - self.lambda_soft) * hard
