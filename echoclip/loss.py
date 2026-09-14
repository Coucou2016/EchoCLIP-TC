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

    ``soft_weight`` scales the soft EF multi-positive term (1.0 = full soft
    loss). When some EF labels are NaN/missing, those rows fall back to hard
    diagonal InfoNCE — the whole batch is **not** dropped.
    """

    def __init__(self, ef_temperature: float = 5.0, soft_weight: float = 1.0):
        super().__init__()
        self.ef_temperature = float(ef_temperature)
        # soft_weight: multiplier on the soft multi-positive KL term (not a mix ratio).
        self.soft_weight = float(soft_weight)
        self.hard = ClipLoss()

    def forward(
        self,
        video_features: torch.Tensor,
        text_features: torch.Tensor,
        logit_scale: torch.Tensor,
        ef: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if ef is None:
            return self.hard(video_features, text_features, logit_scale)

        finite = torch.isfinite(ef)
        if not bool(finite.any()):
            return self.hard(video_features, text_features, logit_scale)

        video_features = F.normalize(video_features, dim=-1)
        text_features = F.normalize(text_features, dim=-1)
        scale = logit_scale.exp()
        logits = scale * video_features @ text_features.T  # (B, B)
        batch_size = video_features.shape[0]
        labels = torch.arange(batch_size, device=video_features.device)

        # Soft targets only among finite-EF pairs; missing EF → hard one-hot.
        ef_f = torch.where(finite, ef.float(), torch.zeros_like(ef.float())).view(-1, 1)
        dist = torch.abs(ef_f - ef_f.T)
        soft = torch.exp(-dist / max(self.ef_temperature, 1e-3))
        # Zero out rows/cols with non-finite EF so they do not pollute soft targets
        mask = finite.float().view(-1, 1) * finite.float().view(1, -1)
        soft = soft * mask
        row_sum = soft.sum(dim=1, keepdim=True).clamp(min=1e-8)
        soft = soft / row_sum

        log_prob = F.log_softmax(logits, dim=1)
        soft_loss_i = -(soft * log_prob).sum(dim=1)
        log_prob_t = F.log_softmax(logits.T, dim=1)
        soft_loss_t = -(soft * log_prob_t).sum(dim=1)

        hard_loss_i = F.cross_entropy(logits, labels, reduction="none")
        hard_loss_t = F.cross_entropy(logits.T, labels, reduction="none")

        # Finite EF → soft; missing → hard (do not drop the batch)
        loss_i = torch.where(finite, soft_loss_i, hard_loss_i)
        loss_t = torch.where(finite, soft_loss_t, hard_loss_t)
        return self.soft_weight * (loss_i.mean() + loss_t.mean()) / 2.0
