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
