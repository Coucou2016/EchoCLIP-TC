"""Evaluation metrics for image–text alignment and retrieval."""

from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F


@torch.no_grad()
def pairwise_retrieval_metrics(
    image_features: torch.Tensor,
    text_features: torch.Tensor,
    logit_scale: Optional[torch.Tensor] = None,
) -> Dict[str, float]:
    """
    Diagonal retrieval on paired batches (image_i matches text_i).

    image_features, text_features: (N, D), L2-normalized or raw (will normalize).
    """
    image_features = F.normalize(image_features, dim=-1)
    text_features = F.normalize(text_features, dim=-1)
    scale = logit_scale.exp() if logit_scale is not None else 1.0
    logits = scale * image_features @ text_features.T
    n = logits.shape[0]
    labels = torch.arange(n, device=logits.device)

    i2t_sorted = logits.argsort(dim=1, descending=True)
    i2t_ranks = (i2t_sorted == labels.view(-1, 1)).nonzero(as_tuple=True)[1].float() + 1
    t2i_sorted = logits.argsort(dim=0, descending=True)
    t2i_ranks = (t2i_sorted == labels.view(1, -1)).nonzero(as_tuple=True)[0].float() + 1

    def summarize(ranks: torch.Tensor) -> Dict[str, float]:
        return {
            "r1": (ranks <= 1).float().mean().item() * 100,
            "r5": (ranks <= min(5, n)).float().mean().item() * 100,
            "median_rank": ranks.median().item(),
            "mean_rank": ranks.mean().item(),
        }

    i2t = summarize(i2t_ranks)
    t2i = summarize(t2i_ranks)
    matched_cosine = (image_features * text_features).sum(dim=-1).mean().item()
    return {
        "i2t_r1": i2t["r1"],
        "i2t_r5": i2t["r5"],
        "i2t_median_rank": i2t["median_rank"],
        "i2t_mean_rank": i2t["mean_rank"],
        "t2i_r1": t2i["r1"],
        "t2i_r5": t2i["r5"],
        "t2i_median_rank": t2i["median_rank"],
        "t2i_mean_rank": t2i["mean_rank"],
        "mean_matched_cosine": matched_cosine,
        "mean_diagonal_logit": logits.diag().mean().item(),
    }


def demo_pacemaker_labels(texts: List[str]) -> List[bool]:
    """Heuristic labels for synthetic demo (NOT clinical ground truth)."""
    keywords = ("PACER", "ICD LEAD", "CATHETER, PACER")
    return [any(k in t.upper() for k in keywords) for t in texts]


@torch.no_grad()
def zero_shot_pacemaker_accuracy(
    frame_embeddings: torch.Tensor,
    texts: List[str],
    engine,
) -> Dict[str, float]:
    """
    Demo-only: compare zero-shot prediction to keyword-derived pseudo-labels.
    """
    from echoclip.prompts import ZERO_SHOT_PROMPTS

    labels = demo_pacemaker_labels(texts)
    neg = ["NORMAL RIGHT VENTRICLE WITHOUT PACER LEAD. ", "NO INTRACARDIAC DEVICE SEEN. "]
    correct = 0
    for i in range(len(texts)):
        emb = frame_embeddings[i]
        pred = engine.zero_shot_binary(emb, ZERO_SHOT_PROMPTS["pacemaker"], neg)["prediction"]
        if pred == labels[i]:
            correct += 1
    return {
        "accuracy_percent": 100.0 * correct / max(len(texts), 1),
        "n_samples": len(texts),
        "note": "Labels are keyword heuristics on demo text only.",
    }
