"""Zero-shot classification, regression, and cross-modal retrieval."""

from typing import Dict, List, Optional, Sequence, Union

import torch

from echoclip.model import EchoCLIP
from echoclip.prompts import ZERO_SHOT_PROMPTS
from echoclip.text import EchoTokenizer

PathLike = Union[str, torch.Tensor]


def _as_frame_batch(embeddings: torch.Tensor) -> torch.Tensor:
    """Normalize embeddings to ``(batch, frames, dim)``.

    Conventions (deliberate; do not change casually):
    - ``(D,)`` → one video, one frame
    - ``(T, D)`` → one video, T frames  (**not** a batch of video vectors)
    - ``(B, T, D)`` → already batched

    Batched mean/temporal video vectors ``(B, D)`` must be reshaped to
    ``(B, 1, D)`` by the caller (see ``EchoCLIPInference.zero_shot_ef_batch``).
    """
    if embeddings.dim() == 1:
        return embeddings.unsqueeze(0).unsqueeze(0)
    if embeddings.dim() == 2:
        return embeddings.unsqueeze(0)
    return embeddings


def _as_prompt_batch(embeddings: torch.Tensor) -> torch.Tensor:
    """(prompts, dim) -> (1, prompts, dim)."""
    if embeddings.dim() == 2:
        return embeddings.unsqueeze(0)
    return embeddings


def compute_binary_score(
    video_embeddings: torch.Tensor,
    prompt_embeddings: torch.Tensor,
) -> torch.Tensor:
    """Average frame × candidate similarities (echonet/echo_CLIP)."""
    video = _as_frame_batch(video_embeddings)
    prompts = _as_prompt_batch(prompt_embeddings)
    per_frame = torch.matmul(video, prompts.transpose(-1, -2))
    return per_frame.mean(dim=-1).mean(dim=-1)


def compute_regression_score(
    video_embeddings: torch.Tensor,
    prompt_embeddings: torch.Tensor,
    prompt_values: Sequence[float],
) -> torch.Tensor:
    video = _as_frame_batch(video_embeddings)
    prompts = _as_prompt_batch(prompt_embeddings)
    per_frame = torch.matmul(video, prompts.transpose(-1, -2))
    ranked = torch.argsort(per_frame, dim=-1, descending=True)
    values = torch.tensor(prompt_values, device=video_embeddings.device)
    ranked_values = values[ranked]
    avg_frames = ranked_values.float().mean(dim=1)
    top_k = max(1, int(avg_frames.shape[1] * 0.2))
    return avg_frames[:, :top_k].median(dim=-1).values


class EchoCLIPInference:
    """Encode echoes and text; run zero-shot tasks without task-specific heads."""

    def __init__(
        self,
        model: EchoCLIP,
        device: Optional[str] = None,
        tokenizer: Optional[EchoTokenizer] = None,
    ):
        self.model = model
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device).eval()
        self.tokenizer = tokenizer or EchoTokenizer(context_length=model.config.context_length)

    @torch.inference_mode()
    def encode_texts(self, texts: List[str], clean: bool = True) -> torch.Tensor:
        tokens = self.tokenizer.encode(texts, clean=clean).to(self.device)
        return self.model.encode_text(tokens)

    @torch.inference_mode()
    def encode_images(self, images: torch.Tensor) -> torch.Tensor:
        images = images.to(self.device)
        if images.dim() == 3:
            images = images.unsqueeze(0)
        return self.model.encode_image(images)

    @torch.inference_mode()
    def encode_video_frames(self, frames: torch.Tensor, pool: str = "auto") -> torch.Tensor:
        frames = frames.to(self.device)
        if frames.dim() == 3:
            frames = frames.unsqueeze(0)
        return self.model.encode_video_frames(frames, pool=pool)

    @torch.inference_mode()
    def encode_video(self, frames: torch.Tensor) -> torch.Tensor:
        """(B, T, C, H, W) or (T, C, H, W) → video embedding(s)."""
        frames = frames.to(self.device)
        if frames.dim() == 4:
            frames = frames.unsqueeze(0)
            return self.model.encode_video(frames).squeeze(0)
        return self.model.encode_video(frames)

    def similarity(self, image_emb: torch.Tensor, text_emb: torch.Tensor) -> torch.Tensor:
        if image_emb.dim() == 1:
            image_emb = image_emb.unsqueeze(0)
        if text_emb.dim() == 1:
            text_emb = text_emb.unsqueeze(0)
        scale = self.model.logit_scale.exp()
        return scale * (image_emb @ text_emb.T)

    def zero_shot_binary(
        self,
        frame_embeddings: torch.Tensor,
        positive_prompts: List[str],
        negative_prompts: List[str],
    ) -> Dict[str, float]:
        """
        Compare mean similarity to positive vs negative prompt sets.
        frame_embeddings: (n_frames, embed_dim)
        """
        pos_emb = self.encode_texts(positive_prompts)
        neg_emb = self.encode_texts(negative_prompts)
        pos_score = compute_binary_score(frame_embeddings, pos_emb).item()
        neg_score = compute_binary_score(frame_embeddings, neg_emb).item()
        return {
            "positive_score": pos_score,
            "negative_score": neg_score,
            "prediction": pos_score > neg_score,
        }

    def zero_shot_ef(
        self,
        frame_embeddings: torch.Tensor,
        ef_values: Optional[List[int]] = None,
        prompt_embeddings: Optional[torch.Tensor] = None,
        prompt_values: Optional[Sequence[float]] = None,
    ) -> float:
        """Zero-shot left ventricular ejection fraction (regression prompts).

        Aggregation matches official echo_CLIP: rank prompts per embedding,
        average across the T axis, take the median of the top 20% EF values.
        Only ``frame_embeddings`` (how z_v is obtained) should change for TC.
        """
        values, emb = self._ef_prompt_pack(
            ef_values=ef_values,
            prompt_embeddings=prompt_embeddings,
            prompt_values=prompt_values,
        )
        return compute_regression_score(frame_embeddings, emb, values).reshape(-1)[0].item()

    def zero_shot_ef_batch(
        self,
        video_embeddings: torch.Tensor,
        ef_values: Optional[List[int]] = None,
        prompt_embeddings: Optional[torch.Tensor] = None,
        prompt_values: Optional[Sequence[float]] = None,
    ) -> torch.Tensor:
        """Vectorized EF regression.

        video_embeddings: (B, D) video-level or (B, T, D) frame-level.
        (T, D) for a single video is handled by ``zero_shot_ef``.
        Aggregation is the official top-20% median over prompt values.
        """
        values, emb = self._ef_prompt_pack(
            ef_values=ef_values,
            prompt_embeddings=prompt_embeddings,
            prompt_values=prompt_values,
        )
        if video_embeddings.dim() == 2:
            video_embeddings = video_embeddings.unsqueeze(1)
        return compute_regression_score(video_embeddings, emb, values)

    def _ef_prompt_pack(
        self,
        ef_values: Optional[List[int]] = None,
        prompt_embeddings: Optional[torch.Tensor] = None,
        prompt_values: Optional[Sequence[float]] = None,
    ):
        if prompt_embeddings is not None and prompt_values is not None:
            return list(prompt_values), prompt_embeddings
        templates = ZERO_SHOT_PROMPTS["ejection_fraction"]
        ef_values = ef_values or list(range(15, 81, 5))
        prompts = [self.tokenizer.fill_prompt(t, v) for t in templates for v in ef_values]
        values = [float(v) for _ in templates for v in ef_values]
        return values, self.encode_texts(prompts)

    def retrieve_text(
        self,
        image_emb: torch.Tensor,
        candidate_texts: List[str],
        top_k: int = 5,
    ) -> List[Dict[str, Union[str, float]]]:
        text_emb = self.encode_texts(candidate_texts)
        scores = (image_emb @ text_emb.T).squeeze(0)
        ranked = torch.argsort(scores, descending=True)[:top_k]
        return [
            {"text": candidate_texts[i], "score": scores[i].item()}
            for i in ranked.tolist()
        ]

    def classify_labels(
        self,
        frame_embeddings: torch.Tensor,
        class_prompts: Dict[str, List[str]],
    ) -> str:
        """Pick the class whose prompt set has highest mean similarity."""
        best_label = None
        best_score = float("-inf")
        for label, prompts in class_prompts.items():
            emb = self.encode_texts(prompts)
            score = compute_binary_score(frame_embeddings, emb).item()
            if score > best_score:
                best_score = score
                best_label = label
        if best_label is None:
            raise ValueError("class_prompts must contain at least one label")
        return best_label
