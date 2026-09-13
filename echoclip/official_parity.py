"""Reference logic aligned with echonet/echo_CLIP ``utils.py`` (public).

Used for golden tests and ``scripts/compare_official_b0.py``.
Does **not** claim bit-exact end-to-end parity (tokenizer / crop path / dtype
gaps remain; see ``PARITY_GAPS``).
"""

from __future__ import annotations

from typing import Sequence, Tuple

import numpy as np
import torch

# Documented remaining gaps vs echonet/echo_CLIP zero_shot_example.py
PARITY_GAPS: Tuple[str, ...] = (
    "Tokenizer: open_clip tokenize vs local EchoTokenizer / CLIPTokenizer quirks",
    "Crop path: official read_avi often crops directly to 224×224; default dataset path uses 640×480 letterbox then 224 CLIP resize",
    "Color: official read_avi leaves OpenCV BGR; we convert BGR→RGB before tensorize",
    "Dtype: official example uses bf16 on CUDA; CPU path is float32",
    "open_clip preprocess_val when hub load succeeds vs local CLIP mean/std resize",
)


def official_stride_indices(num_frames: int, max_span: int = 40, stride: int = 2) -> np.ndarray:
    """Match ``test_video[0:min(40, len(test_video)):2]`` in zero_shot_example.py."""
    end = min(int(max_span), int(num_frames))
    return np.arange(0, end, int(stride), dtype=int)


def compute_regression_metric_official(
    video_embeddings: torch.Tensor,
    prompt_embeddings: torch.Tensor,
    prompt_values: Sequence[float],
) -> torch.Tensor:
    """Bit-aligned copy of echonet/echo_CLIP ``utils.compute_regression_metric``.

    Expects ``video_embeddings`` as ``(N, Frames, Dim)`` and
    ``prompt_embeddings`` as ``(Candidates, Dim)`` (already L2-normalized).
    """
    per_frame_similarities = video_embeddings @ prompt_embeddings.T
    ranked_candidate_phrase_indices = torch.argsort(
        per_frame_similarities, dim=-1, descending=True
    )
    values = torch.tensor(prompt_values, device=video_embeddings.device)
    all_frames_ranked_values = values[ranked_candidate_phrase_indices]
    avg_frame_ranked_values = all_frames_ranked_values.float().mean(dim=1)
    twenty_percent = int(avg_frame_ranked_values.shape[1] * 0.2)
    # Official uses bare int(...); empty slice if candidates < 5.
    final_prediction = avg_frame_ranked_values[:, :twenty_percent].median(dim=-1)[0]
    return final_prediction


def crop_and_scale_official(
    img: np.ndarray,
    res: Tuple[int, int] = (640, 480),
    interpolation=None,
    zoom: float = 0.1,
) -> np.ndarray:
    """Match echonet/echo_CLIP ``utils.crop_and_scale`` (no empty-slice guards)."""
    import cv2

    if interpolation is None:
        interpolation = cv2.INTER_CUBIC
    in_res = (img.shape[1], img.shape[0])
    r_in = in_res[0] / in_res[1]
    r_out = res[0] / res[1]

    if r_in > r_out:
        padding = int(round((in_res[0] - r_out * in_res[1]) / 2))
        img = img[:, padding:-padding]
    if r_in < r_out:
        padding = int(round((in_res[1] - in_res[0] / r_out) / 2))
        img = img[padding:-padding]
    if zoom != 0:
        pad_x = round(int(img.shape[1] * zoom))
        pad_y = round(int(img.shape[0] * zoom))
        img = img[pad_y:-pad_y, pad_x:-pad_x]

    return cv2.resize(img, res, interpolation=interpolation)
