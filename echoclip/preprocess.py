"""Echocardiogram-specific image/video preprocessing (EchoCLIP paper / echonet utils)."""

from pathlib import Path
from typing import List, Optional, Tuple, Union

import cv2
import numpy as np
import torch
from PIL import Image

PathLike = Union[str, Path]

# OpenAI CLIP normalization constants
_CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
_CLIP_STD = (0.26862954, 0.26130258, 0.27577711)


def _pil_to_clip_tensor(pil: Image.Image, image_size: int) -> torch.Tensor:
    """Resize, to tensor, and CLIP-normalize without torchvision."""
    pil = pil.resize((image_size, image_size), Image.BICUBIC)
    arr = np.asarray(pil, dtype=np.float32) / 255.0
    if arr.ndim == 2:
        arr = np.stack([arr, arr, arr], axis=-1)
    tensor = torch.from_numpy(arr).permute(2, 0, 1)
    mean = torch.tensor(_CLIP_MEAN).view(3, 1, 1)
    std = torch.tensor(_CLIP_STD).view(3, 1, 1)
    return (tensor - mean) / std


def crop_and_scale(
    img: np.ndarray,
    res: Tuple[int, int] = (640, 480),
    interpolation: int = cv2.INTER_CUBIC,
    zoom: float = 0.1,
) -> np.ndarray:
    """Letterbox crop and resize echo frames (matches echonet/echo_CLIP utils)."""
    in_res = (img.shape[1], img.shape[0])
    r_in = in_res[0] / in_res[1]
    r_out = res[0] / res[1]

    if r_in > r_out:
        padding = int(round((in_res[0] - r_out * in_res[1]) / 2))
        if padding > 0:
            img = img[:, padding:-padding]
    elif r_in < r_out:
        padding = int(round((in_res[1] - in_res[0] / r_out) / 2))
        if padding > 0:
            img = img[padding:-padding]
    if zoom > 0:
        pad_x = max(1, round(int(img.shape[1] * zoom)))
        pad_y = max(1, round(int(img.shape[0] * zoom)))
        if img.shape[0] > 2 * pad_y and img.shape[1] > 2 * pad_x:
            img = img[pad_y:-pad_y, pad_x:-pad_x]

    return cv2.resize(img, res, interpolation=interpolation)


def read_video_frames(
    path: PathLike,
    res: Optional[Tuple[int, int]] = (640, 480),
    max_frames: Optional[int] = None,
) -> np.ndarray:
    """Load AVI/MP4 frames; returns (T, H, W, C) uint8 RGB."""
    cap = cv2.VideoCapture(str(path))
    frames: List[np.ndarray] = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        if res is not None:
            frame = crop_and_scale(frame, res)
        frames.append(frame)
        if max_frames and len(frames) >= max_frames:
            break
    cap.release()
    if not frames:
        raise ValueError(f"No frames read from video: {path}")
    return np.array(frames)


def sample_frame_indices(
    num_frames: int,
    n_samples: int,
    seed: Optional[int] = None,
    strategy: str = "random",
) -> np.ndarray:
    """Sample frame indices: random (training) or uniform (inference)."""
    if n_samples >= num_frames:
        return np.arange(num_frames)
    if strategy == "uniform":
        return np.linspace(0, num_frames - 1, n_samples, dtype=int)
    rng = np.random.default_rng(seed)
    return np.sort(rng.choice(num_frames, size=n_samples, replace=False))


def frames_to_tensor(
    frames: np.ndarray,
    indices: np.ndarray,
    image_size: int = 224,
) -> torch.Tensor:
    """Convert selected frames to normalized CHW tensor batch."""
    tensors = []
    for idx in indices:
        pil = Image.fromarray(frames[int(idx)])
        tensors.append(_pil_to_clip_tensor(pil, image_size))
    return torch.stack(tensors, dim=0)


def load_image_tensor(path: PathLike, image_size: int = 224) -> torch.Tensor:
    """Load a single echo still frame."""
    return _pil_to_clip_tensor(Image.open(path).convert("RGB"), image_size)
