"""Echocardiogram image/video + report text dataset."""

import csv
import json
import random
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import torch
from torch.utils.data import Dataset

from echoclip.cycle_sample import pad_or_trim_indices, sample_cycle_indices
from echoclip.preprocess import (
    frames_to_tensor,
    load_image_tensor,
    read_video_frames,
    sample_frame_indices,
)
from echoclip.text import EchoTokenizer

PathLike = Union[str, Path]


def load_manifest(path: PathLike) -> List[Dict[str, str]]:
    """Load pairs from JSON manifest or CSV (columns: image_path, text)."""
    path = Path(path)
    if path.suffix.lower() == ".json":
        data = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(data, dict) and "pairs" in data:
            return data["pairs"]
        if isinstance(data, list):
            return data
        raise ValueError("JSON manifest must be a list or {\"pairs\": [...]}")
    if path.suffix.lower() == ".csv":
        rows = []
        with path.open(newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                img = row.get("image_path") or row.get("image") or row.get("path")
                text = row.get("text") or row.get("report") or row.get("caption")
                if img and text:
                    rows.append({"image": img, "text": text})
        return rows
    raise ValueError(f"Unsupported manifest format: {path}")


def validate_manifest(
    pairs: List[Dict[str, str]],
    root: PathLike,
    check_files: bool = True,
) -> List[str]:
    """Return list of validation error messages (empty if OK)."""
    errors: List[str] = []
    root = Path(root)
    if not pairs:
        errors.append("manifest has zero pairs")
        return errors
    for i, item in enumerate(pairs):
        if "image" not in item or "text" not in item:
            errors.append(f"pair[{i}]: missing 'image' or 'text' key")
            continue
        if not str(item["text"]).strip():
            errors.append(f"pair[{i}]: empty text")
        if check_files:
            rel = item["image"]
            path = Path(rel) if Path(rel).is_absolute() else root / rel
            if not path.exists():
                errors.append(f"pair[{i}]: file not found: {path}")
    return errors


class EchoCLIPDataset(Dataset):
    """
    Each item: echo still or sampled video frames + paired report text.

    Manifest entries:
      - image: path to .png/.jpg or video .avi/.mp4
      - text: clinical report string
      - optional: ef, edv, esv, ed_frame, es_frame, captions, split
    Paths may be relative to manifest_dir.

    ``video_frames == 1`` keeps the original single-frame (C, H, W) behavior.
    ``video_frames > 1`` returns a clip (T, C, H, W) unless ``frame_pool='mean'``.
    Still images are repeated along T so batched TC training does not crash.
    """

    VIDEO_EXTENSIONS = {".avi", ".mp4", ".mov", ".mkv"}

    def __init__(
        self,
        manifest_path: PathLike,
        manifest_dir: Optional[PathLike] = None,
        image_size: int = 224,
        context_length: int = 77,
        video_frames: int = 1,
        crop_res: tuple = (640, 480),
        seed: Optional[int] = None,
        tokenizer: Optional[EchoTokenizer] = None,
        sample_strategy: str = "random",
        frame_pool: str = "stack",
        two_views: bool = False,
        caption_mode: str = "primary",
        use_edv_captions: bool = False,
    ):
        self.manifest_path = Path(manifest_path)
        self.root = Path(manifest_dir) if manifest_dir else self.manifest_path.parent
        self.image_size = image_size
        self.video_frames = video_frames
        self.crop_res = crop_res
        self.seed = seed
        self.pairs = load_manifest(self.manifest_path)
        self.tokenizer = tokenizer or EchoTokenizer(context_length=context_length)
        self.sample_strategy = sample_strategy
        self.frame_pool = frame_pool
        self.two_views = two_views
        self.caption_mode = caption_mode
        # Primary R5 path: EF-only captions. EDV dilation is opt-in ablation.
        self.use_edv_captions = bool(use_edv_captions)
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        """Set epoch for deterministic per-item seeds (VAL/TEST should stay 0)."""
        self.epoch = int(epoch)

    def __len__(self) -> int:
        return len(self.pairs)

    def _resolve(self, rel: str) -> Path:
        p = Path(rel)
        return p if p.is_absolute() else self.root / p

    def _item_seed(self, index: int, view: int = 0) -> Optional[int]:
        if self.seed is None:
            return None
        # Include epoch so train reshuffles across epochs; keep VAL/TEST at epoch=0.
        return int(self.seed + index + 10007 * view + 100003 * int(self.epoch))

    @staticmethod
    def _is_ef_caption(text: str) -> bool:
        t = str(text).upper()
        return "EJECTION FRACTION" in t or "LVEF" in t

    def _ef_only_captions(self, item: Dict[str, Any]) -> List[str]:
        """Prefer EF prompt sentences; rebuild from ``ef`` when needed."""
        captions = item.get("captions")
        if isinstance(captions, list) and captions:
            ef_caps = [str(c) for c in captions if self._is_ef_caption(c)]
            if ef_caps:
                return ef_caps
        if item.get("ef") is not None and str(item.get("ef")) != "":
            from echoclip.structured_text import captions_from_measurements

            return captions_from_measurements(
                ef=float(item["ef"]), include_dilation=False
            )
        text = item.get("text") or ""
        return [str(text)] if text else []

    def _choose_text(self, item: Dict[str, Any], index: int) -> str:
        if self.use_edv_captions:
            captions = item.get("captions")
        else:
            captions = self._ef_only_captions(item)
        if self.caption_mode == "random" and isinstance(captions, list) and captions:
            rng = random.Random(self._item_seed(index) or index)
            return str(rng.choice(captions))
        if self.caption_mode == "join" and isinstance(captions, list) and captions:
            return " ".join(str(c) for c in captions)
        if not self.use_edv_captions and isinstance(captions, list) and captions:
            return str(captions[0])
        return item["text"]

    def _sample_video_tensor(
        self,
        frames,
        index: int,
        item: Dict[str, Any],
        view: int = 0,
    ) -> torch.Tensor:
        n_total = len(frames)
        n_want = self.video_frames
        seed = self._item_seed(index, view)
        ed = item.get("ed_frame", item.get("ed_index"))
        es = item.get("es_frame", item.get("es_index"))
        ed_i = int(ed) if ed is not None and str(ed) != "" else None
        es_i = int(es) if es is not None and str(es) != "" else None

        use_legacy = n_want == 1 and self.sample_strategy in ("random", "uniform")
        if use_legacy:
            idx = sample_frame_indices(
                n_total, n_want, seed=seed, strategy=self.sample_strategy
            )
            idx = pad_or_trim_indices(idx, n_want, n_total)
        else:
            idx = sample_cycle_indices(
                n_total,
                n_want,
                strategy=self.sample_strategy,
                ed_index=ed_i,
                es_index=es_i,
                seed=seed,
            )
            # Official EchoCLIP path: keep 0:min(40,T):2 length — do NOT pad/trim to 16.
            strat = str(self.sample_strategy).strip().lower()
            if strat not in ("official_stride", "official", "echo_clip_stride"):
                idx = pad_or_trim_indices(idx, n_want, n_total)
        clip = frames_to_tensor(frames, idx, self.image_size)
        if n_want == 1 and str(self.sample_strategy).strip().lower() not in (
            "official_stride",
            "official",
            "echo_clip_stride",
        ):
            return clip[0]
        if self.frame_pool == "mean":
            return clip.mean(dim=0)
        return clip

    def _as_clip(self, tensor: torch.Tensor) -> torch.Tensor:
        """Still (C,H,W) → (T,C,H,W) when video_frames > 1 and stacking."""
        if self.video_frames <= 1 or self.frame_pool == "mean":
            return tensor
        if tensor.dim() == 3:
            return tensor.unsqueeze(0).expand(self.video_frames, -1, -1, -1).contiguous()
        return tensor

    def __getitem__(self, index: int) -> Dict[str, Any]:
        item = self.pairs[index]
        media_path = self._resolve(item["image"])
        text = self._choose_text(item, index)

        if media_path.suffix.lower() in self.VIDEO_EXTENSIONS:
            frames = read_video_frames(media_path, res=self.crop_res)
            tensor = self._sample_video_tensor(frames, index, item, view=0)
            tensor2 = None
            if self.two_views:
                tensor2 = self._sample_video_tensor(frames, index, item, view=1)
        else:
            tensor = load_image_tensor(media_path, self.image_size)
            tensor2 = tensor.clone() if self.two_views else None

        tensor = self._as_clip(tensor)
        if tensor2 is not None:
            tensor2 = self._as_clip(tensor2)

        tokens = self.tokenizer.encode([text])[0]
        out: Dict[str, Any] = {"image": tensor, "text": tokens, "raw_text": text}
        if tensor2 is not None:
            out["image_2"] = tensor2
        if item.get("ef") is not None and str(item.get("ef")) != "":
            out["ef"] = float(item["ef"])
        else:
            out["ef"] = None
        if item.get("edv") is not None and str(item.get("edv")) != "":
            out["edv"] = float(item["edv"])
        if item.get("esv") is not None and str(item.get("esv")) != "":
            out["esv"] = float(item["esv"])
        return out


def collate_batch(batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
    """Stack images; pad variable-T clips (e.g. official_stride) to max T in batch."""
    imgs = [b["image"] for b in batch]
    if imgs[0].dim() == 4:  # (T,C,H,W) — may differ in T
        max_t = max(im.shape[0] for im in imgs)
        padded = []
        for im in imgs:
            if im.shape[0] == max_t:
                padded.append(im)
            elif im.shape[0] > max_t:
                padded.append(im[:max_t])
            else:
                # Repeat last frame (same convention as pad_or_trim_indices)
                pad_n = max_t - im.shape[0]
                last = im[-1:].expand(pad_n, -1, -1, -1)
                padded.append(torch.cat([im, last], dim=0))
        images = torch.stack(padded)
    else:
        images = torch.stack(imgs)
    texts = torch.stack([b["text"] for b in batch])
    out: Dict[str, torch.Tensor] = {"image": images, "text": texts}
    if "image_2" in batch[0]:
        imgs2 = [b["image_2"] for b in batch]
        if imgs2[0].dim() == 4:
            max_t = max(im.shape[0] for im in imgs2)
            padded2 = []
            for im in imgs2:
                if im.shape[0] == max_t:
                    padded2.append(im)
                elif im.shape[0] > max_t:
                    padded2.append(im[:max_t])
                else:
                    pad_n = max_t - im.shape[0]
                    last = im[-1:].expand(pad_n, -1, -1, -1)
                    padded2.append(torch.cat([im, last], dim=0))
            out["image_2"] = torch.stack(padded2)
        else:
            out["image_2"] = torch.stack(imgs2)
    efs = [b.get("ef") for b in batch]
    if any(v is not None for v in efs):
        out["ef"] = torch.tensor(
            [float("nan") if v is None else float(v) for v in efs],
            dtype=torch.float32,
        )
    return out


def split_manifest(
    pairs: List[Dict[str, str]], val_ratio: float = 0.1, seed: int = 42
) -> tuple:
    if len(pairs) < 2:
        raise ValueError("Need at least 2 samples to split train/val")
    rng = random.Random(seed)
    indices = list(range(len(pairs)))
    rng.shuffle(indices)
    n_val = min(len(pairs) - 1, max(1, int(len(pairs) * val_ratio)))
    val_idx = set(indices[:n_val])
    train = [pairs[i] for i in range(len(pairs)) if i not in val_idx]
    val = [pairs[i] for i in val_idx]
    return train, val
