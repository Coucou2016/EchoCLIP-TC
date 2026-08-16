"""Inference: embeddings, zero-shot tasks, and text retrieval."""

import argparse
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from echoclip.checkpoint import load_checkpoint
from echoclip.preprocess import frames_to_tensor, load_image_tensor, read_video_frames, sample_frame_indices
from echoclip.prompts import ZERO_SHOT_PROMPTS
from echoclip.text import clean_report_text
from echoclip.zeroshot import EchoCLIPInference


def load_media(path: Path, image_size: int, n_frames: int = 8):
    video_ext = {".avi", ".mp4", ".mov", ".mkv"}
    if path.suffix.lower() in video_ext:
        frames = read_video_frames(path)
        idx = sample_frame_indices(
            len(frames), min(n_frames, len(frames)), strategy="uniform"
        )
        return frames_to_tensor(frames, idx, image_size)
    return load_image_tensor(path, image_size).unsqueeze(0)


def main() -> None:
    parser = argparse.ArgumentParser(description="EchoCLIP inference")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--image", type=Path, help="Echo still or video path")
    parser.add_argument("--text", type=str, help="Report text for similarity")
    parser.add_argument("--task", choices=["similarity", "pacemaker", "ef", "retrieve"], default="similarity")
    parser.add_argument("--candidates", nargs="*", help="Texts for retrieval task")
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    model, _ = load_checkpoint(args.checkpoint, device=device)
    engine = EchoCLIPInference(model, device=device)
    image_size = model.config.image_size

    if not args.image:
        parser.error("--image is required")
    media = load_media(args.image, image_size)
    if media.dim() == 4:
        if getattr(model, "temporal", None) is not None:
            frame_emb = engine.encode_video(media)
        else:
            frame_emb = engine.encode_video_frames(media, pool="mean")
    else:
        frame_emb = engine.encode_images(media).squeeze(0)

    if args.task == "similarity":
        if not args.text:
            parser.error("--text required for similarity")
        text_emb = engine.encode_texts([clean_report_text(args.text)])[0]
        cosine = (frame_emb @ text_emb).item()
        scale = engine.model.logit_scale.exp().item()
        print(f"cosine_similarity: {cosine:.4f}")
        print(f"logit_scale_exp: {scale:.4f}")
        print(f"scaled_similarity: {cosine * scale:.4f}")

    elif args.task == "pacemaker":
        result = engine.zero_shot_binary(
            frame_emb,
            ZERO_SHOT_PROMPTS["pacemaker"],
            ["NORMAL RIGHT VENTRICLE WITHOUT PACER LEAD. ", "NO INTRACARDIAC DEVICE SEEN. "],
        )
        print(f"pacemaker_detected: {result['prediction']}")
        print(f"positive_score: {result['positive_score']:.4f}")
        print(f"negative_score: {result['negative_score']:.4f}")

    elif args.task == "ef":
        ef = engine.zero_shot_ef(frame_emb)
        print(f"predicted_ejection_fraction: {ef:.1f}%")

    elif args.task == "retrieve":
        candidates = args.candidates or list(ZERO_SHOT_PROMPTS["pacemaker"]) + list(
            ZERO_SHOT_PROMPTS["ejection_fraction"][:2]
        )
        hits = engine.retrieve_text(frame_emb, candidates, top_k=5)
        for rank, hit in enumerate(hits, 1):
            text_preview = hit["text"][:80]
            suffix = "..." if len(hit["text"]) > 80 else ""
            print(f"{rank}. score={hit['score']:.4f}  {text_preview}{suffix}")


if __name__ == "__main__":
    main()
