"""Attention / ED–ES analysis skeleton (toy tensors; no EchoNet required).

Runs on synthetic frame embeddings to exercise temporal attention weights
and ED/ES index alignment helpers. Real clinical analysis needs EchoNet
videos + VolumeTracings — marked 待补充.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def toy_frame_embeddings(
    n_frames: int = 16,
    dim: int = 32,
    seed: int = 0,
    ed_index: int = 2,
    es_index: int = 10,
) -> Tuple[torch.Tensor, int, int]:
    """Synthetic clip where ED/ES frames have distinct directions."""
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(n_frames, dim)).astype(np.float32)
    x[ed_index] += 3.0
    x[es_index] -= 3.0
    return torch.from_numpy(x), ed_index, es_index


def attention_pool_weights(
    frames: torch.Tensor,
    query: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Simple attention weights over time (softmax of query·frame)."""
    # frames: (T, D)
    if query is None:
        query = frames.mean(dim=0, keepdim=True)  # (1, D)
    scores = (frames @ query.T).squeeze(-1) / (frames.shape[-1] ** 0.5)
    return F.softmax(scores, dim=0)


def ed_es_attention_summary(
    weights: torch.Tensor,
    ed_index: int,
    es_index: int,
) -> Dict[str, Any]:
    w = weights.detach().cpu().numpy().astype(np.float64)
    return {
        "n_frames": int(w.size),
        "ed_index": int(ed_index),
        "es_index": int(es_index),
        "weight_ed": float(w[ed_index]),
        "weight_es": float(w[es_index]),
        "weight_ed_es_sum": float(w[ed_index] + w[es_index]),
        "argmax_frame": int(np.argmax(w)),
        "entropy": float(-np.sum(w * np.log(np.clip(w, 1e-12, 1.0)))),
        "note": (
            "Toy-tensor skeleton only. Real ED/ES attention analysis requires "
            "EchoNet VolumeTracings + a trained temporal module — 待补充."
        ),
    }


def run_toy_analysis(seed: int = 0, n_frames: int = 16) -> Dict[str, Any]:
    frames, ed, es = toy_frame_embeddings(n_frames=n_frames, seed=seed)
    # Prefer AttentionPool if available
    try:
        from echoclip.temporal import AttentionPool

        pool = AttentionPool(frames.shape[-1])
        with torch.no_grad():
            _ = pool(frames.unsqueeze(0))
            weights = attention_pool_weights(frames)
    except Exception:
        weights = attention_pool_weights(frames)
    summary = ed_es_attention_summary(weights, ed, es)
    summary["mode"] = "toy"
    summary["seed"] = seed
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description="Attention/ED-ES analysis (toy skeleton)")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--frames", type=int, default=16)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    out = run_toy_analysis(seed=args.seed, n_frames=args.frames)
    text = json.dumps(out, indent=2)
    print(text)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")
        print(f"Wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
