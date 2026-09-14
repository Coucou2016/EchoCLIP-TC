"""Attention / ED–ES analysis for EchoCLIP-TA temporal modules.

Produces a SciencePlots-styled figure + CSV. Runs on synthetic demo tensors
by default; with EchoNet VolumeTracings + a trained temporal checkpoint, pass
``--manifest`` / ``--checkpoint`` for real analysis.

Clinical attention–EF claims remain 待补充 without EchoNet assets.
"""

from __future__ import annotations

import argparse
import csv
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
    """Softmax attention weights over time (query·frame / sqrt(D))."""
    if query is None:
        query = frames.mean(dim=0, keepdim=True)
    scores = (frames @ query.T).squeeze(-1) / (frames.shape[-1] ** 0.5)
    return F.softmax(scores, dim=0)


def attention_pool_module_weights(frames: torch.Tensor) -> torch.Tensor:
    """Weights from ``AttentionPool`` MultiheadAttention when available."""
    from echoclip.temporal import AttentionPool

    pool = AttentionPool(frames.shape[-1])
    x = frames.unsqueeze(0)  # (1, T, D)
    with torch.no_grad():
        q = pool.query.expand(x.size(0), -1, -1)
        kv = pool.ln(x)
        _, w = pool.attn(q, kv, kv, need_weights=True, average_attn_weights=True)
        # w: (B, 1, T) or (B, T)
        w = w.reshape(-1)
    return w


def ed_es_attention_summary(
    weights: torch.Tensor,
    ed_index: int,
    es_index: int,
) -> Dict[str, Any]:
    w = weights.detach().cpu().numpy().astype(np.float64)
    w = w / max(w.sum(), 1e-12)
    return {
        "n_frames": int(w.size),
        "ed_index": int(ed_index),
        "es_index": int(es_index),
        "weight_ed": float(w[ed_index]),
        "weight_es": float(w[es_index]),
        "weight_ed_es_sum": float(w[ed_index] + w[es_index]),
        "argmax_frame": int(np.argmax(w)),
        "entropy": float(-np.sum(w * np.log(np.clip(w, 1e-12, 1.0)))),
    }


def run_analysis(
    seed: int = 0,
    n_frames: int = 16,
    n_clips: int = 8,
    use_module: bool = True,
) -> Dict[str, Any]:
    rows: List[Dict[str, Any]] = []
    for i in range(n_clips):
        ed = 1 + (i % max(n_frames // 4, 1))
        es = min(n_frames - 2, ed + max(n_frames // 3, 2))
        frames, ed, es = toy_frame_embeddings(
            n_frames=n_frames, seed=seed + i, ed_index=ed, es_index=es
        )
        if use_module:
            try:
                weights = attention_pool_module_weights(frames)
            except Exception:
                weights = attention_pool_weights(frames)
        else:
            weights = attention_pool_weights(frames)
        summary = ed_es_attention_summary(weights, ed, es)
        summary["clip_id"] = i
        summary["seed"] = seed + i
        summary["weights"] = weights.detach().cpu().numpy().astype(float).tolist()
        rows.append(summary)

    ed_sum = float(np.mean([r["weight_ed"] for r in rows]))
    es_sum = float(np.mean([r["weight_es"] for r in rows]))
    return {
        "mode": "demo_toy",
        "demo_is_not_clinical": True,
        "n_clips": n_clips,
        "n_frames": n_frames,
        "mean_weight_ed": ed_sum,
        "mean_weight_es": es_sum,
        "mean_weight_ed_es_sum": float(np.mean([r["weight_ed_es_sum"] for r in rows])),
        "mean_entropy": float(np.mean([r["entropy"] for r in rows])),
        "rows": rows,
        "note": (
            "Synthetic embeddings only. Real ED/ES attention on EchoNet requires "
            "VolumeTracings + trained temporal module — clinical figure 待补充."
        ),
        "blocked_command": (
            "set ECHONET_ROOT=<AIMI_root> && "
            "python scripts/build_echonet_manifest.py --echonet-root %ECHONET_ROOT% && "
            "python scripts/analyze_attention_edes.py --manifest data/echonet_dynamic/test.json "
            "--checkpoint checkpoints/protocol/R5/best.pt --output-dir reports/attention_edes"
        ),
    }


def run_toy_analysis(seed: int = 0, n_frames: int = 16) -> Dict[str, Any]:
    """Backward-compatible single-clip toy summary (tests / smoke)."""
    result = run_analysis(seed=seed, n_frames=n_frames, n_clips=1, use_module=True)
    row = dict(result["rows"][0])
    row.pop("weights", None)
    row["mode"] = "toy"
    row["seed"] = seed
    row["note"] = result["note"]
    return row


def write_csv(result: Dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "clip_id",
        "seed",
        "n_frames",
        "ed_index",
        "es_index",
        "weight_ed",
        "weight_es",
        "weight_ed_es_sum",
        "argmax_frame",
        "entropy",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        for row in result["rows"]:
            w.writerow(row)


def write_figure(result: Dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise SystemExit(f"matplotlib required for figures: {exc}") from exc

    try:
        import scienceplots  # noqa: F401

        plt.style.use(["science", "no-latex"])
    except Exception:
        plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "ggplot")

    # Mean attention curve across clips (pad/trim to common T)
    T = int(result["n_frames"])
    stack = np.zeros((len(result["rows"]), T), dtype=np.float64)
    for i, row in enumerate(result["rows"]):
        w = np.asarray(row["weights"], dtype=np.float64)
        stack[i, : min(T, w.size)] = w[:T]
    mean_w = stack.mean(axis=0)
    std_w = stack.std(axis=0)

    fig, ax = plt.subplots(figsize=(5.2, 3.4))
    x = np.arange(T)
    ax.plot(x, mean_w, color="#1b4f72", lw=1.8, label="mean attention")
    ax.fill_between(x, mean_w - std_w, mean_w + std_w, color="#1b4f72", alpha=0.2)
    # Mark mean ED/ES indices
    ed_idxs = [r["ed_index"] for r in result["rows"]]
    es_idxs = [r["es_index"] for r in result["rows"]]
    ax.axvline(np.mean(ed_idxs), color="#c0392b", ls="--", lw=1.2, label="mean ED idx")
    ax.axvline(np.mean(es_idxs), color="#27ae60", ls="--", lw=1.2, label="mean ES idx")
    ax.set_xlabel("Frame index")
    ax.set_ylabel("Attention weight")
    title = "Attention vs ED/ES (DEMO toy)" if result.get("demo_is_not_clinical") else "Attention vs ED/ES"
    ax.set_title(title)
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    fig.savefig(path.with_suffix(".pdf"))
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(description="Attention / ED–ES analysis")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--frames", type=int, default=16)
    parser.add_argument("--n-clips", type=int, default=8)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "reports" / "attention_edes",
    )
    parser.add_argument("--output", type=Path, default=None, help="Optional JSON path")
    parser.add_argument(
        "--manifest",
        type=Path,
        default=None,
        help="EchoNet test manifest (if absent → demo toy path)",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help="Trained temporal checkpoint (real path 待补充 without weights)",
    )
    args = parser.parse_args()

    if args.manifest is not None and not args.manifest.exists():
        print(f"Manifest not found: {args.manifest}")
        print("Falling back to demo toy analysis.")
        args.manifest = None

    if args.manifest is not None and args.checkpoint is not None and args.checkpoint.exists():
        # Full real path not implemented without hub+AVI decode in this env —
        # fall through with clear blocked note after writing template.
        result = run_analysis(seed=args.seed, n_frames=args.frames, n_clips=args.n_clips)
        result["mode"] = "blocked_real_path"
        result["demo_is_not_clinical"] = True
        result["note"] = (
            "Checkpoint/manifest paths were provided, but end-to-end EchoNet frame "
            "decode + temporal attention export still requires local AVI + matching "
            "preprocess. Toy figure written; clinical attention CSV 待补充."
        )
        result["provided_manifest"] = str(args.manifest)
        result["provided_checkpoint"] = str(args.checkpoint)
    else:
        result = run_analysis(seed=args.seed, n_frames=args.frames, n_clips=args.n_clips)
        if args.manifest is None:
            result["status"] = "demo_complete"
        else:
            result["status"] = "blocked_missing_checkpoint"
            result["blocked_command"] = (
                "Train R5 then re-run:\n"
                "  python scripts/run_protocol.py --experiments R5 --paper\n"
                "  python scripts/analyze_attention_edes.py "
                f"--manifest {args.manifest} "
                "--checkpoint checkpoints/protocol/R5/best.pt"
            )

    out_dir = args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = args.output or (out_dir / "attention_edes_summary.json")
    csv_path = out_dir / "attention_edes.csv"
    fig_path = out_dir / "attention_edes.png"

    # Strip heavy weights from JSON rows copy for summary readability
    summary = {k: v for k, v in result.items() if k != "rows"}
    summary["rows"] = [
        {kk: vv for kk, vv in r.items() if kk != "weights"} for r in result["rows"]
    ]
    json_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    write_csv(result, csv_path)
    write_figure(result, fig_path)
    print(f"Wrote {json_path}")
    print(f"Wrote {csv_path}")
    print(f"Wrote {fig_path} and {fig_path.with_suffix('.pdf')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
