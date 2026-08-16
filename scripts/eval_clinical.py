"""Clinical evaluation for EchoCLIP-TC: EF MAE/RMSE/R², threshold AUCs, calibration.

This — not scripts/eval.py retrieval R@k — is the paper primary metric script.
Demo manifests are accepted for pipeline smoke tests; metrics.json will record
that labels were parsed from synthetic text and are not clinical results.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from echoclip.checkpoint import load_checkpoint
from echoclip.clinical import parse_ef_from_text, summarize_clinical
from echoclip.data import EchoCLIPDataset, collate_batch, load_manifest, validate_manifest
from echoclip.model import EchoCLIP
from echoclip.text import EchoTokenizer
from echoclip.utils import set_seed
from echoclip.zeroshot import EchoCLIPInference


def _collect_ef_labels(ds: EchoCLIPDataset) -> Tuple[np.ndarray, str, int]:
    values: List[float] = []
    missing = 0
    n_manifest = 0
    n_parsed = 0
    for item in ds.pairs:
        if item.get("ef") is not None and str(item.get("ef")) != "":
            values.append(float(item["ef"]))
            n_manifest += 1
            continue
        parsed = parse_ef_from_text(str(item.get("text", "")))
        if parsed is None:
            values.append(float("nan"))
            missing += 1
        else:
            values.append(parsed)
            n_parsed += 1
    if n_manifest > 0 and n_parsed == 0:
        source = "manifest"
    elif n_parsed > 0 and n_manifest == 0:
        source = "text_parse_demo_only"
    elif n_parsed > 0 and n_manifest > 0:
        source = "mixed_manifest_and_text_parse"
    else:
        source = "missing"
    return np.asarray(values, dtype=np.float64), source, missing


@torch.no_grad()
def predict_ef(
    engine: EchoCLIPInference,
    loader: DataLoader,
    pool: str,
) -> np.ndarray:
    """pool: frames (official per-frame) | mean | temporal."""
    from echoclip.temporal import pool_frame_features

    preds: List[torch.Tensor] = []
    prompt_values, prompt_emb = engine._ef_prompt_pack()
    for batch in tqdm(loader, desc="clinical-ef"):
        images = batch["image"].to(engine.device)
        if images.dim() == 5:
            if pool == "frames":
                feats = engine.model.encode_frame_features(images)
                feats = torch.nn.functional.normalize(feats, dim=-1)
                batch_pred = engine.zero_shot_ef_batch(
                    feats, prompt_embeddings=prompt_emb, prompt_values=prompt_values
                )
            elif pool == "mean":
                feats = engine.model.encode_frame_features(images)
                video_z = pool_frame_features(feats, aggregator=None, normalize=True)
                batch_pred = engine.zero_shot_ef_batch(
                    video_z, prompt_embeddings=prompt_emb, prompt_values=prompt_values
                )
            else:
                # temporal / auto: use attached aggregator (mean if none)
                video_z = engine.model.encode_video(images)
                batch_pred = engine.zero_shot_ef_batch(
                    video_z, prompt_embeddings=prompt_emb, prompt_values=prompt_values
                )
        else:
            feats = engine.model.encode_image(images)
            batch_pred = engine.zero_shot_ef_batch(
                feats, prompt_embeddings=prompt_emb, prompt_values=prompt_values
            )
        preds.append(batch_pred.detach().cpu())
    if not preds:
        return np.zeros((0,), dtype=np.float64)
    return torch.cat(preds).numpy().reshape(-1)


def resolve_pool(args_pool: str, model: EchoCLIP) -> str:
    """Map CLI pool flag to frames | mean | temporal."""
    if args_pool == "auto":
        return "temporal" if getattr(model, "temporal", None) is not None else "frames"
    if args_pool == "frames":
        return "frames"
    if args_pool == "mean":
        return "mean"
    if args_pool == "temporal":
        return "temporal"
    raise ValueError(f"Unknown --pool {args_pool!r}")


def _run_split(
    engine: EchoCLIPInference,
    manifest: Path,
    manifest_dir: Path,
    cfg: dict,
    args,
    pool: str,
) -> Tuple[np.ndarray, np.ndarray, dict]:
    ds = EchoCLIPDataset(
        manifest,
        manifest_dir=manifest_dir,
        image_size=engine.model.config.image_size,
        context_length=engine.model.config.context_length,
        tokenizer=EchoTokenizer(context_length=engine.model.config.context_length),
        video_frames=args.video_frames or cfg.get("video_frames", 1),
        sample_strategy=args.sample_strategy
        or cfg.get("val_sample_strategy", cfg.get("sample_strategy", "uniform")),
        seed=args.seed,
    )
    loader = DataLoader(
        ds,
        batch_size=min(args.batch_size, max(len(ds), 1)),
        shuffle=False,
        collate_fn=collate_batch,
    )
    y_true, source, n_missing = _collect_ef_labels(ds)
    y_pred = predict_ef(engine, loader, pool=pool)
    info = {
        "n": len(ds),
        "ef_source": source,
        "n_missing_ef": n_missing,
        "manifest": str(manifest),
        "pool": pool,
    }
    return y_true, y_pred, info


def main() -> int:
    parser = argparse.ArgumentParser(description="EchoCLIP-TC clinical evaluation")
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--manifest", type=Path, default=None)
    parser.add_argument("--manifest-dir", type=Path, default=None)
    parser.add_argument("--cal-manifest", type=Path, default=None, help="Validation split for T / conformal")
    parser.add_argument("--config", type=Path, default=ROOT / "configs" / "echonet_dynamic.yaml")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--video-frames", type=int, default=None)
    parser.add_argument("--sample-strategy", type=str, default=None)
    parser.add_argument(
        "--pool",
        choices=["auto", "temporal", "frames", "mean"],
        default="auto",
        help="frames=official per-frame EF; mean=mean-pool then EF; temporal=aggregator",
    )
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--init-official", action="store_true",
                        help="Evaluate official hub/local weights without a TC checkpoint")
    parser.add_argument("--official-checkpoint", type=Path, default=None)
    parser.add_argument(
        "--experiment-id",
        type=str,
        default=None,
        help="Optional protocol label (B0/M1/M2/M4) written into metrics.json",
    )
    args = parser.parse_args()

    set_seed(args.seed)
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    cfg = {}
    if args.config.exists():
        cfg = yaml.safe_load(args.config.read_text(encoding="utf-8")) or {}

    manifest = args.manifest or Path(
        cfg.get("test_manifest") or cfg.get("manifest") or ROOT / "data" / "demo" / "manifest.json"
    )
    if args.manifest_dir:
        manifest_dir = args.manifest_dir
    elif args.manifest:
        manifest_dir = args.manifest.parent
    else:
        manifest_dir = Path(cfg.get("manifest_dir", manifest.parent))
    if not manifest.exists():
        print(f"Manifest not found: {manifest}")
        print("For EchoNet: python scripts/build_echonet_manifest.py --echonet-root <root>")
        print("For pipeline smoke: python scripts/make_demo_data.py")
        return 1

    pairs = load_manifest(manifest)
    errors = validate_manifest(pairs, manifest_dir)
    if errors:
        print("Manifest errors:")
        for err in errors[:15]:
            print(f"  - {err}")
        return 1

    if args.checkpoint and args.checkpoint.exists():
        model, ckpt = load_checkpoint(args.checkpoint, device=device)
        ckpt_epoch = ckpt.get("epoch", "?")
        load_source = getattr(model, "load_source", "checkpoint")
    elif args.init_official:
        from echoclip.config import EchoCLIPConfig
        from echoclip.utils import config_from_dict

        model_cfg = config_from_dict(cfg) if cfg else EchoCLIPConfig()
        model_cfg.pretrained_vision = False
        if model_module_needs_simple_cnn():
            model_cfg.vision_backbone = "simple_cnn"
        model = EchoCLIP.from_official_echo_clip(
            model_cfg, checkpoint_path=str(args.official_checkpoint) if args.official_checkpoint else None
        )
        model.to(device)
        ckpt_epoch = None
        load_source = model.load_source
    else:
        print("Provide --checkpoint PATH or --init-official")
        return 1

    engine = EchoCLIPInference(model, device=device)
    pool = resolve_pool(args.pool, model)
    if pool == "temporal" and getattr(model, "temporal", None) is None:
        print(
            "Error: --pool temporal requires an attached temporal aggregator, "
            "but this model has temporal=None (would silently mean-pool).\n"
            "Train M2 first, load a TC checkpoint, or use --pool mean|frames."
        )
        return 1

    y_true, y_pred, info = _run_split(
        engine, manifest, manifest_dir, cfg, args, pool
    )
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    n_eval = int(mask.sum())
    protocol_note = (
        "Labels parsed from report text on demo data are NOT clinical ground truth. "
        "Do not report these numbers as EchoNet / paper EF MAE."
        if info["ef_source"] in ("text_parse_demo_only", "mixed_manifest_and_text_parse", "missing")
        else "EF labels taken from the manifest (EchoNet FileList or equivalent)."
    )

    cal_true = cal_pred = None
    cal_info = None
    if args.cal_manifest and args.cal_manifest.exists():
        cal_dir = args.manifest_dir or args.cal_manifest.parent
        cy, cp, cal_info = _run_split(
            engine, args.cal_manifest, cal_dir, cfg, args, pool
        )
        cmask = np.isfinite(cy) & np.isfinite(cp)
        cal_true, cal_pred = cy[cmask], cp[cmask]

    metrics = {
        "task": "clinical_ef",
        "n_eval": n_eval,
        "n_manifest": info["n"],
        "ef_source": info["ef_source"],
        "n_missing_ef": info["n_missing_ef"],
        "checkpoint": str(args.checkpoint) if args.checkpoint else None,
        "checkpoint_epoch": ckpt_epoch,
        "load_source": load_source,
        "pool": pool,
        "use_temporal": pool == "temporal",
        "video_frames": args.video_frames or cfg.get("video_frames", 1),
        "sample_strategy": args.sample_strategy
        or cfg.get("val_sample_strategy", cfg.get("sample_strategy", "uniform")),
        "seed": args.seed,
        "note": protocol_note,
        "paper_primary": True,
        "demo_is_not_clinical": info["ef_source"] != "manifest",
    }
    if args.experiment_id:
        metrics["experiment_id"] = str(args.experiment_id).upper()
    if n_eval >= 2:
        clinical = summarize_clinical(
            y_true[mask],
            y_pred[mask],
            cal_true=cal_true,
            cal_pred=cal_pred,
            seed=args.seed,
        )
        metrics.update(clinical)
    else:
        metrics["error"] = "Need at least 2 samples with EF labels"

    if cal_info:
        metrics["calibration_manifest"] = cal_info["manifest"]
        metrics["n_calibration_raw"] = cal_info["n"]

    metrics = _json_safe(metrics)
    print(json.dumps(metrics, indent=2, allow_nan=False))
    out = args.output
    if out is None:
        out = Path(cfg.get("output_dir", ROOT / "checkpoints")) / "clinical_metrics.json"
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(metrics, indent=2, allow_nan=False), encoding="utf-8")
    print(f"Wrote {out}")
    return 0


def _json_safe(value):
    if isinstance(value, dict):
        return {k: _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, (np.floating, float)):
        x = float(value)
        if not np.isfinite(x):
            return None
        return x
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, np.ndarray):
        return _json_safe(value.tolist())
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    return value


def model_module_needs_simple_cnn() -> bool:
    from echoclip import model as model_module

    return model_module.timm is None


if __name__ == "__main__":
    raise SystemExit(main())
