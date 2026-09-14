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
from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from echoclip.checkpoint import (
    is_supervised_ef_checkpoint,
    load_checkpoint,
    load_supervised_ef_checkpoint,
    predict_direct_ef,
)
from echoclip.clinical import parse_ef_from_text, summarize_clinical
from echoclip.data import EchoCLIPDataset, collate_batch, load_manifest, validate_manifest
from echoclip.model import EchoCLIP
from echoclip.protocol import (
    DEFAULT_EF_VALUES,
    OFFICIAL_EF_VALUES,
    assert_primary_eval_sampling,
    get_experiment,
    resolve_eval_sample_strategy,
)
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
    """Zero-shot EF via prompt pack. pool: frames | mean | temporal."""
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


@torch.no_grad()
def predict_ef_direct(
    backbone: EchoCLIP,
    head: nn.Module,
    loader: DataLoader,
    *,
    head_kind: str,
    device: str,
) -> np.ndarray:
    """Direct EF regression — does NOT call zero-shot prompt pack."""
    preds: List[torch.Tensor] = []
    for batch in tqdm(loader, desc="clinical-ef-direct"):
        images = batch["image"].to(device)
        batch_pred = predict_direct_ef(backbone, head, images, head_kind=head_kind)
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


def _infer_split_name(manifest: Path, cfg: dict) -> str:
    name = manifest.name.lower()
    for key in ("test", "val", "valid", "train"):
        if key in name:
            return "val" if key.startswith("val") else key
    # Heuristic from config paths
    for key, label in (
        ("test_manifest", "test"),
        ("cal_manifest", "val"),
        ("manifest", "train"),
    ):
        p = cfg.get(key)
        if p and Path(p).name == manifest.name:
            return label
    return "test"


def _run_split(
    engine: Optional[EchoCLIPInference],
    manifest: Path,
    manifest_dir: Path,
    cfg: dict,
    args,
    pool: str,
    *,
    sample_strategy: str,
    split: str = "test",
    prediction_mode: str = "zeroshot",
    backbone: Optional[EchoCLIP] = None,
    head: Optional[nn.Module] = None,
    head_kind: Optional[str] = None,
    device: str = "cpu",
) -> Tuple[np.ndarray, np.ndarray, dict]:
    assert_primary_eval_sampling(
        split=split,
        strategy=sample_strategy,
        experiment_id=getattr(args, "experiment_id", None),
        allow_annotation_assisted=bool(
            getattr(args, "allow_annotation_assisted", False)
        ),
    )
    image_size = (
        engine.model.config.image_size
        if engine is not None
        else backbone.config.image_size  # type: ignore[union-attr]
    )
    context_length = (
        engine.model.config.context_length
        if engine is not None
        else backbone.config.context_length  # type: ignore[union-attr]
    )
    # Official stride keeps variable T (do not force pad to 16).
    vf = args.video_frames or cfg.get("video_frames", 1)
    if str(sample_strategy).lower() in ("official_stride", "official", "echo_clip_stride"):
        # Upper bound for dataset; actual count comes from stride (no pad-to-16).
        vf = max(int(vf or 20), 20)
    ds = EchoCLIPDataset(
        manifest,
        manifest_dir=manifest_dir,
        image_size=image_size,
        context_length=context_length,
        tokenizer=EchoTokenizer(context_length=context_length),
        video_frames=vf,
        sample_strategy=sample_strategy,
        seed=args.seed,
    )
    ds.set_epoch(0)  # VAL/TEST: fixed epoch for reproducibility
    loader = DataLoader(
        ds,
        batch_size=min(args.batch_size, max(len(ds), 1)),
        shuffle=False,
        collate_fn=collate_batch,
    )
    y_true, source, n_missing = _collect_ef_labels(ds)
    if prediction_mode == "direct_regression":
        if backbone is None or head is None or head_kind is None:
            raise RuntimeError("direct_regression requires loaded backbone+head")
        y_pred = predict_ef_direct(
            backbone, head, loader, head_kind=head_kind, device=device
        )
    else:
        if engine is None:
            raise RuntimeError("zeroshot prediction requires EchoCLIPInference engine")
        y_pred = predict_ef(engine, loader, pool=pool)
    info = {
        "n": len(ds),
        "ef_source": source,
        "n_missing_ef": n_missing,
        "manifest": str(manifest),
        "pool": pool,
        "sample_strategy": sample_strategy,
        "split": split,
        "prediction_mode": prediction_mode,
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
        help="Optional protocol label (R0–R6 / B0/M1/… / ORACLE_EDES) in metrics.json",
    )
    parser.add_argument(
        "--paper",
        "--official-reproduction",
        dest="paper",
        action="store_true",
        help="Strict official reproduction: EF 0–100, no scratch fallback, prefer official stride",
    )
    parser.add_argument(
        "--allow-annotation-assisted",
        action="store_true",
        help="Permit ed_es/mixed on VAL/TEST (Oracle-EDES only)",
    )
    parser.add_argument(
        "--calibration-method",
        choices=["temperature", "affine_logistic"],
        default="temperature",
        help="VAL-fit calibration for P(EF<50/40/30); affine_logistic preferred over pseudo-logit T",
    )
    parser.add_argument(
        "--adaptive-conformal",
        action="store_true",
        help="Also report normalized residual conformal + risk-coverage / AURC",
    )
    parser.add_argument(
        "--adaptive-scale",
        choices=["heuristic", "head"],
        default="heuristic",
        help="Scale s(x) for adaptive conformal (heuristic or learned PositiveScaleHead)",
    )
    parser.add_argument(
        "--prediction-mode",
        choices=["zeroshot", "direct_regression", "auto"],
        default="auto",
        help="zeroshot=EF prompt pack; direct_regression=supervised head (R2–R4); "
        "auto=infer from experiment / checkpoint",
    )
    parser.add_argument(
        "--split",
        type=str,
        default=None,
        help="Split name for sampling guards (test|val|train). Inferred from manifest if omitted.",
    )
    args = parser.parse_args()

    set_seed(args.seed)
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    cfg = {}
    if args.config.exists():
        cfg = yaml.safe_load(args.config.read_text(encoding="utf-8")) or {}

    paper = bool(args.paper)
    if paper:
        # Paper path must not silently use random towers.
        import os

        if os.environ.get("ECHOCLIP_SKIP_HUB", "").strip() in ("1", "true", "yes"):
            print(
                "Error: --paper/--official-reproduction cannot run with ECHOCLIP_SKIP_HUB=1. "
                "Unset it and provide hub access or --official-checkpoint."
            )
            return 1

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

    # Resolve experiment + eval sampling (honor val_sample_strategy; force uniform primary)
    spec = None
    if args.experiment_id:
        try:
            spec = get_experiment(args.experiment_id)
            args.experiment_id = spec.id
            if spec.annotation_assisted:
                args.allow_annotation_assisted = True
        except KeyError as exc:
            print(exc)
            return 1

    split_name = args.split or _infer_split_name(manifest, cfg)
    if spec is not None:
        sample_strategy = resolve_eval_sample_strategy(
            spec=spec,
            cli_strategy=args.sample_strategy,
            cfg=cfg,
            split=split_name,
            paper=paper,
        )
    else:
        sample_strategy = args.sample_strategy or cfg.get(
            "val_sample_strategy", cfg.get("sample_strategy", "uniform")
        )
        assert_primary_eval_sampling(
            split=split_name,
            strategy=sample_strategy,
            experiment_id=args.experiment_id,
            allow_annotation_assisted=args.allow_annotation_assisted,
        )
        # Paper + no experiment: prefer official stride when CLI left unset
        if paper and args.sample_strategy is None and sample_strategy == "uniform":
            sample_strategy = "official_stride"

    # Resolve prediction mode
    prediction_mode = args.prediction_mode
    if prediction_mode == "auto":
        if spec is not None and getattr(spec, "prediction_mode", None):
            prediction_mode = spec.prediction_mode
        elif spec is not None and spec.pool == "supervised":
            prediction_mode = "direct_regression"
        else:
            prediction_mode = "zeroshot"

    engine: Optional[EchoCLIPInference] = None
    model: Optional[EchoCLIP] = None
    head: Optional[nn.Module] = None
    head_kind: Optional[str] = None
    ckpt: dict = {}
    ckpt_epoch = None
    load_source = "unknown"
    train_seed = None

    if args.checkpoint and args.checkpoint.exists():
        try:
            raw = torch.load(args.checkpoint, map_location=device, weights_only=False)
        except TypeError:
            raw = torch.load(args.checkpoint, map_location=device)
        if prediction_mode == "direct_regression" or is_supervised_ef_checkpoint(raw):
            prediction_mode = "direct_regression"
            model, head, ckpt = load_supervised_ef_checkpoint(
                args.checkpoint,
                device=device,
                allow_scratch_fallback=not paper,
            )
            head_kind = str(ckpt.get("head_kind") or "linear")
            ckpt_epoch = ckpt.get("epoch", "?")
            load_source = (
                ckpt.get("backbone_load_source")
                or getattr(model, "load_source", "supervised_checkpoint")
            )
            train_seed = ckpt.get("train_seed")
            if paper and str(load_source).startswith("scratch"):
                print(
                    "Error: --paper requires real EchoCLIP weights; checkpoint load_source="
                    f"{load_source}"
                )
                return 1
        else:
            model, ckpt = load_checkpoint(args.checkpoint, device=device)
            ckpt_epoch = ckpt.get("epoch", "?")
            load_source = getattr(model, "load_source", "checkpoint")
            train_seed = ckpt.get("train_seed") or (ckpt.get("config") or {}).get("seed")
            if paper and str(load_source).startswith("scratch"):
                print(
                    "Error: --paper requires real EchoCLIP weights; checkpoint load_source="
                    f"{load_source}"
                )
                return 1
    elif args.init_official:
        from echoclip.config import EchoCLIPConfig
        from echoclip.utils import config_from_dict

        if prediction_mode == "direct_regression":
            print(
                "Error: --prediction-mode direct_regression requires --checkpoint "
                "with a supervised EF head (R2–R4)."
            )
            return 1
        model_cfg = config_from_dict(cfg) if cfg else EchoCLIPConfig()
        model_cfg.pretrained_vision = False
        if model_module_needs_simple_cnn() and not paper:
            model_cfg.vision_backbone = "simple_cnn"
        try:
            model = EchoCLIP.from_official_echo_clip(
                model_cfg,
                checkpoint_path=str(args.official_checkpoint)
                if args.official_checkpoint
                else None,
                allow_scratch_fallback=not paper,
            )
        except RuntimeError as exc:
            print(f"Error: {exc}")
            return 1
        model.to(device)
        ckpt_epoch = None
        load_source = model.load_source
        if paper and (
            str(load_source).startswith("scratch")
            or load_source == "scratch_fallback"
        ):
            print(
                "Error: --paper/--official-reproduction refused scratch_fallback. "
                "Provide hub weights or --official-checkpoint."
            )
            return 1
    else:
        print("Provide --checkpoint PATH or --init-official")
        return 1

    ef_values = list(OFFICIAL_EF_VALUES) if paper else list(DEFAULT_EF_VALUES)
    pool = "mean"
    if prediction_mode == "zeroshot":
        engine = EchoCLIPInference(
            model,
            device=device,
            official_reproduction=paper,
            ef_values=ef_values,
        )
        pool = resolve_pool(args.pool, model)
        if pool == "temporal" and getattr(model, "temporal", None) is None:
            print(
                "Error: --pool temporal requires an attached temporal aggregator, "
                "but this model has temporal=None (would silently mean-pool).\n"
                "Train R5/M2 first, load a TC checkpoint, or use --pool mean|frames."
            )
            return 1
    else:
        pool = "supervised_direct"

    y_true, y_pred, info = _run_split(
        engine,
        manifest,
        manifest_dir,
        cfg,
        args,
        pool,
        sample_strategy=sample_strategy,
        split=split_name,
        prediction_mode=prediction_mode,
        backbone=model,
        head=head,
        head_kind=head_kind,
        device=device,
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
            engine,
            args.cal_manifest,
            cal_dir,
            cfg,
            args,
            pool,
            sample_strategy=sample_strategy,
            split="val",
            prediction_mode=prediction_mode,
            backbone=model,
            head=head,
            head_kind=head_kind,
            device=device,
        )
        cmask = np.isfinite(cy) & np.isfinite(cp)
        cal_true, cal_pred = cy[cmask], cp[cmask]

    # official_reproduction_verified: only true after parity evidence; paper alone ≠ verified
    official_verified = False
    if paper and str(sample_strategy).lower() in (
        "official_stride",
        "official",
        "echo_clip_stride",
    ):
        # Verified only when caller/env marks parity (compare_official_b0 / eval_official_r0).
        import os

        official_verified = os.environ.get("ECHOCLIP_OFFICIAL_PARITY_OK", "").strip() in (
            "1",
            "true",
            "yes",
        )

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
        "prediction_mode": prediction_mode,
        "head_kind": head_kind,
        "use_temporal": pool == "temporal",
        "video_frames": args.video_frames or cfg.get("video_frames", 1),
        "sample_strategy": sample_strategy,
        "seed": args.seed,
        "eval_seed": args.seed,
        "train_seed": train_seed,
        "note": protocol_note,
        "paper_primary": True,
        "demo_is_not_clinical": info["ef_source"] != "manifest",
        "official_reproduction": paper,
        "paper_mode": paper,
        "official_reproduction_verified": official_verified,
        "ef_grid": "0_100_step1" if paper else "15_80_step5",
        "ef_grid_n": len(ef_values),
        "baseline_name": (
            "Official EchoCLIP zero-shot (reproduction path)"
            if paper and prediction_mode == "zeroshot"
            else (
                "Supervised EF regression (direct head)"
                if prediction_mode == "direct_regression"
                else "EchoCLIP-based zero-shot baseline"
            )
        ),
    }
    if args.experiment_id:
        metrics["experiment_id"] = str(args.experiment_id).upper()
        if spec is not None and spec.annotation_assisted:
            metrics["annotation_assisted"] = True
    if n_eval >= 2:
        # Paper default calibration: affine_logistic when calibrate path used
        cal_method = args.calibration_method
        if paper and cal_method == "temperature" and args.cal_manifest:
            # Prefer affine_logistic for paper unless user overrode via CLI default —
            # keep CLI explicit; protocol passes affine_logistic under --paper.
            cal_method = args.calibration_method
        clinical = summarize_clinical(
            y_true[mask],
            y_pred[mask],
            cal_true=cal_true,
            cal_pred=cal_pred,
            seed=args.seed,
            calibration_method=cal_method,
            adaptive_conformal=bool(args.adaptive_conformal),
            adaptive_scale=args.adaptive_scale,
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
