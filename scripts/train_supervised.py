"""Train supervised EF heads (S0/S1/S2 / R2–R4) on frozen EchoCLIP features.

Works with toy/demo tensors when EchoNet or hub weights are absent
(``--no-official`` + ``simple_cnn``). Not a clinical result path by itself.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from echoclip.checkpoint import save_checkpoint
from echoclip.config import EchoCLIPConfig
from echoclip.data import EchoCLIPDataset, collate_batch, load_manifest, split_manifest, validate_manifest
from echoclip.model import EchoCLIP
from echoclip.supervised import (
    LinearEFHead,
    MLPEFHead,
    TemporalEFRegressor,
    ef_regression_loss,
    extract_mean_pooled_features,
    fit_ridge_ef,
)
from echoclip.temporal import build_temporal
from echoclip.text import EchoTokenizer
from echoclip.utils import set_seed
from echoclip import model as model_module


def load_config(path: Path) -> dict:
    with path.open(encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def build_backbone(cfg: dict) -> EchoCLIP:
    backbone = cfg.get("vision_backbone", "simple_cnn")
    pretrained = cfg.get("pretrained_vision", False)
    if model_module.timm is None and backbone != "simple_cnn":
        backbone = "simple_cnn"
        pretrained = False
        cfg["init_official_echo_clip"] = False
    model_cfg = EchoCLIPConfig(
        embed_dim=cfg.get("embed_dim", 512),
        image_size=cfg.get("image_size", 224),
        context_length=cfg.get("context_length", 77),
        vision_backbone=backbone,
        text_layers=cfg.get("text_layers", 2),
        text_heads=cfg.get("text_heads", 4),
        text_width=cfg.get("text_width", 512),
        pretrained_vision=pretrained,
        temporal_type="none",
    )
    if cfg.get("init_official_echo_clip"):
        model = EchoCLIP.from_official_echo_clip(
            model_cfg,
            checkpoint_path=cfg.get("official_checkpoint"),
            allow_scratch_fallback=True,
        )
    else:
        model = EchoCLIP(model_cfg)
    for p in model.parameters():
        p.requires_grad = False
    model.eval()
    return model


@torch.no_grad()
def collect_features(model, loader, device, use_temporal_feats: bool = False):
    zs, ys = [], []
    for batch in tqdm(loader, desc="features", leave=False):
        images = batch["image"].to(device)
        ef = batch.get("ef")
        if ef is None:
            continue
        if images.dim() == 5:
            if use_temporal_feats:
                feats = model.encode_frame_features(images)
                zs.append(feats.cpu())
            else:
                z = extract_mean_pooled_features(model, images)
                zs.append(z.cpu())
        else:
            z = model.encode_image(images)
            zs.append(z.cpu())
        ys.append(ef.cpu())
    if not zs:
        raise RuntimeError("No EF-labeled batches found for supervised training")
    if use_temporal_feats:
        return torch.cat(zs, dim=0), torch.cat(ys, dim=0)
    return torch.cat(zs, dim=0), torch.cat(ys, dim=0)


def main() -> int:
    parser = argparse.ArgumentParser(description="Supervised EF head training (R2/R3/R4)")
    parser.add_argument("--config", type=Path, default=ROOT / "configs" / "echonet_dynamic.yaml")
    parser.add_argument("--manifest", type=Path, default=None)
    parser.add_argument("--manifest-dir", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--head", type=str, default="linear", help="linear|mlp|temporal_l1")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--video-frames", type=int, default=None)
    parser.add_argument("--vision-backbone", type=str, default=None)
    parser.add_argument("--no-official", action="store_true")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--ridge-alpha", type=float, default=1.0)
    parser.add_argument(
        "--demo",
        action="store_true",
        help="Force demo wiring: no official init, prefer simple_cnn (NOT clinical).",
    )
    parser.add_argument(
        "--paper",
        action="store_true",
        dest="paper",
        help="Official reproduction path: hard-fail without real EchoCLIP weights.",
    )
    parser.add_argument(
        "--official-reproduction",
        action="store_true",
        dest="paper",
        help=argparse.SUPPRESS,
    )
    args = parser.parse_args()

    if args.demo and args.paper:
        print("Error: --paper and --demo are mutually exclusive.")
        return 1

    cfg = load_config(args.config)
    if args.manifest:
        cfg["manifest"] = str(args.manifest)
    if args.manifest_dir:
        cfg["manifest_dir"] = str(args.manifest_dir)
    if args.output_dir:
        cfg["output_dir"] = str(args.output_dir)
    if args.epochs is not None:
        cfg["epochs"] = args.epochs
    if args.batch_size is not None:
        cfg["batch_size"] = args.batch_size
    if args.video_frames is not None:
        cfg["video_frames"] = args.video_frames
    if args.vision_backbone:
        cfg["vision_backbone"] = args.vision_backbone
    if args.demo:
        cfg["init_official_echo_clip"] = False
        cfg.setdefault("vision_backbone", "simple_cnn")
        if not args.vision_backbone:
            cfg["vision_backbone"] = "simple_cnn"
    if args.paper:
        if cfg.get("vision_backbone") == "simple_cnn" or args.no_official:
            print(
                "Error: --paper cannot combine with --no-official / simple_cnn. "
                "Need official EchoCLIP weights."
            )
            return 1
        cfg["init_official_echo_clip"] = True
    if args.no_official or cfg.get("vision_backbone") == "simple_cnn":
        cfg["init_official_echo_clip"] = False

    set_seed(cfg.get("seed", 42))
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    manifest = Path(cfg.get("manifest") or ROOT / "data" / "demo" / "manifest.json")
    if args.demo and not args.manifest:
        manifest = ROOT / "data" / "demo" / "manifest.json"
        cfg["manifest"] = str(manifest)
    if not manifest.exists():
        print(f"Manifest not found: {manifest}")
        return 1
    pairs = load_manifest(manifest)
    manifest_dir = Path(cfg.get("manifest_dir", manifest.parent))
    errs = validate_manifest(pairs, manifest_dir)
    if errs:
        for e in errs[:10]:
            print(e)
        return 1

    train_pairs, val_pairs = split_manifest(pairs, cfg.get("val_ratio", 0.1), cfg.get("seed", 42))
    out_dir = Path(cfg.get("output_dir", ROOT / "checkpoints" / "supervised"))
    out_dir.mkdir(parents=True, exist_ok=True)
    split_dir = out_dir / "splits"
    split_dir.mkdir(exist_ok=True)
    train_m = split_dir / "train_split.json"
    val_m = split_dir / "val_split.json"
    train_m.write_text(json.dumps({"pairs": train_pairs}, indent=2), encoding="utf-8")
    val_m.write_text(json.dumps({"pairs": val_pairs}, indent=2), encoding="utf-8")

    tokenizer = EchoTokenizer(context_length=cfg.get("context_length", 77))
    ds_kwargs = dict(
        manifest_dir=manifest_dir,
        image_size=cfg.get("image_size", 224),
        context_length=cfg.get("context_length", 77),
        video_frames=cfg.get("video_frames", 16),
        tokenizer=tokenizer,
        sample_strategy=cfg.get("sample_strategy", "mixed"),
        seed=cfg.get("seed", 42),
    )
    train_ds = EchoCLIPDataset(train_m, **ds_kwargs)
    val_ds = EchoCLIPDataset(
        val_m,
        **{**ds_kwargs, "sample_strategy": cfg.get("val_sample_strategy", "uniform")},
    )
    val_ds.set_epoch(0)
    batch_size = min(cfg.get("batch_size", 8), max(len(train_ds), 1))
    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, collate_fn=collate_batch
    )
    val_loader = DataLoader(
        val_ds, batch_size=min(batch_size, max(len(val_ds), 1)), shuffle=False, collate_fn=collate_batch
    )

    backbone = build_backbone(cfg).to(device)
    if args.paper:
        src = str(getattr(backbone, "load_source", "") or "")
        if src.startswith("scratch") or "simple_cnn" in src.lower():
            print(
                "Error: --paper requires real EchoCLIP weights; got load_source="
                f"{src!r}"
            )
            return 1
    dim = backbone.config.embed_dim
    head_kind = str(args.head).lower()
    epochs = int(cfg.get("epochs", 5))
    if args.demo:
        print(
            "DEMO MODE — supervised head training on synthetic/demo data only; "
            "not EchoNet / paper EF MAE."
        )

    # Bundle: backbone config + head weights saved together for eval hooks.
    if head_kind in ("linear", "ridge", "s0"):
        train_ds.set_epoch(0)
        z_tr, y_tr = collect_features(backbone, train_loader, device, use_temporal_feats=False)
        ridge = fit_ridge_ef(z_tr, y_tr, alpha=args.ridge_alpha)
        head = LinearEFHead(dim)
        with torch.no_grad():
            head.fc.weight.copy_(ridge.weight.view(1, -1))
            head.fc.bias.copy_(torch.tensor([ridge.bias]))
        # Optional fine-tune a few SGD steps (API completeness)
        opt = torch.optim.SGD(head.parameters(), lr=args.lr)
        head.to(device)
        for epoch in range(1, min(epochs, 3) + 1):
            train_ds.set_epoch(epoch)
            head.train()
            total = 0.0
            n = 0
            for batch in train_loader:
                images = batch["image"].to(device)
                y = batch["ef"].to(device)
                with torch.no_grad():
                    z = extract_mean_pooled_features(backbone, images) if images.dim() == 5 else backbone.encode_image(images)
                loss = ef_regression_loss(head(z), y, kind="l1")
                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()
                total += loss.item()
                n += 1
            print(f"epoch {epoch}: train_l1={total / max(n, 1):.4f}")
        payload = {"head_kind": "linear", "ridge_alpha": args.ridge_alpha}
        torch.save(
            {
                "epoch": epochs,
                "head_state_dict": head.state_dict(),
                "head_kind": "linear",
                "embed_dim": dim,
                "train_cfg": cfg,
                "meta": payload,
                "model_config": backbone.config.__dict__,
            },
            out_dir / "best.pt",
        )
        # Also attach a thin EchoCLIP wrapper for load_checkpoint compatibility when possible
        print(f"Saved linear/ridge head → {out_dir / 'best.pt'}")
        return 0

    if head_kind in ("mlp", "s1"):
        head = MLPEFHead(dim, hidden=cfg.get("ef_head_hidden", 256)).to(device)
        opt = torch.optim.Adam(head.parameters(), lr=args.lr)
        best = float("inf")
        for epoch in range(1, epochs + 1):
            train_ds.set_epoch(epoch)
            head.train()
            total = 0.0
            n = 0
            for batch in train_loader:
                images = batch["image"].to(device)
                y = batch["ef"].to(device)
                with torch.no_grad():
                    z = (
                        extract_mean_pooled_features(backbone, images)
                        if images.dim() == 5
                        else backbone.encode_image(images)
                    )
                loss = ef_regression_loss(head(z), y, kind="huber")
                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()
                total += loss.item()
                n += 1
            # Val
            head.eval()
            vtotal = 0.0
            vn = 0
            with torch.no_grad():
                for batch in val_loader:
                    images = batch["image"].to(device)
                    y = batch["ef"].to(device)
                    z = (
                        extract_mean_pooled_features(backbone, images)
                        if images.dim() == 5
                        else backbone.encode_image(images)
                    )
                    vtotal += ef_regression_loss(head(z), y, kind="l1").item()
                    vn += 1
            val_l1 = vtotal / max(vn, 1)
            print(f"epoch {epoch}: train={total / max(n, 1):.4f} val_l1={val_l1:.4f}")
            if val_l1 < best:
                best = val_l1
                torch.save(
                    {
                        "epoch": epoch,
                        "head_state_dict": head.state_dict(),
                        "head_kind": "mlp",
                        "embed_dim": dim,
                        "train_cfg": cfg,
                        "model_config": backbone.config.__dict__,
                    },
                    out_dir / "best.pt",
                )
        print(f"Saved MLP head → {out_dir / 'best.pt'}")
        return 0

    if head_kind in ("temporal_l1", "temporal_huber", "s2"):
        agg = build_temporal(
            "transformer",
            dim,
            n_layers=cfg.get("temporal_layers", 2),
            n_heads=cfg.get("temporal_heads", 8),
            max_frames=max(cfg.get("temporal_max_frames", 64), cfg.get("video_frames", 16)),
        )
        assert agg is not None
        reg = TemporalEFRegressor(agg, dim).to(device)
        opt = torch.optim.Adam(reg.parameters(), lr=args.lr)
        best = float("inf")
        for epoch in range(1, epochs + 1):
            train_ds.set_epoch(epoch)
            reg.train()
            total = 0.0
            n = 0
            for batch in train_loader:
                images = batch["image"].to(device)
                y = batch["ef"].to(device)
                with torch.no_grad():
                    feats = backbone.encode_frame_features(images)
                pred = reg(feats)
                loss = ef_regression_loss(pred, y, kind="huber")
                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()
                total += loss.item()
                n += 1
            reg.eval()
            vtotal = 0.0
            vn = 0
            with torch.no_grad():
                for batch in val_loader:
                    images = batch["image"].to(device)
                    y = batch["ef"].to(device)
                    feats = backbone.encode_frame_features(images)
                    vtotal += ef_regression_loss(reg(feats), y, kind="l1").item()
                    vn += 1
            val_l1 = vtotal / max(vn, 1)
            print(f"epoch {epoch}: train={total / max(n, 1):.4f} val_l1={val_l1:.4f}")
            if val_l1 < best:
                best = val_l1
                # Attach temporal to a backbone clone for zero-shot-style eval fallback
                backbone.attach_temporal(
                    "transformer",
                    n_layers=cfg.get("temporal_layers", 2),
                    n_heads=cfg.get("temporal_heads", 8),
                    max_frames=max(cfg.get("temporal_max_frames", 64), cfg.get("video_frames", 16)),
                )
                backbone.temporal.load_state_dict(reg.aggregator.state_dict())
                save_checkpoint(
                    out_dir / "best.pt",
                    backbone,
                    epoch,
                    train_cfg={**cfg, "supervised_head": "temporal_l1", "head_state": reg.head.state_dict()},
                )
        print(f"Saved temporal L1 model → {out_dir / 'best.pt'}")
        return 0

    print(f"Unknown head {args.head!r}")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
