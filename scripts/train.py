"""Train EchoCLIP with CLIP contrastive loss on image-report pairs."""

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
from echoclip.data import EchoCLIPDataset, collate_batch, split_manifest, load_manifest, validate_manifest
from echoclip.loss import ClipLoss, TemporalClipLoss
from echoclip.text import EchoTokenizer
from echoclip import model as model_module
from echoclip.model import EchoCLIP
from echoclip.utils import set_seed


def load_config(path: Path) -> dict:
    with path.open(encoding="utf-8") as f:
        return yaml.safe_load(f)


def build_model(cfg: dict) -> EchoCLIP:
    backbone = cfg.get("vision_backbone", "resnet18")
    pretrained = cfg.get("pretrained_vision", True)
    if model_module.timm is None and backbone != "simple_cnn":
        print("timm unavailable; falling back to vision_backbone=simple_cnn")
        backbone = "simple_cnn"
        pretrained = False
        cfg["init_open_clip"] = False
        cfg["init_official_echo_clip"] = False
    model_cfg = EchoCLIPConfig(
        embed_dim=cfg.get("embed_dim", 512),
        image_size=cfg.get("image_size", 224),
        context_length=cfg.get("context_length", 77),
        vision_backbone=backbone,
        text_layers=cfg.get("text_layers", 12),
        text_heads=cfg.get("text_heads", 8),
        text_width=cfg.get("text_width", 512),
        pretrained_vision=pretrained,
        open_clip_tag=cfg.get("open_clip_tag"),
        open_clip_model=cfg.get("open_clip_model", "convnext_base_w_320"),
        temporal_type=cfg.get("temporal_type", "none"),
        temporal_layers=cfg.get("temporal_layers", 2),
        temporal_heads=cfg.get("temporal_heads", 8),
        temporal_max_frames=max(
            cfg.get("temporal_max_frames", 64), cfg.get("video_frames", 16)
        ),
    )
    if cfg.get("init_official_echo_clip"):
        model = EchoCLIP.from_official_echo_clip(
            model_cfg, checkpoint_path=cfg.get("official_checkpoint")
        )
        print(f"pretrained source: {model.load_source}")
    elif cfg.get("init_open_clip"):
        model = EchoCLIP.from_open_clip(model_cfg)
        print(f"pretrained source: {model.load_source}")
    else:
        model = EchoCLIP(model_cfg)
    temporal_type = cfg.get("temporal_type", "none")
    if temporal_type and str(temporal_type).lower() not in ("none", "mean", ""):
        if model.temporal is None:
            model.attach_temporal(
                temporal_type,
                n_layers=cfg.get("temporal_layers", 2),
                n_heads=cfg.get("temporal_heads", 8),
                max_frames=max(
                    cfg.get("temporal_max_frames", 64), cfg.get("video_frames", 16)
                ),
            )
    apply_freeze(model, cfg)
    return model


def apply_freeze(model: EchoCLIP, cfg: dict) -> None:
    freeze_backbone = bool(cfg.get("freeze_backbone", False))
    freeze_text = bool(cfg.get("freeze_text", freeze_backbone))
    if freeze_backbone:
        for p in model.visual.parameters():
            p.requires_grad = False
        if getattr(model, "external_clip", None) is not None:
            for p in model.external_clip.parameters():
                p.requires_grad = False
    if freeze_text:
        for p in model.textual.parameters():
            p.requires_grad = False
    if not cfg.get("train_logit_scale", True):
        model.logit_scale.requires_grad = False
    if getattr(model, "temporal", None) is not None:
        for p in model.temporal.parameters():
            p.requires_grad = True
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_all = sum(p.numel() for p in model.parameters())
    print(f"trainable parameters: {n_train:,} / {n_all:,}")
    if n_train == 0:
        print("Warning: no trainable parameters; unfreezing logit_scale")
        model.logit_scale.requires_grad = True


def train_epoch(model, loader, optimizer, criterion, device, scaler=None):
    model.train()
    total_loss = 0.0
    for batch in tqdm(loader, desc="train", leave=False):
        images = batch["image"].to(device)
        texts = batch["text"].to(device)
        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast("cuda", enabled=scaler is not None):
            img_f, txt_f, scale = model(images, texts)
            if isinstance(criterion, TemporalClipLoss) and "image_2" in batch:
                img_f2 = model.encode_image(batch["image_2"].to(device))
                loss = criterion(img_f, txt_f, scale, video_features_2=img_f2)
            else:
                loss = criterion(img_f, txt_f, scale)
        if scaler:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()
        total_loss += loss.item()
    return total_loss / max(len(loader), 1)


@torch.no_grad()
def eval_epoch(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    for batch in loader:
        images = batch["image"].to(device)
        texts = batch["text"].to(device)
        img_f, txt_f, scale = model(images, texts)
        total_loss += criterion(img_f, txt_f, scale).item()
    return total_loss / max(len(loader), 1)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=ROOT / "configs" / "default.yaml")
    parser.add_argument("--manifest", type=Path, default=None)
    parser.add_argument("--manifest-dir", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--vision-backbone", type=str, default=None)
    parser.add_argument("--video-frames", type=int, default=None)
    parser.add_argument("--temporal-type", type=str, default=None)
    parser.add_argument("--freeze-backbone", action="store_true")
    parser.add_argument("--no-official", action="store_true")
    parser.add_argument("--sample-strategy", type=str, default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    if args.manifest:
        cfg["manifest"] = str(args.manifest)
        if args.manifest_dir is None:
            cfg["manifest_dir"] = str(Path(args.manifest).parent)
    if args.manifest_dir:
        cfg["manifest_dir"] = str(args.manifest_dir)
    if args.output_dir:
        cfg["output_dir"] = str(args.output_dir)
    if args.epochs is not None:
        cfg["epochs"] = args.epochs
    if args.batch_size is not None:
        cfg["batch_size"] = args.batch_size
    if args.vision_backbone:
        cfg["vision_backbone"] = args.vision_backbone
    if args.video_frames is not None:
        cfg["video_frames"] = args.video_frames
    if args.temporal_type:
        cfg["temporal_type"] = args.temporal_type
    if args.freeze_backbone:
        cfg["freeze_backbone"] = True
        cfg["freeze_text"] = True
    if args.no_official:
        cfg["init_official_echo_clip"] = False
        cfg["init_open_clip"] = False
    if args.sample_strategy:
        cfg["sample_strategy"] = args.sample_strategy
    if cfg.get("vision_backbone") == "simple_cnn":
        cfg["init_official_echo_clip"] = False
        cfg["init_open_clip"] = False

    set_seed(cfg.get("seed", 42))
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    manifest = Path(cfg["manifest"])
    if not manifest.exists():
        print(f"Manifest not found: {manifest}")
        print("Run: python scripts/make_demo_data.py")
        sys.exit(1)

    pairs = load_manifest(manifest)
    manifest_dir = Path(cfg.get("manifest_dir", manifest.parent))
    manifest_errors = validate_manifest(pairs, manifest_dir)
    if manifest_errors:
        print("Manifest validation failed:")
        for err in manifest_errors[:15]:
            print(f"  - {err}")
        sys.exit(1)
    train_pairs, val_pairs = split_manifest(pairs, cfg.get("val_ratio", 0.1), cfg.get("seed", 42))
    out_dir = Path(cfg["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    split_dir = out_dir / "splits"
    split_dir.mkdir(parents=True, exist_ok=True)

    def write_split(name, data):
        p = split_dir / f"{name}.json"
        p.write_text(json.dumps({"pairs": data}, indent=2), encoding="utf-8")
        return p

    train_manifest = write_split("train_split", train_pairs)
    val_manifest = write_split("val_split", val_pairs)

    context_length = cfg.get("context_length", 77)
    tokenizer = EchoTokenizer(context_length=context_length)
    ds_kwargs = dict(
        manifest_dir=manifest_dir,
        image_size=cfg.get("image_size", 224),
        context_length=context_length,
        video_frames=cfg.get("video_frames", 1),
        tokenizer=tokenizer,
        sample_strategy=cfg.get("sample_strategy", "random"),
        frame_pool=cfg.get("frame_pool", "stack"),
        two_views=float(cfg.get("view_weight", 0.0)) > 0,
        caption_mode=cfg.get("caption_mode", "random"),
    )
    train_ds = EchoCLIPDataset(
        train_manifest,
        seed=cfg.get("seed", 42),
        **ds_kwargs,
    )
    val_ds = EchoCLIPDataset(
        val_manifest,
        **{**ds_kwargs, "sample_strategy": cfg.get("val_sample_strategy", "uniform")},
    )

    model = build_model(cfg).to(device)
    view_weight = float(cfg.get("view_weight", 0.0))
    if cfg.get("video_frames", 1) > 1 or view_weight > 0:
        criterion = TemporalClipLoss(
            clip_weight=cfg.get("clip_weight", 1.0),
            view_weight=view_weight,
        )
    else:
        criterion = ClipLoss()
    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(
        trainable if trainable else model.parameters(),
        lr=cfg.get("lr", 5e-5),
        weight_decay=cfg.get("weight_decay", 0.2),
    )
    scaler = torch.amp.GradScaler("cuda") if device.startswith("cuda") else None

    batch_size = min(cfg.get("batch_size", 32), len(train_ds))
    val_batch_size = min(cfg.get("batch_size", 32), len(val_ds))
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=cfg.get("num_workers", 0),
        collate_fn=collate_batch,
        pin_memory=device.startswith("cuda"),
        drop_last=len(train_ds) > batch_size,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=val_batch_size,
        shuffle=False,
        num_workers=cfg.get("num_workers", 0),
        collate_fn=collate_batch,
    )

    best_val = float("inf")
    for epoch in range(1, cfg.get("epochs", 10) + 1):
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device, scaler)
        val_loss = eval_epoch(model, val_loader, criterion, device)
        print(f"epoch {epoch}: train_loss={train_loss:.4f} val_loss={val_loss:.4f}")

        save_checkpoint(out_dir / "last.pt", model, epoch, train_cfg=cfg)
        if val_loss < best_val:
            best_val = val_loss
            save_checkpoint(out_dir / "best.pt", model, epoch, train_cfg=cfg)
            print(f"  saved best.pt (val_loss={val_loss:.4f})")

    print(f"Training complete. Checkpoints in {out_dir}")


if __name__ == "__main__":
    main()
