"""Evaluate retrieval metrics and demo zero-shot tasks on a manifest."""

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

from echoclip.checkpoint import load_checkpoint
from echoclip.data import EchoCLIPDataset, collate_batch, load_manifest, validate_manifest
from echoclip.eval import pairwise_retrieval_metrics, zero_shot_pacemaker_accuracy
from echoclip.text import EchoTokenizer
from echoclip.utils import set_seed
from echoclip.zeroshot import EchoCLIPInference


@torch.no_grad()
def encode_dataset(model, loader, device):
    all_img, all_txt, all_raw = [], [], []
    model.eval()
    for batch in tqdm(loader, desc="encode"):
        images = batch["image"].to(device)
        texts = batch["text"].to(device)
        if images.dim() == 5:
            img_f = model.encode_video(images)
        else:
            img_f = model.encode_image(images)
        txt_f = model.encode_text(texts)
        all_img.append(img_f.cpu())
        all_txt.append(txt_f.cpu())
    return torch.cat(all_img), torch.cat(all_txt)


def main() -> None:
    parser = argparse.ArgumentParser(description="EchoCLIP evaluation")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, default=None)
    parser.add_argument("--manifest-dir", type=Path, default=None)
    parser.add_argument("--config", type=Path, default=ROOT / "configs" / "default.yaml")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, default=None, help="Write JSON metrics here")
    parser.add_argument("--skip-zeroshot", action="store_true")
    args = parser.parse_args()

    set_seed(args.seed)
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    cfg = {}
    if args.config.exists():
        cfg = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    manifest = args.manifest or Path(cfg.get("manifest", ROOT / "data" / "demo" / "manifest.json"))
    manifest_dir = args.manifest_dir or Path(cfg.get("manifest_dir", manifest.parent))

    pairs = load_manifest(manifest)
    errors = validate_manifest(pairs, manifest_dir)
    if errors:
        print("Manifest errors:")
        for e in errors[:20]:
            print(f"  - {e}")
        if len(errors) > 20:
            print(f"  ... and {len(errors) - 20} more")
        sys.exit(1)

    model, ckpt = load_checkpoint(args.checkpoint, device=device)
    context_length = model.config.context_length
    tokenizer = EchoTokenizer(context_length=context_length)
    ds = EchoCLIPDataset(
        manifest,
        manifest_dir=manifest_dir,
        image_size=model.config.image_size,
        context_length=context_length,
        tokenizer=tokenizer,
        video_frames=cfg.get("video_frames", 1),
        sample_strategy=cfg.get("val_sample_strategy", cfg.get("sample_strategy", "uniform")),
    )
    loader = DataLoader(
        ds,
        batch_size=min(args.batch_size, len(ds)),
        shuffle=False,
        collate_fn=collate_batch,
    )

    img_f, txt_f = encode_dataset(model, loader, device)
    metrics = pairwise_retrieval_metrics(img_f, txt_f, model.logit_scale.cpu())
    metrics["n_pairs"] = len(ds)
    metrics["checkpoint_epoch"] = ckpt.get("epoch", "?")
    metrics["note"] = (
        "Retrieval R@k is a diagnostic/demo metric, not the EchoCLIP-TC paper "
        "primary. Use scripts/eval_clinical.py for EF MAE/RMSE/R2, threshold "
        "AUCs, and calibration."
    )

    if not args.skip_zeroshot:
        engine = EchoCLIPInference(model, device=device, tokenizer=tokenizer)
        frame_embs = img_f.to(device)
        raw_texts = [ds.pairs[i]["text"] for i in range(len(ds))]
        zs = zero_shot_pacemaker_accuracy(frame_embs, raw_texts, engine)
        metrics.update({f"demo_{k}": v for k, v in zs.items()})

    print(json.dumps(metrics, indent=2))
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
        print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
