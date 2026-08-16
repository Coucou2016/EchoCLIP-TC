"""Minimal import and one training-step smoke test."""

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main() -> int:
    demo_dir = ROOT / "data" / "demo"
    manifest = demo_dir / "manifest.json"
    if not manifest.exists():
        subprocess.check_call([sys.executable, str(ROOT / "scripts" / "make_demo_data.py")])

    from echoclip.config import EchoCLIPConfig
    from echoclip.data import EchoCLIPDataset, collate_batch
    from echoclip.loss import ClipLoss
    from echoclip.model import EchoCLIP
    from torch.utils.data import DataLoader

    print("imports OK", flush=True)

    cfg = EchoCLIPConfig(
        vision_backbone="simple_cnn",
        pretrained_vision=False,
        embed_dim=128,
        text_width=256,
        text_layers=4,
        text_heads=8,
        temporal_type="none",
    )
    model = EchoCLIP(cfg)
    ds = EchoCLIPDataset(manifest, manifest_dir=demo_dir, image_size=128, context_length=32)
    loader = DataLoader(ds, batch_size=4, collate_fn=collate_batch)
    batch = next(iter(loader))
    loss_fn = ClipLoss()
    img_f, txt_f, scale = model(batch["image"], batch["text"])
    loss = loss_fn(img_f, txt_f, scale)
    loss.backward()
    print(f"forward/backward OK, loss={loss.item():.4f}")

    import torch
    from echoclip.zeroshot import compute_binary_score

    frame = torch.randn(128)
    prompts = torch.randn(3, 128)
    score = compute_binary_score(frame, prompts)
    assert score.shape == torch.Size([1]), score.shape
    print("zeroshot shapes OK")

    from echoclip.calibrate import expected_calibration_error, split_conformal_quantile
    from echoclip.temporal import TemporalTransformer

    agg = TemporalTransformer(dim=128, n_layers=1, n_heads=4, max_frames=8)
    clip = torch.randn(2, 4, 128)
    pooled = agg(clip)
    assert pooled.shape == (2, 128), pooled.shape
    model.attach_temporal("attn_pool", n_heads=4, max_frames=8)
    video = torch.randn(2, 4, 3, 64, 64)
    z_v = model.encode_video(video)
    assert z_v.shape == (2, 128), z_v.shape
    print("temporal forward OK")

    ds_t = EchoCLIPDataset(
        manifest,
        manifest_dir=demo_dir,
        image_size=64,
        context_length=32,
        video_frames=4,
        sample_strategy="uniform",
        seed=0,
    )
    batch_t = collate_batch([ds_t[0], ds_t[1]])
    assert batch_t["image"].dim() == 5, batch_t["image"].shape
    assert batch_t["image"].shape[1] == 4, batch_t["image"].shape
    print(f"video_frames stack OK, image={tuple(batch_t['image'].shape)}")

    labels = torch.tensor([0.0, 1.0, 1.0, 0.0])
    ece = expected_calibration_error(labels.numpy(), labels.numpy())
    q = split_conformal_quantile([0.2, 0.5, 1.0, 1.5], alpha=0.1)
    assert q >= 1.0, q
    print(f"calibration helpers OK, ece={ece:.4f} conformal_q={q:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
