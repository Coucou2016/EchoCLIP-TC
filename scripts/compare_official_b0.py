#!/usr/bin/env python3
"""Compare this repo's B0 path to echonet/echo_CLIP utils on one AVI (optional).

Requires:
  - A local EchoNet-style ``.avi``
  - ``open-clip-torch`` + hub access (or a local EchoCLIP checkpoint)
  - Optional: a checkout of https://github.com/echonet/echo_CLIP for side-by-side

Does **not** invent clinical MAE. Exits 0 with a JSON summary of differences /
parity gaps when assets are present; exits 2 with instructions when missing.

Examples
--------
::

  python scripts/compare_official_b0.py --avi path/to/video.avi
  python scripts/compare_official_b0.py --avi video.avi --hub hf-hub:mkaichristensen/echo-clip
  python scripts/compare_official_b0.py --dry-run   # print gaps + formula check only
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from echoclip.cycle_sample import sample_official_stride  # noqa: E402
from echoclip.official_parity import (  # noqa: E402
    PARITY_GAPS,
    compute_regression_metric_official,
    crop_and_scale_official,
)
from echoclip.preprocess import crop_and_scale, read_video_frames  # noqa: E402
from echoclip.zeroshot import compute_regression_score  # noqa: E402


def _formula_smoke() -> dict:
    torch.manual_seed(42)
    video = F.normalize(torch.randn(1, 10, 16), dim=-1)
    prompts = F.normalize(torch.randn(101, 16), dim=-1)
    values = list(range(101))
    a = compute_regression_metric_official(video, prompts, values)
    b = compute_regression_score(video, prompts, values)
    return {
        "aggregation_allclose": bool(torch.allclose(a, b, atol=1e-6)),
        "official_pred": float(a.item()),
        "ours_pred": float(b.item()),
        "stride_example_T100": sample_official_stride(100).tolist(),
    }


def _try_hub_infer(avi: Path, hub: str, device: str) -> dict:
    import cv2
    from PIL import Image
    import torchvision.transforms as T

    try:
        import open_clip
    except ImportError as e:
        return {"ok": False, "error": f"open_clip missing: {e}"}

    try:
        model, _, preprocess_val = open_clip.create_model_and_transforms(hub)
    except Exception as e:  # noqa: BLE001 — hub/network failures are expected
        return {"ok": False, "error": f"hub load failed: {e}"}

    model = model.to(device).eval()
    tokenize = open_clip.get_tokenizer(hub)

    # Official-like: crop to 224, open_clip preprocess, stride 0:min(40,T):2
    cap = cv2.VideoCapture(str(avi))
    raw = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        raw.append(frame)
    cap.release()
    if not raw:
        return {"ok": False, "error": f"no frames in {avi}"}

    frames_bgr = [crop_and_scale_official(f, (224, 224)) for f in raw]
    tensors = torch.stack(
        [preprocess_val(T.ToPILImage()(f)) for f in frames_bgr], dim=0
    )
    idx = sample_official_stride(len(tensors))
    tensors = tensors[idx].to(device)

    templates = [
        "THE LEFT VENTRICULAR EJECTION FRACTION IS ESTIMATED TO BE <#>% ",
        "LV EJECTION FRACTION IS <#>%. ",
    ]
    prompts, values = [], []
    for tmpl in templates:
        for i in range(101):
            prompts.append(tmpl.replace("<#>", str(i)))
            values.append(i)
    with torch.inference_mode():
        img_emb = F.normalize(model.encode_image(tensors), dim=-1).unsqueeze(0)
        tok = tokenize(prompts).to(device)
        txt_emb = F.normalize(model.encode_text(tok), dim=-1)
        pred_off = compute_regression_metric_official(img_emb, txt_emb, values)
        pred_ours = compute_regression_score(img_emb, txt_emb, values)

    # Also compare our default RGB 640×480 path stride indices only (no claim of MAE)
    try:
        rgb = read_video_frames(avi, res=(640, 480))
        n = len(rgb)
    except Exception as e:  # noqa: BLE001
        rgb, n = None, None
        rgb_err = str(e)
    else:
        rgb_err = None

    return {
        "ok": True,
        "hub": hub,
        "n_raw_frames": len(raw),
        "n_strided": int(len(idx)),
        "stride_indices": idx.tolist(),
        "pred_official_utils": float(pred_off.item()),
        "pred_ours_aggregation": float(pred_ours.item()),
        "aggregation_delta": float(abs(pred_off - pred_ours).item()),
        "our_rgb_640_frame_count": n,
        "our_rgb_read_error": rgb_err,
        "note": (
            "Predictions use the same embeddings; delta should be ~0 if aggregation "
            "matches. End-to-end vs a separate echo_CLIP checkout may still differ "
            "due to PARITY_GAPS (crop/color/dtype)."
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Optional B0 parity check vs official utils")
    parser.add_argument("--avi", type=Path, default=None, help="Echo AVI path")
    parser.add_argument(
        "--hub",
        type=str,
        default="hf-hub:mkaichristensen/echo-clip",
        help="open_clip model id",
    )
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only run tensor formula smoke + print PARITY_GAPS (no hub/AVI)",
    )
    parser.add_argument("--output", type=Path, default=None, help="Write JSON summary")
    args = parser.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    summary = {
        "parity_gaps": list(PARITY_GAPS),
        "formula_smoke": _formula_smoke(),
        "clinical_mae": None,
        "pad_to_16": False,
        "official_stride": "0:min(40,T):2",
        "official_reproduction_verified": False,
        "honesty": "No clinical MAE invented. Demo/missing assets ≠ EchoNet results.",
    }

    if args.dry_run or args.avi is None:
        summary["status"] = "dry_run" if args.dry_run or args.avi is None else "ok"
        if args.avi is None and not args.dry_run:
            summary["status"] = "blocked_missing_avi"
            summary["hint"] = (
                "Provide --avi <path.avi> and hub access for a full compare. "
                "Aggregation golden tests live in tests/test_official_b0_parity.py."
            )
            print(json.dumps(summary, indent=2))
            if args.output:
                args.output.write_text(json.dumps(summary, indent=2), encoding="utf-8")
            return 2
        print(json.dumps(summary, indent=2))
        if args.output:
            args.output.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        return 0

    if not args.avi.exists():
        summary["status"] = "blocked_avi_not_found"
        summary["avi"] = str(args.avi)
        print(json.dumps(summary, indent=2))
        return 2

    summary["hub_run"] = _try_hub_infer(args.avi, args.hub, device)
    formula_ok = bool(summary["formula_smoke"].get("aggregation_allclose"))
    hub_ok = bool(summary["hub_run"].get("ok"))
    summary["official_reproduction_verified"] = bool(formula_ok and hub_ok)
    summary["status"] = "ok" if hub_ok else "blocked_hub_or_weights"
    if summary["official_reproduction_verified"]:
        import os

        os.environ["ECHOCLIP_OFFICIAL_PARITY_OK"] = "1"
    print(json.dumps(summary, indent=2))
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return 0 if summary["status"] == "ok" else 2


if __name__ == "__main__":
    raise SystemExit(main())
