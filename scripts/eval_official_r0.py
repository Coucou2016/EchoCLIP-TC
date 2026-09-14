#!/usr/bin/env python3
"""Official EchoCLIP R0 reproduction helper (open_clip loader / stride / prompts).

Mirrors upstream echonet/echo_CLIP zero-shot EF path when assets allow:
  - ``open_clip.create_model_and_transforms`` + ``get_tokenizer`` + ``preprocess_val``
  - AVI frames cropped/scaled to 224, then open_clip preprocess
  - Frame indices ``0:min(40,T):2`` (no pad-to-16)
  - EF prompts 0–100; top-20% median aggregation

Prefer this script under ``--paper``. It writes metrics with ``paper_mode=true`` and
``official_reproduction_verified`` only if aggregation parity + hub load OK.

Does **not** invent clinical MAE. Exits 2 when assets/weights are missing.

Examples
--------
::

  python scripts/eval_official_r0.py --demo
  python scripts/eval_official_r0.py --manifest data/echonet_dynamic/test.json --paper
  python scripts/eval_official_r0.py --avi path/to/video.avi --dry-run
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from echoclip.cycle_sample import sample_official_stride  # noqa: E402
from echoclip.official_parity import (  # noqa: E402
    PARITY_GAPS,
    compute_regression_metric_official,
    crop_and_scale_official,
)
from echoclip.protocol import OFFICIAL_EF_VALUES  # noqa: E402
from echoclip.zeroshot import compute_regression_score  # noqa: E402

EF_TEMPLATES = (
    "THE LEFT VENTRICULAR EJECTION FRACTION IS ESTIMATED TO BE <#>% ",
    "LV EJECTION FRACTION IS <#>%. ",
)


def _formula_parity_ok() -> bool:
    import torch
    import torch.nn.functional as F

    torch.manual_seed(0)
    video = F.normalize(torch.randn(2, 8, 16), dim=-1)
    prompts = F.normalize(torch.randn(101, 16), dim=-1)
    values = list(range(101))
    a = compute_regression_metric_official(video, prompts, values)
    b = compute_regression_score(video, prompts, values)
    return bool(torch.allclose(a, b, atol=1e-6, rtol=1e-5))


def _build_ef_prompts() -> Tuple[List[str], List[int]]:
    prompts: List[str] = []
    values: List[int] = []
    for tmpl in EF_TEMPLATES:
        for i in OFFICIAL_EF_VALUES:
            prompts.append(tmpl.replace("<#>", str(i)))
            values.append(int(i))
    return prompts, values


def _try_open_clip_hub(hub: str, device: str):
    try:
        import open_clip
    except ImportError as exc:
        return None, f"open_clip missing: {exc}"
    try:
        model, _, preprocess_val = open_clip.create_model_and_transforms(hub)
        tokenize = open_clip.get_tokenizer(hub)
        model = model.to(device).eval()
        return (model, tokenize, preprocess_val, open_clip), None
    except Exception as exc:  # noqa: BLE001
        return None, str(exc)


def _predict_avi_open_clip(
    avi: Path,
    *,
    model,
    tokenize,
    preprocess_val,
    device: str,
) -> float:
    """Direct official-style AVI → EF (bypasses EchoCLIPDataset / pad_or_trim)."""
    import cv2
    import torch
    import torch.nn.functional as F
    import torchvision.transforms as T

    cap = cv2.VideoCapture(str(avi))
    raw = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        raw.append(frame)
    cap.release()
    if not raw:
        raise RuntimeError(f"no frames in {avi}")

    # Official example often uses 224×224 crop; keep BGR as upstream read_avi does.
    frames = [crop_and_scale_official(f, (224, 224)) for f in raw]
    tensors = torch.stack(
        [preprocess_val(T.ToPILImage()(f)) for f in frames], dim=0
    )
    idx = sample_official_stride(len(tensors))
    tensors = tensors[idx].to(device)
    prompts, values = _build_ef_prompts()
    with torch.inference_mode():
        img_emb = F.normalize(model.encode_image(tensors), dim=-1).unsqueeze(0)
        tok = tokenize(prompts).to(device)
        txt_emb = F.normalize(model.encode_text(tok), dim=-1)
        pred = compute_regression_metric_official(img_emb, txt_emb, values)
    return float(pred.item())


def _manifest_records(manifest: Path) -> List[dict]:
    data = json.loads(manifest.read_text(encoding="utf-8"))
    if isinstance(data, dict) and "pairs" in data:
        return list(data["pairs"])
    if isinstance(data, list):
        return list(data)
    raise ValueError(f"Unrecognized manifest schema: {manifest}")


def _resolve_media(rec: dict, manifest_dir: Path) -> Optional[Path]:
    name = rec.get("image") or rec.get("file_name") or rec.get("video") or ""
    if not name:
        return None
    p = Path(name)
    if p.is_file():
        return p
    cand = manifest_dir / name
    if cand.is_file():
        return cand
    return None


def _ef_label(rec: dict) -> Optional[float]:
    for key in ("ef", "EF", "ejection_fraction", "EjectionFraction"):
        if key in rec and rec[key] is not None and str(rec[key]) != "":
            try:
                return float(rec[key])
            except (TypeError, ValueError):
                return None
    return None


def _run_open_clip_manifest(
    manifest: Path,
    manifest_dir: Path,
    *,
    hub: str,
    device: str,
    seed: int,
) -> Dict[str, Any]:
    import numpy as np

    packed, err = _try_open_clip_hub(hub, device)
    if packed is None:
        return {"ok": False, "error": err, "bypassed_dataset": True}
    model, tokenize, preprocess_val, _ = packed

    preds: List[float] = []
    labels: List[float] = []
    n_skip = 0
    for rec in _manifest_records(manifest):
        avi = _resolve_media(rec, manifest_dir)
        if avi is None or avi.suffix.lower() not in (".avi", ".mp4", ".mov"):
            n_skip += 1
            continue
        try:
            pred = _predict_avi_open_clip(
                avi,
                model=model,
                tokenize=tokenize,
                preprocess_val=preprocess_val,
                device=device,
            )
        except Exception:  # noqa: BLE001
            n_skip += 1
            continue
        y = _ef_label(rec)
        if y is None or not np.isfinite(y):
            n_skip += 1
            continue
        preds.append(pred)
        labels.append(float(y))

    if len(preds) < 1:
        return {
            "ok": False,
            "error": "no AVI+EF pairs evaluated via open_clip path",
            "n_skip": n_skip,
            "bypassed_dataset": True,
            "hub": hub,
        }

    y = np.asarray(labels, dtype=float)
    p = np.asarray(preds, dtype=float)
    mae = float(np.mean(np.abs(y - p)))
    return {
        "ok": True,
        "bypassed_dataset": True,
        "sample_strategy": "official_stride",
        "pad_to_16": False,
        "image_size": 224,
        "ef_grid": "0_100_step1",
        "ef_grid_n": len(OFFICIAL_EF_VALUES),
        "n_eval": int(len(preds)),
        "n_skip": int(n_skip),
        "mae": mae,
        "rmse": float(np.sqrt(np.mean((y - p) ** 2))),
        "load_source": hub,
        "prediction_mode": "zeroshot",
        "pool": "frames",
        "eval_seed": int(seed),
        "seed": int(seed),
        "hub": hub,
        "note": (
            "Official open_clip AVI path (no EchoCLIPDataset pad_or_trim). "
            "Do not invent additional clinical claims beyond this run."
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Official R0 EchoCLIP evaluation path")
    parser.add_argument("--manifest", type=Path, default=None)
    parser.add_argument("--manifest-dir", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--demo", action="store_true")
    parser.add_argument("--paper", action="store_true", default=False)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--hub", type=str, default="hf-hub:mkaichristensen/echo-clip")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--avi", type=Path, default=None, help="Optional single-AVI smoke")
    args = parser.parse_args()
    if not args.demo and not args.dry_run:
        # Default to paper semantics when not demo/dry-run (CLI may omit --paper).
        args.paper = True

    device = args.device or ("cuda" if __import__("torch").cuda.is_available() else "cpu")
    formula_ok = _formula_parity_ok()
    summary: Dict[str, Any] = {
        "experiment_id": "R0",
        "script": "eval_official_r0.py",
        "paper_mode": bool(args.paper or not args.demo),
        "ef_grid_n": len(OFFICIAL_EF_VALUES),
        "sample_strategy": "official_stride",
        "pad_to_16": False,
        "aggregation_formula_parity": formula_ok,
        "parity_gaps": list(PARITY_GAPS),
        "official_reproduction_verified": False,
        "eval_seed": int(args.seed),
        "seed": int(args.seed),
        "note": (
            "Verified only after hub load + formula parity. "
            "Clinical MAE requires EchoNet + official weights — never invent."
        ),
    }

    if args.dry_run:
        print(json.dumps(summary, indent=2))
        return 0 if formula_ok else 1

    if args.avi is not None:
        packed, err = _try_open_clip_hub(args.hub, device)
        if packed is None:
            summary.update({"ok": False, "error": err})
            print(json.dumps(summary, indent=2))
            return 2
        model, tokenize, preprocess_val, _ = packed
        try:
            pred = _predict_avi_open_clip(
                args.avi,
                model=model,
                tokenize=tokenize,
                preprocess_val=preprocess_val,
                device=device,
            )
        except Exception as exc:  # noqa: BLE001
            summary.update({"ok": False, "error": str(exc)})
            print(json.dumps(summary, indent=2))
            return 2
        summary.update(
            {
                "ok": True,
                "avi": str(args.avi),
                "pred_ef": pred,
                "bypassed_dataset": True,
                "load_source": args.hub,
                "official_reproduction_verified": bool(formula_ok),
            }
        )
        if formula_ok:
            os.environ["ECHOCLIP_OFFICIAL_PARITY_OK"] = "1"
        out = args.output or (ROOT / "checkpoints" / "protocol" / "R0" / "official_r0_avi.json")
        Path(out).parent.mkdir(parents=True, exist_ok=True)
        Path(out).write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(json.dumps(summary, indent=2))
        return 0

    if args.demo:
        manifest = ROOT / "data" / "demo" / "manifest.json"
        if not manifest.exists():
            print("Demo manifest missing; run: python scripts/make_demo_data.py")
            return 2
        os.environ.pop("ECHOCLIP_OFFICIAL_PARITY_OK", None)
        os.environ.setdefault("ECHOCLIP_SKIP_HUB", "1")
        # Demo cannot claim official hub path — fall back to eval_clinical wiring.
        cmd = [
            sys.executable,
            str(ROOT / "scripts" / "eval_clinical.py"),
            "--manifest",
            str(manifest),
            "--init-official",
            "--experiment-id",
            "R0",
            "--sample-strategy",
            "official_stride",
            "--prediction-mode",
            "zeroshot",
            "--pool",
            "frames",
            "--seed",
            str(args.seed),
            "--video-frames",
            "20",
        ]
        if args.output:
            cmd.extend(["--output", str(args.output)])
        print(">>", " ".join(cmd))
        code = subprocess.run(cmd, cwd=str(ROOT)).returncode
        summary["demo_mode"] = True
        summary["demo_is_not_clinical"] = True
        summary["official_reproduction_verified"] = False
        summary["bypassed_dataset"] = False
        if args.output and Path(args.output).exists():
            metrics = json.loads(Path(args.output).read_text(encoding="utf-8"))
            metrics.update(
                {
                    k: summary[k]
                    for k in (
                        "paper_mode",
                        "official_reproduction_verified",
                        "aggregation_formula_parity",
                        "pad_to_16",
                        "eval_seed",
                    )
                    if k in summary
                }
            )
            metrics["demo_is_not_clinical"] = True
            Path(args.output).write_text(json.dumps(metrics, indent=2), encoding="utf-8")
        print(json.dumps(summary, indent=2))
        return code

    if os.environ.get("ECHOCLIP_SKIP_HUB", "").strip() in ("1", "true", "yes"):
        print("Error: eval_official_r0 --paper cannot run with ECHOCLIP_SKIP_HUB=1")
        return 1

    manifest = args.manifest or ROOT / "data" / "echonet_dynamic" / "test.json"
    if not manifest.exists():
        print(f"Manifest not found: {manifest}")
        print("Provide --manifest or use --demo / --dry-run")
        return 2
    manifest_dir = args.manifest_dir or manifest.parent

    # Prefer direct open_clip AVI path (bypass generic Dataset / pad_or_trim).
    direct = _run_open_clip_manifest(
        manifest,
        Path(manifest_dir),
        hub=args.hub,
        device=device,
        seed=args.seed,
    )
    verified = bool(formula_ok and direct.get("ok") and direct.get("load_source"))
    if verified:
        os.environ["ECHOCLIP_OFFICIAL_PARITY_OK"] = "1"
    else:
        os.environ.pop("ECHOCLIP_OFFICIAL_PARITY_OK", None)

    if direct.get("ok"):
        metrics = {
            "task": "clinical_ef",
            "experiment_id": "R0",
            "baseline_name": "Official EchoCLIP zero-shot (reproduction path)",
            "paper_mode": True,
            "official_reproduction": True,
            "official_reproduction_verified": verified,
            "aggregation_formula_parity": formula_ok,
            "parity_gaps": list(PARITY_GAPS),
            **direct,
        }
        out = args.output or (ROOT / "checkpoints" / "protocol" / "R0" / "metrics.json")
        out = Path(out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
        print(f"Wrote {out}")
        print(json.dumps({**summary, **metrics}, indent=2))
        return 0

    # Fallback: eval_clinical with official_stride (still no pad-to-16 in Dataset).
    print(
        f"open_clip direct path unavailable ({direct.get('error')}); "
        "falling back to eval_clinical official_stride."
    )
    cmd = [
        sys.executable,
        str(ROOT / "scripts" / "eval_clinical.py"),
        "--manifest",
        str(manifest),
        "--init-official",
        "--paper",
        "--experiment-id",
        "R0",
        "--sample-strategy",
        "official_stride",
        "--prediction-mode",
        "zeroshot",
        "--pool",
        "frames",
        "--seed",
        str(args.seed),
        "--video-frames",
        "20",
    ]
    if args.manifest_dir:
        cmd.extend(["--manifest-dir", str(args.manifest_dir)])
    if args.output:
        cmd.extend(["--output", str(args.output)])
    if args.device:
        cmd.extend(["--device", args.device])
    print(">>", " ".join(cmd))
    code = subprocess.run(cmd, cwd=str(ROOT)).returncode
    summary.update(
        {
            "hub_ok": False,
            "hub_error": direct.get("error"),
            "official_reproduction_verified": False,
            "paper_mode": True,
            "bypassed_dataset": False,
            "fallback": "eval_clinical",
        }
    )
    meta_out = (
        args.output.parent / "official_r0_meta.json"
        if args.output
        else ROOT / "checkpoints" / "protocol" / "R0" / "official_r0_meta.json"
    )
    Path(meta_out).parent.mkdir(parents=True, exist_ok=True)
    Path(meta_out).write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return code


if __name__ == "__main__":
    raise SystemExit(main())
