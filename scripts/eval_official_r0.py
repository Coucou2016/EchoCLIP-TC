#!/usr/bin/env python3
"""Official EchoCLIP R0 reproduction helper (open_clip loader / stride / prompts).

Prefer this script under ``--paper`` when hub weights are available. It:
  - loads via open_clip when possible
  - uses EF grid 0..100 and templates matching echonet/echo_CLIP
  - samples frames with ``0:min(40,T):2`` (no pad-to-16)
  - writes metrics with ``paper_mode=true`` and
    ``official_reproduction_verified`` only if aggregation parity + hub load OK

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
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from echoclip.official_parity import PARITY_GAPS, compute_regression_metric_official  # noqa: E402
from echoclip.protocol import OFFICIAL_EF_VALUES  # noqa: E402
from echoclip.zeroshot import compute_regression_score  # noqa: E402


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


def main() -> int:
    parser = argparse.ArgumentParser(description="Official R0 EchoCLIP evaluation path")
    parser.add_argument("--manifest", type=Path, default=None)
    parser.add_argument("--manifest-dir", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--demo", action="store_true")
    parser.add_argument("--paper", action="store_true", default=True)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--hub", type=str, default="hf-hub:mkaichristensen/echo-clip")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    formula_ok = _formula_parity_ok()
    summary = {
        "experiment_id": "R0",
        "script": "eval_official_r0.py",
        "paper_mode": True,
        "ef_grid_n": len(OFFICIAL_EF_VALUES),
        "sample_strategy": "official_stride",
        "pad_to_16": False,
        "aggregation_formula_parity": formula_ok,
        "parity_gaps": list(PARITY_GAPS),
        "official_reproduction_verified": False,
        "note": (
            "Verified only after hub load + formula parity. "
            "Clinical MAE requires EchoNet + official weights — never invent."
        ),
    }

    if args.dry_run:
        print(json.dumps(summary, indent=2))
        return 0 if formula_ok else 1

    # Delegate clinical run to eval_clinical with official_stride when possible.
    # Prefer open_clip path via compare helpers; clinical path uses EchoCLIP wrapper.
    if args.demo:
        manifest = ROOT / "data" / "demo" / "manifest.json"
        if not manifest.exists():
            print("Demo manifest missing; run: python scripts/make_demo_data.py")
            return 2
        # Demo cannot verify official hub parity
        os.environ.pop("ECHOCLIP_OFFICIAL_PARITY_OK", None)
        os.environ.setdefault("ECHOCLIP_SKIP_HUB", "1")
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
        code = os.spawnv(os.P_WAIT, sys.executable, cmd) if False else __import__(
            "subprocess"
        ).run(cmd, cwd=str(ROOT)).returncode
        summary["demo_mode"] = True
        summary["demo_is_not_clinical"] = True
        summary["official_reproduction_verified"] = False
        print(json.dumps(summary, indent=2))
        return code

    # Paper path: require hub; mark verified only if formula OK and env/hub success
    if os.environ.get("ECHOCLIP_SKIP_HUB", "").strip() in ("1", "true", "yes"):
        print("Error: eval_official_r0 --paper cannot run with ECHOCLIP_SKIP_HUB=1")
        return 1

    manifest = args.manifest or ROOT / "data" / "echonet_dynamic" / "test.json"
    if not manifest.exists():
        print(f"Manifest not found: {manifest}")
        print("Provide --manifest or use --demo / --dry-run")
        return 2

    # Try open_clip hub load as a gate
    hub_ok = False
    hub_err = None
    try:
        import open_clip

        open_clip.create_model_and_transforms(args.hub)
        hub_ok = True
    except Exception as exc:  # noqa: BLE001
        hub_err = str(exc)

    verified = bool(formula_ok and hub_ok)
    if verified:
        os.environ["ECHOCLIP_OFFICIAL_PARITY_OK"] = "1"
    else:
        os.environ.pop("ECHOCLIP_OFFICIAL_PARITY_OK", None)

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
    code = __import__("subprocess").run(cmd, cwd=str(ROOT)).returncode
    summary.update(
        {
            "hub_ok": hub_ok,
            "hub_error": hub_err,
            "official_reproduction_verified": verified,
            "paper_mode": True,
        }
    )
    out = args.output or (ROOT / "checkpoints" / "protocol" / "R0" / "official_r0_meta.json")
    out = Path(out)
    if out.suffix == ".json" and "metrics" not in out.name:
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(f"Wrote {out}")
    print(json.dumps(summary, indent=2))
    return code


if __name__ == "__main__":
    raise SystemExit(main())
