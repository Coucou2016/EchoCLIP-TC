"""Run EchoCLIP-TC paper protocol experiments B0 / M1 / M2 / M4.

Writes comparable ``metrics.json`` under ``checkpoints/protocol/<ID>/``.

Demo mode (``--demo``) exercises the wiring on synthetic data only —
metrics are never clinical.

Examples
--------
List modes::

  python scripts/run_protocol.py --list

B0 official zero-shot (needs EchoNet + hub weights for paper numbers)::

  python scripts/run_protocol.py --experiments B0

Full matrix with training for M2/M4::

  python scripts/run_protocol.py --experiments B0,M1,M2,M4

Windows CPU smoke (demo, simple_cnn)::

  python scripts/run_protocol.py --demo --experiments B0,M1 --vision-backbone simple_cnn
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import List, Optional, Sequence

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from echoclip.protocol import (  # noqa: E402
    EXPERIMENT_IDS,
    get_experiment,
    list_experiments,
    merge_metrics_meta,
    metrics_path,
    protocol_output_dir,
    write_protocol_comparison,
)


def _print_catalog() -> None:
    print("EchoCLIP-TC protocol experiments (Table 1 path)\n")
    for spec in list_experiments():
        print(f"  {spec.id}: {spec.title}")
        print(f"      {spec.description}")
        print(
            f"      train={spec.train} pool={spec.pool} calibrate={spec.calibrate}"
        )
        if spec.notes:
            print(f"      note: {spec.notes}")
        print()


def _parse_experiments(raw: Optional[str]) -> List[str]:
    if not raw or raw.strip().lower() in ("all", "*"):
        return list(EXPERIMENT_IDS)
    out: List[str] = []
    for part in raw.split(","):
        part = part.strip()
        if not part:
            continue
        out.append(get_experiment(part).id)
    if not out:
        raise SystemExit("No experiments selected")
    return out


def _load_cfg(path: Path) -> dict:
    if not path.exists():
        return {}
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


def _run(cmd: Sequence[str], cwd: Path = ROOT) -> int:
    print("\n>>", " ".join(str(c) for c in cmd))
    return subprocess.run(list(cmd), cwd=str(cwd)).returncode


def _resolve_manifests(args, cfg: dict, demo: bool):
    if demo:
        demo_manifest = ROOT / "data" / "demo" / "manifest.json"
        return demo_manifest, demo_manifest, demo_manifest.parent, demo_manifest
    train = Path(
        args.train_manifest
        or cfg.get("manifest")
        or ROOT / "data" / "echonet_dynamic" / "train.json"
    )
    test = Path(
        args.test_manifest
        or cfg.get("test_manifest")
        or ROOT / "data" / "echonet_dynamic" / "test.json"
    )
    cal = Path(
        args.cal_manifest
        or cfg.get("cal_manifest")
        or ROOT / "data" / "echonet_dynamic" / "val.json"
    )
    if args.manifest_dir:
        mdir = Path(args.manifest_dir)
    else:
        mdir = Path(cfg.get("manifest_dir", test.parent))
    return train, test, mdir, cal


def _missing_data_help(path: Path) -> str:
    return (
        f"Required data not found: {path}\n\n"
        "EchoNet-Dynamic (Stanford AIMI, non-commercial):\n"
        "  https://echonet.github.io/dynamic/\n"
        "  https://stanfordaimi.azurewebsites.net/\n\n"
        "Then:\n"
        f"  python scripts/build_echonet_manifest.py "
        f"--echonet-root <root> --subset-5000\n\n"
        "For pipeline wiring only:\n"
        "  python scripts/run_protocol.py --demo --experiments B0,M1\n"
    )


def _train_m2(args, cfg: dict, train_manifest: Path, manifest_dir: Path, out_dir: Path) -> int:
    cmd = [
        sys.executable,
        str(ROOT / "scripts" / "train.py"),
        "--config",
        str(args.config),
        "--manifest",
        str(train_manifest),
        "--manifest-dir",
        str(manifest_dir),
        "--output-dir",
        str(out_dir),
        "--temporal-type",
        args.temporal_type or cfg.get("temporal_type", "transformer"),
        "--freeze-backbone",
    ]
    if args.video_frames is not None:
        cmd.extend(["--video-frames", str(args.video_frames)])
    if args.epochs is not None:
        cmd.extend(["--epochs", str(args.epochs)])
    if args.batch_size is not None:
        cmd.extend(["--batch-size", str(args.batch_size)])
    if args.vision_backbone:
        cmd.extend(["--vision-backbone", args.vision_backbone])
    if args.no_official or (args.vision_backbone == "simple_cnn"):
        cmd.append("--no-official")
    if args.sample_strategy:
        cmd.extend(["--sample-strategy", args.sample_strategy])
    if args.device:
        cmd.extend(["--device", args.device])
    return _run(cmd)


def _eval_experiment(
    args,
    spec,
    *,
    test_manifest: Path,
    cal_manifest: Path,
    manifest_dir: Path,
    checkpoint: Optional[Path],
    out_metrics: Path,
    demo: bool,
) -> int:
    cmd = [
        sys.executable,
        str(ROOT / "scripts" / "eval_clinical.py"),
        "--config",
        str(args.config),
        "--manifest",
        str(test_manifest),
        "--manifest-dir",
        str(manifest_dir),
        "--pool",
        spec.pool,
        "--seed",
        str(args.seed),
        "--output",
        str(out_metrics),
        "--experiment-id",
        spec.id,
    ]
    if spec.calibrate and cal_manifest.exists():
        # Non-demo: refuse fitting temperature/conformal on the same split as TEST.
        try:
            same_split = cal_manifest.resolve() == test_manifest.resolve()
        except OSError:
            same_split = str(cal_manifest) == str(test_manifest)
        if same_split and not demo:
            print(
                f"Error: {spec.id} calibration leak — cal_manifest equals test_manifest:\n"
                f"  {cal_manifest}\n"
                "Fit temperature / conformal on VAL only; never retune on TEST."
            )
            return 1
        if same_split and demo:
            print(
                f"Warning: {spec.id} demo uses the same manifest for cal and test "
                "(pipeline smoke only — not clinical)."
            )
        cmd.extend(["--cal-manifest", str(cal_manifest)])
    elif spec.calibrate and not demo:
        print(
            f"Error: {spec.id} requires a calibration manifest (VAL only), missing: "
            f"{cal_manifest}\n"
            "Build with: python scripts/build_echonet_manifest.py --echonet-root <root>\n"
            "Or pass --cal-manifest explicitly. Do not fit temperature/conformal on TEST."
        )
        return 1

    vf = args.video_frames if args.video_frames is not None else spec.video_frames
    if vf is not None:
        cmd.extend(["--video-frames", str(vf)])
    ss = args.sample_strategy or spec.sample_strategy
    if ss:
        # eval uses val strategy for reproducibility unless overridden
        cmd.extend(["--sample-strategy", ss if spec.id != "B0" else (args.sample_strategy or "uniform")])
    if args.batch_size is not None:
        cmd.extend(["--batch-size", str(args.batch_size)])
    if args.device:
        cmd.extend(["--device", args.device])

    if checkpoint and checkpoint.exists():
        cmd.extend(["--checkpoint", str(checkpoint)])
    else:
        cmd.append("--init-official")
        if args.official_checkpoint:
            cmd.extend(["--official-checkpoint", str(args.official_checkpoint)])
        # Demo / --no-official plumbing: skip hub download attempts.
        if demo or args.no_official or args.vision_backbone == "simple_cnn":
            import os

            os.environ.setdefault("ECHOCLIP_SKIP_HUB", "1")

    code = _run(cmd)
    if code != 0:
        return code

    # Stamp protocol metadata onto metrics.json
    if out_metrics.exists():
        metrics = json.loads(out_metrics.read_text(encoding="utf-8"))
        metrics = merge_metrics_meta(
            metrics,
            experiment=spec,
            demo=demo,
            extra={
                "protocol_output": str(out_metrics.parent),
                "b0_reproduce_hint": (
                    "Official EchoCLIP external ~7.1% EF MAE: seed=42 subset_5000 "
                    "(see subset_5000_ids.json) AND/OR full TEST; "
                    "load_source must be hf-hub:mkaichristensen/echo-clip "
                    "(not scratch_fallback / simple_cnn)."
                    if spec.id == "B0"
                    else None
                ),
            },
        )
        # Drop null hint for non-B0
        if metrics.get("b0_reproduce_hint") is None:
            metrics.pop("b0_reproduce_hint", None)
        out_metrics.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
        print(f"Updated {out_metrics} with protocol metadata")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description="EchoCLIP-TC paper protocol runner (B0/M1/M2/M4)"
    )
    parser.add_argument("--list", action="store_true", help="Print experiment catalog")
    parser.add_argument(
        "--experiments",
        type=str,
        default="all",
        help="Comma-separated IDs or 'all' (default). Example: B0,M1,M2,M4",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "configs" / "echonet_dynamic.yaml",
    )
    parser.add_argument("--train-manifest", type=Path, default=None)
    parser.add_argument("--test-manifest", type=Path, default=None)
    parser.add_argument("--cal-manifest", type=Path, default=None)
    parser.add_argument("--manifest-dir", type=Path, default=None)
    parser.add_argument("--output-root", type=Path, default=ROOT)
    parser.add_argument("--checkpoint", type=Path, default=None, help="Reuse TC ckpt for M2/M4")
    parser.add_argument("--official-checkpoint", type=Path, default=None)
    parser.add_argument("--no-official", action="store_true")
    parser.add_argument("--vision-backbone", type=str, default=None)
    parser.add_argument("--video-frames", type=int, default=None)
    parser.add_argument("--sample-strategy", type=str, default=None)
    parser.add_argument("--temporal-type", type=str, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--demo",
        action="store_true",
        help="Use data/demo manifests; skip missing-EchoNet hard fail (NOT clinical)",
    )
    parser.add_argument(
        "--skip-train",
        action="store_true",
        help="Do not train M2/M4; require existing checkpoint",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print planned actions without running train/eval",
    )
    args = parser.parse_args()

    if args.list:
        _print_catalog()
        return 0

    experiments = _parse_experiments(args.experiments)
    cfg = _load_cfg(args.config)
    train_m, test_m, mdir, cal_m = _resolve_manifests(args, cfg, args.demo)

    if not args.demo:
        # Fail clearly before launching GPU work
        need = test_m if test_m.exists() or not train_m.exists() else train_m
        if not test_m.exists() and not train_m.exists():
            print(_missing_data_help(test_m))
            return 1
        if not test_m.exists():
            print(_missing_data_help(test_m))
            return 1
    else:
        if not test_m.exists():
            print(f"Demo manifest missing: {test_m}")
            print("Run: python scripts/make_demo_data.py")
            return 1
        print(
            "DEMO MODE — results are pipeline smoke only, not EchoNet / paper EF MAE.\n"
        )

    # Shared M2 checkpoint directory under protocol/
    m2_dir = protocol_output_dir(args.output_root, "M2")
    shared_ckpt = args.checkpoint
    if shared_ckpt is None and (m2_dir / "best.pt").exists():
        shared_ckpt = m2_dir / "best.pt"

    results = {}
    protocol_root = Path(args.output_root) / "checkpoints" / "protocol"
    for exp_id in experiments:
        spec = get_experiment(exp_id)
        out_dir = protocol_output_dir(args.output_root, spec.id)
        out_dir.mkdir(parents=True, exist_ok=True)
        out_metrics = metrics_path(args.output_root, spec.id)
        print(f"\n===== {spec.id}: {spec.title} =====")

        if args.dry_run:
            print(
                json.dumps(
                    {
                        "id": spec.id,
                        "train": spec.train and not args.skip_train,
                        "pool": spec.pool,
                        "calibrate": spec.calibrate,
                        "test": str(test_m),
                        "cal": str(cal_m) if spec.calibrate else None,
                        "metrics": str(out_metrics),
                    },
                    indent=2,
                )
            )
            results[spec.id] = "dry-run"
            continue

        ckpt: Optional[Path] = None
        if spec.train:
            train_out = m2_dir  # M2 and M4 share the trained temporal ckpt
            train_out.mkdir(parents=True, exist_ok=True)
            ckpt = shared_ckpt if shared_ckpt and shared_ckpt.exists() else train_out / "best.pt"
            if not args.skip_train and (spec.id == "M2" or not ckpt.exists()):
                if not train_m.exists() and not args.demo:
                    print(_missing_data_help(train_m))
                    return 1
                # Demo train: force short schedule if user did not override
                if args.demo and args.epochs is None:
                    args.epochs = 1
                if args.demo and args.video_frames is None:
                    args.video_frames = min(4, spec.video_frames or 4)
                if args.demo and args.vision_backbone is None:
                    args.vision_backbone = "simple_cnn"
                code = _train_m2(args, cfg, train_m, mdir, train_out)
                if code != 0:
                    results[spec.id] = f"train_failed:{code}"
                    continue
                ckpt = train_out / "best.pt"
                shared_ckpt = ckpt
            elif not ckpt.exists():
                print(
                    f"{spec.id} needs a temporal checkpoint. Train M2 first or pass --checkpoint."
                )
                results[spec.id] = "missing_checkpoint"
                continue
        elif args.checkpoint and args.checkpoint.exists() and not spec.init_official:
            ckpt = args.checkpoint

        # B0/M1: prefer --init-official unless user forced a checkpoint for plumbing
        eval_ckpt = None
        if spec.requires_checkpoint:
            eval_ckpt = ckpt
        elif args.checkpoint and args.checkpoint.exists() and args.no_official:
            # plumbing path: local TC/simple ckpt
            eval_ckpt = args.checkpoint

        code = _eval_experiment(
            args,
            spec,
            test_manifest=test_m,
            cal_manifest=cal_m,
            manifest_dir=mdir,
            checkpoint=eval_ckpt,
            out_metrics=out_metrics,
            demo=args.demo,
        )
        results[spec.id] = "ok" if code == 0 else f"eval_failed:{code}"

    # Dry-run must not clobber a prior clinical/demo summary.json.
    if args.dry_run:
        print("\nDry-run complete — summary.json / comparison.* left unchanged.")
        print(json.dumps({"experiments": results, "dry_run": True}, indent=2))
        return 0

    summary_path = protocol_root / "summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary = {
        "experiments": results,
        "demo": bool(args.demo),
        "seed": args.seed,
        "note": (
            "Demo protocol run — not clinical."
            if args.demo
            else "Clinical metrics only valid with EchoNet labels + official weights."
        ),
    }
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"\nSummary → {summary_path}")
    print(json.dumps(summary, indent=2))

    # Refresh cross-experiment comparison whenever any metrics exist.
    try:
        paths = write_protocol_comparison(protocol_root)
        print(f"Comparison → {paths['md']}")
    except Exception as exc:  # noqa: BLE001 — table is best-effort after runs
        print(f"Warning: could not write protocol comparison table: {exc}")

    failed = [k for k, v in results.items() if v != "ok" and v != "dry-run"]
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())

