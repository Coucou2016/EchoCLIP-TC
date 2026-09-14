"""One-shot paper matrix runner for EchoCLIP-TA (R0–R6 + ablations).

Runs official R0, R0U16, R1–R6, Oracle, R5-hard, R5-soft, R5_EDESTRAIN under
``--paper`` (hard-fails without assets) or ``--demo`` (wiring proof only).

Trainable models use seeds 0–4 by default; aggregates mean±SD and paired ΔMAE
vs R0 when both have finite MAE. Never invents clinical numbers.

Examples
--------
Demo wiring (short)::

  python scripts/run_paper_matrix.py --demo --epochs 1 --seeds 0,1 --vision-backbone simple_cnn

Paper (requires EchoNet + hub weights)::

  python scripts/run_paper_matrix.py --paper --seeds 0,1,2,3,4
"""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from echoclip.protocol import get_experiment, metrics_path  # noqa: E402

# Full paper matrix (explicit; includes ablations not in run_protocol "all")
PAPER_MATRIX_IDS = (
    "R0",
    "R0U16",
    "R1",
    "R2",
    "R3",
    "R4",
    "R5",
    "R6",
    "ORACLE_EDES",
    "R5_EDESTRAIN",
)

# Multi-seed only for trainable / calibration-reuse paths
TRAINABLE_IDS = frozenset({"R2", "R3", "R4", "R5", "R6", "R5_EDESTRAIN"})

AGG_KEYS = (
    "mae",
    "rmse",
    "r2",
    "auc_ef_lt_50",
    "auc_ef_lt_40",
    "auc_ef_lt_30",
    "ece_ef_lt_50",
    "conformal_coverage",
    "aurc",
    "n_trainable_params",
    "trainable_pct_of_backbone",
)


def _parse_seeds(raw: str) -> List[int]:
    parts = [p.strip() for p in raw.split(",") if p.strip()]
    if not parts:
        raise SystemExit("No seeds provided")
    return [int(p) for p in parts]


def _mean_sd(vals: Sequence[float]) -> Dict[str, Optional[float]]:
    xs = [float(v) for v in vals if v is not None and math.isfinite(float(v))]
    if not xs:
        return {"mean": None, "sd": None, "n": 0}
    m = sum(xs) / len(xs)
    if len(xs) == 1:
        return {"mean": m, "sd": 0.0, "n": 1}
    var = sum((x - m) ** 2 for x in xs) / (len(xs) - 1)
    return {"mean": m, "sd": math.sqrt(var), "n": len(xs)}


def _run(cmd: Sequence[str]) -> int:
    print("\n>>", " ".join(str(c) for c in cmd))
    return subprocess.run(list(cmd), cwd=str(ROOT)).returncode


def _check_paper_assets() -> Optional[str]:
    """Return error message if paper assets missing; else None."""
    import os

    echonet = os.environ.get("ECHONET_ROOT", "").strip()
    train = ROOT / "data" / "echonet_dynamic" / "train.json"
    test = ROOT / "data" / "echonet_dynamic" / "test.json"
    if not test.exists() and not train.exists():
        if not echonet or not Path(echonet).exists():
            return (
                "PAPER MODE BLOCKED: EchoNet-Dynamic manifests / ECHONET_ROOT missing.\n"
                "Download AIMI EchoNet-Dynamic, then:\n"
                "  set ECHONET_ROOT=<AIMI_root>\n"
                "  python scripts/build_echonet_manifest.py --echonet-root %ECHONET_ROOT% --subset-5000\n"
                "  python scripts/run_paper_matrix.py --paper\n"
                "See PAPER.md / DATA.md. Demo wiring: --demo (not clinical)."
            )
    return None


def _aggregate_exp(per_seed: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    out: Dict[str, Any] = {"seeds": sorted(per_seed.keys(), key=lambda s: int(s))}
    for key in AGG_KEYS:
        vals = []
        for metrics in per_seed.values():
            if key in metrics and metrics[key] is not None:
                try:
                    vals.append(float(metrics[key]))
                except (TypeError, ValueError):
                    pass
        stats = _mean_sd(vals)
        if stats["n"]:
            out[key] = stats
    out["per_seed"] = per_seed
    return out


def _paired_delta(
    ref_mae: Optional[float], other_mae: Optional[float]
) -> Optional[Dict[str, Any]]:
    if ref_mae is None or other_mae is None:
        return None
    if not (math.isfinite(ref_mae) and math.isfinite(other_mae)):
        return None
    return {
        "delta_mae_vs_R0": float(other_mae - ref_mae),
        "note": "Point ΔMAE from aggregated means; paired bootstrap requires same-split preds.",
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="One-shot EchoCLIP-TA paper matrix (R0–R6 + ablations)"
    )
    parser.add_argument("--demo", action="store_true")
    parser.add_argument("--paper", action="store_true")
    parser.add_argument(
        "--seeds",
        type=str,
        default="0,1,2,3,4",
        help="Seeds for trainable models (default 0–4)",
    )
    parser.add_argument(
        "--experiments",
        type=str,
        default=None,
        help="Override matrix (comma IDs). Default = full paper matrix.",
    )
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--video-frames", type=int, default=None)
    parser.add_argument("--vision-backbone", type=str, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=ROOT / "checkpoints" / "paper_matrix",
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--include-r5-hard",
        action="store_true",
        default=True,
        help="Also run R5 with --no-ef-soft-contrastive (hard InfoNCE)",
    )
    parser.add_argument(
        "--skip-r5-hard",
        action="store_true",
        help="Skip R5-hard ablation",
    )
    args = parser.parse_args()

    if args.paper and args.demo:
        print("Error: --paper and --demo are mutually exclusive.")
        return 1
    if not args.paper and not args.demo:
        print("Error: pass --demo (wiring) or --paper (clinical assets required).")
        return 1

    if args.paper:
        err = _check_paper_assets()
        if err:
            print(err)
            return 2

    seeds = _parse_seeds(args.seeds)
    if args.experiments:
        exp_ids = [get_experiment(x.strip()).id for x in args.experiments.split(",") if x.strip()]
    else:
        exp_ids = list(PAPER_MATRIX_IDS)

    args.output_root.mkdir(parents=True, exist_ok=True)
    report: Dict[str, Any] = {
        "protocol": "EchoCLIP-TA paper_matrix",
        "demo": bool(args.demo),
        "paper": bool(args.paper),
        "seeds": seeds,
        "experiments": {},
        "r5_hard": None,
        "r5_soft": None,
        "aggregate": {},
        "paired_delta_vs_R0": {},
        "cardiacclip": None,
        "note": (
            "DEMO — not clinical."
            if args.demo
            else "Paper path; clinical MAE only with EchoNet + hub weights."
        ),
        "clinical_numbers": "待补充" if args.demo else "from metrics.json when assets present",
    }

    # CardiacCLIP comparison row (never invents MAE)
    try:
        from echoclip.cardiacclip import comparison_table_row, load_cardiacclip_interface

        cc = load_cardiacclip_interface()
        report["cardiacclip"] = comparison_table_row(cc)
        if args.paper and not cc.available:
            report["cardiacclip"]["paper_status"] = "待补充"
            report["cardiacclip"]["blocked_command"] = (
                "set CARDIACCLIP_WEIGHTS=<path> && "
                "python -c \"from echoclip.cardiacclip import load_cardiacclip_interface; "
                "print(load_cardiacclip_interface().to_dict())\""
            )
    except Exception as exc:  # noqa: BLE001
        report["cardiacclip"] = {"available": False, "error": str(exc), "clinical_numbers": "待补充"}

    for exp_id in exp_ids:
        spec = get_experiment(exp_id)
        use_seeds = seeds if exp_id in TRAINABLE_IDS else [seeds[0] if seeds else 42]
        per_seed: Dict[str, Dict[str, Any]] = {}
        for seed in use_seeds:
            seed_root = args.output_root / f"seed_{seed}"
            cmd = [
                sys.executable,
                str(ROOT / "scripts" / "run_protocol.py"),
                "--experiments",
                exp_id,
                "--seed",
                str(seed),
                "--output-root",
                str(seed_root),
            ]
            if args.demo:
                cmd.append("--demo")
            if args.paper:
                cmd.append("--paper")
            if args.epochs is not None:
                cmd.extend(["--epochs", str(args.epochs)])
            if args.batch_size is not None:
                cmd.extend(["--batch-size", str(args.batch_size)])
            if args.video_frames is not None:
                cmd.extend(["--video-frames", str(args.video_frames)])
            if args.vision_backbone:
                cmd.extend(["--vision-backbone", args.vision_backbone])
            if args.device:
                cmd.extend(["--device", args.device])
            if args.demo and args.vision_backbone is None:
                cmd.extend(["--vision-backbone", "simple_cnn"])
            if args.dry_run:
                cmd.append("--dry-run")

            code = 0 if args.dry_run else _run(cmd)
            if args.dry_run:
                print(f"[dry-run] {exp_id} seed={seed}")
                per_seed[str(seed)] = {"dry_run": True}
                continue
            if code != 0:
                per_seed[str(seed)] = {"error": f"run_protocol_exit_{code}"}
                continue
            mpath = metrics_path(seed_root, exp_id)
            if mpath.exists():
                per_seed[str(seed)] = json.loads(mpath.read_text(encoding="utf-8"))
            else:
                per_seed[str(seed)] = {"error": f"missing_metrics:{mpath}"}

        report["experiments"][exp_id] = {
            "title": spec.title,
            "prediction_mode": spec.prediction_mode,
            "aggregate": _aggregate_exp(per_seed) if not args.dry_run else {"per_seed": per_seed},
        }
        if exp_id == "R5":
            report["r5_soft"] = report["experiments"][exp_id]

    # R5-hard ablation (same seeds, no EF soft contrastive)
    if not args.skip_r5_hard and args.include_r5_hard and "R5" in exp_ids:
        per_seed_h: Dict[str, Dict[str, Any]] = {}
        for seed in seeds:
            seed_root = args.output_root / f"seed_{seed}_r5hard"
            cmd = [
                sys.executable,
                str(ROOT / "scripts" / "run_protocol.py"),
                "--experiments",
                "R5",
                "--seed",
                str(seed),
                "--output-root",
                str(seed_root),
                "--no-ef-soft-contrastive",
            ]
            if args.demo:
                cmd.append("--demo")
                if args.vision_backbone is None:
                    cmd.extend(["--vision-backbone", "simple_cnn"])
            if args.paper:
                cmd.append("--paper")
            if args.epochs is not None:
                cmd.extend(["--epochs", str(args.epochs)])
            if args.batch_size is not None:
                cmd.extend(["--batch-size", str(args.batch_size)])
            if args.video_frames is not None:
                cmd.extend(["--video-frames", str(args.video_frames)])
            if args.vision_backbone:
                cmd.extend(["--vision-backbone", args.vision_backbone])
            if args.device:
                cmd.extend(["--device", args.device])
            if args.dry_run:
                cmd.append("--dry-run")
                per_seed_h[str(seed)] = {"dry_run": True}
                continue
            code = _run(cmd)
            if code != 0:
                per_seed_h[str(seed)] = {"error": f"exit_{code}"}
                continue
            mpath = metrics_path(seed_root, "R5")
            if mpath.exists():
                m = json.loads(mpath.read_text(encoding="utf-8"))
                m["r5_variant"] = "hard_infonce"
                per_seed_h[str(seed)] = m
            else:
                per_seed_h[str(seed)] = {"error": "missing_metrics"}
        report["r5_hard"] = {
            "title": "R5 hard InfoNCE (--no-ef-soft-contrastive)",
            "aggregate": _aggregate_exp(per_seed_h) if not args.dry_run else {"per_seed": per_seed_h},
        }

    # Paired ΔMAE vs R0 (from aggregate means)
    r0_agg = report["experiments"].get("R0", {}).get("aggregate", {})
    r0_mae = None
    if isinstance(r0_agg.get("mae"), dict):
        r0_mae = r0_agg["mae"].get("mean")
    for eid, block in report["experiments"].items():
        if eid == "R0":
            continue
        mae_block = block.get("aggregate", {}).get("mae")
        other = mae_block.get("mean") if isinstance(mae_block, dict) else None
        delta = _paired_delta(r0_mae, other)
        if delta:
            report["paired_delta_vs_R0"][eid] = delta

    out_json = args.output_root / "paper_matrix_summary.json"
    out_json.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nWrote {out_json}")

    # Markdown table (placeholders stay 待补充 when MAE missing)
    lines = [
        "# EchoCLIP-TA paper matrix summary",
        "",
        f"- demo={args.demo} paper={args.paper}",
        f"- seeds={seeds}",
        "",
        "| ID | MAE mean±SD | ΔMAE vs R0 | n_seeds |",
        "|----|-------------|------------|---------|",
    ]
    for eid in exp_ids:
        agg = report["experiments"].get(eid, {}).get("aggregate", {})
        mae = agg.get("mae") or {}
        if mae.get("mean") is None:
            mae_s = "待补充"
        else:
            mae_s = f"{mae['mean']:.4f}±{mae.get('sd', 0):.4f}"
        d = report["paired_delta_vs_R0"].get(eid, {})
        d_s = f"{d['delta_mae_vs_R0']:.4f}" if d else "—"
        n = mae.get("n", 0) if mae else 0
        lines.append(f"| {eid} | {mae_s} | {d_s} | {n} |")
    if report.get("r5_hard"):
        mae = report["r5_hard"].get("aggregate", {}).get("mae") or {}
        mae_s = (
            "待补充"
            if mae.get("mean") is None
            else f"{mae['mean']:.4f}±{mae.get('sd', 0):.4f}"
        )
        lines.append(f"| R5-hard | {mae_s} | — | {mae.get('n', 0)} |")
    lines.append("")
    lines.append("CardiacCLIP: " + str(report.get("cardiacclip", {}).get("clinical_numbers", "待补充")))
    md_path = args.output_root / "paper_matrix_summary.md"
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {md_path}")

    # Fail paper if any experiment hard-failed (missing assets already gated)
    if args.paper and not args.dry_run:
        failed = []
        for eid, block in report["experiments"].items():
            for seed, m in block.get("aggregate", {}).get("per_seed", {}).items():
                if isinstance(m, dict) and m.get("error"):
                    failed.append(f"{eid}@seed{seed}:{m['error']}")
        if failed:
            print("Paper matrix failures:\n  " + "\n  ".join(failed))
            return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
