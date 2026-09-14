"""Label-efficiency runner: TRAIN subsets 1/5/10/25/100% with fixed TEST.

Compares R2 / R4 / R5-hard / R5-soft. Writes a JSON table. Runs on demo
manifests when EchoNet is absent (metrics labeled DEMO — not clinical).

Examples
--------
  python scripts/run_label_efficiency.py --demo --epochs 1 --vision-backbone simple_cnn
  python scripts/run_label_efficiency.py --fractions 0.01,0.05,0.1,0.25,1.0
"""

from __future__ import annotations

import argparse
import json
import random
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from echoclip.data import load_manifest  # noqa: E402
from echoclip.protocol import metrics_path  # noqa: E402

DEFAULT_FRACTIONS = (0.01, 0.05, 0.10, 0.25, 1.0)
METHODS = ("R2", "R4", "R5_soft", "R5_hard")


def _run(cmd: Sequence[str]) -> int:
    print("\n>>", " ".join(str(c) for c in cmd))
    return subprocess.run(list(cmd), cwd=str(ROOT)).returncode


def _subset_pairs(pairs: List[dict], fraction: float, seed: int) -> List[dict]:
    n = len(pairs)
    if n == 0:
        return []
    k = max(1, int(round(n * float(fraction))))
    k = min(k, n)
    rng = random.Random(seed)
    idx = list(range(n))
    rng.shuffle(idx)
    chosen = sorted(idx[:k])
    return [pairs[i] for i in chosen]


def _write_manifest(pairs: List[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps({"pairs": pairs}, indent=2),
        encoding="utf-8",
    )


def _resolve_manifests(args) -> tuple[Path, Path, Path, bool]:
    """Return train, test, cal, demo_flag."""
    if args.demo:
        demo = ROOT / "data" / "demo" / "manifest.json"
        if not demo.exists():
            raise SystemExit(f"Demo manifest missing: {demo}\nRun: python scripts/make_demo_data.py")
        return demo, demo, demo, True
    train = Path(
        args.train_manifest or ROOT / "data" / "echonet_dynamic" / "train.json"
    )
    test = Path(
        args.test_manifest or ROOT / "data" / "echonet_dynamic" / "test.json"
    )
    cal = Path(
        args.cal_manifest or ROOT / "data" / "echonet_dynamic" / "val.json"
    )
    if not train.exists() or not test.exists():
        print(
            "EchoNet manifests missing — falling back to --demo wiring.\n"
            "Blocked (external):\n"
            "  set ECHONET_ROOT=<AIMI_root>\n"
            "  python scripts/build_echonet_manifest.py --echonet-root %ECHONET_ROOT% --subset-5000\n"
            "  python scripts/run_label_efficiency.py\n"
        )
        demo = ROOT / "data" / "demo" / "manifest.json"
        if not demo.exists():
            raise SystemExit("No EchoNet and no demo data.")
        return demo, demo, demo, True
    return train, test, cal, False


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Label-efficiency curves for R2/R4/R5-hard/R5-soft"
    )
    parser.add_argument("--demo", action="store_true")
    parser.add_argument("--paper", action="store_true")
    parser.add_argument("--train-manifest", type=Path, default=None)
    parser.add_argument("--test-manifest", type=Path, default=None)
    parser.add_argument("--cal-manifest", type=Path, default=None)
    parser.add_argument("--manifest-dir", type=Path, default=None)
    parser.add_argument(
        "--fractions",
        type=str,
        default="0.01,0.05,0.1,0.25,1.0",
        help="Comma-separated TRAIN label fractions",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--video-frames", type=int, default=None)
    parser.add_argument("--vision-backbone", type=str, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "checkpoints" / "label_efficiency",
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if args.paper and args.demo:
        print("Error: --paper and --demo are mutually exclusive.")
        return 1

    fractions = [float(x.strip()) for x in args.fractions.split(",") if x.strip()]
    train_m, test_m, cal_m, demo = _resolve_manifests(args)
    if args.demo:
        demo = True
    if args.paper and demo:
        print(
            "PAPER MODE BLOCKED: need EchoNet TRAIN/TEST manifests.\n"
            "  set ECHONET_ROOT=<AIMI_root>\n"
            "  python scripts/build_echonet_manifest.py --echonet-root %ECHONET_ROOT% --subset-5000\n"
            "  python scripts/run_label_efficiency.py --paper\n"
        )
        return 2

    train_pairs = load_manifest(train_m)
    test_pairs = load_manifest(test_m)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    table: Dict[str, Any] = {
        "protocol": "label_efficiency",
        "fractions": fractions,
        "methods": list(METHODS),
        "n_train_full": len(train_pairs),
        "n_test_fixed": len(test_pairs),
        "test_manifest": str(test_m),
        "demo": demo,
        "paper": bool(args.paper),
        "seed": args.seed,
        "rows": [],
        "note": (
            "DEMO — not clinical."
            if demo
            else "Fixed TEST; TRAIN subsets only. Do not invent MAE."
        ),
        "clinical_numbers": "待补充" if demo else "from metrics when run completes",
    }

    for frac in fractions:
        subset = _subset_pairs(train_pairs, frac, seed=args.seed)
        frac_dir = args.output_dir / f"frac_{frac:.4f}".replace(".", "p")
        frac_dir.mkdir(parents=True, exist_ok=True)
        train_sub = frac_dir / "train_subset.json"
        _write_manifest(subset, train_sub)
        # Keep TEST fixed (full test manifest)
        row: Dict[str, Any] = {
            "fraction": frac,
            "n_train": len(subset),
            "methods": {},
        }
        method_specs = [
            ("R2", "R2", []),
            ("R4", "R4", []),
            ("R5_soft", "R5", []),
            ("R5_hard", "R5", ["--no-ef-soft-contrastive"]),
        ]
        for label, exp_id, extra in method_specs:
            out_root = frac_dir / label
            out_root.mkdir(parents=True, exist_ok=True)
            cmd = [
                sys.executable,
                str(ROOT / "scripts" / "run_protocol.py"),
                "--experiments",
                exp_id,
                "--train-manifest",
                str(train_sub),
                "--test-manifest",
                str(test_m),
                "--cal-manifest",
                str(cal_m),
                "--seed",
                str(args.seed),
                "--output-root",
                str(out_root),
            ]
            if args.manifest_dir:
                cmd.extend(["--manifest-dir", str(args.manifest_dir)])
            elif not demo:
                # Default: media next to EchoNet root / manifest parent
                cmd.extend(["--manifest-dir", str(test_m.parent)])
            if demo:
                cmd.append("--demo")
                # demo mode ignores custom manifests in run_protocol — write note
                # For true subset on demo, we still pass manifests but run_protocol
                # forces demo_manifest. Document this limitation.
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
            elif demo:
                cmd.extend(["--vision-backbone", "simple_cnn"])
            if args.device:
                cmd.extend(["--device", args.device])
            for e in extra:
                cmd.append(e)
            if args.dry_run:
                cmd.append("--dry-run")

            if demo and not args.dry_run:
                # run_protocol --demo remaps manifests; still exercise wiring per method
                pass

            code = 0 if args.dry_run else _run(cmd)
            entry: Dict[str, Any] = {
                "experiment": exp_id,
                "label": label,
                "exit_code": code,
            }
            if args.dry_run:
                entry["dry_run"] = True
            else:
                mpath = metrics_path(out_root, exp_id)
                if mpath.exists():
                    metrics = json.loads(mpath.read_text(encoding="utf-8"))
                    entry["mae"] = metrics.get("mae")
                    entry["rmse"] = metrics.get("rmse")
                    entry["n_trainable_params"] = metrics.get("n_trainable_params")
                    entry["metrics_path"] = str(mpath)
                    if demo:
                        entry["demo_is_not_clinical"] = True
                else:
                    entry["error"] = f"missing {mpath}"
                    entry["mae"] = None
            row["methods"][label] = entry
        table["rows"].append(row)

    if demo:
        table["demo_limitation"] = (
            "run_protocol --demo uses the fixed demo manifest for all fractions; "
            "fraction rows still exercise each method end-to-end. Real label curves "
            "require EchoNet TRAIN/TEST (see blocked_command)."
        )
        table["blocked_command"] = (
            "set ECHONET_ROOT=<AIMI_root> && "
            "python scripts/build_echonet_manifest.py --echonet-root %ECHONET_ROOT% --subset-5000 && "
            "python scripts/run_label_efficiency.py --paper"
        )

    out_json = args.output_dir / "label_efficiency_table.json"
    out_json.write_text(json.dumps(table, indent=2), encoding="utf-8")
    print(f"\nWrote {out_json}")

    # Markdown
    lines = [
        "# Label efficiency",
        "",
        f"demo={demo} n_train_full={len(train_pairs)} n_test={len(test_pairs)}",
        "",
        "| fraction | n_train | R2 MAE | R4 MAE | R5-soft MAE | R5-hard MAE |",
        "|----------|---------|--------|--------|-------------|-------------|",
    ]
    for row in table["rows"]:
        cells = [f"{row['fraction']:.2%}", str(row["n_train"])]
        for m in METHODS:
            mae = row["methods"].get(m, {}).get("mae")
            cells.append("待补充" if mae is None else f"{float(mae):.4f}")
        lines.append("| " + " | ".join(cells) + " |")
    md = args.output_dir / "label_efficiency_table.md"
    md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
