"""Multi-seed protocol helper: run seeds and aggregate mean±SD JSON.

Paper recommendation: report 5 seeds (e.g. 0–4). Default helper runs
seeds 0,1,2 for faster iteration; pass ``--seeds 0,1,2,3,4`` for paper.

Does not invent clinical MAE — aggregates whatever ``run_protocol.py`` writes.
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

# Metrics to aggregate when present in metrics.json
AGG_KEYS = (
    "mae",
    "rmse",
    "r2",
    "auc_ef_lt_50",
    "auc_ef_lt_40",
    "auc_ef_lt_30",
    "ece_ef_lt_50",
    "ece_ef_lt_40",
    "ece_ef_lt_30",
    "brier_ef_lt_50",
    "conformal_coverage",
    "conformal_mean_width",
    "aurc",
    "adaptive_conformal_coverage",
)


def _mean_sd(vals: Sequence[float]) -> Dict[str, Optional[float]]:
    xs = [float(v) for v in vals if v is not None and math.isfinite(float(v))]
    if not xs:
        return {"mean": None, "sd": None, "n": 0}
    m = sum(xs) / len(xs)
    if len(xs) == 1:
        return {"mean": m, "sd": 0.0, "n": 1}
    var = sum((x - m) ** 2 for x in xs) / (len(xs) - 1)
    return {"mean": m, "sd": math.sqrt(var), "n": len(xs)}


def _parse_seeds(raw: str) -> List[int]:
    parts = [p.strip() for p in raw.split(",") if p.strip()]
    if not parts:
        raise SystemExit("No seeds provided")
    return [int(p) for p in parts]


def aggregate_metrics(
    per_seed: Dict[str, Dict[str, Any]],
) -> Dict[str, Any]:
    """Aggregate numeric keys across seeds → mean±SD."""
    out: Dict[str, Any] = {"seeds": sorted(per_seed.keys(), key=lambda s: int(s))}
    for key in AGG_KEYS:
        vals = []
        for seed, metrics in per_seed.items():
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


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run protocol across seeds and write mean±SD aggregate JSON"
    )
    parser.add_argument(
        "--seeds",
        type=str,
        default="0,1,2",
        help="Comma-separated seeds (default 0,1,2; use 0,1,2,3,4 for paper)",
    )
    parser.add_argument(
        "--experiments",
        type=str,
        default="R0,R1",
        help="Passed through to run_protocol.py",
    )
    parser.add_argument("--config", type=Path, default=ROOT / "configs" / "echonet_dynamic.yaml")
    parser.add_argument("--output-root", type=Path, default=ROOT / "checkpoints" / "seeds")
    parser.add_argument("--demo", action="store_true")
    parser.add_argument("--paper", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--vision-backbone", type=str, default=None)
    parser.add_argument("--extra", type=str, default="", help="Extra args for run_protocol")
    args = parser.parse_args()

    if args.paper and args.demo:
        print("Error: --paper and --demo are mutually exclusive.")
        return 1

    seeds = _parse_seeds(args.seeds)
    args.output_root.mkdir(parents=True, exist_ok=True)
    aggregate: Dict[str, Any] = {
        "protocol": "multi_seed",
        "seeds_requested": seeds,
        "paper_recommendation": "Use five seeds (0–4) for paper tables; default helper uses 0,1,2.",
        "experiments": {},
        "demo": bool(args.demo),
        "paper": bool(args.paper),
        "note": (
            "Demo aggregates are not clinical."
            if args.demo
            else "Clinical only with EchoNet + official weights; never invent MAE."
        ),
    }

    for seed in seeds:
        seed_root = args.output_root / f"seed_{seed}"
        seed_root.mkdir(parents=True, exist_ok=True)
        cmd = [
            sys.executable,
            str(ROOT / "scripts" / "run_protocol.py"),
            "--experiments",
            args.experiments,
            "--config",
            str(args.config),
            "--seed",
            str(seed),
            "--output-root",
            str(seed_root),
        ]
        if args.demo:
            cmd.append("--demo")
        if args.paper:
            cmd.append("--paper")
        if args.dry_run:
            cmd.append("--dry-run")
        if args.vision_backbone:
            cmd.extend(["--vision-backbone", args.vision_backbone])
        if args.extra.strip():
            cmd.extend(args.extra.split())
        print("\n>>", " ".join(cmd))
        code = subprocess.run(cmd, cwd=str(ROOT)).returncode
        if code != 0:
            print(f"Seed {seed} failed with code {code}")
            aggregate.setdefault("failures", []).append({"seed": seed, "code": code})
            if args.paper:
                return code
            continue

        # Collect metrics per experiment
        protocol_dir = seed_root / "checkpoints" / "protocol"
        if not protocol_dir.exists():
            continue
        for exp_dir in sorted(protocol_dir.iterdir()):
            if not exp_dir.is_dir():
                continue
            mpath = exp_dir / "metrics.json"
            if not mpath.exists():
                continue
            metrics = json.loads(mpath.read_text(encoding="utf-8"))
            exp_id = metrics.get("experiment_id") or exp_dir.name
            bucket = aggregate["experiments"].setdefault(exp_id, {})
            bucket[str(seed)] = {
                k: metrics.get(k) for k in AGG_KEYS if k in metrics
            }
            bucket[str(seed)]["load_source"] = metrics.get("load_source")
            bucket[str(seed)]["demo_mode"] = metrics.get("demo_mode")

    # Mean±SD per experiment
    summary = {}
    for exp_id, per_seed in aggregate["experiments"].items():
        numeric_only = {
            s: {k: v for k, v in m.items() if k in AGG_KEYS}
            for s, m in per_seed.items()
        }
        summary[exp_id] = aggregate_metrics(numeric_only)
    aggregate["summary_mean_sd"] = summary

    out_path = args.output_root / "seeds_aggregate.json"
    out_path.write_text(json.dumps(aggregate, indent=2), encoding="utf-8")
    print(f"\nWrote {out_path}")
    print(json.dumps({"summary_mean_sd": summary, "seeds": seeds}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
