"""Sequential experiment runner for the paper matrix.

Runs every row in dependency order, one process per command, logging to
logs/run_<name>.log. Designed to be restarted safely: existing JSON files are
skipped unless --force is given.

Order
  1 main rows      R2 R3 R4 R5 R6 (3 seeds each)
  2 zero-shot      R0 R1 ORACLE_EDES
  3 pooling        R4_MEAN (3 seeds), plus R5_EDES
  4 frames         R4 at 8 and 16 frames (32 comes from the main row)
  5 train scale    R4 at 500 and 1000 (2000 comes from the main row)
  6 top-k          R1 at 10% and 30% (20% comes from the main row)
  7 statistics     paired bootstrap and EF distribution check
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from _paths import logs_dir, results_dir  # noqa: E402

RES = results_dir()
LOGS = logs_dir()
PY = sys.executable
TRAINER = str(ROOT / "scripts" / "train_heads_from_cache.py")


def tag_for(exp: str, frames: int, seed: int, extra_tag: str = "", pool: str = "cls") -> Path:
    t = f"{exp}_s{seed}_f{frames}_{pool}"
    if extra_tag:
        t += f"_{extra_tag}"
    return RES / f"{t}.json"


def run(cmd: list, name: str) -> int:
    LOGS.mkdir(parents=True, exist_ok=True)
    log = LOGS / f"run_{name}.log"
    print(f"\n>>> {name}\n$ {' '.join(cmd)}", flush=True)
    t0 = time.time()
    with open(log, "w", encoding="utf-8") as fh:
        proc = subprocess.run(cmd, stdout=fh, stderr=subprocess.STDOUT, cwd=str(ROOT))
    tail = log.read_text(encoding="utf-8", errors="replace").splitlines()[-4:]
    print("\n".join(tail), flush=True)
    print(f"    [{time.time() - t0:.0f}s] exit={proc.returncode}", flush=True)
    return proc.returncode


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", default="all")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--pretrain-epochs", type=int, default=20)
    args = ap.parse_args()

    RES.mkdir(parents=True, exist_ok=True)
    stages = args.stage.split(",") if args.stage != "all" else [
        "main", "zeroshot", "pool", "frames", "scale", "topk", "stat"
    ]

    def maybe(cmd, name, exists: Path):
        if exists.exists() and not args.force:
            print(f"skip {name} (exists)", flush=True)
            return
        run(cmd, name)

    for stage in stages:
        if stage == "main":
            for exp in ("R2", "R3", "R4", "R5", "R6"):
                for seed in args.seeds:
                    maybe(
                        [PY, TRAINER, "--experiment", exp, "--seeds", str(seed),
                         "--frames", "32", "--epochs", str(args.epochs),
                         "--pretrain-epochs", str(args.pretrain_epochs)],
                        f"{exp}_s{seed}",
                        tag_for(exp, 32, seed) if exp != "R6" else RES / f"R6_s{seed}_f32_cls_a10_20.json",
                    )
        elif stage == "zeroshot":
            for exp, seed in (("R0", 42), ("R1", 42), ("ORACLE_EDES", 42)):
                frames = 20 if exp == "R0" else 32
                maybe(
                    [PY, TRAINER, "--experiment", exp, "--seeds", str(seed), "--frames", "32"],
                    f"{exp}",
                    tag_for(exp, frames if exp != "ORACLE_EDES" else 32, seed),
                )
        elif stage == "pool":
            for seed in args.seeds:
                maybe(
                    [PY, TRAINER, "--experiment", "R4_MEAN", "--seeds", str(seed),
                     "--frames", "32", "--epochs", str(args.epochs), "--tag", "abpoolmean"],
                    f"R4_MEAN_s{seed}",
                    tag_for("R4_MEAN", 32, seed, "abpoolmean"),
                )
            for seed in args.seeds[:1]:
                maybe(
                    [PY, TRAINER, "--experiment", "R5_EDES", "--seeds", str(seed),
                     "--frames", "32", "--epochs", str(args.epochs),
                     "--pretrain-epochs", str(args.pretrain_epochs), "--tag", "edes"],
                    f"R5_EDES_s{seed}",
                    tag_for("R5_EDES", 32, seed, "edes"),
                )
        elif stage == "frames":
            for nf in (8, 16):
                for seed in args.seeds[:2]:
                    maybe(
                        [PY, TRAINER, "--experiment", "R4", "--seeds", str(seed),
                         "--frames", str(nf), "--epochs", str(args.epochs),
                         "--tag", f"abframes{nf}"],
                        f"frames{nf}_s{seed}",
                        tag_for("R4", nf, seed, f"abframes{nf}"),
                    )
        elif stage == "scale":
            for n in (500, 1000):
                for seed in args.seeds[:2]:
                    maybe(
                        [PY, TRAINER, "--experiment", "R4", "--seeds", str(seed),
                         "--frames", "32", "--epochs", str(args.epochs),
                         "--train-limit", str(n), "--tag", f"abscale{n}"],
                        f"scale{n}_s{seed}",
                        tag_for("R4", 32, seed, f"abscale{n}"),
                    )
        elif stage == "topk":
            for frac in (0.1, 0.3):
                pct = int(frac * 100)
                maybe(
                    [PY, TRAINER, "--experiment", "R1", "--seeds", "42",
                     "--frames", "32", "--top-frac", str(frac), "--tag", f"abtop{pct}"],
                    f"topk{pct}",
                    tag_for("R1", 32, 42, f"abtop{pct}"),
                )
        elif stage == "stat":
            run([PY, str(ROOT / "scripts" / "run_ablation_matrix.py"), "--group", "stat"], "stat")
        else:
            raise SystemExit(f"unknown stage {stage}")
    print("\nALL STAGES DONE", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
