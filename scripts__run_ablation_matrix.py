"""Ablation and robustness runs for the EchoCLIP-TA/TC paper.

Groups (each writes one JSON per run into results/zh and appends to a log):
  frames    8 / 16 / 32 uniform frames, R4 structure
  pool      mean vs CLS attention pooling, R4 structure
  calib     R5 (uncalibrated) vs R6 (Platt + temperature + conformal)
  topk      zero-shot top-10% / 20% / 30% (R1)
  scale     TRAIN 500 / 1000 / 2000 / 4000 (learning curve, R4)
  stat      paired bootstrap of R5 vs R6, EF distribution of TEST-500 vs TEST

Usage
-----
python scripts/run_ablation_matrix.py --group frames
python scripts/run_ablation_matrix.py --group all
"""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from _paths import cache_dir, logs_dir, results_dir  # noqa: E402

RESULTS = results_dir()
CACHE = cache_dir()
LOGS = logs_dir()
PY = sys.executable
TRAINER = str(ROOT / "scripts" / "train_heads_from_cache.py")


def run(cmd: list, name: str) -> int:
    LOGS.mkdir(parents=True, exist_ok=True)
    log = LOGS / f"ablation_{name}.log"
    print(f"$ {' '.join(cmd)}", flush=True)
    with open(log, "w", encoding="utf-8") as fh:
        proc = subprocess.run(cmd, stdout=fh, stderr=subprocess.STDOUT, cwd=str(ROOT))
    tail = log.read_text(encoding="utf-8", errors="replace").splitlines()[-6:]
    print("\n".join(tail), flush=True)
    return proc.returncode


def base(exp: str, extra: list | None = None) -> list:
    cmd = [PY, TRAINER, "--experiment", exp, "--seeds", "42", "43", "44"]
    if extra:
        cmd += extra
    return cmd


def group_frames() -> None:
    # 32 frames is the main R4 row already; only 8 and 16 need extra runs.
    for nf in (8, 16):
        for seed in (42, 43, 44):
            run(
                [PY, TRAINER, "--experiment", "R4", "--seeds", str(seed),
                 "--frames", str(nf), "--cache-tag", "32_bgr", "--tag", f"abframes{nf}"],
                f"frames_{nf}_s{seed}",
            )


def group_pool() -> None:
    # R4 (CLS attention pooling) already exists from group_frames;
    # here we add the mean-pooling variant over the encoder outputs.
    for seed in (42, 43, 44):
        run(
            [PY, TRAINER, "--experiment", "R4_MEAN", "--seeds", str(seed),
             "--frames", "32", "--tag", "abpoolmean"],
            f"pool_mean_s{seed}",
        )


def group_calib() -> None:
    for exp in ("R5", "R6"):
        for seed in (42, 43, 44):
            run(
                [PY, TRAINER, "--experiment", exp, "--seeds", str(seed),
                 "--frames", "32", "--tag", "abcalib"],
                f"calib_{exp}_s{seed}",
            )


def group_topk() -> None:
    for frac in (0.1, 0.2, 0.3):
        run(
            [PY, TRAINER, "--experiment", "R1", "--seeds", "42",
             "--frames", "32", "--top-frac", str(frac), "--tag", f"abtop{int(frac*100)}"],
            f"topk_{int(frac*100)}",
        )


def group_scale() -> None:
    for n in (500, 1000, 2000, 4000):
        for seed in (42, 43, 44):
            run(
                [PY, TRAINER, "--experiment", "R4", "--seeds", str(seed),
                 "--frames", "32", "--train-limit", str(n),
                 "--tag", f"abscale{n}"],
                f"scale_{n}_s{seed}",
            )


def group_stat() -> None:
    """Paired bootstrap R5 vs R6 and TEST-500 vs TEST-1277 distribution check."""
    import pandas as pd

    out = {}

    def load(tag: str, seed: int) -> dict:
        cands = [
            RESULTS / f"{tag}_s{seed}_f32_cls.json",
            RESULTS / f"{tag}_s{seed}_f32_cls_a10_20.json",
        ]
        for p in cands:
            if p.exists():
                return json.loads(p.read_text(encoding="utf-8"))
        return {}

    pairs = []
    for seed in (42, 43, 44):
        a = load("R5", seed)
        b = load("R6", seed)
        if not a or not b:
            continue
        pa = np.array(a["predictions"]["test_pred"])
        pb = np.array(b["predictions"]["test_pred"])
        y = np.array(a["predictions"]["test_true"])
        pairs.append((y, pa, pb))
    if not pairs:
        print("stat: R5/R6 predictions missing; run group_calib first", flush=True)
        return

    def paired_boot(pairs, seed=42):
        rng = np.random.default_rng(seed)
        diffs = []
        for _ in range(1000):
            per_seed = []
            for y, pa, pb in pairs:
                idx = rng.integers(0, len(y), len(y))
                per_seed.append(
                    float(np.mean(np.abs(pa[idx] - y[idx])) - np.mean(np.abs(pb[idx] - y[idx])))
                )
            diffs.append(float(np.mean(per_seed)))
        d = np.array(diffs)
        return {
            "mean": float(d.mean()),
            "lo": float(np.percentile(d, 2.5)),
            "hi": float(np.percentile(d, 97.5)),
            "n_boot": 1000,
        }

    r56 = paired_boot(pairs, 42)
    r56["note"] = "positive value means R5 has larger MAE than R6"
    out["r5_minus_r6_mae"] = r56

    # R2 (mean-pooled linear) vs R4 (temporal transformer): same seeds, same cache
    p24 = []
    for seed in (42, 43, 44):
        a, b = load("R2", seed), load("R4", seed)
        if not a or not b:
            continue
        p24.append((
            np.array(a["predictions"]["test_true"]),
            np.array(a["predictions"]["test_pred"]),
            np.array(b["predictions"]["test_pred"]),
        ))
    if p24:
        r24 = paired_boot(p24, 43)
        r24["note"] = "positive value means R2 has larger MAE than R4"
        out["r2_minus_r4_mae"] = r24

    # R4 vs R5: does the soft-label pretraining shift error at all
    p45 = []
    for seed in (42, 43, 44):
        a, b = load("R4", seed), load("R5", seed)
        if not a or not b:
            continue
        p45.append((
            np.array(a["predictions"]["test_true"]),
            np.array(a["predictions"]["test_pred"]),
            np.array(b["predictions"]["test_pred"]),
        ))
    if p45:
        r45 = paired_boot(p45, 44)
        r45["note"] = "positive value means R4 has larger MAE than R5"
        out["r4_minus_r5_mae"] = r45

    # EF distribution: TEST-500 vs full TEST
    fl = pd.read_csv(Path(r"E:\Datasets\EchoNet-Dynamic\FileList.csv"))
    fl["stem"] = fl["FileName"].astype(str).str.replace(r"\.avi$", "", regex=True)
    test_all = fl[fl["Split"] == "TEST"]
    sub = np.load(CACHE / "test500_32_bgr.npz", allow_pickle=False)
    sub_ids = set(str(x) for x in sub["ids"])
    sub_rows = test_all[test_all["stem"].isin(sub_ids)]
    from scipy import stats

    ks = stats.ks_2samp(sub_rows["EF"].values, test_all["EF"].values)
    out["ef_distribution"] = {
        "test_full_n": int(len(test_all)),
        "test_full_mean": float(test_all["EF"].mean()),
        "test_full_sd": float(test_all["EF"].std()),
        "test_full_lt50": float(np.mean(test_all["EF"].values < 50)),
        "test_full_lt40": float(np.mean(test_all["EF"].values < 40)),
        "test_full_lt30": float(np.mean(test_all["EF"].values < 30)),
        "test500_n": int(len(sub_rows)),
        "test500_mean": float(sub_rows["EF"].mean()),
        "test500_sd": float(sub_rows["EF"].std()),
        "test500_lt50": float(np.mean(sub_rows["EF"].values < 50)),
        "test500_lt40": float(np.mean(sub_rows["EF"].values < 40)),
        "test500_lt30": float(np.mean(sub_rows["EF"].values < 30)),
        "ks_stat": float(ks.statistic),
        "ks_p": float(ks.pvalue),
        "coverage_of_full_test": float(len(sub_rows) / len(test_all)),
    }
    (RESULTS / "stat_robustness.json").write_text(
        json.dumps(out, indent=1, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(out, indent=1), flush=True)


GROUPS = {
    "frames": group_frames,
    "pool": group_pool,
    "calib": group_calib,
    "topk": group_topk,
    "scale": group_scale,
    "stat": group_stat,
}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--group", default="all")
    args = ap.parse_args()
    names = list(GROUPS) if args.group == "all" else args.group.split(",")
    for n in names:
        print(f"### group {n}", flush=True)
        GROUPS[n]()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
