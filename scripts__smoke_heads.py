"""Smoke test of the head architectures and calibration code on synthetic data.

Runs in a few seconds and does not touch the real caches. Verifies that the
R2/R3/R4/R5 code paths, the contrastive pretraining loop, the calibration block
and the metric functions execute and return finite numbers.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT))

import train_heads_from_cache as th  # noqa: E402

rng = np.random.default_rng(0)
N_TR, N_VA, N_TE, T, D = 160, 60, 60, 8, 512


def make(n, T):
    emb = rng.normal(size=(n, T, D)).astype(np.float32)
    emb /= np.linalg.norm(emb, axis=-1, keepdims=True)
    # EF is a weak linear function of the mean embedding: learnable signal
    w = rng.normal(size=D)
    ef = 55 + 10 * (emb.mean(1) @ w) / np.linalg.norm(w) + rng.normal(scale=3, size=n)
    return emb, np.clip(ef, 5, 95).astype(np.float64)


xtr, ytr = make(N_TR, T)
xva, yva = make(N_VA, T)
xte, yte = make(N_TE, T)

print("--- R2 linear ---")
head = th.LinearRegressor(dim=D)
head, state = th.fit_regression(head, xtr.mean(1), ytr, xva.mean(1), yva, epochs=40, device="cpu")
p = th.predict_regression(head, xte.mean(1), state)
print("  MAE", round(float(np.mean(np.abs(p - yte))), 3))

print("--- R3 MLP ---")
head = th.MLPRegressor(dim=D)
head, state = th.fit_regression(head, xtr.mean(1), ytr, xva.mean(1), yva, epochs=40, device="cpu")
p3 = th.predict_regression(head, xte.mean(1), state)
print("  MAE", round(float(np.mean(np.abs(p3 - yte))), 3))

print("--- R4 temporal ---")
head = th.TemporalRegressor(dim=D, n_layers=2, n_heads=8, max_frames=16)
head, state = th.fit_regression(head, xtr, ytr, xva, yva, epochs=40, device="cpu")
p4 = th.predict_regression(head, xte, state)
print("  MAE", round(float(np.mean(np.abs(p4 - yte))), 3))

print("--- R4 mean pooling ---")
head = th.TemporalRegressor(dim=D, n_layers=2, n_heads=8, max_frames=16, pool="mean")
head, state = th.fit_regression(head, xtr, ytr, xva, yva, epochs=40, device="cpu")
pm = th.predict_regression(head, xte, state)
print("  MAE", round(float(np.mean(np.abs(pm - yte))), 3))

print("--- R5 contrastive pretrain + finetune ---")
head = th.TemporalRegressor(dim=D, n_layers=2, n_heads=8, max_frames=16)
head, curve = th.pretrain_contrastive(head, xtr, ytr, epochs=10, batch_size=32, device="cpu")
print("  pretrain curve", [round(c, 3) for c in curve[:3]], "...", round(curve[-1], 3))
head, state = th.fit_regression(head, xtr, ytr, xva, yva, epochs=40, device="cpu")
p5 = th.predict_regression(head, xte, state)
print("  MAE", round(float(np.mean(np.abs(p5 - yte))), 3))

print("--- calibration block ---")
cal = th.calibrate_block(yva, th.predict_regression(head, xva, state), yte, p5)
for thr, rec in cal["thresholds"].items():
    print(f"  {thr}: ece raw {rec['test']['raw']['ece_15']:.3f} -> platt "
          f"{rec['test']['platt']['ece_15']:.3f}, T={rec['temperature']:.3f}")
for a, rec in cal["conformal"].items():
    print(f"  {a}: coverage {rec['coverage']:.3f} target {rec['target_coverage']:.2f} "
          f"width {rec['width_mean']:.3f}")

print("--- metrics ---")
m = th.regression_metrics(yte, p5)
print("  ", {k: (round(v, 3) if isinstance(v, float) else v) for k, v in m.items()})
print("--- bootstrap ---")
print("  ", th.bootstrap_ci(yte, p5, "mae", n_boot=200))
print("OK")
