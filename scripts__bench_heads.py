"""Benchmark one R4 training epoch on CPU vs GPU for the real tensor shapes."""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT))

import torch  # noqa: E402

import train_heads_from_cache as th  # noqa: E402

N, T, D = 2000, 32, 512
x = torch.randn(N, T, D)
y = torch.randn(N)

for device in ("cpu", "cuda"):
    if device == "cuda" and not torch.cuda.is_available():
        continue
    for bs in (64, 128):
        head = th.TemporalRegressor(dim=D, n_layers=2, n_heads=8, max_frames=32).to(device)
        opt = torch.optim.AdamW(head.parameters(), lr=1e-3)
        xb_all = x.to(device)
        yb_all = y.to(device)
        # warm up
        for _ in range(2):
            idx = torch.randint(0, N, (bs,))
            loss = torch.nn.functional.smooth_l1_loss(head(xb_all[idx]), yb_all[idx])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
        if device == "cuda":
            torch.cuda.synchronize()
        steps = 0
        t0 = time.time()
        while time.time() - t0 < 6.0:
            idx = torch.randint(0, N, (bs,))
            loss = torch.nn.functional.smooth_l1_loss(head(xb_all[idx]), yb_all[idx])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            steps += 1
        if device == "cuda":
            torch.cuda.synchronize()
        dt = time.time() - t0
        per_epoch = dt / steps * (N / bs)
        print(
            f"{device} bs={bs}: {steps} steps in {dt:.1f}s -> {dt/steps*1000:.1f} ms/step "
            f"-> {per_epoch:.2f} s/epoch -> 120 epochs = {per_epoch*120/60:.1f} min",
            flush=True,
        )
