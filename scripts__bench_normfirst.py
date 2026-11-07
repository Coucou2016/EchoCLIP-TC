"""Compare TransformerEncoder settings for head training throughput."""

from __future__ import annotations

import sys
import time
from pathlib import Path

import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

D, T = 512, 32
N, BS = 2000, 64


class Net(nn.Module):
    def __init__(self, norm_first: bool, n_layers: int = 2, ff: int = 4):
        super().__init__()
        self.cls = nn.Parameter(torch.zeros(1, 1, D))
        self.pos = nn.Parameter(torch.zeros(1, T + 1, D))
        layer = nn.TransformerEncoderLayer(
            d_model=D, nhead=8, dim_feedforward=D * ff, dropout=0.1,
            activation="gelu", batch_first=True, norm_first=norm_first,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=n_layers, enable_nested_tensor=False)
        self.ln = nn.LayerNorm(D)
        self.head = nn.Linear(D, 1)

    def forward(self, x):
        b, t, _ = x.shape
        tok = torch.cat([self.cls.expand(b, -1, -1), x], dim=1) + self.pos[:, : t + 1]
        z = self.encoder(tok)[:, 0]
        return self.head(self.ln(z)).squeeze(-1)


x_cpu = torch.randn(N, T, D)
y_cpu = torch.randn(N)

for device in ("cuda", "cpu"):
    if device == "cuda" and not torch.cuda.is_available():
        continue
    x = x_cpu.to(device)
    y = y_cpu.to(device)
    for norm_first in (True, False):
        head = Net(norm_first).to(device)
        opt = torch.optim.AdamW(head.parameters(), lr=1e-3)
        if device == "cuda":
            torch.cuda.synchronize()
        for _ in range(3):
            idx = torch.randint(0, N, (BS,))
            loss = torch.nn.functional.smooth_l1_loss(head(x[idx]), y[idx])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
        if device == "cuda":
            torch.cuda.synchronize()
        steps, t0 = 0, time.time()
        while time.time() - t0 < 5.0:
            idx = torch.randint(0, N, (BS,))
            loss = torch.nn.functional.smooth_l1_loss(head(x[idx]), y[idx])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            steps += 1
        if device == "cuda":
            torch.cuda.synchronize()
        dt = time.time() - t0
        per_epoch = dt / steps * (N / BS)
        print(
            f"{device:4s} norm_first={str(norm_first):5s} {dt/steps*1000:7.1f} ms/step "
            f"-> {per_epoch:6.2f} s/epoch -> 40 epochs {per_epoch*40/60:5.1f} min",
            flush=True,
        )
