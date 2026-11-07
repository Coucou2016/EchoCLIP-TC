"""Small metric helpers shared by the figure scripts."""

from __future__ import annotations

import numpy as np


def reliability_bins(y: np.ndarray, p: np.ndarray, bins: int = 10):
    """Return (bin center, observed frequency, count) for a reliability diagram."""
    p = np.asarray(p, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    edges = np.linspace(0.0, 1.0, bins + 1)
    cx, cy, cnt = [], [], []
    for i in range(bins):
        m = (p >= edges[i]) & (p < edges[i + 1]) if i < bins - 1 else (p >= edges[i]) & (p <= edges[i + 1])
        cx.append((edges[i] + edges[i + 1]) / 2.0)
        cy.append(float(y[m].mean()) if m.any() else float("nan"))
        cnt.append(int(m.sum()))
    return np.array(cx), np.array(cy), np.array(cnt)
