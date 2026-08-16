"""Calibration, conformal intervals, and abstention for EchoCLIP-TC.

Temperature scaling and ECE/Brier apply to *binary* scores (e.g. P(EF < 50)).
Split conformal intervals apply to *regression* residuals (e.g. EF in %).
Fit temperature / quantiles on a validation split only; report test once.
"""

from typing import Dict, Optional, Sequence, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F

ArrayLike = Union[np.ndarray, Sequence[float], torch.Tensor]


def _as_numpy(x: ArrayLike) -> np.ndarray:
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def sigmoid(x: np.ndarray) -> np.ndarray:
    x = np.clip(x, -50.0, 50.0)
    return 1.0 / (1.0 + np.exp(-x))


def softmax_np(logits: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    z = logits / max(float(temperature), 1e-6)
    z = z - z.max(axis=-1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=-1, keepdims=True)


def temperature_scale_logits(
    logits: ArrayLike, temperature: float
) -> np.ndarray:
    """Divide logits (N, C) or (N,) by T > 0."""
    t = max(float(temperature), 1e-6)
    return _as_numpy(logits) / t


def fit_temperature(
    logits: ArrayLike,
    labels: ArrayLike,
    max_iter: int = 50,
    binary: Optional[bool] = None,
) -> float:
    """
    Minimize NLL of softmax(logits / T) (multiclass) or BCE (1-D logits).

    labels: class indices (N,) for multiclass, or {0,1} for binary.
    """
    z = torch.as_tensor(_as_numpy(logits), dtype=torch.float64)
    y = torch.as_tensor(_as_numpy(labels), dtype=torch.float64)
    if z.ndim == 1:
        is_binary = True
    elif binary is True or z.shape[-1] == 1:
        is_binary = True
        z = z.reshape(-1)
    else:
        is_binary = False

    log_t = torch.nn.Parameter(torch.zeros((), dtype=torch.float64))
    opt = torch.optim.LBFGS([log_t], lr=0.25, max_iter=max_iter, line_search_fn="strong_wolfe")

    def closure():
        opt.zero_grad()
        temperature = log_t.exp().clamp(1e-3, 100.0)
        scaled = z / temperature
        if is_binary:
            target = y.clamp(0.0, 1.0)
            loss = F.binary_cross_entropy_with_logits(scaled, target)
        else:
            loss = F.cross_entropy(scaled.float(), y.long())
        loss.backward()
        return loss

    opt.step(closure)
    return float(log_t.exp().clamp(1e-3, 100.0).detach())


def expected_calibration_error(
    probs: ArrayLike,
    labels: ArrayLike,
    n_bins: int = 15,
) -> float:
    """ECE for P(positive) vs binary labels. Uniform bins on [0, 1]."""
    p = _as_numpy(probs).reshape(-1).astype(np.float64)
    y = _as_numpy(labels).reshape(-1).astype(np.float64)
    if p.size == 0:
        return float("nan")
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    n = p.size
    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]
        if i == 0:
            mask = (p >= lo) & (p <= hi)
        else:
            mask = (p > lo) & (p <= hi)
        count = int(mask.sum())
        if count == 0:
            continue
        acc = float(y[mask].mean())
        conf = float(p[mask].mean())
        ece += (count / n) * abs(acc - conf)
    return float(ece)


def brier_score(probs: ArrayLike, labels: ArrayLike) -> float:
    p = _as_numpy(probs).reshape(-1).astype(np.float64)
    y = _as_numpy(labels).reshape(-1).astype(np.float64)
    if p.size == 0:
        return float("nan")
    return float(np.mean((p - y) ** 2))


def split_conformal_quantile(
    residuals: ArrayLike,
    alpha: float = 0.1,
) -> float:
    """
    Finite-sample split-conformal quantile of |y - yhat|.

    q_level = ceil((n+1)(1-alpha)) / n, then the corresponding residual quantile.
    """
    r = np.abs(_as_numpy(residuals).reshape(-1).astype(np.float64))
    n = int(r.size)
    if n == 0:
        return float("nan")
    alpha = float(alpha)
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must be in (0, 1)")
    q_level = min(1.0, np.ceil((n + 1) * (1.0 - alpha)) / n)
    try:
        return float(np.quantile(r, q_level, method="higher"))
    except TypeError:
        return float(np.quantile(r, q_level, interpolation="higher"))


def conformal_intervals(
    predictions: ArrayLike,
    quantile: float,
) -> np.ndarray:
    """Return (N, 2) array of [lo, hi] symmetric intervals."""
    pred = _as_numpy(predictions).reshape(-1).astype(np.float64)
    q = float(quantile)
    return np.stack([pred - q, pred + q], axis=1)


def conformal_coverage(
    y_true: ArrayLike,
    intervals: np.ndarray,
) -> float:
    y = _as_numpy(y_true).reshape(-1).astype(np.float64)
    lo, hi = intervals[:, 0], intervals[:, 1]
    return float(np.mean((y >= lo) & (y <= hi)))


def interval_widths(intervals: np.ndarray) -> np.ndarray:
    return (intervals[:, 1] - intervals[:, 0]).astype(np.float64)


def abstain_by_width(
    intervals: np.ndarray,
    max_width: Optional[float] = None,
    width_quantile: Optional[float] = None,
) -> np.ndarray:
    """
    True = abstain.

    Provide either an absolute ``max_width`` or ``width_quantile`` (abstain on
    the widest fraction, e.g. 0.8 keeps the narrowest 80%).
    """
    widths = interval_widths(intervals)
    if max_width is not None:
        return widths > float(max_width)
    if width_quantile is not None:
        thresh = float(np.quantile(widths, float(width_quantile)))
        return widths > thresh
    raise ValueError("Provide max_width or width_quantile")


def abstain_by_probability(
    probs: ArrayLike,
    min_confidence: float = 0.7,
) -> np.ndarray:
    """True = abstain when max(p, 1-p) < min_confidence."""
    p = _as_numpy(probs).reshape(-1).astype(np.float64)
    conf = np.maximum(p, 1.0 - p)
    return conf < float(min_confidence)


def apply_abstention(
    y_true: ArrayLike,
    y_pred: ArrayLike,
    abstain: np.ndarray,
) -> Dict[str, float]:
    """MAE / coverage on the non-abstained subset."""
    y = _as_numpy(y_true).reshape(-1).astype(np.float64)
    p = _as_numpy(y_pred).reshape(-1).astype(np.float64)
    keep = ~np.asarray(abstain, dtype=bool)
    n_keep = int(keep.sum())
    coverage = float(keep.mean()) if keep.size else 0.0
    if n_keep == 0:
        return {
            "n_keep": 0,
            "coverage": coverage,
            "mae": float("nan"),
            "note": "abstained on every sample",
        }
    mae = float(np.mean(np.abs(p[keep] - y[keep])))
    return {"n_keep": n_keep, "coverage": coverage, "mae": mae}


def reliability_table(
    probs: ArrayLike,
    labels: ArrayLike,
    n_bins: int = 10,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Bin centers, accuracy, confidence — for reliability diagrams."""
    p = _as_numpy(probs).reshape(-1).astype(np.float64)
    y = _as_numpy(labels).reshape(-1).astype(np.float64)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    centers, accs, confs = [], [], []
    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]
        mask = (p >= lo) & (p <= hi) if i == 0 else (p > lo) & (p <= hi)
        if mask.sum() == 0:
            continue
        centers.append(0.5 * (lo + hi))
        accs.append(float(y[mask].mean()))
        confs.append(float(p[mask].mean()))
    return np.array(centers), np.array(accs), np.array(confs)
