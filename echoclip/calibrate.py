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

    Note: for EchoCLIP EF threshold scores, ``logits`` are often *pseudo-logits*
    ``(threshold - predicted_EF)``, not calibrated classifier logits. Prefer
    ``fit_affine_logistic`` when reporting calibrated P(EF < t).
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


def fit_affine_logistic(
    scores: ArrayLike,
    labels: ArrayLike,
    max_iter: int = 100,
) -> Tuple[float, float]:
    """Fit P(y=1) = σ(a * score + b) on VAL; returns (a, b).

    Intended for EF-threshold calibration where ``score`` is a pseudo-logit
    such as ``(threshold - pred_EF)``. Fit on VAL only; apply on TEST.
    """
    z = torch.as_tensor(_as_numpy(scores), dtype=torch.float64).reshape(-1)
    y = torch.as_tensor(_as_numpy(labels), dtype=torch.float64).reshape(-1)
    if z.numel() < 2 or y.min() == y.max():
        return 1.0, 0.0
    a = torch.nn.Parameter(torch.ones((), dtype=torch.float64))
    b = torch.nn.Parameter(torch.zeros((), dtype=torch.float64))
    opt = torch.optim.LBFGS([a, b], lr=0.25, max_iter=max_iter, line_search_fn="strong_wolfe")

    def closure():
        opt.zero_grad()
        logits = a * z + b
        loss = F.binary_cross_entropy_with_logits(logits, y.clamp(0.0, 1.0))
        loss.backward()
        return loss

    opt.step(closure)
    return float(a.detach()), float(b.detach())


def apply_affine_logistic(
    scores: ArrayLike, a: float, b: float
) -> np.ndarray:
    """σ(a * score + b)."""
    z = _as_numpy(scores).reshape(-1).astype(np.float64)
    return sigmoid(float(a) * z + float(b))


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


# ---------------------------------------------------------------------------
# Adaptive / normalized conformal (optional)
# Basic split conformal uses a *fixed* absolute residual quantile → constant
# interval width. Adaptive conformal uses score |y-ŷ|/s(x) so widths vary.
#
# Honesty: when s(x) and the conformal quantile are both estimated on the same
# VAL split (no VAL-scale vs VAL-cal holdout), treat results as an **empirical
# heuristic** — not a fully split conformal guarantee. Prefer splitting VAL
# when sample size allows; metrics should record this caveat.
# ---------------------------------------------------------------------------


def heuristic_uncertainty_scale(
    predictions: ArrayLike,
    *,
    cal_true: Optional[ArrayLike] = None,
    cal_pred: Optional[ArrayLike] = None,
    floor: float = 1.0,
) -> np.ndarray:
    """Positive per-sample scale s(x) for normalized residuals.

    Heuristic (no learned head): blend (i) distance of ŷ from mid-EF (harder
    extremes) with (ii) global VAL residual MAD when calibration preds exist.
    Always returns s(x) ≥ ``floor`` > 0.
    """
    p = _as_numpy(predictions).reshape(-1).astype(np.float64)
    # EF typically ~15–80; mid ~50. Larger |ŷ-50| → larger scale.
    extremity = 1.0 + np.abs(p - 50.0) / 50.0
    base = float(floor)
    if cal_true is not None and cal_pred is not None:
        cy = _as_numpy(cal_true).reshape(-1).astype(np.float64)
        cp = _as_numpy(cal_pred).reshape(-1).astype(np.float64)
        if cy.size:
            mad = float(np.median(np.abs(cp - cy)))
            base = max(base, mad if mad > 1e-6 else base)
    return np.maximum(base * extremity, floor)


class PositiveScaleHead:
    """Tiny positive scale head: s = softplus(a) * |ŷ - c| + softplus(b) + eps.

    Fit on VAL by minimizing NLL of a Laplace residual model with scale s(x),
    or fall back to heuristic if optimization fails / n too small.
    """

    def __init__(self, a: float = 0.0, b: float = 0.0, center: float = 50.0):
        self.a = float(a)
        self.b = float(b)
        self.center = float(center)

    def scale(self, predictions: ArrayLike) -> np.ndarray:
        p = _as_numpy(predictions).reshape(-1).astype(np.float64)
        sa = float(np.log1p(np.exp(self.a)))  # softplus
        sb = float(np.log1p(np.exp(self.b)))
        return np.maximum(sa * np.abs(p - self.center) + sb + 1e-3, 1e-3)

    @classmethod
    def fit(
        cls,
        cal_true: ArrayLike,
        cal_pred: ArrayLike,
        max_iter: int = 80,
    ) -> "PositiveScaleHead":
        y = torch.as_tensor(_as_numpy(cal_true), dtype=torch.float64).reshape(-1)
        p = torch.as_tensor(_as_numpy(cal_pred), dtype=torch.float64).reshape(-1)
        if y.numel() < 4:
            return cls()
        a = torch.nn.Parameter(torch.zeros((), dtype=torch.float64))
        b = torch.nn.Parameter(torch.zeros((), dtype=torch.float64))
        center = torch.tensor(50.0, dtype=torch.float64)
        opt = torch.optim.LBFGS([a, b], lr=0.25, max_iter=max_iter, line_search_fn="strong_wolfe")

        def closure():
            opt.zero_grad()
            sa = F.softplus(a)
            sb = F.softplus(b)
            s = sa * (p - center).abs() + sb + 1e-3
            # Laplace NLL ∝ log(s) + |r|/s
            loss = (torch.log(s) + (p - y).abs() / s).mean()
            loss.backward()
            return loss

        try:
            opt.step(closure)
        except RuntimeError:
            return cls()
        return cls(a=float(a.detach()), b=float(b.detach()), center=50.0)


def normalized_residuals(
    y_true: ArrayLike,
    y_pred: ArrayLike,
    scale: ArrayLike,
) -> np.ndarray:
    y = _as_numpy(y_true).reshape(-1).astype(np.float64)
    p = _as_numpy(y_pred).reshape(-1).astype(np.float64)
    s = np.maximum(_as_numpy(scale).reshape(-1).astype(np.float64), 1e-6)
    return np.abs(y - p) / s


def adaptive_conformal_intervals(
    predictions: ArrayLike,
    scale: ArrayLike,
    quantile: float,
) -> np.ndarray:
    """ŷ ± q · s(x) — variable-width intervals from normalized conformal."""
    pred = _as_numpy(predictions).reshape(-1).astype(np.float64)
    s = np.maximum(_as_numpy(scale).reshape(-1).astype(np.float64), 1e-6)
    q = float(quantile)
    half = q * s
    return np.stack([pred - half, pred + half], axis=1)


def risk_coverage_curve(
    y_true: ArrayLike,
    y_pred: ArrayLike,
    uncertainty: ArrayLike,
    n_levels: int = 20,
) -> Tuple[np.ndarray, np.ndarray]:
    """Selective prediction: keep lowest-uncertainty fraction → (coverage, MAE).

    ``uncertainty`` high = less confident. Coverage goes from ~1/n to 1.
    """
    y = _as_numpy(y_true).reshape(-1).astype(np.float64)
    p = _as_numpy(y_pred).reshape(-1).astype(np.float64)
    u = _as_numpy(uncertainty).reshape(-1).astype(np.float64)
    n = int(y.size)
    if n == 0:
        return np.array([]), np.array([])
    order = np.argsort(u)  # most confident first
    coverages, risks = [], []
    for k in range(1, n_levels + 1):
        frac = k / float(n_levels)
        n_keep = max(1, int(np.ceil(frac * n)))
        idx = order[:n_keep]
        coverages.append(n_keep / float(n))
        risks.append(float(np.mean(np.abs(p[idx] - y[idx]))))
    return np.asarray(coverages), np.asarray(risks)


def area_under_risk_coverage(
    coverages: ArrayLike,
    risks: ArrayLike,
) -> float:
    """AURC via trapezoid rule on the risk–coverage curve (lower is better)."""
    c = _as_numpy(coverages).reshape(-1).astype(np.float64)
    r = _as_numpy(risks).reshape(-1).astype(np.float64)
    if c.size < 2:
        return float("nan")
    order = np.argsort(c)
    try:
        return float(np.trapezoid(r[order], c[order]))
    except AttributeError:
        return float(np.trapz(r[order], c[order]))
