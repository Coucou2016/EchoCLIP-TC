"""Clinical metrics for EchoCLIP-TC (EF regression + threshold AUCs).

These — not in-batch retrieval R@k — are the paper primary numbers.
Demo manifests without an ``ef`` field can parse EF from report text; that
path is labelled and is not a clinical result.
"""

import re
from typing import Dict, Optional, Sequence, Tuple, Union

import numpy as np
import torch

from echoclip.calibrate import (
    abstain_by_probability,
    apply_abstention,
    brier_score,
    conformal_coverage,
    conformal_intervals,
    expected_calibration_error,
    fit_temperature,
    sigmoid,
    split_conformal_quantile,
)

ArrayLike = Union[np.ndarray, Sequence[float], torch.Tensor]

EF_THRESHOLDS = (50, 40, 30)

_EF_FROM_TEXT = re.compile(
    r"(?:EJECTION FRACTION|LVEF)[^\d]{0,48}(\d+(?:\.\d+)?)",
    re.IGNORECASE,
)


def _as_numpy(x: ArrayLike) -> np.ndarray:
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x, dtype=np.float64)


def parse_ef_from_text(text: str) -> Optional[float]:
    """Best-effort EF parse for demo manifests. Not clinical ground truth."""
    if not text:
        return None
    match = _EF_FROM_TEXT.search(text)
    if not match:
        return None
    value = float(match.group(1))
    if value < 5 or value > 90:
        return None
    return value


def regression_metrics(y_true: ArrayLike, y_pred: ArrayLike) -> Dict[str, float]:
    y = _as_numpy(y_true).reshape(-1).astype(np.float64)
    p = _as_numpy(y_pred).reshape(-1).astype(np.float64)
    if y.size == 0:
        return {"mae": float("nan"), "rmse": float("nan"), "r2": float("nan"), "n": 0}
    err = p - y
    mae = float(np.mean(np.abs(err)))
    rmse = float(np.sqrt(np.mean(err ** 2)))
    ss_res = float(np.sum(err ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    r2 = float("nan") if ss_tot <= 0 else float(1.0 - ss_res / ss_tot)
    return {"mae": mae, "rmse": rmse, "r2": r2, "n": int(y.size)}


def roc_auc(y_true: ArrayLike, y_score: ArrayLike) -> float:
    """ROC-AUC via Mann–Whitney (no sklearn). Returns NaN if one class is missing."""
    y = _as_numpy(y_true).reshape(-1).astype(np.int64)
    s = _as_numpy(y_score).reshape(-1).astype(np.float64)
    n_pos = int((y == 1).sum())
    n_neg = int((y == 0).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    order = np.argsort(s)
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(1, y.size + 1, dtype=np.float64)
    # Average ranks for ties
    _, start_idx, counts = np.unique(s, return_index=True, return_counts=True)
    if np.any(counts > 1):
        sorted_s = s[order]
        i = 0
        while i < y.size:
            j = i
            while j + 1 < y.size and sorted_s[j + 1] == sorted_s[i]:
                j += 1
            if j > i:
                avg = 0.5 * (i + 1 + j + 1)
                ranks[order[i : j + 1]] = avg
            i = j + 1
    sum_pos_ranks = float(ranks[y == 1].sum())
    auc = (sum_pos_ranks - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)
    return float(auc)


def threshold_aucs(
    y_true_ef: ArrayLike,
    y_pred_ef: ArrayLike,
    thresholds: Sequence[int] = EF_THRESHOLDS,
) -> Dict[str, float]:
    """AUC for EF < t. Score is -predicted EF (lower EF → more likely reduced)."""
    y = _as_numpy(y_true_ef).reshape(-1)
    p = _as_numpy(y_pred_ef).reshape(-1)
    out: Dict[str, float] = {}
    for t in thresholds:
        y_bin = (y < float(t)).astype(np.int64)
        out[f"auc_ef_lt_{int(t)}"] = roc_auc(y_bin, -p)
    return out


def bootstrap_mae_ci(
    y_true: ArrayLike,
    y_pred: ArrayLike,
    n_boot: int = 1000,
    seed: int = 42,
    alpha: float = 0.05,
) -> Tuple[float, float]:
    y = _as_numpy(y_true).reshape(-1)
    p = _as_numpy(y_pred).reshape(-1)
    n = y.size
    if n == 0:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    maes = np.empty(n_boot, dtype=np.float64)
    for i in range(n_boot):
        idx = rng.integers(0, n, size=n)
        maes[i] = np.mean(np.abs(p[idx] - y[idx]))
    lo = float(np.quantile(maes, alpha / 2.0))
    hi = float(np.quantile(maes, 1.0 - alpha / 2.0))
    return lo, hi


def ef_threshold_logits(y_pred_ef: ArrayLike, threshold: float) -> np.ndarray:
    """Logit for P(EF < threshold): (threshold - pred). Uncalibrated."""
    p = _as_numpy(y_pred_ef).reshape(-1).astype(np.float64)
    return float(threshold) - p


def summarize_clinical(
    y_true: ArrayLike,
    y_pred: ArrayLike,
    *,
    cal_true: Optional[ArrayLike] = None,
    cal_pred: Optional[ArrayLike] = None,
    thresholds: Sequence[int] = EF_THRESHOLDS,
    conformal_alpha: float = 0.1,
    n_boot: int = 1000,
    seed: int = 42,
    abstain_width_quantile: float = 0.8,
) -> Dict:
    """Regression + threshold AUC + optional val-fitted calibration/conformal."""
    y = _as_numpy(y_true).reshape(-1)
    p = _as_numpy(y_pred).reshape(-1)
    metrics = regression_metrics(y, p)
    metrics.update(threshold_aucs(y, p, thresholds=thresholds))
    lo, hi = bootstrap_mae_ci(y, p, n_boot=n_boot, seed=seed)
    metrics["mae_bootstrap_ci95"] = [lo, hi]

    # Uncalibrated P(EF < 50) from (50 - pred) via sigmoid
    primary_t = 50
    test_logits = ef_threshold_logits(p, primary_t)
    test_labels = (y < primary_t).astype(np.float64)
    temperature = 1.0
    conformal_q = None

    if cal_true is not None and cal_pred is not None:
        cy = _as_numpy(cal_true).reshape(-1)
        cp = _as_numpy(cal_pred).reshape(-1)
        cal_logits = ef_threshold_logits(cp, primary_t)
        cal_labels = (cy < primary_t).astype(np.float64)
        if cy.size >= 2 and cal_labels.min() != cal_labels.max():
            temperature = fit_temperature(cal_logits, cal_labels)
        residuals = np.abs(cp - cy)
        if residuals.size:
            conformal_q = split_conformal_quantile(residuals, alpha=conformal_alpha)
            metrics["conformal_fitted_on"] = "calibration_split"
        metrics["n_calibration"] = int(cy.size)

    probs = sigmoid(test_logits / max(temperature, 1e-6))
    metrics["temperature_ef_lt_50"] = float(temperature)
    metrics["ece_ef_lt_50"] = expected_calibration_error(probs, test_labels)
    metrics["brier_ef_lt_50"] = brier_score(probs, test_labels)

    if conformal_q is not None and np.isfinite(conformal_q):
        intervals = conformal_intervals(p, conformal_q)
        metrics["conformal_alpha"] = float(conformal_alpha)
        metrics["conformal_quantile"] = float(conformal_q)
        metrics["conformal_coverage"] = conformal_coverage(y, intervals)
        widths = intervals[:, 1] - intervals[:, 0]
        metrics["conformal_mean_width"] = float(np.mean(widths))
        # Fixed-width split conformal → all widths identical; width-quantile
        # abstention never fires. Fall back to probability confidence.
        if widths.size and float(np.ptp(widths)) < 1e-12:
            abstain = abstain_by_probability(probs, min_confidence=0.7)
            metrics["abstention_rule"] = "probability_confidence"
            metrics["abstention_min_confidence"] = 0.7
        else:
            thresh = float(np.quantile(widths, abstain_width_quantile))
            abstain = widths > thresh
            metrics["abstention_rule"] = "interval_width"
            metrics["abstention_width_quantile"] = float(abstain_width_quantile)
        abs_stats = apply_abstention(y, p, abstain)
        metrics["abstention_coverage"] = abs_stats["coverage"]
        metrics["abstention_mae"] = abs_stats["mae"]
        metrics["abstention_n_keep"] = abs_stats["n_keep"]

    return metrics
