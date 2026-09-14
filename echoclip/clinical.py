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

    PositiveScaleHead,

    abstain_by_probability,

    adaptive_conformal_intervals,

    apply_abstention,

    apply_affine_logistic,

    area_under_risk_coverage,

    brier_score,

    conformal_coverage,

    conformal_intervals,

    expected_calibration_error,

    fit_affine_logistic,

    fit_temperature,

    heuristic_uncertainty_scale,

    normalized_residuals,

    risk_coverage_curve,

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





def bootstrap_metric_ci(
    y_true: ArrayLike,
    y_pred: ArrayLike,
    metric_fn,
    n_boot: int = 1000,
    seed: int = 42,
    alpha: float = 0.05,
    *,
    stratified: bool = False,
    stratify_labels: Optional[ArrayLike] = None,
) -> Tuple[float, float, float]:
    """Bootstrap CI for a scalar metric_fn(y, p) → float. Returns (point, lo, hi).

    NaN replicate metrics are filtered before quantiles. When ``stratified=True``
    and ``stratify_labels`` is a binary vector (same length), each bootstrap
    draw preserves class counts (useful for AUC).
    """
    y = _as_numpy(y_true).reshape(-1)
    p = _as_numpy(y_pred).reshape(-1)
    n = y.size
    if n == 0:
        return float("nan"), float("nan"), float("nan")

    point = float(metric_fn(y, p))
    rng = np.random.default_rng(seed)
    vals = []

    strat = None
    if stratified and stratify_labels is not None:
        strat = _as_numpy(stratify_labels).reshape(-1).astype(np.int64)
        if strat.size != n:
            strat = None

    for _ in range(n_boot):
        if strat is not None and set(np.unique(strat).tolist()) == {0, 1}:
            pos = np.where(strat == 1)[0]
            neg = np.where(strat == 0)[0]
            if pos.size == 0 or neg.size == 0:
                idx = rng.integers(0, n, size=n)
            else:
                idx = np.concatenate(
                    [
                        rng.choice(pos, size=pos.size, replace=True),
                        rng.choice(neg, size=neg.size, replace=True),
                    ]
                )
                rng.shuffle(idx)
        else:
            idx = rng.integers(0, n, size=n)
        v = float(metric_fn(y[idx], p[idx]))
        if np.isfinite(v):
            vals.append(v)

    if not vals:
        return point, float("nan"), float("nan")
    arr = np.asarray(vals, dtype=np.float64)
    lo = float(np.quantile(arr, alpha / 2.0))
    hi = float(np.quantile(arr, 1.0 - alpha / 2.0))
    return point, lo, hi





def bootstrap_mae_ci(

    y_true: ArrayLike,

    y_pred: ArrayLike,

    n_boot: int = 1000,

    seed: int = 42,

    alpha: float = 0.05,

) -> Tuple[float, float]:

    _, lo, hi = bootstrap_metric_ci(

        y_true,

        y_pred,

        lambda yt, yp: float(np.mean(np.abs(yp - yt))),

        n_boot=n_boot,

        seed=seed,

        alpha=alpha,

    )

    return lo, hi





def paired_bootstrap_delta_mae(

    y_true: ArrayLike,

    y_pred_a: ArrayLike,

    y_pred_b: ArrayLike,

    n_boot: int = 1000,

    seed: int = 42,

    alpha: float = 0.05,

) -> Dict[str, float]:

    """Paired bootstrap CI for ΔMAE = MAE(A) - MAE(B) on the same labels.



    Negative Δ favors method A. Same indices resampled for both predictions.

    """

    y = _as_numpy(y_true).reshape(-1)

    a = _as_numpy(y_pred_a).reshape(-1)

    b = _as_numpy(y_pred_b).reshape(-1)

    if y.size == 0 or a.size != y.size or b.size != y.size:

        return {

            "delta_mae": float("nan"),

            "delta_mae_ci95_lo": float("nan"),

            "delta_mae_ci95_hi": float("nan"),

            "n": 0,

        }

    n = y.size

    delta = float(np.mean(np.abs(a - y)) - np.mean(np.abs(b - y)))

    rng = np.random.default_rng(seed)

    boots = np.empty(n_boot, dtype=np.float64)

    for i in range(n_boot):

        idx = rng.integers(0, n, size=n)

        mae_a = float(np.mean(np.abs(a[idx] - y[idx])))

        mae_b = float(np.mean(np.abs(b[idx] - y[idx])))

        boots[i] = mae_a - mae_b

    return {

        "delta_mae": delta,

        "delta_mae_ci95_lo": float(np.quantile(boots, alpha / 2.0)),

        "delta_mae_ci95_hi": float(np.quantile(boots, 1.0 - alpha / 2.0)),

        "n": int(n),

    }





def _rmse(y: np.ndarray, p: np.ndarray) -> float:

    return float(np.sqrt(np.mean((p - y) ** 2)))





def _r2(y: np.ndarray, p: np.ndarray) -> float:

    ss_res = float(np.sum((p - y) ** 2))

    ss_tot = float(np.sum((y - y.mean()) ** 2))

    return float("nan") if ss_tot <= 0 else float(1.0 - ss_res / ss_tot)





def _auc_lt(threshold: float):

    def _fn(y: np.ndarray, p: np.ndarray) -> float:

        y_bin = (y < float(threshold)).astype(np.int64)

        return roc_auc(y_bin, -p)



    return _fn





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

    calibration_method: str = "temperature",

    adaptive_conformal: bool = False,

    adaptive_scale: str = "heuristic",

) -> Dict:

    """Regression + threshold AUC + optional val-fitted calibration/conformal.



    ``calibration_method``:

      - ``temperature``: scale pseudo-logit ``(t - pred)`` by T (legacy; limited

        because scores are not true classifier logits — see PAPER.md).

      - ``affine_logistic``: fit σ(a·score + b) on VAL for EF<50/40/30.



    ``adaptive_conformal``: if True, also fit normalized residual conformal

    ``|y-ŷ|/s(x)`` (variable width) and report risk–coverage / AURC. Basic

    fixed-width split conformal remains the default primary interval.

    """

    y = _as_numpy(y_true).reshape(-1)

    p = _as_numpy(y_pred).reshape(-1)

    metrics = regression_metrics(y, p)

    metrics.update(threshold_aucs(y, p, thresholds=thresholds))

    lo, hi = bootstrap_mae_ci(y, p, n_boot=n_boot, seed=seed)

    metrics["mae_bootstrap_ci95"] = [lo, hi]



    # Bootstrap CIs for RMSE, R², and threshold AUCs

    _, rmse_lo, rmse_hi = bootstrap_metric_ci(y, p, _rmse, n_boot=n_boot, seed=seed + 1)

    _, r2_lo, r2_hi = bootstrap_metric_ci(y, p, _r2, n_boot=n_boot, seed=seed + 2)

    metrics["rmse_bootstrap_ci95"] = [rmse_lo, rmse_hi]

    metrics["r2_bootstrap_ci95"] = [r2_lo, r2_hi]

    for t in thresholds:

        y_bin = (y < float(t)).astype(np.int64)

        _, a_lo, a_hi = bootstrap_metric_ci(

            y,
            p,
            _auc_lt(float(t)),
            n_boot=n_boot,
            seed=seed + int(t),
            stratified=True,
            stratify_labels=y_bin,
        )

        metrics[f"auc_ef_lt_{int(t)}_bootstrap_ci95"] = [a_lo, a_hi]



    # Uncalibrated P(EF < 50) from (50 - pred) via sigmoid — pseudo-logit limitation.

    primary_t = 50

    test_logits = ef_threshold_logits(p, primary_t)

    test_labels = (y < primary_t).astype(np.float64)

    temperature = 1.0

    affine_a, affine_b = 1.0, 0.0

    conformal_q = None

    method = str(calibration_method).strip().lower()

    metrics["calibration_score_note"] = (

        "EF threshold scores use pseudo-logit (threshold - pred_EF), not true "

        "classifier logits; temperature scaling is limited. Prefer affine_logistic."

    )

    # Per-threshold calibration params / metrics (50 primary + 40/30)

    thresh_temps: Dict[int, float] = {int(t): 1.0 for t in thresholds}

    thresh_affine: Dict[int, Tuple[float, float]] = {

        int(t): (1.0, 0.0) for t in thresholds

    }



    if cal_true is not None and cal_pred is not None:

        cy = _as_numpy(cal_true).reshape(-1)

        cp = _as_numpy(cal_pred).reshape(-1)

        for t in thresholds:

            cal_logits_t = ef_threshold_logits(cp, float(t))

            cal_labels_t = (cy < float(t)).astype(np.float64)

            if cy.size >= 2 and cal_labels_t.min() != cal_labels_t.max():

                if method in ("affine", "affine_logistic", "logistic"):

                    a_t, b_t = fit_affine_logistic(cal_logits_t, cal_labels_t)

                    thresh_affine[int(t)] = (a_t, b_t)

                    metrics[f"affine_a_ef_lt_{int(t)}"] = float(a_t)

                    metrics[f"affine_b_ef_lt_{int(t)}"] = float(b_t)

                else:

                    t_fit = fit_temperature(cal_logits_t, cal_labels_t)

                    thresh_temps[int(t)] = float(t_fit)

                    metrics[f"temperature_ef_lt_{int(t)}"] = float(t_fit)

        if method in ("affine", "affine_logistic", "logistic"):

            metrics["calibration_method"] = "affine_logistic"

            affine_a, affine_b = thresh_affine[primary_t]

        else:

            metrics["calibration_method"] = "temperature"

            temperature = thresh_temps[primary_t]

        residuals = np.abs(cp - cy)

        if residuals.size:

            conformal_q = split_conformal_quantile(residuals, alpha=conformal_alpha)

            metrics["conformal_fitted_on"] = "calibration_split"

            metrics["conformal_width_note"] = (

                "Basic split conformal uses a fixed absolute residual quantile → "

                "constant interval width; width-based abstention is vacuous. "

                "Enable adaptive_conformal for |y-ŷ|/s(x) variable-width intervals."

            )

        metrics["n_calibration"] = int(cy.size)



    # Report ECE/Brier at 50/40/30 with the chosen calibration method

    for t in thresholds:

        logits_t = ef_threshold_logits(p, float(t))

        labels_t = (y < float(t)).astype(np.float64)

        if method in ("affine", "affine_logistic", "logistic"):

            a_t, b_t = thresh_affine[int(t)]

            probs_t = apply_affine_logistic(logits_t, a_t, b_t)

            if f"affine_a_ef_lt_{int(t)}" not in metrics:

                metrics[f"affine_a_ef_lt_{int(t)}"] = float(a_t)

                metrics[f"affine_b_ef_lt_{int(t)}"] = float(b_t)

        else:

            t_scale = thresh_temps[int(t)]

            probs_t = sigmoid(logits_t / max(t_scale, 1e-6))

            metrics[f"temperature_ef_lt_{int(t)}"] = float(t_scale)

        metrics[f"ece_ef_lt_{int(t)}"] = expected_calibration_error(probs_t, labels_t)

        metrics[f"brier_ef_lt_{int(t)}"] = brier_score(probs_t, labels_t)



    if method in ("affine", "affine_logistic", "logistic") and (

        "affine_a_ef_lt_50" in metrics or (cal_true is not None)

    ):

        probs = apply_affine_logistic(test_logits, affine_a, affine_b)

        metrics["temperature_ef_lt_50"] = float(thresh_temps.get(50, temperature))

    else:

        probs = sigmoid(test_logits / max(temperature, 1e-6))

        metrics["temperature_ef_lt_50"] = float(temperature)



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



    # Optional adaptive / normalized conformal + risk–coverage / AURC

    if adaptive_conformal and cal_true is not None and cal_pred is not None:

        cy = _as_numpy(cal_true).reshape(-1)

        cp = _as_numpy(cal_pred).reshape(-1)

        scale_mode = str(adaptive_scale).strip().lower()

        if scale_mode in ("head", "learned", "positive_scale"):

            head = PositiveScaleHead.fit(cy, cp)

            s_cal = head.scale(cp)

            s_te = head.scale(p)

            metrics["adaptive_scale"] = "positive_scale_head"

            metrics["adaptive_scale_a"] = float(head.a)

            metrics["adaptive_scale_b"] = float(head.b)

        else:

            s_cal = heuristic_uncertainty_scale(cp, cal_true=cy, cal_pred=cp)

            s_te = heuristic_uncertainty_scale(p, cal_true=cy, cal_pred=cp)

            metrics["adaptive_scale"] = "heuristic"

        norm_r = normalized_residuals(cy, cp, s_cal)

        aq = split_conformal_quantile(norm_r, alpha=conformal_alpha)

        a_intervals = adaptive_conformal_intervals(p, s_te, aq)

        metrics["adaptive_conformal_quantile"] = float(aq)

        metrics["adaptive_conformal_coverage"] = conformal_coverage(y, a_intervals)

        a_widths = a_intervals[:, 1] - a_intervals[:, 0]

        metrics["adaptive_conformal_mean_width"] = float(np.mean(a_widths))

        metrics["adaptive_conformal_width_std"] = float(np.std(a_widths))

        # Uncertainty for selective prediction = interval half-width (= q * s)

        cov_c, risk_c = risk_coverage_curve(y, p, a_widths)

        metrics["risk_coverage_coverage"] = cov_c.tolist()

        metrics["risk_coverage_mae"] = risk_c.tolist()

        metrics["aurc"] = area_under_risk_coverage(cov_c, risk_c)

        # Same-VAL scale+quantile fit → empirical heuristic (not fully split).
        metrics["adaptive_conformal_empirical_heuristic"] = True
        metrics["adaptive_conformal_note"] = (
            "s(x) and conformal quantile fit on the same VAL split; "
            "label as empirical heuristic unless VAL-scale vs VAL-cal are split."
        )



    return metrics


