"""Tests for adaptive conformal, bootstrap CIs, and @40/@30 calibration."""

import unittest

import numpy as np

from echoclip.calibrate import (
    PositiveScaleHead,
    adaptive_conformal_intervals,
    area_under_risk_coverage,
    heuristic_uncertainty_scale,
    normalized_residuals,
    risk_coverage_curve,
    split_conformal_quantile,
)
from echoclip.clinical import (
    paired_bootstrap_delta_mae,
    roc_auc,
    stratified_bootstrap_auc,
    summarize_clinical,
)


class TestAdaptiveConformal(unittest.TestCase):
    def test_normalized_residuals_and_variable_width(self):
        rng = np.random.default_rng(1)
        y = rng.uniform(20, 70, size=100)
        p = y + rng.normal(0, 4, size=100)
        s = heuristic_uncertainty_scale(p, cal_true=y[:50], cal_pred=p[:50])
        self.assertTrue(np.all(s > 0))
        r = normalized_residuals(y[:50], p[:50], s[:50])
        q = split_conformal_quantile(r, alpha=0.1)
        intervals = adaptive_conformal_intervals(p[50:], s[50:], q)
        widths = intervals[:, 1] - intervals[:, 0]
        self.assertGreater(float(np.std(widths)), 1e-6)
        self.assertEqual(intervals.shape, (50, 2))

    def test_val_scale_val_cal_split_separates_roles(self):
        """P1-7: s(x) and the conformal quantile must not share the same labels."""
        rng = np.random.default_rng(21)
        y = rng.uniform(15, 75, size=200)
        p = y + rng.normal(0, 4, size=200)
        out = summarize_clinical(
            y[100:],
            p[100:],
            cal_true=y[:100],
            cal_pred=p[:100],
            n_boot=10,
            seed=0,
            adaptive_conformal=True,
            adaptive_val_split=0.5,
        )
        self.assertTrue(out["adaptive_val_split"])
        self.assertFalse(out["adaptive_conformal_empirical_heuristic"])
        self.assertGreater(out["adaptive_n_scale"], 0)
        self.assertGreater(out["adaptive_n_cal"], 0)
        self.assertNotIn("same VAL split", out["adaptive_conformal_note"])

    def test_small_cal_weakens_claim_wording(self):
        rng = np.random.default_rng(22)
        y = rng.uniform(15, 75, size=10)
        p = y + rng.normal(0, 3, size=10)
        out = summarize_clinical(
            y[5:],
            p[5:],
            cal_true=y[:5],
            cal_pred=p[:5],
            n_boot=5,
            seed=0,
            adaptive_conformal=True,
            adaptive_val_split=0.5,
        )
        self.assertFalse(out["adaptive_val_split"])
        self.assertTrue(out["adaptive_conformal_empirical_heuristic"])
        self.assertIn("empirical heuristic", out["adaptive_conformal_note"])

    def test_positive_scale_head_fit(self):
        rng = np.random.default_rng(2)
        y = rng.uniform(20, 70, size=80)
        p = y + rng.normal(0, 3, size=80)
        head = PositiveScaleHead.fit(y, p)
        s = head.scale(p)
        self.assertEqual(s.shape, (80,))
        self.assertTrue(np.all(s > 0))

    def test_aurc_finite(self):
        rng = np.random.default_rng(3)
        y = rng.uniform(20, 70, size=60)
        p = y + rng.normal(0, 5, size=60)
        u = np.abs(p - 50.0)
        cov, risk = risk_coverage_curve(y, p, u, n_levels=10)
        aurc = area_under_risk_coverage(cov, risk)
        self.assertTrue(np.isfinite(aurc))
        self.assertGreater(aurc, 0.0)


class TestStratifiedBootstrapAuc(unittest.TestCase):
    """P1-9: stratified bootstrap keeps binary AUC CIs finite with tiny classes."""

    def test_rare_minority_class_never_nan(self):
        rng = np.random.default_rng(11)
        n = 40
        y = np.zeros(n, dtype=np.int64)
        y[:3] = 1  # only 3 positives — unstratified draws often empty them
        score = rng.normal(size=n) + y * 1.5
        point, lo, hi = stratified_bootstrap_auc(y, score, n_boot=200, seed=0)
        self.assertTrue(np.isfinite(point))
        self.assertTrue(np.isfinite(lo))
        self.assertTrue(np.isfinite(hi))
        self.assertLessEqual(lo, hi)
        self.assertAlmostEqual(point, roc_auc(y, score), places=6)

    def test_single_class_returns_nan(self):
        y = np.zeros(10, dtype=np.int64)
        score = np.arange(10, dtype=float)
        point, lo, hi = stratified_bootstrap_auc(y, score, n_boot=20, seed=0)
        self.assertTrue(np.isnan(point))
        self.assertTrue(np.isnan(lo))
        self.assertTrue(np.isnan(hi))

    def test_summarize_uses_finite_auc_ci(self):
        rng = np.random.default_rng(12)
        y = rng.uniform(15, 75, size=60)
        p = y + rng.normal(0, 4, size=60)
        out = summarize_clinical(y[20:], p[20:], cal_true=y[:20], cal_pred=p[:20], n_boot=30)
        ci = out.get("auc_ef_lt_50_bootstrap_ci95")
        self.assertIsNotNone(ci)
        self.assertTrue(np.isfinite(ci[0]))
        self.assertTrue(np.isfinite(ci[1]))


class TestCalibrationThresholds(unittest.TestCase):
    def test_summarize_reports_40_30(self):
        rng = np.random.default_rng(4)
        y = rng.uniform(15, 75, size=120)
        p = y + rng.normal(0, 4, size=120)
        out = summarize_clinical(
            y[40:],
            p[40:],
            cal_true=y[:40],
            cal_pred=p[:40],
            n_boot=40,
            seed=0,
            calibration_method="affine_logistic",
            adaptive_conformal=True,
            adaptive_scale="heuristic",
        )
        for t in (50, 40, 30):
            self.assertIn(f"ece_ef_lt_{t}", out)
            self.assertIn(f"brier_ef_lt_{t}", out)
            self.assertIn(f"affine_a_ef_lt_{t}", out)
            self.assertIn(f"auc_ef_lt_{t}_bootstrap_ci95", out)
        self.assertIn("rmse_bootstrap_ci95", out)
        self.assertIn("r2_bootstrap_ci95", out)
        self.assertIn("adaptive_conformal_coverage", out)
        self.assertIn("aurc", out)
        self.assertIn("adaptive_val_split", out)
        # n_cal=40 >= 8 with default split → proper VAL-scale / VAL-cal
        self.assertTrue(out["adaptive_val_split"])
        self.assertFalse(out["adaptive_conformal_empirical_heuristic"])
        self.assertIn("conformal_width_note", out)


class TestPairedBootstrap(unittest.TestCase):
    def test_delta_mae_sign(self):
        y = np.array([10.0, 20.0, 30.0, 40.0, 50.0, 60.0])
        good = y + 1.0
        bad = y + 5.0
        # Δ = MAE(good) - MAE(bad) < 0
        out = paired_bootstrap_delta_mae(y, good, bad, n_boot=200, seed=0)
        self.assertLess(out["delta_mae"], 0.0)
        self.assertLess(out["delta_mae_ci95_hi"], 0.5)
        self.assertEqual(out["n"], 6)


if __name__ == "__main__":
    unittest.main()
