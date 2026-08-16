import unittest

import numpy as np

from echoclip.calibrate import (
    abstain_by_width,
    apply_abstention,
    brier_score,
    conformal_coverage,
    conformal_intervals,
    expected_calibration_error,
    fit_temperature,
    softmax_np,
    split_conformal_quantile,
    temperature_scale_logits,
)


class TestCalibration(unittest.TestCase):
    def test_temperature_softens_probabilities(self):
        logits = np.array([[5.0, 0.0], [4.0, 0.1]], dtype=np.float64)
        p1 = softmax_np(logits, temperature=1.0)
        p10 = softmax_np(logits, temperature=10.0)
        self.assertGreater(p1.max(), p10.max())
        scaled = temperature_scale_logits(logits, 10.0)
        self.assertTrue(np.allclose(scaled * 10.0, logits))

    def test_fit_temperature_positive(self):
        logits = np.array([2.0, -1.0, 0.5, -0.2])
        labels = np.array([1.0, 0.0, 1.0, 0.0])
        t = fit_temperature(logits, labels)
        self.assertGreater(t, 0.0)
        self.assertLess(t, 100.0)

    def test_ece_perfect_is_zero(self):
        labels = np.array([0, 1, 0, 1], dtype=np.float64)
        ece = expected_calibration_error(labels, labels, n_bins=4)
        self.assertAlmostEqual(ece, 0.0, places=6)

    def test_brier_range(self):
        labels = np.array([0.0, 1.0, 1.0, 0.0])
        self.assertAlmostEqual(brier_score(labels, labels), 0.0)
        self.assertGreater(brier_score(np.array([0.5, 0.5, 0.5, 0.5]), labels), 0.0)

    def test_conformal_coverage_monotonic_in_alpha(self):
        rng = np.random.default_rng(0)
        y = rng.normal(size=400)
        pred = y + rng.normal(size=400) * 0.4
        cal_r = np.abs(y[:200] - pred[:200])
        test_y, test_p = y[200:], pred[200:]
        coverages = []
        for alpha in (0.5, 0.2, 0.1):
            q = split_conformal_quantile(cal_r, alpha=alpha)
            intervals = conformal_intervals(test_p, q)
            coverages.append(conformal_coverage(test_y, intervals))
        # Smaller alpha (wider target coverage) → coverage should not decrease.
        self.assertGreaterEqual(coverages[1], coverages[0] - 1e-6)
        self.assertGreaterEqual(coverages[2], coverages[1] - 1e-6)

    def test_abstention_improves_or_keeps_mae(self):
        y = np.array([10.0, 20.0, 30.0, 40.0])
        p = np.array([11.0, 21.0, 50.0, 80.0])  # last two badly off
        intervals = np.array([[9, 13], [19, 23], [20, 80], [10, 120]], dtype=float)
        mask = abstain_by_width(intervals, width_quantile=0.5)
        stats = apply_abstention(y, p, mask)
        full_mae = float(np.mean(np.abs(p - y)))
        self.assertLessEqual(stats["mae"], full_mae + 1e-9)
        self.assertGreater(stats["coverage"], 0.0)
        self.assertLess(stats["coverage"], 1.0)


if __name__ == "__main__":
    unittest.main()
