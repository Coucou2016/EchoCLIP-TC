import unittest

import numpy as np
import torch

from echoclip.clinical import (
    parse_ef_from_text,
    regression_metrics,
    roc_auc,
    summarize_clinical,
    threshold_aucs,
)
from echoclip.loss import TemporalClipLoss


class TestClinicalMetrics(unittest.TestCase):
    def test_perfect_regression(self):
        y = torch.tensor([20.0, 40.0, 60.0, 80.0])
        m = regression_metrics(y, y)
        self.assertAlmostEqual(m["mae"], 0.0)
        self.assertAlmostEqual(m["rmse"], 0.0)
        self.assertAlmostEqual(m["r2"], 1.0)
        self.assertEqual(m["n"], 4)

    def test_mae_rmse(self):
        y = np.array([0.0, 0.0, 0.0, 0.0])
        p = np.array([1.0, -1.0, 1.0, -1.0])
        m = regression_metrics(y, p)
        self.assertAlmostEqual(m["mae"], 1.0)
        self.assertAlmostEqual(m["rmse"], 1.0)

    def test_auc_perfect_threshold(self):
        y = np.array([20.0, 25.0, 60.0, 70.0])
        p = np.array([18.0, 22.0, 65.0, 72.0])
        aucs = threshold_aucs(y, p, thresholds=(50,))
        self.assertAlmostEqual(aucs["auc_ef_lt_50"], 1.0)

    def test_roc_auc_nan_single_class(self):
        self.assertTrue(np.isnan(roc_auc(np.array([1, 1, 1]), np.array([0.1, 0.2, 0.3]))))

    def test_parse_ef_from_demo_text(self):
        self.assertEqual(
            parse_ef_from_text("LEFT VENTRICULAR EJECTION FRACTION IS ESTIMATED TO BE 55%."),
            55.0,
        )
        self.assertEqual(parse_ef_from_text("LV EJECTION FRACTION IS 30%."), 30.0)
        self.assertIsNone(parse_ef_from_text("NO INTRACARDIAC DEVICE SEEN."))

    def test_summarize_clinical_toy(self):
        rng = np.random.default_rng(0)
        y = rng.uniform(20, 70, size=80)
        p = y + rng.normal(0, 3, size=80)
        y_cal, p_cal = y[:30], p[:30]
        y_te, p_te = y[30:], p[30:]
        out = summarize_clinical(y_te, p_te, cal_true=y_cal, cal_pred=p_cal, n_boot=50, seed=0)
        self.assertIn("mae", out)
        self.assertIn("auc_ef_lt_50", out)
        self.assertIn("ece_ef_lt_50", out)
        self.assertIn("conformal_coverage", out)
        self.assertGreater(out["conformal_coverage"], 0.5)
        # Fixed-width conformal → probability abstention (not vacuous width rule).
        self.assertEqual(out.get("abstention_rule"), "probability_confidence")
        self.assertIn("abstention_coverage", out)
        self.assertIn("abstention_n_keep", out)


class TestTemporalClipLoss(unittest.TestCase):
    def test_matches_clip_without_second_view(self):
        from echoclip.loss import ClipLoss

        z = torch.nn.functional.normalize(torch.randn(6, 16), dim=-1)
        t = torch.nn.functional.normalize(torch.randn(6, 16), dim=-1)
        scale = torch.tensor(2.0)
        a = ClipLoss()(z, t, scale)
        b = TemporalClipLoss(clip_weight=1.0, view_weight=0.5)(z, t, scale)
        self.assertAlmostEqual(a.item(), b.item(), places=5)

    def test_second_view_changes_loss(self):
        z = torch.nn.functional.normalize(torch.eye(4), dim=-1)
        t = z.clone()
        z2 = torch.nn.functional.normalize(torch.flip(z, dims=[0]), dim=-1)
        scale = torch.tensor(1.0)
        loss_fn = TemporalClipLoss(clip_weight=1.0, view_weight=1.0)
        one = loss_fn(z, t, scale)
        two = loss_fn(z, t, scale, video_features_2=z2)
        self.assertGreater(two.item(), one.item())


if __name__ == "__main__":
    unittest.main()
