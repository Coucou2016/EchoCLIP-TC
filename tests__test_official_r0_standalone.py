"""P0-5 / P0-6: standalone official R0 module + honest metadata gating.

Covers:
  * official stride ``0:min(40,T):2`` (standalone module),
  * EF prompt grid 0..100 (every integer) and the two documented templates,
  * aggregation parity on synthetic inputs (module vs zero-shot reference),
  * metadata gating: ``--paper`` alone must NEVER set verified=true,
  * golden-parity harness skips gracefully (never fabricates numbers).
"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from echoclip import official_r0 as r0
from echoclip.official_parity import compute_regression_metric_official
from echoclip.protocol import OFFICIAL_EF_VALUES
from echoclip.zeroshot import compute_regression_score


class TestOfficialStride(unittest.TestCase):
    def test_stride_matches_0_min40_t_2(self):
        for t in (1, 5, 20, 39, 40, 41, 100):
            idx = r0.frame_indices(t)
            np.testing.assert_array_equal(idx, np.arange(0, min(40, t), 2))

    def test_select_frames_returns_indices(self):
        frames = np.arange(3 * 4 * 4 * 3, dtype=np.uint8).reshape(3, 4, 4, 3)
        big = np.zeros((50, 4, 4, 3), dtype=np.uint8)
        big[: len(frames)] = frames
        selected, idx = r0.select_official_frames(big)
        self.assertEqual(idx, [int(i) for i in range(0, 40, 2)])
        self.assertEqual(selected.shape[0], 20)


class TestPromptGrid(unittest.TestCase):
    def test_grid_is_0_to_100_every_integer(self):
        prompts, values = r0.build_ef_prompt_grid()
        self.assertEqual(len(values), 2 * 101)
        self.assertEqual(values[:101], list(OFFICIAL_EF_VALUES))
        self.assertEqual(values[101:], list(OFFICIAL_EF_VALUES))
        self.assertEqual(min(values), 0)
        self.assertEqual(max(values), 100)
        self.assertEqual(sorted(set(values)), list(range(101)))

    def test_prompt_strings_use_both_templates(self):
        prompts, _ = r0.build_ef_prompt_grid()
        self.assertEqual(prompts[0], "THE LEFT VENTRICULAR EJECTION FRACTION IS ESTIMATED TO BE 0% ")
        self.assertIn("LV EJECTION FRACTION IS 100%. ", prompts)
        for p in prompts:
            self.assertNotIn("<#>", p)

    def test_prompt_hash_is_stable_and_sensitive(self):
        h1 = r0.prompt_grid_sha256()
        h2 = r0.prompt_grid_sha256()
        self.assertEqual(h1, h2)
        prompts, values = r0.build_ef_prompt_grid()
        prompts[0] = prompts[0] + "X"
        self.assertNotEqual(r0.prompt_grid_sha256(prompts, values), h1)


class TestAggregationParitySynthetic(unittest.TestCase):
    def test_module_matches_official_utils(self):
        torch.manual_seed(0)
        video = F.normalize(torch.randn(2, 8, 16), dim=-1)
        prompts = F.normalize(torch.randn(202, 16), dim=-1)
        values = [i % 101 for i in range(202)]
        ref = compute_regression_metric_official(video, prompts, values)
        ours = compute_regression_score(video, prompts, values)
        self.assertTrue(torch.allclose(ref, ours, atol=1e-6, rtol=1e-5))


class TestHonestMetadata(unittest.TestCase):
    def test_paper_mode_alone_never_verifies(self):
        meta = r0.official_metadata(paper_mode=True)
        self.assertTrue(meta["paper_mode"])
        self.assertTrue(meta["official_reproduction_requested"])
        self.assertFalse(meta["official_reproduction_verified"])
        self.assertFalse(meta["official_reproduction"])

    def test_verified_requires_explicit_proof(self):
        meta = r0.official_metadata(paper_mode=True, verified=True)
        self.assertTrue(meta["official_reproduction_verified"])
        self.assertTrue(meta["official_reproduction"])

    def test_meta_records_upstream_provenance(self):
        meta = r0.official_metadata(paper_mode=True)
        self.assertIn("upstream_repo", meta)
        self.assertIn("upstream_commit", meta)
        self.assertEqual(meta["official_hub"], r0.OFFICIAL_HUB)
        self.assertGreaterEqual(len(meta["parity_gaps"]), 3)


class TestParityHarnessSkipsGracefully(unittest.TestCase):
    def test_missing_avi_skips_without_numbers(self):
        report = r0.parity_report(example_avi=None)
        self.assertFalse(report.available)
        self.assertTrue(report.skipped)
        self.assertFalse(report.verified)
        self.assertIsNone(report.max_abs_diff)
        self.assertIn("skipped", report.reason)
        self.assertEqual(report.n_cases, 0)

    def test_nonexistent_path_skips(self):
        with tempfile.TemporaryDirectory() as td:
            missing = Path(td) / "nope.avi"
            report = r0.parity_report(example_avi=missing)
            self.assertFalse(report.verified)
            self.assertTrue(report.skipped)
            self.assertIsNone(report.max_abs_diff)

    def test_report_serializes(self):
        report = r0.parity_report(example_avi=None)
        d = report.to_dict()
        self.assertIn("skipped", d)
        self.assertIn("verified", d)


class TestInputHash(unittest.TestCase):
    def test_input_hash_changes_with_pixels(self):
        a = np.zeros((3, 4, 4, 3), dtype=np.uint8)
        b = a.copy()
        b[0, 0, 0, 0] = 7
        self.assertEqual(r0.input_sha256(a), r0.input_sha256(a))
        self.assertNotEqual(r0.input_sha256(a), r0.input_sha256(b))


if __name__ == "__main__":
    unittest.main()
