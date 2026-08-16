import unittest

import numpy as np

from echoclip.cycle_sample import (
    pad_or_trim_indices,
    sample_cycle_indices,
    sample_ed_es,
    sample_uniform,
)


class TestCycleSample(unittest.TestCase):
    def test_uniform_length_and_bounds(self):
        idx = sample_uniform(100, 16)
        self.assertEqual(len(idx), 16)
        self.assertEqual(idx[0], 0)
        self.assertEqual(idx[-1], 99)
        self.assertTrue(np.all(idx >= 0) and np.all(idx < 100))

    def test_random_deterministic(self):
        a = sample_cycle_indices(80, 16, strategy="random", seed=0)
        b = sample_cycle_indices(80, 16, strategy="random", seed=0)
        c = sample_cycle_indices(80, 16, strategy="random", seed=1)
        self.assertTrue(np.array_equal(a, b))
        self.assertFalse(np.array_equal(a, c))
        self.assertEqual(len(a), 16)
        self.assertTrue(np.all(np.diff(a) >= 0))

    def test_ed_es_includes_phase_frames(self):
        idx = sample_ed_es(120, 16, ed_index=10, es_index=70, seed=0)
        self.assertIn(10, set(idx.tolist()))
        self.assertIn(70, set(idx.tolist()))
        self.assertEqual(len(idx), 16)

    def test_ed_es_falls_back_without_indices(self):
        idx = sample_cycle_indices(50, 8, strategy="ed_es")
        self.assertTrue(np.array_equal(idx, sample_uniform(50, 8)))

    def test_mixed_only_uses_known_strategies(self):
        idx = sample_cycle_indices(40, 8, strategy="mixed", ed_index=3, es_index=20, seed=7)
        self.assertEqual(len(idx), 8)

    def test_pad_or_trim(self):
        padded = pad_or_trim_indices([0, 5], n_samples=4, num_frames=10)
        self.assertEqual(len(padded), 4)
        self.assertEqual(int(padded[-1]), 5)
        trimmed = pad_or_trim_indices(np.arange(10), n_samples=4, num_frames=10)
        self.assertEqual(len(trimmed), 4)

    def test_unknown_strategy(self):
        with self.assertRaises(ValueError):
            sample_cycle_indices(10, 4, strategy="bogus")


if __name__ == "__main__":
    unittest.main()
