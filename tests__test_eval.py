import unittest

import torch

from echoclip.eval import demo_pacemaker_labels, pairwise_retrieval_metrics


class TestEvalMetrics(unittest.TestCase):
    def test_perfect_retrieval(self):
        n = 8
        z = torch.eye(n)
        m = pairwise_retrieval_metrics(z, z)
        self.assertEqual(m["i2t_r1"], 100.0)
        self.assertEqual(m["t2i_r1"], 100.0)

    def test_demo_labels(self):
        texts = [
            "PACER LEAD SEEN",
            "NORMAL STUDY",
        ]
        labels = demo_pacemaker_labels(texts)
        self.assertTrue(labels[0])
        self.assertFalse(labels[1])


if __name__ == "__main__":
    unittest.main()
