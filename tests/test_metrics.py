import unittest

import numpy as np

from progen2_structure_probe.metrics import matched_concordance, roc_auc


class MetricTests(unittest.TestCase):
    def test_auc_with_ties(self):
        self.assertEqual(roc_auc(np.asarray([1.0, 2.0]), np.asarray([0.0, 1.0])), 0.875)

    def test_matched_concordance(self):
        value = matched_concordance(np.asarray([2.0, 1.0, 1.0]), np.asarray([1.0, 2.0, 1.0]))
        self.assertEqual(value, 0.5)


if __name__ == "__main__":
    unittest.main()

