import unittest

import numpy as np

from progen2_structure_probe.attention import (
    average_product_correction,
    score_attention,
    symmetrize,
)


class AttentionTests(unittest.TestCase):
    def test_symmetrize(self):
        matrix = np.asarray([[1.0, 2.0], [3.0, 4.0]])
        np.testing.assert_array_equal(symmetrize(matrix), [[2.0, 5.0], [5.0, 8.0]])

    def test_apc_has_near_zero_row_and_column_sums(self):
        matrix = np.asarray([[1.0, 2.0], [3.0, 5.0]])
        corrected = average_product_correction(matrix)
        np.testing.assert_allclose(corrected.sum(axis=0), 0.0, atol=1e-12)
        np.testing.assert_allclose(corrected.sum(axis=1), 0.0, atol=1e-12)

    def test_score_shapes(self):
        rng = np.random.default_rng(7)
        attention = rng.random((3, 2, 8, 8))
        attention /= attention.sum(axis=-1, keepdims=True)
        global_score, layer_score = score_attention(
            attention, np.ones(8, dtype=bool), min_separation_exclusive=1
        )
        self.assertEqual(global_score.shape, (8, 8))
        self.assertEqual(layer_score.shape, (3, 8, 8))


if __name__ == "__main__":
    unittest.main()

