import unittest

import numpy as np

from progen2_structure_probe.copy_bias import (
    duplication_context,
    periodic_copy_identity,
    repeat_prompt,
    summarize_duplicate_log_probs,
)


class CopyBiasTests(unittest.TestCase):
    def test_context_and_floor_quarter_prefix(self):
        sequence = "ACDEFGHIKLMNPQRSTVW"  # 19 residues; floor(25%) = 4
        self.assertEqual(duplication_context(sequence), "1" + sequence * 2 + "2")
        prompt, prefix_length = repeat_prompt(sequence)
        self.assertEqual(prefix_length, 4)
        self.assertEqual(prompt, "1" + sequence + sequence[:4])

    def test_mean_per_position_perplexity_for_halves(self):
        first = np.log(np.asarray([1 / 20, 1 / 10]))
        second = np.log(np.asarray([1 / 2, 1 / 1]))
        original, repeated = summarize_duplicate_log_probs(np.r_[first, second], 2)
        self.assertAlmostEqual(original, 15.0)
        self.assertAlmostEqual(repeated, 1.5)

    def test_periodic_copy_identity_respects_prompt_offset(self):
        source = "ACDE"
        # Six prompt residues means expected continuation starts at D: DEAC...
        self.assertEqual(periodic_copy_identity("DEACDE", source, 6), 1.0)
        self.assertAlmostEqual(periodic_copy_identity("DEACDA", source, 6), 5 / 6)


if __name__ == "__main__":
    unittest.main()
