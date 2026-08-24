import unittest

from progen2_structure_probe.sequences import validate_protein_sequence


class SequenceTests(unittest.TestCase):
    def test_validation_normalizes_canonical_sequence(self):
        self.assertEqual(validate_protein_sequence(" acdefg \n"), "ACDEFG")

    def test_validation_rejects_empty_and_noncanonical_sequences(self):
        with self.assertRaisesRegex(ValueError, "empty"):
            validate_protein_sequence("  ")
        with self.assertRaisesRegex(ValueError, "noncanonical residues: BX"):
            validate_protein_sequence("ACBX")


if __name__ == "__main__":
    unittest.main()
