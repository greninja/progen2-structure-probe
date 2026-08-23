import unittest

import numpy as np

from progen2_structure_probe.contacts import (
    contact_map,
    match_contacts_to_decoys,
    stable_seed,
    virtual_cb,
)


class ContactTests(unittest.TestCase):
    def test_virtual_cb_matches_fixed_reference(self):
        backbone = np.asarray(
            [[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0]]]
        )
        actual = virtual_cb(backbone)
        expected = np.asarray([[1.56802827, -0.54067466, -0.58273431]])
        np.testing.assert_allclose(actual, expected, atol=1e-8)

    def test_contact_map_masks_missing_residue(self):
        coords = np.asarray([[0.0, 0.0, 0.0], [7.9, 0.0, 0.0], [20.0, 0.0, 0.0]])
        contacts, valid = contact_map(coords, np.asarray([True, True, False]))
        self.assertTrue(contacts[0, 1])
        self.assertFalse(valid[0, 2])
        self.assertFalse(contacts[0, 0])

    def test_decoys_are_exactly_matched_and_deterministic(self):
        length = 9
        valid = np.ones((length, length), dtype=bool)
        np.fill_diagonal(valid, False)
        contacts = np.zeros_like(valid)
        for i, j in [(0, 3), (1, 4), (0, 4), (2, 6)]:
            contacts[i, j] = contacts[j, i] = True
        seed = stable_seed(20260822, "TEST", "A")
        first = match_contacts_to_decoys(contacts, valid, 2, seed)
        second = match_contacts_to_decoys(contacts, valid, 2, seed)
        self.assertGreater(len(first), 0)
        np.testing.assert_array_equal(first.decoy_i, second.decoy_i)
        np.testing.assert_array_equal(first.separation, first.contact_j - first.contact_i)
        np.testing.assert_array_equal(first.separation, first.decoy_j - first.decoy_i)
        self.assertTrue(np.all(contacts[first.contact_i, first.contact_j]))
        self.assertTrue(np.all(~contacts[first.decoy_i, first.decoy_j]))


if __name__ == "__main__":
    unittest.main()

