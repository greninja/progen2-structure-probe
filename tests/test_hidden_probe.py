import unittest
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import tempfile

import numpy as np

from progen2_structure_probe.hidden_probe import (
    _fit_classifier,
    _load_validation_checkpoint,
    _write_validation_checkpoint,
    extract_hidden_representations,
    make_probe_splits,
    pair_features,
    paired_protein_bootstrap,
)
from progen2_structure_probe.metrics import roc_auc
from progen2_structure_probe.mmcif import PolymerChain


class HiddenProbeTests(unittest.TestCase):
    def test_pair_features_are_symmetric_elementwise_products(self):
        hidden = np.asarray(
            [
                [1.0, 2.0],
                [3.0, 4.0],
                [-1.0, 5.0],
                [2.0, -2.0],
            ]
        )
        pairs = {
            "contact_i": np.asarray([0, 1]),
            "contact_j": np.asarray([1, 2]),
            "decoy_i": np.asarray([0, 2]),
            "decoy_j": np.asarray([2, 3]),
            "separation": np.asarray([1, 1]),
        }
        positive, negative = pair_features(hidden, pairs)
        np.testing.assert_array_equal(positive, [[3.0, 8.0], [-3.0, 20.0]])
        np.testing.assert_array_equal(negative, [[-1.0, 10.0], [-2.0, -10.0]])

    def test_protein_splits_have_frozen_counts_and_length_strata(self):
        entries = []
        sizes = [38, 38, 37, 37]
        for bin_index, size in enumerate(sizes):
            for offset in range(size):
                entries.append(
                    {
                        "structure_id": f"P{bin_index}{offset:03d}",
                        "label_asym_id": "A",
                        "cluster_id": f"C{bin_index}{offset:03d}",
                        "length": 100 * (bin_index + 1),
                    }
                )
        config = {
            "unit": "protein-cluster",
            "train_count": 90,
            "validation_count": 30,
            "test_count": 30,
        }
        first = make_probe_splits(entries, config, 20260822)
        second = make_probe_splits(entries, config, 20260822)
        self.assertEqual(first, second)
        self.assertEqual(first["counts"], {"train": 90, "validation": 30, "test": 30})
        self.assertEqual(len(first["content_sha256"]), 64)

        length_by_id = {row["structure_id"]: row["length"] for row in entries}
        observed = {}
        identifiers = set()
        for split_name, rows in first["splits"].items():
            bins = [0, 0, 0, 0]
            for row in rows:
                self.assertNotIn(row["structure_id"], identifiers)
                identifiers.add(row["structure_id"])
                bins[(length_by_id[row["structure_id"]] - 100) // 100] += 1
            observed[split_name] = bins
        self.assertEqual(observed["train"], [23, 23, 22, 22])
        self.assertEqual(observed["validation"], [8, 8, 7, 7])
        self.assertEqual(observed["test"], [7, 7, 8, 8])

    def test_low_capacity_classifier_recovers_simple_product_signal(self):
        try:
            import sklearn  # noqa: F401
        except ImportError:
            self.skipTest("probe dependencies are not installed")
        positive = np.asarray([[2.0, 1.0], [3.0, 0.5], [1.5, 2.0], [2.5, 1.5]])
        negative = -positive
        features = np.r_[positive, negative]
        labels = np.r_[np.ones(len(positive)), np.zeros(len(negative))]
        scaler, classifier = _fit_classifier(features, labels, 0.0001, 7)
        positive_scores = classifier.decision_function(scaler.transform(positive))
        negative_scores = classifier.decision_function(scaler.transform(negative))
        self.assertEqual(roc_auc(positive_scores, negative_scores), 1.0)

    def test_paired_bootstrap_is_deterministic_and_protein_level(self):
        contextual = [
            {"structure_id": f"P{i}", "label_asym_id": "A", "auc": value}
            for i, value in enumerate([0.7, 0.8, 0.6])
        ]
        baseline = [
            {"structure_id": f"P{i}", "label_asym_id": "A", "auc": value}
            for i, value in enumerate([0.5, 0.6, 0.4])
        ]
        first = paired_protein_bootstrap(contextual, baseline, 1000, 0.95, 11)
        second = paired_protein_bootstrap(contextual, baseline, 1000, 0.95, 11)
        self.assertEqual(first, second)
        self.assertAlmostEqual(first["mean_auc_difference"], 0.2)
        self.assertAlmostEqual(first["lower"], 0.2)
        self.assertAlmostEqual(first["upper"], 0.2)

    def test_validation_stage_checkpoint_is_fingerprint_bound(self):
        results = [
            {"stage": 3, "alpha": 0.00001, "validation": {"mean_per_protein_auc": 0.6}},
            {"stage": 3, "alpha": 0.0001, "validation": {"mean_per_protein_auc": 0.61}},
            {"stage": 3, "alpha": 0.001, "validation": {"mean_per_protein_auc": 0.59}},
        ]
        alphas = [0.00001, 0.0001, 0.001]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "stage_03.json"
            _write_validation_checkpoint(path, 3, "frozen-inputs", results)
            self.assertEqual(
                _load_validation_checkpoint(path, 3, alphas, "frozen-inputs"),
                results,
            )
            self.assertIsNone(
                _load_validation_checkpoint(path, 3, alphas, "different-inputs")
            )
            self.assertIsNone(
                _load_validation_checkpoint(path, 4, alphas, "frozen-inputs")
            )

    def test_hidden_extraction_cache_round_trip_avoids_attention(self):
        class FakeCuda:
            @staticmethod
            def is_available():
                return False

        class FakeModel:
            torch = SimpleNamespace(cuda=FakeCuda())

            def __init__(self):
                self.calls = 0

            def extract_hidden_states(self, sequence):
                self.calls += 1
                hidden = np.ones((28, len(sequence), 1536), dtype=np.float32)
                return SimpleNamespace(hidden_states=hidden)

        length = 14
        backbone = np.zeros((length, 3, 3), dtype=np.float64)
        chain = PolymerChain("TEST", "A", "A" * length, backbone)
        contacts = np.zeros((length, length), dtype=bool)
        contacts[0, 11] = contacts[11, 0] = True
        valid_pairs = np.ones((length, length), dtype=bool)
        np.fill_diagonal(valid_pairs, False)
        config = {
            "run": {"seed": 7, "manifest": "unused.csv"},
            "contact": {"cutoff_angstrom": 8.0, "minimum_separation_exclusive": 10},
            "decoy": {"match": "exact-sequence-separation"},
            "probe": {
                "representation": "post-token-hidden-state",
                "pair_feature": "elementwise-product",
            },
        }
        record = {
            "structure_id": "TEST",
            "label_asym_id": "A",
            "mmcif_path": "TEST.cif",
            "cluster_id": "TEST_1",
        }
        model = FakeModel()
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            with patch(
                "progen2_structure_probe.hidden_probe.load_structure_manifest",
                return_value=[record],
            ), patch(
                "progen2_structure_probe.hidden_probe.load_polymer_chain",
                return_value=chain,
            ), patch(
                "progen2_structure_probe.hidden_probe.contact_map",
                return_value=(contacts, valid_pairs),
            ):
                first = extract_hidden_representations(config, model, output)
                second = extract_hidden_representations(config, model, output)
            self.assertEqual(model.calls, 1)
            self.assertEqual(first, second)
            with (output / "representations" / "extraction_run.json").open() as handle:
                extraction_run = json.load(handle)
            self.assertTrue(extraction_run["chains"][0]["cache_reused"])
            hidden = np.load(
                output / "representations" / "TEST_A.hidden.npy", mmap_mode="r"
            )
            self.assertEqual(hidden.shape, (28, length, 1536))
            self.assertEqual(hidden.dtype, np.float16)


if __name__ == "__main__":
    unittest.main()
