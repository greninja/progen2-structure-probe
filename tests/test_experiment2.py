import csv
import hashlib
import tempfile
import unittest
from pathlib import Path

import numpy as np

from progen2_structure_probe.config import load_config
from progen2_structure_probe.experiment2 import run_experiment2


class FakeModel:
    def duplicate_perplexity(self, sequence):
        return 20.0, 1.5, np.zeros(2 * len(sequence))

    def generate_repeat(self, sequence, **kwargs):
        prefix = len(sequence) // 4
        prompt_count = len(sequence) + prefix
        generated = "".join(
            sequence[(prompt_count + offset) % len(sequence)] for offset in range(12)
        )
        return generated, prefix


def write_manifest(path, rows):
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=["name", "source_id", "sequence", "sequence_sha256"]
        )
        writer.writeheader()
        for name, length in rows:
            sequence = ("ACDEFGHIKLMNPQRSTVWY" * ((length + 19) // 20))[:length]
            writer.writerow(
                {
                    "name": name,
                    "source_id": f"TEST-{name}",
                    "sequence": sequence,
                    "sequence_sha256": hashlib.sha256(sequence.encode("ascii")).hexdigest(),
                }
            )


class Experiment2Tests(unittest.TestCase):
    def test_runner_enforces_layout_and_writes_results(self):
        config = load_config(Path("configs/experiment2_reproduction.yaml"))
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            perplexity = root / "perplexity.csv"
            generation = root / "generation.csv"
            write_manifest(perplexity, [(name, 40 + index) for index, name in enumerate(config["perplexity"]["expected_names"])])
            write_manifest(
                generation,
                [(item["name"], item["reported_length"]) for item in config["generation"]["expected_proteins"]],
            )
            config["perplexity"]["manifest"] = str(perplexity)
            config["generation"]["manifest"] = str(generation)
            result = run_experiment2(config, FakeModel(), root / "output")
            self.assertEqual(len(result["perplexity"]), 12)
            self.assertEqual(len(result["generation"]), 14)
            self.assertTrue(all(row["periodic_copy_identity"] == 1.0 for row in result["generation"]))
            self.assertTrue((root / "output/results.json").exists())


if __name__ == "__main__":
    unittest.main()

