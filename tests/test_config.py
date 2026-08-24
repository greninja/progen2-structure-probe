import tempfile
import unittest
from pathlib import Path

from progen2_structure_probe.config import load_config


class ConfigTests(unittest.TestCase):
    def test_repository_configs_validate(self):
        path = Path("configs/experiment1_pilot.yaml")
        self.assertEqual(load_config(path)["experiment"], 1)

    def test_missing_seed_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "bad.yaml"
            path.write_text(
                "schema_version: 1\nexperiment: 1\nprotocol: {}\nrun: {}\nmodel: {}\n",
                encoding="utf-8",
            )
            with self.assertRaises(ValueError):
                load_config(path)

    def test_non_experiment1_config_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "bad.yaml"
            path.write_text(
                "schema_version: 1\nexperiment: 3\nprotocol: {}\nrun: {seed: 1}\nmodel: {}\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "only Experiment 1"):
                load_config(path)


if __name__ == "__main__":
    unittest.main()
