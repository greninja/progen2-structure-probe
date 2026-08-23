import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from progen2_structure_probe.models.progen2 import ProGenExtraction
from progen2_structure_probe.smoke import run_model_smoke, synthetic_sequence


class FakeCuda:
    @staticmethod
    def is_available():
        return False


class FakeTorch:
    cuda = FakeCuda()


class FakeProGen:
    torch = FakeTorch()

    @staticmethod
    def extract(sequence):
        length = len(sequence)
        attention = np.zeros((27, 16, length, length), dtype=np.float32)
        hidden = np.zeros((28, length, 1536), dtype=np.float32)
        return ProGenExtraction(np.arange(length + 2), attention, hidden)


class FakeESM:
    torch = FakeTorch()

    @staticmethod
    def extract_attention(sequence):
        return np.zeros((12, 20, len(sequence), len(sequence)), dtype=np.float32)


class SmokeTests(unittest.TestCase):
    def test_synthetic_sequence_length(self):
        self.assertEqual(len(synthetic_sequence(41)), 41)

    def test_smoke_runner_writes_passed_result(self):
        with tempfile.TemporaryDirectory() as directory:
            result = run_model_smoke(FakeProGen(), FakeESM(), Path(directory), 16)
            self.assertEqual(result["status"], "passed")
            self.assertTrue((Path(directory) / "smoke.json").exists())


if __name__ == "__main__":
    unittest.main()

