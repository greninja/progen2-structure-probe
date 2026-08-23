import csv
import hashlib
import tempfile
import unittest
from pathlib import Path

from progen2_structure_probe.experiment1 import load_structure_manifest


class ManifestTests(unittest.TestCase):
    def test_provenance_columns_and_matching_mmcif_hash_are_accepted(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            mmcif = root / "TEST.cif"
            mmcif.write_text("data_TEST\n", encoding="utf-8")
            digest = hashlib.sha256(mmcif.read_bytes()).hexdigest()
            manifest = root / "manifest.csv"
            with manifest.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(
                    handle,
                    fieldnames=[
                        "structure_id",
                        "label_asym_id",
                        "mmcif_path",
                        "mmcif_sha256",
                        "cluster_id",
                    ],
                )
                writer.writeheader()
                writer.writerow(
                    {
                        "structure_id": "TEST",
                        "label_asym_id": "A",
                        "mmcif_path": "TEST.cif",
                        "mmcif_sha256": digest,
                        "cluster_id": "TEST_1",
                    }
                )
            records = load_structure_manifest(manifest)
            self.assertEqual(records[0]["mmcif_path"], str(mmcif.resolve()))

    def test_mmcif_hash_mismatch_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            mmcif = root / "TEST.cif"
            mmcif.write_text("data_TEST\n", encoding="utf-8")
            manifest = root / "manifest.csv"
            manifest.write_text(
                "structure_id,label_asym_id,mmcif_path,mmcif_sha256\n"
                "TEST,A,TEST.cif,wrong\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "hash mismatch"):
                load_structure_manifest(manifest)


if __name__ == "__main__":
    unittest.main()
