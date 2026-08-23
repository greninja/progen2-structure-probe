import io
from http.client import RemoteDisconnected
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from progen2_structure_probe.cohort import (
    _post_json,
    _resolve_mmseqs,
    fetch_candidates,
    read_clusters,
    rcsb_search_payload,
    select_five_chain_pilot,
)


class CohortTests(unittest.TestCase):
    def test_mmseqs_resolves_from_active_environment_when_absent_from_path(self):
        with tempfile.TemporaryDirectory() as directory:
            environment_bin = Path(directory) / "bin"
            environment_bin.mkdir()
            python = environment_bin / "python"
            mmseqs = environment_bin / "mmseqs"
            mmseqs.touch()
            with patch("progen2_structure_probe.cohort.shutil.which", return_value=None):
                with patch("progen2_structure_probe.cohort.sys.executable", str(python)):
                    self.assertEqual(_resolve_mmseqs(), str(mmseqs))

    def test_post_json_retries_remote_disconnect(self):
        response = io.BytesIO(b'{"status": "ok"}')
        with patch(
            "progen2_structure_probe.cohort.urlopen",
            side_effect=[RemoteDisconnected("closed"), response],
        ):
            with patch("progen2_structure_probe.cohort.time.sleep") as sleep:
                self.assertEqual(
                    _post_json("https://example.test", {}), {"status": "ok"}
                )
        sleep.assert_called_once_with(1)

    def test_rcsb_query_contains_declared_primary_filters(self):
        payload = rcsb_search_payload(100, 500)
        self.assertEqual(payload["return_type"], "polymer_entity")
        self.assertTrue(payload["request_options"]["return_all_hits"])
        nodes = payload["query"]["nodes"]
        parameters = {node["parameters"]["attribute"]: node["parameters"] for node in nodes}
        self.assertEqual(
            parameters["entity_poly.rcsb_entity_polymer_type"]["value"], "Protein"
        )
        self.assertEqual(
            parameters["entity_poly.rcsb_sample_sequence_length"]["value"],
            {"from": 100, "include_lower": True, "to": 500, "include_upper": True},
        )
        self.assertEqual(parameters["exptl.method"]["value"], "X-RAY DIFFRACTION")
        self.assertEqual(parameters["rcsb_entry_info.resolution_combined"]["value"], 2.0)

    def test_cluster_reader_preserves_representative_membership(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "clusters.tsv"
            path.write_text("A\tA\nA\tB\nC\tC\n", encoding="utf-8")
            self.assertEqual(read_clusters(path), {"A": ["A", "B"], "C": ["C"]})

    def test_noncanonical_candidate_rejection_is_recorded(self):
        response = {
            "data": {
                "polymer_entities": [
                    {
                        "rcsb_id": "TEST_1",
                        "entity_poly": {
                            "pdbx_seq_one_letter_code_can": "A" * 99 + "X",
                            "rcsb_entity_polymer_type": "Protein",
                            "rcsb_sample_sequence_length": 100,
                        },
                        "rcsb_polymer_entity_container_identifiers": {
                            "entry_id": "TEST",
                            "asym_ids": ["A"],
                        },
                        "entry": {
                            "rcsb_entry_info": {"resolution_combined": [1.5]},
                            "exptl": [{"method": "X-RAY DIFFRACTION"}],
                        },
                    }
                ]
            }
        }
        rejections = []
        with patch("progen2_structure_probe.cohort._post_json", return_value=response):
            candidates = fetch_candidates(["TEST_1"], rejections=rejections)
        self.assertEqual(candidates, [])
        self.assertEqual(rejections[0]["stage"], "data_api_filter")
        self.assertIn("noncanonical", rejections[0]["reason"])

    def test_pilot_is_fixed_length_quantiles_with_lexical_ties(self):
        rows = [
            {
                "length": length,
                "structure_id": f"P{index:03d}",
                "label_asym_id": "A",
            }
            for index, length in enumerate(range(100, 250))
        ]
        pilot = select_five_chain_pilot(rows)
        self.assertEqual([row["length"] for row in pilot], [100, 137, 174, 212, 249])


if __name__ == "__main__":
    unittest.main()
