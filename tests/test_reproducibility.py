import csv
import hashlib
import json
from pathlib import Path


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_tracked_frozen_cohort_matches_recorded_hashes():
    root = Path("artifacts/cohort")
    summary = json.loads((root / "summary.json").read_text(encoding="utf-8"))
    manifest = Path("data/manifests/experiment1_150.csv")
    pilot = Path("data/manifests/experiment1_pilot.csv")
    assert _sha256(manifest) == summary["manifest_sha256"]
    assert _sha256(pilot) == summary["pilot_manifest_sha256"]
    with manifest.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 150
    assert len({row["cluster_id"] for row in rows}) == 150
    assert len({row["sequence_sha256"] for row in rows}) == 150


def test_tracked_hidden_probe_outputs_match_reported_hashes():
    root = Path("artifacts/experiment1-hidden-probe")
    summary_path = root / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    split = json.loads((root / "splits.json").read_text(encoding="utf-8"))
    assert _sha256(summary_path) == (
        "31f8c7ac37d7c0cbc0eed77bbdf5104a86e086b1c1cc3e76ba4bc947758ef251"
    )
    assert _sha256(root / "representation_index.json") == summary[
        "representation_index_sha256"
    ]
    assert split["content_sha256"] == summary["split_sha256"]
    assert split["counts"] == {"train": 90, "validation": 30, "test": 30}
    assert len(summary["validation_grid"]) == 84
