"""Strict loading and provenance capture for experiment YAML files."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from .artifacts import canonical_json_sha256, sha256_file


REQUIRED_TOP_LEVEL = {"schema_version", "experiment", "protocol", "run", "model"}


def load_config(path: Path) -> dict[str, Any]:
    source = Path(path)
    with source.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError("configuration root must be a mapping")
    missing = REQUIRED_TOP_LEVEL - set(config)
    if missing:
        raise ValueError(f"configuration is missing required keys: {sorted(missing)}")
    if config["schema_version"] != 1:
        raise ValueError("only configuration schema_version 1 is supported")
    if config["experiment"] != 1:
        raise ValueError("only Experiment 1 configurations are supported")
    if not isinstance(config["run"].get("seed"), int):
        raise ValueError("run.seed must be an integer")
    required_cohort = {
        "min_length",
        "max_length",
        "maximum_resolution_angstrom",
        "target_count",
        "minimum_coordinate_coverage",
        "mmseqs_version",
        "sequence_identity",
        "bidirectional_coverage",
    }
    cohort = config.get("cohort")
    if not isinstance(cohort, dict) or not required_cohort.issubset(cohort):
        missing_cohort = required_cohort - set(cohort or {})
        raise ValueError(f"Experiment 1 cohort config is missing: {sorted(missing_cohort)}")
    if not 0 < float(cohort["sequence_identity"]) <= 1:
        raise ValueError("cohort.sequence_identity must be in (0, 1]")
    if not 0 < float(cohort["bidirectional_coverage"]) <= 1:
        raise ValueError("cohort.bidirectional_coverage must be in (0, 1]")
    if not 0 < float(cohort["minimum_coordinate_coverage"]) <= 1:
        raise ValueError("cohort.minimum_coordinate_coverage must be in (0, 1]")
    return config


def resolved_config_record(path: Path) -> dict[str, Any]:
    config = load_config(path)
    return {
        "source_path": str(Path(path).resolve()),
        "source_sha256": sha256_file(Path(path)),
        "resolved_sha256": canonical_json_sha256(config),
        "config": config,
    }
