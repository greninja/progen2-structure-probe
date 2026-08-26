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
    probe = config.get("probe")
    if probe is not None:
        stages = probe.get("stages")
        contextual = probe.get("contextual_stages")
        alphas = probe.get("regularization_alpha")
        split = probe.get("split", {})
        if stages != list(range(28)):
            raise ValueError("probe.stages must contain ProGen2 stages 0 through 27")
        if contextual != list(range(1, 28)):
            raise ValueError("probe.contextual_stages must contain stages 1 through 27")
        if not isinstance(alphas, list) or not alphas or any(
            float(value) <= 0 for value in alphas
        ):
            raise ValueError("probe.regularization_alpha must contain positive values")
        if int(probe.get("max_training_matched_pairs_per_protein", 0)) <= 0:
            raise ValueError(
                "probe.max_training_matched_pairs_per_protein must be positive"
            )
        split_count = sum(
            int(split.get(key, 0))
            for key in ("train_count", "validation_count", "test_count")
        )
        if split_count != int(cohort["target_count"]):
            raise ValueError("probe split counts must equal cohort.target_count")
        split_hash = split.get("content_sha256")
        if not isinstance(split_hash, str) or len(split_hash) != 64:
            raise ValueError("probe.split.content_sha256 must be a SHA-256 digest")
        bootstrap = probe.get("bootstrap", {})
        if int(bootstrap.get("replicates", 0)) <= 0:
            raise ValueError("probe.bootstrap.replicates must be positive")
    return config


def resolved_config_record(path: Path) -> dict[str, Any]:
    config = load_config(path)
    return {
        "source_path": str(Path(path).resolve()),
        "source_sha256": sha256_file(Path(path)),
        "resolved_sha256": canonical_json_sha256(config),
        "config": config,
    }
