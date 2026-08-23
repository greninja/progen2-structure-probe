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
    if config["experiment"] not in {1, 2}:
        raise ValueError("experiment must be 1 or 2")
    if not isinstance(config["run"].get("seed"), int):
        raise ValueError("run.seed must be an integer")
    return config


def resolved_config_record(path: Path) -> dict[str, Any]:
    config = load_config(path)
    return {
        "source_path": str(Path(path).resolve()),
        "source_sha256": sha256_file(Path(path)),
        "resolved_sha256": canonical_json_sha256(config),
        "config": config,
    }

