"""Orchestration for Mandrake Experiment 2."""

from __future__ import annotations

import csv
import hashlib
from pathlib import Path
from typing import Any

from .artifacts import write_json_atomic
from .copy_bias import periodic_copy_identity, random_protein, validate_protein_sequence


def _sequence_hash(sequence: str) -> str:
    return hashlib.sha256(sequence.encode("ascii")).hexdigest()


def load_sequence_manifest(path: Path) -> list[dict[str, str]]:
    required = {"name", "source_id", "sequence", "sequence_sha256"}
    with Path(path).open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None or set(reader.fieldnames) != required:
            raise ValueError(f"sequence manifest columns must be exactly {sorted(required)}")
        records = []
        for line_number, row in enumerate(reader, start=2):
            sequence = validate_protein_sequence(row["sequence"])
            actual_hash = _sequence_hash(sequence)
            if row["sequence_sha256"] != actual_hash:
                raise ValueError(f"sequence hash mismatch on manifest line {line_number}")
            records.append({**row, "sequence": sequence})
    if not records:
        raise ValueError("sequence manifest is empty")
    return records


def run_experiment2(config: dict[str, Any], model: object, output_dir: Path) -> dict[str, Any]:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    seed = int(config["run"]["seed"])

    perplexity_records = load_sequence_manifest(Path(config["perplexity"]["manifest"]))
    expected_names = config["perplexity"]["expected_names"]
    if [record["name"] for record in perplexity_records] != expected_names:
        raise ValueError("perplexity manifest names/order do not match the recovered figure")

    perplexity_results: list[dict[str, Any]] = []
    for index, record in enumerate(perplexity_records):
        original, repeated, _ = model.duplicate_perplexity(record["sequence"])
        perplexity_results.append(
            {
                "name": record["name"],
                "kind": "real",
                "source_id": record["source_id"],
                "length": len(record["sequence"]),
                "sequence_sha256": record["sequence_sha256"],
                "original_mean_per_position_perplexity": original,
                "repeated_mean_per_position_perplexity": repeated,
            }
        )
        random_sequence = random_protein(len(record["sequence"]), seed + index)
        random_original, random_repeated, _ = model.duplicate_perplexity(random_sequence)
        perplexity_results.append(
            {
                "name": f"Random #{index + 1}",
                "kind": "uniform-random-fallback",
                "source_id": None,
                "length": len(random_sequence),
                "sequence_sha256": _sequence_hash(random_sequence),
                "random_seed": seed + index,
                "original_mean_per_position_perplexity": random_original,
                "repeated_mean_per_position_perplexity": random_repeated,
            }
        )

    generation_records = load_sequence_manifest(Path(config["generation"]["manifest"]))
    expected_generation = config["generation"]["expected_proteins"]
    if [record["name"] for record in generation_records] != [item["name"] for item in expected_generation]:
        raise ValueError("generation manifest names/order do not match the recovered figure")
    for record, expectation in zip(generation_records, expected_generation):
        if len(record["sequence"]) != expectation["reported_length"]:
            raise ValueError(f"{record['name']} length does not match the published figure")

    generation_results: list[dict[str, Any]] = []
    for protein_index, record in enumerate(generation_records):
        for strategy_index, strategy in enumerate(config["generation"]["strategies"]):
            generation_seed = seed + 1000 * protein_index + strategy_index
            generated, prefix_length = model.generate_repeat(
                record["sequence"],
                new_residues=int(config["generation"]["generated_residues"]),
                strategy=strategy["name"],
                top_p=float(strategy.get("top_p", 0.95)),
                temperature=float(strategy.get("temperature", 0.8)),
                seed=generation_seed,
            )
            prompt_residue_count = len(record["sequence"]) + prefix_length
            identity = periodic_copy_identity(
                generated, record["sequence"], prompt_residue_count
            )
            generation_results.append(
                {
                    "name": record["name"],
                    "source_id": record["source_id"],
                    "length": len(record["sequence"]),
                    "sequence_sha256": record["sequence_sha256"],
                    "strategy": strategy,
                    "seed": generation_seed,
                    "repeat_prefix_length": prefix_length,
                    "generated_residue_count": len(generated),
                    "periodic_copy_identity": identity,
                    "generated_sequence": generated,
                }
            )

    result = {
        "experiment": 2,
        "result_label": config["protocol"]["result_label"],
        "perplexity": perplexity_results,
        "generation": generation_results,
    }
    write_json_atomic(output / "results.json", result)
    return result

