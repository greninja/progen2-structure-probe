"""Experiment 2 sequence-duplication and periodic-copy utilities."""

from __future__ import annotations

import math
import numpy as np


CANONICAL_AA = frozenset("ACDEFGHIKLMNPQRSTVWY")


def validate_protein_sequence(sequence: str) -> str:
    cleaned = sequence.strip().upper()
    invalid = sorted(set(cleaned) - CANONICAL_AA)
    if not cleaned:
        raise ValueError("protein sequence is empty")
    if invalid:
        raise ValueError(f"sequence contains noncanonical residues: {''.join(invalid)}")
    return cleaned


def duplication_context(sequence: str, include_end_token: bool = True) -> str:
    seq = validate_protein_sequence(sequence)
    return "1" + seq + seq + ("2" if include_end_token else "")


def repeat_prompt(sequence: str, prefix_fraction: float = 0.25) -> tuple[str, int]:
    seq = validate_protein_sequence(sequence)
    if not 0 < prefix_fraction <= 1:
        raise ValueError("prefix_fraction must be in (0, 1]")
    prefix_length = math.floor(len(seq) * prefix_fraction)
    if prefix_length == 0:
        raise ValueError("prefix_fraction yields an empty repeat prefix")
    return "1" + seq + seq[:prefix_length], prefix_length


def mean_per_position_perplexity(log_probabilities: np.ndarray) -> float:
    """Arithmetic mean of tokenwise perplexities, matching the plot label."""

    values = np.asarray(log_probabilities, dtype=np.float64)
    if values.ndim != 1 or values.size == 0:
        raise ValueError("log_probabilities must be a nonempty vector")
    return float(np.exp(-values).mean())


def summarize_duplicate_log_probs(
    residue_log_probabilities: np.ndarray, sequence_length: int
) -> tuple[float, float]:
    values = np.asarray(residue_log_probabilities, dtype=np.float64)
    if values.shape != (2 * sequence_length,):
        raise ValueError("expected exactly two residue copies")
    return (
        mean_per_position_perplexity(values[:sequence_length]),
        mean_per_position_perplexity(values[sequence_length:]),
    )


def periodic_copy_identity(
    generated_sequence: str,
    source_sequence: str,
    prompt_residue_count: int,
) -> float:
    generated = validate_protein_sequence(generated_sequence)
    source = validate_protein_sequence(source_sequence)
    expected = "".join(
        source[(prompt_residue_count + offset) % len(source)]
        for offset in range(len(generated))
    )
    return float(np.mean(np.fromiter((a == b for a, b in zip(generated, expected)), bool)))


def random_protein(length: int, seed: int) -> str:
    """Uniform canonical-AA fallback; this is not recovered Mandrake methodology."""

    if length <= 0:
        raise ValueError("length must be positive")
    alphabet = np.asarray(sorted(CANONICAL_AA))
    rng = np.random.default_rng(seed)
    return "".join(rng.choice(alphabet, size=length).tolist())

