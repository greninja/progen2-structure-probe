"""Shared protein-sequence validation."""

from __future__ import annotations


CANONICAL_AA = frozenset("ACDEFGHIKLMNPQRSTVWY")


def validate_protein_sequence(sequence: str) -> str:
    cleaned = sequence.strip().upper()
    invalid = sorted(set(cleaned) - CANONICAL_AA)
    if not cleaned:
        raise ValueError("protein sequence is empty")
    if invalid:
        raise ValueError(f"sequence contains noncanonical residues: {''.join(invalid)}")
    return cleaned
