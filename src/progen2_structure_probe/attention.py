"""Parameter-free attention scoring for Experiment 1."""

from __future__ import annotations

import numpy as np


def symmetrize(attention: np.ndarray) -> np.ndarray:
    matrix = np.asarray(attention)
    if matrix.shape[-1] != matrix.shape[-2]:
        raise ValueError("attention matrices must be square")
    return matrix + np.swapaxes(matrix, -1, -2)


def average_product_correction(matrix: np.ndarray) -> np.ndarray:
    """Apply the same row/column product correction used by ESM."""

    values = np.asarray(matrix, dtype=np.float64)
    row_sum = values.sum(axis=-1, keepdims=True)
    col_sum = values.sum(axis=-2, keepdims=True)
    total = values.sum(axis=(-2, -1), keepdims=True)
    if np.any(np.isclose(total, 0.0)):
        raise ValueError("APC is undefined for a channel whose matrix sum is zero")
    return values - row_sum * col_sum / total


def eligible_upper_triangle(
    length: int,
    valid_residues: np.ndarray,
    min_separation_exclusive: int,
) -> np.ndarray:
    valid = np.asarray(valid_residues, dtype=bool)
    if valid.shape != (length,):
        raise ValueError("valid_residues must have shape [length]")
    i, j = np.indices((length, length))
    return valid[:, None] & valid[None, :] & (j - i > min_separation_exclusive)


def zscore_channels(matrix: np.ndarray, reference_mask: np.ndarray) -> np.ndarray:
    """Z-standardize each leading channel over the selected pair population."""

    values = np.asarray(matrix, dtype=np.float64)
    mask = np.asarray(reference_mask, dtype=bool)
    if values.shape[-2:] != mask.shape:
        raise ValueError("reference_mask must match the final matrix dimensions")
    selected = values[..., mask]
    if selected.shape[-1] < 2:
        raise ValueError("at least two reference pairs are required for z-scoring")
    mean = selected.mean(axis=-1)
    std = selected.std(axis=-1, ddof=0)
    if np.any(std == 0):
        raise ValueError("cannot z-score a constant attention channel")
    return (values - mean[..., None, None]) / std[..., None, None]


def score_attention(
    attention: np.ndarray,
    valid_residues: np.ndarray,
    min_separation_exclusive: int = 10,
) -> tuple[np.ndarray, np.ndarray]:
    """Return global and layer-wise maximum APC-corrected z-scores.

    Args:
        attention: Post-softmax attention with shape ``[layers, heads, L, L]``.

    Returns:
        ``(global_score[L,L], layer_score[layers,L,L])``.
    """

    values = np.asarray(attention)
    if values.ndim != 4 or values.shape[-1] != values.shape[-2]:
        raise ValueError("attention must have shape [layers, heads, length, length]")
    length = values.shape[-1]
    reference = eligible_upper_triangle(
        length, valid_residues, min_separation_exclusive
    )
    corrected = average_product_correction(symmetrize(values))
    standardized = zscore_channels(corrected, reference)
    return standardized.max(axis=(0, 1)), standardized.max(axis=1)


def paired_scores(score: np.ndarray, pairs: object) -> tuple[np.ndarray, np.ndarray]:
    """Extract contact and decoy scores from a ``MatchedPairs``-like object."""

    values = np.asarray(score)
    return (
        values[pairs.contact_i, pairs.contact_j],
        values[pairs.decoy_i, pairs.decoy_j],
    )

