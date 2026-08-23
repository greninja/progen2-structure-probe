"""Small, explicit statistical summaries used by Experiment 1."""

from __future__ import annotations

import numpy as np


def _average_ranks(values: np.ndarray) -> np.ndarray:
    """One-based average ranks with deterministic tie handling."""

    order = np.argsort(values, kind="mergesort")
    sorted_values = values[order]
    sorted_ranks = np.empty(len(values), dtype=np.float64)
    start = 0
    while start < len(values):
        stop = start + 1
        while stop < len(values) and sorted_values[stop] == sorted_values[start]:
            stop += 1
        sorted_ranks[start:stop] = (start + 1 + stop) / 2
        start = stop
    ranks = np.empty_like(sorted_ranks)
    ranks[order] = sorted_ranks
    return ranks


def roc_auc(positive: np.ndarray, negative: np.ndarray) -> float:
    """Pooled ROC-AUC with ties counting as one half."""

    pos = np.asarray(positive, dtype=np.float64)
    neg = np.asarray(negative, dtype=np.float64)
    if len(pos) == 0 or len(neg) == 0:
        raise ValueError("ROC-AUC requires both positive and negative scores")
    combined = np.concatenate([pos, neg])
    ranks = _average_ranks(combined)
    rank_sum = ranks[: len(pos)].sum()
    return float((rank_sum - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def cohens_d_pooled(positive: np.ndarray, negative: np.ndarray) -> float:
    pos = np.asarray(positive, dtype=np.float64)
    neg = np.asarray(negative, dtype=np.float64)
    if len(pos) < 2 or len(neg) < 2:
        raise ValueError("pooled Cohen's d requires at least two values per group")
    variance = ((len(pos) - 1) * pos.var(ddof=1) + (len(neg) - 1) * neg.var(ddof=1))
    variance /= len(pos) + len(neg) - 2
    if variance == 0:
        raise ValueError("pooled Cohen's d is undefined when pooled variance is zero")
    return float((pos.mean() - neg.mean()) / np.sqrt(variance))


def matched_concordance(positive: np.ndarray, negative: np.ndarray) -> float:
    pos = np.asarray(positive)
    neg = np.asarray(negative)
    if pos.shape != neg.shape or pos.size == 0:
        raise ValueError("matched concordance requires equal nonempty arrays")
    return float(np.mean((pos > neg) + 0.5 * (pos == neg)))


def mandrake_like_mann_whitney(positive: np.ndarray, negative: np.ndarray) -> float:
    """Two-sided pair-level statistic for figure reconstruction, not inference."""

    from scipy.stats import mannwhitneyu

    return float(mannwhitneyu(positive, negative, alternative="two-sided").pvalue)
