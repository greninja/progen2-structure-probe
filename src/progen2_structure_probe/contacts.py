"""Contact geometry and deterministic sequence-distance-matched decoys."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib

import numpy as np


@dataclass(frozen=True)
class MatchedPairs:
    """One contact and one noncontact per row, matched by sequence separation."""

    contact_i: np.ndarray
    contact_j: np.ndarray
    decoy_i: np.ndarray
    decoy_j: np.ndarray
    separation: np.ndarray

    def __post_init__(self) -> None:
        arrays = (
            self.contact_i,
            self.contact_j,
            self.decoy_i,
            self.decoy_j,
            self.separation,
        )
        lengths = {len(value) for value in arrays}
        if len(lengths) != 1:
            raise ValueError("all matched-pair columns must have equal length")
        if np.any(self.contact_j - self.contact_i != self.separation):
            raise ValueError("contact separations do not match the separation column")
        if np.any(self.decoy_j - self.decoy_i != self.separation):
            raise ValueError("decoy separations do not match the separation column")

    def __len__(self) -> int:
        return len(self.separation)


def virtual_cb(backbone: np.ndarray) -> np.ndarray:
    """Return ESM-style virtual C-beta coordinates from [N, CA, C].

    Args:
        backbone: Array with shape ``[..., 3, 3]``. The atom axis is N, CA, C.
    """

    coords = np.asarray(backbone, dtype=np.float64)
    if coords.shape[-2:] != (3, 3):
        raise ValueError("backbone must have shape [..., 3, 3] for N, CA, C")
    n, ca, c = coords[..., 0, :], coords[..., 1, :], coords[..., 2, :]
    b = ca - n
    c_vec = c - ca
    a = np.cross(b, c_vec)
    return -0.58273431 * a + 0.56802827 * b - 0.54067466 * c_vec + ca


def contact_map(
    cb_coords: np.ndarray,
    valid_residues: np.ndarray,
    cutoff_angstrom: float = 8.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Return contact labels and valid-pair mask for one chain."""

    coords = np.asarray(cb_coords, dtype=np.float64)
    valid = np.asarray(valid_residues, dtype=bool)
    if coords.ndim != 2 or coords.shape[1] != 3:
        raise ValueError("cb_coords must have shape [length, 3]")
    if valid.shape != (len(coords),):
        raise ValueError("valid_residues must have shape [length]")
    delta = coords[:, None, :] - coords[None, :, :]
    distances = np.linalg.norm(delta, axis=-1)
    valid_pairs = valid[:, None] & valid[None, :]
    contacts = (distances < cutoff_angstrom) & valid_pairs
    np.fill_diagonal(contacts, False)
    np.fill_diagonal(valid_pairs, False)
    return contacts, valid_pairs


def stable_seed(base_seed: int, structure_id: str, chain_id: str) -> int:
    payload = f"{base_seed}:{structure_id}:{chain_id}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")


def match_contacts_to_decoys(
    contacts: np.ndarray,
    valid_pairs: np.ndarray,
    min_separation_exclusive: int,
    seed: int,
) -> MatchedPairs:
    """Balance contacts/noncontacts independently at each exact separation."""

    labels = np.asarray(contacts, dtype=bool)
    valid = np.asarray(valid_pairs, dtype=bool)
    if labels.ndim != 2 or labels.shape[0] != labels.shape[1]:
        raise ValueError("contacts must be a square matrix")
    if valid.shape != labels.shape:
        raise ValueError("valid_pairs shape must match contacts")
    if not np.array_equal(labels, labels.T):
        raise ValueError("contacts must be symmetric")
    if not np.array_equal(valid, valid.T):
        raise ValueError("valid_pairs must be symmetric")

    rng = np.random.default_rng(seed)
    length = labels.shape[0]
    contact_rows: list[tuple[int, int]] = []
    decoy_rows: list[tuple[int, int]] = []
    separations: list[int] = []

    for separation in range(min_separation_exclusive + 1, length):
        i = np.arange(0, length - separation, dtype=np.int64)
        j = i + separation
        eligible = valid[i, j]
        positives = np.flatnonzero(eligible & labels[i, j])
        negatives = np.flatnonzero(eligible & ~labels[i, j])
        count = min(len(positives), len(negatives))
        if count == 0:
            continue
        selected_pos = rng.choice(positives, size=count, replace=False)
        selected_neg = rng.choice(negatives, size=count, replace=False)
        rng.shuffle(selected_pos)
        rng.shuffle(selected_neg)
        contact_rows.extend(zip(i[selected_pos].tolist(), j[selected_pos].tolist()))
        decoy_rows.extend(zip(i[selected_neg].tolist(), j[selected_neg].tolist()))
        separations.extend([separation] * count)

    if contact_rows:
        contact_array = np.asarray(contact_rows, dtype=np.int64)
        decoy_array = np.asarray(decoy_rows, dtype=np.int64)
    else:
        contact_array = np.empty((0, 2), dtype=np.int64)
        decoy_array = np.empty((0, 2), dtype=np.int64)
    return MatchedPairs(
        contact_i=contact_array[:, 0],
        contact_j=contact_array[:, 1],
        decoy_i=decoy_array[:, 0],
        decoy_j=decoy_array[:, 1],
        separation=np.asarray(separations, dtype=np.int64),
    )

