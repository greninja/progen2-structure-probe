"""mmCIF polymer-to-coordinate mapping using label sequence identifiers."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np


CANONICAL_THREE_TO_ONE = {
    "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C",
    "GLN": "Q", "GLU": "E", "GLY": "G", "HIS": "H", "ILE": "I",
    "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F", "PRO": "P",
    "SER": "S", "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V",
}


@dataclass(frozen=True)
class PolymerChain:
    structure_id: str
    label_asym_id: str
    sequence: str
    backbone: np.ndarray  # [L, N/CA/C, xyz], NaN for missing atoms

    @property
    def valid_backbone(self) -> np.ndarray:
        return np.isfinite(self.backbone).all(axis=(1, 2))


def _as_list(value: object) -> list[str]:
    if isinstance(value, list):
        return [str(item) for item in value]
    return [str(value)]


def _column(table: dict[str, object], key: str) -> list[str]:
    if key not in table:
        raise ValueError(f"mmCIF is missing required column {key}")
    return _as_list(table[key])


def _rows(columns: Iterable[list[str]]) -> Iterable[tuple[str, ...]]:
    materialized = list(columns)
    lengths = {len(column) for column in materialized}
    if len(lengths) != 1:
        raise ValueError("mmCIF loop columns have inconsistent lengths")
    return zip(*materialized)


def load_polymer_chain(path: Path, label_asym_id: str) -> PolymerChain:
    """Load one model-1 chain while preserving unobserved polymer positions.

    Modified or unknown monomers are rejected rather than silently mapped.
    """

    try:
        from Bio.PDB.MMCIF2Dict import MMCIF2Dict
    except ImportError as error:
        raise RuntimeError("Biopython is required for mmCIF loading") from error

    table = MMCIF2Dict(str(path))
    scheme_rows = _rows(
        [
            _column(table, "_pdbx_poly_seq_scheme.asym_id"),
            _column(table, "_pdbx_poly_seq_scheme.seq_id"),
            _column(table, "_pdbx_poly_seq_scheme.mon_id"),
        ]
    )
    polymer: dict[int, str] = {}
    for asym, sequence_id, monomer in scheme_rows:
        if asym != label_asym_id:
            continue
        position = int(sequence_id)
        if monomer not in CANONICAL_THREE_TO_ONE:
            raise ValueError(
                f"noncanonical monomer {monomer!r} at {label_asym_id}:{position}; "
                "the primary cohort rejects modified residues"
            )
        polymer[position] = CANONICAL_THREE_TO_ONE[monomer]
    if not polymer:
        raise ValueError(f"chain {label_asym_id!r} is absent from polymer scheme")
    expected_positions = list(range(1, max(polymer) + 1))
    if sorted(polymer) != expected_positions:
        raise ValueError("polymer label_seq_id values are not contiguous from one")

    atom_columns = [
        _column(table, "_atom_site.label_asym_id"),
        _column(table, "_atom_site.label_seq_id"),
        _column(table, "_atom_site.label_atom_id"),
        _column(table, "_atom_site.Cartn_x"),
        _column(table, "_atom_site.Cartn_y"),
        _column(table, "_atom_site.Cartn_z"),
        _column(table, "_atom_site.occupancy"),
        _column(table, "_atom_site.label_alt_id"),
        _column(table, "_atom_site.pdbx_PDB_model_num"),
    ]
    candidates: dict[tuple[int, str], list[tuple[float, str, np.ndarray]]] = {}
    for asym, sequence_id, atom, x, y, z, occupancy, alt_id, model in _rows(atom_columns):
        if asym != label_asym_id or model not in {"1", ".", "?"}:
            continue
        if sequence_id in {".", "?"} or atom not in {"N", "CA", "C"}:
            continue
        position = int(sequence_id)
        if position not in polymer:
            continue
        normalized_alt = "" if alt_id in {".", "?"} else alt_id
        candidates.setdefault((position, atom), []).append(
            (float(occupancy), normalized_alt, np.asarray([float(x), float(y), float(z)]))
        )

    backbone = np.full((len(polymer), 3, 3), np.nan, dtype=np.float64)
    atom_index = {"N": 0, "CA": 1, "C": 2}
    for (position, atom), choices in candidates.items():
        # Protocol: highest occupancy, then alt A, then lexicographic alt identifier.
        selected = sorted(choices, key=lambda item: (-item[0], item[1] != "A", item[1]))[0]
        backbone[position - 1, atom_index[atom]] = selected[2]

    structure_id = Path(path).stem.split(".")[0].upper()
    return PolymerChain(
        structure_id=structure_id,
        label_asym_id=label_asym_id,
        sequence="".join(polymer[position] for position in expected_positions),
        backbone=backbone,
    )

