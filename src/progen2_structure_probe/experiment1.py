"""Five-chain and full-cohort orchestration for Experiment 1."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

import numpy as np

from .artifacts import write_json_atomic
from .attention import paired_scores, score_attention
from .contacts import contact_map, match_contacts_to_decoys, stable_seed, virtual_cb
from .metrics import (
    cohens_d_pooled,
    mandrake_like_mann_whitney,
    matched_concordance,
    roc_auc,
)
from .mmcif import load_polymer_chain


def load_structure_manifest(path: Path) -> list[dict[str, str]]:
    required = {"structure_id", "label_asym_id", "mmcif_path"}
    manifest = Path(path)
    with manifest.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None or set(reader.fieldnames) != required:
            raise ValueError(f"structure manifest columns must be exactly {sorted(required)}")
        records = list(reader)
    if not records:
        raise ValueError("structure manifest is empty")
    for record in records:
        source = Path(record["mmcif_path"])
        if not source.is_absolute():
            source = (manifest.parent / source).resolve()
        record["mmcif_path"] = str(source)
    return records


def _summary(positive: np.ndarray, negative: np.ndarray) -> dict[str, float]:
    return {
        "auc": roc_auc(positive, negative),
        "cohens_d_pooled": cohens_d_pooled(positive, negative),
        "matched_concordance": matched_concordance(positive, negative),
        "mann_whitney_pair_level_p": mandrake_like_mann_whitney(positive, negative),
    }


def _safe_summary(positive: np.ndarray, negative: np.ndarray) -> dict[str, Any]:
    if len(positive) < 2 or len(negative) < 2:
        return {"count_per_class": int(min(len(positive), len(negative))), "status": "insufficient"}
    return {
        "count_per_class": int(min(len(positive), len(negative))),
        "status": "ok",
        **_summary(positive, negative),
    }


def run_experiment1(
    config: dict[str, Any],
    progen_model: object,
    esm_model: object,
    output_dir: Path,
) -> dict[str, Any]:
    output = Path(output_dir)
    chain_dir = output / "chains"
    chain_dir.mkdir(parents=True, exist_ok=True)
    manifest = load_structure_manifest(Path(config["run"]["manifest"]))
    base_seed = int(config["run"]["seed"])
    cutoff = float(config["contact"]["cutoff_angstrom"])
    min_separation = int(config["contact"]["minimum_separation_exclusive"])

    all_progen_positive: list[np.ndarray] = []
    all_progen_negative: list[np.ndarray] = []
    all_esm_positive: list[np.ndarray] = []
    all_esm_negative: list[np.ndarray] = []
    all_separations: list[np.ndarray] = []
    all_progen_layer_positive: list[np.ndarray] = []
    all_progen_layer_negative: list[np.ndarray] = []
    all_esm_layer_positive: list[np.ndarray] = []
    all_esm_layer_negative: list[np.ndarray] = []
    chain_summaries: list[dict[str, Any]] = []

    for record in manifest:
        chain = load_polymer_chain(Path(record["mmcif_path"]), record["label_asym_id"])
        if chain.structure_id != record["structure_id"].upper():
            raise ValueError(f"structure ID mismatch for {record['mmcif_path']}")
        cb = virtual_cb(chain.backbone)
        contacts, valid_pairs = contact_map(cb, chain.valid_backbone, cutoff)
        pairs = match_contacts_to_decoys(
            contacts,
            valid_pairs,
            min_separation,
            stable_seed(base_seed, chain.structure_id, chain.label_asym_id),
        )
        if len(pairs) == 0:
            raise ValueError(f"{chain.structure_id}:{chain.label_asym_id} has no matched pairs")

        progen = progen_model.extract(chain.sequence)
        if progen.attentions.shape[-2:] != (len(chain.sequence), len(chain.sequence)):
            raise ValueError("ProGen2 residue attention shape is misaligned")
        if progen.attentions.shape[:2] != (27, 16):
            raise ValueError("ProGen2-base must expose 27 layers and 16 heads")
        causal_upper = np.triu(progen.attentions, k=1)
        if not np.allclose(causal_upper, 0.0, atol=1e-6):
            raise ValueError("ProGen2 attention violates the expected causal mask")
        progen_score, progen_layer_score = score_attention(
            progen.attentions, chain.valid_backbone, min_separation
        )
        progen_positive, progen_negative = paired_scores(progen_score, pairs)
        progen_layer_positive = progen_layer_score[:, pairs.contact_i, pairs.contact_j]
        progen_layer_negative = progen_layer_score[:, pairs.decoy_i, pairs.decoy_j]

        esm_attention = esm_model.extract_attention(chain.sequence)
        if esm_attention.shape[0] != 12 or esm_attention.shape[-2:] != (
            len(chain.sequence), len(chain.sequence)
        ):
            raise ValueError("ESM-2 attention shape is not the expected 12-layer residue map")
        esm_score, esm_layer_score = score_attention(
            esm_attention, chain.valid_backbone, min_separation
        )
        esm_positive, esm_negative = paired_scores(esm_score, pairs)
        esm_layer_positive = esm_layer_score[:, pairs.contact_i, pairs.contact_j]
        esm_layer_negative = esm_layer_score[:, pairs.decoy_i, pairs.decoy_j]

        slug = f"{chain.structure_id}_{chain.label_asym_id}"
        np.savez_compressed(
            chain_dir / f"{slug}.npz",
            contact_i=pairs.contact_i,
            contact_j=pairs.contact_j,
            decoy_i=pairs.decoy_i,
            decoy_j=pairs.decoy_j,
            separation=pairs.separation,
            progen_positive=progen_positive,
            progen_negative=progen_negative,
            esm_positive=esm_positive,
            esm_negative=esm_negative,
            progen_layer_positive=progen_layer_positive,
            progen_layer_negative=progen_layer_negative,
            esm_layer_positive=esm_layer_positive,
            esm_layer_negative=esm_layer_negative,
            progen_layer_score=progen_layer_score,
            esm_layer_score=esm_layer_score,
            valid_backbone=chain.valid_backbone,
        )
        chain_summaries.append(
            {
                "structure_id": chain.structure_id,
                "label_asym_id": chain.label_asym_id,
                "length": len(chain.sequence),
                "valid_backbone_residues": int(chain.valid_backbone.sum()),
                "matched_pairs": len(pairs),
                "progen2": _summary(progen_positive, progen_negative),
                "esm2": _summary(esm_positive, esm_negative),
            }
        )
        all_progen_positive.append(progen_positive)
        all_progen_negative.append(progen_negative)
        all_esm_positive.append(esm_positive)
        all_esm_negative.append(esm_negative)
        all_separations.append(pairs.separation)
        all_progen_layer_positive.append(progen_layer_positive)
        all_progen_layer_negative.append(progen_layer_negative)
        all_esm_layer_positive.append(esm_layer_positive)
        all_esm_layer_negative.append(esm_layer_negative)

    pooled_progen_positive = np.concatenate(all_progen_positive)
    pooled_progen_negative = np.concatenate(all_progen_negative)
    pooled_esm_positive = np.concatenate(all_esm_positive)
    pooled_esm_negative = np.concatenate(all_esm_negative)
    pooled_separation = np.concatenate(all_separations)
    pooled_progen_layer_positive = np.concatenate(all_progen_layer_positive, axis=1)
    pooled_progen_layer_negative = np.concatenate(all_progen_layer_negative, axis=1)
    pooled_esm_layer_positive = np.concatenate(all_esm_layer_positive, axis=1)
    pooled_esm_layer_negative = np.concatenate(all_esm_layer_negative, axis=1)

    distance_summaries: list[dict[str, Any]] = []
    for lower, upper in config["distance_bins"]["intervals"]:
        selected = (pooled_separation > lower) & (pooled_separation <= upper)
        distance_summaries.append(
            {
                "label": f"({lower},{upper}]",
                "lower_exclusive": lower,
                "upper_inclusive": upper,
                "progen2": _safe_summary(
                    pooled_progen_positive[selected], pooled_progen_negative[selected]
                ),
                "esm2": _safe_summary(
                    pooled_esm_positive[selected], pooled_esm_negative[selected]
                ),
            }
        )

    progen_layers = [
        {"layer": layer, **_safe_summary(positive, negative)}
        for layer, (positive, negative) in enumerate(
            zip(pooled_progen_layer_positive, pooled_progen_layer_negative)
        )
    ]
    esm_layers = [
        {"layer": layer, **_safe_summary(positive, negative)}
        for layer, (positive, negative) in enumerate(
            zip(pooled_esm_layer_positive, pooled_esm_layer_negative)
        )
    ]
    result = {
        "experiment": 1,
        "result_label": config["protocol"]["result_label"],
        "chains": chain_summaries,
        "pooled": {
            "matched_pairs": len(pooled_progen_positive),
            "progen2": _summary(pooled_progen_positive, pooled_progen_negative),
            "esm2": _summary(pooled_esm_positive, pooled_esm_negative),
        },
        "distance_bins": distance_summaries,
        "layers": {"progen2": progen_layers, "esm2": esm_layers},
    }
    write_json_atomic(output / "summary.json", result)
    return result
