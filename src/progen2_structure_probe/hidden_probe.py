"""Frozen hidden-state extraction and a low-capacity contact probe."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import tempfile
from typing import Any, Iterable

import numpy as np

from .artifacts import canonical_json_sha256, sha256_file, write_json_atomic
from .contacts import contact_map, match_contacts_to_decoys, stable_seed, virtual_cb
from .experiment1 import load_structure_manifest
from .metrics import matched_concordance, roc_auc
from .mmcif import load_polymer_chain
from .smoke import _measure_cuda


def _sequence_sha256(sequence: str) -> str:
    return hashlib.sha256(sequence.encode("ascii")).hexdigest()


def _atomic_save_npy(path: Path, values: np.ndarray) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", dir=str(destination.parent)
    )
    try:
        with os.fdopen(descriptor, "wb") as handle:
            np.save(handle, values, allow_pickle=False)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, destination)
    except BaseException:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise


def _atomic_save_npz(path: Path, **arrays: Any) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", dir=str(destination.parent)
    )
    try:
        with os.fdopen(descriptor, "wb") as handle:
            np.savez_compressed(handle, **arrays)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, destination)
    except BaseException:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise


def _cached_representation_matches(
    hidden_path: Path,
    pairs_path: Path,
    sequence_sha256: str,
    expected_pairs: object,
    length: int,
) -> bool:
    if not hidden_path.exists() or not pairs_path.exists():
        return False
    try:
        hidden = np.load(hidden_path, mmap_mode="r", allow_pickle=False)
        if hidden.shape != (28, length, 1536) or hidden.dtype != np.float16:
            return False
        if not np.isfinite(hidden).all():
            return False
        with np.load(pairs_path, allow_pickle=False) as stored:
            if str(stored["sequence_sha256"].item()) != sequence_sha256:
                return False
            for name in ("contact_i", "contact_j", "decoy_i", "decoy_j", "separation"):
                if not np.array_equal(stored[name], getattr(expected_pairs, name)):
                    return False
    except (KeyError, OSError, ValueError):
        return False
    return True


def extract_hidden_representations(
    config: dict[str, Any],
    progen_model: object,
    output_dir: Path,
) -> dict[str, Any]:
    """Extract restartable float16 hidden-state caches for the frozen cohort."""

    root = Path(output_dir) / "representations"
    root.mkdir(parents=True, exist_ok=True)
    manifest = load_structure_manifest(Path(config["run"]["manifest"]))
    base_seed = int(config["run"]["seed"])
    cutoff = float(config["contact"]["cutoff_angstrom"])
    min_separation = int(config["contact"]["minimum_separation_exclusive"])
    entries: list[dict[str, Any]] = []
    run_records: list[dict[str, Any]] = []

    for record in manifest:
        chain = load_polymer_chain(Path(record["mmcif_path"]), record["label_asym_id"])
        if chain.structure_id != record["structure_id"].upper():
            raise ValueError(f"structure ID mismatch for {record['mmcif_path']}")
        sequence_hash = _sequence_sha256(chain.sequence)
        declared_sequence_hash = record.get("sequence_sha256")
        if declared_sequence_hash and declared_sequence_hash != sequence_hash:
            raise ValueError(f"sequence hash mismatch for {chain.structure_id}:{chain.label_asym_id}")

        contacts, valid_pairs = contact_map(
            virtual_cb(chain.backbone), chain.valid_backbone, cutoff
        )
        pairs = match_contacts_to_decoys(
            contacts,
            valid_pairs,
            min_separation,
            stable_seed(base_seed, chain.structure_id, chain.label_asym_id),
        )
        if len(pairs) == 0:
            raise ValueError(f"{chain.structure_id}:{chain.label_asym_id} has no matched pairs")

        slug = f"{chain.structure_id}_{chain.label_asym_id}"
        hidden_path = root / f"{slug}.hidden.npy"
        pairs_path = root / f"{slug}.pairs.npz"
        reused = _cached_representation_matches(
            hidden_path, pairs_path, sequence_hash, pairs, len(chain.sequence)
        )
        resources: dict[str, Any] | None = None
        if not reused:
            extraction, seconds, memory = _measure_cuda(
                progen_model.torch,
                lambda: progen_model.extract_hidden_states(chain.sequence),
            )
            hidden = np.asarray(extraction.hidden_states)
            if hidden.shape != (28, len(chain.sequence), 1536):
                raise ValueError(f"unexpected ProGen2 hidden-state shape {hidden.shape}")
            if not np.isfinite(hidden).all():
                raise ValueError("ProGen2 hidden-state extraction contains non-finite values")
            _atomic_save_npy(hidden_path, hidden.astype(np.float16))
            _atomic_save_npz(
                pairs_path,
                structure_id=np.asarray(chain.structure_id),
                label_asym_id=np.asarray(chain.label_asym_id),
                sequence_sha256=np.asarray(sequence_hash),
                contact_i=pairs.contact_i,
                contact_j=pairs.contact_j,
                decoy_i=pairs.decoy_i,
                decoy_j=pairs.decoy_j,
                separation=pairs.separation,
            )
            resources = {"extraction_seconds": seconds, "memory": memory}

        entry = {
            "structure_id": chain.structure_id,
            "label_asym_id": chain.label_asym_id,
            "cluster_id": record.get("cluster_id", f"{chain.structure_id}:{chain.label_asym_id}"),
            "length": len(chain.sequence),
            "sequence_sha256": sequence_hash,
            "matched_pairs": len(pairs),
            "hidden_file": hidden_path.name,
            "hidden_sha256": sha256_file(hidden_path),
            "pairs_file": pairs_path.name,
            "pairs_sha256": sha256_file(pairs_path),
        }
        entries.append(entry)
        run_record = {
            "structure_id": chain.structure_id,
            "label_asym_id": chain.label_asym_id,
            "cache_reused": reused,
        }
        if resources is not None:
            run_record["resources"] = resources
        run_records.append(run_record)

    index = {
        "schema_version": 1,
        "representation": config["probe"]["representation"],
        "dtype": "float16",
        "shape": [28, "protein_length", 1536],
        "pair_feature": config["probe"]["pair_feature"],
        "pairing": {
            "seed": base_seed,
            "contact_cutoff_angstrom": cutoff,
            "minimum_separation_exclusive": min_separation,
            "decoy_match": config["decoy"]["match"],
        },
        "chains": entries,
    }
    index["content_sha256"] = canonical_json_sha256(index)
    write_json_atomic(root / "index.json", index)
    write_json_atomic(root / "extraction_run.json", {"chains": run_records})
    return index


def _length_bin(length: int) -> int:
    for index, (lower, upper) in enumerate(((100, 199), (200, 299), (300, 399), (400, 500))):
        if lower <= length <= upper:
            return index
    raise ValueError(f"length {length} is outside the frozen cohort bins")


def _allocate_proportionally(total: int, sizes: list[int]) -> list[int]:
    population = sum(sizes)
    if total < 0 or total > population:
        raise ValueError("allocation total is outside the available population")
    exact = [total * size / population for size in sizes]
    allocation = [int(np.floor(value)) for value in exact]
    remaining = total - sum(allocation)
    order = sorted(range(len(sizes)), key=lambda i: (exact[i] - allocation[i], -i), reverse=True)
    for index in order[:remaining]:
        allocation[index] += 1
    return allocation


def make_probe_splits(
    entries: list[dict[str, Any]], split_config: dict[str, Any], seed: int
) -> dict[str, Any]:
    """Create deterministic protein-level splits stratified by original length bin."""

    counts = {
        "train": int(split_config["train_count"]),
        "validation": int(split_config["validation_count"]),
        "test": int(split_config["test_count"]),
    }
    if sum(counts.values()) != len(entries):
        raise ValueError("split counts do not match representation count")
    cluster_ids = [str(entry["cluster_id"]) for entry in entries]
    if len(cluster_ids) != len(set(cluster_ids)):
        raise ValueError("representation index contains repeated protein clusters")

    bins: list[list[dict[str, Any]]] = [[] for _ in range(4)]
    for entry in entries:
        bins[_length_bin(int(entry["length"]))].append(entry)
    for rows in bins:
        rows.sort(
            key=lambda row: hashlib.sha256(
                f"{seed}:{row['structure_id']}:{row['label_asym_id']}".encode("utf-8")
            ).hexdigest()
        )

    train_by_bin = _allocate_proportionally(counts["train"], [len(rows) for rows in bins])
    remaining_sizes = [len(rows) - selected for rows, selected in zip(bins, train_by_bin)]
    validation_by_bin = _allocate_proportionally(counts["validation"], remaining_sizes)
    split_rows: dict[str, list[dict[str, str]]] = {
        "train": [],
        "validation": [],
        "test": [],
    }
    for rows, train_count, validation_count in zip(bins, train_by_bin, validation_by_bin):
        boundaries = (train_count, train_count + validation_count)
        groups = (
            ("train", rows[: boundaries[0]]),
            ("validation", rows[boundaries[0] : boundaries[1]]),
            ("test", rows[boundaries[1] :]),
        )
        for split_name, selected in groups:
            split_rows[split_name].extend(
                {
                    "structure_id": str(row["structure_id"]),
                    "label_asym_id": str(row["label_asym_id"]),
                    "cluster_id": str(row["cluster_id"]),
                }
                for row in selected
            )
    for rows in split_rows.values():
        rows.sort(key=lambda row: (row["structure_id"], row["label_asym_id"]))
    result = {
        "seed": seed,
        "unit": split_config["unit"],
        "stratification": "original-protein-length-bin",
        "counts": {name: len(rows) for name, rows in split_rows.items()},
        "splits": split_rows,
    }
    result["content_sha256"] = canonical_json_sha256(result)
    return result


def pair_features(
    hidden: np.ndarray, pairs: Any, selected: np.ndarray | None = None
) -> tuple[np.ndarray, np.ndarray]:
    """Return diagonal-bilinear features for contacts and their matched decoys."""

    indices = np.arange(len(pairs["separation"])) if selected is None else np.asarray(selected)
    values = np.asarray(hidden, dtype=np.float32)
    positive = values[pairs["contact_i"][indices]] * values[pairs["contact_j"][indices]]
    negative = values[pairs["decoy_i"][indices]] * values[pairs["decoy_j"][indices]]
    return positive, negative


def _entry_key(entry: dict[str, Any]) -> tuple[str, str]:
    return str(entry["structure_id"]), str(entry["label_asym_id"])


def _split_entries(
    entries: list[dict[str, Any]], split: dict[str, Any], name: str
) -> list[dict[str, Any]]:
    requested = {
        (str(row["structure_id"]), str(row["label_asym_id"]))
        for row in split["splits"][name]
    }
    selected = [entry for entry in entries if _entry_key(entry) in requested]
    if len(selected) != len(requested):
        raise ValueError(f"split {name} refers to missing representations")
    return selected


def _load_pair_file(root: Path, entry: dict[str, Any]) -> Any:
    return np.load(root / str(entry["pairs_file"]), allow_pickle=False)


def _training_matrix(
    root: Path,
    entries: Iterable[dict[str, Any]],
    stage: int,
    max_pairs: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    features: list[np.ndarray] = []
    labels: list[np.ndarray] = []
    for entry in entries:
        hidden = np.load(root / str(entry["hidden_file"]), mmap_mode="r", allow_pickle=False)[stage]
        with _load_pair_file(root, entry) as pairs:
            count = len(pairs["separation"])
            if count > max_pairs:
                local_seed = stable_seed(seed, str(entry["structure_id"]), str(entry["label_asym_id"]))
                selected = np.random.default_rng(local_seed).choice(
                    count, size=max_pairs, replace=False
                )
            else:
                selected = np.arange(count)
            positive, negative = pair_features(hidden, pairs, selected)
        features.extend((positive, negative))
        labels.extend((np.ones(len(positive), dtype=np.int8), np.zeros(len(negative), dtype=np.int8)))
    return np.concatenate(features), np.concatenate(labels)


def _fit_classifier(
    features: np.ndarray, labels: np.ndarray, alpha: float, seed: int
) -> tuple[Any, Any]:
    try:
        from sklearn.linear_model import SGDClassifier
        from sklearn.preprocessing import StandardScaler
    except ImportError as error:
        raise RuntimeError("install the probe dependencies before fitting") from error

    scaler = StandardScaler()
    standardized = scaler.fit_transform(features)
    classifier = SGDClassifier(
        loss="log_loss",
        penalty="l2",
        alpha=float(alpha),
        max_iter=2000,
        tol=1e-4,
        random_state=seed,
        average=True,
    )
    classifier.fit(standardized, labels)
    return scaler, classifier


def _evaluate_classifier(
    root: Path,
    entries: Iterable[dict[str, Any]],
    stage: int,
    scaler: Any,
    classifier: Any,
    distance_bins: list[list[int]],
) -> dict[str, Any]:
    all_positive: list[np.ndarray] = []
    all_negative: list[np.ndarray] = []
    all_separation: list[np.ndarray] = []
    per_protein: list[dict[str, Any]] = []
    for entry in entries:
        hidden = np.load(root / str(entry["hidden_file"]), mmap_mode="r", allow_pickle=False)[stage]
        with _load_pair_file(root, entry) as pairs:
            positive_features, negative_features = pair_features(hidden, pairs)
            separation = np.asarray(pairs["separation"])
        positive = classifier.decision_function(scaler.transform(positive_features))
        negative = classifier.decision_function(scaler.transform(negative_features))
        per_protein.append(
            {
                "structure_id": str(entry["structure_id"]),
                "label_asym_id": str(entry["label_asym_id"]),
                "matched_pairs": len(positive),
                "auc": roc_auc(positive, negative),
                "matched_concordance": matched_concordance(positive, negative),
            }
        )
        all_positive.append(np.asarray(positive))
        all_negative.append(np.asarray(negative))
        all_separation.append(separation)
    pooled_positive = np.concatenate(all_positive)
    pooled_negative = np.concatenate(all_negative)
    pooled_separation = np.concatenate(all_separation)
    distance_results: list[dict[str, Any]] = []
    for lower, upper in distance_bins:
        selected = (pooled_separation > lower) & (pooled_separation <= upper)
        row: dict[str, Any] = {
            "label": f"({lower},{upper}]",
            "count_per_class": int(selected.sum()),
        }
        if selected.sum() == 0:
            row["status"] = "insufficient"
        else:
            row.update(
                status="ok",
                auc=roc_auc(pooled_positive[selected], pooled_negative[selected]),
                matched_concordance=matched_concordance(
                    pooled_positive[selected], pooled_negative[selected]
                ),
            )
        distance_results.append(row)
    return {
        "matched_pairs": len(pooled_positive),
        "pooled_auc": roc_auc(pooled_positive, pooled_negative),
        "pooled_matched_concordance": matched_concordance(
            pooled_positive, pooled_negative
        ),
        "mean_per_protein_auc": float(np.mean([row["auc"] for row in per_protein])),
        "per_protein": per_protein,
        "distance_bins": distance_results,
    }


def paired_protein_bootstrap(
    contextual: list[dict[str, Any]],
    baseline: list[dict[str, Any]],
    replicates: int,
    confidence_level: float,
    seed: int,
) -> dict[str, float]:
    baseline_by_key = {
        (row["structure_id"], row["label_asym_id"]): float(row["auc"])
        for row in baseline
    }
    differences = np.asarray(
        [
            float(row["auc"]) - baseline_by_key[(row["structure_id"], row["label_asym_id"])]
            for row in contextual
        ],
        dtype=np.float64,
    )
    if len(differences) < 2:
        raise ValueError("paired protein bootstrap requires at least two proteins")
    rng = np.random.default_rng(seed)
    sampled = rng.choice(differences, size=(replicates, len(differences)), replace=True).mean(axis=1)
    tail = (1.0 - confidence_level) / 2.0
    return {
        "mean_auc_difference": float(differences.mean()),
        "confidence_level": float(confidence_level),
        "lower": float(np.quantile(sampled, tail)),
        "upper": float(np.quantile(sampled, 1.0 - tail)),
        "replicates": int(replicates),
    }


def _save_fitted_models(
    path: Path,
    contextual_stage: int,
    contextual_alpha: float,
    contextual_scaler: Any,
    contextual_classifier: Any,
    baseline_alpha: float,
    baseline_scaler: Any,
    baseline_classifier: Any,
) -> None:
    _atomic_save_npz(
        path,
        contextual_stage=np.asarray(contextual_stage),
        contextual_alpha=np.asarray(contextual_alpha),
        contextual_scaler_mean=contextual_scaler.mean_,
        contextual_scaler_scale=contextual_scaler.scale_,
        contextual_coef=contextual_classifier.coef_,
        contextual_intercept=contextual_classifier.intercept_,
        baseline_stage=np.asarray(0),
        baseline_alpha=np.asarray(baseline_alpha),
        baseline_scaler_mean=baseline_scaler.mean_,
        baseline_scaler_scale=baseline_scaler.scale_,
        baseline_coef=baseline_classifier.coef_,
        baseline_intercept=baseline_classifier.intercept_,
    )


def run_hidden_probe(
    config: dict[str, Any], representation_index: Path, output_dir: Path
) -> dict[str, Any]:
    """Select on validation proteins and evaluate once on held-out proteins."""

    index_path = Path(representation_index)
    with index_path.open("r", encoding="utf-8") as handle:
        index = json.load(handle)
    claimed_content_hash = index.get("content_sha256")
    content_without_hash = {key: value for key, value in index.items() if key != "content_sha256"}
    if claimed_content_hash != canonical_json_sha256(content_without_hash):
        raise ValueError("representation index content hash does not match")
    entries = list(index["chains"])
    root = index_path.parent
    for entry in entries:
        hidden_path = root / entry["hidden_file"]
        if sha256_file(hidden_path) != entry["hidden_sha256"]:
            raise ValueError(f"hidden-state hash mismatch for {_entry_key(entry)}")
        if sha256_file(root / entry["pairs_file"]) != entry["pairs_sha256"]:
            raise ValueError(f"pair-cache hash mismatch for {_entry_key(entry)}")
        hidden = np.load(hidden_path, mmap_mode="r", allow_pickle=False)
        expected_shape = (28, int(entry["length"]), 1536)
        if hidden.shape != expected_shape or hidden.dtype != np.float16:
            raise ValueError(f"invalid hidden-state cache for {_entry_key(entry)}")

    seed = int(config["run"]["seed"])
    split = make_probe_splits(entries, config["probe"]["split"], seed)
    expected_split_hash = config["probe"]["split"]["content_sha256"]
    if split["content_sha256"] != expected_split_hash:
        raise ValueError(
            f"frozen split hash {split['content_sha256']} does not match "
            f"{expected_split_hash}"
        )
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    write_json_atomic(output / "splits.json", split)
    train_entries = _split_entries(entries, split, "train")
    validation_entries = _split_entries(entries, split, "validation")
    test_entries = _split_entries(entries, split, "test")
    max_pairs = int(config["probe"]["max_training_matched_pairs_per_protein"])
    bins = config["distance_bins"]["intervals"]

    validation_results: list[dict[str, Any]] = []
    for stage in config["probe"]["stages"]:
        features, labels = _training_matrix(root, train_entries, int(stage), max_pairs, seed)
        for alpha in config["probe"]["regularization_alpha"]:
            scaler, classifier = _fit_classifier(features, labels, float(alpha), seed)
            evaluation = _evaluate_classifier(
                root, validation_entries, int(stage), scaler, classifier, bins
            )
            validation_results.append(
                {
                    "stage": int(stage),
                    "alpha": float(alpha),
                    "classifier_iterations": int(classifier.n_iter_),
                    "validation": evaluation,
                }
            )

    def selection_key(row: dict[str, Any]) -> tuple[float, float, int]:
        return (
            float(row["validation"]["mean_per_protein_auc"]),
            float(row["alpha"]),
            -int(row["stage"]),
        )

    contextual_candidates = [
        row for row in validation_results if row["stage"] in config["probe"]["contextual_stages"]
    ]
    selected_contextual = max(contextual_candidates, key=selection_key)
    selected_baseline = max(
        (row for row in validation_results if row["stage"] == 0), key=selection_key
    )

    refit_entries = train_entries + validation_entries
    contextual_stage = int(selected_contextual["stage"])
    contextual_alpha = float(selected_contextual["alpha"])
    contextual_features, contextual_labels = _training_matrix(
        root, refit_entries, contextual_stage, max_pairs, seed
    )
    contextual_scaler, contextual_classifier = _fit_classifier(
        contextual_features, contextual_labels, contextual_alpha, seed
    )
    baseline_alpha = float(selected_baseline["alpha"])
    baseline_features, baseline_labels = _training_matrix(
        root, refit_entries, 0, max_pairs, seed
    )
    baseline_scaler, baseline_classifier = _fit_classifier(
        baseline_features, baseline_labels, baseline_alpha, seed
    )
    contextual_test = _evaluate_classifier(
        root,
        test_entries,
        contextual_stage,
        contextual_scaler,
        contextual_classifier,
        bins,
    )
    baseline_test = _evaluate_classifier(
        root, test_entries, 0, baseline_scaler, baseline_classifier, bins
    )
    bootstrap_config = config["probe"]["bootstrap"]
    bootstrap = paired_protein_bootstrap(
        contextual_test["per_protein"],
        baseline_test["per_protein"],
        int(bootstrap_config["replicates"]),
        float(bootstrap_config["confidence_level"]),
        seed,
    )
    _save_fitted_models(
        output / "selected_models.npz",
        contextual_stage,
        contextual_alpha,
        contextual_scaler,
        contextual_classifier,
        baseline_alpha,
        baseline_scaler,
        baseline_classifier,
    )
    result = {
        "experiment": "experiment1-hidden-state-follow-up",
        "result_label": config["protocol"]["result_label"],
        "representation_index": str(index_path.resolve()),
        "representation_index_sha256": sha256_file(index_path),
        "split_sha256": split["content_sha256"],
        "selection_metric": config["probe"]["selection_metric"],
        "selected_contextual": {
            "stage": contextual_stage,
            "alpha": contextual_alpha,
            "validation": selected_contextual["validation"],
            "test": contextual_test,
        },
        "stage0_baseline": {
            "stage": 0,
            "alpha": baseline_alpha,
            "validation": selected_baseline["validation"],
            "test": baseline_test,
        },
        "primary_paired_protein_bootstrap": bootstrap,
        "validation_grid": validation_results,
    }
    write_json_atomic(output / "hidden_probe_summary.json", result)
    return result
