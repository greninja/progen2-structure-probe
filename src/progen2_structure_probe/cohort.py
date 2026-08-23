"""Reproducible construction of the Experiment 1 replacement cohort."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
from http.client import IncompleteRead, RemoteDisconnected
import json
from pathlib import Path
import subprocess
import time
from typing import Any, Iterable, Optional
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import numpy as np

from .artifacts import sha256_file, write_json_atomic
from .mmcif import load_polymer_chain


SEARCH_URL = "https://search.rcsb.org/rcsbsearch/v2/query"
GRAPHQL_URL = "https://data.rcsb.org/graphql"
MMCIF_URL = "https://files.rcsb.org/download/{structure_id}.cif"
CANONICAL_AA = frozenset("ACDEFGHIKLMNPQRSTVWY")


@dataclass(frozen=True)
class Candidate:
    entity_id: str
    structure_id: str
    label_asym_ids: tuple[str, ...]
    sequence: str
    resolution: float

    @property
    def length(self) -> int:
        return len(self.sequence)

    @property
    def sequence_sha256(self) -> str:
        return hashlib.sha256(self.sequence.encode("ascii")).hexdigest()


def rcsb_search_payload(
    min_length: int = 100,
    max_length: int = 500,
    maximum_resolution: float = 2.0,
) -> dict[str, Any]:
    """Return the exact public RCSB query used before independent clustering."""

    def terminal(attribute: str, operator: str, value: Any) -> dict[str, Any]:
        return {
            "type": "terminal",
            "service": "text",
            "parameters": {"attribute": attribute, "operator": operator, "value": value},
        }

    return {
        "query": {
            "type": "group",
            "logical_operator": "and",
            "nodes": [
                terminal("entity_poly.rcsb_entity_polymer_type", "exact_match", "Protein"),
                terminal(
                    "entity_poly.rcsb_sample_sequence_length",
                    "range",
                    {
                        "from": min_length,
                        "include_lower": True,
                        "to": max_length,
                        "include_upper": True,
                    },
                ),
                terminal("exptl.method", "exact_match", "X-RAY DIFFRACTION"),
                terminal(
                    "rcsb_entry_info.resolution_combined",
                    "less_or_equal",
                    maximum_resolution,
                ),
            ],
        },
        "request_options": {"return_all_hits": True, "results_verbosity": "compact"},
        "return_type": "polymer_entity",
    }


def _post_json(url: str, payload: dict[str, Any], attempts: int = 5) -> dict[str, Any]:
    body = json.dumps(payload, sort_keys=True).encode("utf-8")
    for attempt in range(attempts):
        request = Request(
            url,
            data=body,
            headers={
                "Content-Type": "application/json",
                "User-Agent": "progen2-structure-probe/0.1",
            },
            method="POST",
        )
        try:
            with urlopen(request, timeout=120) as response:
                return json.load(response)
        except (
            HTTPError,
            URLError,
            TimeoutError,
            ConnectionResetError,
            IncompleteRead,
            RemoteDisconnected,
        ):
            if attempt + 1 == attempts:
                raise
            time.sleep(2**attempt)
    raise AssertionError("unreachable")


def query_entity_ids(
    query_path: Path,
    response_path: Path,
    min_length: int = 100,
    max_length: int = 500,
    maximum_resolution: float = 2.0,
) -> list[str]:
    payload = rcsb_search_payload(min_length, max_length, maximum_resolution)
    if response_path.exists() and not query_path.exists():
        raise ValueError("cached RCSB response is missing its query record")
    if query_path.exists():
        with query_path.open("r", encoding="utf-8") as handle:
            previous_payload = json.load(handle)
        if previous_payload != payload:
            raise ValueError("cached RCSB response belongs to a different search query")
    else:
        write_json_atomic(query_path, payload)
    if response_path.exists():
        with response_path.open("r", encoding="utf-8") as handle:
            response = json.load(handle)
    else:
        response = _post_json(SEARCH_URL, payload)
        write_json_atomic(response_path, response)
        write_json_atomic(
            response_path.with_name("rcsb_retrieval.json"),
            {
                "retrieved_at_utc": datetime.now(timezone.utc).isoformat(),
                "search_url": SEARCH_URL,
                "data_api_url": GRAPHQL_URL,
            },
        )
    identifiers = response.get("result_set")
    if not isinstance(identifiers, list) or not identifiers:
        raise ValueError("RCSB search returned no polymer entities")
    if not all(isinstance(identifier, str) for identifier in identifiers):
        raise ValueError("compact RCSB response did not contain string identifiers")
    return sorted(set(identifiers))


ENTITY_QUERY = """
query CandidateEntities($ids: [String!]!) {
  polymer_entities(entity_ids: $ids) {
    rcsb_id
    entity_poly {
      pdbx_seq_one_letter_code_can
      rcsb_entity_polymer_type
      rcsb_sample_sequence_length
    }
    rcsb_polymer_entity_container_identifiers {
      entry_id
      asym_ids
    }
    entry {
      rcsb_entry_info { resolution_combined }
      exptl { method }
    }
  }
}
"""


def _chunks(values: list[str], size: int) -> Iterable[list[str]]:
    for start in range(0, len(values), size):
        yield values[start : start + size]


def fetch_candidates(
    entity_ids: list[str],
    batch_size: int = 100,
    raw_directory: Optional[Path] = None,
    min_length: int = 100,
    max_length: int = 500,
    maximum_resolution: float = 2.0,
    rejections: Optional[list[dict[str, Any]]] = None,
) -> list[Candidate]:
    candidates: list[Candidate] = []
    if raw_directory is not None:
        Path(raw_directory).mkdir(parents=True, exist_ok=True)
    for batch_index, batch in enumerate(_chunks(entity_ids, batch_size)):
        payload = {"query": ENTITY_QUERY, "variables": {"ids": batch}}
        response_path = (
            Path(raw_directory) / f"batch_{batch_index:05d}_response.json"
            if raw_directory is not None
            else None
        )
        request_path = (
            response_path.with_name(f"batch_{batch_index:05d}_request.json")
            if response_path is not None
            else None
        )
        if response_path is not None and response_path.exists():
            if request_path is None or not request_path.exists():
                raise ValueError(f"cached Data API response lacks request: {response_path}")
            with request_path.open("r", encoding="utf-8") as handle:
                if json.load(handle) != payload:
                    raise ValueError(f"cached Data API request does not match: {request_path}")
            with response_path.open("r", encoding="utf-8") as handle:
                response = json.load(handle)
        else:
            response = _post_json(GRAPHQL_URL, payload)
            if response_path is not None:
                write_json_atomic(request_path, payload)
                write_json_atomic(response_path, response)
        if response.get("errors"):
            raise ValueError(f"RCSB GraphQL error: {response['errors']}")
        entities = response.get("data", {}).get("polymer_entities")
        if not isinstance(entities, list):
            raise ValueError("RCSB GraphQL response is missing polymer_entities")
        for entity in entities:
            polymer = entity["entity_poly"]
            identifiers = entity["rcsb_polymer_entity_container_identifiers"]
            entry = entity["entry"]
            sequence = "".join(polymer["pdbx_seq_one_letter_code_can"].split()).upper()
            methods = {item["method"] for item in entry.get("exptl", [])}
            resolutions = entry["rcsb_entry_info"].get("resolution_combined") or []
            reason: Optional[str] = None
            if polymer["rcsb_entity_polymer_type"] != "Protein":
                reason = "entity is not a protein"
            elif "X-RAY DIFFRACTION" not in methods:
                reason = "entry does not include X-ray diffraction"
            elif not resolutions or min(resolutions) > maximum_resolution:
                reason = "entry has no qualifying resolution"
            elif not min_length <= len(sequence) <= max_length:
                reason = "canonical sequence length is outside configured bounds"
            elif set(sequence) - CANONICAL_AA:
                reason = "canonical sequence contains noncanonical residue letters"
            asym_ids = tuple(sorted(identifiers.get("asym_ids") or []))
            if reason is None and not asym_ids:
                reason = "entity has no label asym IDs"
            if reason is not None:
                if rejections is not None:
                    rejections.append(
                        {
                            "stage": "data_api_filter",
                            "entity_id": entity.get("rcsb_id"),
                            "structure_id": identifiers.get("entry_id"),
                            "label_asym_id": "|".join(asym_ids),
                            "reason": reason,
                        }
                    )
                continue
            candidates.append(
                Candidate(
                    entity_id=entity["rcsb_id"],
                    structure_id=identifiers["entry_id"].upper(),
                    label_asym_ids=asym_ids,
                    sequence=sequence,
                    resolution=float(min(resolutions)),
                )
            )
    return sorted(candidates, key=lambda item: item.entity_id)


def write_candidates(candidates: list[Candidate], directory: Path) -> tuple[Path, Path]:
    root = Path(directory)
    root.mkdir(parents=True, exist_ok=True)
    csv_path = root / "candidates.csv"
    fasta_path = root / "candidates.fasta"
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        fieldnames = [
            "entity_id", "structure_id", "label_asym_ids", "length",
            "resolution", "sequence", "sequence_sha256",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for candidate in candidates:
            writer.writerow(
                {
                    "entity_id": candidate.entity_id,
                    "structure_id": candidate.structure_id,
                    "label_asym_ids": "|".join(candidate.label_asym_ids),
                    "length": candidate.length,
                    "resolution": candidate.resolution,
                    "sequence": candidate.sequence,
                    "sequence_sha256": candidate.sequence_sha256,
                }
            )
    with fasta_path.open("w", encoding="ascii") as handle:
        for candidate in candidates:
            handle.write(f">{candidate.entity_id}\n{candidate.sequence}\n")
    return csv_path, fasta_path


def run_mmseqs(
    fasta_path: Path,
    directory: Path,
    expected_version: str = "15.6f452",
    threads: int = 8,
    sequence_identity: float = 0.30,
    bidirectional_coverage: float = 0.80,
) -> Path:
    version = subprocess.check_output(["mmseqs", "version"], text=True).strip()
    if version != expected_version:
        raise ValueError(f"MMseqs2 version {version!r} does not match {expected_version!r}")
    root = Path(directory)
    prefix = root / "sequence_clusters"
    temporary = root / "mmseqs_tmp"
    clusters = root / "sequence_clusters_cluster.tsv"
    input_record = {
        "fasta_sha256": sha256_file(Path(fasta_path)),
        "mmseqs_version": version,
        "sequence_identity": sequence_identity,
        "bidirectional_coverage": bidirectional_coverage,
        "cov_mode": 0,
    }
    input_record_path = root / "mmseqs_input.json"
    if clusters.exists():
        if not input_record_path.exists():
            raise ValueError("cached MMseqs2 clusters lack their input record")
        with input_record_path.open("r", encoding="utf-8") as handle:
            if json.load(handle) != input_record:
                raise ValueError("cached MMseqs2 clusters belong to different inputs")
        return clusters
    write_json_atomic(input_record_path, input_record)
    subprocess.run(
        [
            "mmseqs", "easy-cluster", str(fasta_path), str(prefix), str(temporary),
            "--min-seq-id", str(sequence_identity),
            "-c", str(bidirectional_coverage), "--cov-mode", "0",
            "--threads", str(threads),
        ],
        check=True,
    )
    if not clusters.exists():
        raise ValueError("MMseqs2 did not create the expected cluster table")
    return clusters


def read_clusters(path: Path) -> dict[str, list[str]]:
    clusters: dict[str, list[str]] = {}
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            columns = line.rstrip("\n").split("\t")
            if len(columns) != 2:
                raise ValueError(f"invalid MMseqs cluster row {line_number}")
            clusters.setdefault(columns[0], []).append(columns[1])
    return clusters


def _download(url: str, destination: Path, attempts: int = 5) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".part")
    for attempt in range(attempts):
        try:
            request = Request(url, headers={"User-Agent": "progen2-structure-probe/0.1"})
            with urlopen(request, timeout=180) as response, temporary.open("wb") as handle:
                while True:
                    chunk = response.read(1024 * 1024)
                    if not chunk:
                        break
                    handle.write(chunk)
            temporary.replace(destination)
            return
        except (HTTPError, URLError, TimeoutError):
            temporary.unlink(missing_ok=True)
            if attempt + 1 == attempts:
                raise
            time.sleep(2**attempt)


def _length_bin(length: int) -> int:
    for index, (lower, upper) in enumerate(((100, 199), (200, 299), (300, 399), (400, 500))):
        if lower <= length <= upper:
            return index
    raise ValueError(f"length {length} is outside the cohort bounds")


def select_cohort(
    candidates: list[Candidate],
    clusters: dict[str, list[str]],
    directory: Path,
    target_count: int = 150,
    seed: int = 20260822,
    minimum_coordinate_coverage: float = 0.95,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Choose deterministic cluster representatives and record every rejection."""

    by_entity = {candidate.entity_id: candidate for candidate in candidates}
    cluster_ids = sorted(clusters)
    np.random.default_rng(seed).shuffle(cluster_ids)
    quotas = [target_count // 4] * 4
    for index in range(target_count % 4):
        quotas[index] += 1
    selected_by_bin: list[list[dict[str, Any]]] = [[] for _ in range(4)]
    rejections: list[dict[str, Any]] = []
    mmcif_dir = Path(directory) / "mmcif"

    for cluster_id in cluster_ids:
        if all(len(rows) >= quota for rows, quota in zip(selected_by_bin, quotas)):
            break
        members = [by_entity[item] for item in clusters[cluster_id] if item in by_entity]
        members.sort(key=lambda item: (item.resolution, item.structure_id, item.entity_id))
        chosen: Optional[dict[str, Any]] = None
        for resolution in sorted({member.resolution for member in members}):
            valid_at_resolution: list[dict[str, Any]] = []
            for member in (item for item in members if item.resolution == resolution):
                mmcif_path = mmcif_dir / f"{member.structure_id}.cif"
                if not mmcif_path.exists():
                    _download(
                        MMCIF_URL.format(structure_id=member.structure_id.lower()), mmcif_path
                    )
                for asym_id in member.label_asym_ids:
                    try:
                        chain = load_polymer_chain(mmcif_path, asym_id)
                        if chain.sequence != member.sequence:
                            raise ValueError("RCSB API and mmCIF polymer sequences differ")
                        coverage = float(chain.valid_backbone.mean())
                        if coverage < minimum_coordinate_coverage:
                            raise ValueError(
                                f"backbone coordinate coverage {coverage:.4f} is below threshold"
                            )
                    except ValueError as error:
                        rejections.append(
                            {
                                "stage": "structure_filter",
                                "cluster_id": cluster_id,
                                "entity_id": member.entity_id,
                                "structure_id": member.structure_id,
                                "label_asym_id": asym_id,
                                "reason": str(error),
                            }
                        )
                        continue
                    valid_at_resolution.append(
                        {
                            "cluster_id": cluster_id,
                            "entity_id": member.entity_id,
                            "structure_id": member.structure_id,
                            "label_asym_id": asym_id,
                            "length": len(chain.sequence),
                            "resolution": resolution,
                            "coordinate_coverage": coverage,
                            "sequence_sha256": member.sequence_sha256,
                            "mmcif_path": f"mmcif/{member.structure_id}.cif",
                            "mmcif_sha256": sha256_file(mmcif_path),
                            "selection_seed": seed,
                        }
                    )
            if valid_at_resolution:
                chosen = sorted(
                    valid_at_resolution,
                    key=lambda item: (
                        -item["coordinate_coverage"],
                        item["structure_id"],
                        item["label_asym_id"],
                    ),
                )[0]
                break
        if chosen is None:
            continue
        bin_index = _length_bin(chosen["length"])
        if len(selected_by_bin[bin_index]) < quotas[bin_index]:
            selected_by_bin[bin_index].append(chosen)
        else:
            rejections.append(
                {
                    "stage": "stratified_selection",
                    "cluster_id": cluster_id,
                    "entity_id": chosen["entity_id"],
                    "structure_id": chosen["structure_id"],
                    "label_asym_id": chosen["label_asym_id"],
                    "reason": f"length-bin {bin_index} quota already filled",
                }
            )

    selected = [row for rows in selected_by_bin for row in rows]
    if len(selected) != target_count:
        counts = [len(rows) for rows in selected_by_bin]
        raise ValueError(f"could select only {len(selected)} chains; bin counts are {counts}")
    selected.sort(key=lambda item: (item["length"], item["structure_id"], item["label_asym_id"]))
    return selected, rejections


def write_manifest(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError("cannot write an empty manifest")
    with Path(path).open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def select_five_chain_pilot(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    ordered = sorted(rows, key=lambda item: (int(item["length"]), item["structure_id"], item["label_asym_id"]))
    indices = [0, round(0.25 * (len(ordered) - 1)), round(0.5 * (len(ordered) - 1)), round(0.75 * (len(ordered) - 1)), len(ordered) - 1]
    return [ordered[index] for index in indices]


def build_cohort(config: dict[str, Any], directory: Path, threads: int = 8) -> dict[str, Any]:
    root = Path(directory)
    root.mkdir(parents=True, exist_ok=True)
    cohort_config = config["cohort"]
    protocol = {
        "min_length": int(cohort_config["min_length"]),
        "max_length": int(cohort_config["max_length"]),
        "maximum_resolution_angstrom": float(
            cohort_config["maximum_resolution_angstrom"]
        ),
        "target_count": int(cohort_config["target_count"]),
        "minimum_coordinate_coverage": float(
            cohort_config["minimum_coordinate_coverage"]
        ),
        "mmseqs_version": str(cohort_config["mmseqs_version"]),
        "sequence_identity": float(cohort_config["sequence_identity"]),
        "bidirectional_coverage": float(cohort_config["bidirectional_coverage"]),
        "mmseqs_cov_mode": 0,
        "entity_query_sha256": hashlib.sha256(ENTITY_QUERY.encode("utf-8")).hexdigest(),
        "selection_seed": int(config["run"]["seed"]),
    }
    protocol_path = root / "cohort_protocol.json"
    if protocol_path.exists():
        with protocol_path.open("r", encoding="utf-8") as handle:
            if json.load(handle) != protocol:
                raise ValueError(
                    "work directory contains artifacts from a different cohort protocol"
                )
    else:
        write_json_atomic(protocol_path, protocol)
    ids = query_entity_ids(
        root / "rcsb_query.json",
        root / "rcsb_response.json",
        min_length=protocol["min_length"],
        max_length=protocol["max_length"],
        maximum_resolution=protocol["maximum_resolution_angstrom"],
    )
    candidate_rejections: list[dict[str, Any]] = []
    candidates = fetch_candidates(
        ids,
        raw_directory=root / "rcsb_data_api",
        min_length=protocol["min_length"],
        max_length=protocol["max_length"],
        maximum_resolution=protocol["maximum_resolution_angstrom"],
        rejections=candidate_rejections,
    )
    candidates_csv, fasta = write_candidates(candidates, root)
    clusters_path = run_mmseqs(
        fasta,
        root,
        expected_version=protocol["mmseqs_version"],
        threads=threads,
        sequence_identity=protocol["sequence_identity"],
        bidirectional_coverage=protocol["bidirectional_coverage"],
    )
    clusters = read_clusters(clusters_path)
    selected, selection_rejections = select_cohort(
        candidates,
        clusters,
        root,
        target_count=protocol["target_count"],
        seed=protocol["selection_seed"],
        minimum_coordinate_coverage=protocol["minimum_coordinate_coverage"],
    )
    manifest = root / "experiment1_150.csv"
    pilot = root / "experiment1_pilot.csv"
    write_manifest(manifest, selected)
    write_manifest(pilot, select_five_chain_pilot(selected))
    rejections = candidate_rejections + selection_rejections
    write_json_atomic(root / "rejections.json", rejections)
    summary = {
        "query_sha256": sha256_file(root / "rcsb_query.json"),
        "protocol_sha256": sha256_file(protocol_path),
        "response_sha256": sha256_file(root / "rcsb_response.json"),
        "candidate_csv_sha256": sha256_file(candidates_csv),
        "cluster_tsv_sha256": sha256_file(clusters_path),
        "manifest_sha256": sha256_file(manifest),
        "pilot_manifest_sha256": sha256_file(pilot),
        "search_entity_count": len(ids),
        "eligible_candidate_count": len(candidates),
        "cluster_count": len(clusters),
        "selected_count": len(selected),
        "rejection_count": len(rejections),
    }
    write_json_atomic(root / "cohort_summary.json", summary)
    return summary
