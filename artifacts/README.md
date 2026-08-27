# Reproducibility artifacts

This directory contains the compact machine-readable records needed to audit the
reported results without committing multi-gigabyte model and representation files.

- `cohort/` records the RCSB retrieval, resolved selection protocol, rejection audit,
  runtime provenance, and hashes of the exact manifests under `data/manifests/`.
- `experiment1-attention/` contains the complete 150-chain attention summary,
  resolved configuration, model/source hashes, and runtime provenance.
- `experiment1-hidden-probe/` contains the complete validation and held-out test
  summary, frozen 90/30/30 split, selected model coefficients, representation-cache
  index and hashes, extraction record, resolved configuration, and runtime
  provenance.

Not stored in Git:

- official ProGen2 and ESM checkpoints;
- downloaded public RCSB mmCIF files;
- per-chain attention arrays;
- the approximately 3.5 GiB float16 hidden-state cache.

The pinned source revisions and commands in `docs/runpod.md` regenerate those large
artifacts. `progen2-probe fetch-structures data/manifests/experiment1_150.csv`
downloads the exact public structures and verifies their recorded SHA-256 hashes.
