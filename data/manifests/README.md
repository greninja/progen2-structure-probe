# Input manifests

These manifests are deliberately not populated with guessed biological identifiers.

An externally recovered `experiment1_pilot.csv` must contain at least:

```text
structure_id,label_asym_id,mmcif_path
```

The fallback cohort builder creates richer `experiment1_150.csv` and
`experiment1_pilot.csv` manifests containing the cluster, resolution, coordinate
coverage, sequence hash, mmCIF hash, and selection seed. The Experiment 1 runner
accepts these extra provenance columns and verifies each available mmCIF hash.
