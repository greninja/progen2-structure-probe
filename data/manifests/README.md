# Input manifests

`experiment1_150.csv` is the exact frozen public-information replacement cohort used
for the reported attention and hidden-state runs. `experiment1_pilot.csv` contains
the fixed five-chain engineering subset. These are not Mandrake's unpublished
identifiers.

Each row records the cluster, RCSB structure and chain identifiers, resolution,
coordinate coverage, sequence hash, mmCIF hash, and selection seed. Relative
`mmcif_path` values resolve beneath this directory. Materialize and verify the public
structure files with:

```bash
progen2-probe fetch-structures data/manifests/experiment1_150.csv
```

The command is restartable and refuses an existing or downloaded file whose SHA-256
does not match the frozen manifest. The full cohort-construction audit is tracked
under `artifacts/cohort/`.
