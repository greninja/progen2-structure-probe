# Input manifests

These manifests are deliberately not populated with guessed biological identifiers.

`experiment1_pilot.csv` must contain:

```text
structure_id,label_asym_id,mmcif_path
```

`experiment2_perplexity.csv` and `experiment2_generation.csv` must contain:

```text
name,source_id,sequence,sequence_sha256
```

Every sequence must use the 20 canonical amino-acid letters. `sequence_sha256` is
the SHA-256 of the uppercase sequence bytes. Experiment 2 execution refuses missing
or mismatched hashes. Mandrake did not publish the exact sequence identifiers, except
that the generation figure labels BSA as UniProt P02769; all other identifiers must
remain unresolved until recovered or explicitly selected as fallback inputs.

