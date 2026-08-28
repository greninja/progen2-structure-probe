# Protein dataset

`experiment1_150.csv` lists the 150 public RCSB proteins used in both experiments.
It includes the protein identifiers, sequences, structure details, and information
needed to check the downloaded files.

Download the structures with:

```bash
progen2-probe fetch-structures data/manifests/experiment1_150.csv
```

The command checks that every downloaded structure is the expected file. It can be
run again if a download is interrupted.
