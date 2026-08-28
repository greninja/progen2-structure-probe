# ProGen2 Structure Probe

This repository first performs a **best-effort reproduction of Experiment 1** from
Mandrake Bio's blog post,
[“Protein Language Models: Fluent, but clueless”](https://research.mandrake.bio/p/protein-language-models-fluent-but).

The experiment asks a simple question: **does a protein language model give higher
attention scores to residue pairs that touch in the folded protein than to pairs
that do not touch?** We reproduced the reported comparison between ProGen2 and
ESM-2, then ran one follow-up asking whether contact information is easier to decode
from ProGen2's hidden states than from its attention scores.

While reproducing the experiment, we made several assumptions about details such
as the protein set, model checkpoints, and scoring procedure. We based these choices
on the information provided in the post and its figures, the official model
implementations, and standard contact-prediction methods. For transparency, all of
our assumptions are listed below.

## Dataset

We used public structures from the
[RCSB Protein Data Bank](https://www.rcsb.org/) (RCSB PDB).

The blog described 150 non-redundant protein structures with resolution at most
2.0 Å and sequence lengths from 100 to 500 residues. We constructed a replacement dataset of **150 protein chains** using
the following rules:

- X-ray structure with resolution ≤ 2.0 Å
- protein length from 100 to 500 residues
- at least 95% of residues have the backbone coordinates needed by the analysis
- sequences clustered at 30% identity and 80% bidirectional coverage
- one chain selected per cluster to reduce near-duplicate proteins
- balanced across four length ranges: 38, 38, 37, and 37 chains in 100–199,
  200–299, 300–399, and 400–500 residues

The RCSB search was recorded on 23 August 2026. The exact 150 identifiers,
sequences, structure paths, and file hashes are tracked in
[`data/manifests/experiment1_150.csv`](data/manifests/experiment1_150.csv).

## Assumptions required for the reproduction

The following details were not fully specified in the blog. These are **our fixed
choices**, not claims about Mandrake's unpublished implementation.

| Missing detail | Choice used here |
|---|---|
| Original 150 structures | Public replacement cohort described above |
| Meaning of “non-redundant” | MMseqs2 clustering at 30% sequence identity and 80% coverage in both directions |
| ProGen2 checkpoint | Official `progen2-base`; 27 layers matched the blog's description |
| ESM-2 checkpoint | Official `esm2_t12_35M_UR50D`; inferred from the blog's 12-layer description |
| Structure handling | First structural model, one PDB polymer chain, intrachain contacts only; modified residues rejected |
| Missing/alternate coordinates | Require ≥95% backbone coverage; select alternate locations by occupancy, then `A`, then lexicographically |
| True contact | Virtual Cβ distance below 8 Å and sequence separation greater than 10 residues |
| Decoy selection | One noncontact per true contact at exactly the same sequence separation, sampled without replacement |
| Randomness | Seed `20260822` |
| Attention values | Post-softmax attention returned by each official model; terminal tokens removed |
| ProGen2's directional attention | Combine the two directions as `A + Aᵀ` |
| Attention correction | Apply APC separately to every layer and attention head, then z-score valid residue pairs |
| Final pair score | Maximum corrected z-score across all layers and heads; inferred from the figure label “Max Z-Score” |
| Distance bins | `(10,20]`, `(20,40]`, `(40,60]`, `(60,100]`, `(100,150]`, `(150,500]` |
| Overall AUC | Pool all selected contact scores and all selected decoy scores across the 150 proteins |

Our replacement cohort produced **88,473 matched contact/decoy units**—each unit is
one true contact plus one decoy. The blog reported 38,286. This difference is
expected because its structures and several sampling details are unavailable; we
did not discard valid pairs merely to force the same count.

## Experiment 1 result

| Pooled ROC-AUC | Blog | This reproduction |
|---|---:|---:|
| ProGen2 | 0.527 | 0.537 |
| ESM-2 | 0.611 | 0.615 |

An AUC of 0.5 means random ranking. Our replacement experiment reproduced the
blog's main qualitative result: **ProGen2 attention was only slightly better than
random, while ESM-2 attention contained a clearer contact signal.** The close
numbers are encouraging, but they should not be presented as an exact numerical
replication because the underlying protein set and some method choices differ.

Detailed plots, per-distance results, and statistical tests are recorded in
[`docs/experiment_log.md`](docs/experiment_log.md). The complete reconstructed
protocol and its limitations are in [`docs/methodology.md`](docs/methodology.md).

## Follow-up: hidden-state probe

We next asked whether ProGen2's hidden representation contains contact information
that is not obvious in attention scores. A small supervised classifier was trained
on frozen residue representations, using separate proteins for training,
validation, and testing (90/30/30 proteins).

| Held-out test result | Mean per-protein ROC-AUC |
|---|---:|
| Input embedding before contextual processing | 0.626 |
| Best validation-selected contextual stage (stage 26) | 0.674 |
| Improvement | **+0.048** (95% protein-bootstrap CI **+0.033 to +0.062**) |

![Hidden-state probe result](docs/figures/hidden_state_probe.png)

This means contact information was more easily decoded after ProGen2 processed the
sequence. It does **not** show that ProGen2 can fold proteins, stores an explicit 3D
map, or uses this information during generation. See
[`docs/hidden-state-probe.md`](docs/hidden-state-probe.md) for the exact probe.

## Reproducing the runs

Create the local environment and run the tests:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e '.[probe,test]'
python -m pytest
```

Download and verify the public structures in the frozen manifest:

```bash
progen2-probe fetch-structures data/manifests/experiment1_150.csv
```

The compact machine-readable results are tracked in [`artifacts/`](artifacts/).
Large model checkpoints, downloaded structures, attention arrays, and the hidden-
state cache are regenerated rather than stored in Git. End-to-end GPU commands are
in [`docs/runpod.md`](docs/runpod.md).

## Repository map

- [`data/manifests/`](data/manifests/) — exact public protein cohorts
- [`configs/`](configs/) — frozen experiment settings
- [`artifacts/`](artifacts/) — compact results and provenance
- [`docs/experiment_log.md`](docs/experiment_log.md) — results and interpretations
- [`src/progen2_structure_probe/`](src/progen2_structure_probe/) — experiment code
