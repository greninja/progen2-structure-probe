# Probing Structural Information in ProGen2

Does ProGen2 fail to encode protein structure, or is structural information simply
hard to see in its attention weights?

This repository tests that question with a small, reproducible study of residue
contacts. It is motivated by Mandrake Bio's
["Protein Language Models: Fluent, but clueless"](https://research.mandrake.bio/p/protein-language-models-fluent-but),
which found that ProGen2 attention weakly distinguishes residues that contact in 3D.
Here, that attention analysis is reconstructed and extended to ProGen2's hidden-state
representations.

## Approach

The study uses a frozen replacement cohort of 150 non-redundant, experimentally
determined protein chains. A residue pair is labeled as a contact when its virtual
Cβ distance is below 8 Å and the residues are more than 10 positions apart in the
sequence. Each contact is paired with a non-contact at exactly the same sequence
separation, preventing ordinary locality bias from masquerading as structural signal.

Two complementary measurements are made:

1. **Attention baseline.** APC-corrected attention from ProGen2-base is evaluated for
   contact discrimination, with ESM-2 as a bidirectional comparison model.
2. **Hidden-state probe.** Frozen representations from every ProGen2-base stage are
   tested with a deliberately limited L2-regularized logistic probe. Proteins, rather
   than residue pairs, are split across training, validation, and test sets. The input
   embedding is evaluated as a baseline for the added value of contextualization.

The probe measures whether contact information is *decodable*. It does not establish
that ProGen2 performs folding, represents explicit 3D geometry, or uses the decoded
signal during generation.

## Results

The 150-chain attention run reproduced the blog post's qualitative result: raw
ProGen2 attention was only weakly contact-discriminative, while ESM-2 was stronger.

| Attention result | ProGen2-base | ESM-2 35M |
|---|---:|---:|
| Pooled ROC-AUC | 0.5374 | 0.6147 |

On the frozen 90/30/30 protein split, the hidden-state follow-up selected ProGen2
stage 26 using validation proteins and evaluated it once on 30 untouched proteins.

| Hidden-state test result | Mean per-protein ROC-AUC |
|---|---:|
| Stage 0 input embedding | 0.6257 |
| Stage 26 contextual representation | 0.6738 |
| Paired improvement | **+0.0481** (95% protein bootstrap CI **+0.0331 to +0.0621**) |

![Hidden-state probe result](docs/figures/hidden_state_probe.png)

The attention and hidden-state AUC values are not direct competitors: attention is
an untrained pair score, whereas the hidden-state result comes from a supervised
probe. The confirmatory comparison is stage 26 versus the identically trained stage
0 baseline. Detailed results and interpretation limits are in
[`docs/experiment_log.md`](docs/experiment_log.md).

## Local development

Python 3.8 or newer is required. From a checkout of the repository:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e '.[probe,test]'
python -m pytest
```

The unit tests use synthetic fixtures and do not require a GPU, model checkpoint, or
protein-structure download. Experiment configurations can also be checked locally:

```bash
progen2-probe validate-config configs/experiment1_pilot.yaml
progen2-probe validate-config configs/experiment1_hidden_probe.yaml
```

Full inference uses pinned versions of the official
[ProGen2](https://github.com/salesforce/progen) and
[ESM](https://github.com/facebookresearch/esm) implementations and is designed for a
CUDA machine.

## Exact inputs and machine-readable results

The exact 150-chain replacement cohort and five-chain pilot are tracked under
[`data/manifests/`](data/manifests/). Download the public RCSB structures named by
the frozen manifest and verify every recorded file hash with:

```bash
progen2-probe fetch-structures data/manifests/experiment1_150.csv
```

Compact outputs for the cohort, attention run, and hidden-state probe are tracked
under [`artifacts/`](artifacts/). They include full JSON summaries, the frozen split,
representation-cache hashes, runtime provenance, and selected probe coefficients.
Large model checkpoints, downloaded mmCIF structures, per-chain attention arrays,
and the 3.5 GiB hidden-state cache remain untracked because the pinned workflow can
regenerate them. See [`docs/runpod.md`](docs/runpod.md) for the end-to-end commands.

## Repository guide

- [`docs/methodology.md`](docs/methodology.md) documents the attention experiment and
  the limits of reconstructing an unpublished cohort and protocol.
- [`docs/hidden-state-probe.md`](docs/hidden-state-probe.md) defines the frozen probe,
  controls, split policy, and interpretation boundaries.
- [`configs/`](configs/) contains versioned experiment configurations.
- [`data/manifests/`](data/manifests/) contains the exact frozen public-structure
  cohort used for the reported runs.
- [`artifacts/`](artifacts/) contains compact machine-readable inputs, outputs, and
  provenance needed to audit the reported numbers.
- [`src/progen2_structure_probe/`](src/progen2_structure_probe/) contains the data,
  model-adapter, scoring, and probing pipelines.
- [`docs/runpod.md`](docs/runpod.md) covers reproducible remote execution and model
  setup.

Runs record resolved configurations, source revisions, hashes, and runtime provenance
so that engineering changes remain distinguishable from scientific changes.
