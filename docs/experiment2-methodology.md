# Mandrake Experiment 2 replication protocol

Status: implementation scaffold; no model result has been produced.  
Protocol version: 0.1 (2026-08-23)

## Objective

Reproduce two copy-bias demonstrations from Mandrake's post:

1. compare mean per-position perplexity for the first and second copies in `ABCABC`;
2. prompt with one full protein plus the first 25% of a repeat, generate 500 residues,
   and measure identity to the infinite periodic continuation of the source protein.

These tests measure copying behavior. They do not establish why a particular circuit
produces it and do not show that the model lacks all structural information.

## Recovered from the post and original figures

- The figure footer specifies ProGen2-small (151M parameters).
- Perplexity uses six named real proteins and six random strings.
- The plot axis says `Mean Per-Position Perplexity`; the underlying aggregation code
  is not public.
- Generation uses seven proteins of reported lengths 129–240, a full sequence plus
  the first 25% as the prompt, and 500 generated residues.
- Generation compares greedy decoding with nucleus sampling at `top-p=0.95` and
  `temperature=0.8`.
- BSA is identified as UniProt P02769; the example shows 190 residues and a 47-residue
  repeat prefix, consistent with flooring 25%.

Published perplexity bars:

| Input | First copy | Second copy |
|---|---:|---:|
| Lysozyme | 19.0 | 2.05 |
| GFP | 21.0 | 1.24 |
| Ubiquitin | 2.9 | 1.02 |
| Insulin B-chain | 23.4 | 3.37 |
| Thioredoxin | 8.8 | 4.33 |
| Calmodulin | 2.3 | 4.10 |
| Random 1 | 25.9 | 1.89 |
| Random 2 | 26.0 | 1.67 |
| Random 3 | 27.1 | 1.85 |
| Random 4 | 27.3 | 1.85 |
| Random 5 | 25.9 | 1.59 |
| Random 6 | 24.6 | 1.32 |

The figure therefore does not support an unqualified claim that repetition always
reduces perplexity for real proteins: Calmodulin's displayed second-copy value is
higher, and several real-protein values fall outside the prose's stated ranges.

Published generation identity percentages:

| Protein | Length | Greedy | Nucleus |
|---|---:|---:|---:|
| HEWL | 129 | 100% | 100% |
| BSA | 190 | 100% | 100% |
| HbA alpha | 142 | 100% | 100% |
| AdK | 214 | 75% | 63% |
| p53 | 177 | 100% | 100% |
| BglB | 237 | 99% | 96% |
| AtpA | 240 | 100% | 100% |

## Still missing

- exact sequence identifiers and sequence versions for every protein except the
  displayed BSA identifier;
- whether signal peptides, initiator methionines, chains, or mature forms were used;
- exact random-string lengths, amino-acid distribution, and random seed;
- checkpoint archive hash;
- treatment of ProGen2 control tokens in perplexity;
- arithmetic versus geometric aggregation beyond the axis wording;
- stochastic sample count and seed;
- whether “500 generated” means new residues or total token length;
- stopping-token handling and the exact copy-identity denominator.

An exact reproduction is blocked until the sequence manifests are recovered. The
runner refuses unhashed or length-mismatched sequences.

## Documented fallback implemented in configuration

The public-information partial reproduction uses the official ProGen2-small release,
literal `1`/`2` control tokens, arithmetic mean tokenwise perplexity, uniform canonical
random strings matched to real-sequence lengths, floor rounding for the 25% prefix,
500 new residues, one seeded nucleus sample per protein, and identity against the
periodic source continuation at the correct prompt offset. Every one of these choices
is recorded in `configs/experiment2_reproduction.yaml` and must not be described as
Mandrake's unpublished method.

## Static source artifacts

| Asset | Original URL | Bytes | SHA-256 |
|---|---|---:|---|
| Duplication perplexity | [PNG](https://substack-post-media.s3.amazonaws.com/public/images/a3110d9f-84cf-4c96-9c12-58ac2cdaf5c9_4211x1691.png) | 361,665 | `ea0b2fdd83837a2d888cc20b653db4b9e705f808e965cf620737d054e9b58b3f` |
| Repeat-primed generation | [PNG](https://substack-post-media.s3.amazonaws.com/public/images/d5551859-30aa-47d0-a9fc-28ef1edbd645_3953x2547.png) | 568,073 | `093909a8e92eb0f4a583c1ba55a81fcc9428cd5ae4bf98a613d587df7a1b59dd` |

