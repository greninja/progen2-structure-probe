# Experiment log

## 2026-08-24 — Experiment 1 engineering gate and fallback cohort

No contact-discrimination result was generated in this run. The work completed the
model smoke gate, constructed the predeclared public-information fallback cohort,
and audited the frozen five-chain pilot manifest.

### Model smoke test

The 128-residue deterministic smoke test passed on one RTX 4090 using the isolated
Python 3.8.20, PyTorch 2.0.1, and CUDA 11.8 runtime.

| Check | ProGen2-base | ESM-2 35M |
|---|---:|---:|
| Attention shape | `27 x 16 x 128 x 128` | `12 x 20 x 128 x 128` |
| Hidden-state shape | `28 x 128 x 1536` | Not requested |
| Repeat-run maximum delta | `0.0` | `0.0` |
| Peak allocated GPU memory | 1,866,924,544 bytes | 1,835,224,576 bytes |

The smoke JSON SHA-256 is
`4fb9797829e43d784e3c5d0aaf303c546366cc29f470c537c6ec39c2f1e5cc86`.
This is an engineering result, not evidence of structural signal.

### Fallback cohort construction

- RCSB query retrieval: `2026-08-23T17:58:38.046531+00:00`
- selection seed: `20260822`
- search hits: 96,666 polymer entities
- eligible candidates: 95,589
- MMseqs2 version: `15.6f452`
- MMseqs2 clusters: 12,793
- selected chains: 150 unique clusters and 150 unique sequences
- length-bin counts: 38, 38, 37, and 37 for 100–199, 200–299,
  300–399, and 400–500 residues
- recorded rejections: 2,124
- manifest SHA-256:
  `10b89e0f3a6dccbacf081ca03a18aebc9001e4dd83f1e4c364e15ecbe3022b09`
- pilot manifest SHA-256:
  `7873e16b49516d4ff760f3c0e91989262630ecb66550e76f1ea7d04272a77996`

Rejections comprise 1,077 noncanonical API sequences, 667 structures below 95%
backbone-coordinate coverage, 258 structures with modified/noncanonical mmCIF
monomers, two API/mmCIF sequence mismatches, and 120 otherwise valid representatives
encountered after a length-bin quota was filled. These are declared filters rather
than post-result exclusions.

The first resumed run stopped before clustering because MMseqs2 was installed beside
the isolated Python interpreter but was absent from the parent shell's `PATH`. Commit
`2b76fb3` made executable resolution environment-local. Local tests (24) and remote
Python 3.8 cohort tests (6) passed before the cached run resumed. No cohort criterion,
seed, cached API response, or candidate was changed by this fix.

### Frozen five-chain pilot audit

| Chain | Annotation | Length | Resolution (Å) | Backbone coverage | Matched pairs | Separation range |
|---|---|---:|---:|---:|---:|---:|
| 4B6I:C | SMA2266/Rap2b | 102 | 1.95 | 1.0000 | 126 | 11–89 |
| 6MRO:A | Methyl transferase | 194 | 1.60 | 1.0000 | 411 | 11–169 |
| 9T47:A | SUN4 domain | 295 | 1.10 | 0.9559 | 733 | 11–179 |
| 8OI4:C | GH154 beta-galactosidase | 398 | 1.76 | 0.9824 | 721 | 11–366 |
| 7EYO:A | Hyaluronoglucuronidase | 496 | 1.85 | 0.9798 | 1,181 | 11–476 |

The audit verified:

- the full and pilot manifest hashes against the cohort summary;
- exact replay of the fixed-quantile five-chain selection;
- unique cluster, entity, sequence, and `(structure, chain)` keys;
- every selected mmCIF hash in the 150-chain manifest;
- all four predeclared length quotas;
- pilot sequence hashes, residue counts, and recorded coordinate coverage;
- virtual-Cβ contact construction at `<8 Å` and `|i-j|>10`;
- nonempty exact-separation matched contacts and noncontacts for every pilot chain.

This audit establishes that the pilot is technically runnable under the declared
fallback protocol. It does not establish equivalence to Mandrake's unpublished
cohort and does not provide a contact-attention result.

## 2026-08-24 — Experiment 1 full replacement-cohort run

The existing attention pipeline was run on all 150 frozen chains at project commit
`7355110`. It produced 88,473 exact-sequence-distance-matched contact/decoy pairs.

| Result | ProGen2-base | ESM-2 35M |
|---|---:|---:|
| Pooled ROC-AUC | 0.5374 | 0.6147 |
| Pooled Cohen's d | 0.1501 | 0.3930 |
| Mean per-protein ROC-AUC | 0.5380 | 0.6210 |
| Best layer ROC-AUC | 0.5759 (layer 25) | 0.6834 (layer 11) |
| Best layer Cohen's d | 0.2095 | 0.5213 |

Distance-binned ROC-AUC:

| Separation | Pairs per class | ProGen2-base | ESM-2 35M |
|---|---:|---:|---:|
| (10,20] | 14,255 | 0.5697 | 0.6209 |
| (20,40] | 25,326 | 0.5704 | 0.6712 |
| (40,60] | 14,033 | 0.5512 | 0.6630 |
| (60,100] | 14,031 | 0.5518 | 0.6626 |
| (100,150] | 7,658 | 0.5524 | 0.6238 |
| (150,500] | 13,170 | 0.5113 | 0.5781 |

Mandrake reported pooled AUC values of 0.527 for ProGen2 and 0.611 for ESM-2.
The replacement-cohort result therefore reproduces the main qualitative finding:
ProGen2 attention has weak contact-discrimination signal that approaches random at
long sequence separation, while ESM-2 is consistently stronger. Numerical differences
are expected because Mandrake did not publish its structures or complete method and
reported 38,286 comparisons rather than the 88,473 used here.

Accepted summary SHA-256:
`a7a35aaeaeb2d6b692c1152054b3b050224b71293e37d831234c149ee48304a4`.

## 2026-08-27 — Experiment 1 hidden-state follow-up

This follow-up tested whether ProGen2-base contextual hidden states contain
contact-predictive information beyond the model's non-contextual input embedding.
It reused the frozen 150-chain replacement cohort and the same virtual-Cβ contacts,
`|i-j|>10` filter, and exact-separation matched decoys as Experiment 1.

Proteins were split before fitting into 90 train, 30 validation, and 30 untouched
test proteins. The split hash exactly matched the preregistered value
`1c4a1b025cf34c37f95eff729039814a378e0dff4a5ffe225416682d9afe2647`.
At each of the 28 representation stages, the probe used only the 1,536-dimensional
symmetric elementwise product `h_i * h_j` and an L2-regularized logistic regression.
Layer and regularization strength were selected by mean per-protein validation AUC.

Stage 26 with regularization `alpha=0.0001` was selected as the contextual model.
Stage 0 with `alpha=0.001` was selected independently as the non-contextual baseline.
The final test set contained 18,166 matched contact/decoy units.

| Held-out test result | Stage 0 embedding | Stage 26 contextual |
|---|---:|---:|
| Mean per-protein ROC-AUC | 0.6257 | 0.6738 |
| Pooled ROC-AUC | 0.6131 | 0.6666 |
| Pooled matched concordance | 0.6095 | 0.6687 |

The preregistered primary effect, contextual minus stage-0 mean test-protein AUC,
was **+0.0481**. Its 95% interval from 10,000 paired protein-level bootstrap
resamples was **[+0.0331, +0.0621]**.

Distance-binned pooled test ROC-AUC:

| Separation | Pairs per class | Stage 0 embedding | Stage 26 contextual | Difference |
|---|---:|---:|---:|---:|
| (10,20] | 3,065 | 0.6271 | 0.6721 | +0.0450 |
| (20,40] | 4,492 | 0.6255 | 0.6792 | +0.0537 |
| (40,60] | 2,650 | 0.6093 | 0.6678 | +0.0584 |
| (60,100] | 2,971 | 0.6101 | 0.6888 | +0.0787 |
| (100,150] | 1,844 | 0.6066 | 0.6694 | +0.0628 |
| (150,500] | 3,144 | 0.5914 | 0.6218 | +0.0304 |

The result supports the narrow claim that late ProGen2 contextual representations
contain generalizable contact-predictive information beyond residue embeddings under
this low-capacity supervised probe. The gain remains positive descriptively in the
longest-range bin, although both models weaken there. It does not show that ProGen2
simulates folding, uses this information during generation, or contains a purely
geometric representation. The probe receives contact labels, and the stage-0 AUC of
0.626 shows that residue identity and other sequence-level regularities already carry
substantial contact-predictive signal. Sequence-only controls are therefore required
before making a stronger structural-mechanism claim.

The eight-worker execution changed only scheduling. All 28 stages used the frozen
protocol, and each completed validation stage was stored in an input-bound atomic
checkpoint. The compact result artifacts were copied from the persistent volume and
verified locally. Accepted summary SHA-256:
`31f8c7ac37d7c0cbc0eed77bbdf5104a86e086b1c1cc3e76ba4bc947758ef251`.

The result figure is available as [SVG](figures/hidden_state_probe.svg) and
[PNG](figures/hidden_state_probe.png).
