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
