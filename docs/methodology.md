# Mandrake Experiment 1 replication protocol

Status: public-information partial reproduction completed on the 150-chain replacement cohort.
Protocol version: 0.3 (2026-08-22)  
Scope: reproduce the attention-contact experiment in Mandrake Bio's post, *Protein Language Models: Fluent, but clueless*.

## 1. Immediate objective

The only current experimental objective is:

> Reproduce, as closely as public information permits, Mandrake's finding that APC-corrected ProGen2 attention weakly discriminates true intraprotein residue contacts from sequence-distance-matched noncontacts, with ESM-2 as a comparison.

Hidden-state probing is deferred. We will not train a contact probe, propose a new benchmark, or interpret decodability until the attention baseline is working and its differences from Mandrake are understood.

An exact numerical reproduction is possible only if Mandrake's missing artifacts are recovered. Without them, the result must be labeled **public-information partial reproduction**, not exact reproduction.

## 2. Target results from the blog post

Source: [Mandrake Experiment 1](https://research.mandrake.bio/p/protein-language-models-fluent-but), inspected 2026-08-22.

The replication should attempt to recover these reported outputs:

| Target | Mandrake report |
|---|---|
| Dataset | 150 nonredundant X-ray structures, resolution ≤2.0 Å, length 100–500 residues |
| Comparisons | 38,286 true-contact/decoy pairs |
| Negative matching | Decoy is a noncontact at the same sequence distance as the true contact |
| Correction | APC applied to attention-derived pair scores |
| ProGen2 global AUC | 0.527 |
| ProGen2 maximum Cohen's d | 0.184 |
| ProGen2 long-range result | Last two bins, beyond about 100 residues, reported nonsignificant with p=0.468 and p=0.223 |
| ESM-2 global AUC | 0.611 |
| ESM-2 maximum Cohen's d | About 0.52 |
| ESM-2 long-range result | All bins significant, including p<10⁻¹⁰ for 150–500 separation |
| Layer result | ESM-2 layer 11 reported gaps of 1.51, 1.71, and 1.45 in its first three distance bins |
| Required figures | Effect size and significance by distance; true/decoy z-score scatter and LOWESS gap; layer-wise analysis |

The post's original-resolution Experiment 1 figures add the following machine-readable-looking labels, but not the underlying data:

- sequence-distance bins `(10,20]`, `(20,40]`, `(40,60]`, `(60,100]`, `(100,150]`, and `(150,500]`;
- pair score `Max Z-Score (APC-corrected)`;
- a significance panel titled `Mann-Whitney U Test`, plotted as `-log10(p-value)`;
- a continuous-separation curve labeled as a LOWESS true-minus-decoy z-score gap.

The word “Max” establishes that a maximum was taken, but the figure does not say over which axes or at what stage. The plot images contain no alt text, captions, downloadable tables, or provenance metadata for the underlying arrays.

These values are comparison targets, not acceptance thresholds. We must not change methodology merely to move our result toward them.

## 3. Replication status: known versus missing

### 3.1 Explicitly specified by Mandrake

- 150 nonredundant experimentally determined structures;
- X-ray resolution at most 2.0 Å;
- sequence length between 100 and 500 residues;
- 38,286 contact/decoy comparisons;
- one true contact compared with a noncontact at the same sequence distance;
- APC correction;
- AUC, Cohen's d, p-values, sequence-distance analysis, and layer-wise analysis;
- the six plotted sequence-distance bin labels listed in Section 2;
- a plotted score described as the maximum APC-corrected z-score;
- a Mann-Whitney U significance analysis and a LOWESS gap curve;
- ProGen2 has 27 layers according to the layer discussion;
- the ESM-2 comparison has 12 layers according to the comparison discussion.

### 3.2 Recoverable from official public code

Sources:

- [ProGen2 code at pinned commit](https://github.com/salesforce/progen/tree/c27a419c234a0997923761e1fe7daffcebf0eaf5/progen2)
- [ProGen2 paper](https://arxiv.org/html/2206.13517v1)
- [ESM code at pinned archived commit](https://github.com/facebookresearch/esm/tree/2b369911bb5b4b0dda914521b9475cad1656b2ac)
- [ESM contact module](https://github.com/facebookresearch/esm/blob/2b369911bb5b4b0dda914521b9475cad1656b2ac/esm/modules.py)
- [ESM contact tutorial](https://github.com/facebookresearch/esm/blob/2b369911bb5b4b0dda914521b9475cad1656b2ac/examples/contact_prediction.ipynb)

Recoverable details:

- ProGen2-base and ProGen2-medium both have 27 layers, 16 heads, and hidden width 1,536. Base has context length 2,048; medium has context length 1,024. The official paper calls them 764M-parameter models.
- The official ProGen2 tokenizer is character-level. The standard forward input is the literal token `1`, followed by one token per residue, followed by literal token `2`.
- ProGen2 returns post-softmax attention with shape `[batch, head, query, key]` when `output_attentions=True`.
- ProGen2 returns the embedding state plus all transformer-layer states when `output_hidden_states=True`.
- The official ESM contact pipeline removes terminal tokens, symmetrizes every layer/head channel, applies APC per channel, and then applies a sparse logistic regression.
- ESM's released logistic contact head is supervised: its weights were fit using 20 protein structures. It must not be presented as a parameter-free attention score.
- The public ESM tutorial uses virtual-Cβ distance below 8 Å as its contact definition and defaults to minimum sequence separation 6.
- The only official 12-layer ESM-2 checkpoint is `esm2_t12_35M_UR50D`.

### 3.3 Missing artifacts required for exact reproduction

| Missing item | Why it matters | Current status |
|---|---|---|
| The 150 PDB/chain identifiers | Determines every label, sequence, and result | Not public in the post |
| Dataset-construction query and date | PDB contents change; “nonredundant” is undefined | Not public |
| Redundancy threshold and coverage rule | Alters protein/family composition | Not public |
| Exact ProGen2 checkpoint and weight hash | Twenty-seven layers does not distinguish base from medium | Not public for Experiment 1 |
| Exact ESM-2 checkpoint | “12 layers” suggests 35M ESM-2 but does not name it | Not public |
| Sequence orientation and ProGen2 terminal tokens | Causal attention is directional | Not public |
| Chain/model/assembly handling | Changes which residues and contacts exist | Not public |
| Missing residues and alternate-location handling | Can create indexing errors or different labels | Not public |
| Contact atom and distance cutoff | Defines the positive class | Not public |
| Minimum sequence separation and bin assignment code | Controls local-contact prevalence and boundary membership | Plot labels imply `|i-j|>10`, but filtering code is unavailable |
| Decoy matching algorithm | “Same sequence distance” does not specify exact/tolerant matching, replacement, or multiplicity | Not public |
| Random seed | Affects the selected decoys | Not public |
| Attention tensor used | Post-softmax attention versus pre-softmax QK scores | Not public |
| Meaning and stage of `Max` aggregation | Maximum over heads, layers, directions, or another quantity gives different results | Axis label only; axes and order are not public |
| Symmetrization rule for causal attention | ProGen2 supplies only later-to-earlier off-diagonal attention | Not public |
| APC ordering and scope | Before/after aggregation and inclusion of diagonal affect scores | Not public |
| Z-score reference population | Required to reproduce the plotted z-scores | Not public |
| Distance-bin implementation | Labels are public, but the exact filtering/binning code is not | Six labels recovered from the figure |
| AUC aggregation | Pooled pairs versus per-protein averaging differ | Not public |
| Cohen's d definition | Paired and unpaired definitions differ | Not public |
| Mann-Whitney U options and multiple-testing correction | Alternative, unit of analysis, tie handling, and correction affect p-values | Test family is in the figure; remaining settings are not public |
| LOWESS parameters | Required to reproduce the plotted curves | Not public |
| Analysis code and package versions | Required for exact numerical reproduction | Not public |

### 3.4 Contradictions that cannot be silently resolved

1. The post says the ESM-2 comparator has 12 layers and is matched to ProGen2's depth, but later says ProGen2 has 27 layers. Those depths are not matched.
2. Elsewhere the post calls ProGen2-base 754M parameters, while the official ProGen2 paper and release call it 764M.
3. The post's claim that APC leaves “pure structural awareness” is stronger than the method supports. APC removes a row/column background term; it does not separate evolution from structure causally.

## 4. Artifact-recovery gate

Before implementing the full analysis, attempt to obtain from Mandrake or the post author:

1. structure manifest with PDB and chain IDs;
2. Experiment 1 code or notebook;
3. exact ProGen2 and ESM-2 checkpoints;
4. contact and decoy definitions;
5. attention preprocessing, aggregation, APC, and z-scoring order;
6. distance-bin edges;
7. statistical tests, LOWESS settings, and seeds.

A concise request should ask for machine-readable artifacts rather than prose clarification. No external message will be sent without explicit user authorization.

### 4.1 Public artifact search log

Search performed 2026-08-22:

- inspected the complete rendered post HTML and its embedded metadata;
- inspected every outbound article link and every original-resolution image asset;
- searched the public web and GitHub for the exact pair count, reported AUC, plot-axis phrase, post title, author name, and ProGen2/contact combinations;
- checked the only exact-name GitHub user returned by the public user search, including public repositories, gists, and organizations.

Result: no Experiment 1 repository, notebook, structure manifest, result table, supplement, downloadable attachment, or linked data archive was found. The exact-name GitHub profile appears unrelated and is not attributed to the post author. The only additional recoverable artifacts were the static plot images described in Section 2. Therefore, the exact-reproduction gate remains unresolved; this negative search result is not evidence that private or unindexed artifacts do not exist.

Static Experiment 1 assets recovered from the post:

| Asset | Original URL | Bytes | SHA-256 |
|---|---|---:|---|
| Effect-size and Mann-Whitney U panels | [PNG](https://substack-post-media.s3.amazonaws.com/public/images/0aa80dba-4216-421d-85b5-1359e05459e2_4800x1800.png) | 354,546 | `d6289946682b7dd21814b77a5be31f0d899de06c32c0e506be6d457423492ea8` |
| True/decoy scatter and LOWESS gap panels | [PNG](https://substack-post-media.s3.amazonaws.com/public/images/83c31426-8bfc-4ac5-b34e-1ffc22b4a705_6600x1800.png) | 1,679,166 | `80336e560aca48fbdef5e70fd8f6a080a31fdf81ed01d7e96ea6df9b999104ec` |

If the artifacts are obtained, hash them and follow them unchanged for the first run. Any identified defect should be documented and reproduced first; a corrected analysis can be reported separately.

If the artifacts cannot be obtained, proceed with the public-information partial reproduction in Section 5. Every assumption must remain visible in the result report.

## 5. Public-information partial reproduction

This section specifies one minimal, deterministic interpretation of the missing details. These settings are not claimed to be Mandrake's settings.

### 5.1 Models

Primary ProGen2 candidate: official `progen2-base`.

Reason: it has the reported 27 layers, appears elsewhere in the post, and supports every stated protein length without approaching its 2,048-token context limit. This is an inference, not verification. If compute permits, run `progen2-medium` as a checkpoint-sensitivity analysis because it also has 27 layers.

Comparison candidate: official `esm2_t12_35M_UR50D` because it is the public 12-layer ESM-2 checkpoint.

Use the parameter-free raw-attention pipeline defined below for both models. Run ESM-2's released supervised contact head only as a separately labeled diagnostic; it is not the primary comparison.

Record model archive, extracted weight, config, tokenizer, and source-code SHA-256 hashes.

### 5.2 Replacement structure cohort

If Mandrake's manifest remains unavailable:

- query an archived or timestamped RCSB PDB snapshot for protein chains solved by X-ray diffraction at resolution ≤2.0 Å and sequence length 100–500 inclusive;
- save the exact query, raw response, timestamp, and mmCIF hashes;
- use model 1 and one polymer chain as the analysis unit;
- use intrachain contacts only;
- require at least 95% of polymer positions to map to N, Cα, and C coordinates;
- cluster sequences at 30% identity and 80% bidirectional coverage with a pinned MMseqs2 version;
- select one chain per cluster by lowest resolution, then highest coordinate coverage, then lexicographic PDB/chain ID;
- use seed `20260822` to choose 150 representatives, stratified evenly across lengths 100–199, 200–299, 300–399, and 400–500;
- freeze the resulting manifest before model inference.

The clustering and stratification are replacement choices. They are not inferred Mandrake methodology.

Execution status (2026-08-24): the fallback cohort was built from a live RCSB query
recorded at `2026-08-23T17:58:38.046531+00:00`. The run found 96,666 search hits,
retained 95,589 candidates after the declared API filters, produced 12,793 MMseqs2
clusters, and froze 150 unique chains. The manifest SHA-256 is
`10b89e0f3a6dccbacf081ca03a18aebc9001e4dd83f1e4c364e15ecbe3022b09`.
The four length-bin counts are 38, 38, 37, and 37. The five-chain pilot manifest
SHA-256 is `7873e16b49516d4ff760f3c0e91989262630ecb66550e76f1ea7d04272a77996`.
An independent audit replayed the pilot selection, verified every selected mmCIF
hash, and confirmed that every pilot chain produces nonempty exact-distance-matched
contact/decoy pairs. Full details are in `docs/experiment_log.md`.

The implementation uses the live RCSB Search and Data APIs, saves the exact query,
every raw response, and a UTC retrieval timestamp, and therefore implements the
“timestamped query” branch rather than an archived PDB snapshot. It pins MMseqs2
`15.6f452` and invokes `easy-cluster` with `--min-seq-id 0.30 -c 0.80 --cov-mode 0`;
coverage mode 0 requires the declared coverage on both sequences. Restarting in the
same work directory reuses downloaded responses, cluster output, and mmCIF files,
but refuses to proceed if the cohort-defining configuration has changed. This
implementation and the completed fallback cohort are not evidence that Mandrake
used any of these choices or structures.

### 5.3 Structure and contact labels

- Preserve the mmCIF polymer-to-observed-residue mapping; never assume author residue numbering is contiguous.
- For alternate atom locations, choose highest occupancy, then `A`, then lexicographic order.
- Mask pairs involving residues without N, Cα, and C coordinates.
- Reconstruct virtual Cβ coordinates using the exact constants and operation from the pinned ESM tutorial.
- Define a contact as virtual-Cβ distance `<8.0 Å` and a noncontact as distance `≥8.0 Å`.
- Exclude pairs with `|i-j|<=10` for the Mandrake-like primary analysis, interpreting the first plotted interval `(10,20]` literally. This boundary interpretation remains a documented inference until the original binning code is available.

Also report a sensitivity using the ESM default minimum separation 6 and ESM's conventional long-range threshold of at least 24. Do not substitute the sensitivity for the primary result after seeing outcomes.

### 5.4 Distance-matched decoys

For each protein and exact sequence separation `δ`:

1. enumerate all eligible contacts and noncontacts;
2. take the smaller of the two counts;
3. sample that many from each class without replacement;
4. pair independently shuffled contacts and noncontacts.

Derive each protein seed from SHA-256 of `20260822:<pdb>:<chain>`. Save the complete eligible table and selected matched-pair table. No selected contact or decoy may be reused within the primary sample.

This exactly matches sequence separation but may not match Mandrake's unknown sampler. Run 20 predeclared alternative decoy seeds only to quantify sampling sensitivity.

### 5.5 ProGen2 input and tensor extraction

For sequence length `L`, encode the literal input `1` + native N-to-C sequence + `2`. Do not add the tokenizer's separate BOS/EOS tokens. Verify `L+2` tokens and map residue `r` to token `r+1`.

Run frozen evaluation-mode inference with gradients and KV caching disabled. Request return dictionaries, attention tensors, and hidden states. For ProGen2-base verify:

- 27 attention tensors, each `[1,16,L+2,L+2]`;
- 28 hidden-state tensors, each `[1,L+2,1536]`;
- causal future-key entries are zero;
- attention rows sum to approximately one;
- repeated extraction is deterministic within recorded tolerance.

Remove the two terminal-token rows and columns before scoring. Hidden states are extracted only to validate the official API and cache format; they are not used for this replication.

### 5.6 Attention score

For each layer/head channel:

1. use returned post-softmax attention;
2. remove terminal tokens;
3. symmetrize using the ESM operation `X=A+Aᵀ`;
4. apply APC independently to that channel as `X - rowsum(X)×colsum(X)/sum(X)`;
5. z-standardize using all valid off-diagonal pairs with separation greater than 10 in that protein;
6. take the maximum standardized value over all layer/head channels to obtain the primary score;
7. take the maximum over heads within each layer for layer-wise results.

For causal ProGen2 and `i<j`, symmetrization makes the pair score derive from the available later-to-earlier edge `A[j,i]`. It does not create evidence from an unavailable reverse attention direction.

The maximum is the smallest aggregation choice consistent with the published `Max Z-Score` axis label. Its axes and ordering remain an inference, not recovered methodology. Report all layers. Do not select a layer after observing labels as a separate global primary result.

Apply the same steps to ESM-2 raw attention. The released ESM contact head is reported separately as “supervised ESM contact-head diagnostic.”

### 5.7 Reconstructed distance bins

Use the labels printed in the published figure, interpreted literally for integer sequence separation:

- `(10,20]` → 11–20;
- `(20,40]` → 21–40;
- `(40,60]` → 41–60;
- `(60,100]` → 61–100;
- `(100,150]` → 101–150;
- `(150,500]` → 151–500.

The labels are published, while their literal right-closed implementation is inferred from notation because the binning code is missing. Also report continuous separation and ESM categories so conclusions are not dependent on this interpretation.

### 5.8 Metrics

Attempt to match the blog outputs while making statistical units explicit:

- pooled ROC-AUC over selected contact and decoy scores;
- matched-pair concordance, counting ties as 0.5;
- conventional pooled Cohen's d, named `d_pooled`;
- paired standardized mean difference, named `d_z`;
- per-protein AUC distribution;
- every metric by distance bin and layer;
- sample counts for every metric.

For a direct blog-figure reconstruction, also compute two-sided Mann-Whitney U tests on the selected true and decoy score arrays and label them as pair-level, pseudo-replicated statistics. Do not treat those p-values as confirmatory evidence because pairs within a protein are dependent and the decoys are matched.

For valid uncertainty, use 10,000 protein-cluster bootstrap replicates with seed `20260822` for 95% confidence intervals. Test matched differences with protein-level sign flips rather than treating residue pairs as independent. Apply Holm correction across layer/bin tests. These robust inference choices are additions because Mandrake's test settings and multiple-testing handling are unspecified.

## 6. Minimal pilot before the 150-protein run

The pilot validates the reproduction pipeline; it is not a scientific result.

After freezing the replacement manifest, select five chains at the minimum, 25th-percentile, median, 75th-percentile, and maximum lengths, breaking ties lexicographically.

The pilot must demonstrate:

- exact sequence/token/residue alignment;
- correct missing-residue masks and virtual-Cβ coordinates;
- all contacts are `<8 Å` and all decoys are `≥8 Å`;
- every matched pair has identical sequence separation;
- no primary pair is reused;
- correct ProGen2 attention shapes and causal mask;
- APC agreement with the pinned ESM functions on a fixed synthetic tensor within `1e-6` FP32 tolerance;
- deterministic scores and metrics from two identical runs;
- successful raw-attention scoring for ProGen2-base and 12-layer ESM-2;
- recorded peak memory and wall time at all five lengths.

There is no pilot AUC threshold. A favorable five-protein AUC is neither required nor sufficient.

## 7. Full-run decision rules

Proceed from five proteins to 150 only when all technical pilot checks pass.

After the full run:

1. compare cohort composition and pair counts with Mandrake's 38,286 comparisons;
2. compare global AUC and effect sizes without treating exact agreement as required;
3. compare the direction and shape of separation-dependent and layer-dependent trends;
4. investigate discrepancies in this order: dataset identity, checkpoint, residue alignment, contact geometry, decoy sampling, attention aggregation, APC/z-scoring, then statistical definitions;
5. never tune an undocumented choice solely because it moves AUC toward 0.527;
6. report every deviation and whether it could plausibly explain the discrepancy.

Possible result labels:

- **Exact reproduction:** Mandrake artifacts were obtained and followed.
- **Close reproduction:** artifacts were mostly obtained, with named compatibility substitutions.
- **Public-information partial reproduction:** replacement choices in Section 5 were required.
- **Failed technical replication:** the pipeline could not satisfy its validation checks.
- **Scientific disagreement:** the validated pipeline and sufficiently matched methods produce a materially different result.

Do not use “scientific disagreement” when the manifest, checkpoint, or score definition is still materially unmatched.

## 8. Leakage and interpretation controls

The attention-only baseline fits no parameters, so train/test leakage is not its main risk. Its primary risks are pretraining familiarity, hidden dataset redundancy, and statistical pseudo-replication.

- Report within-cohort sequence clusters and nearest recoverable pretraining matches.
- Do not claim the test proteins were unseen during ProGen2 or ESM pretraining unless this is verified against the correct training snapshots.
- Bootstrap and test at protein level.
- Keep all residue pairs from a protein together in any later learned analysis.
- Treat ESM's released contact head as supervised and potentially overlapping with this cohort.
- Report native-orientation ProGen2 as primary; reverse-orientation averaging, if run, is a sensitivity analysis.
- Interpret a positive result only as contact discrimination by the declared attention score.
- Interpret a negative result only as weak signal under that score and cohort.
- Do not claim either result demonstrates or disproves mechanistic understanding of protein physics.

## 9. Resource estimate for the likely checkpoint

For ProGen2-base in FP16, batch size 1, length 500 plus two terminal tokens:

- weights: approximately 1.53 GB (1.42 GiB);
- all 27×16 returned attention maps: approximately 208 MiB;
- all 28 hidden-state tensors: approximately 41 MiB;
- theoretical forward compute: approximately 0.81 TFLOP;
- practical GPU target: at least 8 GiB after accounting for masks, temporary tensors, framework overhead, and allocator workspace.

The initial remote Pod is an RTX 4090, whose working base runtime is Python 3.11,
PyTorch 2.4.1, and CUDA 12.4. Because the pinned ProGen2-era tokenizer stack requires
an older Python, remote execution uses an isolated Python 3.8.20 environment with
PyTorch 2.0.1 and its CUDA 11.8 wheel. This compatibility substitution is recorded
separately from model and analysis methodology and must pass the deterministic model
smoke test before any protein is analyzed.

At average protein length 300, storing every FP16 attention and hidden-state tensor for 150 proteins would require about 15 GiB. For this replication, aggregate attention per protein and retain score tables; raw tensor caching is optional. Measure actual peak memory and seconds per protein in the pilot before extrapolating full-run time.

## 10. Reproducibility artifacts

Every pilot and full run must save:

- frozen structure manifest and mmCIF hashes;
- model, tokenizer, config, code, and environment hashes;
- resolved configuration and seeds;
- residue/token mappings and exclusions;
- eligible and selected contact/decoy tables;
- attention scores before and after aggregation;
- per-protein, pooled, layer-wise, and distance-wise results;
- bootstrap/randomization outputs;
- validation, runtime, and peak-memory logs;
- a deviation report tied to the missing-item table in Section 3.3.

Only after this replication is complete should a separate protocol introduce frozen hidden-state representations and a deliberately weak probe.
