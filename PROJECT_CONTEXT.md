# PROJECT_CONTEXT.md

## Project title

**Probing Structural Information in ProGen2 Beyond Attention**

## Why this project exists

This project is inspired by Mandrake Bio's April 18, 2026 post:

> **Protein Language Models: Fluent, but clueless**  
> https://research.mandrake.bio/p/protein-language-models-fluent-but

The post argues that autoregressive protein language models (PLMs), especially ProGen2, are very good at learning **1D protein-sequence statistics ("grammar")** but are brittle when tasks require **3D structural reasoning, robust generation, or mutation-effect prediction**.

The immediate goal here is **not** to broadly "fix protein language models."  
The goal is to take one specific claim from the post, formulate a sharp scientific question, and run a small but rigorous experiment that could plausibly produce an interesting result either way.

The most promising question is:

> **Does ProGen2 genuinely fail to encode 3D structural information, or is that information simply not visible in raw attention weights?**

Mandrake's Experiment 1 primarily probes attention. We want to probe the model's **hidden-state representations** as well.

---

# 1. Minimal protein-language-model background

A protein is a sequence of amino acids:

```text
M K L A V G ...
```

Each letter represents one amino-acid residue. A protein may contain hundreds of residues.

The sequence is one-dimensional, but the protein folds into a three-dimensional structure. Two residues that are far apart in sequence can end up physically adjacent after folding.

Example:

```text
Sequence:

residue 30 ------------------------------ residue 280
              250 positions apart

3D fold:

residue 30  ●──●  residue 280
             contact
```

Those long-range contacts matter for folding, stability, binding, catalysis, and function.

---

# 2. The two model families relevant here

## ProGen2

ProGen2 is an **autoregressive / causal** protein language model.

It predicts one amino acid at a time:

```text
Given: M
predict: K

Given: M K
predict: L

Given: M K L
predict: A
```

Formally:

$$
P(x_1,\ldots,x_L)=\prod_t P(x_t \mid x_{<t})
$$

At position $t$, the model can only attend to positions before $t$.

So residue 280 can attend to residue 30, but residue 30 cannot attend to residue 280.

Official implementation:
https://github.com/salesforce/progen

The implementation can return both:
- attention tensors
- hidden states from every transformer layer

using `output_attentions=True` and `output_hidden_states=True`.

## ESM-2

ESM-2 is a **masked/bidirectional** protein language model.

Rather than always predicting left-to-right, it is trained to reconstruct masked amino acids using context from both sides.

Official repository:
https://github.com/facebookresearch/esm

ESM-2 is useful here primarily as a positive/comparison model because its representations and attention have previously been shown to contain significant structural information.

---

# 3. Mandrake Experiment 1: residue-contact discrimination

## Question

Mandrake asks:

> When ProGen2 processes a protein, do its attention patterns reflect which residue pairs physically contact each other in 3D?

## Dataset

According to the blog:

- 150 non-redundant experimentally determined protein structures
- X-ray resolution <= 2.0 Å
- sequence lengths 100–500 amino acids
- 38,286 true-contact / decoy comparisons

## True contacts vs decoys

For a real contacting pair:

$$
(i,j)
$$

they compare against a **non-contact pair with similar sequence separation**.

This is critical.

Without sequence-distance matching, a model could appear good merely because transformers often attend more strongly to nearby sequence positions.

Example:

```text
True contact:
30 ------------------------- 280
     sequence distance 250

Decoy:
40 ------------------------- 290
     sequence distance 250
```

The two pairs are equally far apart in the 1D sequence, but only the first pair touches in 3D.

The test becomes:

> Does the real 3D-contact pair receive a larger model-derived score than the matched non-contact?

---

# 4. APC: Average Product Correction

Mandrake applies APC before testing attention/contact correspondence.

For a pairwise score matrix $A$, a simplified form is:

$$
A^{APC}_{ij}
=
A_{ij}
-
\frac{\bar A_i \bar A_j}{\bar A}
$$

Intuition:

Some positions may have unusually high scores with many positions simply because of broad statistical/evolutionary effects.

APC subtracts an estimate of that background tendency so that unusually **pair-specific** scores stand out.

Important caveat:

Mandrake describes this as removing "evolutionary background noise" and leaving "pure structural awareness."

That is too strong.

APC does **not** mathematically decompose:

$$
\text{attention} =
\text{evolutionary signal} +
\text{structural signal}
$$

and perfectly subtract the first term.

Evolutionary covariation and structural contact are themselves strongly related.

A safer interpretation is:

> APC reduces broad pairwise/statistical biases before asking whether true structural contacts still score unusually highly.

---

# 5. Experiment 1 metrics

## AUC

AUC is the **Area Under the ROC Curve**.

For this experiment, the most intuitive interpretation is:

> The probability that a randomly selected true contact receives a higher score than a randomly selected non-contact.

Examples:

- AUC = 0.50 → random ranking
- AUC = 0.70 → true contact wins roughly 70% of random contact/non-contact comparisons
- AUC = 1.00 → perfect ranking

Important:

AUC 0.61 does **not** mean "61% of protein structure is understood" or "61% accuracy."

It measures ranking/discrimination.

## Cohen's d

Cohen's $d$ measures separation between the score distributions of true contacts and decoys:

$$
d =
\frac{
\mu_{\text{contact}}-\mu_{\text{decoy}}
}{
s_{\text{pooled}}
}
$$

Very roughly:

- $d=0$: no separation
- $d\approx0.2$: small
- $d\approx0.5$: moderate
- $d\approx0.8$: large

---

# 6. Mandrake Experiment 1 results

Reported in the blog:

### ProGen2

- AUC: **0.527**
- maximum Cohen's $d$: **0.184**
- at sequence separations beyond roughly 100 residues, the final distance bins were not statistically significant
- contact discrimination worsened with increasing sequence separation

### ESM-2

- AUC: **0.611**
- Cohen's $d$ peaks around **0.52**
- all distance bins statistically significant, including 150–500-residue separation
- structural signal appears particularly strongly in later layers

Their broad interpretation:

> ProGen2 learns strong 1D sequence grammar but very weak long-range 3D structural relationships.

---

# 7. The key methodological concern that motivates this project

Mandrake's direct evidence establishes:

> **Raw ProGen2 attention weights contain little contact-discrimination signal.**

That is not identical to:

> **ProGen2's internal representations contain little structural information.**

Attention weights are only one intermediate component of a transformer.

A transformer layer roughly performs:

```text
hidden state
    |
    +--> attention --> residual update
    |
    +--> MLP -------> residual update
    |
    v
new hidden state
```

The hidden state at residue $i$,

$$
h_i^{(\ell)}
$$

is the model's accumulated representation of that residue after layer $\ell$.

It incorporates:
- residue identity
- information received through many attention heads
- information accumulated over previous layers
- nonlinear MLP transformations
- contextual information from many other residues

Therefore:

$$
A_{ij}\text{ is weak}
$$

does **not** imply:

$$
(h_i,h_j)\text{ contain no structural signal}.
$$

Structural information may be **distributed across representations** rather than appearing as a single large attention edge.

This is the central motivation for probing hidden states.

---

# 8. Primary research question

> **Can 3D residue contacts be decoded from frozen ProGen2 hidden states even when they cannot be cleanly decoded from raw attention weights?**

This distinguishes two hypotheses.

## H1: representation exists, attention is a poor readout

Expected result:

```text
ProGen2 attention AUC        ~ 0.53
ProGen2 hidden-state probe   substantially > 0.53
```

Interpretation:

> ProGen2 contains recoverable structural information, but raw attention maps are not where that information is most directly expressed.

This would qualify Mandrake's broader interpretation.

## H2: structural information is genuinely weak

Expected result:

```text
ProGen2 attention AUC        ~ 0.53
ProGen2 hidden-state probe   ~ 0.53–0.55
```

Interpretation:

> Even the internal hidden representations contain little linearly/simple-decodable long-range contact information.

This would strengthen Mandrake's conclusion.

Both outcomes are scientifically useful.

---

# 9. Why hidden-state probing is meaningful

Suppose residues 30 and 280 contact each other.

Raw attention might not satisfy:

$$
A_{280,30} \gg A_{280,j}.
$$

But the representations might still encode information such as:

```text
h_30:
"buried residue, hydrophobic-core environment, beta-sheet context, ..."

h_280:
"compatible buried residue, same structural environment, ..."
```

A downstream function of $h_{30}$ and $h_{280}$ may therefore distinguish the pair from a non-contacting pair.

This is similar to asking:

> What information is *represented* by the network, rather than what a single attention head visibly attends to?

---

# 10. Critical caveat: the probe itself must not learn protein folding

If we train a huge neural network on top of hidden states and it predicts contacts well, that does not necessarily show that ProGen2 already encoded the information.

The downstream network may simply learn the contact problem itself.

Therefore the primary probe should be deliberately weak.

Recommended order:

1. **logistic regression / linear classifier**
2. optionally a tiny MLP as a secondary experiment
3. avoid large nonlinear contact-prediction networks initially

The key claim should be phrased as:

> "How easily can structural information be decoded from the representation?"

not:

> "The model understands protein physics."

---

# 11. Proposed main experiment

For every residue pair $(i,j)$, extract layer-$\ell$ representations:

$$
h_i^{(\ell)},\quad h_j^{(\ell)}.
$$

Construct a simple pair representation.

Possible starting point:

$$
z_{ij}^{(\ell)}
=
[
h_i^{(\ell)};
h_j^{(\ell)};
|h_i^{(\ell)}-h_j^{(\ell)}|;
h_i^{(\ell)}\odot h_j^{(\ell)}
]
$$

where:
- `;` = concatenation
- $|h_i-h_j|$ = element-wise absolute difference
- $h_i\odot h_j$ = element-wise product

Then train:

$$
P(\text{contact}\mid z_{ij})
=
\sigma(w^\top z_{ij}+b)
$$

using logistic regression.

Alternative simpler/symmetric pair features should also be tested.

Because contacts are symmetric, a representation that does not depend strongly on pair ordering may be preferable.

Candidate symmetric representation:

$$
z_{ij}
=
[
|h_i-h_j|;
h_i\odot h_j
]
$$

This should probably be the first probe.

---

# 12. Experimental comparisons

At minimum compare:

| Method | Model | Learned downstream parameters? |
|---|---|---|
| Raw/APC attention | ProGen2 | No |
| Hidden-state linear probe | ProGen2 | Yes, tiny |
| Raw/APC attention | ESM-2 | No or matched contact head |
| Hidden-state linear probe | ESM-2 | Yes, tiny |

The most important comparison is:

$$
\text{ProGen2 attention}
\quad\text{vs}\quad
\text{ProGen2 hidden-state probe}
$$

ESM-2 is useful as a sanity check / positive control.

---

# 13. Layer-wise probing

Probe each transformer layer independently:

$$
h^{(0)},h^{(1)},\ldots,h^{(L)}.
$$

Plot:

```text
layer number  --> contact AUC
```

Questions:

1. Does structural information appear at all?
2. If yes, when does it emerge?
3. Does it strengthen in later layers?
4. Does it peak and then disappear?
5. Does ProGen2 differ qualitatively from ESM-2?

A particularly interesting outcome would be:

```text
ProGen2 attention: weak everywhere

ProGen2 hidden states:
early layers     ~0.52
middle layers    ~0.60
late layers      ~0.65
```

That would show that the network accumulates structural information even though raw attention does not expose it cleanly.

---

# 14. Long-range contacts are the important test

Do not only report a single global AUC.

Measure performance as a function of sequence separation:

$$
|i-j|.
$$

Example bins could follow Mandrake's setup once the exact bin definitions are recovered.

At minimum distinguish:

- short-range
- medium-range
- long-range

The key question is whether hidden states rescue the failure Mandrake observes at **long sequence separations**.

Example desired plot:

```text
Contact AUC
|
|          hidden probe
|          --------
|        /
|  attention
|  ----\________
|
+---------------------- sequence distance
```

---

# 15. Train/test splitting is extremely important

Never randomly split residue pairs from the same proteins across train and test.

Bad:

```text
Protein A pair 1 -> train
Protein A pair 2 -> test
Protein A pair 3 -> train
```

That creates leakage because the probe can learn protein-specific characteristics.

Instead split **by protein**:

```text
train proteins
validation proteins
test proteins
```

Every pair from a test protein must remain entirely unseen during probe training.

Ideally use a structure/family-aware split that also limits homology between train and test proteins.

---

# 16. Decoy construction

The negative examples must control for trivial sequence-distance effects.

For true contact:

$$
(i,j)
$$

with:

$$
d=|i-j|,
$$

choose a non-contact pair:

$$
(k,l)
$$

with approximately the same:

$$
|k-l|\approx d.
$$

The exact matching rule should reproduce Mandrake's method if possible.

Before implementation, determine:

- exact contact geometry
- exact distance cutoff
- whether contact uses C-alpha, C-beta, minimum heavy-atom distance, etc.
- exact sequence-distance matching tolerance
- number of decoys per contact
- whether contacts close along sequence are excluded
- exact distance bins

Do not silently invent these choices while claiming to reproduce Mandrake.

If their precise methodology cannot be recovered, document our chosen definition clearly and call the experiment a **replication-inspired benchmark**, not an exact reproduction.

---

# 17. Contact definition

The blog does not appear to specify the exact geometric contact criterion.

This must be resolved before claiming direct replication.

Common choices in the literature include:
- C-beta distance threshold (C-alpha for glycine)
- minimum heavy-atom distance
- threshold around 8 Å for certain contact benchmarks

Do not assume one without checking the relevant code/paper or contacting the authors.

---

# 18. Important control baselines

A hidden-state probe can exploit nonstructural shortcuts, so include baselines.

## Baseline A: amino-acid identity only

Represent pair using only the two residue identities.

Question:

> How much contact prediction is possible from residue types alone?

## Baseline B: position / sequence distance only

Input:

$$
|i-j|
$$

or normalized positions.

A distance-matched dataset should make this weak, but verify it.

## Baseline C: raw embedding layer

Probe $h^{(0)}$, before contextual transformer processing.

If the contextual layers do no better than the embedding layer, the supposed structural information may just reflect amino-acid identity.

## Baseline D: shuffled hidden states

Shuffle representations across positions/proteins while keeping labels.

Expected AUC:

$$
\approx0.5.
$$

## Baseline E: ESM-2

The positive control should show stronger structural signal under a reasonable probing setup.

---

# 19. Evaluation metrics

Primary:

- ROC-AUC
- preferably PR-AUC as well if class imbalance is introduced
- Cohen's $d$ for direct comparability with Mandrake
- bootstrap confidence intervals by **protein**, not by individual residue pair

Also report:

- AUC by sequence-distance bin
- AUC by layer
- per-protein AUC distributions

Avoid treating tens of thousands of residue pairs as fully independent observations when they originate from only ~150 proteins.

---

# 20. A potential stronger extension: contact head

Only after the probing experiment is understood:

Freeze ProGen2 and train a tiny contact head:

```text
protein sequence
      |
      v
frozen ProGen2
      |
      v
hidden states
      |
      v
small pairwise contact head
      |
      v
contact probability matrix
```

Question:

> Can a very small structure-aware head recover meaningful long-range contacts without retraining the language model?

This is a possible lightweight "fix."

But the scientific probe should come first.

---

# 21. Experiment 3 from the blog: DMS variant scoring

DMS = Deep Mutational Scanning.

Experimentally:

1. start with a real protein
2. mutate residues systematically
3. measure the effect of each mutation on function/fitness

This gives laboratory ground truth for mutation effects.

Mandrake compares:

- ProGen2-base, 754M parameters
- ESM-C, 600M parameters
- 7 DMS assays

For ProGen2, a mutation is scored using a log-likelihood difference:

$$
\Delta LL
=
LL(\text{mutant})-LL(\text{wildtype})
$$

Because ProGen2 is causal, a mutation at position 42 can affect:
- position 42
- positions 43 onward

but cannot affect likelihood terms at positions 1–41.

Mandrake hypothesizes that this directionality creates a blind spot.

## Metric: Spearman correlation

Spearman correlation asks whether the model ranks mutations in roughly the same order as the experimentally measured fitness effects.

- $\rho=1$: perfect rank agreement
- $\rho=0$: no rank relationship
- $\rho=-1$: perfectly reversed ranking

Reported result:

- ESM-C beats ProGen2 on all 7 assays
- mean Spearman advantage: **+0.136**
- range: **+0.028 (HIS7)** to **+0.268 (GAL4)**
- ProGen2 on GFP: **-0.003**, essentially no correlation

Important caveat:

This comparison does **not** prove causality/directionality is the sole explanation.

The models differ in:
- training objectives
- training data
- architecture
- representations

---

# 22. DMS benchmark caveat

DMS is experimental ground truth, but model performance can still be hard to interpret.

If a model saw many homologs of a test protein family during pretraining, it may have learned evolutionary regularities such as:

```text
position 42:
A common
V sometimes observed
W almost never observed
```

Then it can predict mutation tolerance well without necessarily reasoning from biophysical mechanism.

Therefore:

$$
\text{DMS performance}
$$

can reflect a mixture of:

$$
\text{training-family familiarity}
+
\text{evolutionary statistics}
+
\text{transferable biological information}.
$$

So "good DMS performance" should not automatically be interpreted as "deep mechanistic understanding."

---

# 23. Broader eval problem raised by Mandrake

Mandrake argues common PLM benchmarks emphasize:

1. perplexity
2. DMS mutation scoring

but can miss failure modes such as:

- long-range structural-contact blindness
- inability to combine multiple functional constraints
- catastrophic behavior outside familiar sequence distributions

Their proposed broader philosophy is:

> Evaluate PLMs on the failure modes that matter for actual protein engineering, not only on sequence likelihood.

---

# 24. Perplexity vs function caveat

The post also argues that optimizing perplexity too aggressively may favor highly consensus-like sequences.

Their biological intuition:

Real functional proteins are not necessarily maximally stable or statistically typical.

Functions such as catalysis may require:
- flexible loops
- multiple conformational states
- locally suboptimal packing
- active-site dynamics
- energetic frustration

Therefore:

$$
\text{low perplexity}
\not\Rightarrow
\text{high function}.
$$

This should be treated as a plausible design concern, not a universal law that low-perplexity proteins are always inert.

---

# 25. Why AlphaFold does not make this project irrelevant

AlphaFold is directly designed for structure prediction.

If the task is simply:

> "Given this protein sequence, predict its 3D structure"

then AlphaFold/related folding models are the appropriate tools.

The research question here is different:

> **What structural information emerges inside a generic sequence language model that was not explicitly trained as a structure predictor?**

That matters because PLMs are used for:
- embeddings
- protein generation
- mutation scoring
- design
- downstream learned predictors

The project is therefore about the **representational content and limitations of ProGen2**, not about replacing AlphaFold.

---

# 26. What NOT to claim

Even if the hidden-state probe works extremely well, avoid:

> "ProGen2 understands protein physics."

Instead say:

> "3D contact information is linearly/simple-decodable from ProGen2 representations."

Likewise, if it fails:

Do not say:

> "ProGen2 contains zero structural information."

Say:

> "Under this probing setup, little recoverable long-range contact information was found."

Probe results depend on:
- target definition
- pair representation
- layer
- model checkpoint
- data split
- probe capacity

---

# 27. First implementation milestone

Do **not** begin by building the full hidden-state experiment.

First reproduce a minimal attention-contact baseline.

Target:

```text
input protein structure
        |
        +--> amino-acid sequence
        |
        +--> true contact labels
        |
        v
     ProGen2
        |
        v
attention tensors
        |
       APC
        |
        v
contact vs matched-decoy AUC
```

Only after this works should hidden-state probing be added.

Reason:

If our baseline disagrees drastically with Mandrake, we need to understand why before interpreting the probe.

---

# 28. Suggested implementation phases

## Phase 0 — resolve methodology

Before writing substantial code, determine:

- exact ProGen2 checkpoint used
- exact tokenizer details
- exact contact definition
- exact contact-vs-decoy construction
- exact sequence-distance bins
- exact attention aggregation across heads/layers
- where APC is applied
- exact ESM-2 checkpoint
- whether Mandrake code/data are publicly available

Write findings to:

```text
docs/methodology.md
```

Mark each item as:
- verified
- inferred
- chosen by us because unspecified

## Phase 1 — data pipeline

Implement:

```text
PDB/mmCIF
   |
   v
sequence + residue coordinates
   |
   v
contact labels
   |
   v
matched contact/decoy pairs
```

Tests should verify:
- sequence/residue alignment
- missing residues
- chain handling
- glycine/contact atom handling if relevant
- sequence separation
- no accidental overlap/leakage

## Phase 2 — ProGen2 extraction

Implement frozen inference returning:

```python
{
    "hidden_states": ...,
    "attentions": ...
}
```

Cache extracted representations to disk so probe experiments do not repeatedly rerun the PLM.

## Phase 3 — reproduce attention baseline

Implement:
- attention aggregation
- APC
- contact/decoy scores
- ROC-AUC
- Cohen's d
- distance-bin plots

Goal:

Get qualitatively close to Mandrake:
- weak ProGen2 signal
- decreasing long-range performance

Do not force exact numerical agreement if methodology remains underspecified.

## Phase 4 — hidden-state probes

For every layer:
- build symmetric pair features
- fit logistic regression on training proteins
- tune regularization on validation proteins
- evaluate on held-out proteins

Store:

```text
layer
AUC
Cohen_d
AUC_by_distance_bin
per_protein_AUC
```

## Phase 5 — ESM-2 control

Run the same pipeline with ESM-2.

Main purpose:
- verify that the benchmark can recover known structural signal
- contextualize ProGen2 results

## Phase 6 — robustness

Try:
- pair-feature variants
- different contact thresholds
- stricter long-range-only subsets
- family/homology-aware splits if feasible
- bootstrap confidence intervals by protein

---

# 29. Suggested repository structure

```text
progen2-structure-probe/
├── PROJECT_CONTEXT.md
├── AGENTS.md
├── CLAUDE.md
├── README.md
├── pyproject.toml
├── configs/
│   ├── progen2.yaml
│   ├── esm2.yaml
│   └── contacts.yaml
├── docs/
│   ├── methodology.md
│   ├── literature_notes.md
│   └── experiment_log.md
├── scripts/
│   ├── prepare_structures.py
│   ├── extract_progen2.py
│   ├── extract_esm2.py
│   ├── run_attention_baseline.py
│   └── run_probe.py
├── src/
│   └── structure_probe/
│       ├── data.py
│       ├── contacts.py
│       ├── models.py
│       ├── attention.py
│       ├── features.py
│       ├── probes.py
│       ├── metrics.py
│       └── plotting.py
├── tests/
└── results/
```

Do not over-engineer this at the start.

---

# 30. Reproducibility requirements

Every result should be reproducible from config + random seed.

Record:

- model checkpoint
- model revision
- dataset/version
- protein IDs
- chain IDs
- contact definition
- split
- probe hyperparameters
- random seed
- package versions

Prefer cached immutable intermediate datasets.

---

# 31. First plots worth producing

## Plot 1 — reproduce Mandrake

```text
x: sequence-distance bin
y: contact discrimination
series:
- ProGen2 attention
- ESM-2 attention
```

## Plot 2 — central result

```text
x: transformer layer
y: test ROC-AUC
series:
- ProGen2 hidden-state probe
- ESM-2 hidden-state probe
```

## Plot 3 — central long-range result

```text
x: sequence-distance bin
y: ROC-AUC
series:
- ProGen2 attention
- ProGen2 hidden-state probe
```

This is probably the single most important figure.

## Plot 4 — per-protein distribution

Show whether improvement is broad or driven by a few proteins.

---

# 32. What would make the result genuinely interesting?

## Strong positive result

Attention:

$$
AUC\approx0.53
$$

Hidden-state linear probe:

$$
AUC\gg0.53
$$

especially for long-range contacts.

Interesting conclusion:

> The structural information is present but not reflected directly in raw attention scores.

## Strong negative result

Attention:

$$
AUC\approx0.53
$$

Hidden-state probes across all layers:

$$
AUC\approx0.53
$$

while ESM-2 probes work well.

Interesting conclusion:

> ProGen2 appears to contain genuinely weaker easily decodable long-range contact representations, rather than attention simply being a poor readout.

## Less interesting result

A large nonlinear MLP gets good AUC while linear probes fail.

This is ambiguous because the probe may be learning the task.

---

# 33. Possible follow-up project from Experiment 3

A second, independent project idea:

> **Does ProGen2's causal directionality actually explain its DMS deficit?**

A mutation near the N-terminus has many downstream likelihood terms that can change.

A mutation near the C-terminus has very few.

Therefore Mandrake's directionality hypothesis predicts a positional effect.

Test:

```text
mutation position (N -> C)
vs
DMS prediction quality
```

Then compare:
- forward ProGen2 likelihood
- reverse-direction likelihood if available
- bidirectional/ensemble score
- ESM-C

This is a clean mechanistic follow-up but secondary to the structural-probing project.

---

# 34. Recommended first prompt to Claude Code / Codex

Use this before asking it to implement anything:

```text
Read PROJECT_CONTEXT.md completely before doing anything else.

Do not write implementation code yet.

Your first task is to turn the proposed Experiment 1 follow-up into a concrete,
reproducible experimental protocol.

Specifically:

1. Inspect the Mandrake blog's Experiment 1 and identify every methodological
   detail we need in order to reproduce the attention-contact baseline.
2. Inspect the official ProGen2 implementation and determine exactly how to
   extract per-layer hidden states and attention tensors.
3. Inspect established ESM contact-prediction methodology for sensible contact
   definitions, APC usage, attention aggregation, and structural data splits.
4. Separate every methodological choice into:
   - explicitly specified by Mandrake
   - recoverable from cited/public code
   - unspecified and therefore requiring our own documented choice
5. Propose the smallest experiment that can validate the pipeline before
   scaling to all 150 proteins.
6. Identify leakage risks and controls.
7. Estimate memory/compute requirements for the likely ProGen2 checkpoint.
8. Write the resulting protocol to docs/methodology.md.

Do not silently invent missing experimental details.
Do not train a powerful probe.
Do not claim that decodable contact information implies mechanistic
understanding of protein physics.
```

---

# 35. Suggested CLAUDE.md

```markdown
@PROJECT_CONTEXT.md

Work as a careful ML research engineer.

Before changing experimental methodology:
- explain the scientific reason
- identify possible confounds
- distinguish replication from a new methodological choice

Prioritize:
1. scientific validity
2. reproducibility
3. minimal experiments
4. implementation simplicity

Do not optimize for writing lots of code.
```

---

# 36. Suggested AGENTS.md for Codex

```markdown
# Research instructions

Read PROJECT_CONTEXT.md before starting any task.

This is an ML research project, not a product engineering project.

Preserve these constraints:
- ProGen2 remains frozen for the core probing experiment.
- Split data by protein, never randomly by residue pair.
- Use weak probes first.
- Always include trivial baselines.
- Evaluate long-range contacts separately.
- Clearly document methodological choices that differ from Mandrake.
- Do not overclaim what probing establishes.

Prefer a small verified pipeline over a large speculative implementation.
```

---

# 37. Core project summary

The entire project can be reduced to:

```text
Mandrake:
ProGen2 attention -> contact?
AUC ≈ 0.527
          |
          v
"weak structural awareness"

Our question:
Was attention simply the wrong place to look?
          |
          v
ProGen2 hidden states
          |
          v
tiny linear probe
          |
          v
3D contact discrimination
```

The key scientific distinction is:

$$
\boxed{\text{weak structural signal in attention}}
$$

versus:

$$
\boxed{\text{weak structural representation in the model overall}}
$$

Mandrake convincingly provides evidence for the first.

This project tests whether the stronger interpretation survives a simple hidden-state probe.

---

# 38. Primary source links

Mandrake Bio blog:
https://research.mandrake.bio/p/protein-language-models-fluent-but

Official ProGen repository:
https://github.com/salesforce/progen

Official ESM repository:
https://github.com/facebookresearch/esm

ESM contact-prediction background:
https://doi.org/10.1101/2020.12.15.422761

ESM-2 / ESMFold:
https://www.science.org/doi/10.1126/science.ade2574

Mandrake-cited DMS caveat:
"Protein Language Model Fitness Is a Matter of Preference"
https://www.biorxiv.org/content/10.1101/2024.10.03.616542v1

---

# 39. Current status

The Experiment 1 attention-contact pipeline has been completed on the frozen
150-chain public-information fallback cohort. It produced pooled ROC-AUC values of
0.5374 for ProGen2-base and 0.6147 for ESM-2 35M, reproducing Mandrake's main
qualitative finding despite the unpublished original cohort and methodological gaps.

The next step is the planned minimal hidden-state probe. Freeze the language models,
split by protein before training, use a deliberately small regularized pairwise head,
and report held-out and long-range performance without claiming that decodability
implies mechanistic understanding.
