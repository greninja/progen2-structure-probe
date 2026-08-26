# Experiment 1 follow-up: minimal hidden-state probe

Status: protocol frozen before implementation or hidden-state result inspection.  
Protocol version: 0.1 (2026-08-26)

## Question and claim boundary

The primary question is:

> Do contextual ProGen2-base hidden states contain contact-predictive information
> that is not available in the model's input embedding and is not visible in the
> raw-attention Experiment 1 baseline?

This experiment measures held-out decodability. It cannot establish that ProGen2
simulates folding, uses the decoded feature during generation, or learned contact
geometry rather than sequence regularities shaped by evolution.

The representation at residue `j` is recorded after token `j` has been consumed. For
a pair `i < j`, the probe therefore tests whether the relationship is decodable after
both residues are present, not whether ProGen2 anticipated the contact before
generating residue `j`.

## Frozen data and labels

- Reuse the frozen 150-chain Experiment 1 replacement cohort without exclusions based
  on attention or probe results.
- Reconstruct virtual-Cβ contacts at `<8 Å` and require `|i-j|>10`.
- Recreate the same one-to-one, exact-sequence-separation contact/decoy matching with
  seed `20260822`.
- Treat a whole protein, which is already unique at the cohort's 30%-identity/80%-
  coverage clustering rule, as the split unit.
- Stratify the deterministic 90/30/30 train/validation/test split by the four original
  protein-length bins. Freeze and hash the resulting identifiers before fitting.

The frozen split-record content SHA-256 is
`1c4a1b025cf34c37f95eff729039814a378e0dff4a5ffe225416682d9afe2647`.

No residue pair from a validation or test protein may enter feature scaling, model
fitting, regularization selection, or layer selection.

## Frozen representation and probe

Extract all 28 ProGen2-base stages: stage 0 is the input embedding and stages 1–27
are contextual transformer outputs. Terminal tokens are removed and cached arrays
are stored as float16; pair features and fitting use float32.

For residues `i` and `j` with hidden width 1,536, use only the symmetric feature

```text
x(i,j) = h_i elementwise-multiplied-by h_j
```

Fit an L2-regularized logistic regression independently at each stage. This is a
diagonal bilinear contact score with 1,536 feature weights and one intercept. It
cannot learn a full residue-by-residue interaction matrix or a folding network.

For training only, sample at most 256 matched units per protein without replacement;
both the contact and its paired decoy are retained. This limits compute and prevents
large proteins from dominating. Use every available pair for validation and test.

Evaluate regularization strengths `1e-5`, `1e-4`, and `1e-3`. Select the contextual
stage and strength using mean per-protein validation ROC-AUC, with deterministic
tie-breaking toward stronger regularization and then the earlier stage. Refit the
selected model on train plus validation proteins before one final test evaluation.
Apply the same selection and refit procedure to stage 0 as the input baseline.

## Primary and secondary outputs

Primary output:

```text
mean test-protein AUC(contextual probe)
minus
mean test-protein AUC(stage-0 probe)
```

Report a 95% interval from 10,000 paired bootstrap resamples of test proteins.

Secondary descriptive outputs:

- pooled and mean per-protein test AUC for both selected models;
- AUC in the six published sequence-separation bins;
- validation curves for every stage and regularization strength;
- long-range performance for separations above 100 residues;
- the existing ProGen2 attention baseline, clearly labeled as coming from the full
  replacement cohort unless recalculated on exactly the test proteins.

No pair-level p-value is confirmatory because residue pairs within a protein are not
independent.

## Interpretation gates

- Contextual probe approximately equals stage 0: no evidence that contextualization
  adds generally decodable contact information under this probe.
- Contextual probe exceeds stage 0 but not simple sequence controls added later: the
  result may come from contact supervision in the probe rather than ProGen2.
- Contextual probe exceeds stage 0 on held-out proteins, especially beyond 100
  residues: evidence for generalizable contact-predictive information in the frozen
  representation, not proof of mechanistic 3D understanding.
- An unexpectedly large result triggers leakage, split, pair-matching, and probe-
  capacity audits before scientific interpretation.

The five-chain pilot is an engineering test only: verify extraction shape, terminal
removal, cache round-trip, deterministic pair recreation, and fitting on synthetic
fixtures. It is not used to choose thresholds or estimate scientific performance.

The 43,773 cohort residues require approximately 3.51 GiB for the uncompressed
float16 hidden-state cache. Probe fitting is CPU work and must not keep an expensive
GPU Pod running after extraction merely for convenience.
