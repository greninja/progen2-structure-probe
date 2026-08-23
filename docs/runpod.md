# RunPod execution

Use one Secure Cloud Pod with a single RTX 4090 (24 GB), at least 40 GB host RAM,
and an 80–100 GB persistent volume mounted at `/workspace/volume`. Heavy data work
and all model inference run there. Only compact results, tests, plots, and analysis
belong on the local machine.

## Create the isolated runtime

The observed RTX 4090 Pod base environment uses Python 3.11, PyTorch 2.4.1, and
CUDA 12.4. Preserve that functioning base environment. Create a persistent isolated
Python 3.8 runtime with the hash-pinned micromamba bootstrap:

```bash
bash scripts/runpod_create_env.sh /workspace/volume /workspace/project
```

The isolated runtime pins PyTorch 2.0.1 with CUDA 11.8, selected for RTX 4090
compatibility with the older ProGen2 dependency stack. This is an
environment compatibility substitution, not a recovered Mandrake setting. Its
package freeze and CUDA runtime check are written under
`/workspace/volume/bootstrap-artifacts`.

Run commands through:

```bash
/workspace/volume/envs/progen2-probe/bin/python
/workspace/volume/envs/progen2-probe/bin/progen2-probe
```

## Source and checkpoint bootstrap

The custom Dockerfile encodes the same Python/PyTorch/CUDA compatibility choice for
future image builds. The current Pod uses the isolated persistent environment above.
Then run:

```bash
cd /workspace/project
bash scripts/runpod_bootstrap.sh /workspace/volume progen2-base
```

The bootstrap script checks out the pinned Salesforce and ESM commits, downloads only
the requested official ProGen2 archive, and records archive, environment, driver, and
source provenance. The Experiment 1 default is `progen2-base`; do not download
`progen2-small` unless Experiment 2 is explicitly resumed. The persistent volume
prevents repeated model downloads.

Before inference, populate and hash the manifests described in
`data/manifests/README.md`. Do not fill unresolved Mandrake sequence or structure IDs
by guesswork.

## Validate locally or remotely

```bash
progen2-probe validate-config configs/experiment1_pilot.yaml
progen2-probe validate-config configs/experiment2_reproduction.yaml
python -m pytest
```

## Model and memory smoke test

Run deterministic extraction before downloading structures or launching the pilot:

```bash
TORCH_HOME=/workspace/volume/torch-cache \
/workspace/volume/envs/progen2-probe/bin/progen2-probe \
  smoke-models configs/experiment1_pilot.yaml \
  --progen-repo /workspace/volume/upstream/progen \
  --progen-checkpoint /workspace/volume/checkpoints/progen2-base \
  --esm-repo /workspace/volume/upstream/esm \
  --sequence-length 128 \
  --output-dir /workspace/volume/results/smoke
```

Review `smoke.json` for tensor shapes, determinism, causal masking, wall time, and
peak CUDA allocation. This is an engineering check, not a contact result.

## Build the documented fallback cohort

Only use this when the unpublished Mandrake manifest remains unavailable:

```bash
progen2-probe build-cohort configs/experiment1_pilot.yaml \
  --work-dir /workspace/volume/data/experiment1-cohort \
  --threads 16
```

The command is restartable and produces raw RCSB request/response records,
candidate CSV/FASTA, MMseqs2 clusters, rejection records, downloaded mmCIF files,
the frozen 150-chain manifest, and the five-chain pilot manifest. Do not delete or
edit individual candidates after seeing model results; rebuild under a new work
directory if the predeclared cohort protocol changes. The runner resolves the pinned
MMseqs2 executable either from `PATH` or beside the active Python interpreter, so
invoking the environment's `progen2-probe` entry point by absolute path remains safe.

## Five-protein Experiment 1 pilot

```bash
progen2-probe experiment1 configs/experiment1_pilot.yaml \
  --progen-repo /workspace/volume/upstream/progen \
  --progen-checkpoint /workspace/volume/checkpoints/progen2-base \
  --esm-repo /workspace/volume/upstream/esm \
  --manifest /workspace/volume/data/experiment1-cohort/experiment1_pilot.csv \
  --output-dir /workspace/volume/results/experiment1/pilot
```

Do not launch the 150-chain run until the five pilot chains pass all methodology
checks and measured memory/runtime have been reviewed.

## Experiment 2

```bash
progen2-probe experiment2 configs/experiment2_reproduction.yaml \
  --progen-repo /workspace/volume/upstream/progen \
  --progen-checkpoint /workspace/volume/checkpoints/progen2-small \
  --output-dir /workspace/volume/results/experiment2
```

Terminate GPU compute when a batch completes. Retain only the persistent volume while
reviewing results. Sync `resolved_config.json`, JSON summaries, per-chain NPZ files,
bootstrap metadata, and logs back to the local repository's ignored `results/` tree.
