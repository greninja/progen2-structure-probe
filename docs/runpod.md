# RunPod execution

Use one Secure Cloud Pod with a single RTX 4090 (24 GB), at least 40 GB host RAM,
and an 80–100 GB persistent volume mounted at `/workspace/volume`. Heavy data work
and all model inference run there. Only compact results, tests, plots, and analysis
belong on the local machine.

## Build and bootstrap

Build `Dockerfile.runpod` as a RunPod custom image or build it in a Pod after syncing
the repository. Then run:

```bash
cd /workspace/project
bash scripts/runpod_bootstrap.sh /workspace/volume
```

The bootstrap script checks out the pinned Salesforce and ESM commits, downloads the
official ProGen2 base and small archives, and records archive, environment, driver,
and source provenance. The persistent volume prevents repeated model downloads.

Before inference, populate and hash the manifests described in
`data/manifests/README.md`. Do not fill unresolved Mandrake sequence or structure IDs
by guesswork.

## Validate locally or remotely

```bash
progen2-probe validate-config configs/experiment1_pilot.yaml
progen2-probe validate-config configs/experiment2_reproduction.yaml
python -m pytest
```

## Five-protein Experiment 1 pilot

```bash
progen2-probe experiment1 configs/experiment1_pilot.yaml \
  --progen-repo /workspace/volume/upstream/progen \
  --progen-checkpoint /workspace/volume/checkpoints/progen2-base \
  --esm-repo /workspace/volume/upstream/esm \
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

