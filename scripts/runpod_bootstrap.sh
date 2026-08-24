#!/usr/bin/env bash
set -euo pipefail

volume_root=${1:-/workspace/volume}
requested_models=${2:-progen2-base}
upstream_root="${volume_root}/upstream"
checkpoint_root="${volume_root}/checkpoints"
artifact_root="${volume_root}/bootstrap-artifacts"
environment_python="${volume_root}/envs/progen2-probe/bin/python"

mkdir -p "${upstream_root}" "${checkpoint_root}" "${artifact_root}"

progen_commit=c27a419c234a0997923761e1fe7daffcebf0eaf5
esm_commit=2b369911bb5b4b0dda914521b9475cad1656b2ac

if [[ ! -d "${upstream_root}/progen/.git" ]]; then
  git clone https://github.com/salesforce/progen.git "${upstream_root}/progen"
fi
git -C "${upstream_root}/progen" fetch --all --tags
git -C "${upstream_root}/progen" checkout --detach "${progen_commit}"

if [[ ! -d "${upstream_root}/esm/.git" ]]; then
  git clone https://github.com/facebookresearch/esm.git "${upstream_root}/esm"
fi
git -C "${upstream_root}/esm" fetch --all --tags
git -C "${upstream_root}/esm" checkout --detach "${esm_commit}"

checkpoint_archives=()
for model_name in ${requested_models}; do
  case "${model_name}" in
    progen2-base) ;;
    *)
      printf 'Unsupported checkpoint requested: %s\n' "${model_name}" >&2
      exit 2
      ;;
  esac
  model_dir="${checkpoint_root}/${model_name}"
  archive="${checkpoint_root}/${model_name}.tar.gz"
  if [[ ! -f "${archive}" ]]; then
    wget -O "${archive}" "https://storage.googleapis.com/sfr-progen-research/checkpoints/${model_name}.tar.gz"
  fi
  mkdir -p "${model_dir}"
  if [[ ! -f "${model_dir}/config.json" ]]; then
    tar -xzf "${archive}" -C "${model_dir}"
  fi
  checkpoint_archives+=("${archive}")
done

git -C "${upstream_root}/progen" rev-parse HEAD > "${artifact_root}/progen.commit"
git -C "${upstream_root}/esm" rev-parse HEAD > "${artifact_root}/esm.commit"
sha256sum "${checkpoint_archives[@]}" > "${artifact_root}/checkpoint-archives.sha256"
"${environment_python}" -m pip freeze > "${artifact_root}/pip-freeze.txt"
nvidia-smi -q > "${artifact_root}/nvidia-smi.txt"

printf 'Bootstrap complete under %s\n' "${volume_root}"
