#!/usr/bin/env bash
set -euo pipefail

volume_root=${1:-/workspace/volume}
project_root=${2:-/workspace/project}
tool_root="${volume_root}/tools"
mamba_root="${volume_root}/micromamba-root"
environment_prefix="${volume_root}/envs/progen2-probe"
artifact_root="${volume_root}/bootstrap-artifacts"
micromamba="${tool_root}/micromamba"

micromamba_version=2.8.1-0
micromamba_sha256=9689782d863c05a1bf5d2d371ba527104e7a4eb4310c1637d8653b751aed9c82
micromamba_url="https://github.com/mamba-org/micromamba-releases/releases/download/${micromamba_version}/micromamba-linux-64"

mkdir -p "${tool_root}" "${mamba_root}" "${artifact_root}"

if [[ ! -x "${micromamba}" ]]; then
  wget -O "${micromamba}.download" "${micromamba_url}"
  printf '%s  %s\n' "${micromamba_sha256}" "${micromamba}.download" | sha256sum -c -
  mv "${micromamba}.download" "${micromamba}"
  chmod 755 "${micromamba}"
fi

if [[ ! -x "${environment_prefix}/bin/python" ]]; then
  "${micromamba}" --root-prefix "${mamba_root}" create -y \
    --prefix "${environment_prefix}" \
    --channel conda-forge \
    --channel bioconda \
    python=3.8.20 pip=23.3.2 setuptools=69.5.1 wheel=0.42.0 mmseqs2=15.6f452
fi

"${micromamba}" --root-prefix "${mamba_root}" run --prefix "${environment_prefix}" \
  python -m pip install --index-url https://download.pytorch.org/whl/cu118 \
  'torch==2.0.1+cu118'

"${micromamba}" --root-prefix "${mamba_root}" run --prefix "${environment_prefix}" \
  python -m pip install --no-cache-dir -r "${project_root}/requirements/remote.txt"

"${micromamba}" --root-prefix "${mamba_root}" run --prefix "${environment_prefix}" \
  python -m pip install --no-deps -e "${project_root}"

"${micromamba}" --version > "${artifact_root}/micromamba-version.txt"
"${environment_prefix}/bin/python" -m pip freeze > "${artifact_root}/runpod-pip-freeze.txt"
"${environment_prefix}/bin/python" -c \
  'import torch; print(torch.__version__, torch.version.cuda, torch.cuda.get_device_name(0))' \
  > "${artifact_root}/runpod-torch-runtime.txt"

printf 'Environment ready: %s\n' "${environment_prefix}"
