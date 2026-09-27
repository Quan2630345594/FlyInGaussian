#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENV_NAME="gaussiangym"
ISAAC_GYM_ROOT="${ISAAC_GYM_ROOT:-$HOME/isaacgym}"
RLPX4_ROOT="${RLPX4_ROOT:-$HOME/rlPx4Controller}"

if ! command -v conda >/dev/null 2>&1; then
    echo "Conda is required. Install Miniconda or Anaconda first." >&2
    exit 1
fi

CONDA_DIR="$(conda info --base)"
source "${CONDA_DIR}/etc/profile.d/conda.sh"

if conda env list | grep -Eq "^[[:space:]]*${ENV_NAME}[[:space:]]"; then
    echo "Using existing Conda environment: ${ENV_NAME}"
else
    conda env create -f "${ROOT_DIR}/conda_env.yml"
fi
conda activate "${ENV_NAME}"

python -m pip install --upgrade pip
python -m pip install \
    torch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 \
    --index-url https://download.pytorch.org/whl/cu118

# PyTorch3D has no matching PyPI wheel for this stack; build the tagged source.
python -m pip install \
    "git+https://github.com/facebookresearch/pytorch3d.git@V0.7.8"

if [ ! -d "${RLPX4_ROOT}/.git" ]; then
    git clone https://github.com/emNavi/rlPx4Controller.git "${RLPX4_ROOT}"
fi
python -m pip install -e "${RLPX4_ROOT}"
python -m pip install -e "${ROOT_DIR}"

if [ -d "${ISAAC_GYM_ROOT}/python" ]; then
    python -m pip install -e "${ISAAC_GYM_ROOT}/python"
else
    echo "Isaac Gym was not found at ${ISAAC_GYM_ROOT}."
    echo "Download Preview 4 from NVIDIA, extract it, then run:"
    echo "  ISAAC_GYM_ROOT=/path/to/isaacgym ./configuration.sh"
fi

echo "Environment setup complete. Activate it with: conda activate ${ENV_NAME}"
