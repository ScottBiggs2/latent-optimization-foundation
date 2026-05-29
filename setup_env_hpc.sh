#!/bin/bash
# setup_env_hpc.sh
#
# Creates the 'llm_vae' conda environment on Explorer HPC.
# Run once from the login node BEFORE submitting any SLURM jobs:
#
#   bash setup_env_hpc.sh
#
# Safe to re-run: removes any stale env first.

set -e

ENV_NAME="llm_vae"

echo "=== Setting up conda env: $ENV_NAME ==="

source ~/miniconda/etc/profile.d/conda.sh

if conda env list | grep -q "^${ENV_NAME}[[:space:]]"; then
    echo "Removing existing env '$ENV_NAME' …"
    conda env remove -n "$ENV_NAME" -y
fi

# Python 3.11 — stable for PyTorch 2.x
conda create -n "$ENV_NAME" python=3.11 -y

conda activate "$ENV_NAME"

# PyTorch with CUDA 12.1 (matches cuda/12.8.0 driver on Explorer)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# All project dependencies
pip install \
    numpy \
    scipy \
    scikit-learn \
    transformers \
    accelerate \
    huggingface-hub \
    datasets \
    tqdm

echo ""
echo "=== Environment '$ENV_NAME' ready ==="
echo "Activate with:  conda activate $ENV_NAME"
conda deactivate
