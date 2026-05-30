#!/bin/bash
# setup_env_hpc.sh
#
# Creates the 'llm_vae' conda environment on Explorer HPC.
#
# Usage — run from the login node, it handles the allocation itself:
#
#   bash setup_env_hpc.sh
#
# The pip/conda install is too heavy for the login node, so the script
# detects if it is running there and automatically re-launches itself
# on a short CPU allocation (15 min, 4 CPUs, 8 GB RAM) via srun.
# If you are already on a compute node the allocation step is skipped.
#
# Safe to re-run: removes any stale env first.

set -e

ENV_NAME="llm_vae"

# ── Login-node guard ──────────────────────────────────────────────────────────
# SLURM_JOB_ID is only set inside an active allocation.
# If it is absent we are still on the login node and must request a node first.
if [ -z "$SLURM_JOB_ID" ]; then
    echo "=== Login node detected — requesting a 15-min CPU compute node ==="
    srun \
        --partition=short \
        --nodes=1 \
        --ntasks=1 \
        --cpus-per-task=4 \
        --mem=8G \
        --time=00:15:00 \
        bash "$0" "$@"
    exit $?   # propagate the job's exit code back to the login shell
fi
# ─────────────────────────────────────────────────────────────────────────────

echo "=== Running on compute node: ${SLURMD_NODENAME:-unknown} ==="
echo "=== Setting up conda env: $ENV_NAME ==="

source ~/miniconda/etc/profile.d/conda.sh

if conda env list | grep -q "^${ENV_NAME}[[:space:]]"; then
    echo "Removing existing env '$ENV_NAME' …"
    conda env remove -n "$ENV_NAME" -y
fi

# Python 3.11 — stable for PyTorch 2.x
conda create -n "$ENV_NAME" python=3.11 -y

conda activate "$ENV_NAME"

# PyTorch with CUDA 12.1 (matches Explorer's cuda/12.8.0 driver)
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
