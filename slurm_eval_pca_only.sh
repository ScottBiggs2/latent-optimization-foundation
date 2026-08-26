#!/bin/bash
# slurm_eval_pca_only.sh
#
# PCA-only reconstruction + PPL evaluation (eval_pca_only.py's CLI) — the
# VAE-free control. Runs against an existing blocks/pca artifact set produced by
# slurm_train.sh; needs no VAE checkpoint and re-fits nothing.
#
# Why: every other number in this repo measures PCA and VAE jointly, so when a
# family reconstructs badly (qwen3 0.586, pythia_160m 0.617, pythia_410m 0.375 on
# the 2026-08-21 run) there is no way to attribute it. This separates them:
#   bad here too                 -> the shared PCA basis is the problem
#   fine here, bad with the VAE  -> the VAE is the problem
#
# Usage (from the project directory on Explorer login node):
#   sbatch slurm_eval_pca_only.sh                        # full fitted basis
#   N_COMPONENTS="79 40 20 10" sbatch slurm_eval_pca_only.sh   # rank sweep
#
# Optional overrides via environment:
#   ARCH_LIST="pythia_410m"                # restrict to these archs
#   ARTIFACT_DIR=/scratch/biggs.s/llm_vae  # where blocks/pca/results live
#   PCA_DIR=...                            # override artifact_dir/pca
#   N_COMPONENTS="40 20"                   # rank sweep (space-separated)
#   EXCLUDE_1D=1 / EXCLUDE_1D=0            # override dataset_meta.json's setting
#   EVAL_SEQ_LEN=1024 / EVAL_N_SEQUENCES=64
#
# No gated assets are touched (WikiText-2 only), so no HF_TOKEN is needed.

#SBATCH --job-name=llm_vae_pca_only
#SBATCH --output=/scratch/biggs.s/llm_vae/slurm_pca_only_%j.out
#SBATCH --error=/scratch/biggs.s/llm_vae/slurm_pca_only_%j.err
#SBATCH --chdir=/home/biggs.s/llm_vae

#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --time=02:00:00

set -e

source ~/miniconda/etc/profile.d/conda.sh
conda activate llm_vae

export HF_HOME=/scratch/biggs.s/hf_cache
export HF_DATASETS_CACHE=/scratch/biggs.s/hf_cache
export TRITON_CACHE_DIR=/scratch/biggs.s/triton_cache
export ARTIFACT_DIR="${ARTIFACT_DIR:-/scratch/biggs.s/llm_vae}"

mkdir -p "$ARTIFACT_DIR" "$HF_HOME" "$TRITON_CACHE_DIR"

ARCH_LIST_FLAG=""
if [ -n "${ARCH_LIST:-}" ]; then
    ARCH_LIST_FLAG="--arch_list $ARCH_LIST"
fi
PCA_DIR_FLAG=""
if [ -n "${PCA_DIR:-}" ]; then
    PCA_DIR_FLAG="--pca_dir $PCA_DIR"
fi
N_COMPONENTS_FLAG=""
if [ -n "${N_COMPONENTS:-}" ]; then
    N_COMPONENTS_FLAG="--n_components $N_COMPONENTS"
fi
# Tri-state: unset = inherit dataset_meta.json; 1 = force on; 0 = force off.
EXCLUDE_1D_FLAG=""
if [ "${EXCLUDE_1D:-}" = "1" ]; then
    EXCLUDE_1D_FLAG="--exclude_1d"
fi
SEQ_LEN_FLAG=""
if [ -n "${EVAL_SEQ_LEN:-}" ]; then
    SEQ_LEN_FLAG="--eval_seq_len $EVAL_SEQ_LEN"
fi
N_SEQ_FLAG=""
if [ -n "${EVAL_N_SEQUENCES:-}" ]; then
    N_SEQ_FLAG="--eval_n_sequences $EVAL_N_SEQUENCES"
fi

echo "=============================================="
echo "Job ID       : $SLURM_JOB_ID"
echo "Node         : $SLURMD_NODENAME"
echo "GPU          : $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
echo "Artifact     : $ARTIFACT_DIR"
echo "Arch list    : ${ARCH_LIST:-<from dataset_meta.json>}"
echo "n_components : ${N_COMPONENTS:-<full fitted basis>}"
echo "exclude_1d   : ${EXCLUDE_1D:-<from dataset_meta.json>}"
echo "=============================================="

python -u eval_pca_only.py \
    --artifact_dir "$ARTIFACT_DIR" \
    $ARCH_LIST_FLAG \
    $PCA_DIR_FLAG \
    $N_COMPONENTS_FLAG \
    $EXCLUDE_1D_FLAG \
    $SEQ_LEN_FLAG \
    $N_SEQ_FLAG

echo "Done — results in $ARTIFACT_DIR/results/pca_only_eval_results.json"
