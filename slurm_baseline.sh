#!/bin/bash
# slurm_baseline.sh
#
# Baseline evaluation on stock HuggingFace checkpoints — no PCA/VAE required.
# Gives a clean, independently reproducible reference point for the
# before/after deltas reported by eval_lm.py / eval_mc.py.
#
# Usage (from the project directory on Explorer login node):
#   sbatch slurm_baseline.sh
#
# Optional overrides via environment:
#   ARCH_LIST="gpt2_medium smollm2_360m" sbatch slurm_baseline.sh
#   MC_BENCHMARKS="mmlu hellaswag" sbatch slurm_baseline.sh   # skip GPQA
#   MC_N_QUESTIONS=200 sbatch slurm_baseline.sh
#   SKIP_PPL=1 sbatch slurm_baseline.sh                       # MC only
#   SKIP_MC=1 sbatch slurm_baseline.sh                        # PPL only
#
# GPQA requires accepting terms at huggingface.co/datasets/Idavidrein/gpqa
# and setting HF_TOKEN below (same gating pattern as gemma3_270m).

#SBATCH --job-name=llm_vae_baseline
#SBATCH --output=/scratch/biggs.s/llm_vae/slurm_baseline_%j.out
#SBATCH --error=/scratch/biggs.s/llm_vae/slurm_baseline_%j.err
#SBATCH --chdir=/home/biggs.s/llm_vae

#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --time=03:00:00

set -e

source ~/miniconda/etc/profile.d/conda.sh
conda activate llm_vae

export HF_HOME=/scratch/biggs.s/hf_cache
# Uncomment for GPQA / gemma3_270m:
# export HF_TOKEN=hf_your_token_here
export HF_DATASETS_CACHE=/scratch/biggs.s/hf_cache
export TRITON_CACHE_DIR=/scratch/biggs.s/triton_cache
export ARTIFACT_DIR=/scratch/biggs.s/llm_vae

mkdir -p "$ARTIFACT_DIR" "$HF_HOME" "$TRITON_CACHE_DIR"

ARCH_LIST_FLAG=""
if [ -n "${ARCH_LIST:-}" ]; then
    ARCH_LIST_FLAG="--arch_list $ARCH_LIST"
fi
MC_BENCHMARKS_FLAG=""
if [ -n "${MC_BENCHMARKS:-}" ]; then
    MC_BENCHMARKS_FLAG="--mc_benchmarks $MC_BENCHMARKS"
fi
MC_N_Q_FLAG=""
if [ -n "${MC_N_QUESTIONS:-}" ]; then
    MC_N_Q_FLAG="--mc_n_questions $MC_N_QUESTIONS"
fi
SKIP_PPL_FLAG=""
if [ "${SKIP_PPL:-0}" != "0" ]; then
    SKIP_PPL_FLAG="--skip_ppl"
fi
SKIP_MC_FLAG=""
if [ "${SKIP_MC:-0}" != "0" ]; then
    SKIP_MC_FLAG="--skip_mc"
fi

echo "=============================================="
echo "Job ID   : $SLURM_JOB_ID"
echo "Node     : $SLURMD_NODENAME"
echo "GPU      : $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
echo "Artifact : $ARTIFACT_DIR"
echo "=============================================="

python -u eval_baseline.py \
    --artifact_dir "$ARTIFACT_DIR" \
    $ARCH_LIST_FLAG \
    $MC_BENCHMARKS_FLAG \
    $MC_N_Q_FLAG \
    $SKIP_PPL_FLAG \
    $SKIP_MC_FLAG

echo "Done — results in $ARTIFACT_DIR/results/baseline_results.json"
