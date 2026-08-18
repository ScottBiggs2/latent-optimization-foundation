#!/bin/bash
# slurm_eval_mc.sh
#
# Standalone multiple-choice benchmark evaluation (eval_mc.py's CLI).
# Hooks into either raw HuggingFace checkpoints or a trained PCA+VAE artifact
# set produced by slurm_train.sh — does not require re-running extraction or
# training, and does not require sbatch'ing the whole run_hpc.py pipeline.
#
# Usage (from the project directory on Explorer login node):
#   sbatch slurm_eval_mc.sh                       # reconstruct mode against
#                                                  # $ARTIFACT_DIR's trained PCA/VAE
#   RAW=1 sbatch slurm_eval_mc.sh                 # hf-model-path mode: stock
#                                                  # checkpoints, no PCA/VAE needed
#   MODE=generate sbatch slurm_eval_mc.sh         # generative mode: sample fresh
#                                                  # blocks from the VAE prior
#                                                  # instead of reconstructing real
#                                                  # ones, then MC-eval the result
#
# Optional overrides via environment:
#   ARCH_LIST="gpt2_medium smollm2_360m"          # restrict to these archs
#   ARTIFACT_DIR=/scratch/biggs.s/llm_vae         # where blocks/pca/vae/results live
#   PCA_DIR=... / VAE_DIR=...                     # override artifact_dir/{pca,vae}
#   MC_BENCHMARKS="mmlu hellaswag"                # skip GPQA
#   MC_N_QUESTIONS=200
#   SKIP_SIMPLE_EVAL=1                            # skip the cheap block-level
#                                                  # cosine-sim/MSE pass
#
# gemma3_270m / GPQA are gated: accept terms at huggingface.co/google/gemma-3-270m
# and huggingface.co/datasets/Idavidrein/gpqa, then export HF_TOKEN in your shell
# before submitting — sbatch inherits it automatically:
#   export HF_TOKEN=hf_...  &&  sbatch slurm_eval_mc.sh
# Do NOT hardcode a token into this file — it's tracked by git.

#SBATCH --job-name=llm_vae_eval_mc
#SBATCH --output=/scratch/biggs.s/llm_vae/slurm_eval_mc_%j.out
#SBATCH --error=/scratch/biggs.s/llm_vae/slurm_eval_mc_%j.err
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
# HF_TOKEN (if needed for GPQA / gemma3_270m) is picked up from the
# submitting shell's environment via sbatch — nothing to export here.
export HF_DATASETS_CACHE=/scratch/biggs.s/hf_cache
export TRITON_CACHE_DIR=/scratch/biggs.s/triton_cache
export ARTIFACT_DIR="${ARTIFACT_DIR:-/scratch/biggs.s/llm_vae}"

mkdir -p "$ARTIFACT_DIR" "$HF_HOME" "$TRITON_CACHE_DIR"

RAW_FLAG=""
if [ "${RAW:-0}" != "0" ]; then
    RAW_FLAG="--raw"
fi
MODE_FLAG=""
if [ -n "${MODE:-}" ]; then
    MODE_FLAG="--mode $MODE"
fi
ARCH_LIST_FLAG=""
if [ -n "${ARCH_LIST:-}" ]; then
    ARCH_LIST_FLAG="--arch_list $ARCH_LIST"
fi
PCA_DIR_FLAG=""
if [ -n "${PCA_DIR:-}" ]; then
    PCA_DIR_FLAG="--pca_dir $PCA_DIR"
fi
VAE_DIR_FLAG=""
if [ -n "${VAE_DIR:-}" ]; then
    VAE_DIR_FLAG="--vae_dir $VAE_DIR"
fi
MC_BENCHMARKS_FLAG=""
if [ -n "${MC_BENCHMARKS:-}" ]; then
    MC_BENCHMARKS_FLAG="--mc_benchmarks $MC_BENCHMARKS"
fi
MC_N_Q_FLAG=""
if [ -n "${MC_N_QUESTIONS:-}" ]; then
    MC_N_Q_FLAG="--mc_n_questions $MC_N_QUESTIONS"
fi
SKIP_SIMPLE_EVAL_FLAG=""
if [ "${SKIP_SIMPLE_EVAL:-0}" != "0" ]; then
    SKIP_SIMPLE_EVAL_FLAG="--skip_simple_eval"
fi

echo "=============================================="
echo "Job ID   : $SLURM_JOB_ID"
echo "Node     : $SLURMD_NODENAME"
echo "GPU      : $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
echo "Artifact : $ARTIFACT_DIR"
echo "Raw      : ${RAW:-0}"
echo "Mode     : ${MODE:-reconstruct}"
echo "=============================================="

python -u eval_mc.py \
    --artifact_dir "$ARTIFACT_DIR" \
    $RAW_FLAG \
    $MODE_FLAG \
    $ARCH_LIST_FLAG \
    $PCA_DIR_FLAG \
    $VAE_DIR_FLAG \
    $MC_BENCHMARKS_FLAG \
    $MC_N_Q_FLAG \
    $SKIP_SIMPLE_EVAL_FLAG

echo "Done — results in $ARTIFACT_DIR/results/"
