#!/bin/bash
# slurm_train.sh
#
# Submit the LLM-VAE training pipeline to Explorer HPC.
#
# Usage (from the project directory on Explorer login node):
#   sbatch slurm_train.sh
#
# Optional overrides via environment:
#   MODE=tiny sbatch slurm_train.sh    # smoke test without downloads
#   EVAL_LM=0 sbatch slurm_train.sh    # skip LM eval to save time
#   MC_EVAL=1 sbatch slurm_train.sh    # also run MMLU/HellaSwag/GPQA eval
#
# MC_EVAL requires GPQA access: accept terms at
# huggingface.co/datasets/Idavidrein/gpqa and set HF_TOKEN below (same
# gating pattern as gemma3_270m). Training/PCA/VAE stages are checkpointed,
# so resubmitting with MC_EVAL=1 on an existing artifact_dir skips straight
# to the new evaluation stage.

#SBATCH --job-name=llm_vae_train
#SBATCH --output=/scratch/biggs.s/llm_vae/slurm_%j.out
#SBATCH --error=/scratch/biggs.s/llm_vae/slurm_%j.err
# SLURM copies the script to its spool dir before running it, so
# $(dirname "$0") would point there instead of the project root.
# --chdir is processed before the script executes and reliably sets CWD.
#SBATCH --chdir=/home/biggs.s/llm_vae

# Resources: V100 (32 GB) is sufficient; upgrade to A100 if OOM
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --time=06:00:00

set -e

# ---- Environment ----
source ~/miniconda/etc/profile.d/conda.sh
conda activate llm_vae

# All large files go to scratch — never to $HOME
export HF_HOME=/scratch/biggs.s/hf_cache
# HF_TOKEN is required for gated models/datasets (e.g. google/gemma-3-270m,
# or the GPQA dataset used by MC_EVAL). The default config needs no token.
# Uncomment and fill in if you add gemma3_270m to --arch_list, or set MC_EVAL=1:
# export HF_TOKEN=hf_your_token_here
export HF_DATASETS_CACHE=/scratch/biggs.s/hf_cache
export TRITON_CACHE_DIR=/scratch/biggs.s/triton_cache
export ARTIFACT_DIR=/scratch/biggs.s/llm_vae

mkdir -p "$ARTIFACT_DIR" "$HF_HOME" "$TRITON_CACHE_DIR"

# ---- Config ----
MODE="${MODE:-full}"
EVAL_LM_FLAG=""
if [ "${EVAL_LM:-1}" != "0" ]; then
    EVAL_LM_FLAG="--eval_lm"
fi
MC_EVAL_FLAG=""
if [ "${MC_EVAL:-0}" != "0" ]; then
    MC_EVAL_FLAG="--eval_mc"
fi

echo "=============================================="
echo "Job ID   : $SLURM_JOB_ID"
echo "Node     : $SLURMD_NODENAME"
echo "GPU      : $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
echo "Mode     : $MODE"
echo "Artifact : $ARTIFACT_DIR"
echo "=============================================="

python run_hpc.py \
    --mode "$MODE" \
    --artifact_dir "$ARTIFACT_DIR" \
    --n_components 97 \
    --pca_batch_size 10 \
    --latent_dim 32 \
    --hidden_dim 256 \
    --cond_dim 64 \
    --val_fraction 0.0 \
    --epochs 500 \
    --patience 50 \
    --warmup_epochs 50 \
    --beta 1.0 \
    --lr 3e-4 \
    --batch_size 32 \
    --eval_seq_len 512 \
    --eval_n_sequences 32 \
    $EVAL_LM_FLAG \
    $MC_EVAL_FLAG

echo "Done — results in $ARTIFACT_DIR/results/"
