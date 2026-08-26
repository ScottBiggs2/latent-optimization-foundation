#!/bin/bash
#SBATCH --job-name=llm_vae_diag_refit
#SBATCH --output=/scratch/biggs.s/llm_vae/logs/diag_refit_%j.out
#SBATCH --error=/scratch/biggs.s/llm_vae/logs/diag_refit_%j.err
#SBATCH --chdir=/home/biggs.s/llm_vae
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --time=02:00:00
set -e
mkdir -p /scratch/biggs.s/llm_vae/logs
source ~/miniconda/etc/profile.d/conda.sh
conda activate llm_vae
export HF_HOME=/scratch/biggs.s/hf_cache
export HF_DATASETS_CACHE=/scratch/biggs.s/hf_cache
export TRITON_CACHE_DIR=/scratch/biggs.s/triton_cache
R=/scratch/biggs.s/llm_vae
OUT=$R/runs/2026-08-21-shared-6family/pca_eigh_refit

echo "### Stage 1: refit shared basis with eigh"
python -u diag_refit_shared_pca.py --artifact_dir "$R" --out_dir "$OUT"

echo; echo "### Stage 2: PCA-only eval through the refit basis"
python -u eval_pca_only.py --artifact_dir "$R" --pca_dir "$OUT" \
    --eval_seq_len 1024 --eval_n_sequences 64 --no_wandb

echo "Done."
