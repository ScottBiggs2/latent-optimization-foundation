#!/bin/bash
#SBATCH --job-name=llm_vae_stack_typ
#SBATCH --output=/scratch/biggs.s/llm_vae/logs/stack_typ_%j.out
#SBATCH --error=/scratch/biggs.s/llm_vae/logs/stack_typ_%j.err
#SBATCH --chdir=/home/biggs.s/llm_vae
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=96G
#SBATCH --cpus-per-task=8
#SBATCH --time=02:00:00
set -e
source ~/miniconda/etc/profile.d/conda.sh
conda activate llm_vae
export HF_HOME=/scratch/biggs.s/hf_cache
export HF_DATASETS_CACHE=/scratch/biggs.s/hf_cache
export TRITON_CACHE_DIR=/scratch/biggs.s/triton_cache
# Reconstruct a TYPICAL ensemble member instead of sample 0, so the rank sweep
# actually measures truncation loss. Prediction: k=99 exact (rank bound);
# k=50 residual ~ s*sigma/sqrt(2) = 7.1e-3*sigma -> roughly +0.5% for pythia_410m.
python -u eval_stack.py --artifact_dir /scratch/biggs.s/llm_vae --run_name perfam \
    --mode full --k 99 50 --arms pca_only vae --sample_idx 5 \
    --eval_seq_len 1024 --eval_n_sequences 64
