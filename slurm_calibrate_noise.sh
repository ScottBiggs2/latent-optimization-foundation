#!/bin/bash
#SBATCH --job-name=llm_vae_calib
#SBATCH --output=/scratch/biggs.s/llm_vae/logs/calib_%j.out
#SBATCH --error=/scratch/biggs.s/llm_vae/logs/calib_%j.err
#SBATCH --chdir=/home/biggs.s/llm_vae
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --time=03:00:00
set -e
mkdir -p /scratch/biggs.s/llm_vae/logs
source ~/miniconda/etc/profile.d/conda.sh
conda activate llm_vae
export HF_HOME=/scratch/biggs.s/hf_cache
export HF_DATASETS_CACHE=/scratch/biggs.s/hf_cache
export TRITON_CACHE_DIR=/scratch/biggs.s/triton_cache
python -u calibrate_noise.py \
    --arch_list ${ARCH_LIST:-gpt2_medium smollm2_360m pythia_410m} \
    --scales ${SCALES:-1e-5 1e-4 1e-3 3e-3 1e-2 3e-2 1e-1} \
    --eval_seq_len 1024 --eval_n_sequences 64 \
    --artifact_dir /scratch/biggs.s/llm_vae
