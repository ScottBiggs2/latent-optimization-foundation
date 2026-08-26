#!/bin/bash
# slurm_stack_run.sh
#
# Full whole-stack run: per-family Gram PCA -> StackVAE -> evaluation, at both
# rank arms. One PCA sample is one complete decoder stack (DeepWeightFlow framing).
#
#   sbatch slurm_stack_run.sh
#   N_SAMPLES=143 NOISE_SCALE=3e-3 RUN=perfam-s3e3 sbatch slurm_stack_run.sh
#
# noise_scale=1e-2 is CALIBRATED, not a guess (job 9678992, WikiText-2 64x1024):
#   s        gpt2_medium   smollm2_360m   pythia_410m
#   1e-3        +0.004%        -0.000%       -0.003%
#   3e-3        +0.016%        +0.016%       +0.057%
#   1e-2        +0.097%        +0.250%       +0.956%   <- largest s within +1% for all
#   3e-2        +0.699%        +2.378%       +9.633%
#   1e-1        +8.158%       +37.036%     +102.694%
# 1e-2 is the largest scale that keeps every family under +1%. It is also ~5 orders
# of magnitude above float32 epsilon, so the ensemble members separate cleanly in the
# Gram matrix, and it makes the k=N/2 arm measurable instead of lost in noise.
#
# Prediction to check against: at k=N/2 the truncation residual is ~s*sigma/sqrt(2)
# = 7.1e-3*sigma, i.e. between the 3e-3 and 1e-2 rows above -- so expect roughly
# +0.5% for pythia_410m. At k=N-1 reconstruction is exact by the rank bound
# (cosine 1.0, dPPL ~0); anything else there is a bug, not a finding.

#SBATCH --job-name=llm_vae_stack
#SBATCH --output=/scratch/biggs.s/llm_vae/logs/stack_%j.out
#SBATCH --error=/scratch/biggs.s/llm_vae/logs/stack_%j.err
#SBATCH --chdir=/home/biggs.s/llm_vae
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=96G
#SBATCH --cpus-per-task=8
#SBATCH --time=06:00:00

set -e
mkdir -p /scratch/biggs.s/llm_vae/logs
source ~/miniconda/etc/profile.d/conda.sh
conda activate llm_vae
export HF_HOME=/scratch/biggs.s/hf_cache
export HF_DATASETS_CACHE=/scratch/biggs.s/hf_cache
export TRITON_CACHE_DIR=/scratch/biggs.s/triton_cache

ROOT=/scratch/biggs.s/llm_vae
RUN="${RUN:-perfam}"
N_SAMPLES="${N_SAMPLES:-100}"
NOISE_SCALE="${NOISE_SCALE:-1e-2}"
K_FULL=$((N_SAMPLES - 1))
K_HALF=$((N_SAMPLES / 2))

echo "=============================================="
echo "Job      : $SLURM_JOB_ID on $SLURMD_NODENAME"
echo "GPU      : $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
echo "Run      : $RUN   N=$N_SAMPLES  s=$NOISE_SCALE"
echo "Ranks    : k=$K_FULL (N-1)  and  k=$K_HALF (N/2)"
echo "=============================================="

COMMON="--mode full --run_name $RUN --artifact_dir $ROOT \
        --n_samples $N_SAMPLES --noise_scale $NOISE_SCALE \
        --exclude_1d --chunk_budget_mb 512 \
        --latent_dim 32 --hidden_dim 256 --cond_dim 64 \
        --epochs 3000 --patience 300 --warmup_epochs 150 \
        --beta 1.0 --free_bits 0.05 --cond_dropout 0.15 \
        --code_noise_std 0.02 --lr 3e-4 --batch_size 64"

echo; echo "######## Stage A: PCA fits + VAE at k=$K_FULL ########"
python -u train_stack.py $COMMON --k "$K_FULL"

echo; echo "######## Stage B: VAE at k=$K_HALF (reuses the PCA fits) ########"
python -u train_stack.py $COMMON --k "$K_HALF"

echo; echo "######## Stage C: evaluation, both ranks, all arms ########"
python -u eval_stack.py --artifact_dir "$ROOT" --run_name "$RUN" --mode full \
    --k "$K_FULL" "$K_HALF" --arms pca_only vae generate \
    --eval_seq_len 1024 --eval_n_sequences 64

echo; echo "Done — results in $ROOT/runs/$RUN/results/"
