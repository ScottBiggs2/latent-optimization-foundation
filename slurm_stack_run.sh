#!/bin/bash
# slurm_stack_run.sh
#
# Full whole-stack run: per-family Gram PCA -> StackVAE -> evaluation, at both
# rank arms. One PCA sample is one complete decoder stack (DeepWeightFlow framing).
#
#   sbatch slurm_stack_run.sh
#   N_SAMPLES=143 NOISE_SCALE=3e-3 RUN=perfam-s3e3 sbatch slurm_stack_run.sh
#
# noise_scale=3e-3 is CALIBRATED on the CURRENT layout -- embeddings, final norm and
# the untied LM head are all in D (job 9894023, WikiText-2 64x1024):
#   s        gpt2_medium   smollm2_360m   pythia_410m
#   1e-4        +0.003%        -0.001%       -0.000%
#   1e-3        +0.070%        -0.003%       +0.005%
#   3e-3        +0.449%        +0.030%       +0.075%   <- largest s within +0.5% for all
#   5e-3        +1.151%        +0.116%       +0.228%
#   1e-2        +4.363%        +0.567%       +0.994%
#   3e-2       +43.933%        +5.862%      +11.021%
#
# DO NOT reuse the old decoder-blocks-only table (job 9678992). It said 1e-2 cost
# gpt2_medium +0.097%; on the whole stack the same s costs +4.363%, a 45x change.
#
# The reason is NOT the logit-facing fraction, which was the obvious guess and is
# wrong: pythia_410m has the largest extra segment (25.4% of D, and its LM head is
# untied) yet is the second most robust family here. It is the GLOBAL weight_std,
# which is what the noise is scaled by:
#   gpt2_medium   weight_std=0.1083   extra=14.8% of D   D=354,501,632
#   smollm2_360m  weight_std=0.1717   extra=13.0% of D   D=361,758,720
#   pythia_410m   weight_std=0.0277   extra=25.4% of D   D=405,012,480
# gpt2_medium's global std is 3.9x pythia's, so at equal s it takes 3.9x more
# ABSOLUTE perturbation onto embeddings whose own scale is comparable. One global
# sigma mis-scales the noise for any submatrix whose own std differs from it, and
# gpt2_medium has the widest internal spread of the three.
#
# 3e-3 is still ~4.5 orders of magnitude above float32 epsilon, so the members
# separate cleanly in the Gram matrix (misstep 8).
#
# Prediction to check against: at k=N/2 the truncation residual on a TYPICAL member
# is ~s*sigma/sqrt(2) = 2.1e-3*sigma, i.e. between the 1e-3 and 3e-3 rows -- so
# expect roughly +0.2% for gpt2_medium and under +0.05% for the other two. On
# sample_idx 0 it is smaller by a further sqrt(N) (misstep 9), which is why the rank
# axis has to be read on a nonzero --sample_idx. At k=N-1 reconstruction is exact by
# the rank bound (cosine 1.0, dPPL ~0); anything else there is a bug, not a finding.

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
NOISE_SCALE="${NOISE_SCALE:-3e-3}"
# N_FAMILIES is already 6, so the six-family follow-up is purely this variable.
ARCH_LIST="${ARCH_LIST:-gpt2_medium smollm2_360m pythia_410m}"
K_FULL=$((N_SAMPLES - 1))
K_HALF=$((N_SAMPLES / 2))

echo "=============================================="
echo "Job      : $SLURM_JOB_ID on $SLURMD_NODENAME"
echo "GPU      : $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
echo "Run      : $RUN   N=$N_SAMPLES  s=$NOISE_SCALE"
echo "Archs    : $ARCH_LIST"
echo "Ranks    : k=$K_FULL (N-1)  and  k=$K_HALF (N/2)"
echo "=============================================="

COMMON="--mode full --run_name $RUN --artifact_dir $ROOT \
        --arch_list $ARCH_LIST \
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
