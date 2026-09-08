#!/bin/bash
# Full whole-stack run: per-family Gram PCA -> StackVAE -> evaluation, at both
# rank arms. One PCA sample is one complete decoder stack (DeepWeightFlow framing).
#
#   sbatch slurm/stack_run.sbatch
#   N_SAMPLES=143 NOISE_SCALE=3e-3 RUN=perfam-s3e3 sbatch slurm/stack_run.sbatch
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

#SBATCH --job-name=stack_run
#SBATCH --account=p2026_0038_neu
#SBATCH --partition=b200-batch
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=200G
#SBATCH --time=01:30:00
#SBATCH --output=/scratch/%u/logs/%x-%j.out
#SBATCH --error=/scratch/%u/logs/%x-%j.err

# --gres=gpu:1 is deliberate, not timid. AICR shares nodes between jobs, so a
# 1-GPU request backfills into any partially-used node and starts immediately,
# where an 8-GPU request waits for a whole node to drain. This stage measured
# ~19 min / 20 GB on ONE V100; it has never needed a second GPU.
#
# --time=01:30 is likewise honest: backfill only slots a job into a gap it can
# prove the job finishes inside. Asking 6h for a 20min stage forfeits every gap.

set -euo pipefail
# sbatch does not set $0 to the script path, so point at the header explicitly.
CODE_DIR="${CODE_DIR:-/work/neu/p2026_0038_neu/$USER/llm-vae}"
source "$CODE_DIR/slurm/aicr_env.sh"

RUN="${RUN:-perfam}"
N_SAMPLES="${N_SAMPLES:-100}"
NOISE_SCALE="${NOISE_SCALE:-3e-3}"
# N_FAMILIES is already 6, so the six-family follow-up is purely this variable.
ARCH_LIST="${ARCH_LIST:-gpt2_medium smollm2_360m pythia_410m}"
K_FULL=$((N_SAMPLES - 1))
K_HALF=$((N_SAMPLES / 2))

echo "=============================================="
echo "Run      : $RUN   N=$N_SAMPLES  s=$NOISE_SCALE"
echo "Archs    : $ARCH_LIST"
echo "Ranks    : k=$K_FULL (N-1)  and  k=$K_HALF (N/2)"
echo "=============================================="

COMMON="--mode full --run_name $RUN --artifact_dir $ARTIFACT_DIR \
        --arch_list $ARCH_LIST \
        --n_samples $N_SAMPLES --noise_scale $NOISE_SCALE \
        --exclude_1d --chunk_budget_mb 512 \
        --latent_dim 32 --hidden_dim 256 --cond_dim 64 \
        --epochs 3000 --patience 300 --warmup_epochs 150 \
        --beta 1.0 --free_bits 0.05 --cond_dropout 0.15 \
        --code_noise_std 0.02 --lr 3e-4 --batch_size 64"

echo; echo "######## Stage A: PCA fits + VAE at k=$K_FULL ########"
python -u scripts/train_stack.py $COMMON --k "$K_FULL"

echo; echo "######## Stage B: VAE at k=$K_HALF (reuses the PCA fits) ########"
python -u scripts/train_stack.py $COMMON --k "$K_HALF"

echo; echo "######## Stage C: evaluation, both ranks, all arms ########"
python -u scripts/eval_stack.py --artifact_dir "$ARTIFACT_DIR" --run_name "$RUN" \
    --mode full --k "$K_FULL" "$K_HALF" --arms pca_only vae generate \
    --eval_seq_len 1024 --eval_n_sequences 64

echo; echo "Done — results in $ARTIFACT_DIR/runs/$RUN/results/"
