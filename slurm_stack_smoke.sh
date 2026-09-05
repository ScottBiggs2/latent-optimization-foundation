#!/bin/bash
# Tiny-mode end-to-end smoke test of the whole-stack pipeline.
#SBATCH --job-name=llm_vae_stack_smoke
#SBATCH --output=/scratch/biggs.s/llm_vae/logs/stack_smoke_%j.out
#SBATCH --error=/scratch/biggs.s/llm_vae/logs/stack_smoke_%j.err
#SBATCH --chdir=/home/biggs.s/llm_vae
#SBATCH --partition=gpu-short
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --time=00:40:00
set -e
mkdir -p /scratch/biggs.s/llm_vae/logs
source ~/miniconda/etc/profile.d/conda.sh
conda activate llm_vae
export HF_HOME=/scratch/biggs.s/hf_cache
export HF_DATASETS_CACHE=/scratch/biggs.s/hf_cache
export TRITON_CACHE_DIR=/scratch/biggs.s/triton_cache

RUN=stack_smoke
ROOT=/scratch/biggs.s/llm_vae
rm -rf "$ROOT/runs/$RUN"

COMMON="--mode tiny --run_name $RUN --artifact_dir $ROOT --n_samples 12 \
        --noise_scale 1e-2 --chunk_budget_mb 8 --no_wandb \
        --epochs 150 --patience 150 --warmup_epochs 30 --latent_dim 8 \
        --hidden_dim 64 --cond_dim 16 --batch_size 8"

echo "############ train k=11 (N-1) ############"
python -u train_stack.py $COMMON --k 11

echo; echo "############ train k=6 (N/2) — must reuse the Stage 2 fits ############"
python -u train_stack.py $COMMON --k 6

echo; echo "############ eval both ranks, all three arms ############"
# --bench synthetic exercises the MC plumbing without a download. Real benchmarks
# are REFUSED in tiny mode on purpose: a random-init model scores at chance, so an
# mmlu delta here would be noise dressed as a measurement.
python -u eval_stack.py --artifact_dir "$ROOT" --run_name "$RUN" --mode tiny \
    --k 11 6 --arms pca_only vae generate \
    --bench synthetic --bench_n_questions 8 \
    --eval_seq_len 64 --eval_n_sequences 4 --no_wandb

echo; echo "############ tiny mode must REFUSE a real benchmark ############"
if python -u eval_stack.py --artifact_dir "$ROOT" --run_name "$RUN" --mode tiny \
       --k 11 --arms pca_only --bench mmlu --no_wandb 2>&1 | tee /dev/stderr \
       | grep -q "Refusing --bench"; then
    echo "  OK: refused as designed"
else
    echo "  FAIL: tiny mode accepted a real benchmark"; exit 1
fi

echo; echo "SMOKE OK"
