#!/bin/bash
# Tiny-mode end-to-end smoke test of the whole-stack pipeline. The known-green
# ~1-minute bisect point: if this fails, the break is in the last thing you touched.
#
#   sbatch slurm/stack_smoke.sbatch

#SBATCH --job-name=stack_smoke
#SBATCH --account=p2026_0038_neu
#SBATCH --partition=b200-devel
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:30:00
#SBATCH --output=/scratch/%u/logs/%x-%j.out
#SBATCH --error=/scratch/%u/logs/%x-%j.err

# b200-devel, not b200-batch: 4 h wall and a 2-GPU cap, and it is usually idle.
# A smoke test should never queue behind a training job.

set -euo pipefail
# sbatch does not set $0 to the script path, so point at the header explicitly.
CODE_DIR="${CODE_DIR:-/work/neu/p2026_0038_neu/$USER/llm-vae}"
source "$CODE_DIR/slurm/aicr_env.sh"

RUN=stack_smoke
rm -rf "$ARTIFACT_DIR/runs/$RUN"

COMMON="--mode tiny --run_name $RUN --artifact_dir $ARTIFACT_DIR --n_samples 12 \
        --noise_scale 1e-2 --chunk_budget_mb 8 --no_wandb \
        --epochs 150 --patience 150 --warmup_epochs 30 --latent_dim 8 \
        --hidden_dim 64 --cond_dim 16 --batch_size 8"

echo "############ train k=11 (N-1) ############"
python -u scripts/train_stack.py $COMMON --k 11

echo; echo "############ train k=6 (N/2) — must reuse the Stage 2 fits ############"
python -u scripts/train_stack.py $COMMON --k 6

echo; echo "############ eval both ranks, all three arms ############"
# --bench synthetic exercises the MC plumbing without a download. Real benchmarks
# are REFUSED in tiny mode on purpose: a random-init model scores at chance, so an
# mmlu delta here would be noise dressed as a measurement.
python -u scripts/eval_stack.py --artifact_dir "$ARTIFACT_DIR" --run_name "$RUN" \
    --mode tiny --k 11 6 --arms pca_only vae generate \
    --bench synthetic --bench_n_questions 8 \
    --eval_seq_len 64 --eval_n_sequences 4 --no_wandb

echo; echo "############ tiny mode must REFUSE a real benchmark ############"
# Capture first, then grep -- do NOT pipe the python call into grep.
#
# eval_stack refuses via `raise SystemExit(msg)`, which exits 1. Under the
# `set -o pipefail` above, a pipeline's status is the last NON-ZERO status, so
# `python ... | grep -q "Refusing"` returned 1 even when grep matched, and this
# check reported FAIL for the exact behaviour it was asserting. The refusal has
# been working the whole time; the assertion was inverted by pipefail.
#
# `|| true` because a non-zero exit is the EXPECTED outcome here.
refusal=$(python -u scripts/eval_stack.py --artifact_dir "$ARTIFACT_DIR" \
              --run_name "$RUN" --mode tiny --k 11 --arms pca_only \
              --bench mmlu --no_wandb 2>&1 || true)
echo "$refusal"
if grep -q "Refusing --bench" <<< "$refusal"; then
    echo "  OK: refused as designed"
else
    echo "  FAIL: tiny mode accepted a real benchmark"; exit 1
fi

echo; echo "SMOKE OK"
