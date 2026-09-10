#!/bin/bash
# Follow-up to job 749426, which showed the flow was UNDERFIT, not under-expressive:
# at k=50, recovery of the available conditional structure went 27.6% (500 epochs)
# -> 45.2% (2000) -> 90.2% (8000), with pooled dispersion RISING to 1.16 rather
# than collapsing.
#
# One question left before the retrieval sweep is worth re-running: is k=N-1's
# failure a RANK problem or just a harder optimisation problem? Everything measured
# at k=99 so far used 500 epochs. If k=99 also recovers at 8000, rank is a
# convergence-rate axis and the sweep should cover both ranks. If it stays flat,
# rank is structural and the sweep only needs k=N/2.
#
# Also: capacity AND duration together, to find the config the sweep should use.
#SBATCH --job-name=diaglong
#SBATCH --account=p2026_0038_neu
#SBATCH --partition=b200-devel
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=02:00:00
#SBATCH --output=/scratch/%u/logs/%x-%j.out
#SBATCH --error=/scratch/%u/logs/%x-%j.err

set -euo pipefail
CODE_DIR="${CODE_DIR:-/work/neu/p2026_0038_neu/$USER/llm-vae}"
source "$CODE_DIR/slurm/aicr_env.sh"
cd "$CODE_DIR"
export WANDB_PROJECT="${WANDB_PROJECT:-llmzoo-genverify}"
export RUN="zoo_b015_mini_k99"
OUT="$ARTIFACT_DIR/runs/$RUN/diag_cond"

BASE="--artifact_dir $ARTIFACT_DIR --run_name $RUN --space codes --cond_mode pi \
      --cond_dim 64 --time_embed_dim 64 --source_std 1.0 --path_noise 0.0 \
      --patience 100000 --holdout_frac 0.15 --lr 3e-4 --batch_size 64 \
      --weight_decay 1e-4 --n_steps 50 --force --no_wandb"

run_cell () {
  local NAME="$1" K="$2" EP="$3" HID="$4"
  local DIR="$OUT/$NAME"
  [ -f "$DIR/flow_meta.json" ] && { echo "SKIP $NAME"; return 0; }
  echo; echo "==== $NAME | k=$K epochs=$EP hidden=($HID) dropout=0.15 ===="
  python -u scripts/train_flow.py $BASE --k "$K" --epochs "$EP" \
      --hidden_dims $HID --cond_dropout 0.15 2>&1 \
    | grep -E "rms at a fixed|sample rms|within-pi|pi<->code" || true
  rm -rf "$DIR"; mkdir -p "$(dirname "$DIR")"
  mv "$ARTIFACT_DIR/runs/$RUN/flow_k${K}_codes_pi" "$DIR"
}

# Does k=N-1 recover if it is simply trained long enough?
run_cell "long_k99_e8000"  99 8000 "64 128 64"
run_cell "long_k99_e8000w" 99 8000 "256 512 256"
# The candidate config for the sweep redo.
run_cell "long_k50_e8000w" 50 8000 "256 512 256"
# Sanity: does even more duration start to collapse it on REAL data?
run_cell "long_k50_e20000" 50 20000 "64 128 64"
