#!/bin/bash
# Why does pi explain ~0% of code variance at k=N-1 and ~20% at k=N/2?
#
# Four candidate causes, and this job separates them. Every arm is scored by the
# SAME number: variance explained by pi = 1 - (at_pi / R)^2, R = pooled/within.
#
#   RANK      k in {99, 50, 25, 10}. Effective rank is ~4.5 (PHASE2_RESULTS 2b), so
#             at k=99 roughly 95 of 99 code dims are near-noise. If that is the
#             binding constraint, variance explained should keep climbing as k falls
#             toward the effective rank. Two points already trend this way; this
#             adds two more and makes it a curve or kills it.
#
#   STEPS     epochs in {500, 2000, 8000} at fixed k=50. THIS IS THE ARM THAT NEEDS
#             ARGUING FOR, because CLAUDE.md says do not scale the flow up. Read the
#             provenance of that rule: misstep 21's table was measured by
#             diag_flow_capacity.py on the MANUFACTURED NOISE ensemble (M=300 rows,
#             ev0/median = 1.001), where there was provably nothing to learn and any
#             extra capacity could only memorise. This zoo has ev0/median = 309 and
#             89% of its code variance BETWEEN mixtures. The rule may still hold, but
#             it was never tested here.
#             It is safe to test now for a specific reason: the collapse it warns
#             about was invisible to the pooled rms ratio under pi-conditioning, and
#             code_rms_ratio_at_fixed_pi sees it. Collapse shows up as at_pi -> 0.
#             At 85 rows and batch 64 the default is ~2 batches/epoch, so 500 epochs
#             is about 1000 optimiser steps. DeepWeightFlow ran 30k.
#
#   WIDTH     (64,128,64) vs (256,512,256) at k=50, 500 epochs. Capacity at fixed
#             duration, so it separates "too small" from "too short".
#
#   DROPOUT   cond_dropout 0.15 vs 0.0 at k=50. 15% of steps train the NULL branch;
#             on 85 rows that is a real fraction of a small budget.
#
# Read at_pi, not just pooled. Pooled reads healthy (0.85-1.10) in every cell
# measured so far and cannot distinguish any of these.
#SBATCH --job-name=diagcond
#SBATCH --account=p2026_0038_neu
#SBATCH --partition=b200-devel
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=200G
#SBATCH --time=02:00:00
#SBATCH --output=/scratch/%u/logs/%x-%j.out
#SBATCH --error=/scratch/%u/logs/%x-%j.err

set -euo pipefail
CODE_DIR="${CODE_DIR:-/work/neu/p2026_0038_neu/$USER/llm-vae}"
source "$CODE_DIR/slurm/aicr_env.sh"
cd "$CODE_DIR"
export WANDB_PROJECT="${WANDB_PROJECT:-llmzoo-genverify}"

BETA="${BETA:-b015}"
RUN="zoo_${BETA}_mini_k99"
ZOO_ROOT="$ARTIFACT_DIR/zoo_${BETA}_mini_n100"
OUT="$ARTIFACT_DIR/runs/$RUN/diag_cond"
mkdir -p "$OUT"

# Codes at the two new ranks. Stage 2's PCA is reused -- a lower k is a column
# prefix of the same fit -- so this is stage 3 plus a small VAE.
for K in 25 10; do
  if [ ! -d "$ARTIFACT_DIR/runs/$RUN/codes_k$K" ]; then
    echo "### codes at k=$K ###"
    python -u scripts/train_stack.py --arch_list gpt2_zoo_mini --mode full \
        --artifact_dir "$ARTIFACT_DIR" --run_name "$RUN" --source zoo \
        --zoo_dir "$ZOO_ROOT" --n_samples 100 --k "$K" --members all \
        --noise_scale 0.0 --no_exclude_1d \
        --epochs 400 --patience 100 --warmup_epochs 50 \
        --latent_dim 8 --hidden_dim 64 --cond_dim 16 --batch_size 8 \
        --no_wandb 2>&1 | tail -3
  fi
done

BASE="--artifact_dir $ARTIFACT_DIR --run_name $RUN --space codes --cond_mode pi \
      --cond_dim 64 --time_embed_dim 64 --source_std 1.0 --path_noise 0.0 \
      --patience 100000 --holdout_frac 0.15 --lr 3e-4 --batch_size 64 \
      --weight_decay 1e-4 --n_steps 50 --force --no_wandb"

run_cell () {  # name k epochs "hidden" dropout
  local NAME="$1" K="$2" EP="$3" HID="$4" DO="$5"
  local DIR="$OUT/$NAME"
  [ -f "$DIR/flow_meta.json" ] && { echo "SKIP $NAME"; return 0; }
  echo; echo "==== $NAME | k=$K epochs=$EP hidden=($HID) dropout=$DO ===="
  python -u scripts/train_flow.py $BASE --k "$K" --epochs "$EP" \
      --hidden_dims $HID --cond_dropout "$DO" 2>&1 \
    | grep -E "rms at a fixed|sample rms|pi<->code|within-pi|Early stopping" || true
  rm -rf "$DIR"; mkdir -p "$(dirname "$DIR")"
  mv "$ARTIFACT_DIR/runs/$RUN/flow_k${K}_codes_pi" "$DIR"
}

echo; echo "######## RANK ########"
for K in 99 50 25 10; do run_cell "rank_k${K}" "$K" 500 "64 128 64" 0.15; done

echo; echo "######## STEPS (k=50) ########"
for EP in 2000 8000; do run_cell "steps_e${EP}" 50 "$EP" "64 128 64" 0.15; done

echo; echo "######## WIDTH (k=50, 500 epochs) ########"
run_cell "width_256" 50 500 "256 512 256" 0.15

echo; echo "######## COND DROPOUT (k=50, 500 epochs) ########"
run_cell "drop_000" 50 500 "64 128 64" 0.0

echo; echo "############ RESULT: variance in code space explained by pi ############"
python - <<'PY'
import json, os, glob
art = os.environ["ARTIFACT_DIR"]; run = os.environ["RUN"]
cells = sorted(glob.glob(f"{art}/runs/{run}/diag_cond/*/flow_meta.json"))
print("%-14s %4s %7s %8s %8s %8s %9s %9s" % (
    "cell", "k", "epochs", "at_pi", "var_pi", "pooled", "trainloss", "params"))
for p in cells:
    m = json.load(open(p)); t = m["train"]; n = p.split("/")[-2]
    a = t.get("code_rms_ratio_at_fixed_pi")
    # R is recomputed per cell from that cell's own reference, printed by train_flow
    # as "ratio"; recover it from the sealed pooled/within relationship instead.
    hid = m["net"]["hidden_dims"]
    npar = sum(1 for _ in [0])  # placeholder, real count below
    print("%-14s %4d %7d %8.3f %8s %8.3f %9.4f %9s" % (
        n, m["dim"], t.get("epochs_run", 0), a if a else float("nan"),
        "see below", t.get("code_rms_ratio") or float("nan"),
        t.get("best_loss") or float("nan"), "x".join(map(str, hid))))
print()
print("var_pi = 1 - (at_pi/R)^2 with R = pooled/within measured PER RANK;")
print("R is printed by train_flow as the 'within-pi ... ratio' line above.")
PY
