#!/bin/bash
# The retrieval-difficulty sweep, flow-training stage, BOTH ranks.
#
# 5 target mixtures x 6 radii x 2 beta x 2 ranks = 120 flows, seconds each.
#
# Grid B bases only (tr92): the sealed interior+vertex holdout is already outside
# the SPAN there, so "held out" means held out of the basis and not merely out of
# the flow's training rows. The radius then pushes the nearest mixture the flow ever
# saw further from the target, on purpose -- because the sealed interior holdout is
# saturated (retrieval already matches the ceiling to inside the noise floor), and a
# sweep is what turns one dead number into a curve.
#
# Both k=N-1 and k=N/2, since conditioning measured INERT at k=N-1 (variance
# explained ~0%) and ~20% at k=N/2 and the whole point is to see whether that
# carries through to the generative arm.
#SBATCH --job-name=sweepflows
#SBATCH --account=p2026_0038_neu
#SBATCH --partition=b200-devel
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=/scratch/%u/logs/%x-%j.out
#SBATCH --error=/scratch/%u/logs/%x-%j.err

set -euo pipefail
CODE_DIR="${CODE_DIR:-/work/neu/p2026_0038_neu/$USER/llm-vae}"
source "$CODE_DIR/slurm/aicr_env.sh"
cd "$CODE_DIR"
export WANDB_PROJECT="${WANDB_PROJECT:-llmzoo-genverify}"

# The five held-out mixtures, read off zoo_meta.json's sealed holdout.
TARGETS=(
  "d050:0.173871 0.263829 0.224204 0.161555 0.176541"
  "d075:0.208987 0.250865 0.181794 0.270635 0.087719"
  "d043:0.128538 0.189349 0.309987 0.226306 0.145819"
  "d048:0.069343 0.208001 0.164042 0.352318 0.206295"
  "vmath:0.0 0.0 1.0 0.0 0.0"
)
# 0.1/0.2 are dropped: measured, they exclude 0-1 members on every interior target
# and leave the nearest-neighbour distance where r=0 left it. A wasted cell.
RADII="${RADII:-0 0.3 0.4 0.5 0.6 0.8}"

FCOMMON="--artifact_dir $ARTIFACT_DIR --space codes --cond_mode pi \
         --hidden_dims 64 128 64 --cond_dim 64 --time_embed_dim 64 \
         --cond_dropout 0.15 --source_std 1.0 --path_noise 0.0 \
         --epochs 500 --patience 500 --holdout_frac 0.15 \
         --lr 3e-4 --batch_size 64 --weight_decay 1e-4 --n_steps 50 --no_wandb"

for BETA in b015 b030; do
  for K in 91 46; do
    RUN="zoo_${BETA}_mini_tr92_k${K}"
    for T in "${TARGETS[@]}"; do
      NAME="${T%%:*}"; PI="${T#*:}"
      for R in $RADII; do
        RTAG=$(echo "$R" | tr -d '.')
        DIR="$ARTIFACT_DIR/runs/$RUN/sweep/${NAME}_r${RTAG}"
        if [ -f "$DIR/flow_meta.json" ]; then
          echo "SKIP $RUN $NAME r=$R"; continue
        fi
        echo; echo "==== $RUN | target $NAME | r=$R ===="
        # --run_name carries the cell so each gets its own directory; the flow dir
        # is derived from it, and EnsembleDataset's cache gate is keyed per run.
        python -u scripts/train_flow.py $FCOMMON \
            --run_name "$RUN" --k "$K" \
            --exclude_within_l1 "$R" --center_pi $PI \
            --force 2>&1 | grep -E "radius|pi<->code|rms at a fixed|sample rms|round-trip rel" \
          || true
        # train_flow writes flow_k<K>_codes_pi/; move it aside so the next radius
        # does not collide, and so every cell survives for the report.
        mkdir -p "$(dirname "$DIR")"
        rm -rf "$DIR"
        mv "$ARTIFACT_DIR/runs/$RUN/flow_k${K}_codes_pi" "$DIR"
      done
    done
  done
done

echo; echo "############ SWEEP GEOMETRY + DISPERSION ############"
python - <<'PY'
import json, os, glob
art = os.environ["ARTIFACT_DIR"]
print("%-24s %4s %-6s %6s %6s %7s %7s %8s" %
      ("run","k","target","r","kept","nearest","at_pi","piCorr"))
for p in sorted(glob.glob(f"{art}/runs/zoo_b0*_tr92_k*/sweep/*/flow_meta.json")):
    m = json.load(open(p)); t = m["train"]
    parts = p.split("/"); run = parts[-4]; cell = parts[-2]
    name, r = cell.rsplit("_r", 1)
    print("%-24s %4d %-6s %6s %6d %7s %7.3f %8.3f" % (
        run, m["dim"], name, t.get("exclude_within_l1"),
        t.get("n_rows", 0) + t.get("n_holdout", 0),
        "-", t.get("code_rms_ratio_at_fixed_pi") or float("nan"),
        t.get("pi_code_distance_corr") or float("nan")))
PY
