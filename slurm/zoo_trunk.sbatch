#!/bin/bash
# Train ONE shared trunk for a zoo scale (RESEARCH_PLAN §4.1).
# Every branch descends from this checkpoint, so it runs once per (arch, beta).
#
#   ZOO_ROOT=$ARTIFACT_DIR/zoo_b030 ARCH=gpt2_zoo_mini BETA=0.30 \
#       sbatch slurm/zoo_trunk.sbatch
#
# GIVE EVERY (arch, beta) ITS OWN ZOO_ROOT. The trunk's token count is
# (1-beta)*F, so a beta=0.15 trunk is NOT a beta=0.30 trunk -- but trunk.pt is
# gated on os.path.exists, so a second beta pointed at the same directory reports
# "trunk already at ...; nothing to do" and then every branch fails on the beta
# guard. train_zoo.py also refuses a directory whose zoo_meta.json disagrees.
#
# Cost: (1-beta)*F. At Mini that is ~0.09 GPU-hr, at Small ~0.53, at Medium ~4.35
# -- but those assume 30% MFU, which nothing had measured. --time below is set for
# Mini with headroom for a data-bound loop (the tokenizer runs single-threaded in
# the training thread). RAISE IT for the larger scales: Small needs ~4x and Medium
# ~50x, and a killed trunk loses the whole run.

#SBATCH --job-name=zoo_trunk
#SBATCH --account=p2026_0038_neu
#SBATCH --partition=b200-batch
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=200G
#SBATCH --time=02:00:00
#SBATCH --output=/scratch/%u/logs/%x-%j.out
#SBATCH --error=/scratch/%u/logs/%x-%j.err

# One GPU, deliberately. AICR nodes are shared and a 1-GPU job backfills into a
# partially-free node immediately, where --gres=gpu:8 waits for one to drain.

set -euo pipefail
CODE_DIR="${CODE_DIR:-/work/neu/p2026_0038_neu/$USER/llm-vae}"
source "$CODE_DIR/slurm/aicr_env.sh"

ARCH="${ARCH:-gpt2_zoo_mini}"
BETA="${BETA:-0.30}"
N_MEMBERS="${N_MEMBERS:-100}"
EXTRA="${EXTRA:-}"
# ZOO_ROOT is the PARENT of the arch dir, which is also exactly what
# EnsembleDataset(zoo_dir=) wants: it reads <zoo_dir>/<arch>/w_<i>.npy. So the
# same value passes straight to `train_stack.py --zoo_dir` with no second path
# convention to get wrong.
ZOO_ROOT="${ZOO_ROOT:-$ARTIFACT_DIR/zoo}"
ZOO_DIR="$ZOO_ROOT/$ARCH"

echo "=== zoo trunk: arch=$ARCH beta=$BETA -> $ZOO_DIR ==="
python -u scripts/train_zoo.py --mode plan  --arch "$ARCH" --beta "$BETA" \
       --n_members "$N_MEMBERS" --zoo_dir "$ZOO_DIR" $EXTRA
python -u scripts/train_zoo.py --mode trunk --arch "$ARCH" --beta "$BETA" \
       --n_members "$N_MEMBERS" --zoo_dir "$ZOO_DIR" $EXTRA
echo "=== done ==="
