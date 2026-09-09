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

# RESOURCE SHAPE, MEASURED rather than guessed (2026-09-08, sacct on 12 probe
# branches). The committed request had been --cpus-per-task=16 --mem=200G:
#
#   MaxRSS   16.1-17.8 GB   -> 200G was 12x over
#   AveCPU   ~0.87 cores    -> the data path tokenizes one document at a time,
#                              synchronously, in the training thread, so 16 CPUs
#                              was ~16x over
#
# This is not just tidiness. Slurm must find the WHOLE request free on one node,
# so a 200 GB reservation cannot be placed on a node with a free GPU and 150 GB
# free -- backfill lost for nothing (aicr-cluster skill, Rule 1). And 32
# concurrent tasks reserved 6.4 TB against a 2.3 TB node, on a fairshare account
# shared with two collaborators.
#
# 96G is 5.4x Mini's observed peak and still covers Medium, whose extra cost is
# bounded: the model (1.42 GB), Adam's two moments, and the flattened w (1.42 GB)
# -- the 5 x 10k-document shuffle buffers dominate and do not scale with the
# model. Raise it if a Medium branch ever OOMs; do not raise it pre-emptively.
#SBATCH --job-name=zoo_trunk
#SBATCH --account=p2026_0038_neu
#SBATCH --partition=b200-batch
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
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
