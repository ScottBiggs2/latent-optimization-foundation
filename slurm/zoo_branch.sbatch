#!/bin/bash
# Branch the zoo off a trained trunk -- one array task per member.
#
#   ZOO_ROOT=$ARTIFACT_DIR/zoo_b030 ARCH=gpt2_zoo_mini BETA=0.30 \
#       sbatch --array=0-99 slurm/zoo_branch.sbatch
#   ZOO_ROOT=$ARTIFACT_DIR/zoo_b030 ARCH=gpt2_zoo_mini BETA=0.30 N_MEMBERS=12 \
#       sbatch --array=0-11 slurm/zoo_branch.sbatch          # beta calibration
#
# GIVE EVERY (arch, beta) ITS OWN ZOO_ROOT, and note WHY this is not cosmetic.
# The resume check below is `w_<i>.npy exists -> exit 0`, and it runs BEFORE the
# trunk's beta is ever read. So two betas sharing a directory used to hand back
# the FIRST beta's weights under the second beta's label -- silently, exit 0, on
# every member, invalidating a whole calibration with no error anywhere.
# train_zoo.py now also refuses a directory whose zoo_meta.json disagrees on
# beta/arch/n_members, so both layers have to be defeated to repeat that.
#
# ONE GPU PER TASK, and that is the whole point (RESEARCH_PLAN §4.6). Branches are
# independent, so Slurm backfills them across partially-free nodes and they start
# immediately. The 32-GPU account ceiling then runs 100 members in ~4 waves.
# Submitting this as a single --gres=gpu:8 job would be slower AND queue longer.
#
# Re-running is safe and cheap: a member whose w_<i>.npy already exists exits
# immediately, so a partially-failed array is resubmitted verbatim.

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
#SBATCH --job-name=zoo_branch
#SBATCH --account=p2026_0038_neu
#SBATCH --partition=b200-batch
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=01:00:00
#SBATCH --output=/scratch/%u/logs/%x-%A_%a.out
#SBATCH --error=/scratch/%u/logs/%x-%A_%a.err

set -euo pipefail
CODE_DIR="${CODE_DIR:-/work/neu/p2026_0038_neu/$USER/llm-vae}"
source "$CODE_DIR/slurm/aicr_env.sh"

ARCH="${ARCH:-gpt2_zoo_mini}"
BETA="${BETA:-0.30}"
N_MEMBERS="${N_MEMBERS:-100}"
EXTRA="${EXTRA:-}"

# Mid-run checkpointing. OFF unless CKPT_EVERY is set, and with it off the python
# command line below is byte-for-byte what it was before this existed.
#
# Expand as ${CKPT_ARGS[@]+"${CKPT_ARGS[@]}"}, NOT "${CKPT_ARGS[@]}". This script
# runs `set -euo pipefail`, and on bash < 4.4 expanding an EMPTY array under `set
# -u` is a fatal "unbound variable" -- which is the exact class of bug that killed
# all three trunks in one second on 2026-09-08 (see aicr_env.sh's note on
# ${#HF_TOKEN}). Test with `bash -euo pipefail -c`, never `srun bash -c`: that
# shell has no `set -u` and will not reproduce it.
CKPT_EVERY="${CKPT_EVERY:-0}"
CKPT_DIR="${CKPT_DIR:-$S/zoo_ckpt}"
CKPT_ARGS=()
if [ "$CKPT_EVERY" -gt 0 ]; then
    CKPT_ARGS=(--ckpt_every "$CKPT_EVERY" --ckpt_dir "$CKPT_DIR")
    echo "checkpointing: every $CKPT_EVERY steps -> $CKPT_DIR"
fi
# Parent of the arch dir -- the same value `train_stack.py --zoo_dir` takes.
ZOO_ROOT="${ZOO_ROOT:-$ARTIFACT_DIR/zoo}"
ZOO_DIR="$ZOO_ROOT/$ARCH"
IDX="${SLURM_ARRAY_TASK_ID:?this script must be submitted with --array}"

echo "=== zoo branch: arch=$ARCH beta=$BETA member=$IDX -> $ZOO_DIR ==="
python -u scripts/train_zoo.py --mode branch --arch "$ARCH" --beta "$BETA" \
       --n_members "$N_MEMBERS" --member_idx "$IDX" --zoo_dir "$ZOO_DIR" \
       ${CKPT_ARGS[@]+"${CKPT_ARGS[@]}"} $EXTRA
echo "=== done ==="
