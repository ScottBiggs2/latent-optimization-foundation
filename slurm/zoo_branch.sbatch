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

#SBATCH --job-name=zoo_branch
#SBATCH --account=p2026_0038_neu
#SBATCH --partition=b200-batch
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=200G
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
# Parent of the arch dir -- the same value `train_stack.py --zoo_dir` takes.
ZOO_ROOT="${ZOO_ROOT:-$ARTIFACT_DIR/zoo}"
ZOO_DIR="$ZOO_ROOT/$ARCH"
IDX="${SLURM_ARRAY_TASK_ID:?this script must be submitted with --array}"

echo "=== zoo branch: arch=$ARCH beta=$BETA member=$IDX -> $ZOO_DIR ==="
python -u scripts/train_zoo.py --mode branch --arch "$ARCH" --beta "$BETA" \
       --n_members "$N_MEMBERS" --member_idx "$IDX" --zoo_dir "$ZOO_DIR" $EXTRA
echo "=== done ==="
