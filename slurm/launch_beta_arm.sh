#!/bin/bash
# Launch ONE beta arm of the RESEARCH_PLAN §4.2 calibration as a dependency chain:
#
#     trunk  ->  branch array  ->  { §4.3 gate , k=11 spectrum }
#
#   bash slurm/launch_beta_arm.sh 0.30
#   bash slurm/launch_beta_arm.sh 0.15 12 gpt2_zoo_mini
#   TRUNK_TIME=04:00:00 BRANCH_TIME=02:00:00 bash slurm/launch_beta_arm.sh 0.60
#
# WHY THIS IS A SCRIPT AND NOT THREE TYPED COMMANDS
#
# Every beta needs its own ZOO_ROOT, and getting that wrong is silent rather than
# loud. The branch resume check is `w_<i>.npy exists -> exit 0` and it runs before
# the trunk's beta is ever read, so two betas sharing a directory hand back the
# first beta's weights under the second beta's label, on every member, with
# exit 0. Deriving ZOO_ROOT from BETA in one place removes the chance to mistype
# it in three.
#
# afterok, not afterany: a failed trunk must not spawn 12 doomed branches, and the
# gate must not score a half-built zoo. A gate that runs and exits 1 is a RESULT
# (raise beta); a gate that never runs is a build failure. Keep them distinguishable.
#
# Re-running is safe and cheap. A member whose w_<i>.npy exists exits immediately,
# so resubmit this verbatim after a partial array.

set -euo pipefail

BETA="${1:?usage: launch_beta_arm.sh <beta> [n_members] [arch]}"
N_MEMBERS="${2:-12}"
ARCH="${3:-gpt2_zoo_mini}"

PROJ=/work/neu/p2026_0038_neu
ART="${ARTIFACT_DIR:-$PROJ/$USER/llm_vae}"
CODE_DIR="${CODE_DIR:-$PROJ/$USER/llm-vae}"

# 0.30 -> b030, so the directory name is unambiguous in a listing and sorts.
TAG="b$(printf '%03d' "$(python3 -c "print(round(float('$BETA')*100))")")"
ZOO_ROOT="$ART/zoo_$TAG"

# --time is sized for Mini. Small needs ~4x and Medium ~50x; override with the
# env vars rather than editing the sbatch, so the committed default stays honest
# about which scale it was measured for.
TRUNK_TIME="${TRUNK_TIME:-02:00:00}"
BRANCH_TIME="${BRANCH_TIME:-01:00:00}"

# Concurrent array tasks. The QOS ceiling is 32 GPUs per user on b200-batch, and
# three arms running at once would request 3x this. Keep the product at or under
# 32 so nothing sits in QOSMaxGRESPerUser, and leave a little headroom on a
# fairshare account shared with collaborators.
THROTTLE="${THROTTLE:-10}"

# Where the trunk and branches run. b200-batch is the intended pool and is ~1.56x
# faster (234 vs ~150 ktok/s measured at Mini). rtx-batch is the fallback: a
# SEPARATE 32-GPU QOS pool, so it neither consumes nor waits on the b200 ceiling.
# Override when b200-batch is congested or, as on 2026-09-08, when it is refusing
# to launch anything ("launch failed requeued held" on even a trivial job).
# Note --peak_tflops defaults to a B200; on rtx-batch the sealed MFU percentage
# needs a different denominator, though the ktok/s stands either way.
PARTITION="${PARTITION:-b200-batch}"

LAST=$((N_MEMBERS - 1))

echo "=================================================================="
echo " beta arm      : BETA=$BETA  ARCH=$ARCH  N=$N_MEMBERS"
echo " zoo root      : $ZOO_ROOT"
echo " partition     : $PARTITION"
echo " walltimes     : trunk=$TRUNK_TIME  branch=$BRANCH_TIME"
echo "=================================================================="

cd "$CODE_DIR"

t=$(ZOO_ROOT="$ZOO_ROOT" ARCH="$ARCH" BETA="$BETA" N_MEMBERS="$N_MEMBERS" \
    sbatch --parsable --partition="$PARTITION" --time="$TRUNK_TIME" \
           --job-name="trunk_$TAG" \
           slurm/zoo_trunk.sbatch)
echo "trunk      : $t"

# One GPU per array task, always. Branches are independent, AICR nodes are shared,
# and a 1-GPU job backfills immediately where --gres=gpu:8 waits for a node to
# drain. %N throttles concurrency to stay under the 32-GPU account ceiling and to
# leave room for collaborators on the shared fairshare.
b=$(ZOO_ROOT="$ZOO_ROOT" ARCH="$ARCH" BETA="$BETA" N_MEMBERS="$N_MEMBERS" \
    sbatch --parsable --partition="$PARTITION" --time="$BRANCH_TIME" \
           --job-name="branch_$TAG" \
           --array="0-$LAST%$THROTTLE" --dependency="afterok:$t" \
           slurm/zoo_branch.sbatch)
echo "branches   : $b  (array 0-$LAST%$THROTTLE)"

# The §4.3 gate, on rtx-batch: a SEPARATE 32-GPU QOS pool, so it does not consume
# the b200 ceiling and runs alongside the next beta's training.
g=$(ZOO_ROOT="$ZOO_ROOT" ARCH="$ARCH" \
    sbatch --parsable --job-name="gate_$TAG" --dependency="afterok:$b" \
           slurm/eval_domains.sbatch)
echo "gate       : $g  (rtx-batch, exit 1 = gate fired)"

# The k=N-1 spectrum. Four flags and every one is load-bearing:
#   --source zoo / --zoo_dir   point at the zoo at all (parent of the arch dir)
#   --noise_scale 0.0          EnsembleDataset refuses a zoo with augmentation noise
#   --no_exclude_1d            train_zoo writes exclude_1d=False (§2.2); the
#                              default here is True and the cache gate refuses a mismatch
K=$((N_MEMBERS - 1))
s=$(sbatch --parsable --job-name="spec_$TAG" --dependency="afterok:$b" \
           --account=p2026_0038_neu --partition=rtx-batch --nodes=1 --gres=gpu:1 \
           --cpus-per-task=8 --mem=200G --time=01:00:00 \
           --output="/scratch/$USER/logs/spec_$TAG-%j.out" \
           --error="/scratch/$USER/logs/spec_$TAG-%j.err" \
           --wrap "set -euo pipefail
                   source $CODE_DIR/slurm/aicr_env.sh
                   python -u scripts/train_stack.py \
                     --arch_list $ARCH --mode full \
                     --run_name zoo_${TAG}_k$K --artifact_dir \$ARTIFACT_DIR \
                     --source zoo --zoo_dir '$ZOO_ROOT' \
                     --n_samples $N_MEMBERS --k $K \
                     --noise_scale 0.0 --no_exclude_1d \
                     --epochs 400 --patience 100 --warmup_epochs 50 \
                     --latent_dim 8 --hidden_dim 64 --cond_dim 16 --batch_size 8")
echo "spectrum   : $s  (rtx-batch, k=$K)"

echo
echo "watch:  squeue -u \$USER"
echo "gate  : $ZOO_ROOT/$ARCH/domain_separation.json"
echo "spec  : \$ARTIFACT_DIR/runs/zoo_${TAG}_k$K/pipeline_summary_k$K.json"
echo "        -> per_arch.$ARCH.spectrum_ev0_over_median"
echo "        -> per_arch.$ARCH.spectrum_effective_rank_ratio"
echo "        NOT spectrum_ev0_over_evlast (misstep 15b: reads 12.09 on data"
echo "        whose true ratio is 1.01)"
