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

# ZOO_ROOT is keyed on (beta, ARCH, N), not on beta alone, and every part of that
# is load-bearing:
#
#   * ARCH, because the three scales are separate zoos that happen to share a
#     beta. They land in different <root>/<arch> subdirs either way, so this is
#     belt-and-braces -- but the RUN NAME below is not, see there.
#   * N_MEMBERS, because train_zoo.py refuses a directory whose zoo_meta.json
#     disagrees on n_members (:621). The Phase 1 calibration wrote
#     zoo_b030/gpt2_zoo_mini with n_members=12, so a Phase 2 N=100 run pointed at
#     the old `$ART/zoo_$TAG` default would have had EVERY branch exit 2 while
#     the trunk step reported "nothing to do". Loud, but 100 wasted submissions.
#
# Overridable, because the Phase 1 arms live at the OLD `$ART/zoo_b<NNN>` paths:
# to re-run one of those, pass ZOO_ROOT explicitly rather than relying on this.
ZOO_ROOT="${ZOO_ROOT:-$ART/zoo_${TAG}_${ARCH#gpt2_zoo_}_n${N_MEMBERS}}"

# --time is sized for Mini. Small needs ~4x and Medium ~50x; override with the
# env vars rather than editing the sbatch, so the committed default stays honest
# about which scale it was measured for.
TRUNK_TIME="${TRUNK_TIME:-02:00:00}"
BRANCH_TIME="${BRANCH_TIME:-01:00:00}"
# The spectrum fit streams every member TWICE (mean pass, then Gram pass) off
# NFS. The committed 1 h was sized for 12 x 206 MB = 2.5 GB; at N=100 Medium it
# is 100 x 1.42 GB = 142 GB per pass.
SPEC_TIME="${SPEC_TIME:-02:00:00}"
# The gate's own sbatch header hardcodes 01:30:00 and nothing could override it, so
# a scale whose 100 members take longer to score had no lever at all. It rebuilds
# each member in place and runs 5 domains x 64 blocks of forwards, so it grows with
# both D and N.
GATE_TIME="${GATE_TIME:-01:30:00}"

# Mid-run checkpointing for the trunk and the branches. 0 = OFF, which is the
# default and is what Mini and Small run. Turn it on for spans long enough that
# losing one hurts: Medium's trunk is ~16.5 h against a 24 h MaxTime.
CKPT_EVERY="${CKPT_EVERY:-0}"

# W&B project. One per WORKSTREAM, which is what aicr_env.sh:90-93 describes and
# what no zoo sbatch ever actually set -- so every zoo run landed in the historical
# `llm-vae` alongside the flow and eval runs. `llmzoo-genverify` is already the
# generative-verification project; this is its zoo-training sibling.
WANDB_PROJECT="${WANDB_PROJECT:-llmzoo-zoo}"
export WANDB_PROJECT

# Skip the trunk job when the trunk has already been staged into ZOO_ROOT (which
# is valid: total_steps/trunk_steps depend on tokens_per_param, batch,
# grad_accum, n_ctx and beta -- NOT on n_members, so a 12-member trunk serves a
# 100-member plan). The staging step must then have written zoo_plan.json too,
# because that is what zoo_trunk.sbatch would otherwise have done.
SKIP_TRUNK="${SKIP_TRUNK:-0}"

# Concurrent array tasks. The QOS ceiling is 32 GPUs per user on b200-batch, so
# keep the product across simultaneously-running arms at or under 32; anything
# above simply sits in QOSMaxGRESPerUser, which is harmless but invisible.
#
# The default is 32 rather than 10 because the wave count is what sets wall
# clock and 10 was quietly costing 3.3x. At N=100: 32 -> 4 waves, 10 -> 10.
# Measured per-branch times make that Medium in 28 h versus 71 h. Lower it when
# collaborators are queueing -- it is one env var, and the wave count is echoed
# below so a slow choice cannot be made silently.
THROTTLE="${THROTTLE:-32}"

# Extra flags forwarded to train_zoo.py (trunk AND branch) and, separately,
# to eval_domains.py. These were declared nowhere, so --alpha could not be
# passed through this launcher at all -- the singleton probe had to bypass it.
EXTRA="${EXTRA:-}"
GATE_EXTRA="${GATE_EXTRA:-}"

# Where the trunk and branches run. b200-batch is the intended pool and is ~1.56x
# faster (234 vs ~150 ktok/s measured at Mini). rtx-batch is the fallback: a
# SEPARATE 32-GPU QOS pool, so it neither consumes nor waits on the b200 ceiling.
# Override when b200-batch is congested or, as on 2026-09-08, when it is refusing
# to launch anything ("launch failed requeued held" on even a trivial job).
# Note --peak_tflops defaults to a B200; on rtx-batch the sealed MFU percentage
# needs a different denominator, though the ktok/s stands either way.
PARTITION="${PARTITION:-b200-batch}"

# Nodes to keep off. NOT a scheduling optimisation -- --nodelist and --exclusive
# are banned for that reason (skill Rule 2) -- but a genuinely faulty node has to
# be excludable. a0016 on rtx-batch threw `CUDA error: uncorrectable ECC error
# encountered` on three separate jobs across 2026-09-08/09 (a probe gate, then
# gate_b015 and spec_b015 in the same minute) and Slurm had not drained it. Set
# EXCLUDE_NODES= to clear this once it is fixed, and report it to the admins
# rather than routing around it forever.
EXCLUDE_NODES="${EXCLUDE_NODES:-a0016}"
EXC=()
[ -n "$EXCLUDE_NODES" ] && EXC=(--exclude="$EXCLUDE_NODES")

LAST=$((N_MEMBERS - 1))
K=$((N_MEMBERS - 1))
# Which array indices to submit. The default submits every member, which is the
# documented and tested path: a member whose w_<i>.npy exists exits immediately,
# so resubmitting verbatim after a partial array is safe and costs one no-op per
# finished member. Override to skip a staged prefix (ARRAY=12-99) or to retry a
# handful (ARRAY=17,43,81).
ARRAY="${ARRAY:-0-$LAST}"
SLUG="${ARCH#gpt2_zoo_}"
RUN_NAME="zoo_${TAG}_${SLUG}_k$K"
# The W&B cell for this whole zoo. train_zoo.py and eval_domains.py each build this
# same string internally, but train_stack.py has NO fallback -- it passes
# `group=args.wandb_group` straight through, and this launcher never sent one. So the
# spectrum run landed in `job_<jobid>` and sat OUTSIDE the row holding the trunk, the
# 100 branches and the gate it belongs to. Must match train_zoo.py's group exactly,
# including N.
WB_GROUP="zoo_${ARCH}_${TAG}_n${N_MEMBERS}"
WAVES=$(( (N_MEMBERS + THROTTLE - 1) / THROTTLE ))

echo "=================================================================="
echo " beta arm      : BETA=$BETA  ARCH=$ARCH  N=$N_MEMBERS"
echo " zoo root      : $ZOO_ROOT"
echo " spectrum run  : runs/$RUN_NAME   (k=$K)"
echo " partition     : $PARTITION"
echo " walltimes     : trunk=$TRUNK_TIME  branch=$BRANCH_TIME  spec=$SPEC_TIME"
echo " array         : $ARRAY   ($THROTTLE at a time -> ~$WAVES wave(s))"
echo " skip trunk    : $SKIP_TRUNK"
echo " exclude nodes : ${EXCLUDE_NODES:-<none>}"
echo " checkpointing : CKPT_EVERY=$CKPT_EVERY$([ "$CKPT_EVERY" = "0" ] && echo '  (OFF)')"
echo " wandb         : project=$WANDB_PROJECT"
echo "                 group=$WB_GROUP"
echo "=================================================================="

# A long span with no checkpointing is the expensive mistake this warns about.
# Keyed on the actual walltimes rather than on the arch string, so it fires for any
# configuration that has grown long enough to matter.
if [ "$CKPT_EVERY" = "0" ] \
   && { [ "${TRUNK_TIME%%:*}" -ge 8 ] || [ "${BRANCH_TIME%%:*}" -ge 8 ]; }; then
  echo "WARNING: CKPT_EVERY=0 with TRUNK_TIME=$TRUNK_TIME BRANCH_TIME=$BRANCH_TIME." >&2
  echo "         A walltime kill or a bad node loses the WHOLE span -- there is no" >&2
  echo "         resume below whole-member granularity. Suggest CKPT_EVERY=1000." >&2
fi

cd "$CODE_DIR"

# Refuse a mismatched directory HERE rather than 100 times inside the array.
# train_zoo.py's own guard (:621) is correct but per-member, so a mistyped root
# means N failed submissions and N log files instead of one message.
META="$ZOO_ROOT/$ARCH/zoo_meta.json"
if [ -f "$META" ]; then
  python3 - "$META" "$BETA" "$ARCH" "$N_MEMBERS" <<'PYCHECK'
import json, sys
meta, beta, arch, n = sys.argv[1], float(sys.argv[2]), sys.argv[3], int(sys.argv[4])
z = json.load(open(meta))
bad = []
if z.get("beta") is not None and abs(float(z["beta"]) - beta) > 1e-9:
    bad.append(f"beta={z['beta']!r} (asked {beta!r})")
if z.get("arch") not in (None, arch):
    bad.append(f"arch={z['arch']!r} (asked {arch!r})")
if z.get("n_members") not in (None, n):
    bad.append(f"n_members={z['n_members']!r} (asked {n!r})")
if bad:
    print(f"REFUSING: {meta} already describes a zoo with " + ", ".join(bad))
    print("Give this configuration its own ZOO_ROOT. Mixing them would relabel")
    print("already-written members, and the w_<i>.npy resume check returns 0")
    print("before the trunk's beta is ever read.")
    sys.exit(1)
print(f"  existing zoo_meta.json agrees (beta={z.get('beta')} "
      f"arch={z.get('arch')} n_members={z.get('n_members')}); resuming into it")
PYCHECK
fi

if [ "$SKIP_TRUNK" = "1" ]; then
  if [ ! -f "$ZOO_ROOT/$ARCH/trunk.pt" ]; then
    echo "SKIP_TRUNK=1 but no trunk at $ZOO_ROOT/$ARCH/trunk.pt" >&2
    exit 1
  fi
  if [ ! -f "$ZOO_ROOT/$ARCH/zoo_plan.json" ]; then
    echo "SKIP_TRUNK=1 but no zoo_plan.json at $ZOO_ROOT/$ARCH/ -- the staging" >&2
    echo "step must write the plan, since zoo_trunk.sbatch is what normally does." >&2
    exit 1
  fi
  t=""
  echo "trunk      : SKIPPED (staged trunk.pt + zoo_plan.json already present)"
else
  t=$(ZOO_ROOT="$ZOO_ROOT" ARCH="$ARCH" BETA="$BETA" N_MEMBERS="$N_MEMBERS" \
      EXTRA="$EXTRA" CKPT_EVERY="$CKPT_EVERY" \
      sbatch --parsable --partition="$PARTITION" --time="$TRUNK_TIME" \
             --job-name="trunk_$TAG" "${EXC[@]}" \
             slurm/zoo_trunk.sbatch)
  echo "trunk      : $t"
fi

# One GPU per array task, always. Branches are independent, AICR nodes are shared,
# and a 1-GPU job backfills immediately where --gres=gpu:8 waits for a node to
# drain. %N throttles concurrency to stay under the 32-GPU account ceiling and to
# leave room for collaborators on the shared fairshare.
DEP=()
[ -n "$t" ] && DEP=(--dependency="afterok:$t")
b=$(ZOO_ROOT="$ZOO_ROOT" ARCH="$ARCH" BETA="$BETA" N_MEMBERS="$N_MEMBERS" \
    EXTRA="$EXTRA" CKPT_EVERY="$CKPT_EVERY" \
    sbatch --parsable --partition="$PARTITION" --time="$BRANCH_TIME" \
           --job-name="branch_$TAG" \
           --array="$ARRAY%$THROTTLE" "${DEP[@]}" "${EXC[@]}" \
           slurm/zoo_branch.sbatch)
echo "branches   : $b  (array $ARRAY%$THROTTLE, ~$WAVES wave(s))"

# The §4.3 gate, on rtx-batch: a SEPARATE 32-GPU QOS pool, so it does not consume
# the b200 ceiling and runs alongside the next beta's training.
g=$(ZOO_ROOT="$ZOO_ROOT" ARCH="$ARCH" EXTRA="$GATE_EXTRA" \
    sbatch --parsable --job-name="gate_$TAG" --dependency="afterok:$b" \
           --time="$GATE_TIME" "${EXC[@]}" \
           slurm/eval_domains.sbatch)
echo "gate       : $g  (rtx-batch, exit 1 = gate fired)"

# The k=N-1 spectrum. Four flags and every one is load-bearing:
#   --source zoo / --zoo_dir   point at the zoo at all (parent of the arch dir)
#   --noise_scale 0.0          EnsembleDataset refuses a zoo with augmentation noise
#   --no_exclude_1d            train_zoo writes exclude_1d=False (§2.2); the
#                              default here is True and the cache gate refuses a mismatch
# The run name carries the ARCH, and that is the bug this fixes rather than a
# tidy-up: every N=100 scale has k=99, so `zoo_${TAG}_k$K` made Mini, Small and
# Medium all write runs/zoo_b030_k99 and the later fits would have silently
# overwritten the earlier ones.
s=$(sbatch --parsable --job-name="spec_$TAG" --dependency="afterok:$b" \
           "${EXC[@]}" \
           --account=p2026_0038_neu --partition=rtx-batch --nodes=1 --gres=gpu:1 \
           --cpus-per-task=8 --mem=200G --time="$SPEC_TIME" \
           --output="/scratch/$USER/logs/spec_$TAG-%j.out" \
           --error="/scratch/$USER/logs/spec_$TAG-%j.err" \
           --wrap "set -euo pipefail
                   source $CODE_DIR/slurm/aicr_env.sh
                   python -u scripts/train_stack.py \
                     --arch_list $ARCH --mode full \
                     --run_name $RUN_NAME --artifact_dir \$ARTIFACT_DIR \
                     --wandb_group $WB_GROUP --wandb_project $WANDB_PROJECT \
                     --source zoo --zoo_dir '$ZOO_ROOT' \
                     --n_samples $N_MEMBERS --k $K \
                     --noise_scale 0.0 --no_exclude_1d \
                     --epochs 400 --patience 100 --warmup_epochs 50 \
                     --latent_dim 8 --hidden_dim 64 --cond_dim 16 --batch_size 8")
echo "spectrum   : $s  (rtx-batch, k=$K)"

echo
echo "watch:  squeue -u \$USER"
echo "gate  : $ZOO_ROOT/$ARCH/domain_separation.json"
echo "spec  : \$ARTIFACT_DIR/runs/$RUN_NAME/pipeline_summary_k$K.json"
echo "        -> per_arch.$ARCH.spectrum_ev0_over_median"
echo "        -> per_arch.$ARCH.spectrum_effective_rank_ratio"
echo "        NOT spectrum_ev0_over_evlast (misstep 15b: reads 12.09 on data"
echo "        whose true ratio is 1.01)"
