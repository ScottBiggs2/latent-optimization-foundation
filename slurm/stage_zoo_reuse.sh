#!/bin/bash
# Stage an N=100 zoo by REUSING a completed smaller zoo's trunk and members.
#
#   SRC=$ARTIFACT_DIR/zoo_b030/gpt2_zoo_mini \
#   DST=$ARTIFACT_DIR/zoo_b030_mini_n100/gpt2_zoo_mini \
#   ARCH=gpt2_zoo_mini BETA=0.30 N_MEMBERS=100 bash slurm/stage_zoo_reuse.sh
#
# WHY THIS IS VALID, AND WHAT IT CHECKS RATHER THAN ASSUMES
#
# `build_zoo_plan` lays the 20 anchors down FIRST and does not touch the RNG to
# do it, so the first min(N, 20) members of a 12-member plan and a 100-member
# plan are the same mixtures in the same order. And the step counts
#
#     total_steps  = tokens_per_param * D / (batch * grad_accum * n_ctx)
#     trunk_steps  = round((1 - beta) * total_steps)
#
# depend on NOTHING that n_members touches. Each branch's data seed is
# 10_000 + member_idx, and its LR schedule resumes the global cosine at
# trunk_steps. So member i of the 12-member run and member i of the 100-member
# run are the same computation, and the 12-member trunk is the 100-member trunk.
#
# That is an argument. This script MEASURES it: it refuses unless the freshly
# generated N=100 plan agrees with the source zoo_meta.json on every step count
# and on every reused member's pi, kind, mixture_id and branch. If a default ever
# drifts -- tokens_per_param, batch_size, grad_accum, n_ctx -- the reuse becomes
# wrong and this stops rather than silently mixing two token budgets in one zoo.
#
# It also checks the reused members were trained on the SAME GPU TYPE as the
# coming ones will be. bf16 reduction order differs across architectures and
# lands in the within-anchor spread that gate condition 2 divides by
# (docs/PHASE1_HANDOFF.md §8 trap 6), so a zoo half on b200 and half on rtx has
# a manufactured noise floor.
#
# zoo_meta.json is deliberately NOT copied: the source says n_members=12, which
# is exactly what train_zoo.py:621 refuses. The first branch to finish writes the
# correct one.

set -euo pipefail

SRC="${SRC:?set SRC to the completed zoo's ARCH-level dir}"
DST="${DST:?set DST to the new zoo's ARCH-level dir}"
ARCH="${ARCH:-gpt2_zoo_mini}"
BETA="${BETA:?set BETA}"
N_MEMBERS="${N_MEMBERS:-100}"
# ktok/s band the reused members must fall in, so a GPU-type mix is caught.
# Measured 2026-09-08: b200 Mini branches ran 233-272 ktok/s, rtx ~150.
EXPECT_KTOK_MIN="${EXPECT_KTOK_MIN:-200}"
EXPECT_KTOK_MAX="${EXPECT_KTOK_MAX:-400}"

echo "=================================================================="
echo " stage by reuse"
echo "   src  : $SRC"
echo "   dst  : $DST"
echo "   arch : $ARCH   beta : $BETA   N : $N_MEMBERS"
echo "=================================================================="

[ -f "$SRC/zoo_meta.json" ] || { echo "no $SRC/zoo_meta.json" >&2; exit 1; }
[ -f "$SRC/trunk.pt" ]      || { echo "no $SRC/trunk.pt" >&2; exit 1; }
mkdir -p "$DST"

# 1. Generate the new plan FIRST, so the checks below compare real artifacts.
python -u scripts/train_zoo.py --mode plan --arch "$ARCH" --beta "$BETA" \
    --n_members "$N_MEMBERS" --zoo_dir "$DST"

# 2. Refuse unless the reuse is actually sound.
python3 - "$SRC" "$DST" "$ARCH" "$BETA" "$EXPECT_KTOK_MIN" "$EXPECT_KTOK_MAX" <<'PYCHECK'
import glob, json, os, sys
src, dst, arch, beta, kmin, kmax = sys.argv[1:7]
beta, kmin, kmax = float(beta), float(kmin), float(kmax)
meta = json.load(open(os.path.join(src, "zoo_meta.json")))
plan = json.load(open(os.path.join(dst, "zoo_plan.json")))
bad = []

if meta.get("arch") != arch:
    bad.append(f"src arch {meta.get('arch')!r} != {arch!r}")
if abs(float(meta["beta"]) - beta) > 1e-9:
    bad.append(f"src beta {meta['beta']!r} != {beta!r}")

# The n_members-independence claim, checked rather than argued.
for k in ("total_steps", "trunk_steps"):
    if meta.get(k) != plan.get(k):
        bad.append(f"{k}: src {meta.get(k)} != new plan {plan.get(k)} -- a "
                   f"trainer default (tokens_per_param/batch/grad_accum/n_ctx) "
                   f"must have changed, so the trunk is NOT reusable")
sb = plan["total_steps"] - plan["trunk_steps"]
if meta.get("total_steps", 0) - meta.get("trunk_steps", 0) != sb:
    bad.append("branch_steps differ")

# Every reused member must be the same mixture in the same slot.
have = sorted(int(p.rsplit("_", 1)[1].split(".")[0])
              for p in glob.glob(os.path.join(src, "w_*.npy")))
smem = {m["idx"]: m for m in meta["members"]}
pmem = {m["idx"]: m for m in plan["members"]}
for i in have:
    a, b = smem.get(i), pmem.get(i)
    if a is None or b is None:
        bad.append(f"member {i} missing from a plan"); continue
    for f in ("pi", "kind", "mixture_id", "branch"):
        if a[f] != b[f]:
            bad.append(f"member {i}.{f}: src {a[f]!r} != new {b[f]!r}")

# GPU-type homogeneity of what we are importing.
tps = []
for i in have:
    p = os.path.join(src, f"member_{i}.json")
    if os.path.exists(p):
        t = (json.load(open(p)).get("throughput") or {}).get("tokens_per_sec")
        if t:
            tps.append(t / 1e3)
if tps:
    lo, hi = min(tps), max(tps)
    print(f"  reused members ran at {lo:.1f}-{hi:.1f} ktok/s "
          f"(expected band {kmin:.0f}-{kmax:.0f} = b200)")
    if lo < kmin or hi > kmax:
        bad.append(f"reused members span {lo:.1f}-{hi:.1f} ktok/s, outside the "
                   f"{kmin:.0f}-{kmax:.0f} band -- they were probably not all on "
                   f"one GPU type (handoff §8 trap 6)")
else:
    print("  WARNING: no member_*.json throughput to check GPU type against")

if bad:
    print("\nREFUSING to stage by reuse:")
    for x in bad:
        print("  - " + x)
    sys.exit(1)

print(f"  plan agrees with src on total_steps={plan['total_steps']} "
      f"trunk_steps={plan['trunk_steps']} branch_steps={sb}")
print(f"  {len(have)} reusable members verified identical in pi/kind/"
      f"mixture_id/branch: {have[0]}..{have[-1]}")
with open(os.path.join(dst, ".reuse_members"), "w") as f:
    f.write(" ".join(str(i) for i in have))
PYCHECK

REUSE=$(cat "$DST/.reuse_members")
echo "  reusing members: $REUSE"

# 3. Copy. .tmp + mv throughout: a cp killed at the walltime otherwise leaves a
#    truncated trunk.pt that --mode trunk's os.path.exists gate accepts, or a
#    truncated w_<i>.npy that the resume check accepts.
if [ -f "$DST/trunk.pt" ]; then
  echo "  trunk already staged"
else
  cp "$SRC/trunk.pt" "$DST/trunk.pt.tmp"
  mv "$DST/trunk.pt.tmp" "$DST/trunk.pt"
  echo "  trunk staged"
fi
n=0
for i in $REUSE; do
  if [ ! -f "$DST/w_$i.npy" ]; then
    cp "$SRC/w_$i.npy" "$DST/w_$i.npy.tmp"
    mv "$DST/w_$i.npy.tmp" "$DST/w_$i.npy"
    n=$((n + 1))
  fi
  [ -f "$SRC/member_$i.json" ] && cp "$SRC/member_$i.json" "$DST/member_$i.json"
done
echo "  staged $n member weight files"

FIRST=$(python3 -c "print(max(int(x) for x in '$REUSE'.split()) + 1)")
echo
echo "=================================================================="
echo " staged. Remaining branches to run: $FIRST-$((N_MEMBERS - 1))"
echo " Launch with SKIP_TRUNK=1 -- the trunk and zoo_plan.json are in place."
echo " A member whose w_<i>.npy exists exits immediately, so submitting the"
echo " full 0-$((N_MEMBERS - 1)) array is also safe and costs $FIRST no-ops."
echo "=================================================================="
