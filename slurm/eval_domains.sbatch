#!/bin/bash
# The RESEARCH_PLAN §4.3 GATE. Per-domain perplexity across the zoo, then a
# pass/fail verdict on whether mixture identity is measurable in the models
# themselves.
#
#   ZOO_ROOT=$ARTIFACT_DIR/zoo_b030 ARCH=gpt2_zoo_mini sbatch slurm/eval_domains.sbatch
#
# THIS IS A GATE, NOT A REPORT. It exits 1 when the mixture axis is not visible.
# The response to a failure is a LARGER --beta and a re-run -- not a reframe, and
# not something to fix downstream. A flat spectrum measured on a zoo that failed
# this gate says nothing at all about weight space.
#
# rtx-batch, not b200-batch, and that is deliberate: rtx-batch is a SEPARATE
# 32-GPU QOS pool, so the gate runs ALONGSIDE the b200 training instead of queueing
# behind it. One GPU is plenty -- this is 5 domains x 64 blocks of forward passes
# per member, and the model is rebuilt in place rather than reconstructed.
#
# The first run streams and discards 100k documents per domain to build the
# held-out slices, which is the slow part; it caches them under $ARTIFACT_DIR and
# every later beta reuses the same bytes. --time covers the uncached case.

#SBATCH --job-name=eval_domains
#SBATCH --account=p2026_0038_neu
#SBATCH --partition=rtx-batch
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:30:00
#SBATCH --output=/scratch/%u/logs/%x-%j.out
#SBATCH --error=/scratch/%u/logs/%x-%j.err

set -euo pipefail
# sbatch does not set $0 to the script path, so point at the header explicitly.
CODE_DIR="${CODE_DIR:-/work/neu/p2026_0038_neu/$USER/llm-vae}"
source "$CODE_DIR/slurm/aicr_env.sh"

ARCH="${ARCH:-gpt2_zoo_mini}"
ZOO_ROOT="${ZOO_ROOT:-$ARTIFACT_DIR/zoo}"
ZOO_DIR="$ZOO_ROOT/$ARCH"
EXTRA="${EXTRA:-}"

echo "=== §4.3 gate: arch=$ARCH  zoo=$ZOO_DIR ==="

# Do NOT let `set -e` swallow the verdict: exit 1 is the gate firing, which is a
# result to report, not a job failure to debug.
#
# But exit 1 is ALSO what a traceback gives, and those are opposite diagnoses.
# On 2026-09-08 a gate hit `CUDA error: uncorrectable ECC error encountered` on a
# bad node and this block printed "Gate FAILED ... the response is a larger
# --beta" -- advice to spend hundreds of GPU-hours in response to a broken GPU.
# So distinguish them by ARTIFACT, not by exit code: eval_domains.py writes
# domain_separation.json before it returns, so a fresh file means the gate
# actually reached a verdict.
# Freshness by INODE, not by timestamp. `[ file -nt marker ]` compares mtime, and
# two files written in the same second tie -- which would misreport a real
# verdict as a crash. atomic_write_json does .tmp + os.replace, so a write always
# lands a NEW inode; that is exact regardless of clock granularity.
REPORT="$ZOO_DIR/domain_separation.json"
BEFORE="$(stat -c '%i:%Y:%s' "$REPORT" 2>/dev/null || echo absent)"
rc=0
python -u scripts/eval_domains.py --arch "$ARCH" --zoo_dir "$ZOO_DIR" $EXTRA || rc=$?
AFTER="$(stat -c '%i:%Y:%s' "$REPORT" 2>/dev/null || echo absent)"

FRESH=0
if [ "$AFTER" != "absent" ] && [ "$AFTER" != "$BEFORE" ]; then FRESH=1; fi

echo "=== eval_domains exit=$rc  report_written=$FRESH ==="
if [ "$rc" -ne 0 ] && [ "$FRESH" -eq 0 ]; then
    echo "CRASH, NOT A GATE VERDICT. No domain_separation.json was written, so"
    echo "eval_domains.py died before scoring. This is NOT an argument for a"
    echo "larger --beta -- read the stderr. Seen before: an uncorrectable ECC"
    echo "error on one rtx-batch node, fixed by resubmitting with --exclude."
elif [ "$rc" -eq 0 ]; then
    echo "GATE PASS."
else
    # A real verdict. Name what was actually measured rather than assuming N=12:
    # separation is only evaluated over kind=="anchor" groups, so an
    # all-singleton probe zoo reports separation=None -- a MISSING VERDICT, not a
    # failure, and report_singleton_probe.py is what scores that case.
    python3 - "$REPORT" <<'PYV'
import json, sys
r = json.load(open(sys.argv[1]))
v = r.get("verdict") or {}
sep, snr = v.get("separation"), v.get("signal_over_noise")
n = r.get("n_evaluated")
print(f"  evaluated {n} member(s)")
if sep is None:
    print("  separation : NO VERDICT -- this zoo has no anchor GROUPS, so §4.3's")
    print("               condition 1 is undefined. That is a MISSING VERDICT,")
    print("               not a gate failure, and NOT an argument for a larger")
    print("               beta. Score it with scripts/report_singleton_probe.py")
    print("               (RESEARCH_PLAN §6.5's slope).")
else:
    print(f"  separation : {'PASS' if sep.get('pass') else 'FAIL'}  "
          f"{sep.get('wins')}/{sep.get('checks')} anchors best on own domain")
if snr is None:
    print("  signal>noise: NO VERDICT (needs >1 anchor group with >1 branch)")
else:
    per = snr.get("per_domain") or {}
    worst = min(per, key=per.get) if per else None
    print(f"  signal>noise: {'PASS' if snr.get('pass') else 'FAIL'}  "
          f"min_ratio={snr.get('min_ratio')}"
          + (f"  worst domain: {worst}" if worst else ""))
    if per:
        print("    " + "  ".join(f"{d}={per[d]:.2f}" for d in per))
    if sep and sep.get("pass") and not snr.get("pass") and worst:
        print(f"  NOTE: separation PASSED and the failure is on '{worst}'. That is")
        print(f"        a different finding from a separation failure and a much")
        print(f"        weaker argument for raising beta.")
PYV
    echo "  Per RESEARCH_PLAN §4.3, a genuine separation failure means a LARGER"
    echo "  --beta and a re-run -- not a reframe, and not something to fix"
    echo "  downstream."
fi
exit "$rc"
