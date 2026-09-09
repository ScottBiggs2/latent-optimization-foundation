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
rc=0
python -u scripts/eval_domains.py --arch "$ARCH" --zoo_dir "$ZOO_DIR" $EXTRA || rc=$?

echo "=== eval_domains exit=$rc (0 = gate PASS, 1 = gate FAIL) ==="
if [ "$rc" -eq 1 ]; then
    echo "Gate FAILED. Per RESEARCH_PLAN §4.3 the response is a larger --beta and"
    echo "a re-run. Read domain_separation.json for which sub-condition and which"
    echo "domain decided it: at N=12 the zoo has only 3 anchors (web/code/math),"
    echo "but the signal>noise condition is evaluated on all 5 domains, so a"
    echo "failure on books/multilingual is a different diagnosis from a failure"
    echo "on separation."
fi
exit "$rc"
