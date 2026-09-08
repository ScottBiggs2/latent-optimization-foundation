#!/bin/bash
# Run one test module on a compute node.
#
#   TEST=tests/test_ensemble.py sbatch slurm/tests.sbatch
#   TEST=tests/test_ensemble.py TEST_ARGS=--tiny sbatch slurm/tests.sbatch
#   for t in test_ensemble test_bundle test_flow; do
#       TEST=tests/$t.py sbatch slurm/tests.sbatch
#   done
#
# The `cpu` partition, not a GPU one. These tests are numpy/CPU-torch only, and a
# cpu job costs nothing against the 32-GPU per-user ceiling -- so it runs ALONGSIDE
# whatever training is queued instead of behind it.
#
# tests/test_report.py is pure stdlib and runs on a laptop; it does not need this.
# The other three import torch at module level and need the conda env.

#SBATCH --job-name=llmzoo_tests
#SBATCH --account=p2026_0038_neu
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=00:20:00
#SBATCH --output=/scratch/%u/logs/%x-%j.out
#SBATCH --error=/scratch/%u/logs/%x-%j.err

set -euo pipefail
# sbatch does not set $0 to the script path, so point at the header explicitly.
CODE_DIR="${CODE_DIR:-/work/neu/p2026_0038_neu/$USER/llm-vae}"
source "$CODE_DIR/slurm/aicr_env.sh"

TEST="${TEST:-tests/test_report.py}"
TEST_ARGS="${TEST_ARGS:-}"

if [ ! -f "$TEST" ]; then
    echo "No such test: $TEST" >&2
    echo "Available:" >&2
    ls tests/*.py >&2
    exit 1
fi

echo "=============================================="
echo "Test   : $TEST $TEST_ARGS"
echo "=============================================="

python -u "$TEST" $TEST_ARGS
