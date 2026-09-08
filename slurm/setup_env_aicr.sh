#!/bin/bash
# One-time conda environment build on AICR.
#
#   bash slurm/setup_env_aicr.sh
#
# Safe on a login node: this only downloads and installs, it does not compute.
# (That was NOT true on Explorer, whose login nodes SIGKILLed even `conda activate`.)
#
# TORCH IS INSTALLED FIRST, FROM THE cu128 INDEX, AND THIS IS NOT OPTIONAL.
# AICR's B200s are Blackwell / sm_100. The cu124 build the Explorer env used does
# not have sm_100 kernels and fails at the first CUDA launch, not at import -- so a
# job looks healthy through model loading and dies deep into a stage. `pip install
# -e .` runs SECOND and deliberately does not list torch as a dependency, so it
# cannot pull a default-index wheel over this one.

set -euo pipefail

PROJ="${PROJ:-/work/neu/p2026_0038_neu}"
WORK="${WORK:-$PROJ/$USER}"
ENV_PATH="${ENV_PATH:-$WORK/envs/llmzoo}"
CODE_DIR="${CODE_DIR:-$WORK/llm-vae}"

source /apps/aicr/packages/miniforge3/25.3.0-3/3fiwftn/etc/profile.d/conda.sh

if [ ! -d "$ENV_PATH" ]; then
    echo "### creating env at $ENV_PATH"
    conda create -y -p "$ENV_PATH" python=3.11
fi
conda activate "$ENV_PATH"

echo "### torch, cu128 (sm_100 / Blackwell)"
pip install --index-url https://download.pytorch.org/whl/cu128 torch torchvision

echo "### llmzoo + the rest, editable"
pip install -e "$CODE_DIR"

echo "### verify"
python - <<'PY'
import torch
print("torch      ", torch.__version__, "cuda", torch.version.cuda)
# get_arch_list() reads the COMPILED kernel set and works with no GPU attached,
# which is exactly what we need to check on a login node.
caps = torch.cuda.get_arch_list()
print("arch list  ", caps)
assert any("sm_100" in c for c in caps), \
    f"this torch has no sm_100 kernels and will fail on a B200: {caps}"
import llmzoo
print("llmzoo     ", llmzoo.__file__)
PY

cat <<MSG

Done. Environment: $ENV_PATH

Note: torch.cuda.is_available() is False on a login node (no GPU attached) --
the arch-list check above still reads the compiled kernel set, which is the
thing that actually matters. Confirm end to end with:

    sbatch slurm/stack_smoke.sbatch
MSG
