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

# PINNED, not floating. The cu128 index currently serves up to torch 2.11, but
# 2.9.1+cu128 is the build already proven on these B200s inside this project
# space, and a never-executed training loop is the wrong place to also be the
# first user of a new torch. It has sm_100; the assert below still checks.
TORCH_VERSION="${TORCH_VERSION:-2.9.1}"

echo "### torch $TORCH_VERSION, cu128 (sm_100 / Blackwell)"
pip install --index-url https://download.pytorch.org/whl/cu128 \
    "torch==$TORCH_VERSION" torchvision

echo "### llmzoo + the rest, editable"
pip install -e "$CODE_DIR"

echo "### verify"
python - <<'PY'
import torch
print("torch      ", torch.__version__, "cuda", torch.version.cuda)

# Read the compiled kernel set WITHOUT a GPU attached, which is the whole point
# of checking here on the login node.
#
# torch.cuda.get_arch_list() is the obvious call and it is the WRONG one: as of
# torch 2.9 it opens with `if not is_available(): return []`, so on a login node
# it returns an empty list and an `sm_100 in caps` assertion fails for a
# perfectly good cu128 build. That false negative cost a cycle here on
# 2026-09-07. The private binding underneath reads a compile-time string and
# needs no driver, no device and no CUDA init.
caps = torch._C._cuda_getArchFlags()
print("arch flags ", caps)
assert caps and "sm_100" in caps.split(), \
    f"this torch has no sm_100 kernels and will fail on a B200: {caps!r}"
print("sm_100     ", "present (B200 / Blackwell OK)")
import transformers, datasets, numpy
print("transformers", transformers.__version__)
print("datasets    ", datasets.__version__)
print("numpy       ", numpy.__version__)
assert transformers.__version__.startswith("4."), \
    f"pyproject pins transformers<5; got {transformers.__version__}"
assert datasets.__version__.startswith("3."), \
    f"pyproject pins datasets<4; got {datasets.__version__}"
import llmzoo
print("llmzoo     ", llmzoo.__file__)
PY

# Lock file, so "which versions produced this result" is answerable later. /work,
# not /scratch: scratch is purged at 30 days and this outlives the sprint.
pip freeze > "$CODE_DIR/slurm/env_lock_aicr.txt"
echo "### locked -> $CODE_DIR/slurm/env_lock_aicr.txt"

cat <<MSG

Done. Environment: $ENV_PATH

Note: torch.cuda.is_available() is False on a login node (no GPU attached) --
the arch-list check above still reads the compiled kernel set, which is the
thing that actually matters. Confirm end to end with:

    sbatch slurm/stack_smoke.sbatch
MSG
