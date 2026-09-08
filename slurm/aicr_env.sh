#!/bin/bash
# Shared environment header for every AICR job in this repo. SOURCED, not executed.
#
#   source "$(dirname "${BASH_SOURCE[0]}")/aicr_env.sh"
#
# Override PROJ / WORK / CODE_DIR / ENV_PATH / ARTIFACT_DIR from the sbatch script
# or via `sbatch --export=ALL,ARTIFACT_DIR=...` before sourcing.
#
# WHY THE STORAGE IS SPLIT THE WAY IT IS
#
#   $HOME                             100 GiB.  Nothing large -- a full home fails
#                                     jobs in about a second, with an opaque code.
#   /work/neu/p2026_0038_neu/$USER    1.0 TB, persistent.  Code, the conda env, and
#                                     RUN ARTIFACTS.  The zoo is ~160 GB and has to
#                                     survive to the paper deadline.
#   /scratch/$USER                    10 TiB but PURGED AT 30 DAYS.  Caches and logs
#                                     only -- anything here is regenerable.

PROJ="${PROJ:-/work/neu/p2026_0038_neu}"
WORK="${WORK:-$PROJ/$USER}"
CODE_DIR="${CODE_DIR:-$WORK/llm-vae}"
ENV_PATH="${ENV_PATH:-$WORK/envs/llmzoo}"
S="/scratch/$USER"

source /apps/aicr/packages/miniforge3/25.3.0-3/3fiwftn/etc/profile.d/conda.sh
conda activate "$ENV_PATH"

cd "$CODE_DIR" || { echo "ERROR: CODE_DIR not found: $CODE_DIR" >&2; exit 1; }
mkdir -p "$S/logs" "$S/hf_cache/datasets" "$S/triton_cache" "$S/wandb"

export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"

# Caches: regenerable, so scratch. Never $HOME.
export HF_HOME="$S/hf_cache"
export HF_DATASETS_CACHE="$S/hf_cache/datasets"
export TRITON_CACHE_DIR="$S/triton_cache"

# Run artifacts: NOT regenerable on the sprint timeline, so project space.
export ARTIFACT_DIR="${ARTIFACT_DIR:-$WORK/llm_vae}"
mkdir -p "$ARTIFACT_DIR"

# A batch job cannot answer wandb's login prompt; it dies after loading everything.
if grep -qs "api.wandb.ai" "$HOME/.netrc"; then
  export WANDB_MODE="${WANDB_MODE:-online}"
else
  export WANDB_MODE="${WANDB_MODE:-offline}"   # then: wandb sync "$WANDB_DIR/<run>"
fi
export WANDB_DIR="${WANDB_DIR:-$S/wandb}"

# Fail loudly and immediately if the editable install is missing, rather than 20
# minutes in when the first `import llmzoo` runs inside a stage.
python -c "import llmzoo" 2>/dev/null || {
  echo "ERROR: llmzoo is not importable in $ENV_PATH." >&2
  echo "       Run slurm/setup_env_aicr.sh, or: pip install -e $CODE_DIR" >&2
  exit 1
}

echo "--- env ---"
echo "host=$(hostname)  job=${SLURM_JOB_ID:-local}  part=${SLURM_JOB_PARTITION:-local}"
echo "cpus=${SLURM_CPUS_PER_TASK:-?}  mem=${SLURM_MEM_PER_NODE:-?}M  gpus=${SLURM_GPUS_ON_NODE:-0}"
echo "code=$CODE_DIR  artifacts=$ARTIFACT_DIR  scratch=$S  wandb=$WANDB_MODE"
python -c "import torch;print('torch',torch.__version__,'cuda',torch.version.cuda,'gpus',torch.cuda.device_count(),[torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())])" 2>&1
echo "-----------"
