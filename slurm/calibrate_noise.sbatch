#!/bin/bash
# Re-calibrate --noise_scale for the CURRENT ensemble layout.
#
#   sbatch slurm/calibrate_noise.sbatch
#   ARCH_LIST="gpt2_medium pythia_160m" sbatch slurm/calibrate_noise.sbatch
#
# WHEN TO RUN THIS: whenever the layout changes, and only then. Nothing about a
# noise scale transfers across layouts -- one global weight_std mis-scales the
# perturbation for any submatrix whose own std differs from it, so adding the
# embeddings to D changed gpt2_medium's cost at s=1e-2 from +0.097% to +4.363%,
# a 45x move, while pythia_410m barely shifted (misstep 18).
#
# LIKELY OBSOLETE AFTER THE ZOO LANDS. A real zoo of independently-branched models
# needs no augmentation noise at all; source="noise" survives as the machinery test
# and as the beta -> 0 endpoint of the branch-point sweep, and this script with it.

#SBATCH --job-name=calib_noise
#SBATCH --account=p2026_0038_neu
#SBATCH --partition=b200-batch
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=200G
#SBATCH --time=01:30:00
#SBATCH --output=/scratch/%u/logs/%x-%j.out
#SBATCH --error=/scratch/%u/logs/%x-%j.err

set -euo pipefail
# sbatch does not set $0 to the script path, so point at the header explicitly.
CODE_DIR="${CODE_DIR:-/work/neu/p2026_0038_neu/$USER/llm-vae}"
source "$CODE_DIR/slurm/aicr_env.sh"

python -u scripts/calibrate_noise.py \
    --arch_list ${ARCH_LIST:-gpt2_medium smollm2_360m pythia_410m} \
    --scales ${SCALES:-1e-5 1e-4 1e-3 3e-3 1e-2 3e-2 1e-1} \
    --eval_seq_len 1024 --eval_n_sequences 64 \
    --artifact_dir "$ARTIFACT_DIR"
