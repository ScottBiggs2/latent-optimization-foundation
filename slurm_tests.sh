#!/bin/bash
# slurm_tests.sh
#
# Run a synthetic test module on a compute node. Explorer's login nodes SIGKILL
# resource-heavy processes (conda itself gets killed there), so even light numpy
# tests have to go through the scheduler.
#
# Usage:
#   TEST=tests_step1_dual_pca.py sbatch slurm_tests.sh
#   TEST=tests_step3_ensemble.py TEST_ARGS=--tiny sbatch slurm_tests.sh

#SBATCH --job-name=llm_vae_tests
#SBATCH --output=/scratch/biggs.s/llm_vae/logs/tests_%j.out
#SBATCH --error=/scratch/biggs.s/llm_vae/logs/tests_%j.err
#SBATCH --chdir=/home/biggs.s/llm_vae
#SBATCH --partition=short
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4
#SBATCH --time=00:20:00

set -e
mkdir -p /scratch/biggs.s/llm_vae/logs

source ~/miniconda/etc/profile.d/conda.sh
conda activate llm_vae

export HF_HOME=/scratch/biggs.s/hf_cache
export HF_DATASETS_CACHE=/scratch/biggs.s/hf_cache
export TRITON_CACHE_DIR=/scratch/biggs.s/triton_cache

TEST="${TEST:-tests_step1_dual_pca.py}"
TEST_ARGS="${TEST_ARGS:-}"
echo "=============================================="
echo "Job ID : $SLURM_JOB_ID   Node: $SLURMD_NODENAME"
echo "Test   : $TEST $TEST_ARGS"
echo "=============================================="

python -u "$TEST" $TEST_ARGS
