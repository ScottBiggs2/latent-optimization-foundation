#!/bin/bash
# slurm_flow_run.sh
#
# Trains conditional rectified flows in BOTH spaces at BOTH ranks over an existing
# stack run, then evaluates all eight arms and renders the report.
#
#   sbatch slurm_flow_run.sh                          # run=emb3, target = member 0
#   SAMPLE_IDX=5 sbatch slurm_flow_run.sh             # the typical-member target
#   RUN=emb6 N_SAMPLES=100 sbatch slurm_flow_run.sh
#
# Requires slurm_stack_run.sh to have finished first: this reads codes_k<k>/ and
# vae_k<k>/ and never re-fits a PCA.
#
# Flow training is idempotent here. A sealed flow_k<k>_<space>/ is SKIPPED rather
# than retrained, so running this twice with different SAMPLE_IDX trains the flows
# once and evaluates twice. (train_flow.py itself refuses to overwrite a sealed
# flow without --force, which under `set -e` would kill the job -- hence the guard
# rather than a bare invocation.)
#
# PREDICTION TO CHECK AGAINST, on the current manufactured (noise) ensemble:
#
#   `flow_codes` should land on the SAME dPPL as `gauss_codes`.
#
# The ensemble is w_i = w_0 + s*sigma*eps_i, so the per-family code distribution is
# near-Gaussian BY CONSTRUCTION and there is nothing for a flow to learn beyond the
# prior. If the two arms agree, this is a MACHINERY CHECK, not a finding
# (RESEARCH_NOTES section 3, Experiment 3). The comparison that carries information
# is `flow_codes` vs `gauss_codes` -- never `flow_codes` vs `pca_only`, which
# differ for reasons that have nothing to do with the flow.
#
# The flow checkpoints seal the spectrum they were trained on, so report_stack.py
# raises the flat-spectrum banner automatically and the caveat travels with the
# artifact rather than living in this header alone.

#SBATCH --job-name=llm_vae_flow
#SBATCH --output=/scratch/biggs.s/llm_vae/logs/flow_%j.out
#SBATCH --error=/scratch/biggs.s/llm_vae/logs/flow_%j.err
#SBATCH --chdir=/home/biggs.s/llm_vae
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=96G
#SBATCH --cpus-per-task=8
#SBATCH --time=08:00:00

set -e
mkdir -p /scratch/biggs.s/llm_vae/logs
source ~/miniconda/etc/profile.d/conda.sh
conda activate llm_vae
export HF_HOME=/scratch/biggs.s/hf_cache
export HF_DATASETS_CACHE=/scratch/biggs.s/hf_cache
export TRITON_CACHE_DIR=/scratch/biggs.s/triton_cache

ROOT=/scratch/biggs.s/llm_vae
RUN="${RUN:-emb3}"
N_SAMPLES="${N_SAMPLES:-100}"
SPACES="${SPACES:-codes latent}"
SAMPLE_IDX="${SAMPLE_IDX:-0}"
ARMS="${ARMS:-pca_only vae generate gauss_codes flow_codes flow_latent flow_rt_codes flow_rt_latent}"
RT_STEPS="${RT_STEPS:-5 20 100}"
K_FULL=$((N_SAMPLES - 1))
K_HALF=$((N_SAMPLES / 2))
RUN_ROOT="$ROOT/runs/$RUN"

echo "=============================================="
echo "Job        : $SLURM_JOB_ID on $SLURMD_NODENAME"
echo "GPU        : $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
echo "Run        : $RUN   N=$N_SAMPLES"
echo "Ranks      : k=$K_FULL (N-1)  and  k=$K_HALF (N/2)"
echo "Spaces     : $SPACES"
echo "Eval target: sample_idx=$SAMPLE_IDX"
echo "=============================================="

if [ ! -f "$RUN_ROOT/ensemble/ensemble_meta.json" ]; then
    echo "No ensemble at $RUN_ROOT. Run slurm_stack_run.sh first." >&2
    exit 1
fi

# Flows read the 40 KB codes artifact and nothing else (plus vae_k<k>/ for the
# latent space). That is what makes this stage re-runnable on a different ensemble
# source without touching flow code.
# NET SIZE AND EPOCHS ARE CALIBRATED, and small on purpose (job 9912696,
# diag_flow_capacity.py, k=99, M=300 rows, everything else held fixed):
#
#   width           params   epochs   train loss   sample rms ratio
#   64,128,64       51,043      500        1.846              1.039   <- correct
#   64,128,64       51,043     8000        1.568              0.959
#   256,512,256    361,699     2000        0.904              0.360
#   256,512,256    361,699     8000        0.651              0.183
#   512,1024,512 1,234,659     2000        0.650              0.158   <- worst
#
# Training loss falls MONOTONICALLY while the samples collapse toward the family
# mean. More capacity and more epochs make it strictly worse, so this is
# over-fitting, not under-fitting -- and dPPL cannot see it, because a collapsed
# generator lands near the ensemble mean, which on this ensemble is essentially w_0
# (RESEARCH_NOTES missteps 19 and 21).
#
# --holdout_frac is nonzero so early stopping gates on VAL loss. With no holdout it
# gates on train loss and therefore selects the most collapsed checkpoint on the
# curve. Watch the rms= column in the training log; 1.0 is correct.
FCOMMON="--run_name $RUN --artifact_dir $ROOT \
         --hidden_dims ${HIDDEN_DIMS:-64 128 64} --cond_dim 64 --time_embed_dim 64 \
         --cond_dropout 0.15 --source_std 1.0 --path_noise 0.0 \
         --epochs ${FLOW_EPOCHS:-500} --patience ${FLOW_PATIENCE:-500} \
         --holdout_frac ${HOLDOUT:-0.15} \
         --lr 3e-4 --batch_size 64 --weight_decay 1e-4 --n_steps 50"

echo; echo "######## Stage A: flows ########"
for K in "$K_FULL" "$K_HALF"; do
    if [ ! -d "$RUN_ROOT/codes_k$K" ]; then
        echo "  SKIP k=$K — no codes_k$K/. Run train_stack.py --k $K first."
        continue
    fi
    for SPACE in $SPACES; do
        DIR="$RUN_ROOT/flow_k${K}_${SPACE}"
        if [ -f "$DIR/flow_meta.json" ]; then
            echo "  SKIP k=$K space=$SPACE — already sealed at $DIR"
            continue
        fi
        # A latent-space flow decodes through vae.decode_cfg, so it cannot exist
        # without the VAE it was trained against.
        if [ "$SPACE" = "latent" ] && [ ! -f "$RUN_ROOT/vae_k$K/vae_meta.json" ]; then
            echo "  SKIP k=$K space=latent — no sealed vae_k$K/."
            continue
        fi
        echo "--- train_flow k=$K space=$SPACE ---"
        python -u train_flow.py $FCOMMON --k "$K" --space "$SPACE"
    done
done

echo; echo "######## Stage B: evaluation, all arms, both ranks ########"
# No --bench here. Benchmarks are a separate job (slurm_stack_bench.sh) because the
# two axes multiply: see that script's header for the arithmetic.
python -u eval_stack.py --artifact_dir "$ROOT" --run_name "$RUN" --mode full \
    --k "$K_FULL" "$K_HALF" \
    --arms $ARMS \
    --flow_rt_steps $RT_STEPS \
    --sample_idx "$SAMPLE_IDX" \
    --eval_seq_len 1024 --eval_n_sequences 64

echo; echo "######## Stage C: report ########"
python -u report_stack.py --run_dir "$RUN_ROOT" \
    --output "$RUN_ROOT/results/report_stack.md"

echo; echo "Done — results in $RUN_ROOT/results/"
