#!/bin/bash
# slurm_stack_bench.sh
#
# Benchmark scores (MMLU / HellaSwag / GPQA) for an already-evaluated stack run.
# Evaluation only -- fits nothing, trains nothing, writes no artifact except a
# results JSON tagged `_bench`.
#
#   HF_TOKEN=... sbatch slurm_stack_bench.sh
#   BENCH="mmlu hellaswag" N_Q=25 sbatch slurm_stack_bench.sh        # cheap probe
#   SAMPLE_IDX=5 sbatch slurm_stack_bench.sh                        # typical member
#
# GPQA IS A GATED DATASET. `HF_TOKEN` must be exported BEFORE sbatch, so the job
# inherits it; it is already exported from ~/.bashrc on this account. Never hardcode
# it -- this file is tracked by git (RESEARCH_NOTES section 6.7). The pre-flight
# check below fails in ~3 seconds rather than dying in a `datasets` traceback 40
# minutes in.
#
# WHY THIS IS A SEPARATE JOB FROM THE dPPL RUN
# --------------------------------------------
# The two axes multiply. Eight arms x 3 benchmarks x 2 ranks x 3 archs at 200
# questions is ~50 benchmark measurements of ~2,400 forward passes each, i.e. ~120k
# forwards, ON TOP OF 48 full inverse_transform streaming passes over 350-400M
# parameter ensembles. That is borderline for the 8-hour `gpu` partition at three
# families and does not fit at six.
#
# `--arms` and `--bench` are independent flags precisely so this splits: the
# comprehensive dPPL table (all eight arms, no bench) is slurm_flow_run.sh, and this
# job takes a SUBSET of arms across all benchmarks. Default subset below is the four
# arms where a benchmark delta is interpretable -- reconstruction quality
# (`pca_only`, `vae`) and the generative comparison that actually decides whether
# the flows earn their keep (`gauss_codes` vs `flow_codes`).
#
# Runtime scales as len(ARMS) x len(BENCH) x N_Q x 2 ranks x n_archs. At the
# defaults (4 arms, 3 benchmarks, 200q, 2 ranks, 3 archs) expect ~4-6 h. Raising
# N_Q or ARMS raises it linearly; check against the 8 h wall before doing both.
#
# READ THE ABSOLUTE NUMBERS WITH CARE. These are 360-410M parameter base models, so
# MMLU and GPQA sit at chance (~25%) before any perturbation. The DELTA against the
# target model's own score is the signal; the absolute accuracy is not.

#SBATCH --job-name=llm_vae_bench
#SBATCH --output=/scratch/biggs.s/llm_vae/logs/bench_%j.out
#SBATCH --error=/scratch/biggs.s/llm_vae/logs/bench_%j.err
#SBATCH --chdir=/home/biggs.s/llm_vae
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
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
BENCH="${BENCH:-mmlu hellaswag gpqa}"
N_Q="${N_Q:-200}"
ARMS="${ARMS:-pca_only vae gauss_codes flow_codes}"
SAMPLE_IDX="${SAMPLE_IDX:-0}"
K_FULL=$((N_SAMPLES - 1))
K_HALF=$((N_SAMPLES / 2))
RUN_ROOT="$ROOT/runs/$RUN"

# Pre-flight, in order of how cheap the failure is.
if [[ "$BENCH" == *gpqa* ]] && [ -z "$HF_TOKEN" ]; then
    echo "GPQA is gated and HF_TOKEN is not set in this job's environment." >&2
    echo "Export it before sbatch, or drop gpqa:  BENCH=\"mmlu hellaswag\" sbatch $0" >&2
    exit 1
fi
if [ ! -f "$RUN_ROOT/ensemble/ensemble_meta.json" ]; then
    echo "No ensemble at $RUN_ROOT. Run slurm_stack_run.sh first." >&2
    exit 1
fi
for ARM in $ARMS; do
    case "$ARM" in
        flow_*)
            if ! ls -d "$RUN_ROOT"/flow_k*_* >/dev/null 2>&1; then
                echo "ARMS includes '$ARM' but $RUN_ROOT has no flow_k*_*/." >&2
                echo "Run slurm_flow_run.sh first, or drop the flow arms." >&2
                exit 1
            fi
            ;;
    esac
done

echo "=============================================="
echo "Job        : $SLURM_JOB_ID on $SLURMD_NODENAME"
echo "GPU        : $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
echo "Run        : $RUN   ranks k=$K_FULL and k=$K_HALF"
echo "Arms       : $ARMS"
echo "Benchmarks : $BENCH   ($N_Q questions each)"
echo "Eval target: sample_idx=$SAMPLE_IDX"
# Never interpolate the token itself: this line lands in a log file on shared
# scratch. `${VAR:-fallback}` expands to the VALUE when VAR is set, so the
# obvious one-liner leaks it.
if [ -n "$HF_TOKEN" ]; then echo "HF_TOKEN   : set (${#HF_TOKEN} chars)"; else echo "HF_TOKEN   : unset"; fi
echo "=============================================="

# --out_suffix bench keeps this from overwriting slurm_flow_run.sh's dPPL results,
# which land at the same path for the same sample_idx. report_stack.py globs
# stack_eval_results*.json and renders each file as its own section.
python -u eval_stack.py --artifact_dir "$ROOT" --run_name "$RUN" --mode full \
    --k "$K_FULL" "$K_HALF" \
    --arms $ARMS \
    --bench $BENCH --bench_n_questions "$N_Q" --bench_seed 0 \
    --sample_idx "$SAMPLE_IDX" \
    --out_suffix bench \
    --eval_seq_len 1024 --eval_n_sequences 64

echo; echo "######## report ########"
python -u report_stack.py --run_dir "$RUN_ROOT" \
    --output "$RUN_ROOT/results/report_stack.md"

echo; echo "Done — results in $RUN_ROOT/results/"
