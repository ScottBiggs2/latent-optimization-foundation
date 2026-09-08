#!/bin/bash
# Tiny-mode end-to-end smoke test of the FULL stack: ensembles -> per-family PCA ->
# codes artifact -> StackVAE -> flows in both spaces -> all eight eval arms -> report.
#
#   sbatch slurm/flow_smoke.sbatch
#
# Separate from slurm/stack_smoke.sbatch on purpose: that one is the known-green
# ~1-minute baseline for the pipeline without flows, and keeping it fast and green is
# worth more than one fewer file. It is also the bisect point if this script starts
# failing.

#SBATCH --job-name=flow_smoke
#SBATCH --account=p2026_0038_neu
#SBATCH --partition=b200-devel
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:30:00
#SBATCH --output=/scratch/%u/logs/%x-%j.out
#SBATCH --error=/scratch/%u/logs/%x-%j.err

set -euo pipefail
# sbatch does not set $0 to the script path, so point at the header explicitly.
CODE_DIR="${CODE_DIR:-/work/neu/p2026_0038_neu/$USER/llm-vae}"
source "$CODE_DIR/slurm/aicr_env.sh"

RUN=flow_smoke
RUN_ROOT="$ARTIFACT_DIR/runs/$RUN"
rm -rf "$RUN_ROOT"

COMMON="--mode tiny --run_name $RUN --artifact_dir $ARTIFACT_DIR --n_samples 12 \
        --noise_scale 1e-2 --chunk_budget_mb 8 --no_wandb \
        --epochs 150 --patience 150 --warmup_epochs 30 --latent_dim 8 \
        --hidden_dim 64 --cond_dim 16 --batch_size 8"

FCOMMON="--run_name $RUN --artifact_dir $ARTIFACT_DIR --no_wandb \
         --epochs 300 --patience 300 --batch_size 8 \
         --hidden_dims 64 128 64 --cond_dim 16 --time_embed_dim 16 --n_steps 20"

echo "############ 1. train_stack k=11 (N-1) ############"
python -u scripts/train_stack.py $COMMON --k 11

echo; echo "############ 2. train_stack k=6 (N/2) — must reuse Stage 2 ############"
python -u scripts/train_stack.py $COMMON --k 6

echo; echo "############ 3. flows, both spaces, both ranks ############"
for K in 11 6; do
  for SPACE in codes latent; do
    echo "--- k=$K space=$SPACE ---"
    python -u scripts/train_flow.py $FCOMMON --k "$K" --space "$SPACE"
  done
done

echo; echo "############ 4. eval, ALL EIGHT arms, both ranks ############"
python -u scripts/eval_stack.py --artifact_dir "$ARTIFACT_DIR" --run_name "$RUN" \
    --k 11 6 \
    --arms pca_only vae generate gauss_codes flow_codes flow_latent \
           flow_rt_codes flow_rt_latent \
    --flow_rt_steps 5 20 \
    --bench synthetic --bench_n_questions 8 \
    --eval_seq_len 64 --eval_n_sequences 4 --no_wandb

echo; echo "############ 5. report ############"
python -u scripts/report_stack.py --run_dir "$RUN_ROOT" \
    --output "$RUN_ROOT/results/report_stack.md"
head -60 "$RUN_ROOT/results/report_stack.md"

echo; echo "############ 6. refusals must fire ############"
# A sealed VAE exists, so --allow_legacy_vae should be unnecessary; but a bumped
# layout version must be refused rather than silently adopted.
RUN_ROOT="$RUN_ROOT" python - <<'PYCHK'
import json, os, sys
root = os.environ["RUN_ROOT"]
p = os.path.join(root, "vae_k11", "vae_meta.json")
meta = json.load(open(p))
orig = meta["layout_version"]
meta["layout_version"] = orig + 99
json.dump(meta, open(p, "w"), indent=2)
try:
    from llmzoo.gen.vae import StackVAE
    StackVAE.load(os.path.join(root, "vae_k11"), device="cpu")
    print("  FAIL: a bumped layout_version was accepted"); ok = False
except RuntimeError as exc:
    ok = "layout_version" in str(exc)
    print(f"  {'OK' if ok else 'FAIL'}: {str(exc).splitlines()[0]}")
finally:
    meta["layout_version"] = orig
    json.dump(meta, open(p, "w"), indent=2)
sys.exit(0 if ok else 1)
PYCHK

if python -u scripts/eval_stack.py --artifact_dir "$ARTIFACT_DIR" --run_name "$RUN" \
       --mode tiny --k 11 --arms pca_only --bench mmlu --no_wandb 2>&1 \
       | grep -q "Refusing --bench"; then
    echo "  OK: tiny mode refused a real benchmark"
else
    echo "  FAIL: tiny mode accepted a real benchmark"; exit 1
fi

if python -u scripts/train_flow.py $FCOMMON --k 11 --space codes 2>&1 \
       | grep -q "already holds a sealed flow"; then
    echo "  OK: refused to overwrite a sealed flow without --force"
else
    echo "  FAIL: overwrote a sealed flow"; exit 1
fi

# The gate reads ev0/MEDIAN, not ev0/ev[k-1]. On this pure-noise ensemble the
# former is ~1.0 (correctly flat) while the latter is ~12 at k=N-1, because ev[k-1]
# there is the smallest direction surviving the rank floor. Gating on the tail ratio
# would let a flat spectrum straight through, so this assertion pins the choice.
if python -u scripts/train_flow.py $FCOMMON --k 11 --space codes --force \
       --require_spectrum_ratio 10.0 2>&1 | grep -q "require_spectrum_ratio"; then
    echo "  OK: refused a flat spectrum under --require_spectrum_ratio"
else
    echo "  FAIL: trained on a flat spectrum despite the gate"; exit 1
fi

echo; echo "FLOW SMOKE OK"
