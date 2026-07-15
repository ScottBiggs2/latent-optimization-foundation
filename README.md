# LLM-VAE-Early

A block-wise, conditioned Variational Autoencoder for LLM weight spaces.

Rather than treating an entire model as a single high-dimensional point, this project decomposes each LLM into its transformer decoder blocks and trains a shared VAE across all of them. The VAE is conditioned on the block's position in the stack (`block_idx`) and the model family it came from (`family`), allowing a single latent space to represent blocks from architectures of varying shapes and sizes.

---

## Method

### Why block-wise?

Full model weight vectors for even small LLMs (270M–360M params) are millions-dimensional. Working block-by-block reduces each training sample to a few million parameters, and the conditioning mechanism encodes the structural context that full-model approaches lose by flattening everything together.

### Why PCA first?

Transformer blocks still have 3–15M parameters — too large for a direct MLP encoder. We first compress each padded block vector through a shared **Dual/Gram-matrix PCA** (see [`dual_pca.py`](dual_pca.py)) into a ≤100-dimensional code, then train the VAE on those codes. The "dual trick" builds an (N×N) Gram matrix instead of an (N_params × N_params) covariance matrix; with ~100 training blocks the Gram matrix is 100×100 regardless of block size.

### Conditioning and padding

Different model families have different block sizes (e.g., GPT-2-medium blocks have ~12.6M params; OPT-350M blocks have ~3.2M). All blocks are zero-padded to the largest size found across all families. The padded positions:
- contribute zero dot-product mass during PCA fitting (so the PCA basis is naturally mask-safe)
- are excluded from reconstruction loss comparisons

The VAE encoder and decoder each receive a conditioning vector formed by concatenating learned embeddings for `block_idx` (position in stack) and `family` (model architecture), projected through a small MLP.

### Data and augmentation

Training data is the actual pre-trained weights from four models of similar scale:

| Model | Params | Layers | Hidden | Family idx |
|---|---|---|---|---|
| openai-community/gpt2-medium | 355M | 24 | 1024 | 0 |
| HuggingFaceTB/SmolLM2-360M | 360M | 32 | 960 | 1 |
| google/gemma-3-270m | ~270M | ~18 | ~1152 | 2 |
| facebook/opt-350m | 350M | 24 | 512 | 3 |

Extracting all blocks gives ~100 base training samples. On-the-fly noise augmentation (σ ≤ 1e-6 × block std) smooths learning dynamics at negligible cost.

---

## Repository Layout

```
LLM-VAE-Early/
├── models/
│   ├── registry.py          # ARCH_CONFIGS for all 4 families + tiny mocks
│   └── weight_extractor.py  # Block flattening, padding/masking, reconstruction
├── data/
│   ├── block_dataset.py     # BlockDataset: extracts all blocks, writes to memmap
│   ├── val_loader.py        # WikiText-2 DataLoader for perplexity evaluation
│   └── mc_loader.py         # MMLU / HellaSwag / GPQA loaders for eval_mc.py
├── dual_pca.py              # BatchedCovariancePCA with save/load (float16 on disk)
├── vae.py                   # ConditionedBlockVAE + BetaScheduler (KL warmup)
├── train.py                 # Four-stage pipeline with checkpoint/resume
├── evaluate.py              # Block reconstruction metrics (cosine sim, MSE, KL)
├── eval_lm.py               # LM perplexity: original vs. reconstructed model
├── eval_mc.py               # MMLU/HellaSwag/GPQA accuracy: original vs. reconstructed (opt-in, --eval_mc)
├── report.py                # Aggregates results/*.json into a markdown report (pure stdlib)
├── run_demo.py              # Local entry point (--mode tiny skips all downloads)
├── run_hpc.py               # HPC entry point; defaults to /scratch/biggs.s/llm_vae
├── slurm_train.sh           # SLURM job (V100, 64G RAM, 6h)
└── setup_env_hpc.sh         # One-time conda env setup on Explorer HPC
```

---

## Pipeline Stages

`train.py` runs four stages sequentially. Each stage checks for an existing checkpoint and skips if found, so jobs can be interrupted and resumed.

```
Stage 1 ─ Block extraction
   Load each model (one at a time to bound peak RAM)
   Extract all L transformer blocks per model
   Zero-pad to max_block_size; write to memmap
   ↓
Stage 2 ─ Dual PCA fit
   Gram matrix C = X^T X  (N×N, N ≈ 100)
   Randomised SVD on C → back-project to get k components
   Save components as float16 to reduce disk footprint
   ↓
Stage 3 ─ Encode
   Project all N blocks through PCA → (N, k) code matrix
   ↓
Stage 4 ─ VAE training
   ConditionedBlockVAE on codes, conditioned on block_idx + family
   ELBO loss with linear KL warmup (beta 0→1 over first 50 epochs)
   Early stopping, ReduceLROnPlateau
```

---

## HPC Deployment (Explorer)

### One-time setup
```bash
# On Explorer login node, inside the cloned repo:
bash setup_env_hpc.sh
mkdir -p /scratch/biggs.s/llm_vae/scratch/biggs.s/hf_cache

# Set your HuggingFace token (needed for some models):
export HF_TOKEN=hf_...
```

### Submit a job
```bash
sbatch slurm_train.sh
# Output: /scratch/biggs.s/llm_vae/slurm_<jobid>.out
```

To skip LM eval (faster turnaround during debugging):
```bash
EVAL_LM=0 sbatch slurm_train.sh
```

### Val split vs. train-on-all

With ~100 blocks, `--val_fraction 0.0` (default) is correct: the VAE needs to memorize
the training set before it can generalize. A non-zero val split causes immediate
overfitting — the best checkpoint ends up being the random initialization.

Switch to `--val_fraction 0.1` when the dataset grows to 500+ blocks.

### Resume a killed job
All four stages leave checkpoints. Resubmit `slurm_train.sh` unchanged — completed stages are detected and skipped automatically.

To force a specific stage to re-run:
```bash
python run_hpc.py --force_pca          # re-fit PCA, keep existing blocks
python run_hpc.py --force_train        # re-train VAE, keep PCA + codes
```

### Artifact layout on scratch
```
/scratch/biggs.s/llm_vae/
├── blocks/
│   ├── all_blocks.npy        (N × max_block_size, float32, memmap)
│   ├── all_masks.npy         (N × max_block_size, uint8)
│   └── dataset_meta.json
├── pca/
│   ├── components.npy        (k × max_block_size, float16)
│   ├── mean.npy
│   ├── explained_variance.npy
│   └── pca_meta.json
├── vae/
│   ├── vae_best.pt
│   ├── vae_config.json
│   ├── pca_codes.npy
│   └── train_metrics.json
└── results/
    ├── reconstruction_results.json
    └── lm_eval_results.json
```

### Memory budget

| Object | Size (full models) | Notes |
|---|---|---|
| All blocks (N×max_block_size) | ~3.7 GB | memmap on scratch, not all in RAM at once |
| PCA components (k×max_block_size) | ~2.4 GB | stored float16; loaded float32 for compute |
| PCA Gram matrix (N×N) | < 1 KB | trivially small |
| VAE parameters | < 1 MB | tiny MLP on PCA codes |
| PCA streaming batch | ~500 MB peak | `pca_batch_size=10` → 10×12M×4B |

The 64 GB SLURM request is conservative. In practice peak RAM is dominated by one model loaded at a time during extraction (~2–4 GB) plus the PCA component matrix (~2.4 GB).

---

## Results (first full run)

VAE trained on 108 blocks (24 GPT-2 + 32 SmolLM2 + 28 Qwen3 + 24 OPT), 500 epochs,
train-loss plateau stopping (`--val_fraction 0.0`). Best normalized ELBO: 0.0023.

### Weight reconstruction

| Model | Cosine sim | MSE |
|---|---|---|
| openai-community/gpt2-medium | **0.9997** | 5.9e-6 |
| HuggingFaceTB/SmolLM2-360M  | **0.9998** | 7.0e-6 |
| Qwen/Qwen3-0.6B              | **0.9982** | 5.6e-6 |
| facebook/opt-350m            | 0.772  | 3.4e-4 ⚠️ |

### Perplexity (WikiText-2 test, 512-token sequences)

| Model | Original PPL | Reconstructed PPL | Δ PPL |
|---|---|---|---|
| gpt2-medium   | 26.69 | **26.80** | +0.11 (+0.4%) ✅ |
| SmolLM2-360M  | 14.67 | **14.74** | +0.06 (+0.4%) ✅ |
| Qwen3-0.6B    | 26.25 | 51.77  | +25.5 (+97%) ⚠️ |
| OPT-350M      | 33.59 | 1199   | +1165 (+3470%) ❌ |

GPT-2 medium and SmolLM2-360M fully meet the target of < 1 PPL point delta.

### Known issues and next steps

**Qwen3-0.6B PPL sensitivity** — Weight reconstruction is excellent (cosine_sim=0.998)
but perplexity doubles. Models using explicit `head_dim` and RoPE attention appear
sensitive to the small residual weight errors; the block structure is captured correctly
but the architecture is numerically brittle. Possible fix: per-family reconstruction
fine-tuning or tighter convergence.

**OPT-350M poor reconstruction** — OPT applies biases to every attention projection and
feed-forward layer. Bias vectors have a fundamentally different statistical distribution
from weight matrices, which pulls the shared PCA basis in unhelpful directions. Options:
(a) exclude OPT from the joint model and handle it separately, (b) separate bias
parameters from weight matrices before PCA, or (c) replace OPT-350M with a bias-free
alternative (e.g. a LLaMA-family model).

**Dataset size** — With 108 blocks across 4 families, the VAE memorizes rather than
generalizes. `--val_fraction 0.0` (train on all data) is the correct setting at this
scale. Switching to `--val_fraction 0.1` becomes meaningful when the dataset grows to
500+ blocks, e.g. by adding OLMo-2 training checkpoints.

**Gemma 3** — `google/gemma-3-270m` is a gated model. To add it: accept terms at
huggingface.co/google/gemma-3-270m, set `HF_TOKEN` in `slurm_train.sh`, then add
`gemma3_270m` to `--arch_list`. Its registry entry (family_idx=4) is already present.

## Multiple-Choice Benchmarks (MMLU / HellaSwag / GPQA)

`eval_mc.py` extends the reconstruction evaluation beyond WikiText-2 PPL to
downstream multiple-choice accuracy. It's opt-in — enable with `--eval_mc`
(`run_hpc.py`) or `MC_EVAL=1` (`slurm_train.sh`) — since it adds real runtime
and dataset downloads on top of the default pipeline.

All four registered architectures are base pretrained models (no chat
template), so every benchmark is scored by ranking answer continuations via
log-likelihood under the model (`eval_mc.score_choices`) rather than
chat-formatted generation — see `data/mc_loader.py` for per-benchmark prompt
construction (5-shot for MMLU, 0-shot for HellaSwag, few-shot for GPQA).

**GPQA gating** — `Idavidrein/gpqa` is a gated dataset, same pattern as
Gemma 3: accept terms at huggingface.co/datasets/Idavidrein/gpqa and set
`HF_TOKEN` before running with `MC_EVAL=1` (or drop `gpqa` from
`--mc_benchmarks`). Expect near-chance accuracy (~25%) on these small base
models — the before/after delta is the meaningful signal, not absolute score.

Results are saved to `results/mc_eval_results.json`.

### Reporting

`report.py` aggregates whichever of `reconstruction_results.json` /
`lm_eval_results.json` / `mc_eval_results.json` exist in a results directory
into one markdown report. Pure stdlib, no torch/transformers import, so it's
safe to run locally against results copied down from Explorer (e.g. `scp -r
explorer:/scratch/biggs.s/llm_vae/results ./results`):

```bash
python report.py --results_dir ./results              # print to stdout
python report.py --results_dir ./results --output report.md
```

## Evaluation Targets

After a successful run, check `results/`:

| Metric | Target | File |
|---|---|---|
| Block cosine similarity | > 0.999 | reconstruction_results.json |
| Block MSE | < 1e-8 | reconstruction_results.json |
| PPL delta (vs. original) | < 1.0 ppl point per family | lm_eval_results.json |
| MMLU/HellaSwag/GPQA acc delta (vs. original) | as close to 0 as possible per family | mc_eval_results.json |

---

## Adding a New Architecture

1. Add an entry to `ARCH_CONFIGS` in `models/registry.py`:
   - `default_model_id`, `hf_model_type`, `layers_attr`, `family_idx`, `tiny_config`
2. Update `N_FAMILIES` and `MAX_BLOCKS` if needed.
3. No other changes required — block extraction uses `block.named_parameters()` and is architecture-agnostic.

> **Gemma 3 note:** `google/gemma-3-270m` is a multimodal model. Its text decoder blocks live at `model.language_model.layers` (already set in the registry), not `model.layers`. If the model ID is unavailable on HuggingFace, substitute the nearest available variant (e.g., `google/gemma-3-1b-pt`).

---
### Improvement Plan

1. Add a n+1 family to implement Classifier Free Guidance in the VAE to ensure that family embeddings are information rich conditions. 
2. Beef up the VAE, current size is quite light given the size of PCA encodings and the weirdness of weight spaces. 
3. Augment/increase the size of the dataset with:
   - Spamming Olmo-2 checkpoints
   - More noise augmentation
   - Does permuatation augmentation make sense? VAEs should be perm. invariant, but I'm not sure if the PCA is... Although the lesson from DWF was certainly that scale is the critical axis here, so this might be a method? However if the class embeddings are acting as a perm. spec. one-hot type label, then this will eliminate that signal and increase load on PCA or the VAE. 
4. Richer graph based embeddings for classes. This should allow the model to better generalize to new families with known blocks. 

---
## References

- Dual/Gram-matrix PCA algorithm: [NNeuralDynamics/DeepWeightFlow](https://github.com/NNeuralDynamics/DeepWeightFlow)
- Related weight-space work: [ScottBiggs2/DeepWeightFlow-Revisions](https://github.com/ScottBiggs2/DeepWeightFlow-Revisions), [ScottBiggs2/SDAF](https://github.com/ScottBiggs2/SDAF)
