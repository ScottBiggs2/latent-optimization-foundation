# LLM-VAE-Early

A conditioned Variational Autoencoder over LLM weight spaces.

The project compresses transformer decoder weights with dual (Gram-matrix) PCA, then
trains a conditioned VAE over the PCA codes. The goal is a latent space that a
generative model can sample to produce working weights.

**One PCA sample is one complete decoder stack.** Each architecture gets its own PCA
basis. The VAE conditions on the model family. This follows
[DeepWeightFlow](https://arxiv.org/abs/2601.05052) (ICLR 2026).

An earlier design treated one transformer block as one sample, with a single shared
basis across families. That design is still in the repo and still runs, but it is not
the current direction. The next section explains the difference.

**New here?** Read [RESEARCH_NOTES.md](RESEARCH_NOTES.md) first. It records what the
pipeline can and cannot measure, the next experiment, and 17 recorded missteps.

---

## ⚠ Current direction (2026-08-26): the sample unit changed

**Everything below describing one PCA sample as one transformer block is the LEGACY
path.** It still runs, and it produced every result in the Results section, but it is
not the direction of the work.

The project now follows [DeepWeightFlow](https://arxiv.org/abs/2601.05052) (ICLR
2026), which `dual_pca.py` was already adapted from, where **one PCA sample is the
final weight vector of one complete network**:

| | legacy (block-wise) | current (whole-stack) |
|---|---|---|
| one PCA sample | one transformer block | one whole decoder stack |
| `N` | number of blocks (150) | ensemble size per family (~100) |
| `k` | 97, capped by block count | `N-1` (rank bound) or `N/2` |
| basis | **one shared**, all families | **one per family** |
| padding | zero-pad to `max_block_size` | **none** — stacks are fixed-length |
| VAE conditioning | `(block_idx, family_idx)` | `family_idx` only |

Two long-standing problems dissolve rather than needing mitigation:

* **Padding disappears.** Every family has uniform block sizes (verified on real
  configs), so a stack is a fixed length and no mask is needed anywhere.
* **Posterior collapse stops being reachable.** `(family_idx, block_idx)` was a
  *unique key* over the block dataset, so a conditioned decoder could memorise the
  training set and ignore `z` — the measured cause of the 0.001 nats/sample collapse
  documented below. With ~100 samples sharing one `family_idx`, conditioning cannot
  identify a sample.

New modules (legacy ones untouched):

| file | role |
|---|---|
| [`data/ensemble_dataset.py`](data/ensemble_dataset.py) | per-family ensembles of whole stacks; regenerates augmentation noise from a seed instead of storing it |
| [`dual_gram_pca.py`](dual_gram_pca.py) | dual PCA that **never materialises** the `(k, D)` basis |
| [`vae.py`](vae.py) → `StackVAE` | family-only conditioning, per-family code normalization |
| [`train_stack.py`](train_stack.py) | ensembles → per-family PCA → codes → `StackVAE` |
| [`eval_stack.py`](eval_stack.py) | PCA-only / PCA+VAE / generate arms at any rank |
| [`calibrate_noise.py`](calibrate_noise.py) | measures the usable augmentation-noise band |

### Why the basis is never materialised

At `k=99` and `D=302M` an explicit component matrix is **119.6 GB per family**. The
dual formulation exists so it is never needed:

* training codes are exactly `U·√S` from the Gram eigendecomposition — **zero passes
  over the data** (verified to 1.2e-15 against an explicit `Xc @ components`);
* reconstruction is `x̂ = mean + Xcᵀa` with `a = U_k(z/√S_k)`, i.e. a weighted sum of
  ensemble members — one streaming pass.

`DualGramPCA` stores the mean plus a 100×100 eigenbasis instead: ~1.2 GB per family.

### The honest caveat

DeepWeightFlow uses ~100 *independently trained* networks. We have **one** pretrained
model per family, so the ensemble is manufactured as `w_0 + s·σ·ε_i` (sample 0 is the
real model, exactly). An isotropic ensemble spans injected-noise directions, so
**this first pass validates machinery, not weight-space structure.**

Permutation augmentation is *not* a fix: generating permuted copies deliberately adds
the permutation orbit, whereas TransFusion (DeepWeightFlow's canonicalizer) exists to
*remove* it. Permuted copies are functionally identical to the original, so a model
trained on them regenerates the same network — a good sanity harness, not new
structure. Real structure needs genuinely different complete models; **Pythia's ~143
checkpoint revisions** are the cheap source at fixed model scale.

---

## Method (legacy block pipeline)

> The three subsections below describe the **block** pipeline. They are accurate for
> that pipeline and inaccurate for the current one. For the current design, read
> "Current direction" above. The main differences: one sample is a whole stack, each
> family has its own basis, there is no padding, and the VAE conditions on
> `family_idx` alone.

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

Training data is the actual pre-trained weights from the default `--arch_list` (`run_hpc.py`):

| Model | Params | Layers | Hidden | Family idx | Default? |
|---|---|---|---|---|---|
| openai-community/gpt2-medium | 355M | 24 | 1024 | 0 | yes |
| HuggingFaceTB/SmolLM2-360M | 360M | 32 | 960 | 1 | yes |
| Qwen/Qwen3-0.6B | 0.6B | 28 | 1024 | 2 | yes |
| facebook/opt-350m | 350M | 24 | 512 | 3 | **no** — biases break the shared PCA basis, see Known Issues |
| google/gemma-3-270m | ~270M | ~18 | ~1152 | 4 | **no** – bugged `HF_TOKEN` is a huge pain to deal with|
| HuggingFaceTB/SmolLM2-135M | 135M | 30 | 576 | 5 | yes |
| EleutherAI/pythia-160m | 160M | 12 | 768 | 6 | yes |
| EleutherAI/pythia-410m | 410M | 24 | 1024 | 7 | yes |

Extracting all blocks across the default roster gives a few hundred base training samples.

Augmentation noise is applied to the **PCA codes** during VAE training, not to raw
weights. The old weight-space path in `BlockDataset.__getitem__` never actually ran —
PCA reads blocks through `make_loader()` and the VAE trains on precomputed codes, so
nothing ever called `__getitem__`. Code space is the right place for it anyway:
components are unit-norm and orthogonal, so isotropic weight-space noise projects to
isotropic code-space noise of the same std. Two knobs:

| Flag | Scale relative to | Default |
|---|---|---|
| `--noise_scale` | each block's own weight std | 1e-7 (≈9 orders of magnitude below the code scale — effectively off) |
| `--code_noise_std` | each PCA dimension's own std | 0.02 (the one that actually bites at this dataset size) |

Current training runs use a **350–400M band with three distinct families**
(`gpt2_medium` 355M / GPT-2, `smollm2_360m` 360M / LLaMA, `pythia_410m` 410M /
GPT-NeoX): scale is held roughly constant while architectural diversity is kept.
Block sizes are 12.60M / 9.83M / 12.60M, so `max_block_size` = 12.60M and only
smollm2_360m pads (~22%). This matters because the zero-padding pattern correlates
perfectly with family, so heavy padding makes the leading PCs encode *which family
this is* rather than *what kind of block this is* — the earlier 135M–355M roster
forced smollm2_135m to ~72% padding.

---

## Repository Layout

Two pipelines live here. **Use the stack pipeline.** The block pipeline is kept
because it produced the historical results and because `diag_refit_shared_pca.py`
still reads its artifacts.

```
LLM-VAE-Early/
├── RESEARCH_NOTES.md           # READ FIRST: state, next steps, recorded missteps
│
│  ===== STACK PIPELINE (current) =====
├── data/ensemble_dataset.py    # EnsembleDataset: per-family ensembles of whole stacks
├── dual_gram_pca.py            # DualGramPCA: never builds the (k, D) basis
├── vae.py            (StackVAE)# family-only conditioning, per-family code norm
├── train_stack.py              # ensembles -> per-family PCA -> codes -> StackVAE
├── eval_stack.py               # pca_only / vae / generate arms, any rank
├── calibrate_noise.py          # measures the usable augmentation-noise band
├── slurm_stack_run.sh          # full run: both rank arms, all eval arms
├── slurm_stack_smoke.sh        # tiny-mode end-to-end check (~1 min)
├── slurm_calibrate_noise.sh
│
│  ===== TESTS =====
├── tests_step1_dual_pca.py     # 35 checks: numerics, rank floor, codes I/O
├── tests_step3_ensemble.py     # 56 checks: ensemble + DualGramPCA vs brute force
├── slurm_tests.sh              # runs a test module on a compute node
│
│  ===== SHARED =====
├── models/registry.py          # ARCH_CONFIGS, family_idx, tiny mocks
├── models/weight_extractor.py  # block flatten/reconstruct, exclude_1d
├── data/val_loader.py          # WikiText-2 loader
├── data/mc_loader.py           # MMLU / HellaSwag / GPQA loaders
├── wandb_utils.py
├── setup_env_hpc.sh
│
│  ===== BLOCK PIPELINE (legacy) =====
├── data/block_dataset.py       # one block per sample, zero-padded
├── dual_pca.py                 # BatchedCovariancePCA (builds an explicit basis)
├── vae.py  (ConditionedBlockVAE)
├── train.py, run_hpc.py, run_demo.py, slurm_train.sh
├── evaluate.py, eval_lm.py, eval_mc.py, eval_pca_only.py
├── diag_refit_shared_pca.py    # one-off: refit the old shared basis with eigh
├── report.py
└── slurm_train.sh, slurm_eval_mc.sh, slurm_eval_pca_only.sh, slurm_diag_refit.sh
```

`vae.py` holds both VAE classes. `StackVAE` is current. `ConditionedBlockVAE` is
legacy.

---

## Pipeline Stages

### Stack pipeline (current)

`train_stack.py` runs four stages. Each stage skips work that already exists on disk.

```
Stage 1 ─ Ensembles                                    (data/ensemble_dataset.py)
   Load one model per architecture.
   Concatenate its decoder blocks into w_0, length D = n_layers * block_size.
   Check that all blocks in the family have the same size. Fail if they do not.
   Save w_0 only. Members 1..N-1 exist as a seed, never as a file.
   ↓
Stage 2 ─ Per-family Gram PCA at k = N-1              (dual_gram_pca.py)
   Pass 1: accumulate the mean over parameter chunks.
   Pass 2: accumulate the centered Gram C in float64.
   eigh on C. Drop eigenvalues below the rank floor.
   Codes are U*sqrt(S). No pass over the data. No (k, D) basis.
   ↓
Stage 3 ─ Codes at rank k
   Take the first k columns. A k-prefix IS the rank-k PCA.
   Record each family's own code mean and scale.
   ↓
Stage 4 ─ StackVAE
   Conditioned on family_idx only.
   Free bits, conditioning dropout, cosine LR, warmup-gated checkpointing.
```

Fit Stage 2 once. Every lower rank is a column prefix of that fit, so a rank sweep
costs nothing extra.

```bash
sbatch slurm_stack_smoke.sh          # tiny mode, ~1 minute, no downloads
sbatch slurm_stack_run.sh            # full run, ~10 minutes on one V100
RUN=myrun N_SAMPLES=143 NOISE_SCALE=3e-3 sbatch slurm_stack_run.sh
```

### Block pipeline (legacy)

`train.py` runs the original four stages: pad every block to `max_block_size`, fit
ONE shared PCA, encode, then train `ConditionedBlockVAE` on `(block_idx, family_idx)`.
`train.py` now refuses to resume any `blocks/` directory whose
`dataset_meta.json` lacks a matching `layout_version`.

---

## HPC Deployment (Explorer)

### One-time setup
```bash
# On Explorer login node, inside the cloned repo:
bash setup_env_hpc.sh

# Set your HuggingFace token (needed for some models):
export HF_TOKEN=hf_...
```

### Submit a job

Stack pipeline (current). All logs land in `/scratch/biggs.s/llm_vae/logs/`.

```bash
sbatch slurm_stack_smoke.sh          # tiny mode, ~1 min, no downloads. Run this first.
sbatch slurm_stack_run.sh            # full run, ~10 min on one V100
sbatch slurm_calibrate_noise.sh      # measure the usable noise band
sbatch --export=ALL,TEST=tests_step1_dual_pca.py slurm_tests.sh
sbatch --export=ALL,TEST=tests_step3_ensemble.py,TEST_ARGS=--tiny slurm_tests.sh
```

Override the stack run with environment variables:

```bash
RUN=pythia-revs N_SAMPLES=143 NOISE_SCALE=3e-3 sbatch slurm_stack_run.sh
```

Block pipeline (legacy):

```bash
sbatch slurm_train.sh
EVAL_LM=0 sbatch slurm_train.sh      # skip LM eval
```

**Explorer login nodes kill heavy processes.** Even `conda activate` gets killed
there. Submit every job with `sbatch`, including small numpy tests.

### Val split vs. train-on-all

With ~100 blocks, `--val_fraction 0.0` (default) is correct: the VAE needs to memorize
the training set before it can generalize. A non-zero val split causes immediate
overfitting — the best checkpoint ends up being the random initialization.

Two further reasons a random-block split is the wrong instrument here:

1. **It poisons the PPL table.** `eval_lm` reconstructs *every* block of a model, so
   held-out blocks go straight into the reported perplexity untrained, dragging down
   whichever family happens to own them.
2. **It leaks.** PCA is fit on all blocks *before* the split happens
   (`train.py`: `stage_pca` runs ahead of `train_vae`), so the "held-out" blocks are
   already inside the basis.

When the dataset grows to 500+ blocks, hold out a **whole family** or a **whole
checkpoint revision** rather than random blocks.

### Resume a killed job
All four stages leave checkpoints. Resubmit `slurm_train.sh` unchanged — completed stages are detected and skipped automatically.

To force a specific stage to re-run:
```bash
python run_hpc.py --force_pca          # re-fit PCA, keep existing blocks
python run_hpc.py --force_train        # re-train VAE, keep PCA + codes
```

### Artifact layout on scratch

Every stack run writes into its own `runs/<name>/` directory. Runs never overwrite
each other.

```
/scratch/biggs.s/llm_vae/
├── runs/
│   ├── perfam/                            <- a stack run
│   │   ├── ensemble/
│   │   │   ├── ensemble_meta.json         (N, noise_scale, seed, layout_version)
│   │   │   └── <arch>_w0.npy              (D,) float32 — the real stack, ~1.2 GB
│   │   ├── pca/<arch>/
│   │   │   ├── mean.npy                   (D,) float32
│   │   │   ├── gram_evals.npy             (k,)  float64
│   │   │   ├── gram_evecs.npy             (N,k) float64
│   │   │   ├── codes.npy                  (N,k) float32
│   │   │   └── gram_pca_meta.json
│   │   ├── vae_k99/  vae_k50/
│   │   │   ├── vae_best.pt, vae_config.json, train_metrics.json
│   │   ├── results/stack_eval_results*.json
│   │   └── pipeline_summary_k*.json
│   └── 2026-08-21-shared-6family/         <- archived legacy run (JSONs only)
├── logs/                                  <- all slurm .out/.err
└── hf_cache/
```

There is no `components.npy`. `DualGramPCA` stores the mean plus a small
eigenbasis instead. At k=99 and D=302M an explicit basis would be 119.6 GB per
family.

### Memory budget

Stack pipeline, N=100, D≈302M per family:

| Object | Size | Notes |
|---|---|---|
| `w0.npy` per family | ~1.2 GB | on scratch, read as a memmap |
| `mean.npy` per family | ~1.2 GB | on scratch |
| Gram eigenbasis | ~80 KB | (100,100) float64 |
| Codes | ~40 KB | (100,99) float32 |
| Chunk in flight | ~525 MB | (N, chunk) float32, chunk from a byte budget |
| Ensemble on disk | **0 bytes** | members 1..N-1 come from a seed |
| Explicit (k,D) basis | **never built** | would be 119.6 GB per family |
| StackVAE parameters | < 1 MB | |

Measured on the 2026-08-26 full run: **10 minutes wall clock, 20 GB peak RSS** for
three families, three PCA fits, two VAEs, and 18 evaluations. Each PCA fit takes
about 45 seconds.

Set the chunk size with `--chunk_budget_mb`. The code divides that budget by
`4 * n_samples`, so the chunk shrinks as the ensemble grows.

---

## Results

> **Reading order.** Sections are chronological. The 2026-08-26 entries describe the
> current stack pipeline. Everything before them describes the legacy block pipeline
> and is kept as the record of why the design changed.
>
> The first two tables come from the original 4-family roster, before the registry
> gained smollm2_135m / pythia_160m / pythia_410m. They were never re-run on the
> larger roster, and the stack pipeline superseded that question.

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

### Measured posterior collapse (2026-08-24 diagnosis)

Read off `vae/train_metrics.json` from the 150-block run. **The latent was dead.**

| Epoch | beta | train_recon | train_kl (per-dim) | KL nats/sample |
|---|---|---|---|---|
| 1 | 0.02 | 1.079 | 0.094 | 3.0 |
| 10 | 0.20 | 0.931 | **0.113** | **3.6** ← peak |
| 50 | 1.00 | 0.674 | 0.050 | 1.6 |
| 100 | 1.00 | 0.381 | 0.020 | 0.65 |
| 500 | 1.00 | 0.0022 | 0.000033 | **0.001** |

Total KL is `32 × 3.3e-05 = 0.00106 nats/sample` — 0.0015 bits. The decoder was
reconstructing each block from its conditioning alone.

The structural cause: **`(family_idx, block_idx)` is a unique key over the block
dataset.** Every block has its own pair, so a `cond_dim`-wide vector can memorize the
whole training set and `z` is never needed. High `cosine_sim` was memorization through
the conditioning, not learned weight-space structure.

Two consequences worth internalizing:

- **A longer run makes this worse, not better.** Reconstruction was still improving at
  epoch 500; it was converging toward a perfect lookup table.
- **No generative machinery can sit on top of this.** Sampling `z ~ N(0,I)` returns the
  memorized block, and a flow/diffusion model over these latents would just learn
  `N(0,I)` and add nothing.

The encouraging part: 3.6 nats existed at epoch 10, before beta reached 1.0. The
information is recoverable, so the mitigations target something real:

| Mitigation | Flag | What it does |
|---|---|---|
| Conditioning dropout | `--cond_dropout 0.15` | Blanks the conditioning vector during training, so the decoder cannot use the unique key. The learned null embedding doubles as the CFG unconditional branch (`vae.decode_cfg`). |
| Free bits | `--free_bits 0.05` | Per-latent-dim KL floor in nats; beta cannot crush a dimension to exactly zero. |
| Global code scale | (automatic) | `set_code_norm` now reduces the code std to a **single scalar**. Per-dimension whitening equalized PC 0 and PC 96 — which differ 3.4× in std — and spent decoder capacity on directions carrying almost no weight-space energy. |

**Gate every future run on `kl/total_nats_per_sample` and `kl/active_units` in wandb.**
`train/kl` alone cannot tell a healthy latent from a dead one. Note the warning in
`train.py` compares KL against the free-bits floor (`free_bits × latent_dim`), not
against a constant: free bits alone guarantees that floor even when every dimension is
pinned at it and carries no information.

### PCA-only ablation: the basis is the bottleneck, not the VAE

`eval_pca_only.py` runs the pipeline with the VAE removed
(`block → pad → PCA project → PCA inverse → strip padding → write back`), which is
the control that was missing: every other number in this repo measures PCA and VAE
jointly, so a bad family could never be attributed to a stage.

Run against the 150-block / 97-component basis, WikiText-2, 64 × 1024 tokens:

| arch | PCA-only cos | PCA-only ΔPPL | PCA+VAE cos | PCA+VAE ΔPPL |
|---|---|---|---|---|
| smollm2_360m | 0.99989 | **+0.22%** ✅ | 0.99978 | +2% ✅ |
| gpt2_medium | 0.99939 | +72,468% ❌ | 0.99899 | +23,776% ❌ |
| pythia_410m | 0.38825 | +945,552% ❌ | 0.37483 | +568,040% ❌ |

**The VAE is essentially innocent.** Removing it does not fix any family — the
PCA-only numbers are as bad or worse. `pythia_410m` reconstructs at cosine 0.388 with
no VAE in the loop at all, which means the shared 97-component basis simply cannot
represent a GPT-NeoX block: those blocks are largely orthogonal to the span the basis
retained. That is consistent with the near-flat eigenvalue spectrum
(`ev[0]/ev[96]` = 11.8) — 150 independently-trained blocks from six architectures are
close to mutually orthogonal, so 97 components cannot cover all of them.

(Where both numbers are catastrophic, their *ordering* is noise — a PPL of 16,224 vs
6,372 is not a meaningful difference. The signal is which families survive at all.)

**Required fidelity is far tighter than the current target.** smollm2_360m at 0.99989
is fine; gpt2_medium at 0.99939 is destroyed. The usable band is roughly
**cosine ≥ 0.9999**, i.e. relative weight error below ~1e-4. The `> 0.999` target in
"Evaluation Targets" below is too loose by at least an order of magnitude — **gate on
ΔPPL, not on cosine.**

Two fixes follow directly, and both are now wired:

- `--exclude_1d` — pythia/OPT-style per-projection biases are exactly the
  bias-pollution mechanism that got `opt_350m` dropped, and pythia is the family that
  fails hardest. Requires re-fitting the PCA (`--force_pca`) since the flat layout
  changes.
- **Keep `n_components = N_blocks − 1`.** On the 350–400M roster that is 79 of 80
  blocks, so the basis spans the full affine hull and projection is lossless
  in-sample. The cost is that decoded samples are confined to that hull — a real
  ceiling for generation, and the reason block count eventually has to grow.

Reproduce with `sbatch slurm_eval_pca_only.sh`, or sweep rank with
`N_COMPONENTS="79 40 20" sbatch slurm_eval_pca_only.sh`.

### Two measurement bugs found and fixed

**`total_variance_captured` was always exactly 1.0.** `randomized_svd` returns only
`n_comp` singular values, and the ratio was normalized by the sum of *those* — so it
summed to 1.0 by construction and said nothing about what truncation discarded. Now
normalized by `tr(C)/(n−1)`, captured before the SVD. Verified: a 3-of-11 basis on
synthetic data reports 57.73%, not 100%. PCA checkpoints written before this fix print
a notice on load; re-fit with `--force_pca` for a real number.

**`reparameterize` silently returned `mu` inside any `no_grad` block.** That made the
val ELBO a different objective from the train ELBO, and would have silently no-op'd any
sampling code written under `torch.no_grad()`. Now an explicit `sample: bool`; all
reconstruction-fidelity call sites pass `sample=False` deliberately.

### 2026-08-26: the shared-basis failure is real, not numerical

`dual_pca.py` was refitting the 150-block / 97-component shared basis with
`randomized_svd` and storing components as float16. Both are now known to be
unreliable — on a synthetic steep spectrum at n=24, `randomized_svd` returned an
orthogonality error of **0.82** at `k=n-1` with one component's direction 100% wrong,
which the unit-normalisation then promoted to equal footing with the real components.
So the obvious question was whether `pythia_410m`'s cosine of 0.388 was arithmetic
rather than representation.

**It is not.** Refitting the same blocks with `eigh` + a float64 Gram + float32
storage (job 9678739) changes nothing:

| arch | old (`randomized_svd`, fp16) | new (`eigh`, fp64 Gram, fp32) |
|---|---|---|
| smollm2_360m | 0.99992 | 0.99992 |
| gpt2_medium | 0.99939 | 0.99939 |
| pythia_160m | 0.61731 | 0.62074 |
| qwen3_0_6b | 0.58642 | 0.59084 |
| pythia_410m | 0.38825 | **0.38817** |

Eigenvalues agree to 7.4e-6 relative and the spectrum ratio is 11.83 either way. The
shared block basis genuinely cannot represent a GPT-NeoX block. The `eigh` change is
kept anyway — it is principled, and it is what makes the rank floor able to *detect*
and discard numerically-null directions rather than normalise roundoff to unit length.

Side effect of the trace fix: variance captured now reports **94.98%** for 97 of 149
components, instead of a vacuous 100%.

### 2026-08-26: augmentation noise scale, calibrated

`--noise_scale 1e-7` (the old default) is unusable: 1e-7 *is* float32's relative
precision, so the ensemble members sit at the representable limit and the Gram matrix
is roundoff. `calibrate_noise.py` measures the usable band directly — WikiText-2,
64 × 1024 tokens, ΔPPL as a function of `s` (job 9678992):

| `s` | gpt2_medium | smollm2_360m | pythia_410m |
|---|---|---|---|
| 1e-3 | +0.004% | −0.000% | −0.003% |
| 3e-3 | +0.016% | +0.016% | +0.057% |
| **1e-2** | **+0.097%** | **+0.250%** | **+0.956%** |
| 3e-2 | +0.699% | +2.378% | +9.633% |
| 1e-1 | +8.158% | +37.036% | +102.694% |

`s = 1e-2` is the largest scale keeping every family under +1%, and it sits ~5 orders
of magnitude above float32 epsilon so members separate cleanly in the Gram. It also
makes the `k=N/2` arm *measurable*: the truncation residual there is `s·σ/√2 ≈
7.1e-3·σ`, which lands between the 3e-3 and 1e-2 rows — so expect roughly +0.5% for
`pythia_410m`. At `k=N-1` reconstruction is exact by the rank bound, so cosine 1.0
and ΔPPL ≈ 0 there is the *expected* result, not a finding.

`pythia_410m` is consistently the most perturbation-sensitive family despite having
the *smallest* weight std (0.0287 vs 0.107 and 0.178) — so its fragility is about the
function, not the weight scale.

### 2026-08-26: whole-stack run — machinery works, the rank sweep does not

Job 9715570, 10 minutes end to end: 3 per-family PCA fits (N=100, D~302M), two
StackVAEs (k=99 and k=50), and 18 evaluations.

**Posterior collapse is gone, structurally.**

| | block-wise (2026-08-21) | whole-stack (2026-08-26) |
|---|---|---|
| total KL | 0.00106 nats/sample | **11.36** (k=99) / **10.70** (k=50) |
| active units | ~0 | **32/32** both arms |
| behaviour at β=1.0 | decayed to zero | stable |

No hyperparameter tuning was involved. `family_idx` over ~100 samples cannot act as
a primary key, so the decoder has no lookup table to fall back on.

**Reconstruction and generation both land within noise:**

| arch | k | pca_only Δ% | vae Δ% | generate Δ% |
|---|---|---|---|---|
| gpt2_medium | 99 | +0.000 | +0.001 | −0.011 |
| smollm2_360m | 99 | −0.000 | +0.088 | +0.112 |
| pythia_410m | 99 | −0.000 | +0.051 | +0.254 |
| gpt2_medium | 50 | −0.004 | +0.027 | −0.007 |
| smollm2_360m | 50 | −0.007 | −0.007 | +0.042 |
| pythia_410m | 50 | −0.008 | +0.044 | +0.091 |

`pca_only` at k=99 returns cosine **1.000000** for all three families — the rank
bound, exactly as predicted. Per-family PCA plus zero padding removes the failure the
shared block basis had (pythia_410m: 0.388 → 1.000).

**But the k=50 arm measures almost nothing, and that is a design flaw, not a result.**

The evaluation target is ensemble member 0, which is the real pretrained model. For a
noise ensemble the mean is `w_0 + O(s·σ/√N)`, so **sample 0 sits at the centre of the
ensemble** — its deviation from the mean is ~√N smaller than a typical member's.
Truncating its code therefore discards ~√N less. Concretely the residual is
`s·σ/√(2N) ≈ 7.1e-4·σ`, not the `s·σ/√2 ≈ 7.1e-3·σ` that applies to a typical member.
Cross-referencing the calibration table, 7.1e-4 predicts ΔPPL well under 0.01% —
which is what was measured (−0.008%).

So `x̂ ≈ mean ≈ w_0` **at any rank**, and the reconstruction metric cannot fail. Lowering
k further does not help, because the flat spectrum spreads sample 0's (already tiny)
deviation evenly across all components.

This sharpens the caveat at the top of this file. It is not merely that the basis
spans injected-noise directions — it is that the **evaluation target sits at the
ensemble's centre**, so faithful reconstruction is guaranteed by geometry rather than
earned. `eval_stack.py --sample_idx N` reconstructs a typical member instead, which is
the meaningful sweep.

**Consequence for the plan:** the machinery is verified (analytic predictions all
hold, KL healthy, write-back correct, generation runs), but a noise-augmented ensemble
cannot produce an informative rank sweep for the real model. That needs an ensemble
whose members are all genuine models with none at a privileged centre — i.e. **Pythia
checkpoint revisions** (N=143 complete models at fixed scale). That is now unblocked.

### 2026-08-26: the typical-member sweep — PCA truncation acts as a denoiser

`eval_stack.py --sample_idx 5` reconstructs a *typical* ensemble member rather than
sample 0 (job 9715671). Every number matches the analytic prediction:

| arch | k | relL2 | PPL(member 5) | PPL(recon) | Δ% |
|---|---|---|---|---|---|
| gpt2_medium | 99 | 1.98e-07 | 22.3695 | 22.3695 | +0.000 |
| smollm2_360m | 99 | 1.92e-07 | 12.9862 | 12.9862 | −0.000 |
| pythia_410m | 99 | 1.96e-07 | 18.9664 | 18.9664 | −0.000 |
| gpt2_medium | 50 | 7.369e-3 | 22.3695 | 22.3514 | **−0.081** |
| smollm2_360m | 50 | 7.442e-3 | 12.9862 | 12.9682 | **−0.138** |
| pythia_410m | 50 | 7.369e-3 | 18.9664 | 18.8669 | **−0.524** |

Two things are confirmed quantitatively:

1. **The truncation residual formula holds.** Predicted `s·σ/√2 = 7.07e-3`; measured
   7.37–7.44e-3 across all three families, agreement within 5%. And the √N gap
   between sample 0 (7e-4) and a typical member (7.4e-3) is exactly the factor the
   geometry predicts.
2. **ΔPPL at k=N/2 is NEGATIVE — truncation improves the model.** Not a bug. Member 5
   is `w_0 + noise`, i.e. a *degraded* model (pythia: 18.966 vs the real 18.756).
   Discarding half the ensemble variance discards half the injected noise, pulling the
   reconstruction back toward `w_0`. The arithmetic is exact: pythia recovers
   `18.9664 → 18.8669`, and `(18.9664 − 18.8669) / (18.9664 − 18.7561) = 0.47` — it
   removes **half** the noise-induced PPL penalty, precisely as discarding half the
   variance implies.

So **PCA truncation is a denoiser here**, and "reconstruction fidelity" and "model
quality" are *anti-correlated* at low rank. That makes the rank sweep uninterpretable
as a research result for either target: on sample 0 truncation cannot hurt (it sits at
the ensemble centre), and on a typical member truncation *helps* (it removes noise the
augmentation put there). Both are artifacts of an ensemble that is isotropic noise
around a single point.

### The VAE's error is not interchangeable with weight-space noise

At k=99 the PCA is exact (relL2 ~2e-7), so all residual error is the VAE's. It is
~2e-3 relative and costs `pythia_410m` **+0.378%** PPL. For comparison:

* an *isotropic* 2e-3 perturbation costs pythia ~+0.03% (calibration table) — **10×
  less**;
* the k=50 truncation error is 7.4e-3, i.e. **3.7× larger** than the VAE's, and it
  *improves* PPL.

So error magnitude in this space says little about functional damage. The VAE's error
is structured — concentrated in the leading principal directions, which are exactly
the ones that carry function — while isotropic noise and truncation error are not.
Worth knowing before putting a flow on top of these codes: a flow's sample quality
cannot be judged by code-space or weight-space L2 alone.

### Known issues

> **Status note.** The first three entries below are legacy-pipeline problems. The
> stack pipeline removes the mechanism behind all three: per-family bases mean no
> family competes for shared components, `--exclude_1d` keeps biases out of the
> basis, and whole-stack samples make dataset size a question about ensemble size
> rather than block count. They are kept because they explain the design change.
>
> For open problems in the **current** pipeline, read
> [RESEARCH_NOTES.md](RESEARCH_NOTES.md) section 7.

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
alternative (e.g. a LLaMA-family model). **Resolved as (a)**: `opt_350m` stays
registered for reference but is no longer in `run_hpc.py`'s default `--arch_list` —
pass it explicitly via `--arch_list` if you want it back. Note that the new
`pythia_160m`/`pythia_410m` entries carry the same per-projection biases as OPT, so
they carry the same reconstruction-quality risk; watch their cosine-sim/PPL numbers
on the next run.

**Dataset size** — With 108 blocks across 4 families, the VAE memorizes rather than
generalizes. `--val_fraction 0.0` (train on all data) is the correct setting at this
scale. Switching to `--val_fraction 0.1` becomes meaningful when the dataset grows to
500+ blocks, e.g. by adding OLMo-2 or Pythia checkpoint-series training snapshots
(EleutherAI publishes intermediate checkpoints throughout pretraining for every
Pythia size, alongside Olmo-2 and SmolLM — a cheap way to multiply block count
without adding new architectures).

**Gemma 3** — `google/gemma-3-270m` is a gated model, now included in
`run_hpc.py`'s default `--arch_list`. To use it: accept terms at
huggingface.co/google/gemma-3-270m, then `export HF_TOKEN=hf_...` in your
shell before `sbatch slurm_train.sh` (required for the default run now, not
just as an opt-in extra) — `sbatch` inherits it automatically, so it does not
need to be edited into the script. Its registry entry
(family_idx=4) has been in place for a while; this expansion is what actually
exercises the gated download path end-to-end for the first time.

## Multiple-Choice Benchmarks (MMLU / HellaSwag / GPQA)

`eval_mc.py` extends the reconstruction evaluation beyond WikiText-2 PPL to
downstream multiple-choice accuracy. It's opt-in — enable with `--eval_mc`
(`run_hpc.py`) or `MC_EVAL=1` (`slurm_train.sh`) — since it adds real runtime
and dataset downloads on top of the default pipeline.

All registered architectures are base pretrained models (no chat
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

### Stack pipeline

Read `runs/<name>/results/stack_eval_results*.json`.

| Metric | Target | Where |
|---|---|---|
| **ΔPPL, `pca_only`, k=N−1** | **exactly 0, cosine 1.000000** | stack_eval_results.json |
| **ΔPPL, `vae` arm** | this is the real gate — the VAE's own cost | stack_eval_results.json |
| relL2, `pca_only`, k=N−1 | < 1e-6 (rank bound) | stack_eval_results.json |
| relL2, `pca_only`, k=N/2 | ≈ `s/√2` for a typical member | stack_eval_results.json |
| Total KL | 1–20 nats/sample, stable after β reaches 1.0 | vae_k*/train_metrics.json |
| Active latent units | most of `latent_dim` | wandb `kl/active_units` |
| Spectrum `ev[0]/ev[k−2]` | < ~3 for a noise ensemble (flat) | pipeline_summary_k*.json |

Four rules for reading these numbers:

1. **`pca_only` at k=N−1 is a self-test, not a result.** The rank bound makes it exact.
   A value other than cosine 1.000000 means a bug.
2. **Gate on ΔPPL, never on cosine.** A measured cosine of 0.99939 came with +72,468%
   PPL. Cosine only becomes informative above about 0.9999.
3. **Do not compare `--sample_idx 0` against a nonzero index.** Sample 0 sits at the
   ensemble centre, so its truncation residual is smaller by √N.
4. **Error magnitude does not predict functional damage.** The VAE's 2e-3 error cost
   pythia_410m +0.378% PPL. An isotropic 2e-3 perturbation costs about +0.03%.

### Block pipeline (legacy)

| Metric | Target | File |
|---|---|---|
| PPL delta | < 1.0 ppl point per family | lm_eval_results.json |
| Block cosine similarity | > 0.9999 (0.999 is not sufficient) | reconstruction_results.json |
| Block MSE | < 1e-8 | reconstruction_results.json |
| MMLU/HellaSwag/GPQA acc delta | as close to 0 as possible | mc_eval_results.json |

---

## Adding a New Architecture

1. Add an entry to `ARCH_CONFIGS` in `models/registry.py`:
   - `default_model_id`, `hf_model_type`, `layers_attr`, `family_idx`, `tiny_config`
2. Update `N_FAMILIES` and `MAX_BLOCKS` if needed.
3. No other changes required — block extraction uses `block.named_parameters()` and is architecture-agnostic.

> **Gemma 3 note:** `google/gemma-3-270m` is a multimodal model. Its text decoder blocks live at `model.language_model.layers` (already set in the registry), not `model.layers`. If the model ID is unavailable on HuggingFace, substitute the nearest available variant (e.g., `google/gemma-3-1b-pt`).

---
### Next steps

Next steps and the reasoning behind them live in **[RESEARCH_NOTES.md](RESEARCH_NOTES.md)**.
Read that file before you start work. It records:

- what the machinery can and cannot currently measure, and why;
- the next experiment, with the artifacts that would prove or disprove it;
- **17 recorded missteps**, so nobody repeats them.

The single most important item: the noise-augmented ensemble validates machinery but
cannot answer a research question. The next experiment needs an ensemble of genuinely
different complete models. Pythia publishes about 143 checkpoint revisions per size,
which gives that at fixed model scale.

---
## References

- Dual/Gram-matrix PCA algorithm: [NNeuralDynamics/DeepWeightFlow](https://github.com/NNeuralDynamics/DeepWeightFlow)
- Related weight-space work: [ScottBiggs2/DeepWeightFlow-Revisions](https://github.com/ScottBiggs2/DeepWeightFlow-Revisions), [ScottBiggs2/SDAF](https://github.com/ScottBiggs2/SDAF)
