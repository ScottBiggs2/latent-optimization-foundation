# llmzoo — generative modelling over LLM weight space

Dual (Gram-matrix) PCA over whole decoder stacks, then a conditional rectified flow
(and, as an ablation, a conditioned VAE) over the PCA codes. The goal is a code space
a generative model can sample to produce working weights.

**One PCA sample is one complete decoder stack.** Each architecture gets its own PCA
basis; the generative models condition on the model family. This follows
[DeepWeightFlow](https://arxiv.org/abs/2601.05052) (ICLR 2026).

The pipeline runs end to end, versioned and reloadable, and has produced a
pre-registered **null result** on a manufactured ensemble — the machinery is
validated; the ensemble is what cannot answer a research question. Building a real
zoo is the current work.

**Start here:**

| you want | read |
|---|---|
| what we are building and why | [RESEARCH_PLAN.md](RESEARCH_PLAN.md) — authoritative |
| what will bite you | [CLAUDE.md](CLAUDE.md) — the short version |
| the 21 recorded missteps | [docs/RESEARCH_NOTES.md](docs/RESEARCH_NOTES.md) §5 |
| how the last sprint ended | [docs/HANDOFF.md](docs/HANDOFF.md) |

---

## Design

The project previously treated one transformer block as one PCA sample, with a single
shared basis across families. That pipeline has been removed; `git log` has it. The
change is worth understanding, because two long-standing problems dissolved rather
than needing mitigation.

| | removed (block-wise) | current (whole-stack) |
|---|---|---|
| one PCA sample | one transformer block | one whole decoder stack |
| `N` | number of blocks (150) | ensemble size per family (~100) |
| `k` | 97, capped by block count | `N-1` (rank bound) or `N/2` |
| basis | **one shared**, all families | **one per family** |
| padding | zero-pad to `max_block_size` | **none** — stacks are fixed-length |
| conditioning | `(block_idx, family_idx)` | `family_idx` only |

* **Padding disappears.** Every family has uniform block sizes (verified on real
  configs), so a stack is a fixed length and no mask is needed anywhere.
* **Posterior collapse stops being reachable.** `(family_idx, block_idx)` was a
  *unique key* over the block dataset, so a conditioned decoder could memorise the
  training set and ignore `z` — the measured cause of a 0.001 nats/sample collapse.
  With ~100 samples sharing one `family_idx`, conditioning cannot identify a sample.

### Why the basis is never materialised

At `k=99` and `D=302M` an explicit component matrix is **119.6 GB per family**. The
dual formulation exists so it is never needed:

* training codes are exactly `U·√S` from the Gram eigendecomposition — **zero passes
  over the data** (verified to 1.2e-15 against an explicit `Xc @ components`);
* reconstruction is `x̂ = mean + Xcᵀa` with `a = U_k(z/√S_k)`, i.e. a weighted sum of
  ensemble members — one streaming pass.

`DualGramPCA` stores the mean plus a 100×100 eigenbasis instead: ~1.2 GB per family.

The cost, stated up front: decoding a generated code **requires the whole training
ensemble on disk**. Materialising an explicit rank-k basis (64 GB at 100M/k=99) would
buy a free-standing generator. DeepWeightFlow's dual PCA has the same property and
does not mention it.

### The honest caveat

DeepWeightFlow uses ~100 *independently trained* networks. We have **one** pretrained
model per family, so the current ensemble is manufactured as `w_0 + s·σ·ε_i` (sample 0
is the real model, exactly). An isotropic ensemble spans injected-noise directions, so
**this validates machinery, not weight-space structure.** Its spectrum is flat
(`ev0/median = 1.001`) and a flow provably buys nothing over a Gaussian on it — which
is the pre-registered null, and the flat end of a calibration curve rather than a
disappointment. `RESEARCH_PLAN.md` §4 is the plan for a real zoo.

Permutation augmentation is *not* a fix: generating permuted copies deliberately adds
the permutation orbit, whereas TransFusion (DeepWeightFlow's canonicalizer) exists to
*remove* it. Permuted copies are functionally identical to the original, so a model
trained on them regenerates the same network — a sanity harness, not new structure.

---

## Repository layout

```
LLM-VAE-Early/
├── RESEARCH_PLAN.md            the plan. Read first.
├── CLAUDE.md                   the traps, distilled
├── pyproject.toml              `pip install -e .`  (torch NOT declared — see below)
│
├── src/llmzoo/                 the library
│   ├── data/ensemble.py        EnsembleDataset: per-family ensembles of whole stacks
│   ├── pca/gram.py             DualGramPCA: never builds the (k, D) basis
│   ├── gen/vae.py              StackVAE: family conditioning, per-family code norm
│   ├── gen/flow.py             conditional rectified flow, codes OR VAE-latent space
│   ├── eval/core.py            perplexity + multiple-choice scoring primitives
│   ├── artifacts/io.py         leaf: version gates, fingerprints, atomic writes
│   ├── artifacts/bundle.py     CodeStats + load_run(): reconstitute a whole run
│   ├── models/registry.py      ARCH_CONFIGS, family_idx, tiny mocks
│   ├── models/weight_extractor.py   stack flatten / reconstruct, exclude_1d
│   └── data/{val,mc}_loader.py WikiText-2 · MMLU / HellaSwag / GPQA
│
├── scripts/                    the CLIs
│   ├── train_stack.py          ensembles → per-family PCA → codes → StackVAE
│   ├── train_flow.py           trains a flow from the 40 KB codes artifact alone
│   ├── eval_stack.py           eight arms, any rank, optional benchmark axis
│   ├── report_stack.py         arch × k × arm as markdown — PURE STDLIB, keep it so
│   ├── calibrate_noise.py      measures the usable augmentation-noise band
│   ├── diag_flow_dispersion.py do generative arms have the RIGHT SPREAD? (misstep 19)
│   └── diag_flow_capacity.py   is flow contraction under-fitting, or over-fitting?
│
├── slurm/                      AICR job scripts
│   ├── aicr_env.sh             shared header, sourced by every job
│   ├── setup_env_aicr.sh       one-time conda env (cu128 — B200 needs sm_100)
│   ├── stack_run.sbatch        ensembles + PCA + VAEs at both ranks
│   ├── flow_run.sbatch         flows both spaces + all-arm ΔPPL + report
│   ├── stack_bench.sbatch      MMLU / HellaSwag / GPQA, eval only
│   ├── {stack,flow}_smoke.sbatch   tiny-mode end-to-end, b200-devel
│   ├── tests.sbatch            one test module on the cpu partition
│   └── calibrate_noise.sbatch
│
├── tests/                      bare scripts, not pytest — `python tests/<name>.py`
│   ├── test_ensemble.py        ensemble + DualGramPCA vs brute force
│   ├── test_bundle.py          fingerprints, CodeStats, save/load
│   ├── test_flow.py            interpolant, Euler, spaces, save/load
│   └── test_report.py          reporting; pure stdlib, the only one that runs locally
│
└── docs/                       archive
    ├── RESEARCH_NOTES.md       §5 missteps and §6 environment stay authoritative
    ├── HANDOFF.md              close-out of the CFM sprint, 2026-08-31 → 09-04
    └── dwf_authors_qa_2026-09-06.md   primary source on DeepWeightFlow's zoo
```

`src/llmzoo/__init__.py` is empty on purpose: `import llmzoo` must not pull in numpy
or torch, which is what keeps `report_stack.py` renderable on a CPU node and on a
laptop. A test enforces it.

---
## Pipeline stages

`scripts/train_stack.py` runs four stages. Each skips work that already exists on disk.

```
Stage 1 ─ Ensembles                                    (llmzoo/data/ensemble.py)
   Load one model per architecture.
   Flatten the WHOLE network: [extra | block_0 ... block_{L-1}], where `extra` is
   the embeddings, the final norm and the LM head. The extra segment is derived as
   the COMPLEMENT of the layers_attr subtree in named_parameters(), so it needs no
   per-arch config and weight tying is handled free (a tied lm_head/wte pair is
   yielded once). Pass --no_include_extra for a decoder-only run.
   Check that all blocks in the family have the same size. Fail if they do not.
   Save w_0 only. Members 1..N-1 exist as a seed, never as a file.
   ↓
Stage 2 ─ Per-family Gram PCA at k = N-1               (llmzoo/pca/gram.py)
   Pass 1: accumulate the mean over parameter chunks.
   Pass 2: accumulate the centered Gram C in float64.
   eigh on C. Drop eigenvalues below the rank floor.
   Codes are U*sqrt(S). No pass over the data. No (k, D) basis.
   ↓
Stage 3 ─ Codes at rank k                              (llmzoo/artifacts/bundle.py)
   Take the first k columns. A k-prefix IS the rank-k PCA.
   Record each family's own code mean and scale, and SEAL them to
   runs/<run>/codes_k<k>/ with a content fingerprint.
   ↓
Stage 4 ─ StackVAE                                     (llmzoo/gen/vae.py)
   Conditioned on family_idx only.
   Free bits, conditioning dropout, cosine LR, warmup-gated checkpointing.
   Sealed once at the end as vae_weights.pt + vae_meta.json.
```

```
Stage 5 ─ Conditional rectified flow                   (scripts/train_flow.py)
   A SEPARATE job, in either of two interchangeable target spaces:
     codes   dim = k   target = per-family-normalized PCA codes
     latent  dim = 32  target = the StackVAE posterior mean mu(codes)
   x0 ~ N(0, I); x_t = (1-t)x0 + t*x1; target u = x1 - x0; MSE; Euler at sample
   time. Family conditioning follows StackVAE (embedding + learned null_cond +
   conditioning dropout), so classifier-free guidance is available -- DeepWeightFlow
   has no unconditional branch and therefore no CFG.

   train_flow.py reads ONLY runs/<run>/codes_k<k>/ (~40 KB) plus vae_k<k>/ for the
   latent space. It never constructs an EnsembleDataset or touches DualGramPCA.
   That is what makes this stage re-runnable on a different ensemble source with no
   change to flow code: swap the source, re-run stages 1-3, and this stage is
   untouched and unaware.
```

Fit Stage 2 once. Every lower rank is a column prefix of that fit, so a rank sweep
costs nothing extra.

`ARCH_LIST` is the only change needed for the six-family follow-up: `N_FAMILIES` is
already 6.

---

## Running it on AICR

Compute is AICR (`ssh aicr`, account `p2026_0038_neu`). Explorer is retired. The
`aicr-cluster` skill has the queue mechanics; the short version is that nodes are
shared, so every script here asks for **one** GPU and an honest `--time` and starts
immediately rather than waiting for a node to drain.

```bash
# once
rsync -az --exclude .git --exclude __pycache__ ./ \
      aicr:/work/neu/p2026_0038_neu/$USER/llm-vae/
ssh aicr 'bash /work/neu/p2026_0038_neu/$USER/llm-vae/slurm/setup_env_aicr.sh'

# then
sbatch slurm/stack_smoke.sbatch            # tiny mode, ~1 min, no downloads
sbatch slurm/flow_smoke.sbatch             # tiny mode WITH flows + all 8 arms
RUN=emb3 sbatch slurm/stack_run.sbatch     # ensembles + PCA + VAEs, k=N-1 and k=N/2
RUN=emb3 sbatch slurm/flow_run.sbatch      # flows both spaces + all-arm ΔPPL + report
SAMPLE_IDX=5 RUN=emb3 sbatch slurm/flow_run.sbatch   # the typical-member target
RUN=emb3 sbatch slurm/stack_bench.sbatch             # MMLU / HellaSwag / GPQA

TEST=tests/test_ensemble.py sbatch slurm/tests.sbatch
python tests/test_report.py                # the one test that runs on a laptop
```

`flow_run.sbatch` SKIPS a flow that is already sealed rather than retraining it, so
running it twice with different `SAMPLE_IDX` trains the flows once and evaluates
twice. `--out_suffix` keeps the benchmark job's results from overwriting the ΔPPL
job's at the same `--sample_idx`.

**Storage.** Run artifacts go to `$ARTIFACT_DIR`
(`/work/neu/p2026_0038_neu/$USER/llm_vae` — 1.0 TB, persistent). Caches, logs and
wandb go to `/scratch/$USER` — 10 TiB, but **purged at 30 days**, so nothing there is
allowed to be irreplaceable. Nothing goes in `$HOME`: the quota is 100 GiB and filling
it fails jobs in about a second, with an opaque exit code.

**torch is deliberately not a declared dependency.** AICR's B200s are Blackwell /
sm_100 and need a cu128 wheel, which is not on the default PyPI index. If torch were
in `pyproject.toml`, `pip install -e .` could quietly replace a correct cu128 install
and the failure would not surface until the first CUDA kernel.
`slurm/setup_env_aicr.sh` installs torch first, from the cu128 index, then asserts the
compiled kernel set contains sm_100.

---
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
| `code_rms_ratio`, generative arms | ≈ 1.0; below 0.8 is collapse | stack_eval_results*.json |
| Spectrum `ev0/median` | ≈ 1 for a noise ensemble (flat) | pipeline_summary_k*.json |
| Spectrum effective-rank ratio | ≈ 1 for a noise ensemble (flat) | pipeline_summary_k*.json |
| ODE round-trip `rel_l2` | must fall like 1/n_steps | `flow_rt_*` arms |

**The eight arms.** All eight share one code path: write the reconstruction into the
live model, measure, restore.

| arm | path | what it isolates |
|---|---|---|
| `pca_only` | project + inverse_transform | basis capacity (self-test at k=N−1) |
| `vae` | encode + decode the codes | the VAE's own cost |
| `generate` | z ~ N(0,I) → `vae.decode_cfg` | VAE prior sample |
| `gauss_codes` | z ~ N(mean_f, std_f) from `code_stats` | **the null model the flows must beat** |
| `flow_codes` | flow sample in code space | flow over PCA codes |
| `flow_latent` | flow sample in VAE latent space | flow over VAE latents |
| `flow_rt_codes` | encode → reverse ODE → forward ODE | ODE consistency, codes |
| `flow_rt_latent` | same, latent space | ODE consistency, latent |

**`gauss_codes` is the arm that decides whether any `flow_codes` number means
anything.** On a manufactured ensemble the per-family code distribution is
near-Gaussian by construction, so `flow_codes` vs `pca_only` carries no information
about the flow; `flow_codes` vs `gauss_codes` does. It costs one `inverse_transform`
and zero training.

> **ΔPPL alone is NOT sufficient for a generative arm, and `gauss_codes` cannot fix
> that.** The ensemble mean is essentially `w_0`, so a collapsed generator scores
> ΔPPL ≈ 0 — beating an honest sample — while generating nothing. On `emb3`,
> `flow_codes` beat `gauss_codes` by 43× on ΔPPL while emitting codes at **0.25× the
> correct RMS**. Always run `diag_flow_dispersion.py` before reading a generative
> ΔPPL. See RESEARCH_NOTES misstep 19.

Note that `--guidance_scale` and `--flow_guidance_scale` are DIFFERENT mechanisms —
CFG on the VAE decoder and CFG on the velocity field. `flow_latent` applies both.
Multiplying them together is meaningless.

Five rules for reading these numbers:

1. **`pca_only` at k=N−1 is a self-test, not a result.** The rank bound makes it exact.
   A value other than cosine 1.000000 means a bug.
2. **Gate on ΔPPL, never on cosine.** A measured cosine of 0.99939 came with +72,468%
   PPL. Cosine only becomes informative above about 0.9999.
3. **Do not compare `--sample_idx 0` against a nonzero index.** Sample 0 sits at the
   ensemble centre, so its truncation residual is smaller by √N.
4. **Error magnitude does not predict functional damage.** The VAE's 2e-3 error cost
   pythia_410m +0.378% PPL. An isotropic 2e-3 perturbation costs about +0.03%.
5. **Read `ev0/median`, not `ev[0]/ev[k−1]`.** At the rank bound the latter describes
   the smallest direction surviving the rank floor, not the bulk: it reads 12.09 at
   k=N−1 and 1.006 at k=N/2 on the *same* pure-noise ensemble. Same trap as
   `total_variance_captured` being vacuous at k=N−1 (missteps 15 and 15b).

---

## Adding a New Architecture

1. Add an entry to `ARCH_CONFIGS` in `src/llmzoo/models/registry.py`:
   - `default_model_id`, `hf_model_type`, `layers_attr`, `family_idx`, `tiny_config`
2. Update `N_FAMILIES` and `MAX_BLOCKS` if needed.
3. No other changes required — block extraction uses `block.named_parameters()` and is architecture-agnostic.

> **Gemma 3 note:** `google/gemma-3-270m` is a multimodal model. Its text decoder blocks live at `model.language_model.layers` (already set in the registry), not `model.layers`. If the model ID is unavailable on HuggingFace, substitute the nearest available variant (e.g., `google/gemma-3-1b-pt`).

---

## Next steps

They live in **[RESEARCH_PLAN.md](RESEARCH_PLAN.md)**, which supersedes §4 and §7 of
the research notes. The single most important item: the noise-augmented ensemble
validates machinery but cannot answer a research question. The next experiment needs
an ensemble of genuinely different complete models — the plan is to branch ~100 of
them off one shared partially-trained trunk.

---

## References

- Dual/Gram-matrix PCA algorithm: [NNeuralDynamics/DeepWeightFlow](https://github.com/NNeuralDynamics/DeepWeightFlow)
- Related weight-space work: [ScottBiggs2/DeepWeightFlow-Revisions](https://github.com/ScottBiggs2/DeepWeightFlow-Revisions), [ScottBiggs2/SDAF](https://github.com/ScottBiggs2/SDAF)
