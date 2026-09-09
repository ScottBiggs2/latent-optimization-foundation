# Research Plan — a zoo of language models

Written 2026-09-06. Revised 2026-09-08 for the `llmzoo` refactor, exclusive AICR
compute, and a fixed GPT-2 architecture.

Read alongside `CLAUDE.md` (the traps, distilled) and `docs/RESEARCH_NOTES.md` §5–6
(the 21 recorded missteps and the environment facts). This file is the plan; those two
are the guardrails.

Audience: Scott, Bedionita, anyone joining, and any agent picking this up cold.
Target is ICLR 2027 — abstract 18 Sep 2026, paper 25 Sep 2026.

---

## 0. Summary

We have a working generative pipeline over LLM weights and no real data for it. The
data is cheaper than it looked. Branch ~100 GPT-2 models off one shared,
partially-trained trunk: about 3× less compute than independent training, every model
stays in one basin so canonicalization is unnecessary rather than merely assumed
unnecessary, and a mediocre checkpoint becomes useful as a launch pad instead of
harmful as a sample.

Four contributions, three of which need no new science:

1. A zoo of independently-trained language models, and the first eigenvalue spectra
   reported for any weight-space zoo.
2. Conditional generation across pretraining data mixtures, conditioned on the mixture
   simplex itself so an unseen mixture is a real zero-shot test.
3. The evaluation instrument: a null arm, a dispersion statistic, bulk spectrum
   statistics, and a retrieval baseline.
4. *(stretch)* Fine-tuning inside the zoo basis — ~100 trainable parameters adapting a
   whole model.

The full ladder (GPT-2 Mini 51M / Small 124M / Medium 355M, N=100 each) is
**≈219 GPU-hr and ~215 GB**. On AICR's 32-GPU ceiling, submitted as 1-GPU array jobs,
that is ~9 minutes, ~55 minutes and ~7.5 hours of wall clock respectively. Compute is
not the bottleneck. The trainer and the remaining refactor are.

---

## 1. Claim

> Every weight-space model zoo in the literature is a zoo of task-specific networks
> trained from scratch on one narrow objective. Nobody has a zoo of language models, so
> nobody knows whether weight-space structure survives at LM scale, on a general
> objective, or how it changes as models grow.

The ICML 2026 position paper ([2605.18632](https://arxiv.org/abs/2605.18632)) asserts
that "high-performing models occupy low-dimensional, highly structured regions of weight
space." That is a premise, not a measurement. Nobody has checked it, us included.

---

## 2. What already exists

The repo was repackaged on 2026-09-07 (commits `ea36ba1`…`cb899d3`). Current layout:

```
src/llmzoo/     library   artifacts/{io,bundle}  pca/gram  gen/{flow,vae}
                          eval/core  data/{ensemble,val_loader,mc_loader}
                          models/{registry,weight_extractor}
scripts/        CLIs      train_stack  train_flow  eval_stack  report_stack
                          calibrate_noise  diag_flow_{dispersion,capacity}
slurm/          AICR      *.sbatch + aicr_env.sh + setup_env_aicr.sh
tests/          bare scripts, not pytest — python tests/<name>.py
docs/           archive   RESEARCH_NOTES  HANDOFF  dwf_authors_qa_2026-09-06
```

| asset | state |
|---|---|
| ensembles → per-family Gram PCA → StackVAE → conditional rectified flow → 8 eval arms → report | end to end, versioned, fingerprinted, reloadable |
| automated checks across 4 test modules | green |
| null arm `gauss_codes` (`z ~ N(mean_f, std_f)`) | built; decides whether any flow number means anything |
| dispersion diagnostic (`code_rms_ratio`, `scripts/diag_flow_dispersion.py`) | built; catches what ΔPPL cannot see |
| bulk spectrum statistics (`ev0/median`, effective-rank ratio) | sealed into every artifact |
| pre-registered null on a manufactured ensemble | flow ≡ Gaussian to 0.015 pp in all six cells |
| CFG on the velocity field and on the VAE decoder | two independent mechanisms |
| whole-stack layout including embeddings and LM head | `ENSEMBLE_LAYOUT_VERSION = 2` |
| block pipeline | **deleted** (`9ce64f3`), not archived — see `git log` if ever needed |
| 21 recorded missteps | `docs/RESEARCH_NOTES.md` §5 |

The null result is an asset, not a consolation prize: it is the flat endpoint of the
calibration in §4.2, and it was pre-registered in a job header before it ran.

### 2.1 What the refactor did and did not do

It was packaging-deep, not semantics-deep. Useful to know before planning around it:

**Landed.** Installable `llmzoo` package with `scripts/` entry points; block pipeline
deleted; tests moved to `tests/` with a real stdlib-purity check on `report_stack.py`;
every sbatch retargeted from Explorer to AICR; `CLAUDE.md` written.

**Landed 2026-09-08** (Phase 0 partial, plus the new zoo path):

| item | what changed |
|---|---|
| **0.2 — conditioning size decoupled from the registry** | `N_COND_SLOTS = 32`, fixed. Was `len(ARCH_CONFIGS)`, so registering the three zoo archs would have moved every embedding table 6 → 9 and invalidated every checkpoint. `N_FAMILIES` is now an alias for the constant; `N_ARCHS` is the registry size. |
| **0.3 — `MemberSource` abstraction** | `NoiseSource` / `ZooSource` behind one `load_rows`. `EnsembleDataset(source=, zoo_dir=)`. **Acceptance verified:** a pre-0.3 meta and an explicit `source="noise"` both hash to `cdbb077e838a543f`; `source="zoo"` hashes differently. |
| **0.6 — `exclude_1d` decided** | `False` for zoo runs (§2.2). |
| GPT-2 zoo architectures | `gpt2_zoo_{mini,small,medium}` in the registry, plus `zoo_config()`, `build_zoo_model()`, `zoo_param_count()`. `load_model()` now refuses a `from_scratch` arch instead of 404ing on the hub. |
| the zoo itself | `data/mixtures.py`, `scripts/train_zoo.py`, `scripts/eval_domains.py`, `slurm/zoo_{trunk,branch}.sbatch`, `tests/test_zoo.py` |

**Still not landed**, deliberately deferred — none of it blocks the β calibration:

| item | current state | needed by |
|---|---|---|
| **0.1 — `family_idx` means four things** | 150 references across 13 files; heaviest in `gen/flow.py` (44), `gen/vae.py` (28), `artifacts/bundle.py` (25) | Phase 3 (π-conditioning). Do it while Phase 2 zoos run. |
| **0.4 — `eval_stack.py` monolithic** | 36.5 KB, eight arms inline | Phase 4 |
| **0.5 — `--k` asymmetric** | scalar in `train_stack`/`train_flow`, variadic in `eval_stack` | nothing; comfort |

> **Sequencing correction.** An earlier draft put all of Phase 0 ahead of Phase 1.
> That was wrong: the β calibration needs 0.2, 0.3 and 0.6 and nothing else. 0.1 is
> a Phase 3 prerequisite and is better done *while the zoos are training* than
> before they start, because it is the change most likely to need a cluster
> round-trip to debug and the zoos do not depend on it.

### 2.2 Two defects to decide on, not discover

**The VAE decoder contracts.** `generate` (z ~ N(0,I) → `decode_cfg`, no flow involved)
emits codes at 0.59–0.64× the correct RMS at every rank in every run, including
alongside a healthy flow. That is why `flow_latent` is contracted where `flow_codes` is
not. Stays off the critical path — the sprint runs PCA → Flow directly, as
DeepWeightFlow does, and the VAE is an ablation arm (§6.6).

**`exclude_1d` — decided 2026-09-08: `False` for zoo runs.** The flag omits 1-D
tensors — every projection bias and every LayerNorm gain — from `D`, so they keep
member 0's values on write-back. Harmless for a noise ensemble where all members share
biases by construction; wrong for a zoo, where every member has different biases and a
generated model would silently inherit member 0's. Including them costs `13·d` per
layer plus `2·d`, which is **~0.1% of `D`** at every scale — far less than the risk.
`scripts/train_zoo.py --exclude_1d` therefore defaults to `false`, and
`EnsembleDataset` refuses a zoo whose `zoo_meta.json` disagrees with the flag it was
handed.

One consequence to remember: with 1-D tensors included, `weight_std` is computed over
a vector mixing LayerNorm gains (≈1.0) with weights (≈0.02), so it is inflated and
meaningless. It only ever scaled augmentation noise, and a zoo has none — `train_zoo`
records it for schema uniformity and nothing reads it.

---

## 3. Landscape

| capability | who | scale | data |
|---|---|---|---|
| adapter generation from text | Text-to-LoRA, Doc-to-LoRA, Tina | Llama-3.1-8B | adapters, not full weights |
| mid-scale full-weight vision | RPG | ~200M ConvNeXt / ViT | vision zoos |
| full-weight transformer | DeepWeightFlow (ICLR 2026) | BERT-118M | 100 models trained from scratch, unique seeds, Yelp regression |
| frontier-scale full synthesis | — | open | — |
| full-weight pretrained causal LM | nobody | — | — |

Confirmed with the DWF authors (`docs/dwf_authors_qa_2026-09-06.md`): the BERT-118M
models were trained *ab initio* on Yelp with distinct random inits, N=100, k=N−1,
TransFusion applied. BERT was chosen over GPT-2 because its encoder blocks match ViT's,
so the TransFusion code transferred unmodified. They are task-specific regressors with
BERT's architecture, not language models.

Two further openings:

- DWF reports no eigenvalue spectrum and no variance-captured statistic anywhere. Nobody
  looked. We log two bulk statistics that do not lie at the rank bound (misstep 15b).
- DWF Appendix K (multi-class generation) was a late addition — no CFG, single-level
  feature injection, a 3-layer MLP — with deliberately understated prose to avoid
  opening a new contribution during revisions. Its tables are promising. Every ingredient
  it lacked exists in this repo.

We are extending DeepWeightFlow, not competing with it, and checking one of its
conclusions from the inside (§5).

---

## 4. Contribution 1 — the zoo

### 4.1 Design: branch off a shared trunk

All N models descend from one initialization and one shared partially-trained trunk,
then diverge on different data mixtures for the final fraction β of training.

1. **Compute.** Independent training costs `N·F`. Branching costs `(1−β)F + N·β·F`. At
   β = 0.3 that is a 3.3× saving.
2. **No canonicalization.** Models sharing an init and a burn-in stay linearly mode
   connected ([Frankle et al.](https://arxiv.org/abs/1912.05671)), so permutation
   symmetry is broken by construction. We skip TransFusion because it is unnecessary,
   not because we assume it is — which matters, see §5. Building TransFusion for
   decoder-only blocks would be real new code, and it is slow to run.
3. **Checkpoint quality stops being a problem.** A mediocre checkpoint is a bad sample
   and a fine launch pad. Every terminal model in the zoo is fully trained.

Scope limit, stated up front rather than conceded under review: this is a local weight
manifold around one trunk, not the global weight distribution. Same scope as
Text-to-LoRA, and it is the regime that matters for adaptation.

**Decided: we train our own trunk at all three scales.** Public checkpoints are free but
ship no optimizer state, so every branch would restart Adam's moments and eat a
transient — uniform across branches and so probably harmless, but an uncontrolled
confound. Our own trunk costs 0.09 / 0.53 / 4.35 GPU-hr, which is negligible, and buys
full control of data order plus no contamination question about held-out mixtures.

### 4.2 β: a calibration step, not a headline sweep

β sets how far branches diverge. The two limits behave differently:

- **β → 0** (branch very late): branches nearly identical, deltas dominated by gradient
  noise. Degenerates toward `w_0 + noise` — the manufactured ensemble we already
  measured, spectrum flat at `ev0/median = 1.001`, effective-rank ratio 0.990, flow
  provably no better than a Gaussian.
- **β → 1** (branch at init): effectively independent training. Frankle's instability
  analysis says branches with no shared burn-in land in different basins.

The first limit is solid. **The second is weaker than an earlier draft of this document
claimed** — models sharing an init but no burn-in end up far apart, which is not the same
as mutually orthogonal, and if mixture structure dominates basin structure the spectrum
could stay informative. "Both endpoints are flat, therefore there is an interior optimum"
is a plausible hypothesis, not a result to build the paper on.

The costs settle how much to spend finding out:

| β protocol | Mini 51M | Small 124M | Medium 355M |
|---|---|---|---|
| full sweep, N=100, β ∈ {.05, .15, .30, .60} | 14.6 GPU-hr | 85.6 | **701** |
| one zoo at β = 0.30, N=100 | 4.0 | 23.3 | 191 |
| **calibration: N=12, β ∈ {.15, .30, .60}, Mini only** | **1.9 GPU-hr** | — | — |

A full sweep at Medium is indefensible. A calibration at Mini costs **less than half a
single Mini zoo**, and β has to be chosen somehow regardless.

**Decided: run the reduced calibration, report it as a methods figure, drop the full
sweep.** Twelve branches gives k=11, enough to see whether a spectrum is flat, and §4.3
gives a cheaper criterion anyway. If the calibration happens to show a clean interior
optimum, promote it to a result then — on evidence, not on the strength of an argument
made too confidently.

**What the calibration produces**, concretely — this is Phase 1.2:

| output | used by |
|---|---|
| per-domain PPL spread at each β (§4.3 precondition) | picks β; gates Phase 2 |
| `ev0/median` and `effective_rank_ratio` at k=11 for each β | sanity check on §4.4 |
| wall-clock and MFU measurement at Mini scale | recalibrates every estimate in §4.5 |
| a working `train_zoo.py` exercised end to end at N=12 | de-risks the N=100 runs |

### 4.3 Choosing β, and the precondition that gates everything

Before fitting any PCA, measure per-domain perplexity across the zoo and confirm the
mixture axis is visible. It is a loop over N models and costs minutes on `rtx-batch`.

If β is too small, every branch is essentially the trunk, per-domain performance barely
moves, and the spectrum, the conditioning and the slope figure in §6.5 all come out flat
for reasons that have nothing to do with weight space.

The operational rule: **pick the smallest β at which mixture identity is measurable in
the models themselves.** Below that it is not a zoo, it is the noise ensemble with extra
steps.

Concretely, the pass condition: for each anchor mixture, the model trained on it should
be best-in-zoo on its own dominant domain, and the per-domain PPL gap between the best
and worst anchor on a given domain should exceed the within-anchor spread across its 4
branches. If it does not, raise β and re-run — do not proceed and do not reframe.

### 4.4 Pre-registered prediction

Write this into the job header before anything runs, as we did for the flow null.

> With C distinct mixtures across N models, between-mixture variance dominates
> within-mixture variance, so the spectrum shows a small number of large eigenvalues over
> a noise floor.
>
> Predict `effective_rank_ratio` ≈ 0.2–0.3 and `ev0/median` ≫ 1 at β = 0.3.
>
> Disproved by a flat spectrum (ratio → 1.0, `ev0/median` → 1.0), which would mean
> branches drift into near-orthogonal directions and no generative model over these codes
> will generalize.
>
> Gate on the bulk statistics. `ev[0]/ev[k−1]` reads 12.09 on data whose true ratio is
> 1.01 (misstep 15b).

### 4.5 The models: GPT-2, as exactly as possible

Architecture is GPT-2 (Radford, Wu, Child, Luan, Amodei & Sutskever, 2019, *Language
Models are Unsupervised Multitask Learners*, OpenAI technical report; reference
implementation [openai/gpt-2](https://github.com/openai/gpt-2)). Hold every architectural
choice fixed at the published one:

- decoder-only transformer, **pre-LayerNorm** (LN moved to the input of each sub-block),
  with an **additional LN after the final self-attention block**;
- **learned absolute position embeddings**, `n_positions = 1024`;
- GELU activation, `n_inner = 4 · n_embd`;
- **biases on every projection** (`Conv1D`, i.e. weights stored transposed relative to
  `nn.Linear`) — see §2.2, this interacts with `exclude_1d`;
- **tied** `wte` / `lm_head`; `named_parameters()` dedups, so the tied pair enters `D`
  once;
- BPE vocab **50257**;
- residual-layer weights scaled by **1/√N** at initialization, N = number of residual
  layers.

Use `hf_model_type: "gpt2"` and `GPT2LMHeadModel` from a config, so the implementation is
HF's rather than ours. `models/registry.py` already has the `gpt2_medium` entry
(`layers_attr: "transformer.h"`) and a `tiny_config` in GPT-2's `n_embd`/`n_layer`/
`n_head` naming; the three zoo configs slot in beside it.

| name | L | `n_embd` | heads | block params | embed params | **total** | embed % |
|---|---|---|---|---|---|---|---|
| **Mini** *(ours, no published equivalent)* | 8 | 512 | 8 | 25.2 M | 26.3 M | **51.5 M** | 51.0% |
| **Small** *(= published GPT-2 117M/124M)* | 12 | 768 | 12 | 85.1 M | 39.4 M | **124.4 M** | 31.6% |
| **Medium** *(= published GPT-2 345M/355M)* | 24 | 1024 | 16 | 302.3 M | 52.5 M | **354.8 M** | 14.8% |

Two of the three are exactly published GPT-2 configurations, which is worth stating in
the paper. Mini has no published equivalent and should be labelled as ours.

> **Keep the vocab fixed at 50257 across the whole ladder.** Shrinking it for Mini would
> make the embedding fraction more comfortable but would change the tokenizer, and a
> scaling comparison across different tokenizers is not a scaling comparison.
>
> **The confound to declare:** with `V` fixed, the embedding share of `D` falls 51% → 32%
> → 15% across the ladder. The *composition* of `D` changes, not only its size. State
> this next to the scaling figure; do not let a reviewer find it. Reporting the spectrum
> for the block-only sub-vector alongside the whole-stack one costs nothing and largely
> answers it.

### 4.6 Compute, sized for AICR

AICR is the only cluster; Explorer is retired. Chinchilla-optimal at 20 tokens/parameter,
B200 at ~30% MFU, β = 0.3, N = 100. Assume ±2× until Phase 1.2 measures the real MFU.

| scale | tokens/model | `F` | trunk | per branch | **zoo total** | disk |
|---|---|---|---|---|---|---|
| Mini 51.5M | 1.03 B | 0.13 GPU-hr | 0.09 | 2.3 min | **4.0 GPU-hr** | 21 GB |
| Small 124.4M | 2.49 B | 0.76 | 0.53 | 13.7 min | **23.3 GPU-hr** | 50 GB |
| Medium 354.8M | 7.10 B | 6.22 | 4.35 | 112 min | **191 GPU-hr** | 142 GB |
| | | | | | **≈219 GPU-hr** | **213 GB** |

**Submit each zoo as a 100-way array of 1-GPU jobs, never as one multi-GPU job.** This is
AICR rule 1 and 2 and it is exactly the right shape here, because branches are
independent:

```bash
sbatch --array=0-99 --gres=gpu:1 --time=00:20:00 --mem=200G \
       --partition=b200-batch --account=p2026_0038_neu slurm/zoo_branch.sbatch
```

Nodes are shared (`select/cons_tres`, `OverSubscribe=NO`). A `--gres=gpu:8` job waits for
a whole node to drain; 1-GPU jobs backfill into partially-free nodes and start
immediately. With the 32-GPU account ceiling, 100 branches run in 4 waves:

| scale | wall clock at 32 concurrent |
|---|---|
| Mini | ~9 min |
| Small | ~55 min |
| Medium | ~7.5 h |

Partition assignment, per AICR rule 5:

| stage | partition | why |
|---|---|---|
| `train_zoo` trunk + branches | `b200-batch` | the 224-GPU pool; 24 h walltime |
| smoke tests, `stack_smoke.sbatch` | `b200-devel` | 4 h, 2-GPU cap, usually idle |
| per-domain PPL sweep (§4.3), eval arms, benchmarks | `rtx-batch` | **separate 32-GPU QOS pool — runs alongside the B200 jobs without consuming their ceiling** |
| PCA fits, flow training, reports, plotting | `cpu` or `rtx-batch` | flows train in seconds; PCA is I/O bound |

Non-negotiables from the cluster skill: **always set `--time` and `--mem` explicitly**
(defaults are 1 h and 1 GB/CPU, and both kill jobs silently); ask an honest `--time`
because the scheduler backfills; never use `--exclusive` or `--nodelist`. Fairshare is
charged to the shared `p2026_0038_neu` account, so idle held GPUs cost Irene and Nadim
too — kill hung jobs.

Storage: artifacts to `$ARTIFACT_DIR` = `/work/neu/p2026_0038_neu/$USER/llm_vae`
(persistent). Caches and logs to `/scratch/$USER`, **purged at 30 days**. Nothing in
`$HOME`. 213 GB against a 9.2 TB ceiling is not a constraint.

### 4.7 The scaling axis

Run the identical protocol at Mini / Small / Medium with N=100 fixed and ask whether the
effective rank of the weight distribution grows with D or stays flat. Nobody can answer
this because nobody has the zoos, and a 6.9× span in D is enough to see a trend. Report
the block-only spectrum beside the whole-stack one to control for the embedding-share
confound (§4.5).

A second, weaker, free claim: generation cost is independent of model size. The flow's
width is `k` whether D is 5e7 or 5e11, and it trains in seconds. Only the basis fit
scales, and it scales linearly.

---

## 5. The DeepWeightFlow dispersion check

DWF's justification for skipping canonicalization is that its benefit vanishes as flow
capacity grows. Misstep 21, measured later in this repo at fixed k / codes / seed / space:

| width | params | epochs | train loss | sample rms |
|---|---|---|---|---|
| 64,128,64 | 51k | 500 | 1.846 | 1.039 (correct) |
| 64,128,64 | 51k | 8000 | 1.568 | 0.959 |
| 256,512,256 | 362k | 8000 | 0.651 | 0.183 |
| 512,1024,512 | 1.23M | 2000 | 0.650 | 0.158 (worst) |

Capacity × duration collapses the flow monotonically while training loss improves
monotonically.

### 5.1 What the DWF procedure was

Confirmed 2026-09-06: **DWF used no early stopping and no gating.** The flow ran a flat
30k steps on a [512, 1024, 2048] net — "it could have gone longer and kept improving."

Misstep 21 has two independent halves, and only one is excluded:

| mechanism | applies to DWF? |
|---|---|
| best-train-loss *selection* picks the collapsed checkpoint | no — there was no selection |
| capacity × duration *drives* collapse, loss falling throughout | yes, and nothing was there to catch it |

"Kept improving" means improving in loss, which is the quantity misstep 21 shows to be
anti-correlated with dispersion.

The hedge is real. DWF's targets were zoos with genuine cluster structure, not a flat
isotropic ensemble. On a flat target the optimal predictor genuinely is mean-ward; on
structured data the same over-fitting can instead present as memorizing training points,
and memorized points are valid models that disperse and score correctly. DWF may well be
clean. **This is a check, not an accusation.**

### 5.2 The check

- [x] ~~What did DWF gate on?~~ Nothing — 30k steps, no gate.
- [ ] One `std()` over the generated codes from any saved DWF run. Ratio near 1 → clean,
      and it becomes a one-line sanity result. Below ~0.8 → a finding.

If no samples survive, the re-run is the DWF repo's own generation scripts plus
`scripts/diag_flow_dispersion.py`. The asymmetry matters: this is a safe thing to find
because it is our own paper.

The stake is immediate — that ablation is the entire justification for not building
TransFusion here. §4.1 motivation 2 makes us robust either way, which is a second reason
to prefer the shared trunk, but we should still know.

### 5.3 Consequence for this sprint

Do not replicate "let it rip." `train_flow.py` defaults to `--holdout_frac 0.15` (val
gating), logs `--dispersion_n` every print epoch, seals the final ratio into
`flow_meta.json`, and prints a loud block below 0.8. `slurm/flow_run.sbatch` defaults to
(64,128,64) at 500 epochs. Keep all of it. The small net is not a compromise — at 51k
parameters it is the only configuration in the sweep that disperses correctly.

---

## 6. Contribution 2 — conditional generation across mixtures

### 6.1 Only `k` has to match, not `D`

The flow's input width is `k`, and `k = N−1` is set by ensemble size, which we choose.
Give GPT-2, a Mamba/SSM and a diffusion-LM N=100 members each and all three produce
99-dim codes. `train_flow.py` already reads `codes_k<k>/` as a stacked `(M, k)` array with
`family_idxs` alongside, so a joint multi-architecture flow needs no code changes.

| | blocked by differing `D`? |
|---|---|
| training one flow over several architectures | no — only `k` must match |
| conditioning on architecture | no — it is another conditioning axis |
| decoding for an architecture we have no members of | **yes** — no basis exists |

The basis absorbs the architecture; the code carries position within the manifold. Two
tractable multi-architecture experiments follow, both **stretch S1** now that the zoo is
GPT-2-only:

1. Does one flow over A architectures beat A separate flows? A free ablation testing
   whether "shape of a training-outcome manifold" is a shared object or whether
   conditioning merely partitions.
2. Compositional held-out cells — a grid of architectures × mixtures with cells removed.
   Train mixture *m′* on GPT-2 only, train Mamba on other mixtures, generate `(Mamba,
   m′)`. Mamba's basis exists, so it decodes. Unseen-architecture generation never does.

### 6.2 Condition on the simplex, not on interpolated embeddings

A class is a pretraining data mixture `π ∈ Δ`, e.g. `[0.4 web, 0.3 code, 0.2 math,
0.1 books, 0.0 multilingual]`.

The conditioning input must be `π` itself through a small MLP, replacing
`nn.Embedding(n_families, cond_dim)` at `gen/flow.py:273` (and `gen/vae.py:145` if the
VAE arm is run). Roughly 15 lines, but it depends on Phase 0.1 and 0.2 landing first.

With a discrete embedding table the only route to a novel mixture is blending learned rows
(`0.5·e_code + 0.5·e_math`). That is weak evidence — the model never saw intermediate
points, embedding space is unconstrained, and smooth output is what any Lipschitz MLP
does. "Of course it interpolates" is a one-line review. With `π` as input, a novel mixture
is the same kind of object as a training mixture, and the claim upgrades from "the latent
space is smooth" to "the conditioning map generalizes."

### 6.3 Zoo composition: nested, 20 anchors + 80 singletons

**Decided.** The problem with five mixtures alone: the simplex over five domains is
4-dimensional, so five mixtures sample it at five points. Fitting `Δ⁴ → cond_dim` from
five points is underdetermined — every smooth interpolant fits, and "we generated a
held-out mixture" is indistinguishable from "the MLP is continuous." It also caps the
science: between-mixture structure spans ≤4 dimensions, the flow's job reduces to modeling
five clusters, and the generative claim is confined to a 4-dim affine hull.

The problem with 20 mixtures × 5 branches: it under-provisions the within-mixture noise
floor, which is what tells you whether a generation error is meaningful.

| component | count | purpose |
|---|---|---|
| anchors: 5 mixtures × 4 branches | 20 models | within-mixture noise floor; 15 pooled dof |
| singletons at distinct Dirichlet-sampled `π` | 80 models | simplex coverage |
| **total** | **100 models, 85 distinct `π`** | |

The only thing given up is 16 extra repeats per anchor, over-provisioned for a variance
estimate. Place anchors at vertices and edges; hold out interior points *and* one vertex,
so interpolation and extrapolation are measured as separate difficulties.

Anchors are the five one-hot vertices; singletons are `π ~ Dirichlet(α=1)` over the
5-simplex, rejecting draws within ε of an anchor or a previously drawn point.

Domains — **as actually run, 2026-09-08.** An earlier draft of this line named
"FineWeb-Edu / StarCoder / OpenWebMath / books / multilingual", which was never what
the code did. Three of the five ids were unusable and were replaced; the table below
is `src/llmzoo/data/mixtures.py` and is the authority.

| domain | dataset | config / split | text column | note |
|---|---|---|---|---|
| web | `HuggingFaceFW/fineweb-edu` | `sample-10BT` / `train` | `text` | unchanged |
| code | `codeparrot/codeparrot-clean` | — / `train` | `content` | **Python only.** StarCoder/The-Stack are all `gated: auto`, and no HF token reaches a job. |
| math | `open-web-math/open-web-math` | — / `train` | `text` | unchanged |
| books | `manu/project_gutenberg` | — / `en` | `text` | name and split were swapped in the code |
| multilingual | `HuggingFaceFW/fineweb-2` | `fra_Latn` / `train` | `text` | **French only** — as was the CulturaX `name="fr"` it replaced. The label was always broader than the config. |

> **Do not "restore" these ids to match an older draft.** β was calibrated against
> these exact corpora (§4.2, 2026-09-08). Changing a corpus changes the per-domain
> separation the gate measured, so the calibrated β no longer applies and the
> calibration has to be re-run. If a corpus must change, re-run §4.2.
>
> Scope to declare in the paper rather than concede: four of five domains are
> English and the fifth is French; "code" is Python. The mixture axis is real and
> measurable (§4.3 passed on all three β) but its axes are narrower than the domain
> names imply. State it next to the domain table.

### 6.4 Baselines, including the one nobody reports

For each held-out mixture:

| arm | what it rules out | cost |
|---|---|---|
| a model actually trained on that mixture | the ceiling | 0.76 GPU-hr at Small |
| `gauss_codes` conditioned on the novel `π` | the usual null | free |
| **the trained model whose mixture is nearest to `π`** | **retrieval** | free |

The third row decides whether this is a result. If the generated model is no better than
"use the model trained on the closest mixture we already have," the flow did not
generalize, it memorized and retrieved. It is `gauss_codes` one level up, it is the
objection a reviewer raises unprompted, and nobody in this literature reports it.

### 6.5 The evaluation that beats ΔPPL

With `π` as input, do not only ask whether the generated model is good. Ask whether the
requested mixture *controls* the output:

> Plot requested weight on domain *d* against the generated model's measured per-domain
> perplexity advantage on *d*. One line per domain. If conditioning works, the lines have
> positive slope.

A control test rather than a quality test, which is what "did the flow generalize"
actually means. It is also immune to the collapse trap: a collapsed generator produces
flat lines however flattering its ΔPPL — the direct answer to misstep 19, where the most
collapsed arm posted the best ΔPPL.

Depends on §4.3 passing. If the zoo's own models do not differ per-domain, the slopes are
flat for reasons unrelated to weight space.

### 6.6 The mechanism, and where the VAE sits

Appendix K lacked three things: CFG, conditioning injected at every layer, and a net that
does not collapse. All three exist in this repo. That is the mechanism, and it needs no
latent.

The VAE is an ablation arm — `flow_codes` (raw PCA codes) versus `flow_latent` (VAE
latent), same eval path, same eight arms, same null. The hypothesis that raw PCA codes are
rough and a smoother target would help is worth testing, but the measured state runs the
other way: `flow_codes` sits at rms 1.04 while `flow_latent` sits at 0.68, and the
contraction belongs to the decoder (`generate` uses no flow at all). The latent currently
costs dispersion rather than buying smoothness.

Keep it wired for the LS-Merge connection, run it as an ablation, and do not claim it is
the mechanism until it beats `flow_codes` on the same null.

> **Parked fix, if the latent ever needs rescuing.** The contraction is an
> aggregate-posterior-versus-prior mismatch, and the VAE's KL constrains `q(z|x)` per
> sample — structurally the wrong instrument. SIGReg (Sketched Isotropic Gaussian
> Regularization, [LeJEPA](https://arxiv.org/abs/2511.08544)) constrains the aggregate
> directly via random projections plus a 1-D normality test per direction. If adopted:
> apply it per-class, never to the global marginal. Applied to a global latent marginal in
> multi-task settings it compresses the task-dependent clusters together, which would
> present as "conditioning doesn't take" — i.e. it would look exactly like Appendix K.

---

## 7. Contribution 3 — the instrument

Mostly written already. What it says:

1. **Report a null.** `gauss_codes` costs one `inverse_transform` and zero training. The
   field does not report one. On a flat-spectrum ensemble a flow matches it to 0.015
   percentage points, so without it "the flow works" is unfalsifiable.
2. **ΔPPL cannot police a generative arm on a mean-centred ensemble.** The ensemble mean is
   essentially a good model, so a collapsed generator scores better than an honest sample.
   Measured: `flow_codes` beat the null by 43× on ΔPPL while emitting at 0.25× RMS.
3. **Collapse is over-fitting, and loss-based selection favours it.** §5's table.
   Dispersion is the only logged number that gets worse as the loss improves, which is why
   omitting it lets a bad result ship.
4. **Error magnitude does not predict functional damage.** At matched relL2 = 2e-3 the
   VAE's structured error costs `pythia_410m` +0.378% PPL while isotropic noise costs
   ~+0.03% — 10× — and truncation error at 7.4e-3, nearly 4× larger, *improves* PPL.
5. **Use bulk spectrum statistics.** `ev[0]/ev[k−1]` reads 12.09 at k=N−1 and 1.006 at
   k=N/2 on the same isotropic data. Gate on `ev0/median` and effective-rank ratio.

Framed this way, the paper survives the §4.4 prediction failing.

---

## 8. Contribution 4 (stretch) — PEFT in the zoo basis

`w(z) = mean + Xcᵀ U_k S_k^{-1/2} z` is linear in z, so
`dL/dz = S_k^{-1/2} U_kᵀ Xc (dL/dw)` — one `transform_vector`-shaped contraction.

Materialize `B = Xcᵀ U_k S_k^{-1/2}` once as `(D, k)`, make `z` a leaf tensor, set
`w = mean + B @ z`, and `functional_call` the model on `w`. Autograd handles the rest;
about 50 lines. At Small with k=99 that is 49 GB fp32 (141 GB at Medium) — resident on one
B200's 178 GiB either way — and the per-step cost is two matvecs, free next to the LM's
own forward and backward.

~100 trainable parameters adapting a whole model. The comparisons that make it a result:

| baseline | controls for |
|---|---|
| random k-dim subspace | is the subspace doing work, or just the dimension count? (Li et al.; Aghajanyan et al.) |
| LoRA at matched parameter count | is a global k-dim constraint better than a per-matrix low-rank one? |
| single-model trajectory SVD | ["Fine-tuning Happens in Tiny Subspaces"](https://arxiv.org/pdf/2305.17446) — is a zoo-derived subspace transferable where a self-derived one is not? |

Row 3 is the closest prior art and the one to beat: their subspace comes from one model's
own trajectory and is non-transferable by construction. Ours is cross-mixture.

It composes with contribution 2 — the flow generates the initialization, ~100-dim PEFT
refines it — which is the "adaptation cost reduced by orders of magnitude" claim the field
says it wants.

---

## 9. Phases

Six phases. Each lists what it produces, what gates it, and what it costs. Everything
unassigned defaults to Scott.

### Phase 0 — finish the refactor (DONE for what Phase 1 needs)

Landed 2026-09-08. Full detail in §2.1.

| # | change | status |
|---|---|---|
| 0.2 | conditioning table size decoupled from the registry (`N_COND_SLOTS = 32`) | **done** |
| 0.3 | `MemberSource` abstraction; `EnsembleDataset(source=, zoo_dir=)` | **done**, acceptance verified |
| 0.6 | `exclude_1d=False` for zoo runs | **done** |
| 0.1 | split `family_idx` into `basis_key` and `cond` | **deferred to Phase 2** — Phase 3 needs it, Phase 1 does not |
| 0.4 | arm registry in `eval_stack.py` | deferred to Phase 4 |
| 0.5 | `--k` consistency | deferred; comfort only |

Verify on the cluster before anything else:
`TEST=tests/test_zoo.py sbatch slurm/tests.sbatch`.

### Phase 1 — trainer, calibration, pilot (1–2 days + ~3 GPU-hr)

| # | item | produces | cost |
|---|---|---|---|
| 1.0 | **`--verify_domains`** — open all five HF streams, pull a doc each | pass/fail per domain | seconds, `cpu` |
| 1.1 | `scripts/train_zoo.py` + `slurm/zoo_{trunk,branch}.sbatch` (**written**) | `zoo/<arch>/w_<i>.npy`, `zoo_meta.json`, `zoo_plan.json` | done; needs a cluster shakedown |
| 1.2 | β calibration: N=12, β ∈ {.15, .30, .60}, Mini (§4.2) | per-domain PPL spread per β; chosen β; **measured MFU**, which recalibrates every number in §4.6 | 1.9 GPU-hr |
| 1.3 | `scripts/eval_domains.py` — the §4.3 gate | `domain_separation.json`, exit 0/1 | minutes, `rtx-batch` |
| 1.4 | Pilot zoo N=32 @ Mini, PCA fit, spectrum | `pipeline_summary_k31.json` | ~1.3 GPU-hr |

**Do 1.0 first.** The five HF dataset ids in `src/llmzoo/data/mixtures.py` are the
most fragile thing in the repo — datasets get renamed and gated — and finding that out
in seconds beats finding it out 40 minutes into a trunk run. Swapping an id is a
one-line edit and changes nothing downstream.

**Gate, and it lands on day 2 or 3, not day 10.** If 1.3 shows no per-domain separation,
raise β and re-run — this is not a reframe. If 1.3 passes and 1.4 returns a flat spectrum
against §4.4, stop and rebuild the paper around contribution 3.

Smoke first: `sbatch slurm/stack_smoke.sbatch` on `b200-devel` is the known-green bisect
point. If it fails, the environment is broken, not the science.

### Phase 2 — the zoos (~219 GPU-hr, mostly unattended)

| # | item | GPU-hr | wall at 32 concurrent |
|---|---|---|---|
| 2.1 | Zoo @ Small 124M, N=100, nested 20+80 — **the primary result** | 23.3 | ~55 min |
| 2.2 | Zoo @ Mini 51M, N=100 | 4.0 | ~9 min |
| 2.3 | Zoo @ Medium 355M, N=100 | 191 | ~7.5 h |
| 2.4 | Gram PCA + codes at each scale | minutes each | |
| 2.5 | **Phase 0.1 while the zoos run** — split `family_idx` into `basis_key` + `cond` | 1 d, CPU | — |

2.5 is deliberately here rather than in Phase 0: it is the change most likely to need
a cluster round-trip to debug, and nothing in Phase 2 depends on it. Phase 3 does.

Exit: three spectra, one scaling curve (§4.7), block-only spectra alongside whole-stack.

### Phase 3 — conditioning (0.5 day, parallel with Phase 2)

| # | item | depends on |
|---|---|---|
| 3.1 | `π`-vector MLP replacing `nn.Embedding` (`gen/flow.py:273`) | 0.1, 0.2 |
| 3.2 | Flow training per scale — (64,128,64), 500 epochs, `--holdout_frac 0.15`, dispersion logged | 2.4 |

### Phase 4 — evaluation (1.5 days)

| # | item | partition | note |
|---|---|---|---|
| 4.1 | Eight arms per scale; held-out interior mixtures and one held-out vertex | `rtx-batch` | |
| 4.2 | Retrieval baseline (§6.4) | `rtx-batch` | the arm that decides the result |
| 4.3 | Ceiling models — one trained per held-out mixture | `b200-batch` | 0.76 GPU-hr each at Small |
| 4.4 | **Per-domain slope figure** (§6.5) | `cpu` | the headline evaluation |
| 4.5 | DWF dispersion check (§5.2) | `cpu` | independent of everything; hands off cleanly |

### Phase 5 — writing (5–7 days)

Abstract 18 Sep, paper 25 Sep.

### Stretch, in cut order (last cut first)

| # | item | cost |
|---|---|---|
| S1 | Multi-architecture joint flow + compositional cells (§6.1) | 1–2 d + a non-GPT-2 zoo |
| S2 | PCA-space PEFT (§8) | 2 d |
| S3 | VAE latent ablation; SIGReg only if the latent needs rescuing | 2 d |

**Cut order: S3, S2, S1, then 2.3 (Medium), then 4.5.** Phases 0–5 are the paper.

### Timeline

10 days to abstract, 17 to paper. This works only because contributions 1 and 3 are mostly
built and the zoo compute is an afternoon. It does not work if Phase 0 sprawls or if we
chase §11. **Phases 0 and 1 are the only serial bottleneck**; 3.1, 4.5 and all stretch
items parallelize cleanly across people.

---

## 10. Out of scope

State these as limitations rather than letting a reviewer state them.

- **Generating for an architecture we have no members of.** The only thing differing `D`
  actually forecloses (§6.1). Without members there is no basis, so a generated code has
  nothing to decode into. Bedionita's GNN-over-compute-graph architecture embeddings are
  the right mechanism for the conditioning half; the decoding half stays open. Natural
  joint follow-up.
- **Free-standing generators.** `inverse_transform` computes `x̂ = mean·(1−Σa) + Σaⱼ·Xⱼ`,
  so decoding a generated code requires the whole training ensemble on disk. Mitigable by
  materializing an explicit rank-k basis (49 GB at Small, k=99), but state it first. DWF's
  dual PCA has the same property and does not mention it.
- **Global weight-space claims.** We model a local manifold around one trunk.
- **Architectures other than GPT-2.** The zoo is GPT-2-only for this paper; multi-arch is
  stretch S1.
- **Frontier scale.** Not attempted, not claimed.

---

## 11. Parked

**Reward-finetuning the flow, TempFlow-GRPO style**
([2508.04324](https://arxiv.org/abs/2508.04324)). The pitch is specific: text-to-image RL
must use a noisy learned preference model, whereas in weight space the reward oracle is
exact and cheap — decode, write back, measure held-out loss. Their trajectory-branching
credit assignment drops straight in. Cost is the blocker: each reward evaluation is an
`inverse_transform` plus a PPL eval, so a group of 8 over a few hundred steps is tens of
GPU-hours plus new code. Second paper.

> Do not map training step to flow time literally. `x_t = (1−t)x₀ + t·x₁` is a straight
> segment from an isotropic Gaussian and a checkpoint is not on it. That mapping would
> also break the ODE round-trip diagnostic, the only thing separating solver error from
> model inconsistency.

**Quality-conditioned flow.** Condition on a scalar quality or step, train on everything
including mediocre checkpoints, sample at the good end. One scalar into `_condition`, and
it is the correct way to use low-quality data. Largely moot under §4.1 motivation 3, but
the fallback if trunk-branching fails.

**Drift matching / SDE samplers.** Not now, and for a sharper reason than cost.

> An SDE sampler would destroy the diagnostic contribution 3 is built on.
> `code_rms_ratio` is informative *because* the ODE is deterministic — spread in the
> output can only come from spread the model learned. Inject noise at sample time and you
> manufacture dispersion: a mean-ward drift plus diffusion reads rms ≈ 1.0 and is literally
> `gauss_codes`. We would be building the null and measuring it as a success. Adopting an
> SDE requires replacing the diagnostic first — a two-sample test (energy distance / MMD)
> against the real codes, or §6.5's slope figure, which is a control test and stays immune.

As a training-side regularizer it needs no new code: `gen/flow.py` has `path_noise`,
defaulting to 0.0. The docstring records the cost — with `path_noise > 0` the interpolant
leaves the straight segment, so the round-trip residual stops being attributable to Euler
error. One flag if we want the datapoint. Mostly moot regardless: 51k parameters at 500
epochs gives rms 1.039, so the memorization fix is already free.

Two reasons to keep it on the roadmap: it is a *prerequisite* for the GRPO arm above
(trajectory branching injects one-step SDE noise at intermediate latents), and if we ever
fall back to five mixtures the effective structure is ~4-dimensional over 100 points and
the flow will memorize it easily — at which point "we need an SDE" becomes tempting, and
should be read as a symptom of the data design rather than a modeling problem.

**Bedionita's 30 JAX GPT-2s at 8B WebText tokens.** Real data that exists now, and now
architecture-matched to our zoo. Worth one day to convert and spectrum-check even at N=30
(k=29), as an independent read on §4.4 before our own zoo lands. Open: are the inits
distinct, and are the weights retrievable? Flax↔PyTorch conversion is fiddly — transposed
`Conv1D` kernels, param-tree naming — but bounded.

---

## 12. Decisions and open questions

### Decided

| question | decision | where |
|---|---|---|
| Cluster | **AICR only.** Explorer retired. 1-GPU array jobs, never `--gres=gpu:8`. | §4.6 |
| Architecture | **GPT-2, exactly** (Radford et al. 2019). Mini 51M / Small 124M / Medium 355M; Small and Medium are the published configs. | §4.5 |
| Vocab | **50257 throughout.** Changing it would confound the scaling comparison. | §4.5 |
| Trunk | **Ours**, at all three scales. Public trunks ship no optimizer state. | §4.1 |
| Corpora | **FineWeb-Edu / CodeParrot-clean (Python) / OpenWebMath / Project Gutenberg (en) / FineWeb-2 (fra_Latn).** Fixed for the sprint — β was calibrated against these. | §6.3 |
| β | **0.30 recommended** (all of 0.15 / 0.30 / 0.60 passed the §4.3 gate on 2026-09-08). 0.15 is what §4.3's "smallest passing" rule selects and is ~2× cheaper; 0.30 buys divergence headroom for the singleton regime N=12 could not test. | §4.2 |
| Mixture count | **Nested 20 anchors + 80 singletons**, 85 distinct `π`. | §6.3 |
| β | **1.9 GPU-hr calibration at Mini**; full sweep dropped as indefensible at Medium. | §4.2 |
| DWF flow gating | **None** — 30k flat steps. The check is a `std()`, not a question. | §5.1 |
| SIGReg / the VAE | **Off the critical path.** Sprint runs PCA → Flow. | §6.6 |
| Drift matching | **Parked with a verdict.** | §11 |
| `exclude_1d` for zoo runs | **False.** Biases and LN gains enter `D`; ~0.1% cost. | §2.2 |
| Phase 0 scope | **0.2 / 0.3 / 0.6 shipped; 0.1 moved to Phase 2, 0.4 to Phase 4.** | §2.1 |

### Open

1. Bedionita's zoo — distinct inits? retrievable? (§11)
2. Joint submission with KAIST, same deadline?
3. Multi-architecture (§6.1) — stretch S1 as written. Promote if Bedionita re-engages,
   since it is free to *train* and expensive to *build*.
4. Who runs what. Phases 0–1 are the serial bottleneck; 3.1, 4.5 and all stretch items
   parallelize.

---

## 13. Standing rules

`CLAUDE.md` is the full list. The ones that bind this plan:

- **Never run torch or transformers locally.** Everything through `sbatch` on AICR.
  `tests/test_report.py` is the sole exception and `test_stdlib_purity` enforces it.
- **Always set `--time` and `--mem`.** Defaults are 1 h and 1 GB/CPU; both kill jobs
  silently. An honest `--time` gets backfilled sooner.
- **Ask for the fewest GPUs that work.** Nodes are shared; a 1-GPU job starts immediately
  where an 8-GPU job waits for a node to drain. Never `--exclusive`, never `--nodelist`.
- **`rtx-batch` is a separate 32-GPU pool** and does not consume the b200 ceiling — put
  eval there so it runs alongside training.
- Fairshare is charged to the shared `p2026_0038_neu` account. Kill hung jobs.
- Artifacts to `/work/neu/p2026_0038_neu/$USER/llm_vae`; caches and logs to
  `/scratch/$USER`, **purged at 30 days**; nothing in `$HOME`.
- **Print `${#HF_TOKEN}`, never `${HF_TOKEN:-UNSET}`.** Rotation still outstanding.
- `chunk_bounds` must stay a fixed grid (misstep 10). This survives Phase 0.3 unchanged.
- Read `code_rms_ratio` before any generative ΔPPL (misstep 19).
- Do not scale up the flow (misstep 21). Defaults are (64,128,64) at 500 epochs.
- Re-calibrate `--noise_scale` whenever the layout changes (misstep 18). Moot for a real
  zoo, but `source="noise"` stays as the machinery test and the β→0 reference point.
- Do not prune `runs/emb3/flow_k*_collapsed/` — the only evidence for misstep 21.
- Do not edit `eval_stack.py` or `report_stack.py` while a dependency chain is queued.
