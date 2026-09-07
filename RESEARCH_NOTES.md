# Research Notes (ARCHIVE — superseded 2026-09-06) — read this before you write code

> **This is the OLD notepad. The forward plan lives in
> [RESEARCH_PLAN.md](RESEARCH_PLAN.md).**
>
> This file is still authoritative for two things and should not be deleted:
> **§5 (the 21 recorded missteps)** and **§6 (environment facts)**. Both carry
> forward unchanged. Its §4 "next experiments" and §7 "open questions" are
> superseded — read RESEARCH_PLAN.md for those.

This file has one job. It stops the next agent from repeating work that already
happened, and from repeating mistakes that already happened.

Written 2026-08-26, after the whole-stack pipeline landed and ran.
Awarded Scott's Stamp of Tampering and Approval, 2026-08-26
Updated 2026-09-04, after the CFM sprint (tables, save/load, flows).
Archived 2026-09-06, when the project got a real data plan.


---

## 1. Glossary

Define these once. The words are used the same way everywhere in this repo.

| Term | Meaning here |
|---|---|
| **sample** | One complete decoder stack, flattened. NOT one transformer block. |
| **stack** | All L decoder blocks of one model, concatenated. Length `D = L * block_size`. |
| **N** | The number of samples in one family's ensemble. NOT the layer count. |
| **k** | The number of PCA components kept. `k <= N - 1`. |
| **member** | One sample inside an ensemble. Member 0 is the real pretrained model. |
| **family** | One architecture, e.g. `pythia_410m`. Each family has its own PCA basis. |
| **arm** | One evaluation path. Eight exist: `pca_only`, `vae`, `generate`, `gauss_codes`, `flow_codes`, `flow_latent`, `flow_rt_codes`, `flow_rt_latent`. |
| **`gauss_codes`** | The NULL arm: `z ~ N(mean_f, std_f)` from `code_stats`. What a flow has to beat. |
| **`code_rms_ratio`** | A generative arm's sample spread over the real codes', in normalized code space. 1.0 correct, <0.8 collapsing. Read it BEFORE dPPL (misstep 19). |
| **rank bound** | `k = N - 1`. Centering removes one degree of freedom, so the centered data cannot have higher rank. |
| **relL2** | `norm(reconstruction - target) / norm(target)`. |

---

## 2. Where the work stands

The machinery works. Every number it produces matches an analytic prediction.

**Verified**

1. Per-family PCA reconstructs exactly at `k = N-1`. Cosine is 1.000000 for all three
   families. The old shared block basis gave 0.388 for `pythia_410m`.
2. Padding is gone. Every family has uniform block sizes, so a stack is fixed-length.
3. Posterior collapse is gone. Total KL went from 0.00106 to 11.36 nats per sample.
   All 32 latent dimensions stay active. No tuning was needed.
4. The truncation residual formula holds to within 5% on the decoder-only
   layout, and to within 6-10% on the whole-stack layout: at k=N/2 with N=100
   and s=3e-3 it predicts relL2 = 0.002118 and measures 0.002336 / 0.002284 /
   0.002251 for gpt2_medium / smollm2_360m / pythia_410m (job 9894474,
   sample_idx 5). Truncation-as-denoiser also confirmed on the new layout:
   member 5's noise cost gpt2_medium +0.519% PPL and `pca_only` at k=N/2
   recovered -0.320% of it, against a predicted ~-0.26% for half the variance.
5. The full run costs 10 minutes and 20 GB of RAM on one V100.
6. 348 automated checks pass across five test modules: `tests_step1_dual_pca.py`
   (35), `tests_step3_ensemble.py` (56), `tests_step4_run_bundle.py` (60),
   `tests_step5_flow.py` (91), `tests_step6_report_stack.py` (106).

**Added 2026-09-02 (sprint: tables, save/load, CFM)**

7. **`D` now includes the embeddings, final norm and untied LM head** (open question
   2). The extra segment is derived as the *complement* of the `layers_attr` subtree
   in `named_parameters()`, so it stays architecture-agnostic and weight tying is
   handled for free — `named_parameters` deduplicates, so a tied `lm_head`/`wte` pair
   is counted once. A stack is now `[extra | block_0 … block_{L-1}]`.
   `ENSEMBLE_LAYOUT_VERSION` 1 → 2, so old ensembles are refused rather than mixed.
8. **Benchmarks are wired into the stack eval.** `eval_stack.py --bench` measures
   MMLU/HellaSwag/GPQA on the same in-memory reconstructed model, between write-back
   and restore, so every arm inherits them for free.
9. **Artifacts are versioned and fingerprinted.** `StackVAE` has real `save`/`load`,
   per-family code statistics are a first-class artifact (`codes_k<k>/`), and
   `run_bundle.load_run` reconstitutes a whole run in one call. Pre-provenance VAE
   directories are refused by default.
10. **Conditional rectified flow, in both spaces**, with CFG and an ODE round-trip
    diagnostic. `flow.py` / `train_flow.py`.
11. **`--noise_scale` is re-calibrated to `3e-3`** on the whole-stack layout
    (job 9894023). The old `1e-2` was measured on decoder blocks only and is now
    wrong by a wide margin — see misstep 18.
12. **Flow arithmetic is pinned.** `tests_step5_flow.py`, 91 checks: Euler is exact
    to 1e-14 in float64 at every step count, the float32 error sits 1730x below what
    an off-by-one in the step count would cost, and the round-trip residual falls
    4.0x and 5.1x for 4x and 5x the steps — first-order convergence, which is what
    lets `--flow_rt_steps` separate solver error from model inconsistency.
    `tests_step6_report_stack.py`, 76 checks, covers the reporting path.

**The sprint's actual question, answered (2026-09-03)**

13. **Do the flows earn their keep? NO — and for the legitimate reason, on the second
    attempt.** At MATCHED dispersion (both arms at code rms 1.02-1.05x, job 9918746,
    sample_idx 5), `flow_codes` and `gauss_codes` agree in all six arch x rank cells:

    | arch | k | `gauss_codes` | `flow_codes` | gap |
    |---|---|---|---|---|
    | gpt2_medium | 99 | -0.341% | -0.356% | 0.015 |
    | smollm2_360m | 99 | -0.207% | -0.210% | 0.003 |
    | pythia_410m | 99 | -0.077% | -0.077% | 0.000 |
    | gpt2_medium | 50 | -0.502% | -0.510% | 0.008 |
    | smollm2_360m | 50 | -0.194% | -0.192% | 0.002 |
    | pythia_410m | 50 | -0.066% | -0.067% | 0.001 |

    Gaps under 0.015 percentage points, with relL2 matching to four significant
    figures. This is exactly the prediction written into `slurm_flow_run.sh`'s header
    BEFORE any of it ran: the ensemble is `w_0 + s*sigma*eps_i`, so the per-family
    code distribution is Gaussian by construction, the spectrum is flat
    (ev0/median 1.001), and there is nothing for a flow to learn beyond the prior.

    **The machinery is validated; the ensemble is what cannot answer the question.**
    Note it took two attempts to get an honest reading -- the first said the flow beat
    the null by 43x, which was collapse (missteps 19 and 21). Do not compare
    generative arms without checking `code_rms_ratio` first.

    Still open: `generate` and `flow_latent` remain contracted at 0.59-0.64x even with
    a healthy flow, because that contraction is the VAE DECODER's (`generate` uses no
    flow at all). Prior-posterior mismatch, untouched by any of this, and the next
    thing to fix if latent-space sampling is wanted.

**Still not verified, because this ensemble cannot verify it**

The pipeline has now been asked one question and it answered cleanly (item 13): on a
manufactured ensemble a flow adds nothing over a Gaussian. That is a MACHINERY result,
not a research result. The research question — do complete models share
low-dimensional structure in weight space — needs an ensemble of genuinely different
models. See section 4, Experiment 1.

---

## 3. Small Sample Sizes

This repo has exactly one pretrained model per family. This is in contrast to DeepWeightFlow, which uses about 100 **independently trained** networks per task. The ensemble is therefore manufactured with Gaussian noise, as shown below:

```
member 0        = w_0                        (the real model, exactly)
member 1..N-1   = w_0 + s * sigma * eps_i    (isotropic noise)
```

Three consequences follow:

1. **The basis spans injected noise.** The principal directions describe the perturbation, which obscures the trained weights.
2. **The mean is the real model.** `mean = w_0 + O(s*sigma/sqrt(N))`. So
   `x_hat ~= mean ~= w_0` at any rank. Reconstruction of member 0 cannot fail.
3. **Truncation acts as a denoiser.** Discarding half the variance discards half the
   injected noise, so a truncated reconstruction of a noisy member is a BETTER model
   than the member. Measured: `pythia_410m` recovers exactly half the noise-induced
   perplexity penalty at `k = N/2`.

Point 3 indicates that a rank sweep on `k` could be quite interesting, and contextualize findings in LS-Merge as well as providing a avenue for illuminating the conseuqences of this construction. 

## 4. Next experiments

### Experiment 1 — Pythia checkpoint revisions 

**Question.** Do complete models from one training run share low-dimensional structure in weight space? *Obviously this is true for near-terminal checkpoints.*

**Why this ensemble.** EleutherAI publishes about 143 intermediate revisions per
Pythia size. Every revision is a genuine complete model. Model scale is fixed. No
member sits at a privileged centre. This is the closest available thing to
DeepWeightFlow's ensemble.

**Start with `pythia_160m`, not `pythia_410m`.**

| | pythia_160m | pythia_410m |
|---|---|---|
| L | 12 | 24 |
| block (matrix only) | 7,077,888 | 12,582,912 |
| D | 84,934,656 | 301,989,888 |
| 143 members on disk | ~48 GB | ~173 GB |

Both fit on scratch. `pythia_160m` iterates about 3.5 times faster.

**Code change required.** `EnsembleDataset.load_chunk` currently regenerates members
from a seed. Real members must be read from disk instead. Add a second ensemble
source rather than replacing the first:

- Keep `source="noise"` for the machinery test.
- Add `source="revisions"`, which stores one `w_i.npy` per member and reads a chunk
  from each. `models/registry.py` needs a `revision=` argument threaded into
  `from_pretrained`.
- `chunk_bounds` must stay the fixed grid. `DualGramPCA` does not care where a chunk
  comes from, only that the chunk is the same on every pass.

**What proves the hypothesis.** A **steep** eigenvalue spectrum. Steep means the
first few components hold most of the variance, so training trajectories occupy a
low-dimensional subspace. Report `ev[0]/ev[k-1]` and the cumulative variance curve.

**What disproves it.** A **flat** spectrum, like the noise ensemble produces. Flat
means the checkpoints are near mutually orthogonal, so PCA does not compress and no
generative model over these codes will generalize.

**Read the spectrum warning.** `DualGramPCA` prints a warning when the retained
spectrum spans more than 1e6. That warning matters here. Forming the Gram matrix
squares the condition number, so a genuinely steep spectrum may exceed what float32
data supports. If the warning fires, lower `k` rather than lowering `rank_rtol`.

### Experiment 2 — Held-out revision

**Question.** Does the basis generalize to a checkpoint it never saw? *Again, this should be the case for near-terminal checkpoints.*

**Method.** Fit on revisions 0 to 119. Project revision 130 with
`DualGramPCA.transform_vector`. Reconstruct. Measure ΔPPL.

This is the first honest generalization test in the project. Every earlier number
measured in-sample reconstruction. `transform_vector` already exists and is tested.

**Gate.** ΔPPL, not cosine. Cosine below about 0.9999 tells you nothing.

### Experiment 3 — A flow over the codes

**STATUS 2026-09-04: BUILT AND RUN on the noise ensemble; result is the null.**
`flow.py` / `train_flow.py` / four eval arms exist and are tested (91 checks). Run
against the manufactured ensemble, `flow_codes` matched `gauss_codes` in all six
arch x rank cells to within 0.015 percentage points (item 13). That is the expected
answer, not a failure: a flow over codes whose aggregate distribution is already
Gaussian learns the prior and adds nothing.

`train_flow.py --require_spectrum_ratio 10.0` now makes the original warning
machine-checkable — it refuses to train when `ev0/median` says the spectrum is flat.
It defaults OFF so the machinery stays smoke-testable on the noise ensemble; **turn
it on for the revisions run.** Two things to carry in:
  * gate early stopping on val loss (`--holdout_frac`, now 0.15 by default);
  * report `code_rms_ratio` next to every generative dPPL (missteps 19, 21).

**Just Build Flows** Rectified flow, conditional-OT path. Velocity field
`v(z_t, t, family, step)` as a 3 to 4 layer MLP with a sinusoidal time embedding. About 50 Euler steps at sampling time. On 99-dimensional codes it trains in seconds.

`eval_stack.py` already has the seam: the `generate` arm decodes a latent and writes
the result back. Replace the sampler, keep everything else.

The reasoning is simply that they aren't that expensive or challenging to implement, and if they do not work, then the problem is upstream. See *DeepWeightFlow* for a reference here. 

**Judge samples on ΔPPL.** Do not judge them on code-space or weight-space L2. See
misstep 9.

### Experiment 4 — Cross-family transfer 

Per-family PCA gives each family its own basis. Code dimension 0 of `gpt2_medium` and
code dimension 0 of `pythia_410m` are coefficients on unrelated directions. To address this we could either use a VAE as a first pass, or just let the Flows figure it out. 

This design choice makes it difficult -- **but maybe not impossible** to explore transfer to unseen model families. The mechanism for this exploration would be to anneal the novel family class into being through CFG inside the flow system (this is a hunch, I do not have citations supporting this, but I believe it is possible). We may want to use CFG anyways, so it is worth a try with relatively little overhead. This does not answer the PCA decoding question, but we could use the VAE to map the novel family into the PCA space of the known families, and then decode from there? Again, all highly speculative.

---

## 5. Recorded missteps

Every item below cost real time in this project. Do not repeat them.

### About the framing

**1. A PCA sample is a whole model, not a block.**
The repo's legacy code treats one block as one sample, so reading the code teaches the
wrong model. I got this wrong three times in one conversation. Check section 1 before
you reason about shapes.

**2. `N` is the ensemble size, not the layer count.**
`k = N-1` refers to the number of complete models. It has nothing to do with how many
layers a model has.

**3. Two pipelines coexist. Do not mix their artifacts.**
`blocks/` plus `pca/components.npy` belongs to the legacy pipeline. `ensemble/` plus
`pca/<arch>/gram_*.npy` belongs to the current one. Both write into
`<artifact_dir>`. The `layout_version` fields exist to stop a cross-read.

### About numerics

**4. Do not use `randomized_svd` on a symmetric Gram matrix.**
Use `np.linalg.eigh`. `randomized_svd` returns absolute eigenvalues, so a direction
that rounds negative comes back positive with an arbitrary eigenvector. Unit
normalisation then promotes that noise to a full-status component. Measured on a
synthetic case: orthogonality error 0.82 at `k = n-1`, one direction 100% wrong.

**5. The Gram matrix squares the condition number.**
A singular-value ratio of `r` appears as an eigenvalue ratio of `r^2`. So eigenvalues
below about float32 epsilon are unrecoverable. No choice of `rank_rtol` recovers them.
Lower `k` instead. `DEFAULT_RANK_RTOL = 1e-7` is the conservative end of the usable
range. Do not lower it to keep more components.

**6. Do not store components as float16.**
Unit-norm vectors over `D = 302M` have typical entries near 5.7e-5. Float16 is
subnormal there. Against a cosine target of 0.9999 that is a first-order error. This
also made two scripts silently measure two different bases: one used the in-memory
float32 object, the other loaded the float16 file.

**7. Accumulate the Gram matrix in float64.**
Entries are sums over hundreds of millions of products. Float32 carries about 1e-3
relative error at that length, which is the same size as the structure being resolved.

### About the ensemble

**8. `noise_scale = 1e-7` is unusable.**
1e-7 *is* float32's relative precision. The members sit at the representable limit,
the Gram matrix is roundoff, and the rank floor discards the basis. Use
`calibrate_noise.py`. **The calibrated value is `3e-3`** on the current whole-stack
layout (job 9894023). `1e-2` was the decoder-blocks-only answer and is wrong now --
see misstep 18. Re-calibrate whenever the layout changes; nothing else transfers.

**9. Member 0 sits at the ensemble centre.**
`mean = w_0 + O(s*sigma/sqrt(N))`, so member 0's deviation from the mean is smaller by
`sqrt(N)`. Two different residual formulas follow, and mixing them up produces a wrong
prediction:

| target | residual at `k = N/2` |
|---|---|
| a typical member | `s*sigma/sqrt(2)` |
| member 0 | `s*sigma/sqrt(2N)` |

I documented this factor and then failed to apply it when writing a prediction into a
job header. Use `eval_stack.py --sample_idx` with a nonzero index for any rank sweep.

**10. Never change the chunk grid between passes.**
Noise is keyed to `(seed, chunk_idx)`. If the grid changes, pass 2 sees a different
ensemble than pass 1, and the Gram matrix is meaningless. `load_chunk` takes a chunk
index, not raw offsets, so that this cannot happen by accident. Keep it that way.

**18. One global `weight_std` mis-scales the noise per submatrix.** *(2026-09-02)*
The ensemble is `w_0 + s * sigma * eps`, where `sigma` is ONE number for the whole
stack. Once embeddings entered `D` that became the dominant source of cross-family
variation, and the effect is large:

| arch | global `weight_std` | extra (emb + head) | dPPL at s=1e-2, blocks only | dPPL at s=1e-2, whole stack |
|---|---|---|---|---|
| gpt2_medium | 0.1083 | 14.8% of D | +0.097% | **+4.363%** |
| smollm2_360m | 0.1717 | 13.0% of D | +0.250% | +0.567% |
| pythia_410m | 0.0277 | 25.4% of D | +0.956% | +0.994% |

I predicted `pythia_410m` would be worst, because it has the largest extra segment
and its LM head is untied, so ~25% of its stack is logit-facing. **That prediction
was wrong, and the reasoning behind it was wrong.** `pythia_410m` barely moved
(+0.956% -> +0.994%); `gpt2_medium` got 45x worse.

The mechanism is the global `sigma`, not the logit-facing fraction. `gpt2_medium`'s
global `weight_std` is 3.9x `pythia_410m`'s, so at equal `s` it takes 3.9x more
ABSOLUTE perturbation — and it lands on embeddings whose own scale is comparable
across the three families. A family whose internal spread of per-tensor stds is wide
gets its small-magnitude tensors over-perturbed and its large ones under-perturbed.

Two consequences:

* **Re-calibrate whenever the layout changes.** `calibrate_noise.py` perturbs the
  same object `EnsembleDataset` builds, so it transfers; nothing else does. The
  current value is `s = 3e-3`, the largest scale keeping every family within +0.5%.
* This is an argument for **per-parameter-group noise scaling** (`s * std(tensor)`
  rather than `s * std(stack)`), which would make `s` mean the same thing in every
  tensor. Not implemented — it changes the ensemble definition and therefore every
  fingerprint, so it belongs to whatever run adopts it, not retrofitted onto this
  one. See open question 6.

### About measurement

**11. Cosine above 0.999 is not a useful bar.**
A measured cosine of 0.99939 came with +72,468% perplexity. Cosine only becomes
informative above about 0.9999. Gate on ΔPPL.

**12. `pca_only` at `k = N-1` is a self-test, not a result.**
The rank bound makes it exact. Reporting "cosine 1.0" there proves the arithmetic
works and nothing else.

**13. Error magnitude does not predict functional damage.**
Three measurements on `pythia_410m`:

| error source | relL2 | ΔPPL |
|---|---|---|
| VAE round trip at `k=N-1` | 2.0e-3 | **+0.378%** |
| isotropic noise (calibration) | 2.0e-3 | ~+0.03% |
| PCA truncation at `k=N/2` | 7.4e-3 | **−0.524%** (improves) |

The VAE's error is structured. It sits in the leading principal directions, which are
the ones carrying function. Isotropic noise spreads over all 302M dimensions.
Truncation error points back toward `w_0`. So L2 alone ranks these wrongly, and by a
factor of 10.

**14. On a noise ensemble, truncation improves the model.**
Do not read a negative ΔPPL at low rank as success. It means the target was noisy and
truncation removed some of the noise.

**15. `total_variance_captured` is vacuous at `k = N-1`.**
All non-null directions are retained, so the ratio is 1.0 by construction. Report it
at a lower rank as well, or it misleads. `gram_pca_meta.json` carries an explicit
`variance_captured_is_vacuous_at_k_eq_N_minus_1` flag.

**15b. `ev[0]/ev[k-1]` is NOT a flatness measure at `k = N-1`.** *(2026-09-02)*
Same trap as 15, and it nearly let a flat-spectrum flow train unchallenged.
Measured on one pure-noise ensemble, N=12, three families:

| statistic | at `k = N-1` (11) | at `k = N/2` (6) | truth |
|---|---|---|---|
| `ev0/ev[k-1]` | **12.09** | 1.006 | — |
| `ev0/median` | 1.01 | 1.01 | flat |
| effective-rank ratio | 0.92 | 0.93 | flat |

At the rank bound `ev[k-1]` is the smallest direction still standing after the rank
floor, so the ratio describes the *tail* and reports "steep" for an ensemble that is
isotropic by construction. Gate on a **bulk** statistic instead. Two are now recorded
in `pipeline_summary_k*.json` by `train_stack.spectrum_stats`:

* `spectrum_ev0_over_median` — the leading direction as a multiple of the typical
  one. ~1 for isotropic noise.
* `spectrum_effective_rank_ratio` — `(Σev)² / (Σev²) / k`, the participation ratio
  normalised to [0, 1]. 1.0 means every retained direction carries equal variance.
  Cannot be moved by one marginal direction at the tail.

`train_flow.py --require_spectrum_ratio` gates on `ev0/median` for exactly this
reason.

**19. dPPL cannot police a GENERATIVE arm on a mean-centred ensemble.** *(2026-09-02)*
The worst measurement trap found so far, because it produced a result that looked
like a win.

`mean(w) = w_0 + O(s*sigma/sqrt(N))`, so the ensemble mean essentially IS the real
pretrained model. A generator that has collapsed toward its per-family mean therefore
scores dPPL ~ 0 -- BETTER than an honest sample -- while generating nothing.
Measured on `emb3` (job 9894473), dPPL % and code-space dispersion side by side:

| arm | dPPL @k99, gpt2 | code rms (real 0.995) | mean pairwise (real 14.06) |
|---|---|---|---|
| `gauss_codes` | +0.341 | 1.0007 | 14.04 |
| `flow_codes` | **+0.008** | **0.249** | **3.18** |
| `flow_latent` | +0.089 | 0.680 | 9.21 |
| `generate` | +0.267 | 0.627 | 8.78 |

Read on dPPL alone, `flow_codes` beat the Gaussian null by 43x and "the flows earned
their keep". Read with dispersion, `flow_codes` emits samples a QUARTER the right
size and `gauss_codes` is the only honest generator in the table -- it loses on dPPL
*because* it is honest. **The dPPL ranking of generative arms is anti-correlated with
sample fidelity here.** Same shape as misstep 13, one level up: a metric that ranks a
degenerate output above a faithful one.

`gauss_codes` was added specifically as the null model for whether a flow learned
structure, and it CANNOT detect this, because a collapsed flow beats it by
construction. Use `diag_flow_dispersion.py` (cheap: code space only, no ensemble, no
`inverse_transform`) before reading any generative dPPL.

The effect is sharper on a TYPICAL member, where the target is a degraded model and
the ensemble mean is genuinely better than it. `gpt2_medium@k99`, sample_idx 5
(job 9894474):

| arm | code rms | dPPL |
|---|---|---|
| `flow_codes` | 0.249 | **-0.508%** |
| `flow_latent` | 0.680 | -0.427% |
| `generate` | 0.627 | -0.250% |
| `gauss_codes` | 1.001 | -0.177% |

The most collapsed arm shows the largest apparent improvement and the only honestly
dispersed arm the smallest. Not perfectly monotone in rms -- `flow_latent` (0.680)
beats `generate` (0.627) -- so do not read this as a clean rank correlation; the
extremes are what matter.

**Why the flow contracts.** See **misstep 21** — it is over-fitting, and a smaller
net fixes it entirely.

> **Superseded.** This entry originally argued the contraction was irreducible
> finite-sample regression, on this reasoning: with source and target both isotropic
> at rms ~1 the Bayes-optimal field `E[u|x_t,t] = (2t-1)/a(t) * x_t`,
> `a(t) = (1-t)^2 + t^2`, is norm-preserving, but `a(t)` dips to 0.5 at `t = 0.5`, so
> the true path contracts mid-flight and must re-expand; a model fits the shrinkage
> more easily than the re-expansion. The algebra is right and the conclusion was
> wrong. The controlled sweep in misstep 21 shows a 51k-parameter net at 500 epochs
> gives rms 1.039 at k=99, so nothing here is irreducible. Kept only so the argument
> is not re-derived and re-believed.

**There are TWO independent contractions, and they need separating.** Measured
`code_rms_ratio` across the real run (emb3, jobs 9894473/4) and the tiny smoke
(job 9912419), where 1.0 is correct:

| arm | tiny k=6 | tiny k=11 | emb3 k=50 | emb3 k=99 |
|---|---|---|---|---|
| `gauss_codes` | 0.99 | 0.95 | 1.005 | 1.001 |
| `flow_codes` | 1.26 | 1.02 | 0.48 | **0.25** |
| `flow_latent` | 0.34-0.60 | 0.24-0.33 | 0.74 | 0.68 |
| `generate` | 0.35-0.41 | 0.31-0.38 | 0.59 | 0.63 |

1. **The flow's own contraction looked k-dependent, and that reading was
   confounded.** `flow_codes` runs 1.26 -> 1.02 -> 0.48 -> 0.25 for k = 6, 11, 50, 99
   — monotone, and tempting. But the tiny run also used a smaller net and 36 rows
   against 300, so k was never isolated. The sweep that isolates it (misstep 21)
   shows **net capacity, not k**, is the cause: at fixed k=99 and fixed rows, width
   alone moves rms from 1.039 to 0.158. Do not attribute this to k.
2. **Every arm routing through the VAE DECODER contracts, at every k, in both runs.**
   `generate` (VAE prior, no flow at all) sits at 0.31-0.63 everywhere. That is
   ordinary prior-posterior mismatch -- `decode(z ~ N(0,I))` lands inside the
   aggregate posterior rather than covering it -- and it is why `flow_latent` is
   contracted even where `flow_codes` is not. It is NOT the flow's fault, and no
   amount of flow training fixes it.

So `flow_latent` inherits two contractions and `flow_codes` only one. Reporting a
single number per arm hides that; the decomposition is what says where to look.

Corollary worth keeping: in the tiny run the four generative arms gave dPPL -0.837,
-0.861, -0.864 and -0.839 while their rms spanned 0.24 to 1.02. **dPPL is not merely
mis-ranking dispersion there, it is nearly blind to a 4x difference in it.**

**Consequence for Experiment 1: measure dispersion, but do not fear the row count.**

> **Retracted.** This entry originally warned that Pythia's ~143 revisions land in
> "the same data-starved regime", so contraction was expected there too and the
> mitigation would be more samples or a narrower code space. That followed from the
> superseded mechanism above and is wrong. Capacity is the lever, and it is free —
> `--hidden_dims 64 128 64` already gives correct dispersion at k=99 on 300 rows.
> Nothing here argues against `pythia_160m` revisions, and nothing here argues for
> preferring k=N/2 over k=N-1.

What DOES carry over: always report `code_rms_ratio` alongside a generative dPPL, and
gate early stopping on val loss (`--holdout_frac > 0`). Both are now defaults.

**20. A benchmark delta smaller than a few questions is not a measurement.**
*(2026-09-03)*
At `--bench_n_questions 200` the quantum of an accuracy delta is 1/200 = 0.005.
Measured on `emb3` (jobs 9894475/6), every delta across MMLU / HellaSwag / GPQA, all
three families, both ranks and both targets, came in between 0.000 and -0.020 -- i.e.
**0 to 4 questions flipping**. That is the correct result: the perturbations are tiny
(relL2 4e-4 to 3e-3), so the models are barely changed. But a column of "-0.0100"
values invites reading a sign and a ranking off noise. `report_stack.caveats()` now
states the quantum and refuses the column when every delta is within ~4 questions of
zero.

Note this is a PAIRED comparison -- same questions, same model, perturbed weights --
so the relevant scale is the flip count, not the naive binomial SE (~0.031 at p=0.25,
n=200). Most answers cannot flip at all at these perturbation sizes.

Also: **MMLU and GPQA report `acc_norm` identical to `acc` in every cell**, because
they are scored on single-letter choices where length normalization is a no-op.
HellaSwag's genuinely differ (0.330 vs 0.360 baseline). `report_stack` now skips a
duplicate `acc_norm` table rather than doubling the report with zero information.

**21. The flow collapse is OVER-fitting, and best-train-loss selects for it.**
*(2026-09-03)*
Misstep 19 established that a collapsed generator posts a flattering dPPL. This is
the cause, and it is the opposite of what I first wrote there. Controlled sweep
(`diag_flow_capacity.py`, job 9912696): k=99, M=300 rows, same codes, same seed, same
space, same lr/batch -- only net width and epochs vary.

| width | params | epochs | train loss | sample rms ratio |
|---|---|---|---|---|
| 64,128,64 | 51,043 | 500 | 1.846 | **1.039 (correct)** |
| 64,128,64 | 51,043 | 8000 | 1.568 | 0.959 |
| 256,512,256 | 361,699 | 2000 | 0.904 | 0.360 |
| 256,512,256 | 361,699 | 8000 | 0.651 | 0.183 |
| 512,1024,512 | 1,234,659 | 2000 | 0.650 | **0.158 (worst)** |

Same pattern at k=50. Three things follow, and the first two correct misstep 19:

1. **Training loss falls monotonically while the samples die.** Lower loss = worse
   generator. So this is over-fitting, not the irreducible finite-sample regression I
   described in misstep 19 -- and "more capacity may make it worse" understated it:
   capacity is the *cause*.
2. **A small net already solves it.** 51k parameters at 500 epochs gives rms 1.039 at
   k=99. The fix was not more rows or a narrower code space; it was less model. I had
   the diagnosis half right and the remedy backwards.
3. **`train_flow.py` was selecting the collapsed checkpoint on purpose.**
   `--holdout_frac` defaulted to 0.0, so `gate = val if val is not None else tot`
   fell through to TRAIN loss, and best-train-loss is monotonically the most
   collapsed point on the curve. The old default's stated justification -- "the flow
   needs to fit the training set before generalization is a meaningful question" --
   is exactly backwards: fitting the training set IS the failure mode.

Fixed: `--holdout_frac` now defaults to 0.15 so the gate is val loss; `--dispersion_n`
logs the sample-spread ratio every print epoch (**the only logged number that gets
worse as the loss improves**, which is why omitting it let this ship); the final ratio
is measured on the SELECTED weights and sealed into `flow_meta.json`; a sub-0.8 ratio
prints a loud block naming the smaller net. `slurm_flow_run.sh` defaults to
(64,128,64) at 500 epochs with the calibration table in its header.

The `flow_k*_collapsed/` directories and `stack_eval_results*_collapsedflow.json` in
`runs/emb3/` are the preserved evidence. Do not delete them; they are the only
artifacts showing the failure, and `flow_capacity_k*.json` is the controlled sweep.

### About the code

**16. Do not read a codes file with `np.memmap` and an explicit shape.**
`np.memmap(mode="r", shape=...)` **succeeds silently** when the requested shape is
smaller than the file. It raises only when the shape is larger. So a stale
`(150, 97)` file read as `(100, 50)` returns the first 5,000 floats as if they were
the current codes. Use `dual_pca.load_codes`, which validates the shape.

**17. Do not gate a pipeline stage on `os.path.exists` alone.**
Every stage in the legacy pipeline did this, with no version check, so an artifact
directory from a different layout gets resumed instead of rejected. Add a
`layout_version` and hard-error on a mismatch.

---

## 6. Environment facts

1. **Never run torch or transformers on the user's laptop.** Use Explorer through
   `ssh explorer`. Submit every job with `sbatch`.
2. **Explorer login nodes kill heavy processes.** Even `conda activate` gets killed
   there. Submit even a small numpy test as a job. `slurm_tests.sh` exists for this.
3. **Write large files to `/scratch/biggs.s`.** Never write to `$HOME`. `/tmp` on
   Explorer refuses writes.
4. Partitions: `gpu` (8 h), `gpu-short` (2 h), `short` (CPU, 2 days).
5. Sync code with `rsync -az <files> explorer:/home/biggs.s/llm_vae/`. macOS ships an
   old rsync, so `--info=stats1` fails.
6. To pass arguments through `sbatch --export`, expand the variable unquoted in the
   script. `python -u "$TEST"` treats `"file.py --flag"` as one filename.
7. `HF_TOKEN` is needed for the gated GPQA dataset — by `MC_EVAL=1` (legacy) and by
   `eval_stack.py --bench gpqa` / `slurm_stack_bench.sh` (current). It is exported
   from `~/.bashrc`, so `sbatch` inherits it; no drop-in block is needed. **Never
   hardcode it, and never interpolate it into a log line.** `${HF_TOKEN:-UNSET}`
   expands to the VALUE when the variable is set, so the obvious one-liner leaks it
   into a log on shared scratch. Print `${#HF_TOKEN}` instead.
   *A token was leaked into an assistant transcript on 2026-09-03 by exactly that
   mistake and should be rotated if it has not been.*

---

## 7. Open questions

1. Is a steep spectrum across Pythia revisions real, or do checkpoints drift into near-orthogonal directions? Experiment 1 answers this. Nothing else should be built until it does. Scott: This obviously must be the caase for near-terminal checkpoints, by the same reasoning that PEFT literature follows. The reason I am slow to accept using checkpoints as data is because I am concerned that they are too far from the terminal state to be high-quality LLM weights. 
2. ~~Should `D` include the embedding matrix and the LM head?~~ **RESOLVED 2026-09-02: yes, and it is done.** Derived as the complement of the `layers_attr` subtree, so no per-arch config and weight tying is free. `ENSEMBLE_LAYOUT_VERSION` 1 -> 2. Side effect worth knowing: it invalidated the noise calibration and moved `s` from 1e-2 to 3e-3, because embeddings widened each family's internal spread of per-tensor stds (misstep 18). Scott: Yes, absolutely it should. 
3. Is cross-family transfer a goal? Per-family PCA forecloses it. See Experiment 4. Scott: Fuck yeah it is, that would be sick dude.
4. Does the VAE earn its place? At `k = N-1` the PCA is already exact, so the VAE only adds error. Its value has to come from producing a smooth, well-conditioned latent for a generative model. That claim is untested. Scott: We'll see, but let's keep the machinery intact so we can investigate relations with LS-Merge. It might be needed to regularize and condition the PCA codes for smoother flows. 
5. Is 32 latent dimensions right for 99-dimensional codes? The VAE currently compresses 99 to 32. Nobody has swept this. Scott: Meh, it's probably fine. We can check it later. 
6. Should the augmentation noise be scaled per parameter group rather than by one global `weight_std`? Misstep 18 shows the global version over-perturbs small-magnitude tensors and that this, not the logit-facing fraction, is what decides a family's fragility. `s * std(tensor)` would make `s` mean the same thing everywhere, at the cost of redefining the ensemble (and so invalidating every fingerprint). Only worth doing on a run that starts with it.
7. Why does the VAE decoder contract? `generate` (z ~ N(0,I) -> `decode_cfg`, no flow anywhere) emits codes at 0.59-0.64x the correct RMS at every rank, in every run, including with a healthy flow. That is ordinary prior-posterior mismatch and it is the reason `flow_latent` is contracted where `flow_codes` is not. Untouched by the flow fix. It is the next thing to fix if latent-space sampling is wanted, and it also bears on open question 4 — a decoder that does not cover its own aggregate posterior is not "conditioning the codes for smoother flows", it is shrinking them.
