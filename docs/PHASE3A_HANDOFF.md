# Phase 3A handoff — the generative machinery on a real zoo

2026-09-09. Track A of `docs/PHASE3_LAUNCH_PROMPT.md`, run against the two Phase 2
Mini zoos. `RESEARCH_PLAN.md`, `CLAUDE.md` and `docs/PHASE2_RESULTS.md` still govern;
this file is what happened when the generative half was pointed at real data for the
first time.

**Read §2 before reading any number in §3.** The headline statistic is new this phase
and it is not the one the repo used before.

---

## 1. Status

| | state |
|---|---|
| `eval_stack.py` can evaluate a zoo | **done** — it could not, at all, before this phase |
| π-conditioning of the flow (§6.2) | **done**, 20 tests green, backward compatible |
| Member-subset bases, so the holdout is outside the span | **done**, 4 bases × 2 β |
| §6.4 retrieval baseline | **done**, pure stdlib, reproduces offline |
| A1 dispersion, 16 cells | **done** |
| Conditioning diagnosis (12 cells) | **done** — the main result of this phase |
| W&B split by workstream | **done** |
| Retrieval-radius sweep, 120 cells | infrastructure **done**, numbers **void** (§6) |
| ΔPPL vs `gauss_codes` on a zoo | **not started** — needs `--pi` in `eval_stack.py` |
| Generated-model harness | **not started** |
| §6.5 slope on generated models | **not started** |
| β-selection appendix, report figures | **not started** |

Nothing in this phase licenses a generative claim. What it licenses is: the machinery
runs, reconstruction is exact, and the conditioning axis can be made live.

---

## 2. How the impact of the conditioning vector is measured

This is the part to understand before anything else, because the statistic the repo
already had — `code_rms_ratio` — **cannot see conditioning at all**, and under
π-conditioning it actively misleads.

### 2.1 The space

Everything here is in **normalised PCA code space**, not weights and not perplexity.
A member is a point in `D = 51,475,968` dimensions; the Gram PCA projects it to `k`
coordinates; `CodeStats.normalize` then subtracts the per-family mean and divides by
one per-family scalar. So the real training codes have RMS ≈ 1 by construction
(measured 0.995), and the flow's target `x1_all` is that matrix.

Two properties of this normalisation matter later. The mean is per-dimension, so the
codes are centred. The scale is a **single scalar**, not per-dimension, so PCA's
variance ordering survives — component 0 still dominates component 98.

### 2.2 The old statistic, and exactly what it measures

```
code_rms_ratio  =  RMS(generated codes) / RMS(real codes)
```

`train_flow.py:_rms_ratio`. It detects one failure: **shrinkage toward the ensemble
mean**. That failure matters because `mean(w)` is itself roughly a good model, so a
collapsed generator posts a *flattering* ΔPPL while generating nothing — misstep 19,
where the collapsed arm beat the Gaussian null by 43× while emitting at 0.25× RMS.

What it cannot see: whether the spread is in the right directions, whether samples
are diverse from one another, and — the point of this phase — **whether the generator
is answering the question it was asked**. A flow that ignores π entirely still emits
the correct total spread and reads a perfect 1.0.

### 2.3 Why it inverts under π-conditioning

The pooled ratio compares sample spread to the spread of the *whole* training set.
With constant conditioning those are one population and 1.0 is right. Condition on π
and they are not: samples at a fixed π carry the **conditional** spread, while the
target carries the **pooled** one, which also contains all the between-mixture
variation.

Measured on this zoo, `pooled / within = 3.07`. So a **perfectly conditional** flow,
sampled at one mixture, emits `1/3.07 = 0.33 ×` the pooled target — numerically the
same reading misstep 19 recorded for genuine collapse.

Left unfixed, a correct π-conditioned flow and a catastrophically collapsed one print
the identical `[COLLAPSED]` tag. `train_flow.py` therefore emits **two** numbers with
`code_rms_reference` naming which is which:

- `code_rms_ratio` — samples drawn at the **training π's**, so the conditioning
  distribution matches the target's and ~1.0 stays correct in either mode. Continuity
  with every historical number.
- `code_rms_ratio_at_fixed_pi` — the new one, below.

### 2.4 The conditional statistic

```
within_rms = RMS( X_anchor − mean(X_anchor) )     over ONE anchor group
at_pi      = RMS( S − mean(S) ) / within_rms      S = 64 samples at that anchor's π
```

The denominator is the load-bearing part. The four branches of an anchor saw an
**identical** data mixture and differ only by data order and seed, so their spread is
the irreducible noise of "train a model on this mixture." That is the width a correct
conditional distribution should have. This is exactly what RESEARCH_PLAN §6.3
provisioned the 20 anchors for; the 80 singletons have one member each and cannot
supply it.

Three landmarks:

| `at_pi` | meaning |
|---|---|
| ≈ 0 | point mass per mixture — **memorisation / collapse** |
| ≈ 1 | correct conditional width |
| ≈ 3.07 | emitted the whole ensemble's spread — **π was ignored** |

### 2.5 The headline number: recovery

`at_pi` is awkward to compare across ranks because the denominator moves with k
(truncation removes within-mixture noise: `within/pooled` is 0.320 at k=99 but 0.022
at k=10, so a raw `at_pi` of 23 at low rank is a shrinking denominator, not a broken
flow). So convert to a scale-free quantity via the law of total variance:

```
Var_total = E_π[ Var(x | π) ]  +  Var_π[ E(x | π) ]

fraction explained by π   =  1 − E[Var(x|π)] / Var_total
                          =  1 − (cond_rms / pooled_rms)^2
                          =  1 − (at_pi · within/pooled)^2      <- what the flow does

ceiling at this rank      =  1 − (within/pooled)^2              <- what the DATA allows
recovery                  =  flow / ceiling
```

`recovery` is the number to quote. 0% = π ignored; 100% = the flow reproduces all the
conditional structure the data contains at that rank.

**Why this is a real conclusion and not "the widths happen to match":** if π were
shifting the conditional *mean* while leaving the conditional *width* at the marginal
width, the marginal would have to be strictly wider than the conditional. We measure
them equal. So the second term of the decomposition is ~0 and the conditional mean is
not moving. That is what licenses saying "π is ignored" rather than "π narrows nothing."

### 2.6 The pairing control

```
pi_code_distance_corr = corr over all pairs (i,j) of  ‖π_i − π_j‖₁  vs  ‖code_i − code_j‖₂
```

No flow is involved — a property of the data alone. Measured **+0.63 … +0.69**.

It exists because a scrambled row→member→π join trains the flow on shuffled
conditioning and yields a *flat* §6.5 slope, which after the fact is
indistinguishable from an honest negative result. This reads ~0 if the rows are
shuffled, and it fails in one epoch instead of after five hundred.

### 2.7 What none of these can do

All of the above is **code-space geometry**. None of it says a generated model is any
good; that is ΔPPL, and it is not yet measured on a zoo.

More importantly — and this is the limit that decides the paper:

> `at_pi ≈ 1` is consistent with the flow having learned `p(code | π)` **and** with it
> having memorised the four anchor members at that π. Sampling a mixture of four
> memorised deltas produces exactly the right conditional spread.

No dispersion statistic can separate those. **Only a held-out π scored against the
retrieval baseline can.** That is why §6.4 is load-bearing rather than a nicety, and
why it was built before any ΔPPL was read.

---

## 3. What was measured

### 3.1 A1 — dispersion across the 16 (β × grid × k × cond_mode) cells

Pooled `code_rms_ratio` 0.85–1.10 everywhere: **nothing is collapsed**.
`pi_code_distance_corr` +0.63…+0.69: **the join is correct**.

Conditioning, at the committed default of 500 epochs:

| rank | recovery, β=0.15 | recovery, β=0.30 |
|---|---|---|
| k=N−1 (99 / 91) | 1.8% / 2.9% | ~0% / 0.1% |
| k≈N/2 (50 / 46) | 27.6% / 25.2% | 17.6% / 26.6% |

The pooled ratio reads healthy in **all sixteen**, so this is invisible without §2.4.

### 3.2 The conditioning diagnosis — the main result

Twelve cells, β=0.15, grid A. Scored by recovery (§2.5).

| cell | k | epochs | hidden | at_pi | pooled | recovery |
|---|---|---|---|---|---|---|
| rank_k99 | 99 | 500 | 64,128,64 | 3.100 | 1.061 | 1.8% |
| long_k99_e8000 | 99 | 8000 | 64,128,64 | 2.804 | 1.336 | 21.7% |
| **long_k99_e8000w** | 99 | 8000 | 256,512,256 | 1.796 | 1.136 | **74.6%** |
| rank_k50 | 50 | 500 | 64,128,64 | 2.654 | 0.948 | 27.6% |
| drop_000 | 50 | 500 | 64,128,64 | 2.550 | 0.930 | 34.1% |
| width_256 | 50 | 500 | 256,512,256 | 1.751 | 1.027 | 75.3% |
| steps_e2000 | 50 | 2000 | 64,128,64 | 2.362 | 1.115 | 45.2% |
| **steps_e8000** | 50 | 8000 | 64,128,64 | 1.348 | 1.159 | **90.2%** |
| **long_k50_e8000w** | 50 | 8000 | 256,512,256 | 1.243 | 1.144 | **93.5%** |
| long_k50_e20000 | 50 | 20000 | 64,128,64 | **0.955** | 1.081 | 101.1% |
| rank_k25 | 25 | 500 | 64,128,64 | 22.131 | 0.831 | 53.0% |
| rank_k10 | 10 | 500 | 64,128,64 | 23.201 | 0.823 | 74.0% |

Four things follow.

**(a) It was underfitting, not expressivity.** 85 rows at batch 64 is ~2 batches per
epoch, so the committed 500 epochs is **~1000 optimiser steps**. DeepWeightFlow ran
30k. The π-MLP was never the constraint: it is a 2-layer MLP on a 5-dimensional input
and the structure to represent is ~4-dimensional (`dim(Δ⁴)`, PHASE2_RESULTS §2b).

**(b) Rank is a convergence-rate axis, not a structural one.** k=99 looked hopeless at
1.8%, but reaches 74.6% with the wide net at 8000 epochs. Lower k converges faster
because truncation discards the noise directions; it does not unlock anything k=N−1
cannot reach.

**(c) misstep 21 does not transfer to real data as written — but its mechanism does,
in a different statistic.** On the noise ensemble, over-fitting showed up as
**marginal collapse** (pooled rms 1.039 → 0.158). Here, pooled rms *rises* to 1.08–1.16
under exactly the treatments the rule forbids. But at 20000 epochs `at_pi` falls to
**0.955 — below 1**, i.e. the conditional distribution is now *tighter* than the real
within-mixture variation, while the pooled ratio sits at a healthy 1.081 and sees
nothing. Same over-fitting, different symptom:

> noise ensemble → over-fitting collapses the **marginal**, pooled rms catches it.
> real zoo → over-fitting over-tightens the **conditional**, only `at_pi` catches it.

**(d) The working range is ~8000 epochs.** 90.2% at (64,128,64), 93.5% at
(256,512,256), with `at_pi` still above 1. 20000 crosses into over-tightening.

**Caveat, stated plainly:** `within_rms` is estimated from **one anchor group of four
members**, and anchors are the five one-hot vertices — the most extreme mixtures on
the simplex. The k-dependence and the epoch-dependence are trustworthy (consistent
across many independent cells), but the absolute scale is a 4-sample estimate. Treat
k=25 and k=10 as directional only: their denominators (0.031, 0.022) are small enough
to be noise-dominated.

### 3.3 §6.4 retrieval baseline — the interior holdout is saturated

`scripts/report_retrieval.py`, pure stdlib, reproducible off-cluster from
`reports/data/n100/`. Scalar `L(model, π) = Σ_d π_d ln ppl_d`; noise floor σ = the
within-anchor spread of `L`.

β=0.15, σ = 0.0076 nats:

| cell | nearest train L1 | ceiling | retrieval | H = retr − ceil | H/σ |
|---|---|---|---|---|---|
| dirichlet_050 | 0.261 | 3.8067 | 3.8008 | −0.0059 | −0.8 |
| dirichlet_075 | 0.173 | 3.8280 | 3.8376 | +0.0095 | +1.2 |
| dirichlet_043 | 0.239 | 3.9175 | 3.9157 | −0.0018 | −0.2 |
| dirichlet_048 | 0.344 | 3.7454 | 3.7245 | −0.0209 | −2.7 |
| **anchor_math** | 0.568 | 3.8556 | 4.0566 | **+0.2010** | **+26.4** |

β=0.30 is the same shape; the vertex reads +0.2283 (+34.4σ).

**On all four interior holdouts the nearest trained neighbour is as good as — in three
cases better than — the model actually trained there.** This is structural, not bad
luck: `mixtures.holdout_split()` picks the four singletons **nearest the barycentre**,
the densest region of the simplex. Only the vertex has headroom.

Pre-registered rule, enforced in code: report the recovery fraction
`ρ = (L_retr − L_gen)/H` only where `H > 3σ`. Otherwise print `NOT RESOLVABLE`.
Unguarded, ρ explodes with arbitrary sign on exactly the cells where nothing can be
concluded (dirichlet_043 has H = −0.002).

---

## 4. Code changes

| file | change |
|---|---|
| `scripts/eval_stack.py` | zoo compatibility (`build_zoo_model` + `"gpt2"` tokenizer); **unconditional target write-back**; `--wandb_project/--wandb_group` |
| `src/llmzoo/gen/flow.py` | `cond_mode ∈ {family, pi}`, `pi_mlp`, `pi` threaded **last** through `loss`/`integrate`/`sample`/`round_trip`/`sample_codes`/`round_trip_codes`; sealed `cond` block; load-time mode/weights refusal; `FlowModel.cond_mode` |
| `scripts/train_flow.py` | `--cond_mode`, `--exclude_within_l1`/`--center_pi`, dual dispersion statistic, `pi_code_distance_corr`, `_pi` output-dir suffix |
| `src/llmzoo/artifacts/bundle.py` | `build_pi_matrix` / `load_pi_matrix` / `validate_pi`; `member_idxs` threaded into the rebuilt dataset |
| `src/llmzoo/data/ensemble.py` | `member_idxs` on `ZooSource`/`EnsembleDataset`, `members` property, `_load_meta` now compares `zoo_dir` **and** `member_idxs` |
| `src/llmzoo/artifacts/io.py` | `member_idxs` inside `ensemble_fingerprint`, conditionally |
| `src/llmzoo/pca/gram.py` | `inverse_transform_many` — batched cohort decode |
| `scripts/train_stack.py` | `--members all|train|<list>`, `resolve_members` |
| `scripts/report_retrieval.py` | **new**, §6.4, pure stdlib |
| `src/llmzoo/wandb_utils.py` | `WANDB_PROJECT` read explicitly, `project=`/`group=` kwargs, `add_wandb_args` |
| `tests/` | 20 π checks + subset-fingerprint checks + purity extended to 3 more scripts |

All five test modules green on `cpu`.

---

## 5. Traps found this phase (all now in CLAUDE.md)

1. **`eval_stack.py` could not see a zoo at all**, and its ΔPPL baseline was a random
   model. Before the fix: `original_ppl` 57419.25, `pca_only` 175.58, **Δ = −99.694%**.
   After: Δ = −0.000%, `cos = 1.000000`, `relL2 = 2.2e-07`. A large negative ΔPPL on a
   zoo means the baseline is wrong.
2. **`code_rms_ratio` inverts under π-conditioning** (§2.3). A correct flow prints
   `[COLLAPSED]`.
3. **Over-fitting on a real zoo over-tightens the conditional, not the marginal**
   (§3.2c). The pooled ratio reads healthy while it happens.
4. **`EnsembleDataset._load_meta` never compared `zoo_dir`.** A reused `--run_name`
   could silently score against a different zoo's basis. Now compared, along with
   `member_idxs`.
5. **misstep 21's capacity rule was calibrated on the noise ensemble** and does not
   transfer as written. Re-read its provenance before applying it to real data.

---

## 6. The void sweep — an error to not repeat

All 120 retrieval-radius cells were trained at **500 epochs**, before the diagnosis
showed that to be ~28% recovery. The infrastructure is validated — exclusion counts
scale correctly with radius (0 → 57 members at r=0.8; `vmath` excludes almost nothing
until r=0.6, as a vertex should) — but every `at_pi` reads 2.7–2.9 and the numbers say
nothing. **Sequence the diagnosis before the sweep.**

Artifacts remain at `runs/zoo_b0*_mini_tr92_k*/sweep/<target>_r<radius>/` and should be
overwritten, not read.

---

## 7. Next steps, in order

1. **`eval_stack.py --pi` and `--flow_cond_mode`.** Blocks everything functional. It
   needs π for `--sample_idx` (read from the zoo via `load_pi_matrix`) and a
   `--target_pi` for novel mixtures. `eval_stack.py:276/:390/:423/:441/:454` are the
   sites; `real_code_rms` at :361-368 stays pooled and correct. Also emit
   `real_code_rms_within_pi` so the eval side gets the same dual statistic
   `train_flow.py` now has. **Do not edit while a dependency chain is queued.**
2. **Re-run the sweep at the corrected config** — 8000 epochs, both ranks, since
   §3.2b shows k=N−1 is reachable. ~120 cells.
3. **The generated-model harness.** Sample at a π, `inverse_transform_many` the whole
   cohort in one streaming pass (naive per-model decode is 20.6 GB of reads *each*),
   write a pseudo-zoo, score with `eval_domains.py` so the text is byte-identical to
   the trained models' (`domain_evalset_s7_n64_c1024.npz`). `eval_domains.py` needs
   `--out` and `--no_gate`.
4. **The retrieval verdict.** Feed §3 into `report_retrieval.py --gen`. This is the
   result. Report it on the **vertex** and on the resolvable sweep cells; the sealed
   interior cells cannot carry it.
5. **§6.5 slope on generated models.** `report_singleton_probe.py` needs a `--kinds`
   filter and a Monte-Carlo null — its exact enumeration is `n!` with no guard, so
   n=12 is 4.8e8. Keep exact for n ≤ 8 so the published p = 0.0014 floor still
   reproduces.
6. **β appendix and report figures.**

---

## 8. Decisions needed from a human

1. **The flow's training budget is now a live parameter.** `flow_run.sbatch` still
   defaults to 500 epochs on the strength of a noise-ensemble rule. Recommend 8000 at
   (64,128,64) — 90.2% recovery, `at_pi` 1.348, pooled 1.159 — and **not** 20000,
   which crosses into conditional over-tightening. This should be re-measured per
   scale, not assumed to transfer to Small and Medium.

2. **The zoo composition may be the binding constraint, and it is still changeable.**
   85 distinct π over 100 members, 80 of them seen exactly once. To learn
   `p(code | π)` the flow needs variation *at fixed π*, and it gets that from only 5
   anchor groups. This is CcGAN's regime by construction
   ([2011.07466](https://arxiv.org/abs/2011.07466)). The 20+80 split traded conditional
   estimability for simplex coverage — reasonable when the worry was "five mixtures is
   too few," but Small and Medium (~446 GPU-hr) have **not been run**, so the split can
   still be revisited. If the retrieval verdict comes back "memorised," this is the
   first thing to change.

3. **`RESEARCH_PLAN.md` §6.6 contains a false claim.** It states the repo has
   "conditioning injected at every layer." It does not — `flow.py` concatenates `cond`
   into the **first** hidden layer only, and `vae.py` likewise at the encoder/decoder
   inputs. Per-block injection (adaLN-Zero,
   [2212.09748](https://arxiv.org/abs/2212.09748)) exists nowhere. Either correct the
   sentence or build it; §3.2 says it is not currently the binding constraint, so
   correcting the sentence is the cheaper honest option.

---

## 9. Artifacts

`ARTIFACT_DIR = /work/neu/p2026_0038_neu/$USER/llm_vae` — 344 GB free.

| run | codes | contents |
|---|---|---|
| `zoo_b015_mini_k99` | k=99, 50, 25, 10 | grid A; 2 flows; **12 diagnosis cells** under `diag_cond/` |
| `zoo_b030_mini_k99` | k=99, 50 | grid A; 4 flows |
| `zoo_b0{15,30}_mini_tr92_k91` | k=91 | grid B, 92 train members; 30 void sweep cells each |
| `zoo_b0{15,30}_mini_tr92_k46` | k=46 | grid B; 30 void sweep cells each |

Grid B ensembles fingerprint differently from grid A by construction
(`2da5696d190bff7f` vs `0d35b58cad8d6039`), so the two cannot be crossed silently.

W&B: project `llmzoo-genverify`, groups `b015_k99`, `b015_k50`, … Historical runs stay
in `llm-vae`.

Reproduce the retrieval baseline with no cluster at all:

```bash
python scripts/report_retrieval.py --data_dir reports/data/n100 --arms 0.15 0.30
```
