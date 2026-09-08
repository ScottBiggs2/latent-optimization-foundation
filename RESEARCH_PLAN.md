# Research Plan — the LLM weight zoo

Written 2026-09-06. Supersedes §4 and §7 of [docs/RESEARCH_NOTES.md](docs/RESEARCH_NOTES.md),
which stays authoritative for the 21 recorded missteps (§5) and the environment
facts (§6). Read this one first; read that one before you touch code.

**Audience:** Scott, Bedionita, and anyone joining. This is a plan to argue with,
not a spec. Everything below is costed and most of it is falsifiable.

---

## 0. TL;DR

We have a validated generative pipeline over LLM weights and no real data to point
it at. The fix is cheaper than we thought: **branch ~100 language models off a
shared partially-trained trunk**, which costs ~3× less than independent training,
puts every model in one basin so canonicalization is unnecessary *by construction*
rather than by assumption, and turns the "public checkpoints might be low quality"
problem into an advantage — a checkpoint is a launch pad, not a sample.

The paper is four contributions, three of which need no new science:

1. The first zoo of independently-trained **language models**, and the first
   eigenvalue spectra ever reported for any weight-space zoo.
2. Conditional generation across data mixtures that works — conditioned on the
   mixture simplex itself, so an unseen mixture is a real zero-shot test rather than
   an embedding-space interpolation.
3. The evaluation instrument: a null arm, a dispersion statistic, and bulk spectrum
   statistics — including evidence that flow capacity buys apparent quality by
   collapsing the generator, and a **retrieval** baseline (§6.3) that nobody reports.
4. *(stretch)* Fine-tuning inside the zoo basis: ~100 trainable parameters adapting a
   whole model.

**Total compute for the whole zoo ladder — 50M / 100M / 250M, N=100 each — is
≈114 GPU-hr and 160 GB.** Under two hours on eight of AICR's 28 B200 nodes. The
bottleneck is writing the trainer and agreeing the mixture design (§6.2b), not compute.

Target: **ICLR 2027, abstract 18 Sep 2026, paper 25 Sep 2026.** Tight. §9 says what
gets cut first.

---

## 1. The thesis

> Every weight-space model zoo in the literature is a zoo of task-specific networks
> trained from scratch on one narrow objective. **Nobody has a zoo of language
> models.** So nobody knows whether weight-space structure survives at LM scale, on a
> general objective, or how it changes as models grow.

The position paper ([2605.18632](https://arxiv.org/abs/2605.18632), ICML 2026) states
as its premise that "high-performing models occupy low-dimensional, highly structured
regions of weight space." It is a premise. Nobody has measured it, including us — see
§4.

---

## 2. What we already have (do not rebuild any of this)

| asset | state |
|---|---|
| ensembles → per-family Gram PCA → StackVAE → conditional rectified flow → 8 eval arms → report | runs end to end, versioned, fingerprinted, reloadable |
| 348 automated checks across 5 modules | green |
| a **null arm** (`gauss_codes`: `z ~ N(mean_f, std_f)`) | built, and it is the arm that decides whether any flow number means anything |
| a **dispersion diagnostic** (`code_rms_ratio`, `scripts/diag_flow_dispersion.py`) | built; catches the failure that ΔPPL cannot see |
| **bulk** spectrum statistics (`ev0/median`, effective-rank ratio) | sealed into every artifact |
| a pre-registered null result on a manufactured ensemble | flow ≡ Gaussian to 0.015pp in all 6 cells, exactly as predicted |
| 21 recorded missteps | docs/RESEARCH_NOTES.md §5 |
| CFG on the velocity field **and** on the VAE decoder | two independent mechanisms, both wired |
| whole-stack layout incl. embeddings + LM head | `ENSEMBLE_LAYOUT_VERSION = 2` |

The null result is not a consolation prize. It is the **flat end of the calibration
curve** in contribution 3, and it was pre-registered in a job header before it ran.

### What is broken, and why it is NOT on the critical path

**The VAE decoder contracts.** `generate` (z ~ N(0,I) → `decode_cfg`, no flow
anywhere) emits codes at 0.59–0.64× the correct RMS at every rank, in every run,
including alongside a healthy flow. This is why `flow_latent` is contracted where
`flow_codes` is not.

An earlier draft of this plan put the fix on the critical path, on the theory that a
regularized latent was the mechanism for contribution 2. **It isn't** — see §6.5. The
mechanism is CFG plus proper conditioning plus a net that doesn't collapse, and none of
that needs a latent. `flow_codes` already runs at rms 1.04. So this stays open question
7, the VAE stays an ablation arm, and the sprint runs **PCA → Flow directly**, as
DeepWeightFlow does.

---

## 3. Where we sit in the landscape

| what | who | scale | data |
|---|---|---|---|
| adapter generation from text | Text-to-LoRA, Doc-to-LoRA, Tina | Llama-3.1-8B | adapters, not full weights |
| mid-scale full-weight vision | RPG | ~200M ConvNeXt/ViT | vision zoos |
| full-weight transformer | **DeepWeightFlow** (ICLR 2026) | **BERT-118M** | 100 models **trained from scratch**, unique seeds, Yelp regression only |
| frontier-scale full synthesis | — | open | — |
| **full-weight pretrained causal LM** | **— nobody —** | | |

Confirmed with the DWF authors: the BERT-118M models were trained *ab initio* on Yelp
with distinct random inits, N=100, k=N−1, and TransFusion applied (BERT was chosen
over GPT-2 precisely because its encoder blocks match ViT's, so the TransFusion code
transferred unmodified). They are task-specific regressors that happen to have BERT's
architecture. **They are not language models.**

Two more openings:

* **DWF reports no eigenvalue spectrum and no variance-captured statistic anywhere.**
  Nobody looked. We already log two bulk statistics that don't lie at the rank bound
  (misstep 15b).
* **DWF Appendix K (multi-class generation)** was a late addition: no CFG, basic
  feature injection at a single level, a 3-layer MLP, and deliberately understated
  prose to avoid opening a new contribution during revisions. *The tables in it are
  promising.* Every ingredient it lacked exists in this repo.

We are not competing with DeepWeightFlow. We are extending it, and correcting one of
its conclusions from the inside (§5).

---

## 4. Contribution 1 — the zoo, and the spectrum nobody has measured

### 4.1 The design: branch off a shared trunk

All N models descend from **one initialization and one shared partially-trained
trunk**, then diverge on different data order / mixture for the final fraction β of
training.

Three independent motivations, in the order we'd argue them:

1. **Compute.** Independent training costs `N·F`. Branching costs `(1−β)F + N·β·F`.
   At β = 0.3 that is a **3.3× saving**, and with a *public* checkpoint as the trunk
   the first term is free.
2. **No canonicalization.** Models sharing an init and a burn-in stay linearly mode
   connected ([Frankle et al.](https://arxiv.org/abs/1912.05671)), so permutation
   symmetry is broken *by construction*. We skip TransFusion because it is
   unnecessary, not because we assume it is (which matters a great deal — see §5).
   TransFusion for decoder-only blocks would otherwise be real new code, and it is
   slow.
3. **The checkpoint-quality problem inverts.** A mediocre public checkpoint is a bad
   *sample* and a perfectly good *launch pad*. Every terminal model in the zoo is
   fully trained. Nothing low-quality enters the ensemble.

Honest scope, stated up front rather than conceded under review: this is a **local**
weight manifold around one trunk, not the global weight distribution. That is the
same scope as Text-to-LoRA, and it is the regime that matters for adaptation.

### 4.2 β is a scientific knob, not a hyperparameter

This is the part worth being excited about.

* **β → 0** (branch very late): all branches are nearly identical. The ensemble
  degenerates toward `w_0 + noise` — **exactly the manufactured ensemble we already
  measured**, whose spectrum is flat (`ev0/median = 1.001`, effective-rank ratio
  0.990) and on which a flow provably buys nothing over a Gaussian.
* **β → 1** (branch at init): effectively independent training. Per Frankle et al.'s
  instability analysis, branches that share no burn-in land in **different basins**,
  so the members are near mutually orthogonal — and the spectrum goes **flat again**.
  Canonicalization comes back on the table.

**Both endpoints are flat, for opposite reasons — degeneracy at one end, orthogonality
at the other — and all the exploitable structure is in between.** That is the sharpest
prediction in this plan, it is a single sweep to test, and it reframes the whole
question the field has been asking qualitatively:

> *How far must training runs diverge before the weight distribution has structure a
> generative model can exploit — and how far before it has too much?*

Sweep β ∈ {0.05, 0.15, 0.30, 0.60} and plot effective-rank ratio and the (flow − null)
ΔPPL gap against it. **We already own the β → 0 endpoint**, measured and
pre-registered. If the curve is non-monotonic with an interior optimum, that is the
figure the paper is built around; it also retroactively explains *why* a manufactured
noise ensemble was never going to work, in the same units.

### 4.3 Pre-registered prediction

Write this into the job header before anything runs, as we did for the flow null.

> With C mixtures × S branches each, between-mixture variance dominates within-mixture,
> so the spectrum has ~C−1 large eigenvalues over an N−C floor.
> **Predict `effective_rank_ratio` ≈ 0.2–0.3 and `ev0/median` ≫ 1 at β = 0.3.**
>
> Disproved by a flat spectrum (ratio → 1.0, ev0/median → 1.0), which would mean
> branches drift into near-orthogonal directions and no generative model over these
> codes will generalize.
>
> Gate on the **bulk** statistics. `ev[0]/ev[k−1]` reads 12.09 on data whose true
> ratio is 1.01 (misstep 15b).

### 4.3b The precondition, and it is cheap

**Before fitting any PCA: measure per-domain perplexity across the whole zoo and
confirm the mixture axis is visible.** A `for` loop over N models.

If β is too small every branch is essentially the trunk, per-domain performance barely
moves, and *every* downstream result — the spectrum, the conditioning, §6.4 — comes out
flat for a boring reason that has nothing to do with weight space. This check costs
minutes and can save the entire experiment.

It also gives a principled way to choose β: **the smallest β at which mixture identity
is measurable in the models themselves.** Anything below that is not a zoo, it is the
noise ensemble with extra steps.

### 4.4 Compute

Chinchilla-optimal (20 tok/param), B200 at ~30% MFU. **Assume ±2×; the pilot
calibrates.** Small models are memory-bound and may do worse.

| scale | tokens | `F` (GPU-hr) | zoo of 100 at β=0.3 | disk |
|---|---|---|---|---|
| ~50M | 1B | 0.12 | **~4 GPU-hr** | 20 GB |
| ~100M | 2B | 0.49 | **~15 GPU-hr** | 40 GB |
| ~250M | 5B | 3.09 | **~95 GPU-hr** | 100 GB |
| | | | **≈114 GPU-hr total** | **160 GB** |

**All three scales together are under two hours on eight nodes.** Cheaper than the
single 410M point an earlier draft proposed, and it buys three scaling points instead
of one. There is no reason to do fewer than all three.

**Pilot: N=32 at ~50M is ~1 GPU-hr — eight minutes on one node.** There is no reason
not to have a real spectrum within a day of the trainer existing.

AICR: 224 B200s across 28 nodes, association `p2026_0038_neu` has **no `GrpTRES` and
no `MaxJobs` cap**, `/scratch` has 2.5 PB free. Neither compute nor storage is a
constraint anywhere in this plan.

### 4.5 The scaling axis

Run the identical protocol at **50M / 100M / 250M** with N=100 fixed. **Does the
effective rank of the weight distribution grow with D, or stay flat?** Nobody can
answer this because nobody has the zoos. Three points is a headline figure, and a 5×
span of D is enough to see a trend.

Second, weaker but free, scaling claim: **generation cost is independent of model
size.** The flow's width is `k = 99` whether D is 5e7 or 5e11; it trains in seconds.
Only the basis fit scales, and it scales linearly. DWF calls O(1B) "possible but
untested."

---

## 5. The risk that has to be settled first

Answer 15 from the DWF authors: canonicalization is skipped because DWF showed its
benefit vanishes as flow capacity grows.

Misstep 21, measured later, in this repo, controlled at fixed k / codes / seed / space:

| width | params | epochs | train loss | sample rms |
|---|---|---|---|---|
| 64,128,64 | 51k | 500 | 1.846 | **1.039 (correct)** |
| 64,128,64 | 51k | 8000 | 1.568 | 0.959 |
| 256,512,256 | 362k | 8000 | 0.651 | 0.183 |
| 512,1024,512 | 1.23M | 2000 | 0.650 | **0.158 (worst)** |

**Capacity × duration collapses the flow, monotonically, while training loss improves
monotonically.** The high-capacity arm of the canonicalization ablation is exactly the
regime where samples die and every loss-based signal reports improvement.

### 5.1 What the DWF training procedure actually was

**Confirmed 2026-09-06: DWF used no early stopping and no gating — the flow ran a flat
30k steps, on a [512, 1024, 2048] net.** *"It could have gone longer and kept
improving."*

That answers the question this section originally asked, and it answers it the
unhelpful way. Misstep 21 has two independent halves and only one of them is excluded:

| mechanism | applies to DWF? |
|---|---|
| best-train-loss **selection** picks the collapsed checkpoint | **no** — there was no selection |
| capacity × duration **drives** the collapse, with loss falling throughout | **yes, and nothing was there to catch it** |

Taking the final checkpoint of a long run at high capacity is the bottom-right corner
of the table above. And *"kept improving"* is improving in **loss**, which is precisely
the quantity misstep 21 shows to be anti-correlated with sample dispersion. Dispersion
is the only logged number that gets worse as the loss gets better, which is why
omitting it lets this ship.

**The honest hedge, and it is a real one.** DWF's targets were real zoos with genuine
cluster structure, not a flat isotropic ensemble. On a flat target the optimal
predictor genuinely is mean-ward; on structured data the same over-fitting can instead
present as **memorising training points** — and memorised points are valid models that
would disperse and score correctly. So DWF may well be clean. This is a check, not an
accusation.

### 5.2 The check

- [x] ~~What did DWF gate on?~~ **Nothing. 30k steps, no gate.** (§5.1)
- [ ] **One `std()` over the generated codes from any saved DWF run.** Ratio near 1 →
      clean, and it becomes a one-line sanity result in the paper. Ratio below ~0.8 →
      a finding.

If no samples survive, the re-run is the DWF repo's own generation scripts plus
`scripts/diag_flow_dispersion.py`. Note the asymmetry: **this is a safe thing to find, because
it is our own paper.** Us finding it is a contribution. Someone else finding it is a
problem.

The stake is immediate: that ablation is the entire justification for not building
TransFusion here. §4.1 motivation 2 makes us robust either way, which is a second
reason to prefer the shared trunk — but we should still know.

### 5.3 Consequence for this sprint

**Do not replicate "let it rip."** `scripts/train_flow.py` now defaults to `--holdout_frac
0.15` (val gating), logs `--dispersion_n` every print epoch, seals the final ratio into
`flow_meta.json`, and prints a loud block below 0.8. `slurm/flow_run.sbatch` defaults to
(64,128,64) at 500 epochs. Keep all of it. The small net is not a compromise — at 51k
parameters it is the *only* configuration in the sweep that disperses correctly.

---

## 6. Contribution 2 — conditional generation across mixtures

### 6.1 Only `k` has to match, not `D`

Correcting an earlier error in this plan. The flow's input width is `k`, and
`k = N−1` is set by **ensemble size, which we choose**. Give gpt2, a Mamba/SSM and a
dLLM N=100 members each and all three produce 99-dim codes. `scripts/train_flow.py` already
reads `codes_k<k>/` as a stacked `(M, k)` array with `family_idxs` alongside, so a
**joint multi-architecture flow needs no code changes at all.**

What different `D` actually forecloses is narrower than stated before:

| | blocked? |
|---|---|
| training one flow over several architectures | **no** — only `k` must match |
| conditioning on architecture | **no** — it is another conditioning axis |
| **decoding for an architecture we have no models of** | **yes** — no basis exists |

The basis absorbs the architecture; the code carries position-within-manifold. So the
tractable multi-architecture experiments are:

1. **Does one flow over A architectures beat A separate flows?** Free ablation. Tests
   whether "shape of a training-outcome manifold" is a real shared object or whether
   conditioning is merely partitioning.
2. **Compositional held-out cells.** A grid of architectures × mixtures with cells
   removed: train mixture *m′* on gpt2 only, train mamba on other mixtures, then
   generate `(mamba, m′)`. **Mamba's basis exists, so it decodes.** This is genuine
   compositional generalization and it is the reachable version of what Appendix K was
   after — unlike unseen-architecture generation, which never decodes.

### 6.2 Condition on the simplex, not on interpolated embeddings

**Class = pretraining data mixture `π ∈ Δ`**, e.g. `[0.4 web, 0.3 code, 0.2 math,
0.1 books]`. One architecture per basis, so cross-mixture transfer is arithmetic.

The conditioning input must be **`π` itself**, through a small MLP — replacing
`nn.Embedding(n_families, cond_dim)` at `src/llmzoo/gen/flow.py:273`. ~15 lines.

> **Why not interpolate learned embeddings.** With a discrete embedding table the only
> way to reach a novel mixture is to blend rows (`0.5·e_code + 0.5·e_math`). That is
> weak evidence: the model never saw intermediate points, embedding space is
> unconstrained, and smooth output is what any Lipschitz MLP does. *"Of course it
> interpolates"* is a one-line review. With `π` as input, a novel mixture is **the same
> kind of object** as a training mixture, and the claim upgrades from "the latent space
> is smooth" to "the conditioning map generalizes."

### 6.2b How many distinct mixtures — the one genuinely open design call

Two proposals were on the table and the resolution dominates both.

**Why 5 mixtures is not enough on its own.** The simplex over 5 domains is
**4-dimensional**. Five mixtures samples that 4-manifold at five points, so fitting a
map `Δ⁴ → cond_dim` is wildly underdetermined — every smooth interpolant fits, and
*"we generated a held-out mixture"* becomes indistinguishable from *"the MLP is
continuous."* Switching from an embedding table to a `π`-vector does not fix this if
there are only five distinct `π`. It also caps the science: between-mixture structure
spans ≤4 dimensions, so the flow's job is "model 5 clusters," which it memorises
trivially, and the generative claim is confined to a 4-dim affine hull.

**Why 20 mixtures × 5 branches is not enough either.** It under-provisions the
within-mixture noise floor, which is what tells you whether a generation error is
meaningful at all.

**Resolution — a nested design, at identical compute:**

> **20 anchor models** — 5 mixtures × 4 branches — for the within-mixture noise floor.
> Four branches at five anchors gives 15 pooled dof, ample for *"is between-mixture
> variance bigger than within."*
>
> **80 singletons** at distinct Dirichlet-sampled `π`, for simplex coverage.
>
> **= 100 models, 85 distinct `π`.**

The only thing given up is 16 extra repeats per anchor, which are over-provisioned for
a variance estimate. Place anchors at vertices and edges; hold out **interior points
and one vertex** — interior interpolation and vertex extrapolation are different
difficulties, so having both gives a gradient rather than a single pass/fail.

**If we ship 5 × 20 instead**, that is a legitimate scope choice, but the claim must be
written as *"CFG interpolation between learned classes produces sensible models"* —
Appendix K un-sandbagged, still a contribution — and **not** as *"the conditioning map
generalises."* Write the weaker sentence deliberately rather than discover it in review.

CFG annealing into a novel mixture stays available and is now *also* well-posed, since
`π` is continuous.

### 6.3 The baseline that decides whether this is a result

For each held-out mixture:

| arm | what it rules out |
|---|---|
| a model actually trained on that mixture | the ceiling (~0.5 GPU-hr at 100M) |
| `gauss_codes` conditioned on the novel `π` | the usual null |
| **the trained model whose mixture is nearest to `π`** | **retrieval** |

The third row is the one that matters. If the generated model is no better than *"just
use the model trained on the closest mixture we already have,"* the flow did not
generalize — it memorized and retrieved. It is `gauss_codes` one level up, it is the
objection a reviewer raises unprompted, and **nobody in this literature reports it.**

### 6.4 The evaluation that beats ΔPPL — control, not quality

With `π` as the conditioning input, do not only ask *"is the generated model good."*
Ask whether the requested mixture **controls** the output:

> Plot requested weight on domain *d* against the generated model's measured
> per-domain perplexity advantage on *d*. One line per domain. **If conditioning
> works, those lines have positive slope.**

Two reasons this is the best instrument in the plan:

* It is a **control** test, not a quality test — which is what "did the flow
  generalize" actually means.
* It is **immune to the collapse trap.** A collapsed generator produces flat lines no
  matter how flattering its ΔPPL. This is the direct answer to misstep 19, where the
  most collapsed arm posted the best ΔPPL.

Depends on §4.3b passing: if the zoo's own models do not differ per-domain, the slopes
are flat for reasons that have nothing to do with weight space.

### 6.5 The mechanism, and where the VAE actually sits

Appendix K lacked three things and had none of them: **CFG, conditioning injected at
every layer, and a net that does not collapse.** All three exist in this repo. That is
the mechanism. It needs no latent.

**The VAE is an ablation arm, not load-bearing** — `flow_codes` (raw PCA codes) vs
`flow_latent` (VAE latent), same eval path, same 8 arms, same null. Scott's read is
that raw PCA codes may be rough for a flow and a smoother target would help; the
measured state is the opposite of urgent — `flow_codes` runs at rms 1.04 while
`flow_latent` runs at 0.68, and **the contraction is the decoder's** (`generate` uses
no flow at all). So the latent currently costs dispersion rather than buying smoothness.

Keep it wired for the LS-Merge connection and run it as an ablation. Do not put it on
the critical path, and do not claim it is the mechanism until it beats `flow_codes` on
the same null.

> **Parked fix, if the latent ever needs rescuing.** The decoder contraction is an
> *aggregate*-posterior-vs-prior mismatch, and the VAE's KL constrains `q(z|x)` **per
> sample**, which is structurally the wrong instrument. **SIGReg** (Sketched Isotropic
> Gaussian Regularization, [LeJEPA](https://arxiv.org/abs/2511.08544)) constrains the
> aggregate directly — random projections plus a 1-D normality test per direction,
> Cramér–Wold. If we adopt it: **apply it per-class, never to the global marginal.**
> Applied to the global latent marginal in multi-task settings it compresses the
> task-dependent clusters together, which would present as *"conditioning doesn't
> take"* — i.e. it would look exactly like Appendix K.

---

## 7. Contribution 3 — the instrument

Mostly written. What it says:

1. **Report a null.** `gauss_codes` costs one `inverse_transform` and zero training.
   The field does not report one. On a flat-spectrum ensemble a flow matches it to
   0.015 percentage points — so without it, "the flow works" is unfalsifiable.
2. **ΔPPL cannot police a generative arm on a mean-centred ensemble.** The ensemble
   mean is essentially a good model, so a collapsed generator scores *better* than an
   honest sample. Measured: `flow_codes` beat the null by 43× on ΔPPL while emitting
   at 0.25× RMS. The standard evaluation in this field cannot detect this.
3. **Collapse is over-fitting, and best-train-loss selects for it.** §5's table.
   Dispersion is the only logged number that gets *worse* as the loss improves, which
   is why omitting it let a bad result ship.
4. **Error magnitude does not predict functional damage.** At matched relL2 = 2e-3,
   the VAE's structured error costs `pythia_410m` +0.378% PPL and isotropic noise
   costs ~+0.03% — **10×** — while truncation error at 7.4e-3, nearly 4× larger,
   *improves* PPL. Weight-space L2 ranks these wrongly and by an order of magnitude.
5. **Use bulk spectrum statistics.** `ev[0]/ev[k−1]` reads 12.09 at k=N−1 and 1.006 at
   k=N/2 on the *same isotropic* data. Gate on `ev0/median` and effective-rank ratio.

Framed this way the paper survives the §4.3 prediction failing. That is the point.

---

## 8. Contribution 4 (stretch) — PEFT in the zoo basis

`w(z) = mean + Xcᵀ U_k S_k^{-1/2} z` is **linear in z**, so
`dL/dz = S_k^{-1/2} U_kᵀ Xc (dL/dw)` — one `transform_vector`-shaped contraction.

Materialize `B = Xcᵀ U_k S_k^{-1/2}` once as `(D, k)`; make `z` a leaf tensor;
`w = mean + B @ z`; `torch.func.functional_call` the model on `w`. Autograd does the
rest. ~50 lines. At 100M with k=99 that is 40 GB fp32 (99 GB at 250M) — **resident on
one B200 either way**, and
the per-step cost is two matvecs, free next to the LM's own forward/backward.

**~100 trainable parameters adapting a whole model.** The comparison that makes it a
result rather than a curiosity:

| baseline | controls for |
|---|---|
| random k-dim subspace | is the *subspace* doing work, or just the dimension count? (Li et al.; Aghajanyan et al.) |
| LoRA at matched param count | is a *global* k-dim constraint better than a *per-matrix* low-rank one? |
| single-model trajectory SVD | ["Fine-tuning Happens in Tiny Subspaces"](https://arxiv.org/pdf/2305.17446) — is a *zoo-derived* subspace transferable where a self-derived one isn't? |

Row 3 is the closest prior art and the one to beat: their subspace comes from one
model's own trajectory and is non-transferable by construction. Ours is cross-class.

It also composes with contribution 2: **the flow generates the initialization, ~100-dim
PEFT refines it.** That is the "adaptation cost reduced by orders of magnitude" claim
the field says it cares about.

---

## 9. Plan and cut order

| # | item | depends on | est. |
|---|---|---|---|
| 1 | `scripts/train_zoo.py` (trunk + branch, sbatch array) + `source="zoo"` in `EnsembleDataset` | — | 1–2 d |
| 2 | **Pilot: N=32 @ 50M, read the spectrum** (run §4.3b first) | 1 | ~1 GPU-hr |
| 3 | **GO / NO-GO on §4.3** | 2 | — |
| 4 | Zoo @ 100M, N=100, β sweep | 3 | ~15 GPU-hr |
| 5 | Simplex conditioning: `π`-vector MLP replacing `nn.Embedding` (`src/llmzoo/gen/flow.py:273`) | — (parallel) | 0.5 d |
| 6 | Flows, 8 arms, held-out mixtures + retrieval baseline (§6.3) | 4, 5 | 1 d |
| 7 | **Per-domain slope figure** (§6.4) | 6 | 0.5 d |
| 8 | Scaling points @ 50M and 250M | 4 | ~99 GPU-hr |
| 9 | DWF dispersion re-check (§5) | DWF repo | 0.5 d |
| 10 | Writing | 7 | 5–7 d |
| 11 | *stretch* multi-architecture joint flow + compositional cells (§6.1) | 4 | 1–2 d |
| 12 | *stretch* PCA-space PEFT | 4 | 2 d |
| 13 | *stretch* VAE latent as an ablation arm; SIGReg only if it needs rescuing | 6 | 2 d |

**Cut order if we slip: 13, 12, 11, then 8, then 9.** Items 1–7 and 10 are the paper.

Note items 5 and 13 versus the previous draft: the VAE/SIGReg work moved **off** the
critical path (§6.5) and the conditioning change moved **on** to it. The conditioning
change is half a day; the latent rescue was two.

**Step 3 is the real decision point** and it arrives on day 2, not day 7. If the pilot
spectrum comes back flat, we stop and reframe around contribution 3 rather than
discovering it a week from the deadline. **Run §4.3b before step 3** — if the zoo's own
models do not differ per-domain, the spectrum is uninformative and the answer is a
larger β, not a reframe.

**Deadline reality.** 12 days to abstract, 19 to paper, from 2026-09-06. This is
achievable only because contributions 1 and 3 are mostly built and contribution 1's
compute is an afternoon. It is not achievable if the zoo trainer takes a week or if
we chase §11.

---

## 10. Explicitly out of scope

State these as limitations rather than letting a reviewer state them for us.

* **Generating for an architecture we have no models of.** This is the *only* thing
  different `D` forecloses (see §6.1 — an earlier draft of this plan overstated it and
  ruled out joint multi-architecture training, which is wrong: only `k` must match, and
  `k = N−1` is ours to choose). Without members there is no basis, so a generated code
  has nothing to decode into. **Bedionita's GNN-over-compute-graph architecture
  embeddings are the right mechanism for the conditioning half of this**, but the
  decoding half stays open — that is the natural joint follow-up, not a 19-day item.
  Multi-architecture generation for architectures we *do* have members of is **in
  scope** and is §6.1.
* **Free-standing generators.** `inverse_transform` computes
  `x̂ = mean·(1−Σa) + Σaⱼ·Xⱼ`, so decoding a generated code **requires the whole
  training ensemble** on disk. Mitigable (materialize an explicit rank-k basis: 64 GB
  at 100M/k=99) but state it first. DWF's dual PCA has the same property and does not
  mention it.
* **Global weight-space claims.** We model a local manifold around one trunk. Say so.
* **Frontier scale.** Not attempted, not claimed.

---

## 11. Parked (good ideas, wrong sprint)

* **Reward-finetuning the flow, TempFlow-GRPO style**
  ([2508.04324](https://arxiv.org/abs/2508.04324)). The pitch is strong and specific:
  text-to-image RL must use a noisy learned preference model, whereas **in weight space
  the reward oracle is exact and cheap** — decode, write back, measure held-out loss.
  Their trajectory-branching credit assignment drops straight in. Cost is the problem:
  each reward eval is an `inverse_transform` plus a PPL eval (~30–60 s at 360M), so a
  group of 8 over a few hundred steps is tens of GPU-hours *plus* new code. Second
  paper.
  > **Do not** map training-step to flow-time literally. `x_t = (1−t)x₀ + t·x₁` is a
  > straight segment from an isotropic Gaussian; a checkpoint is not on it. That
  > mapping would also silently break the ODE round-trip diagnostic, which is the only
  > thing separating solver error from model inconsistency.
* **Quality-conditioned flow.** Condition on a scalar quality/step, train on
  everything including mediocre checkpoints, sample at the good end. Cheap (one scalar
  into `_condition`) and it is the *correct* way to use low-quality data. Largely moot
  under §4.1 motivation 3, but it is the fallback if the trunk-branch design fails.
* **Drift matching / SDE samplers.** Verdict: **not now, and for a sharper reason than
  cost.**

  > **An SDE sampler would destroy the diagnostic contribution 3 is built on.**
  > `code_rms_ratio` is informative *because* the ODE is deterministic — spread in the
  > output can only come from spread the model learned. Inject noise at sample time and
  > you manufacture dispersion: a mean-ward drift plus diffusion reads rms ≈ 1.0 and
  > **is literally `gauss_codes`**. We would be building the null and measuring it as a
  > success. Adopting an SDE therefore requires replacing the diagnostic first — a
  > two-sample test (energy distance / MMD) against the real codes, or §6.4's slope
  > figure, which is a *control* test and stays immune.

  As a **training-side regulariser** it is already available and needs no new code:
  `flow.py` has `path_noise`, defaulting to 0.0. The docstring records the cost — with
  `path_noise > 0` the interpolant leaves the straight segment, so the round-trip
  residual stops being attributable to Euler error and `--flow_rt_steps` stops
  separating solver error from model inconsistency. One flag if we want the datapoint.

  And it is mostly moot: **the memorisation fix is already known and free.** 51k
  parameters at 500 epochs gives rms 1.039 (misstep 21). An SDE is more machinery for
  the same outcome, minus a diagnostic.

  Two reasons to keep it on the roadmap rather than delete it:
  1. **It is a prerequisite for the GRPO arm above, not a competitor to it.**
     TempFlow-GRPO's trajectory branching works by injecting one-step SDE noise at
     intermediate latents and denoising each descendant. No SDE, no branching.
  2. If we ship **5 mixtures** (§6.2b), the effective structure is ~4-dimensional over
     100 points and the flow will memorise it easily — at which point "we need an SDE
     to avoid memorisation" becomes tempting. Read that as a symptom of the data
     design, not a modelling problem. The two questions are the same question.
* **Bedionita's 30 JAX GPT-2s @ 8B WebText tokens.** Free real data that exists now.
  Worth one day to convert and spectrum-check even at N=30 (k=29), as an independent
  read on §4.3 before our own zoo lands. Needs: are the inits distinct? are the weights
  actually retrievable? Flax↔PyTorch conversion is fiddly (transposed Linear kernels,
  param-tree naming) but bounded.

---

## 12. Open questions

**Blocking:**

1. **Trunk: public checkpoint or train our own?** Public (Pythia / OLMo-2 / SmolLM2
   intermediates) is free, but most public checkpoints ship **no optimizer state**, so
   every branch restarts Adam's moments and eats a transient — probably harmless since
   it is uniform across branches, but it is a confound in the β sweep and must be
   measured. Our own trunk costs `0.7F`, which at these scales is **0.09 / 0.35 / 2.2
   GPU-hr** — negligible — and it buys full control of the data order plus no
   contamination question about the held-out mixtures. **Lean: train our own at all
   three scales.** The public-trunk argument was about affording 410M, and 410M is out. **Decision:** Train our own.
2. **The corpora.** FineWeb-Edu / StarCoder / OpenWebMath / books / multilingual is
   the obvious 5-domain split, all streamable from HF. Anything already staged on AICR
   or Explorer scratch we should prefer? **Decision:** These suggestions seem fine. 
3. **Mixture count — the nested 20 + 80 design, or 5 × 20?** (§6.2b) This is the one
   call that changes what the paper can claim. Nested costs the same and preserves the
   generalisation claim; 5 × 20 is a deliberate scope reduction to "CFG interpolation
   between learned classes." **Decision:** Lean 20+80 nested design, more interesting. 

*Resolved since the first draft: scale ladder is 50M / 100M / 250M (§4.4); DWF used no
gating at all, 30k flat steps (§5.1); SIGReg and the VAE are off the critical path
(§6.5); drift matching is parked with a verdict (§11).*

**Non-blocking:**

4. Bedionita's zoo: distinct inits? retrievable? (§11)
5. Is this a joint submission with KAIST, and on the same deadline?
6. Do we report the β sweep as a headline result or an appendix? It is arguably the
   most novel single measurement in the plan. **Decision:** It seems like a  β sweep would likely be extremely expensive - even in a reduced setting - and I'm unclear on what the contribution would actually be? It would be interesting if we were doing a lora/weight space understanding paper, but that's not really what we're doing.
7. Multi-architecture (§6.1) as a main-paper arm or a stretch? It is free to *train*
   — only `k` must match — but the compositional held-out cells need an extra
   architecture's worth of zoo, so it is not free to *build*.
8. Who runs what, if the group re-engages. Items 1–2 in §9 are the only serial
   bottleneck; §9 items 5, 9, 11, 12, 13 all parallelise cleanly across people.

---

## 13. Carried forward, unchanged

* **Never run torch or transformers locally.** Submit everything with `sbatch`.
  Explorer login nodes kill even `conda activate`.
* **Rotate `HF_TOKEN`** — leaked into a transcript 2026-09-03 via `${HF_TOKEN:-UNSET}`,
  which expands to the *value*. Print `${#HF_TOKEN}`. Still outstanding.
* **Re-calibrate `--noise_scale` whenever the layout changes.** Nothing else transfers
  (misstep 18). Moot for a real zoo, which needs no augmentation noise at all — but
  `source="noise"` stays as the machinery test and as the β→0 endpoint.
* **`chunk_bounds` must stay a fixed grid** (misstep 10).
* Read `code_rms_ratio` **before** any generative ΔPPL (misstep 19).
* `train_stack.py --k` and `train_flow.py --k` are scalar; `eval_stack.py --k` is
  variadic. A codes-space flow's width *is* k, so one flow cannot span ranks.
* Do not prune `runs/emb3/flow_k*_collapsed/` — the only evidence for misstep 21.
