# Phase 2 progress — the singleton probe, and Mini N=100

Written 2026-09-08, continuing from `docs/PHASE1_HANDOFF.md`. Read that first;
this file records only what happened *after* it.

---

## 0. Decisions taken this session

| question | decision | why |
|---|---|---|
| Throughput work (handoff §3, launch-prompt tasks 2–3) | **dropped** | GPT-2 stays stock and canonical. The one read-only check was worth it and is answered below. |
| Mid-run checkpointing (task 3) | **deferred to Medium** | It was only ever a Medium prerequisite. Small's trunk is ~3.5 h; Medium's is 16.5 h. |
| Phase 2 first scale | **Mini 51M, N=100, at BOTH β=0.15 and β=0.30** | Settle β on the real k=99 spectrum with 85 distinct π, not on a 6-member slope. ~59 GPU-hr because both trunks and 12 members each already exist. |
| Probe arms | **both, concurrently** | +1.1 GPU-hr, and if the slopes had come out flat, one arm could not have separated "β too small" from "the 80-singleton design is wrong". |

---

## 1. The singleton / diversity probe — BOTH ARMS PASS

The gap this closed: N=12 contained **only one-hot anchor vertices**, the maximally
separated mixtures. Phase 2 is 80 Dirichlet singletons whose closest pair sits at
L1 = 0.1725 (measured on the real Phase 2 plan, §4 below). Nothing had tested a
branch at that separation.

Six singletons per arm at `--alpha 8.0`, min pair L1 **0.164** — i.e. *harder* than
Phase 2's worst case. Both arms reuse their β's existing trunk, so this cost
branches only: **3.2 GPU-hr**, 26 min of wall clock.

**The π are bit-identical across the two arms** (same `--plan_seed 0`), so β is the
only difference and the comparison is controlled.

### The statistic (RESEARCH_PLAN §6.5, in miniature)

Per domain *d*, regress measured per-domain PPL advantage on requested weight π_d:

```
adv[i][d] = mean_j( ln ppl[j][d] )  -  ln ppl[i][d]
```

Log space because perplexity composes multiplicatively, so the advantage is additive
and comparable across domains whose absolute PPL differs 18× (code ≈ 9.6, web ≈ 178).
A **control** test, not a ranking test — §6.5's point, and why it is immune to
misstep 19's collapse trap.

The null is **exact**: H0 is that the pairing between a member's π and its measured
perplexities carries no information, the exchangeable unit is the member, so all
6! = 720 permutations of that pairing are enumerated rather than sampled. 30 points
do not support a t-distribution. The p-value floor is therefore 1/720 = 0.0014.

`scripts/report_singleton_probe.py`, pure stdlib. Validated against synthetic ground
truth before use: recovers **+1.422** on data built with true slope +1.5 (p = 0.001,
5/5 domains positive) and **+0.071** on pure noise (p = 0.214) — where 4 of 5 signs
came out positive *by chance*, which is exactly why the decision rule needs the exact
p-value and not the sign count.

### Result

| | β = 0.15 | β = 0.30 |
|---|---|---|
| **pooled slope** | **+0.328** | **+0.693** |
| pooled p (exact, 720 perms) | **0.0014** (floor) | **0.0014** (floor) |
| domains with positive slope | **5/5** | 4/5 |
| branch steps | 1178 | 2356 |
| cost | 1.5 GPU-hr | 2.6 GPU-hr |

Per domain — slope, Pearson *r*, exact one-sided *p*, and the effect over the π range
this probe actually covers (as a PPL reduction):

| domain | β=0.15 slope | *r* | *p* | %PPL | β=0.30 slope | *r* | *p* | %PPL |
|---|---|---|---|---|---|---|---|---|
| web | +0.370 | 0.757 | 0.050 | 4.3% | +0.552 | 0.564 | 0.083 | 6.4% |
| code | +0.158 | 0.550 | 0.161 | 2.6% | +0.414 | 0.471 | 0.185 | 6.6% |
| math | **+0.495** | 0.888 | 0.011 | 6.7% | **+1.202** | 0.945 | 0.008 | 15.4% |
| books | +0.038 | 0.563 | 0.128 | 0.5% | **−0.072** | −0.535 | 0.864 | −0.9% |
| multilingual | **+0.498** | 0.934 | 0.010 | 8.0% | **+1.092** | 0.853 | 0.011 | 16.7% |

**Verdict: β is confirmed at Phase 2's hardest separation, and both 0.15 and 0.30
qualify.** β=0.30 is 2.1× stronger on the pooled slope, which is what more divergence
should buy. β=0.15 is what §4.3's "smallest passing" rule selects and is the cheaper
arm — so the choice is *not* settled by the probe, and both Mini N=100 arms are being
run to settle it on the k=99 spectrum instead (§3).

### Probe geometry — the probe is a LOWER BOUND on Phase 2, and here is why

`scripts/diag_zoo_geometry.py` on the two probe arms, against the Phase 1 anchor
zoos at the same β:

| | anchors, N=12 | **singletons, N=6** |
|---|---|---|
| displacement from trunk, β=0.15 | 16.5% | **5.0%** |
| displacement from trunk, β=0.30 | 28.7% | **12.2%** |
| spread / displacement, β=0.15 | 0.950 | **0.748** |
| spread / displacement, β=0.30 | 0.878 | **0.517** |
| centroid fraction | 0.741 (i.i.d. 0.677) | 0.646 (i.i.d. **0.645**) |

`between/within` is undefined here by construction — an all-singleton zoo has no
within-group pair — and the rewritten script says so rather than silently counting
unique-π singleton pairs as "between".

Read these together and they say something the slopes alone do not. The probe's
singletons **moved a third as far** and **much more in a common direction**. The cause
is mechanical: to match Phase 2's *closest pair* (L1 0.173) at only 6 draws, the probe
needed `--alpha 8.0`, which clusters draws near the **barycentre** — and the
barycentre *is* the uniform mixture the trunk itself was trained on. So the probe's
branches barely changed their data distribution relative to the trunk, and most of
their displacement is the common "keep training on ~uniform data" drift rather than
mixture-specific movement. The centroid fraction sitting at 0.646 against an i.i.d.
0.645 confirms the decomposition: a large shared drift plus an essentially i.i.d.
residual, not a huddle around the mean.

**This cuts in our favour, and should be stated that way.** Phase 2's 80 singletons
are drawn at `alpha=1` — uniform on the simplex, mean pairwise L1 **0.889** against
the probe's 0.330 — so they spread across the whole simplex rather than clustering at
the centre. They will move further from the trunk and more independently than the
probe's did. The probe was deliberately the hardest case for *mixture control* and it
is simultaneously the **weakest** case for weight-space signal; Phase 2's typical
member sits on the better side of both. Do not quote the probe's 0.517 as Phase 2's
expected `spread/displacement`.

### `books` is not a discriminative axis — declare it

At **both** β, books is the outlier, and at β=0.30 its slope is **negative**. Its
measured PPL spans 38.80–39.15 (0.9%) at β=0.15 and 38.95–39.72 (2.0%) at β=0.30,
against 15–17% swings on math and multilingual. Three reasons this is expected rather
than alarming, and it should be stated in the paper rather than conceded:

1. Project Gutenberg is generic English prose, so training on *web* or *math* also
   improves books PPL — π_books has little marginal effect over the other four.
2. Books documents are ~195 KB, so with `HOLDOUT_EVERY = 64` the 64-block held-out
   slice comes from a handful of documents. High variance, low signal.
3. Phase 1 already saw it: **books was the min-SNR domain at β=0.15** in the §4.3
   gate.

This does not threaten the probe — the pooled test is at the exact-null floor either
way — but "five domains" is really four discriminative ones plus a near-flat book
axis, and §6.3's scope paragraph should say so alongside "four of five are English
and the fifth is French; code is Python".

---

## 2. The block-only spectrum — §4.5's confound is DOMINANT, not a footnote

§4.5 flagged the embedding-share confound as something to declare next to the scaling
figure. Nothing could compute the control it prescribes: `spectrum_stats` was buried
in `scripts/train_stack.py` and every fit was whole-stack.

`scripts/diag_block_spectrum.py` (new) needs **no PCA, no GPU and no torch**: a Gram
is a sum over coordinates, so it splits exactly along the layout
`w = [extra | block_0 | ... ]`, giving `C_whole = C_extra + C_block`. Verified against
the fitted `DualGramPCA` spectrum on all three N=12 arms to **~1e-10**.

At Mini, k=11:

| β | region | ev0/median | eff_rank_ratio | variance share |
|---|---|---|---|---|
| 0.15 | whole stack | 134.13 | 0.1420 | 100% |
| 0.15 | **blocks only** | 54.34 | **0.2081** | **9.6%** |
| 0.15 | embeddings only | 150.52 | 0.1352 | 90.4% |
| 0.30 | whole stack | 121.99 | 0.1446 | 100% |
| 0.30 | **blocks only** | 50.36 | **0.2111** | **12.5%** |
| 0.30 | embeddings only | 140.56 | 0.1355 | 87.5% |
| 0.60 | whole stack | 52.77 | 0.1836 | 100% |
| 0.60 | **blocks only** | 17.63 | **0.2906** | **21.1%** |
| 0.60 | embeddings only | 79.43 | 0.1605 | 78.9% |

**The embedding block is 51.0% of `D` and carries 87.5% of the centred variance** (at
β=0.30). So the whole-stack spectrum is essentially an embedding spectrum: its
`effective_rank_ratio` (0.145) tracks the embeddings' (0.136), while the transformer
blocks sit at **0.211**.

That flips which side of §4.4's prediction the measurement falls on. §4.4 predicts
`effective_rank_ratio ≈ 0.2–0.3`; the whole-stack number is **below** that band and
the block-only number is **inside** it, at all three β.

**Caveat, stated as loudly as handoff §2 states it:** at N=12 the design ceiling still
binds — 3 anchor groups span rank 2 at most, so the *absolute* effective ranks are
saturated for both regions. What is **not** ceiling-limited is the variance split,
because a trace ratio just measures how much each region moved. So "embeddings are
51% of D and 87.5% of the variance" is a solid N=12 statement; "blocks have higher
effective rank than embeddings" is suggestive and gets its real test at k=99.

Consequence for the paper: **every spectrum claim needs both curves.** A ladder plotted
whole-stack only would be reporting the embedding share falling 51% → 32% → 15%.

---

## 3. Phase 2 — Mini, N=100, both β arms (RUNNING)

Both arms reuse the calibration's trunk **and its 12 members**, which is worth
~4.5 GPU-hr per arm. `slurm/stage_zoo_reuse.sh` does not take that on trust: it
generates the N=100 plan and refuses unless every step count and every reused
member's π / kind / mixture_id / branch match the source, and unless the reused
members sit in the b200 throughput band (handoff §8 trap 6). Both arms passed:

```
plan agrees with src on total_steps=7854 trunk_steps=6676 branch_steps=1178   (β=0.15)
plan agrees with src on total_steps=7854 trunk_steps=5498 branch_steps=2356   (β=0.30)
12 reusable members verified identical in pi/kind/mixture_id/branch: 0..11
reused members ran at 233.6-272.1 ktok/s (b200 band)
```

The Phase 2 plan, verified independently of §6.3's prose:

```
members      : 100  (20 anchor + 80 singleton)
distinct pi  : 85                            <- §6.3 says 85
anchor groups: 5 x 4 branches                 <- all five domains now have an anchor
singleton pairwise L1: min=0.1725  mean=0.8886  max=1.8824
holdout      : 4 interior [70, 95, 63, 68] + 4 vertex [8, 9, 10, 11] (anchor_math)
steps        : total=7854  trunk=5498  branch=2356   (β=0.30)
```

**The gate is a materially stronger test at N=100 than the one β was calibrated
against.** At N=12 only three domains had an anchor group, so §4.3 condition 1 was
checked on 3 domains; now all five have one and condition 2's noise floor pools 15 dof
instead of 9.

Measured cost, from the probe's actual branch times (singleton branches are slower
than anchors — 5-way interleave plus ~3 min of shuffle-buffer fill):

| arm | branches | anchor 12–19 | singletons 20–99 | GPU-hr |
|---|---|---|---|---|
| β=0.15 | 88 | ~10.5 min | ~15.1 min | ~21.5 |
| β=0.30 | 88 | ~20.8 min | ~26.1 min | ~37.6 |
| | | | **total** | **~59** |

~2 h of wall clock at the 32-GPU ceiling.

---

### 3.1 The wall-clock assumption in §4.6 is wrong, and not by a little

RESEARCH_PLAN §4.6 costs every zoo at **32 concurrent** ("100 branches run in 4
waves"), and handoff §4 inherits it. That is our QOS *ceiling*, not our throughput.
Measured at launch, 2026-09-08 22:5x:

```
b200-batch : 248 GPUs, 222 allocated, 2 more nodes drain* (Dell hardware cases)
             -> ~10 genuinely schedulable
             165 pending jobs, priority 698-882
our priority: 135, ALL of it fairshare (Age contributes 0)
our account : RawShares=1, NormShares=0.0082, EffectvUsage=0.0043
concurrency we actually got: 3
rtx-batch  : 152 GPUs, 152 allocated, 0 free, 52 pending
```

So we are **priority-starved, not resource-starved**, and the levers are backfill
ones, not sizing ones:

1. **Honest `--time`.** The committed 60 min against measured maxima of 15:49 (β=0.15)
   and 27:00 (β=0.30) forfeited every backfill gap shorter than an hour
   (`SchedulerType=sched/backfill`; skill Rule 3). Cut to 28 and 42 min on the
   in-flight arrays via `scontrol update TimeLimit=`, which works on *pending* array
   tasks and is permission-denied on running ones (handoff §8 trap 4).
2. **Honest `--mem`.** `--mem=200G` against a measured MaxRSS of 16.1–17.8 GB. Slurm
   must find the whole request free on one node, so that reservation could not be
   placed on a node with a free GPU and 150 GB free. Cut to 96 GB, in the committed
   sbatch and on the pending tasks. `--cpus-per-task` went 16 → 8 (AveCPU ≈ 0.87
   cores: the tokenizer is single-threaded in the training thread).

**Considered and rejected:** `b200-devel` has 8 free B200s and would satisfy trap 6,
but our cap there is 2 GPUs and 35 jobs were already pending on a partition intended
for short sanity checks. Going 3 → 5 concurrent does not change the character of the
timeline and would be poor citizenship. `rtx-batch` is both full and a *different* GPU
type, so it would violate trap 6 for a zoo whose members 0–11 are already b200.

**What this means for Small and Medium.** Their GPU-hour costs (137 and 726) stand —
those are measured. What does not stand is any wall-clock figure derived from 32
concurrent. Medium at 3 concurrent is **10 days**, not 28 hours. Before committing to
Medium, re-measure the achievable concurrency, and note that the fix is a fairshare
conversation with the cluster owners rather than anything in this repo: our account
holds 1 RawShare and our priority is 135 against a field at 698–882.

## 4. Traps found this session, all now in CLAUDE.md

1. **A singleton mixture opens FIVE HF streams; an anchor opens one.** The very first
   singleton-bearing job died in 13 s on a real `429 ... We had to rate limit your IP`.
   Phase 1 could not have found it: `mixture_stream` takes a `len(streams) == 1` fast
   path for a one-hot mixture, so all 39 calibration jobs resolved one dataset each.
   Fixed twice over — `HF_TOKEN` now reaches jobs (it was already at
   `~/.cache/huggingface/token`, but `HF_HOME=/scratch/$USER` redirects the lookup away
   from it), and `mixtures.open_domain_stream` backs off with full jitter, which
   doubles as the de-synchroniser for an array that all started at once. The token
   changes only the auth header; the corpora are unchanged, so the calibration stands.
2. **`_is_retryable` must not match a dead dataset id.** Retrying a renamed or gated
   corpus six times and reporting a rate limit would hide the most fragile thing in the
   repo and invite the §6.3 edit that silently invalidates β. Tested both directions.
3. **`launch_beta_arm.sh` keyed `ZOO_ROOT` and `--run_name` on β alone.** Neither
   could fail at N=12. Phase 2 Mini would have landed in the calibration's directory
   and `train_zoo.py:621` would have refused every branch; and since every N=100 scale
   has k=99, all three scales wrote `runs/zoo_b030_k99` and later fits would have
   overwritten earlier ones. `THROTTLE` also defaulted to 10 — 10 waves instead of 4,
   which is Medium in 71 h rather than 28.
4. **`diag_zoo_geometry.py` was O(N²) per chunk**: 66 pairs at N=12 (35 s), 4950 at
   N=100 — hours. Rewritten as one streaming pass and one BLAS call per chunk via an
   N×N Gram *of displacements* (raw vectors would mean catastrophic cancellation, since
   members share a trunk and `‖w_i − w_j‖` is 16–56% of `‖w_i‖`). Reproduces the old
   implementation to **4.6e-14** on 3 arms × 12 statistics.
5. **A gate crash and a gate verdict both exit 1.** A gate hit
   `CUDA error: uncorrectable ECC error encountered` on one rtx-batch node and the
   wrapper printed "Gate FAILED ... the response is a larger --beta" — advice to spend
   hundreds of GPU-hours in response to a broken GPU. Now distinguished by artifact:
   `atomic_write_json` does `.tmp` + `os.replace`, so a write always lands a **new
   inode**. (`[ report -nt marker ]` does *not* work — same-second writes tie, verified
   failing on the cluster.)
6. **Anchor repo-root `.gitignore` patterns with a leading slash**, not just rsync
   excludes. Unanchored `artifacts*/` matched at any depth and had left
   `src/llmzoo/artifacts/__init__.py` **untracked** — the only subpackage missing its
   marker.

## 5. Two answers about throughput, for whenever Medium is planned

Both read-only; nothing was changed.

- **Attention is `sdpa` at all three scales**, resolved by HF at model init even though
  `zoo_config()` never sets `attn_implementation` (transformers 4.57.6). So eager
  attention is **not** the explanation for Medium's low MFU, and Medium's ~7 µs/token
  of non-head work at ~15% of peak remains unexplained.
- **The LM head's share of `6·D` equals the embedding fraction exactly**: 50.0% at
  Mini, 31.0% at Small, 14.5% at Medium (measured). Since Medium is 81% of Phase 2's
  bill, a fused cross-entropy would have bought ~2× at Mini and little where it
  matters. Dropping it was correct.

