# Phase 1 handoff — β calibration complete, gate passed

Written 2026-09-08, at the end of the session that ran RESEARCH_PLAN §4.2 and §4.3.
Read `RESEARCH_PLAN.md` §4 and `CLAUDE.md` first; this file records what *happened*,
which of the plan's assumptions turned out to be wrong, and what is safe to do next.

Figures: **`reports/beta_calibration.html`** — open it in a browser. Seven charts, both
themes, hover values, table view per chart, self-contained (no network). Regenerate with
`python reports/build_calibration_figures.py && python reports/make_html.py`.

Chart 1 is the one to look at first: 12 members × 5 domains per β, each column
normalised to its own best member. The block-diagonal pattern *is* the result.
Chart 7 is why β=0.30 rather than 0.15 or 0.60.

W&B: **`scottbiggs2001-northeastern-university/llm-vae`**, groups
`zoo_gpt2_zoo_mini_b015` / `_b030` / `_b060` — 15 runs each (trunk, 12 branches, gate,
spectrum). The HTML exists because the numbers that decide this result are
*cross-run* (a 12×5 PPL block structure, one spectrum per arm) and W&B shows one row
per run.

---

## 1. Headline

**All three β arms pass the §4.3 gate.** Separation 3/3 on every arm; signal>noise
clears the >1.0 threshold on all five domains with min ratios 5.96 / 5.40 / 4.54.

| β | min SNR | worst domain | ev0/median | eff. rank (of 11) | trunk | branch | fingerprint |
|---|---|---|---|---|---|---|---|
| 0.15 | **5.96** | books | 134.1 | 1.56 | 70.6 min | 10.5 min | `a899bcd619f7c017` |
| 0.30 | 5.40 | multilingual | 122.0 | 1.59 | 59.2 min | 20.8 min | `b43542b114177538` |
| 0.60 | 4.54 | multilingual | 52.8 | 2.02 | 35.9 min | 41.6 min | `57259d7268b431e1` |

Total ≈18 GPU-hr (§4.2 predicted 1.9). **β = 0.30 is DECIDED** — see §5.

**β genuinely controls divergence.** `weight_std` spread across the 12 members grows
36× from β=0.15 to β=0.60, so this is not the β→0 degenerate case §4.2 warns about.

**Next action: the diversity / singleton probe at β=0.30.** Ready to launch, §5 has
the exact command. It is the one question N=12 could not answer.

---

## 2. The spectrum result is NOT yet a test of §4.4 — read this before quoting it

`ev0/median = 134` and `effective_rank_ratio = 0.142` look like a strong confirmation
of §4.4. They are not, and the reason is structural:

At N=12, `build_zoo_plan` emits **3 anchor groups of 4 identical-π members**.
Mean-centred, 3 group means span **rank 2 at most**. Measured:

| β | top-1 | top-2 | top-3 | eff. rank |
|---|---|---|---|---|
| 0.15 | 78.5% | 93.6% | 95.0% | 1.56 |
| 0.30 | 77.7% | 93.3% | 94.7% | 1.59 |
| 0.60 | 67.4% | 87.2% | 89.6% | 2.02 |

A 3-cluster design with negligible within-cluster noise gives eff. rank → 2.00
(ratio 2/11 = 0.182). **The measurement is saturated at the design ceiling.** So:

- What it *does* establish: the pipeline detects real cluster structure and separates
  cleanly from the manufactured-noise null (`ev0/median` 1.001, ratio 0.990). A 134×
  separation from the null is instrument validation.
- What it does *not* establish: anything about the dimensionality of weight space.
  §4.4's 0.2–0.3 prediction was written for N=100 with 85 distinct π, where the
  ceiling is far higher. **That test is still open.**
- The one number here that *is* weight space: β=0.60's 2.02 slightly **exceeds** the
  2.00 ceiling, because within-anchor divergence grows with β and adds genuine extra
  dimensions.

Do not write "spectrum is highly structured, §4.4 confirmed" into a draft. The
honest sentence is: "at N=12 the spectrum is design-saturated; the §4.4 test requires
the N=100 zoo."

---

## 2b. Weight-space geometry — and why the gate is only half a criterion

The §4.3 gate answers **"is mixture identity detectable"** and nothing else. It is
*maximised* by a tight within-anchor spread — which is exactly the regime where the
ensemble mean is already a good model, `gauss_codes` becomes a strong null, and a flow
can memorise the code distribution. That is misstep 19's mechanism and §11's warning,
and **no number in `domain_separation.json` sees it.** Selecting β on min-SNR alone
optimises the wrong axis.

So it was measured directly. `scripts/diag_zoo_geometry.py` (new; streams the members,
peak RSS ≈ one member, ~35 s on `cpu`):

| | β=0.15 | β=0.30 | β=0.60 |
|---|---|---|---|
| `‖w_i − trunk‖ / ‖trunk‖` — how much actually moved | 16.5% | **28.7%** | 55.9% |
| `spread / displacement` (√2 = fully independent) | 0.950 | 0.878 | 0.868 |
| centroid fraction `‖w_i−mean‖ / ‖w_i−w_j‖` | 0.741 | 0.738 | 0.719 |
| **between/within group distance, in weight space** | **4.075** | 3.975 | **2.882** |
| shared burn-in | 85% | 70% | 40% |

Read these carefully, because two of them overturn intuitions:

- **β=0.15 is not a memorisation hazard.** 16.5% of the weight norm moved — a large
  change, not rounding. The centroid fraction 0.741 sits *above* the 0.677 that an
  i.i.d. spread gives at N=12, so members are not huddled near the mean, which is the
  condition misstep 19 needs. `spread/displacement` is the **highest** of the three:
  at β=0.15 members moved most independently *relative to how far they moved*. Even
  β=0.15 is 154M tokens of single-domain training, and specialisation moves weights
  fast.
- **β=0.60's higher effective rank is not richer structure.** Between/within distance
  in weight space — the quantity a PCA basis is actually built from — *degrades*
  monotonically: 4.08 → 3.98 → 2.88. Combined with 55.9% displacement, β=0.60 is the
  worst arm on the geometry and the biggest risk to §4.1 motivation 2 (shared burn-in
  ⇒ linearly mode connected ⇒ canonicalisation unnecessary). Nothing has measured mode
  connectivity; β=0.60 stresses that untested assumption hardest.

**And the framing correction that matters most:** β cannot fix memorisation and does
not need to. The memorisation objection is answered by the *evaluation protocol* —
§6.4's **retrieval baseline** ("the trained model whose mixture is nearest to π"),
which is the direct control and which the field does not report, plus §6.3's held-out
interior points *and* one held-out vertex. Do not spend compute on larger β hoping to
pre-empt a reviewer; spend it on the retrieval arm.

## 3. What MFU is, and why it matters here

**MFU = Model FLOP Utilisation** — the fraction of a GPU's peak arithmetic throughput
your training loop actually converts into useful model FLOPs.

```
MFU = (6 · D · tokens_per_second) / peak_FLOPS
```

`6·D` is the standard dense count of forward+backward FLOPs per token for a
`D`-parameter model (2·D forward, 4·D backward). So MFU answers: *of the arithmetic
this GPU could have done, what share went into the model?* 50% is a well-tuned large
LM; 30% is a normal planning assumption; single digits means something is wrong or the
model shape is hostile.

**Why it matters here:** every compute number in RESEARCH_PLAN §4.6 — the 219 GPU-hr
total, the 112 min/branch at Medium, the `--time` values committed in the sbatch
scripts — was derived by *assuming* 30% and dividing. Nothing had measured it. If the
real figure is 4.6%, every one of those numbers is wrong by ~6.5×, including the
walltimes, which is the difference between a job finishing and being killed at the
wall with no checkpoint.

**Measured (B200, Mini, 2026-09-08):**

| | value |
|---|---|
| **Achievable bf16 dense peak** | **1662.4 TFLOPS** (datasheet says 2250 — using it was itself a 26% error in the denominator) |
| Branch training loop | 249 ktok/s → **MFU 4.63%** |
| Compute only (synthetic tokens, no dataset) | 325.6 ktok/s → 6.05% |
| Data pipeline only (one-hot) | 564 ktok/s |

The serial model `1/232 = 1/325 + 1/811` reproduces the measured loop rate, so the
decomposition is trustworthy: **the ceiling is the GPU path, not the dataloader.**

**Why it's 4.6% — located precisely.** Timing each GEMM the loop actually issues
(T=16384 tokens, d=512, V=50257):

| GEMM | TFLOPS | % of peak |
|---|---|---|
| mlp down `(T,4d)×(4d,d)` | 900.3 | 54% |
| qkv proj `(T,d)×(d,3d)` | 704.3 | 42% |
| mlp up `(T,d)×(d,4d)` | 697.3 | 42% |
| attn out `(T,d)×(d,d)` | 554.0 | 33% |
| **lm head `(T,d)×(d,V)`** | **83.8** | **5.0%** |

The LM head is **50.6% of Mini's per-token FLOPs** and runs at a twentieth of peak:
`k=512` against `n=50257` is a tall-thin × very-wide matmul, bandwidth-bound writing a
logit tensor 98× wider than the hidden state. Half the model executes at 5%.

Ruled out by measurement, so don't re-litigate:
- **not the dataloader** — a perfect one buys +40% (325 vs 249 ktok/s)
- **not launch overhead / occupancy** — 4× larger micro-batch at identical effective
  batch buys 5% ((16,8)→232, (32,4)→240, (64,2)→245 ktok/s)
- **not the fp32 logit upcast** — removing it buys 4%

**The targeted fix** is a fused / chunked cross-entropy that never materialises
full-vocab logits (Liger-kernel style, or chunk the vocab and accumulate). That
attacks exactly the half of the FLOPs running at 5%. Worth doing before Phase 2, where
it is worth hundreds of GPU-hours.

**MFU rises with scale**, because the embedding share of `D` falls 51%→32%→15%: Mini
6.05% → Small 8.60% → Medium 11.90% (compute-only). This is the *same* confound §4.5
already flags for the spectrum, showing up in the compute budget.

---

## 4. Phase 2 recalibrated — and two things that would break it

| scale | measured | §4.6 | miss | branch each | committed `--time` |
|---|---|---|---|---|---|
| Mini | 34.5 GPU-hr | 4.0 | 8.6× | 21 min | 60 min — ok |
| Small | 137 | 23.3 | 5.9× | **80 min** | 60 min → **killed** |
| Medium | 726 | 191 | 3.8× | **426 min** | 60 min → **killed** |
| total | **~898** | 219 | **4.1×** | | |

**Before launching Phase 2:**

1. **Raise `--time`.** `slurm/zoo_{trunk,branch}.sbatch` are sized for Mini. Small
   needs ≥2 h/branch, Medium ≥9 h. `launch_beta_arm.sh` takes `TRUNK_TIME` /
   `BRANCH_TIME` env vars, so no edit is needed — but they must be set.
2. **Add mid-run checkpointing to `train_span`.** Medium's trunk is 16.5 h with none;
   a single preemption or node failure loses it entirely. There is no resume path
   below whole-member granularity.
3. Storage: the project path is a **1.0 TB quota with 397 GB free** — not the "9.2 TB
   ceiling" §4.6 cites, which is the `/scratch` quota. Phase 2's ~213 GB of weights
   fits but consumes over half the headroom, and it is shared with two collaborators.
   Re-check `df -h /work/neu/p2026_0038_neu` before starting.

---

## 5. β = 0.30, DECIDED — and the probe that is next

`report_beta_calibration.py` mechanically applies §4.3's rule ("smallest β at which
mixture identity is measurable") and returns 0.15, which is also the cleanest on
min-SNR and the cheapest (464 vs 898 GPU-hr for Phase 2). **The decision is 0.30
anyway.** The full reasoning, since it deliberately overrides the rule:

| | β=0.15 | **β=0.30** | β=0.60 |
|---|---|---|---|
| gate min SNR | 5.96 | 5.40 | 4.54 |
| between/within (weights) | 4.08 | 3.98 | 2.88 |
| weights moved from trunk | 16.5% | 28.7% | 55.9% |
| Phase 2 total | 464 GPU-hr | 898 | 1767 |
| saving vs independent training | 6.3× | 3.3× | **1.7×** |
| shared burn-in | 85% | 70% | 40% |

- **0.60 is out.** Worst on the gate, worst on weight-space between/within, 2× the
  compute of 0.30, and its 1.7× saving guts §4.1 motivation 1 — you pay near
  independent-training cost while keeping a shared trunk's constraints. 55.9%
  displacement is also the biggest stress on the untested mode-connectivity
  assumption.
- **0.15 is defensible and was not chosen.** The geometry (§2b) clears it of the
  memorisation worry, and it would save 434 GPU-hr. It loses on one thing only: it has
  the least headroom for the singleton regime nothing has tested. Given Phase 2 costs
  898 GPU-hr at 0.30, buying insurance for 434 is a reasonable trade — and 0.30 is
  what §4.2/§4.6 costed, so nothing needs re-planning.
- If the probe below shows 0.30 comfortably separating nearby singletons, **revisiting
  0.15 is a legitimate 434 GPU-hr saving** and worth one extra probe arm (~1 GPU-hr).

### The diversity / singleton probe — READY TO LAUNCH at β=0.30

The gap this closes: N=12 contained **only one-hot anchors**, the maximally separated
mixtures on the simplex. Phase 2 is 20 anchors + **80 singletons at Dirichlet π**,
whose closest pair sits at **L1 = 0.173** (measured on the actual Phase 2 plan; the
`min_l1_gap=0.15` rejection floor is what sets it). Nothing has tested whether a
β=0.30 branch can separate mixtures that close.

**Calibrating the probe correctly matters, and an earlier draft of this section got it
wrong.** It said "lower `min_l1_gap` to make the singletons close". That is wrong:
`min_l1_gap` only *relaxes a rejection test* and cannot cluster draws. The knob is
**`alpha`**, the Dirichlet concentration. Measured, 6 draws:

| alpha | min pair L1 | mean pair L1 | character |
|---|---|---|---|
| 1.0 (Phase 2 default) | 0.595 | 0.971 | uniform on simplex — **4× too easy at n=6** |
| 4.0 | 0.232 | 0.536 | interior |
| **8.0** | **0.169** | 0.380 | tight around barycentre — **matches Phase 2's 0.173** |

Verified on the actual probe plan (`--n_members 26 --alpha 8.0`, which also avoids the
5 anchors and applies the 0.15 rejection floor): singletons land at **indices 20–25**
with **min pair L1 = 0.164**, mean 0.330 — Phase 2's hardest pair is 0.173. Cost
confirmed at **2.1 GPU-hr** for 6 members (2356 branch steps each, identical to
`zoo_b030`, which is what makes the trunk reusable).
| 16.0 | 0.122 | 0.269 | tighter than Phase 2 ever gets |

`--alpha` and `--min_l1_gap` are now CLI flags on `train_zoo.py`, defaulting to the
Phase 2 values (verified: explicit defaults reproduce the Phase 2 plan bit-identically),
and both are sealed into `zoo_meta.json` so a probe zoo can never be mistaken for a
real one.

```bash
ART=/work/neu/p2026_0038_neu/$USER/llm_vae
PROBE=$ART/zoo_b030_singleton/gpt2_zoo_mini

# 1. Reuse the existing beta=0.30 trunk. total_steps/trunk_steps depend only on
#    tokens_per_param, batch, grad_accum, n_ctx and beta -- NOT on n_members -- so
#    the 12-member trunk is valid for a 26-member plan. Saves 59 min of GPU.
mkdir -p $PROBE && cp $ART/zoo_b030/gpt2_zoo_mini/trunk.pt $PROBE/

# 2. Check the plan BEFORE spending anything: singletons land at indices 20-25.
python scripts/train_zoo.py --mode plan --arch gpt2_zoo_mini --beta 0.30 \
    --n_members 26 --alpha 8.0 --zoo_dir $PROBE

# 3. Six branches, one GPU each. ~21 min per member -> ~2.1 GPU-hr.
ZOO_ROOT=$ART/zoo_b030_singleton ARCH=gpt2_zoo_mini BETA=0.30 N_MEMBERS=26 \
  EXTRA="--alpha 8.0" sbatch --array=20-25%6 slurm/zoo_branch.sbatch
```

**How to score it — not with the gate.** `eval_domains.py` groups only
`kind == "anchor"` (`:234`) and its whole verdict block is behind `if by_anchor:`
(`:240`), so on an all-singleton zoo it exits 1 with `separation: None`. That is a
**missing verdict, not a gate failure** — `report_beta_calibration.py` already
distinguishes those states; read its INCOMPLETE branch.

The right statistic is RESEARCH_PLAN §6.5's slope, in miniature: for each domain `d`,
regress *measured per-domain PPL advantage* on *requested weight* `π_d` across the six
members. 6 members × 5 domains = 30 points. Positive slopes ⇒ the requested mixture
controls the model, at Phase 2's hardest separation. It is a **control** test rather
than a ranking test, and §6.5 notes it is immune to the collapse trap of misstep 19.
Expect a signed, noisy estimate — not a p-value.

Decision rule set in advance, so the result cannot be reinterpreted after the fact:

- **Slopes clearly positive** → β=0.30 confirmed; proceed to Phase 2. Optionally spend
  ~1 GPU-hr on the same probe at β=0.15 to try to bank the 434 GPU-hr saving.
- **Slopes flat or mixed** → the 80-singleton design is the problem, not β. Do not
  simply raise β: β=0.60 is worse on every geometry metric. The response is to
  reconsider §6.3's composition (fewer, better-separated π; or more branches per
  mixture) before spending 898 GPU-hr.

### The follow-up test that decides it on evidence (~1.5 GPU-hr)

Build a small **singleton** zoo at β=0.15 and check whether gate-style separation
survives when the mixtures are blends rather than vertices.

`build_zoo_plan` emits the 20 anchors first, so singletons only appear at
`n_members ≥ 21`. Verified: `build_zoo_plan(26, 4, seed=0)` gives 20 anchors +
**6 singletons at indices 20–25**. Their π, and the pairwise L1 gaps, measured:

```
20 [0.299 0.449 0.009 0.001 0.242]      pairwise L1 gap over the 6:
21 [0.137 0.056 0.063 0.236 0.508]        min 0.323   max 1.483
22 [0.491 0.000 0.339 0.011 0.160]      (sample_simplex rejects < 0.15,
23 [0.138 0.512 0.058 0.050 0.243]       so these are comfortably spread —
24 [0.010 0.036 0.277 0.209 0.468]       i.e. NOT yet the hard case)
25 [0.087 0.094 0.389 0.362 0.068]
```

```bash
# The trunk is shared with the existing beta=0.15 arm, so reuse it rather than
# retraining: point ZOO_ROOT at a fresh dir, copy in zoo_b015's trunk.pt, and run
# only the 6 branch tasks. ~1.5 GPU-hr at Mini.
ZOO_ROOT=$ARTIFACT_DIR/zoo_b015_singleton ARCH=gpt2_zoo_mini BETA=0.15 N_MEMBERS=26 \
  sbatch --array=20-25%6 slurm/zoo_branch.sbatch
```

Design notes for whoever runs it:

- **A min gap of 0.323 is the easy case.** The genuinely hard discrimination is
  neighbouring mixtures, so consider lowering `min_l1_gap` for this probe (it is a
  `sample_simplex` kwarg, not currently reachable from the `train_zoo` CLI) or
  hand-picking 6 nearby draws from a larger plan. Report which you did — an easy-case
  pass does not license β=0.15.
- **`eval_domains.py` will report no verdict on an all-singleton zoo.** Confirmed in
  the code: it groups only `kind == "anchor"` (`:234`) and the whole verdict block is
  behind `if by_anchor:` (`:240`), so it exits 1 with `separation: None`. That is a
  missing-verdict, *not* a gate failure — `report_beta_calibration.py` already
  distinguishes those two states, so read its INCOMPLETE branch rather than
  concluding β=0.15 failed.
- **Score it with the slope, not best-in-zoo.** The right statistic is the per-member
  correlation between *requested* weight on domain `d` and *measured* PPL advantage on
  `d` — RESEARCH_PLAN §6.5's slope figure in miniature. It is a control test (does the
  requested mixture drive the output?) rather than a ranking test, and §6.5 notes it is
  immune to the collapse trap of misstep 19. With 6 members × 5 domains there are 30
  points; expect a noisy but signed estimate, not a p-value.

If the slope is positive and significant at β=0.15, use 0.15 and save ~half the
Phase 2 compute. If it is flat, β=0.30 or higher is required and the calibration
question was genuinely unanswered by anchors alone.

---

## 6. What changed in the code this session

All deployed to `/work/neu/p2026_0038_neu/$USER/llm-vae` and present locally.

| file | change |
|---|---|
| `src/llmzoo/data/mixtures.py` | 3 of 5 dataset ids replaced/fixed (§7). Added `HOLDOUT_EVERY=64` + content-hash `is_holdout()`. |
| `scripts/train_zoo.py` | β-consistency refusal before the resume skip; `remove_columns` on the interleave map; atomic `trunk.pt`; MFU + steady-state throughput instrumentation; `--verify_mixture`; W&B; `--peak_tflops` (default **1662.4**, measured). |
| `scripts/eval_domains.py` | content-hash holdout replacing `skip(100_000)`; eval-slice caching; W&B with gate verdict as summary columns. |
| `scripts/train_stack.py` | `--source` / `--zoo_dir` — previously there was **no way** to fit a PCA on a real zoo. |
| `src/llmzoo/artifacts/io.py` | `zoo_dir` in the ensemble fingerprint (conditionally, so the noise hash `cdbb077e838a543f` is byte-preserved); hostname in the atomic-write temp name. |
| `src/llmzoo/artifacts/bundle.py` | `load_run()` passes `source`/`zoo_dir` — without it every downstream consumer refused a zoo run. |
| `slurm/aicr_env.sh` | sources `~/.config/llmzoo/env` for secrets; accepts `WANDB_API_KEY` as well as `~/.netrc`; warns loudly on the offline fallback; prints key *lengths* only. |
| `slurm/stack_smoke.sbatch` | fixed a `pipefail`-inverted assertion that had been reporting FAIL for correct behaviour. |
| `tests/test_zoo.py` | chunk-grid fixture resized to D=450,000 (the old D=92 made the assertion unsatisfiable). |
| `slurm/launch_beta_arm.sh` | **new** — one β arm as a dependency chain, with per-β `ZOO_ROOT`. |
| `slurm/eval_domains.sbatch` | **new**. |
| `scripts/report_beta_calibration.py` | **new** — pure-stdlib β table renderer. |
| `reports/` | **new** — figure builder + self-contained HTML. |

Environment is pinned and locked: torch 2.9.1+cu128 (sm_100), transformers 4.57.6,
datasets 3.6.0, numpy 2.4.6. `slurm/env_lock_aicr.txt` is a committed `pip freeze`.

---

## 7. Corpora — do not "restore" these

RESEARCH_PLAN §6.3 now carries the authoritative table. Summary of why they changed:

| domain | was | now | why |
|---|---|---|---|
| code | `codeparrot/github-code-clean` | `codeparrot/codeparrot-clean` (**Python only**) | old id ships a loading script — removed in `datasets≥4`, needs `trust_remote_code` on 3.x which this repo never passes. StarCoder/The-Stack are all `gated: auto` and **no HF token reaches a job**. |
| books | `name="en", split="train"` | `name=None, split="en"` | name and split were **swapped**; the dataset has one `default` config with language-named splits |
| multilingual | `uonlp/CulturaX` `name="fr"` | `HuggingFaceFW/fineweb-2` `fra_Latn` | CulturaX is `gated: auto` **and** script-based. Note the old config was **already French-only** — the swap preserved scope; the "multilingual" label was always broader than the config. |

**β was calibrated against these exact corpora. Changing one invalidates the
calibration** — the gate's separation numbers are corpus-specific. If a corpus must
change, re-run §4.2.

Paper scope to declare, not concede: four of five domains are English, the fifth is
French, and "code" is Python.

---

## 8. Traps this session hit that are not yet in CLAUDE.md

1. **`set -u` and `${#VAR}`.** Every sbatch runs `set -euo pipefail`; a bare
   `${#HF_TOKEN}` on an unset var is a fatal "unbound variable" and killed all three
   trunks in 1 s. Testing via `srun bash -c` will **not** reproduce it — that shell has
   no `set -u`. Test with `bash -euo pipefail -c`.
2. **rsync excludes match at any depth.** `--exclude='artifacts*/'` silently ate
   `src/llmzoo/artifacts/`. Anchor repo-root excludes with a leading `/`.
3. **`torch.cuda.get_arch_list()` returns `[]` with no GPU** as of torch 2.9 (it opens
   `if not is_available(): return []`). Use `torch._C._cuda_getArchFlags()` for a
   login-node check.
4. **b200-batch had a partition-wide launch fault** on 2026-09-08 — `launch failed
   requeued held` on even a trivial `hostname` job, 61 jobs affected cluster-wide. It
   cleared on its own in ~40 min. Held jobs never recover unaided; `scontrol release`
   them. `scontrol update TimeLimit=` works on *pending* array tasks and is
   permission-denied on running ones.
5. **rtx-batch is not automatically the right place for eval.** §4.6 assigns eval there
   because it is a separate pool, which is correct when saturating b200. On 2026-09-08
   rtx-batch had **275** one-GPU jobs queued vs b200's 16, and the gates sat pending
   8+ min before being moved. Check both queues; note b200's headline pending count is
   dominated by one user's self-throttled array (`JobArrayTaskLimit`), which is not
   competition.
6. **Keep one zoo on one GPU type.** bf16 reduction order differs across
   architectures, and that variation lands in the within-anchor spread that gate
   condition 2 divides by.
7. **Corpora contain exact duplicates.** Books had 47 duplicates in a 130-doc window
   (81 unique of 126). A *positional* train/eval split puts copies on both sides; only
   a content hash is safe.

---

## 9. Immediate next actions, in order

1. **Decide β** — accept 0.30, or run the §5 singleton probe (~1.5 GPU-hr) and let it
   decide.
2. **Fix the cross-entropy** before Phase 2 (§3). This is the single highest-leverage
   change available; it targets 50% of FLOPs running at 5% of peak.
3. **Add checkpointing to `train_span`** (§4.2) — required for Medium.
4. **Then Phase 2**, with `TRUNK_TIME` / `BRANCH_TIME` set per scale.
5. Phase 0.1 (`family_idx` → `basis_key` + `cond`) is still deferred and is a Phase 3
   prerequisite; §2.1 says do it while zoos train.

Not started, unchanged from the plan: Phases 3–5, all stretch items.
