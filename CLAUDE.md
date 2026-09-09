# CLAUDE.md

Distilled from `RESEARCH_PLAN.md` §13, `docs/RESEARCH_NOTES.md` §5–6, and
`docs/HANDOFF.md` §4. Nothing here is new — it is the set of things that have
already cost this project time.

**Read `RESEARCH_PLAN.md` first. Then `docs/RESEARCH_NOTES.md` §5 (the 21 recorded
missteps) before you write code.**

## Never run torch or transformers locally

The laptop cannot do it, and the login node is not the place either. Everything goes
through `sbatch`.

`tests/test_report.py` is the sole exception: it is pure stdlib by design, and
`test_stdlib_purity` in it enforces that `scripts/report_stack.py` stays that way.
Do not add a numpy/torch/llmzoo import to `report_stack.py` — it would break both
the `cpu`-partition test and rendering locally from copied-down files.

## Compute is AICR

`ssh aicr`, `--account=p2026_0038_neu`. Explorer is retired. See the `aicr-cluster`
skill for partitions and sizing; the short version is that nodes are shared, so ask
for **one** GPU and an honest `--time` and you start immediately.

```bash
bash slurm/setup_env_aicr.sh          # once
sbatch slurm/stack_smoke.sbatch       # ~1 min, the known-green bisect point
sbatch slurm/stack_run.sbatch         # PCA + VAE, both ranks
sbatch slurm/flow_run.sbatch          # flows, 8 arms, report
TEST=tests/test_ensemble.py sbatch slurm/tests.sbatch
```

Artifacts land in `$ARTIFACT_DIR` (`/work/neu/p2026_0038_neu/$USER/llm_vae`, persistent).
Caches and logs go to `/scratch/$USER`, which is **purged at 30 days**. Nothing in `$HOME`.

## The traps, in the order they bite

**Read `code_rms_ratio` before any generative ΔPPL.** (misstep 19) The ensemble is
mean-centred and `mean(w) ≈ w_0`, so a generator collapsed toward its family mean
scores ΔPPL ≈ 0 — *better* than an honest sample — while generating nothing. Measured:
`flow_codes` beat the Gaussian null by 43× while emitting at 0.25× the correct RMS.
`gauss_codes` is the null for this question and **cannot detect it**, because a
collapsed flow beats it by construction.

**`eval_stack.py` could not see a zoo at all until 2026-09-09, and its ΔPPL baseline
was a RANDOM MODEL.** Two separate defects, both found by running it:
`evaluate_arch` called `load_model(arch)`, which `registry.py` refuses for every
`from_scratch: True` arch, and then read `cfg["default_model_id"]`, which the zoo
configs deliberately omit. Fixed with `build_zoo_model(arch, seed=0)` and the `"gpt2"`
tokenizer (the zoo vocab *is* GPT-2 BPE at 50257). Worse, the target write-back was
skipped at `sample_idx == 0` — correct for a noise ensemble, where row 0 *is* w_0 and
the live model already holds it, and silently catastrophic for a zoo, where the live
model is a random init and row 0 is an arbitrary member. Measured before the fix:
`original_ppl` 57419.25, `pca_only` 175.58, **Δ = −99.694%**. After: Δ = −0.000% with
`cos=1.000000`, `relL2=2.2e-07`. **A large negative ΔPPL on a zoo means the baseline
is wrong, not that the arm is good.**

**Under π-conditioning `code_rms_ratio` inverts, and a CORRECT flow prints
`[COLLAPSED]`.** (2026-09-09) The pooled ratio compares sample spread to the spread of
the whole training set. With constant conditioning those are one population and 1.0 is
right. Condition on π and they are not: samples at a fixed π carry the *conditional*
spread while the target carries the *pooled* one. Measured `pooled/within = 3.07` at
Mini, so a perfectly conditional flow reads **0.33 pooled** — the same number misstep
19 recorded for genuine collapse. `train_flow.py` now emits both, with
`code_rms_reference` sealed beside them: `code_rms_ratio` samples at the *training*
π's (so ~1.0 stays correct in either mode) and `code_rms_ratio_at_fixed_pi` compares
one anchor's samples to that anchor's own within-group spread. Only the second can see
memorisation-as-point-mass, which is π-conditioning's own failure mode — π is nearly a
unique key here, 85 distinct mixtures over 100 members with each singleton seen once.

**Measured at Mini, all 8 π cells: conditioning is INERT at k=N−1.** With
`C = (R−at_pi)/(R−1)`, R = 3.07 (π ignored) and 1.0 (fully conditional): C = −0.03…+0.02
at k=99/91, and +0.12…+0.20 at k=50/46. The pooled ratio reads a healthy 0.85–1.10 in
**every** cell, so this is invisible without the conditional statistic. Halving k is
worth ~10× on conditioning strength.

**Verify the π↔code row pairing, or a scrambled join reads as a null result.**
`train_flow.py` seals `pi_code_distance_corr` — the correlation of pairwise L1 in
mixture space against pairwise L2 in code space. Measured +0.63…+0.69; a shuffled
pairing gives ~0. It is a *pairing* check, independent of whether the flow learned
anything, and it fails in one epoch rather than after 500.

**Gate on `ev0/median` and `effective_rank_ratio`, never `ev[0]/ev[k−1]`.**
(misstep 15b) The tail ratio reads 12.09 at k=N−1 and 1.006 at k=N/2 on the *same*
pure-noise data, because at the rank bound `ev[k−1]` is the smallest direction
surviving the rank floor. Both good statistics are in `pipeline_summary_k*.json`.

**Do not scale up the flow.** (misstep 21) Capacity × duration collapses it
monotonically while training loss improves monotonically:

| width | params | epochs | train loss | sample rms |
|---|---|---|---|---|
| 64,128,64 | 51k | 500 | 1.846 | **1.039 (correct)** |
| 256,512,256 | 362k | 8000 | 0.651 | 0.183 |
| 512,1024,512 | 1.23M | 2000 | 0.650 | **0.158 (worst)** |

Defaults are (64,128,64) at 500 epochs with `--holdout_frac 0.15` so early stopping
gates on **val** loss. With no holdout it gates on train loss and therefore selects
the most collapsed checkpoint on the curve. Dispersion is the only logged number that
gets worse as the loss gets better. DeepWeightFlow ran a flat 30k steps on
[512,1024,2048] with no gate at all — do not replicate it.

**Re-calibrate `--noise_scale` whenever the layout changes.** (misstep 18) Nothing
about it transfers. One global `weight_std` mis-scales the perturbation for any
submatrix whose own std differs; adding embeddings to `D` moved gpt2_medium's cost at
`s=1e-2` from +0.097% to +4.363% while pythia_410m barely shifted. Current calibrated
value is **3e-3**, and the table is in `slurm/stack_run.sbatch`'s header.

**`chunk_bounds` must stay a fixed grid.** (misstep 10)

**A benchmark delta smaller than a few questions is not a measurement.** (misstep 20)
At 200 questions the quantum is 0.005.

## API shapes that look symmetric and are not

`train_stack.py --k` and `train_flow.py --k` are **scalar**; `eval_stack.py --k` is
**variadic**. A codes-space flow's width *is* k, so one flow cannot span ranks.

## Operational

- **Do not edit `scripts/eval_stack.py` or `scripts/report_stack.py` while a
  dependency chain is queued.** Queued jobs pick up whatever is deployed at exec time.
- **Do not prune `runs/emb3/flow_k*_collapsed/`.** It is the only evidence for
  misstep 21, which is load-bearing for the DWF argument in `RESEARCH_PLAN.md` §5.
- **Print `${#HF_TOKEN}`, never `${HF_TOKEN:-UNSET}`** — the latter expands to the
  *value*. It leaked into a transcript on 2026-09-03 and **rotation is still
  outstanding**.
- `slurm/flow_run.sbatch` **skips** an already-sealed flow rather than retraining, so
  running it twice with different `SAMPLE_IDX` trains once and evaluates twice.
- Real benchmarks are **refused** in tiny mode by design; `--bench synthetic` covers
  the plumbing.

## Layout

```
src/llmzoo/     the library      (artifacts/ pca/ gen/ eval/ data/ models/)
scripts/        the CLIs         python scripts/<name>.py --args
slurm/          AICR job scripts sbatch slurm/<name>.sbatch
tests/          bare scripts, not pytest — python tests/<name>.py
docs/           archive: RESEARCH_NOTES (missteps §5, environment §6), HANDOFF, DWF Q&A
```

`llmzoo/__init__.py` is empty on purpose: `import llmzoo` must not pull in numpy or
torch. `pip install -e .` is required — the tests and scripts import `llmzoo.*`, not
sibling files. torch is deliberately **not** a declared dependency; it must come from
the cu128 index first (see `slurm/setup_env_aicr.sh`).

## The zoo (added 2026-09-08)

`RESEARCH_PLAN.md` §4. Branch ~100 GPT-2s off one shared trunk; classes are
pretraining data mixtures on the 5-simplex.

```bash
python scripts/train_zoo.py --verify_domains        # DO THIS FIRST, it is free
python scripts/train_zoo.py --mode plan --arch gpt2_zoo_mini --n_members 12
ARCH=gpt2_zoo_mini BETA=0.30 N_MEMBERS=12 sbatch slurm/zoo_trunk.sbatch
ARCH=gpt2_zoo_mini BETA=0.30 N_MEMBERS=12 sbatch --array=0-11 slurm/zoo_branch.sbatch
python scripts/eval_domains.py --arch gpt2_zoo_mini   # the §4.3 gate; exit 1 = stop
TEST=tests/test_zoo.py sbatch slurm/tests.sbatch
```

**One GPU per array task, always.** Branches are independent, AICR nodes are shared,
and a 1-GPU job backfills immediately where `--gres=gpu:8` waits for a node to drain.
A member whose `w_<i>.npy` exists exits immediately, so a failed array is resubmitted
verbatim.

**The five HF dataset ids in `src/llmzoo/data/mixtures.py` are the most fragile thing
in this repo.** Datasets get renamed and gated. `--verify_domains` finds that in
seconds; a trunk job finds it 40 minutes in.

**`eval_domains.py` is a gate, not a report.** It exits 1 when mixture identity is not
measurable in the models themselves. The response is a LARGER `--beta` and a re-run —
not a reframe, and not something to fix downstream. A flat spectrum measured on a zoo
that failed this gate says nothing about weight space (RESEARCH_PLAN §4.3).

**The gate measures DETECTABILITY, not generative difficulty — do not select β on it
alone.** (2026-09-08) Its min-SNR is *maximised* by a tight within-anchor spread, which
is exactly the regime where the ensemble mean is already a good model, `gauss_codes`
is a strong null, and a flow can memorise the codes — misstep 19's mechanism, and
nothing in `domain_separation.json` sees it. Use `scripts/diag_zoo_geometry.py`
alongside: it reports displacement from the trunk, spread/displacement (√2 = members
moved independently), the centroid fraction (√((N−1)/N/2) for an i.i.d. spread) and
between/within distance in weight space. Measured at Mini N=12, β=0.15/0.30/0.60:
between/within **degrades** 4.08 → 3.98 → 2.88 while displacement grows 16.5% → 28.7%
→ 55.9%, so the largest β is worst on the statistic a PCA basis is built from.
And β cannot fix memorisation anyway — §6.4's retrieval baseline is the control that
answers it.

**β is DECIDED at 0.15** (2026-09-09, `docs/PHASE2_RESULTS.md` §6). This supersedes
the 2026-09-08 decision of 0.30. The k=99 spectrum at N=100 did **not** discriminate
(`effective_rank_ratio` 0.0451 vs 0.0452 whole-stack, 0.0727 vs 0.0738 block-only), so
§4.3's smallest-passing rule selects 0.15, and the sole reason for the earlier override
— headroom for the singleton regime N=12 could not test — has now been tested and passed
5/5. Saves ~417 GPU-hr across Small and Medium. The cost: β=0.30's probe slope is 2.1×
stronger, i.e. more conditioning signal. Do not re-open without reading RESULTS §6.

**`effective_rank_ratio` is NOT k-invariant — quote the absolute `effective_rank`, or
state k.** (2026-09-09) It divides by k while the effective rank barely moves, so it
scales as ~1/k. Measured on ONE set of eigenvalues, β=0.15 blocks-only: ratio 0.073 at
k=99, 0.116 at k=50, **0.199 at k=25**, 0.435 at k=10, while the rank goes 7.20 → 5.79 →
4.98 → 4.35. So §4.4's "0.2–0.3" is satisfied or missed purely by choice of k, and the
plan never states one. This is misstep 15b again, in the statistic adopted to *fix*
misstep 15b. `scripts/diag_block_spectrum.py` emits the sweep by default.

**The leading structure is ≈ dim(Δ⁴) = 4.** (2026-09-09) At Mini N=100 the embeddings
measure effective rank 4.03 / 3.98 — the mixture simplex dimension to within 0.05 — and
the blocks 7.20 / 7.31. A class is a point on Δ⁴, so between-mixture structure can span
at most 4 dimensions however many distinct π are drawn. Not a rank-4 claim: 18% of
embedding and 32% of block variance lie beyond c4. Consequence: 100 members in a
~4.5-dimensional manifold is §11's memorisation regime, so **§6.4's retrieval baseline
(not yet built) is the control that decides any generative result.**

**`--alpha`, not `--min_l1_gap`, controls how close singleton mixtures land.**
`min_l1_gap` only relaxes a rejection test and cannot cluster draws. Measured at n=6:
alpha=1 → min pair L1 0.595, alpha=8 → 0.169. Phase 2's 80 singletons have a closest
pair at 0.173, so a probe wanting Phase 2's hardest case needs `--alpha 8.0`.

**Never size a conditioning table with `len(ARCH_CONFIGS)`.** It was `N_FAMILIES`
until 2026-09-08, which meant registering an architecture silently reshaped every
`nn.Embedding` and invalidated every checkpoint. Use `N_COND_SLOTS` (fixed at 32).
`N_ARCHS` is the registry size and is informational.

**A SINGLETON mixture opens five HF streams; an ANCHOR opens one.** (2026-09-08)
`mixture_stream` takes a `len(streams) == 1` fast path for a one-hot mixture, so all
39 Phase 1 jobs resolved exactly one dataset each. Every singleton resolves five, and
a 32-wide array makes ~160 near-simultaneous calls from one cluster IP — which drew
`429 Too Many Requests ... We had to rate limit your IP` on the very first
singleton-bearing job. Two fixes are in place and both matter: `HF_TOKEN` now reaches
jobs via `~/.config/llmzoo/env` (it was already at `~/.cache/huggingface/token`, but
`HF_HOME=/scratch/$USER` redirects the lookup away from it), and every zoo dataset
open goes through `mixtures.open_domain_stream`, which backs off with full jitter.
The token changes only the auth header — **it is not licence to switch to a gated
corpus** (§6.3). `_is_retryable` deliberately does *not* match a dead dataset id or a
schema break: retrying those and reporting a rate limit would hide the most fragile
thing in this repo.

**Run `--verify_mixture` before any singleton zoo, not just `--verify_domains`.**
The interleave path is unreachable from an all-anchor zoo (any N≤20), so it is the
only cheap proof. Seconds on `cpu`; ~4 min because the shuffle buffers fill.

**Quote the BLOCK-ONLY spectrum beside the whole-stack one, always.**
(`scripts/diag_block_spectrum.py`, 2026-09-08) §4.5 flagged the embedding-share
confound as a thing to declare; measured, it is *dominant*. At Mini the embedding
block is **51.0% of `D` but carries 87.5% of the centred variance**, so the
whole-stack `effective_rank_ratio` (0.145) is essentially the embeddings' number
(0.136) while the transformer blocks sit at **0.211**. The whole-stack figure is
below §4.4's 0.2–0.3 prediction and the block figure is inside it. The script needs
no PCA, no GPU and no torch — a Gram is a sum over coordinates, so
`C_whole = C_extra + C_block` exactly.

**`launch_beta_arm.sh` keys `ZOO_ROOT` and `--run_name` on (β, arch, N), not β.**
Both were β-only and neither could fail at N=12. Pointed at the old default, Phase 2
Mini would land in the N=12 calibration's directory and `train_zoo.py:621` would
refuse every branch; and since every N=100 scale has k=99, all three scales wrote
`runs/zoo_b030_k99` and the later fits overwrote the earlier ones. `THROTTLE` now
defaults to 32 — it was 10, which is 10 waves instead of 4 and 71 h instead of 28 at
Medium.

**Anchor repo-root `.gitignore` patterns with a leading slash too**, not just rsync
excludes. Unanchored `artifacts*/` matched at any depth and shadowed
`src/llmzoo/artifacts/`, which is how that package's `__init__.py` sat **untracked**
— the only subpackage missing its marker.

**Attention is `sdpa` at all three scales**, resolved by HF at model init even though
`zoo_config()` never sets `attn_implementation` (measured 2026-09-08, transformers
4.57.6). So eager attention is *not* the explanation for Medium's low MFU. The LM
head's share of `6·D` equals the embedding fraction exactly — 50.0% / 31.0% / 14.5%
— so a cross-entropy fix is worth ~2× at Mini and little at Medium, which is 81% of
Phase 2's bill. Throughput work is deliberately **not** being done: GPT-2 stays
stock.

**Zoo runs use `exclude_1d=False`.** GPT-2 has biases on every projection; with
`True` a generated model would inherit member 0's. Costs ~0.1% of `D`.
