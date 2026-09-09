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

**β is DECIDED at 0.30** (2026-09-08, `docs/PHASE1_HANDOFF.md` §5). All of
0.15/0.30/0.60 passed the gate. 0.15 is what §4.3's "smallest passing" rule selects and
is 434 GPU-hr cheaper; the override buys headroom for the singleton regime N=12 could
not test. Do not re-open this without reading handoff §2b first.

**`--alpha`, not `--min_l1_gap`, controls how close singleton mixtures land.**
`min_l1_gap` only relaxes a rejection test and cannot cluster draws. Measured at n=6:
alpha=1 → min pair L1 0.595, alpha=8 → 0.169. Phase 2's 80 singletons have a closest
pair at 0.173, so a probe wanting Phase 2's hardest case needs `--alpha 8.0`.

**Never size a conditioning table with `len(ARCH_CONFIGS)`.** It was `N_FAMILIES`
until 2026-09-08, which meant registering an architecture silently reshaped every
`nn.Embedding` and invalidated every checkpoint. Use `N_COND_SLOTS` (fixed at 32).
`N_ARCHS` is the registry size and is informational.

**Zoo runs use `exclude_1d=False`.** GPT-2 has biases on every projection; with
`True` a generated model would inherit member 0's. Costs ~0.1% of `D`.
