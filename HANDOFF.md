# Handoff — CFM sprint, 2026-08-31 → 2026-09-04

Point a new session at this file, then at `RESEARCH_NOTES.md`.

**One-line state:** the whole-stack pipeline now runs end to end — ensembles → per-family
Gram PCA → `StackVAE` → conditional rectified flows → eight eval arms → markdown report,
all versioned and reloadable. It has been run on a manufactured ensemble and produced a
clean **null result**. It has never been run on real data.

---

## 1. DONE

### The four sprint deliverables

| # | Ask | State |
|---|---|---|
| 1 | ΔPPL + benchmark tables, PCA at k=N−1 and k=N/2 | **done** |
| 2 | Same with a family-conditioned VAE, round-trip *and* generative | **done** |
| 3 | PCA/VAE save+reload seamlessly | **done** — plus flows, fingerprinted |
| 4 | CFM in raw PCA codes *and* in the VAE latent, to see if flows earn their keep | **done — answer is no, see §2** |

### The run that produced the numbers

`RUN=emb3` on `/scratch/biggs.s/llm_vae/runs/emb3/` — N=100, `s=3e-3`, 3 families
(`gpt2_medium`, `smollm2_360m`, `pythia_410m`), k=99 and k=50, evaluated at
`--sample_idx 0` and `--sample_idx 5`, MMLU+HellaSwag+GPQA at 200 questions.

| job | what |
|---|---|
| 9894023 | noise re-calibration → `s` 1e-2 → **3e-3** |
| 9894365 | ensembles + PCA + VAEs, both ranks (19 min) |
| 9894473/4 | 8-arm ΔPPL, samples 0 and 5 — **collapsed flows, preserved as evidence** |
| 9894475/6 | benchmarks, both targets (1.3 h + 1.5 h) |
| 9894536 | `diag_flow_dispersion.py` → misstep 19 |
| 9912696 | `diag_flow_capacity.py` → misstep 21 |
| 9918744/5/6 | smoke + **corrected** flows and re-eval, both targets |

### New files

`flow.py` · `train_flow.py` · `artifact_io.py` · `run_bundle.py` · `report_stack.py` ·
`diag_flow_dispersion.py` · `diag_flow_capacity.py` ·
`tests_step4_run_bundle.py` · `tests_step5_flow.py` · `tests_step6_report_stack.py` ·
`slurm_flow_run.sh` · `slurm_stack_bench.sh` · `slurm_flow_smoke.sh`

### Modified

`eval_stack.py` (8 arms + benchmarks + dispersion + `--out_suffix`) ·
`train_stack.py` · `vae.py` (`save`/`load`) · `models/weight_extractor.py` (whole-stack) ·
`data/ensemble_dataset.py` (layout v2) · `calibrate_noise.py` · `wandb_utils.py` ·
`slurm_stack_run.sh` (new calibration table, `ARCH_LIST`) · `README.md` · `RESEARCH_NOTES.md`

### Tests: 348 checks, all green

35 + 56 + 60 + 91 + 106 across steps 1, 3, 4, 5, 6.
`tests_step6` is pure stdlib and runs on the `short` partition (and locally).

---

## 2. LEARNED

### The headline result

**The flows do not earn their keep on this ensemble — and that is the correct answer.**
At *matched* dispersion (both arms at code rms 1.02–1.05×), `flow_codes` and
`gauss_codes` agree in all six arch × rank cells to within **0.015 percentage points**,
with `relL2` matching to four significant figures. The ensemble is `w_0 + s·σ·ε_i`, so
the per-family code distribution is Gaussian by construction and the spectrum is flat
(`ev0/median = 1.001`). There is nothing for a flow to learn beyond the prior.

This was pre-registered in `slurm_flow_run.sh`'s header before anything ran.
**The machinery is validated; the ensemble is what cannot answer the question.**

### Five recorded missteps (18–21 + 15b) — read these before touching the flows

Four of the five are cases where a number looked like a result and wasn't.

**15b — `ev[0]/ev[k−1]` is not a flatness measure at k=N−1.** Reads 12.09 at k=N−1 and
1.006 at k=N/2 on the *same* pure-noise data, because at the rank bound `ev[k−1]` is the
smallest direction surviving the rank floor. Gate on `ev0/median` and
`effective_rank_ratio` instead. Both are in `pipeline_summary_k*.json`.

**18 — one global `weight_std` mis-scales the noise per submatrix.** Once embeddings
entered `D`, `s=1e-2` cost `gpt2_medium` **+4.363%** PPL (was +0.097% decoder-only, a 45×
change) while `pythia_410m` barely moved. It is *not* the logit-facing fraction — pythia
has the largest at 25.4% and is second most robust. It is that gpt2's global `weight_std`
is 3.9× pythia's, so at equal `s` it eats 3.9× more absolute perturbation. **Re-calibrate
whenever the layout changes.**

**19 — ΔPPL cannot police a generative arm on a mean-centred ensemble.** `mean(w) ≈ w_0`,
so a generator collapsed toward its family mean scores ΔPPL ≈ 0 — *better* than an honest
sample — while generating nothing. Measured: `flow_codes` "beat" the Gaussian null by 43×
while emitting codes at **0.25× the correct RMS**. `gauss_codes` was built as the null for
exactly this question and **cannot detect it**, because a collapsed flow beats it by
construction. Always read `code_rms_ratio` before ΔPPL.

**20 — a benchmark delta smaller than a few questions is not a measurement.** At 200
questions the quantum is 0.005; every measured delta was 0.000 to −0.020, i.e. 0–4
questions flipping. Also `acc_norm` ≡ `acc` for MMLU/GPQA (single-letter choices, so
length normalization is a no-op).

**21 — the collapse is OVER-fitting, and `train_flow.py` was selecting for it.**
Controlled sweep at fixed k, codes, seed and space:

| width | params | epochs | train loss | sample rms |
|---|---|---|---|---|
| 64,128,64 | 51k | 500 | 1.846 | **1.039 correct** |
| 256,512,256 | 362k | 8000 | 0.651 | 0.183 |
| 512,1024,512 | 1.23M | 2000 | 0.650 | **0.158 worst** |

Training loss falls monotonically while the samples die. `--holdout_frac` defaulted to
0.0, so early stopping gated on **train** loss — monotonically the most collapsed point on
the curve. Fixed: holdout defaults to 0.15 (val gating), `--dispersion_n` logs the spread
ratio every print epoch, the final ratio is sealed into `flow_meta.json`, and
`slurm_flow_run.sh` defaults to (64,128,64) at 500 epochs.

### Two corrections to earlier claims in this repo

* Misstep 19 originally blamed irreducible finite-sample regression and recommended "more
  rows or a narrower code space". **Retracted** — capacity is the cause and it is free.
  Nothing argues against `pythia_160m` revisions, and nothing argues for preferring
  k=N/2 over k=N−1.
* The README previously concluded pythia's fragility "is about the function, not the
  weight scale". **Inverted** by misstep 18.

### Verified against analytic prediction

* `pca_only` at k=N−1: cosine 1.000000, ΔPPL ±0.000%, all three families.
* Truncation residual at k=N/2: predicted 0.002118, measured 0.002336 / 0.002284 /
  0.002251 (6–10% high — the notes' old "within 5%" was the decoder-only layout).
* Truncation-as-denoiser: member 5's noise cost gpt2 +0.519%, `pca_only` at k=N/2
  recovered −0.320% against ≈−0.26% predicted.
* Euler exact to 1e-14 in float64 at every step count; float32 error 1730× below an
  off-by-one; round-trip residual first-order convergent (4.02× and 5.11× for 4× and 5×
  the steps).

---

## 3. TO DO

### Blocking / do first

1. **Rotate `HF_TOKEN`.** It was leaked into an assistant transcript on 2026-09-03 via
   `${HF_TOKEN:-UNSET}` (which expands to the *value* when set). Nothing depends on the
   old value now. `slurm_stack_bench.sh` was fixed to print only the length.
2. **Commit.** ~30 files are modified/untracked; nothing from this sprint is in git.
   Remote repo is behind local.

### The actual next experiment

3. **Experiment 1 — Pythia checkpoint revisions**, `pythia_160m` first (§4 of
   RESEARCH_NOTES). This is the only thing that can answer the research question.
   Code change needed: `EnsembleDataset` gains `source="revisions"` reading one
   `w_i.npy` per member, plus `revision=` threaded through `models/registry.py`.
   Keep `source="noise"`. **`chunk_bounds` must stay the fixed grid** (misstep 10).
   `ensemble_fingerprint` already includes `source`, so every existing artifact
   invalidates automatically the moment this lands.
   Carry in: `--require_spectrum_ratio 10.0`, `--holdout_frac > 0`, and read
   `code_rms_ratio` next to every generative ΔPPL.

### Known open defects

4. **The VAE decoder contracts.** `generate` (z ~ N(0,I) → `decode_cfg`, no flow at all)
   emits codes at 0.59–0.64× the correct RMS at every rank, in every run, *including*
   alongside a healthy flow. Prior-posterior mismatch. Untouched by the flow fix, and the
   reason `flow_latent` is contracted where `flow_codes` is not. Fix this before trusting
   latent-space sampling — and note it bears on open question 4: a decoder that doesn't
   cover its own aggregate posterior isn't conditioning the codes, it's shrinking them.

### Cheap and deferred

5. **Phase 6 — six families.** One variable, no code change:
   `ARCH_LIST="gpt2_medium smollm2_360m qwen3_0_6b smollm2_135m pythia_160m pythia_410m" RUN=emb6 sbatch slurm_stack_run.sh`
   then `slurm_flow_run.sh`, then `slurm_stack_bench.sh`. `N_FAMILIES` is already 6.
6. **Per-parameter-group noise scaling** (`s·std(tensor)` not `s·std(stack)`) — open
   question 6. Redefines the ensemble and invalidates every fingerprint, so only on a run
   that starts with it.
7. Latent-dim sweep (open question 5) — deferred by Scott, "probably fine".

---

## 4. Gotchas that will bite a fresh session

* `train_stack.py --k` and `train_flow.py --k` are **scalar**; `eval_stack.py --k` is
  **variadic**. A codes-space flow's width *is* k, so one flow cannot span ranks.
* `runs/perfam/` is the old decoder-only run and is **refused** by the layout gate. That
  is intended — it's the "before embeddings" comparison point.
* `runs/emb3/` holds **both** the collapsed flows (`flow_k*_collapsed/`,
  `stack_eval_results*_collapsedflow.json`) and the corrected ones. Do not prune the
  collapsed set; it is the only evidence of misstep 21.
* `slurm_flow_run.sh` **skips** an already-sealed flow rather than retraining, so running
  it twice with different `SAMPLE_IDX` trains once and evaluates twice.
* Real benchmarks are **refused** in tiny mode by design; `--bench synthetic` covers the
  plumbing in smoke tests.
* Use `report_stack.py` for the stack pipeline, never `report.py` — the schemas differ in
  shape and `parse_key` raises on a legacy flat key on purpose.
* Never run torch locally. `tests_step6_report_stack.py` is the one exception: pure stdlib
  by design, and tested for that property.
* Do not modify `eval_stack.py` or `report_stack.py` while a dependency chain is queued —
  queued jobs pick up whatever is deployed at exec time.
