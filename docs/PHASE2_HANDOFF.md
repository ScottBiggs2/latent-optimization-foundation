# Phase 2 handoff — state, artifacts, and what is open

2026-09-09. Read `docs/PHASE2_RESULTS.md` for the science; this file is the operational
state. `RESEARCH_PLAN.md` and `CLAUDE.md` still govern.

---

## 1. Decisions

| question | decision | where |
|---|---|---|
| β | **0.15** (Scott, 2026-09-09) | RESULTS §6 |
| Throughput engineering | **dropped.** GPT-2 stays stock. Attention is already `sdpa`; the LM head's FLOP share equals the embedding fraction (50.0 / 31.0 / 14.5%), so a fused cross-entropy buys ~2× at Mini and little at Medium, which is 81% of the bill. | — |
| Mid-run checkpointing | deferred; a hard prerequisite for Medium only | §4 below |
| First Phase 2 scale | Mini, both β arms, **done** | RESULTS |
| Reporting rule | quote absolute `effective_rank`, or state k | RESULTS §2a |

## 2. Artifacts

`ARTIFACT_DIR = /work/neu/p2026_0038_neu/$USER/llm_vae`

| path | contents |
|---|---|
| `zoo_b015_mini_n100/gpt2_zoo_mini/` | **100 members**, trunk, `zoo_meta.json`, `domain_separation.json`, `zoo_geometry.json` |
| `zoo_b030_mini_n100/gpt2_zoo_mini/` | 100 members, same |
| `runs/zoo_b015_mini_k99/` | PCA (197 MB), `codes_k99/{codes,family_idxs}.npy`, `vae_k99/`, `pipeline_summary_k99.json` |
| `runs/zoo_b030_mini_k99/` | same |
| `zoo_b015/`, `zoo_b030/`, `zoo_b060/` | the Phase 1 N=12 calibration arms |
| `zoo_b015_singleton/`, `zoo_b030_singleton/` | the 6-member α=8 probes |
| `zoo_probe/` | stale 3-member leftover; safe to delete |

Committed locally so figures regenerate without the cluster:
`reports/data/n100/`, `reports/data/probe/`, `reports/data/` (Phase 1).

Disk: 346 GB free of 1.0 TB. Small adds ~50 GB, Medium ~142 GB.

W&B: `scottbiggs2001-northeastern-university/llm-vae`, groups
`zoo_gpt2_zoo_mini_b015_n100` and `_b030_n100`. **N is in the group name** — an
(arch, β) pair spans more than one zoo.

## 3. Two things needing a human

1. **Report node `a0016` (rtx-batch) to the AICR admins.** Three
   `CUDA error: uncorrectable ECC error encountered` kills across 2026-09-08/09; Slurm
   never drained it (`state=mix-`, no reason). `EXCLUDE_NODES` in
   `slurm/launch_beta_arm.sh` defaults to excluding it — a stopgap, and it should be
   cleared once the node is fixed.
2. **Rotate the HF token.** `CLAUDE.md` has listed rotation as outstanding since the
   2026-09-03 transcript leak. It now also reaches every compute job via
   `~/.config/llmzoo/env`, which widens the exposure.

## 4. Prerequisites, by what they block

**Blocks Medium (355M) only:**
- Mid-run checkpointing in `train_span`. Medium's trunk is 16.5 h and a branch is
  426 min with no resume below whole-member granularity. Checkpoints must go to
  `/scratch` and be deleted on completion: a Medium checkpoint is 355M × 4 B × 3 ≈
  4.3 GB, and 100 in `/work` want 426 GB against 346 GB free.
- Medium's 9 h `BRANCH_TIME` is intrinsically unbackfillable. The lever that rescued
  Mini (honest `--time`/`--mem`) does not transfer. Measure achievable concurrency at
  Small first.

**Blocks the generative claim:**
- **§6.4's retrieval baseline is not implemented.** Grep confirms it: `eval_stack.py`
  has 8 arms (`pca_only, vae, generate, gauss_codes, flow_codes, flow_latent,
  flow_rt_codes, flow_rt_latent`) and no nearest-mixture arm. RESULTS §2b makes this
  urgent rather than optional: 100 members in a ~4.5-dimensional manifold is precisely
  the regime §11 predicts a flow will memorise, and retrieval is the control that
  distinguishes generalisation from lookup.

**Blocks Phase 3 (π-conditioning):**
- Phase 0.1, `family_idx` → `basis_key` + `cond`. Real scope is **230 references across
  14 files**, not the plan's ~150/13: `gen/flow.py` 53, `tests/test_flow.py` 41,
  `gen/vae.py` 40, `artifacts/bundle.py` 34, `models/registry.py` 18. It touches the
  checkpoint/fingerprint surface, so budget a cluster round-trip.
- Then §6.2's π-vector MLP replacing `nn.Embedding` at `gen/flow.py:273`.

## 5. Traps added to CLAUDE.md this phase

1. A **singleton** mixture opens five HF streams; an **anchor** opens one
   (`mixture_stream`'s `len(streams)==1` fast path). All 39 Phase 1 jobs resolved one
   dataset each; a 32-wide singleton array makes ~160 near-simultaneous calls.
2. Authenticated HF limiting is **per-user with a 5-minute window** ("1000 api requests
   per 5 minutes"). A backoff budget shorter than the window does not help — the first
   attempt at 189 s worst case still lost 8 of 153 branches. Now 756 s.
3. A shard fetch can fail **mid-stream**, after `open_domain_stream` returned.
   `train_span` rebuilds at `seed + restarts` and seals `stream_restarts` into the
   member JSON. Note the asymmetry: a bare `FileNotFoundError` at *open* time means a
   dead dataset id and must **not** be retried.
4. Run `--verify_mixture`, not just `--verify_domains`, before any singleton zoo. The
   interleave path is unreachable from an all-anchor zoo.
5. Quote the **block-only** spectrum beside the whole-stack one, always.
6. `effective_rank_ratio` is **not k-invariant** (RESULTS §2a).
7. `launch_beta_arm.sh` keys `ZOO_ROOT` and `--run_name` on **(β, arch, N)**. Keyed on
   β alone, Phase 2 Mini would have been refused on every branch and all three scales
   would have overwritten `runs/zoo_b030_k99`.
8. A gate **crash** and a gate **verdict** both exit 1. `eval_domains.sbatch`
   distinguishes them by inode change on `domain_separation.json`, not by timestamp
   (same-second writes tie under `-nt`).
9. Anchor repo-root `.gitignore` patterns with a leading slash, not just rsync excludes.

## 6. Sizing, measured

| | branch | trunk | zoo total (β=0.15) | zoo total (β=0.30) |
|---|---|---|---|---|
| Mini 51.5M | 15.1 min singleton / 10.5 anchor | 70.6 min | ~22 GPU-hr | ~38 |
| Small 124.4M | ~80 min | ~3.5 h | **~71 GPU-hr** | ~137 |
| Medium 354.8M | ~426 min | ~16.5 h | **~375 GPU-hr** | ~726 |

MFU is 4.6% at Mini and rises with scale as the embedding share falls. Concurrency
reached 30 of a 32 ceiling once `--time` and `--mem` were honest (28–42 min, 96 GB
against a measured MaxRSS of 17.8 GB); it was 3 with the committed 60 min / 200 GB.

## 7. Reproducing the figures

```bash
python reports/build_phase2_figures.py     # reads reports/data/, pure stdlib
python reports/make_phase2_html.py         # -> reports/phase2_report.html
python scripts/report_singleton_probe.py --arms 0.15 0.30
python scripts/report_beta_calibration.py --betas 0.15 0.30 --k 99 \
    --zoo_name_tmpl 'zoo_{tag}_{slug}_n100' --run_name_tmpl 'zoo_{tag}_{slug}_k{k}' \
    --n_expected 100
```

The palette in `reports/_tokens.css` passed the dataviz validator in both modes
(worst adjacent CVD ΔE 9.2 light / 9.4 dark). Light-mode `--s3` is 2.74:1 against the
surface, which obligates the direct labels and table views the report already carries.
