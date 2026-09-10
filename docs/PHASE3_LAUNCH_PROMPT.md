# Launch prompt — two parallel tracks after Phase 2 Mini

Copy the block below into a fresh session. It assumes the 2026-09-09 state: β decided at
0.15, two complete N=100 Mini zoos, the §4.3 gate passed on both, no flow yet trained on
real zoo data.

The two tracks are independent. **Track A is GPU-light and on the critical path for the
paper's central claim. Track B is GPU-heavy and unattended.** Launch B first so it queues,
then work A while it runs.

---
Claude's self-prompt:
```
You are continuing the LLM weight-zoo project on the AICR B200 cluster. Phase 2 Mini is
COMPLETE at N=100 for both β=0.15 and β=0.30. β IS DECIDED AT 0.15. Your job is two
tracks that run in parallel.

READ FIRST, IN THIS ORDER
  1. docs/PHASE2_RESULTS.md   — what was measured. §2a and §7 change what you may claim.
  2. docs/PHASE2_HANDOFF.md   — state, artifacts, prerequisites, the 9 traps.
  3. CLAUDE.md                — the traps, distilled.
  4. RESEARCH_PLAN.md §6      — contribution 2, which Track A tests.
  Open reports/phase2_report.html in a browser for the figures.

STATE
- β = 0.15, decided 2026-09-09. The k=99 spectrum did NOT discriminate between 0.15 and
  0.30 (effective_rank_ratio 0.0451 vs 0.0452 whole-stack, 0.0727 vs 0.0738 block-only),
  so §4.3's smallest-passing rule selects 0.15. It saves ~417 GPU-hr on Small + Medium.
  The one thing given up: β=0.30's probe slope is 2.1× stronger (+0.693 vs +0.328), i.e.
  more conditioning signal. That trade was made deliberately.
- Both gates PASS: separation 5/5, min SNR 18.37 (β=0.15) / 10.86 (β=0.30).
- PCA, codes and a VAE already exist for both arms at k=99:
  $ARTIFACT_DIR/runs/zoo_b015_mini_k99/ has codes_k99/{codes,family_idxs}.npy, which is
  ALL train_flow.py reads. Track A needs no new plumbing to start.
- Env pinned: torch 2.9.1+cu128, transformers 4.57.6, datasets 3.6.0. Secrets in
  ~/.config/llmzoo/env (mode 600), sourced by slurm/aicr_env.sh. HF_TOKEN now reaches
  jobs — needed because a singleton mixture opens five HF streams where an anchor opens
  one.

DO NOT
- Do not change any dataset id in src/llmzoo/data/mixtures.py. β was calibrated against
  those exact corpora. RESEARCH_PLAN §6.3 says this in two places and it is still the
  most likely unforced error.
- Do not quote effective_rank_ratio without stating k. It divides by k while the
  effective rank is nearly k-invariant, so it scales as ~1/k: measured 0.045 at k=99 and
  0.343 at k=10 on the SAME eigenvalues. Prefer the absolute effective_rank.
  scripts/diag_block_spectrum.py emits the sweep by default.
- Do not quote a whole-stack spectrum alone. Embeddings are 51.0% of D and carry 86.6%
  of the variance, so a whole-stack number is substantially an embedding number.
- Do not re-open β. Read PHASE2_RESULTS §6 first if tempted.
- Do not scale up the flow (misstep 21). Defaults are (64,128,64) at 500 epochs with
  --holdout_frac 0.15. Capacity × duration collapses it while the loss improves.

===========================================================================
TRACK A — validate the generative machinery on a REAL zoo  (GPU-light)
===========================================================================
No flow has ever been trained on anything but the manufactured noise ensemble. This is
the first real test, and PHASE2_RESULTS §2b makes it sharper than it looks: 100 members
in a ~4.5-dimensional manifold is exactly the regime RESEARCH_PLAN §11 predicts a flow
will MEMORISE. So the order below is deliberate — the controls come before the headline.

A1. READ code_rms_ratio BEFORE ANY ΔPPL. (misstep 19) The ensemble is mean-centred and
    mean(w) ≈ w_0, so a generator collapsed toward its family mean scores ΔPPL ≈ 0 —
    BETTER than an honest sample — while generating nothing. Measured previously:
    flow_codes beat the Gaussian null by 43× while emitting at 0.25× the correct RMS.
    gauss_codes CANNOT detect this, because a collapsed flow beats it by construction.

A2. Train the flow on the β=0.15 codes.
      sbatch slurm/flow_run.sbatch      # check its RUN/K env vars point at
                                        # zoo_b015_mini_k99 and k=99 first
    Note flow_run.sbatch SKIPS an already-sealed flow rather than retraining, so running
    it twice with different SAMPLE_IDX trains once and evaluates twice.
    A codes-space flow's width IS k, so one flow cannot span ranks. train_stack.py --k
    and train_flow.py --k are SCALAR; eval_stack.py --k is VARIADIC.

A3. Run the eight arms and the report. The arms are pca_only, vae, generate,
    gauss_codes, flow_codes, flow_latent, flow_rt_codes, flow_rt_latent.
    gauss_codes is the null and decides whether any flow number means anything.

A4. BUILD §6.4's RETRIEVAL BASELINE. It does not exist — eval_stack.py has no
    nearest-mixture arm; grep confirms. This is the highest-value item in either track.
    For a held-out mixture π, the arm is "the trained model whose mixture is nearest to
    π" (L1 on the simplex). If the generated model is no better than that, the flow
    retrieved rather than generalised. It is gauss_codes one level up, it is the
    objection a reviewer raises unprompted, nobody in this literature reports it, and at
    effective rank 4.5 with 100 members it is now the load-bearing control rather than a
    nicety. The holdout is already sealed in zoo_meta.json: 4 interior singletons
    [70, 95, 63, 68] and the whole anchor_math vertex [8, 9, 10, 11].

A5. Report §6.5's slope on the GENERATED models, not just the trained ones. The
    machinery exists: scripts/report_singleton_probe.py computes exactly this statistic
    with an exact 720-permutation null, and it is the headline evaluation because it is a
    CONTROL test — a collapsed generator gives flat lines however flattering its ΔPPL.

A6. Note what Track A CANNOT yet test: conditioning on π itself. That needs Phase 0.1
    (family_idx → basis_key + cond, 230 references across 14 files) and then §6.2's
    π-vector MLP at gen/flow.py:273. Until then the conditioning axis is a discrete
    embedding table and "we generated a held-out mixture" is not yet a zero-shot claim.
    Do Phase 0.1 while Track B's zoos train — §2.1 puts it there deliberately because it
    is the change most likely to need a cluster round-trip.

===========================================================================
TRACK B — the larger zoos  (GPU-heavy, unattended)
===========================================================================
B1. SMALL 124M, N=100, β=0.15. ~71 GPU-hr. This is §2.1's PRIMARY RESULT and the scale
    the paper leads with. Nothing to reuse — it needs its own trunk (~3.5 h).

      df -h /work/neu/p2026_0038_neu          # 346 GB free; Small adds ~50 GB
      python scripts/train_zoo.py --verify_domains && \
        python scripts/train_zoo.py --verify_mixture     # free, on cpu; do BOTH
      TRUNK_TIME=06:00:00 BRANCH_TIME=02:30:00 SPEC_TIME=02:00:00 THROTTLE=32 \
        bash slurm/launch_beta_arm.sh 0.15 100 gpt2_zoo_small

    launch_beta_arm.sh chains trunk → branch array → {gate, spectrum} with afterok,
    keys ZOO_ROOT and --run_name on (β, arch, N), forwards EXTRA, and excludes the
    faulty node via EXCLUDE_NODES. Resubmitting is safe: a member whose w_<i>.npy
    exists exits immediately, and ARRAY= lets you retry a subset.

    BRANCH_TIME MATTERS. Small's branch is ~80 min; the committed default is 60 and
    would kill every one. Also: an honest --time is the single biggest lever on
    concurrency — Mini went from 3 to 30 concurrent when --time dropped 60→28 min and
    --mem 200G→96G. Ask for the real number plus ~30%, not an hour.

B2. MEASURE THE ACHIEVED CONCURRENCY AT SMALL and report it. Do not extrapolate Mini's.
    Medium needs a 9 h BRANCH_TIME, which is intrinsically far less backfillable, so
    Small is the only intermediate data point before committing ~375 GPU-hr.

B3. THEN Medium 355M, N=100, β=0.15. ~375 GPU-hr. TWO HARD PREREQUISITES:
      (a) mid-run checkpointing in train_span. The trunk is 16.5 h with no resume below
          whole-member granularity. Checkpoints go to /scratch and are DELETED on
          member completion — a Medium checkpoint is ~4.3 GB and 100 in /work want
          426 GB against 346 GB free.
      (b) ~142 GB of disk. Re-check df first.
      TRUNK_TIME=20:00:00 BRANCH_TIME=09:00:00 (b200-batch MaxTime is 24 h).

B4. Per scale, report: the §4.3 gate verdict; the absolute effective_rank at k=99 AND
    the k-sweep; and the block-only spectrum beside the whole-stack one. That last is
    §4.7's whole point — the embedding share falls 51.0% → 31.6% → 14.8% along the
    ladder, so a whole-stack-only curve would be plotting the composition of D changing
    rather than its size.

    scripts/diag_block_spectrum.py does this with no PCA, no GPU and no torch.
    scripts/diag_zoo_geometry.py is O(N) streaming and safe at N=100.

CLUSTER NOTES THAT COST TIME (handoff §5 has all nine)
- Every sbatch runs `set -euo pipefail`. A bare ${#VAR} on an unset var is FATAL, and
  `srun bash -c` will NOT reproduce it — that shell has no set -u. Test with
  `bash -euo pipefail -c`.
- rsync excludes match at ANY depth. Anchor repo-root ones with a leading slash, or
  --exclude='artifacts*/' silently deletes src/llmzoo/artifacts/. The same is true of
  .gitignore patterns.
- Node a0016 on rtx-batch throws uncorrectable ECC and Slurm has not drained it.
  EXCLUDE_NODES defaults to excluding it. Clear that once it is fixed.
- Keep one zoo on ONE GPU type. bf16 reduction order differs across architectures and
  lands in the within-anchor spread that gate condition 2 divides by.
- A gate exiting 1 may be a verdict OR a crash. eval_domains.sbatch now tells you which
  (`report_written=0` means crash). A crash is NOT an argument for a larger β.

REPORT BACK WITH
  1. Track A: code_rms_ratio for every generative arm, BEFORE any ΔPPL; then ΔPPL
     against gauss_codes; then the retrieval baseline; then §6.5's slope on generated
     models. State plainly whether the flow beat retrieval, because that is the result.
  2. Track B: per scale, the gate verdict, absolute effective_rank at k=99 with the
     k-sweep, and block-only beside whole-stack.
  3. The §4.7 scaling curve: does effective rank grow with D or stay flat? Report both
     curves. Note the prediction to beat is now concrete — at Mini the leading structure
     is ≈ dim(Δ⁴) = 4 for embeddings and ≈ 7.2 for blocks. If those hold at Small and
     Medium, effective rank is set by the conditioning variable rather than by D, which
     is a sharper claim than §4.4 made.
  4. Achieved concurrency at Small, and whether Medium is affordable in wall clock.
```

Scott's Generative Verifier Prompt:
```
You are continuing the LLM weight-zoo project on the AICR B200 cluster. Phase 2 Mini is
COMPLETE at N=100 for both β=0.15 and β=0.30. β is likely to be fixed at 0.15 for future scaled runs. Your job is to focus on using both β=0.30 and β=0.15 wtih k=N-1 and k=N/2 to verify that the generative machinery is working. 

READ FIRST, IN THIS ORDER
  1. docs/PHASE2_RESULTS.md   — what was measured. §2a and §7 change what you may claim.
  2. docs/PHASE2_HANDOFF.md   — state, artifacts, prerequisites, the 9 traps.
  3. CLAUDE.md                — the traps, distilled.
  4. RESEARCH_PLAN.md §6      — contribution 2, which Track A tests.
  Open reports/phase2_report.html in a browser for the figures.

STATE
- β = 0.15, decided 2026-09-09. The k=99 spectrum did NOT discriminate between 0.15 and
  0.30 (effective_rank_ratio 0.0451 vs 0.0452 whole-stack, 0.0727 vs 0.0738 block-only),
  so §4.3's smallest-passing rule selects 0.15. It saves ~417 GPU-hr on Small + Medium.
  The one thing given up: β=0.30's probe slope is 2.1× stronger (+0.693 vs +0.328), i.e.
  more conditioning signal. That trade was made deliberately.
- Both gates PASS: separation 5/5, min SNR 18.37 (β=0.15) / 10.86 (β=0.30).
- PCA, codes and a VAE already exist for both arms at k=99:
  $ARTIFACT_DIR/runs/zoo_b015_mini_k99/ has codes_k99/{codes,family_idxs}.npy, which is
  ALL train_flow.py reads. Track A needs no new plumbing to start.
- Env pinned: torch 2.9.1+cu128, transformers 4.57.6, datasets 3.6.0. Secrets in
  ~/.config/llmzoo/env (mode 600), sourced by slurm/aicr_env.sh. HF_TOKEN now reaches
  jobs — needed because a singleton mixture opens five HF streams where an anchor opens
  one.

DO NOT
- Do not change any dataset id in src/llmzoo/data/mixtures.py. β was calibrated against
  those exact corpora. RESEARCH_PLAN §6.3 says this in two places and it is still the
  most likely unforced error.
- Do not quote effective_rank_ratio without stating k. It divides by k while the
  effective rank is nearly k-invariant, so it scales as ~1/k: measured 0.045 at k=99 and
  0.343 at k=10 on the SAME eigenvalues. Prefer the absolute effective_rank.
  scripts/diag_block_spectrum.py emits the sweep by default.
- Do not quote a whole-stack spectrum alone. Embeddings are 51.0% of D and carry 86.6%
  of the variance, so a whole-stack number is substantially an embedding number.
- Do not re-open β. Read PHASE2_RESULTS §6 first if tempted.
- Do not scale up the flow (misstep 21). Defaults are (64,128,64) at 500 epochs with
  --holdout_frac 0.15. Capacity × duration collapses it while the loss improves.

===========================================================================
TRACK A — validate the generative machinery on a REAL zoo  (GPU-light)
===========================================================================
No flow has ever been trained on anything but the manufactured noise ensemble. This is
the first real test, and PHASE2_RESULTS §2b makes it sharper than it looks: 100 members
in a ~4.5-dimensional manifold is exactly the regime RESEARCH_PLAN §11 predicts a flow
will MEMORISE. So the order below is deliberate — the controls come before the headline. You may need to develop the π mixture embedding/conditioning machinery, if it does not cleanly fall into the existing family embedding systems. Cite any techniques you draw from directly in the report (include working links). 

A1. READ code_rms_ratio BEFORE ANY ΔPPL. (misstep 19) The ensemble is mean-centred and
    mean(w) ≈ w_0, so a generator collapsed toward its family mean scores ΔPPL ≈ 0 —
    BETTER than an honest sample — while generating nothing. Measured previously:
    flow_codes beat the Gaussian null by 43× while emitting at 0.25× the correct RMS.
    gauss_codes CANNOT detect this, because a collapsed flow beats it by construction.

A2. Train the flow on the β=0.15 codes.
      sbatch slurm/flow_run.sbatch      # check its RUN/K env vars point at
                                        # zoo_b015_mini_k99 and k=99 first
    Note flow_run.sbatch SKIPS an already-sealed flow rather than retraining, so running
    it twice with different SAMPLE_IDX trains once and evaluates twice.
    A codes-space flow's width IS k, so one flow cannot span ranks. train_stack.py --k
    and train_flow.py --k are SCALAR; eval_stack.py --k is VARIADIC.

A3. Run the eight arms and the report. The arms are pca_only, vae, generate,
    gauss_codes, flow_codes, flow_latent, flow_rt_codes, flow_rt_latent.
    gauss_codes is the null and decides whether any flow number means anything.

A4. BUILD §6.4's RETRIEVAL BASELINE. It does not exist — eval_stack.py has no
    nearest-mixture arm; grep confirms. This is the highest-value item in either track.
    For a held-out mixture π, the arm is "the trained model whose mixture is nearest to
    π" (L1 on the simplex). If the generated model is no better than that, the flow
    retrieved rather than generalised. It is gauss_codes one level up, it is the
    objection a reviewer raises unprompted, nobody in this literature reports it, and at
    effective rank 4.5 with 100 members it is now the load-bearing control rather than a
    nicety. The holdout is already sealed in zoo_meta.json: 4 interior singletons
    [70, 95, 63, 68] and the whole anchor_math vertex [8, 9, 10, 11].

A5. Report §6.5's slope on the GENERATED models, not just the trained ones. The
    machinery exists: scripts/report_singleton_probe.py computes exactly this statistic
    with an exact 720-permutation null, and it is the headline evaluation because it is a
    CONTROL test — a collapsed generator gives flat lines however flattering its ΔPPL.

A6. Note what Track A CANNOT yet test: conditioning on π itself. That needs Phase 0.1
    (family_idx → basis_key + cond, 230 references across 14 files) and then §6.2's
    π-vector MLP at gen/flow.py:273. Until then the conditioning axis is a discrete
    embedding table and "we generated a held-out mixture" is not yet a zero-shot claim.
    Do Phase 0.1 while Track B's zoos train — §2.1 puts it there deliberately because it
    is the change most likely to need a cluster round-trip.

  CLUSTER NOTES THAT COST TIME (handoff §5 has all nine)
- Every sbatch runs `set -euo pipefail`. A bare ${#VAR} on an unset var is FATAL, and
  `srun bash -c` will NOT reproduce it — that shell has no set -u. Test with
  `bash -euo pipefail -c`.
- rsync excludes match at ANY depth. Anchor repo-root ones with a leading slash, or
  --exclude='artifacts*/' silently deletes src/llmzoo/artifacts/. The same is true of
  .gitignore patterns.
- Node a0016 on rtx-batch throws uncorrectable ECC and Slurm has not drained it.
  EXCLUDE_NODES defaults to excluding it. Clear that once it is fixed.
- Keep one zoo on ONE GPU type. bf16 reduction order differs across architectures and
  lands in the within-anchor spread that gate condition 2 divides by.
- A gate exiting 1 may be a verdict OR a crash. eval_domains.sbatch now tells you which
  (`report_written=0` means crash). A crash is NOT an argument for a larger β.

REPORT BACK WITH
  1. Track A: code_rms_ratio for every generative arm, BEFORE any ΔPPL; then ΔPPL
     against gauss_codes; then the retrieval baseline; then §6.5's slope on generated
     models. State plainly whether the flow beat retrieval, because that is the result.
  2. Track B: per scale, the gate verdict, absolute effective_rank at k=99 with the
     k-sweep, and block-only beside whole-stack.
  3. The §4.7 scaling curve: does effective rank grow with D or stay flat? Report both
     curves. Note the prediction to beat is now concrete — at Mini the leading structure
     is ≈ dim(Δ⁴) = 4 for embeddings and ≈ 7.2 for blocks. If those hold at Small and
     Medium, effective rank is set by the conditioning variable rather than by D, which
     is a sharper claim than §4.4 made.
  4. Achieved concurrency at Small, and whether Medium is affordable in wall clock.
  5. An analysis of the methods used to decide on β, ready to drop into Overleaf as an appendix to the final publication. 
```

Scotts Zoo Launchers
```
You are continuing the LLM weight-zoo project on the AICR B200 cluster. Phase 2 Mini is
COMPLETE at N=100 for both β=0.15 and β=0.30. β is likely to be fixed at 0.15 for future scaled runs. Your job is to focus on using both β=0.15 to launch the GPU heavy medium and large scaled tiers of the zoo. It's worth going in to tweak the WandB logging of these to make sure they are clearly visible. 

READ FIRST, IN THIS ORDER
  1. docs/PHASE2_RESULTS.md   — what was measured. §2a and §7 change what you may claim.
  2. docs/PHASE2_HANDOFF.md   — state, artifacts, prerequisites, the 9 traps.
  3. CLAUDE.md                — the traps, distilled.
  4. RESEARCH_PLAN.md §6      — contribution 2, which Track A tests.
  Open reports/phase2_report.html in a browser for the figures.

STATE
- β = 0.15, decided 2026-09-09. The k=99 spectrum did NOT discriminate between 0.15 and
  0.30 (effective_rank_ratio 0.0451 vs 0.0452 whole-stack, 0.0727 vs 0.0738 block-only),
  so §4.3's smallest-passing rule selects 0.15. It saves ~417 GPU-hr on Small + Medium.
  The one thing given up: β=0.30's probe slope is 2.1× stronger (+0.693 vs +0.328), i.e.
  more conditioning signal. That trade was made deliberately.
- Both gates PASS: separation 5/5, min SNR 18.37 (β=0.15) / 10.86 (β=0.30).
- PCA, codes and a VAE already exist for both arms at k=99:
  $ARTIFACT_DIR/runs/zoo_b015_mini_k99/ has codes_k99/{codes,family_idxs}.npy, which is
  ALL train_flow.py reads. Track A needs no new plumbing to start.
- Env pinned: torch 2.9.1+cu128, transformers 4.57.6, datasets 3.6.0. Secrets in
  ~/.config/llmzoo/env (mode 600), sourced by slurm/aicr_env.sh. HF_TOKEN now reaches
  jobs — needed because a singleton mixture opens five HF streams where an anchor opens
  one.

DO NOT
- Do not change any dataset id in src/llmzoo/data/mixtures.py. β was calibrated against
  those exact corpora. RESEARCH_PLAN §6.3 says this in two places and it is still the
  most likely unforced error.
- Do not quote effective_rank_ratio without stating k. It divides by k while the
  effective rank is nearly k-invariant, so it scales as ~1/k: measured 0.045 at k=99 and
  0.343 at k=10 on the SAME eigenvalues. Prefer the absolute effective_rank.
  scripts/diag_block_spectrum.py emits the sweep by default.
- Do not quote a whole-stack spectrum alone. Embeddings are 51.0% of D and carry 86.6%
  of the variance, so a whole-stack number is substantially an embedding number.
- Do not re-open β. Read PHASE2_RESULTS §6 first if tempted.
- Do not scale up the flow (misstep 21). Defaults are (64,128,64) at 500 epochs with
  --holdout_frac 0.15. Capacity × duration collapses it while the loss improves.


===========================================================================
TRACK B — the larger zoos  (GPU-heavy, unattended)
===========================================================================
B0. TINY ~50M, N=100, β=0.15 and β=0.30 are DONE. Just go snoop to verify that they're all there and are clearly labeled/readable. Nothing to train here, you're just snooping. 
B1. SMALL ~125M, N=100, β=0.15. ~71 GPU-hr. This is §2.1's PRIMARY RESULT and the scale
    the paper leads with. Nothing to reuse — it needs its own trunk (~3.5 h).

      df -h /work/neu/p2026_0038_neu          # 346 GB free; Small adds ~50 GB
      python scripts/train_zoo.py --verify_domains && \
        python scripts/train_zoo.py --verify_mixture     # free, on cpu; do BOTH
      TRUNK_TIME=06:00:00 BRANCH_TIME=02:30:00 SPEC_TIME=02:00:00 THROTTLE=32 \
        bash slurm/launch_beta_arm.sh 0.15 100 gpt2_zoo_small

    launch_beta_arm.sh chains trunk → branch array → {gate, spectrum} with afterok,
    keys ZOO_ROOT and --run_name on (β, arch, N), forwards EXTRA, and excludes the
    faulty node via EXCLUDE_NODES. Resubmitting is safe: a member whose w_<i>.npy
    exists exits immediately, and ARRAY= lets you retry a subset.

    BRANCH_TIME MATTERS. Small's branch is ~80 min; the committed default is 60 and
    would kill every one. Also: an honest --time is the single biggest lever on
    concurrency — Mini went from 3 to 30 concurrent when --time dropped 60→28 min and
    --mem 200G→96G. Ask for the real number plus ~30%, not an hour.

B2. MEASURE THE ACHIEVED CONCURRENCY AT SMALL and report it. Do not extrapolate Mini's.
    Medium needs a 9 h BRANCH_TIME, which is intrinsically far less backfillable, so
    Small is the only intermediate data point before committing ~375 GPU-hr.

B3. THEN Medium ~250M, N=100, β=0.15. ~375 GPU-hr. TWO HARD PREREQUISITES:
      (a) mid-run checkpointing in train_span. The trunk is 16.5 h with no resume below
          whole-member granularity. Checkpoints go to /scratch and are DELETED on
          member completion — a Medium checkpoint is ~4.3 GB and 100 in /work want
          426 GB against 346 GB free.
      (b) ~142 GB of disk. Re-check df first.
      TRUNK_TIME=20:00:00 BRANCH_TIME=09:00:00 (b200-batch MaxTime is 24 h).

B4. Per scale, report: the §4.3 gate verdict; the absolute effective_rank at k=99 AND
    the k-sweep; and the block-only spectrum beside the whole-stack one. That last is
    §4.7's whole point — the embedding share falls 51.0% → 31.6% → 14.8% along the
    ladder, so a whole-stack-only curve would be plotting the composition of D changing
    rather than its size.

    scripts/diag_block_spectrum.py does this with no PCA, no GPU and no torch.
    scripts/diag_zoo_geometry.py is O(N) streaming and safe at N=100.

CLUSTER NOTES THAT COST TIME (handoff §5 has all nine)
- Every sbatch runs `set -euo pipefail`. A bare ${#VAR} on an unset var is FATAL, and
  `srun bash -c` will NOT reproduce it — that shell has no set -u. Test with
  `bash -euo pipefail -c`.
- rsync excludes match at ANY depth. Anchor repo-root ones with a leading slash, or
  --exclude='artifacts*/' silently deletes src/llmzoo/artifacts/. The same is true of
  .gitignore patterns.
- Node a0016 on rtx-batch throws uncorrectable ECC and Slurm has not drained it.
  EXCLUDE_NODES defaults to excluding it. Clear that once it is fixed.
- Keep one zoo on ONE GPU type. bf16 reduction order differs across architectures and
  lands in the within-anchor spread that gate condition 2 divides by.
- A gate exiting 1 may be a verdict OR a crash. eval_domains.sbatch now tells you which
  (`report_written=0` means crash). A crash is NOT an argument for a larger β.

REPORT BACK WITH
  1. Track A: code_rms_ratio for every generative arm, BEFORE any ΔPPL; then ΔPPL
     against gauss_codes; then the retrieval baseline; then §6.5's slope on generated
     models. State plainly whether the flow beat retrieval, because that is the result.
  2. Track B: per scale, the gate verdict, absolute effective_rank at k=99 with the
     k-sweep, and block-only beside whole-stack.
  3. The §4.7 scaling curve: does effective rank grow with D or stay flat? Report both
     curves. Note the prediction to beat is now concrete — at Mini the leading structure
     is ≈ dim(Δ⁴) = 4 for embeddings and ≈ 7.2 for blocks. If those hold at Small and
     Medium, effective rank is set by the conditioning variable rather than by D, which
     is a sharper claim than §4.4 made.
  4. Achieved concurrency at Small, and whether Medium is affordable in wall clock (old names, but same idea).
```
---

## Notes for whoever is driving

The three things most likely to go wrong, in order:

1. **The flow memorises and it reads as success.** Effective rank 4.5 with 100 training
   points is the §11 regime. `code_rms_ratio` catches collapse; only the retrieval
   baseline catches memorisation, and it is the one arm not yet built. Do A4 before
   believing A2/A3.
2. **Someone quotes a whole-stack `effective_rank_ratio`** without k and without the
   block-only curve. Both defects are now documented, and both would survive review
   only by luck.
3. **Small's walltime.** ~80 min/branch against a committed 60-minute default. The same
   class of error as Phase 1's, and the fix is one env var.
