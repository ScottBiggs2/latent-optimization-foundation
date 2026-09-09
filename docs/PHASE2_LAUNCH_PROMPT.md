# Launch prompt — picking up after Phase 1

Copy the block below into a fresh session. It assumes the 2026-09-08 state: the β
calibration is done, the §4.3 gate passed on all three arms, and nothing of Phase 2
has started.

---

```
You are continuing the LLM weight-zoo project on the AICR B200 cluster. Phase 1 (the
β calibration and the §4.3 gate) is COMPLETE and PASSED. Your job is the pre-Phase-2
work, then Phase 2 itself.

READ FIRST, IN THIS ORDER
  1. docs/PHASE1_HANDOFF.md   — what happened, what was measured, what is still open.
     Sections 2, 3 and 8 contain things you will otherwise get wrong.
  2. CLAUDE.md                — the traps, distilled.
  3. RESEARCH_PLAN.md §4       — the plan. Note §4.6's compute numbers are now known
     to be wrong by 4.1×; the handoff has the measured replacements.
  Open reports/beta_calibration.html in a browser for the figures.

STATE
- All three β arms (0.15 / 0.30 / 0.60) passed the gate: separation 3/3, min SNR
  5.96 / 5.40 / 4.54 against a >1.0 threshold.
- **β = 0.30 is DECIDED.** It is NOT what §4.3's rule mechanically selects (that is
  0.15) and the override is deliberate — handoff §5 has the table. β=0.60 was
  considered and rejected on measured evidence: it is worst on the gate, worst on
  weight-space between/within (2.88 vs 4.08), 2× the compute, and its 1.7× saving vs
  independent training guts §4.1 motivation 1.
- Weight-space geometry was measured (handoff §2b, scripts/diag_zoo_geometry.py). Two
  results worth knowing before you re-open the β question: β=0.15 is NOT a
  memorisation hazard (16.5% of the weight norm moved; centroid fraction 0.741 above
  the 0.677 an i.i.d. spread gives at N=12), and β=0.60's higher effective rank is
  within-anchor noise, not richer structure.
- Artifacts: $ARTIFACT_DIR/zoo_b015, zoo_b030, zoo_b060 (12 members each),
  runs/zoo_b0*_k11/ (PCA + codes + VAE), beta_calibration.json.
- W&B: scottbiggs2001-northeastern-university/llm-vae, groups zoo_gpt2_zoo_mini_b*.
- Env is pinned and locked: torch 2.9.1+cu128, transformers 4.57.6, datasets 3.6.0.
  Secrets live in ~/.config/llmzoo/env (mode 600), sourced by slurm/aicr_env.sh.

DO NOT
- Do not change any dataset id in src/llmzoo/data/mixtures.py. β was calibrated
  against those exact corpora; changing one invalidates the calibration. RESEARCH_PLAN
  §6.3 has the authoritative table and says this explicitly.
- Do not quote the N=12 spectrum (ev0/median 134, effrank 0.142) as confirming §4.4.
  It is saturated at the design ceiling — 3 anchor groups span rank 2 at most, so
  effrank ≈ 2/11 = 0.18 is forced by the plan, not measured from weight space. Handoff
  §2 explains this. The §4.4 test needs the N=100 zoo with 85 distinct π.
- Do not re-run the β sweep at other scales. That was considered and dropped.

TASKS, IN ORDER

1. RUN THE DIVERSITY / SINGLETON PROBE at β=0.30  (~2.1 GPU-hr)  <-- START HERE
   This is the one question N=12 could not answer, and it is the last thing between
   here and 898 GPU-hr of Phase 2. N=12 contained ONLY one-hot anchors, the maximally
   separated mixtures. Phase 2 is 80 singletons whose CLOSEST PAIR sits at L1 = 0.173
   (measured on the real Phase 2 plan). Nothing has tested a β=0.30 branch at that
   separation.

   Copy-paste, in order. The trunk is reused, so this costs branches only:

     ART=/work/neu/p2026_0038_neu/$USER/llm_vae
     PROBE=$ART/zoo_b030_singleton/gpt2_zoo_mini
     mkdir -p $PROBE && cp $ART/zoo_b030/gpt2_zoo_mini/trunk.pt $PROBE/
     python scripts/train_zoo.py --mode plan --arch gpt2_zoo_mini --beta 0.30 \
         --n_members 26 --alpha 8.0 --zoo_dir $PROBE
     ZOO_ROOT=$ART/zoo_b030_singleton ARCH=gpt2_zoo_mini BETA=0.30 N_MEMBERS=26 \
       EXTRA="--alpha 8.0" sbatch --array=20-25%6 slurm/zoo_branch.sbatch

   WHY alpha=8.0 AND NOT min_l1_gap: an earlier draft said to lower min_l1_gap to make
   the singletons close. That is wrong — min_l1_gap only relaxes a rejection test and
   cannot cluster draws. alpha is the Dirichlet concentration and is the actual knob.
   Measured at n=6: alpha=1 gives min pair L1 0.595 (4x too easy), alpha=8 gives 0.169,
   which matches Phase 2's hardest pair at 0.173. Both flags are now on train_zoo.py,
   default to the Phase 2 values, and are sealed into zoo_meta.json.

   Reusing the trunk is valid: total_steps/trunk_steps depend only on tokens_per_param,
   batch, grad_accum, n_ctx and beta — NOT on n_members — so the 12-member trunk serves
   a 26-member plan. Saves 59 min of GPU.

   SCORING — do not use the gate. eval_domains.py groups only kind=="anchor" (:234) and
   its verdict block is behind `if by_anchor:` (:240), so an all-singleton zoo exits 1
   with separation: None. That is a MISSING VERDICT, not a gate failure;
   report_beta_calibration.py already distinguishes those states. Score with §6.5's
   slope: per domain d, regress measured per-domain PPL advantage on requested weight
   pi_d across the 6 members (30 points). It is a CONTROL test, not a ranking test, and
   §6.5 notes it is immune to misstep 19's collapse trap.

   DECISION RULE, fixed in advance so the result cannot be reinterpreted afterwards:
     - slopes clearly positive -> β=0.30 confirmed, go to Phase 2. Optionally spend
       ~1 GPU-hr on the same probe at β=0.15 to try to bank a 434 GPU-hr saving.
     - slopes flat or mixed -> the 80-SINGLETON DESIGN is the problem, not β. Do NOT
       just raise β; 0.60 is worse on every geometry metric. Reconsider §6.3's
       composition (fewer, better-separated π, or more branches per mixture) before
       committing 898 GPU-hr.

2. FIX THE CROSS-ENTROPY  (highest leverage change available)
   Measured MFU is 4.63%, not the 30% §4.6 assumed. The cause is located: the LM head
   is 50.6% of Mini's per-token FLOPs and runs at 5.0% of the measured 1662 TFLOPS
   peak, because k=512 against n=50257 is bandwidth-bound writing logits 98× wider
   than the hidden state. Handoff §3 has the full decomposition and rules out the
   dataloader, launch overhead, and the fp32 upcast by measurement — do not re-derive
   these.
   Implement a fused or chunked cross-entropy that never materialises full-vocab
   logits, in scripts/train_zoo.py's train_span. Verify with the existing instruments:
     - throughput/mfu is sealed into member_<i>.json and logged to W&B
     - reuse /scratch/$USER/diag/computeprobe.py and gemmpeak.py for before/after
   This is worth hundreds of GPU-hours at Medium. Confirm loss curves are unchanged
   on a short run before trusting it.

3. ADD MID-RUN CHECKPOINTING to train_span
   Medium's trunk is 16.5 h with none. One preemption loses it; there is no resume
   below whole-member granularity. Required before 2.3.

4. PHASE 2 — the zoos (RESEARCH_PLAN §2 Phase 2)
   Order: Small (2.1, the primary result) → Mini (2.2) → Medium (2.3).
   MEASURED costs, not §4.6's:
     Mini   34.5 GPU-hr,  branch  21 min
     Small  137  GPU-hr,  branch  80 min
     Medium 726  GPU-hr,  branch 426 min
   The committed --time=01:00:00 would kill EVERY Small and Medium branch. Use:
     TRUNK_TIME=06:00:00 BRANCH_TIME=02:30:00 bash slurm/launch_beta_arm.sh 0.30 100 gpt2_zoo_small
     TRUNK_TIME=20:00:00 BRANCH_TIME=09:00:00 bash slurm/launch_beta_arm.sh 0.30 100 gpt2_zoo_medium
   (b200-batch MaxTime is 24 h, so Medium's trunk fits with margin — but see task 3.)
   Re-check `df -h /work/neu/p2026_0038_neu` first: 397 GB free of a 1.0 TB quota at
   last look, shared with two collaborators, and Phase 2 wants ~213 GB.
   launch_beta_arm.sh chains trunk → branch array → {gate, spectrum} with afterok and
   gives each config its own ZOO_ROOT. Resubmitting is safe: a member whose w_<i>.npy
   exists exits immediately.

5. PHASE 0.1 while the zoos train
   Split family_idx into basis_key + cond (RESEARCH_PLAN §2.1). ~150 references across
   13 files. Phase 3 needs it; Phase 2 does not. §2.1 explicitly says do it while zoos
   are training because it is the change most likely to need a cluster round-trip.

CLUSTER NOTES THAT COST TIME LAST SESSION (handoff §8 has all seven)
- Every sbatch runs `set -euo pipefail`. A bare ${#VAR} on an unset var is FATAL.
  Testing via `srun bash -c` will NOT reproduce it — that shell has no set -u. Test
  with `bash -euo pipefail -c`.
- rsync excludes match at ANY depth: anchor repo-root ones with a leading slash, or
  --exclude='artifacts*/' will silently delete src/llmzoo/artifacts/.
- b200-batch had a partition-wide launch fault (`launch failed requeued held`, 61 jobs)
  that cleared itself in ~40 min. Held jobs never recover unaided — scontrol release.
- rtx-batch is NOT automatically right for eval despite §4.6's table. Check both
  queues; b200's headline pending count is dominated by one user's self-throttled
  array, which is not competition.
- Keep one zoo on ONE GPU type. bf16 reduction order differs across architectures and
  lands in the within-anchor spread that gate condition 2 divides by.

REPORT BACK WITH
  1. The singleton-probe slopes per domain, and whether β=0.30 is confirmed.
  2. Before/after MFU and tokens/sec from the cross-entropy fix, and the revised
     Phase 2 cost.
  3. Per scale: the §4.3 gate verdict, and ev0/median + effective_rank_ratio at
     k=99 — this is the first REAL test of §4.4, since 85 distinct π lifts the design
     ceiling that made the N=12 numbers uninformative.
  4. The §4.7 scaling curve: does effective rank grow with D or stay flat? Report the
     block-only spectrum beside the whole-stack one (§4.5's embedding-share confound).
```

---

## Notes for whoever is driving

The three things most likely to go wrong, in order:

1. **Someone "fixes" a dataset id back to match an old draft of §6.3** and silently
   invalidates the β calibration. The plan now warns about this in two places; it is
   still the most likely unforced error.
2. **The cross-entropy rewrite changes the loss.** It must be numerically equivalent,
   not just faster. Compare loss curves on a short run at fixed seed before trusting
   it — `train_span` is deterministic given `(seed, member_idx)`.
3. **Medium's walltime.** 426 min/branch and a 16.5 h trunk, against a committed
   60 min default and no checkpointing. Two independent ways to lose a day.
