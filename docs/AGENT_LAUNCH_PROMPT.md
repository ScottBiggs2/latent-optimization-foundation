# Agent launch prompt — β calibration and beyond

Copy the block below to start an agent on the cluster work. It assumes the
2026-09-08 state: Phase 0.2 / 0.3 / 0.6 landed, `train_zoo.py` and
`eval_domains.py` written but never run on a GPU.

---

```
You are running the β-calibration stage of the LLM weight-zoo project on the AICR
B200 cluster. Read RESEARCH_PLAN.md §4 and CLAUDE.md first; docs/RESEARCH_NOTES.md
§5 has the 21 recorded missteps and is worth skimming before you debug anything.

CONTEXT
The pipeline (ensembles → per-family Gram PCA → conditional rectified flow → 8 eval
arms) is built and tested, but has only ever run on a manufactured noise ensemble,
where it produced a clean, pre-registered null. We are now building the first real
data: ~100 GPT-2 models branched off one shared partially-trained trunk, where each
model's "class" is its pretraining data mixture on a 5-simplex.

Your job is Phase 1: get train_zoo.py working on the cluster, run the β calibration,
and run the §4.3 gate. You are NOT doing Phase 2+ unless the gate passes.

ENVIRONMENT
- ssh aicr, --account=p2026_0038_neu. Explorer is retired; do not use it.
- Never run torch locally or on the login node. Everything through sbatch.
- Code lives at /work/neu/p2026_0038_neu/$USER/llm-vae. rsync from the laptop, or
  git pull if you have pushed.
- ALWAYS set --time and --mem. Defaults are 1 h and 1 GB/CPU and both kill jobs
  silently, mid-training, with unhelpful exit codes.
- ONE GPU per job. AICR nodes are shared: a 1-GPU job backfills into a
  partially-free node and starts immediately, where --gres=gpu:8 waits for a whole
  node to drain. Never use --exclusive or --nodelist.
- b200-devel (4 h, 2-GPU cap, usually idle) for smoke tests. b200-batch for
  training. rtx-batch is a SEPARATE 32-GPU pool that does not consume the b200
  ceiling — put evaluation there so it runs alongside training. cpu for analysis.
- Fairshare is charged to a shared account. Kill hung jobs; idle held GPUs cost
  your collaborators.

STEP 0 — deploy and verify, before any GPU time
  a) rsync the repo to AICR, then: pip install -e . inside the env
     (slurm/setup_env_aicr.sh if the env does not exist yet).
  b) TEST=tests/test_zoo.py sbatch slurm/tests.sbatch
     This covers Phase 0.2/0.3/0.6 and the mixture plan. It has never run with
     torch present — expect to fix import-level breakage here, not logic.
     The load-bearing assertion is noise_source_fingerprint_unchanged: source=
     "noise" must still hash to cdbb077e838a543f. If that fails, Phase 0.3
     changed the ensemble underneath every existing artifact and must be fixed
     before anything else.
  c) sbatch slurm/stack_smoke.sbatch on b200-devel — the known-green bisect point
     for the pre-existing pipeline. If it fails, the break is in what I changed,
     not in the science.
  d) python scripts/train_zoo.py --verify_domains        (cpu partition, seconds)
     THIS IS THE MOST LIKELY THING TO FAIL. The five HF dataset ids in
     src/llmzoo/data/mixtures.py get renamed and gated. If a domain fails, swap
     the id in DOMAIN_SOURCES — it is a one-line edit and nothing downstream
     depends on which corpus a domain maps to. Prefer ungated alternatives. Do
     not proceed with a broken domain; a mixture that silently drops one corpus
     invalidates the whole conditioning axis.

STEP 1 — shakedown at N=12, β=0.30, Mini
  python scripts/train_zoo.py --mode plan --arch gpt2_zoo_mini --n_members 12
  ARCH=gpt2_zoo_mini BETA=0.30 N_MEMBERS=12 sbatch slurm/zoo_trunk.sbatch
  ARCH=gpt2_zoo_mini BETA=0.30 N_MEMBERS=12 sbatch --array=0-11 slurm/zoo_branch.sbatch

  Expect ~0.09 GPU-hr trunk and ~2.4 min per branch, but treat those as ±2× until
  you measure. Report the ACTUAL tokens/sec and MFU — every compute estimate in
  RESEARCH_PLAN §4.6 is derived from an assumed 30% MFU and needs recalibrating
  against the real number.

  Re-running is safe: a member whose w_<i>.npy exists exits immediately, so a
  partially-failed array is resubmitted verbatim.

STEP 2 — the β calibration (RESEARCH_PLAN §4.2)
  Repeat step 1 for β ∈ {0.15, 0.30, 0.60}, into separate zoo dirs, N=12, Mini.
  Total ~1.9 GPU-hr. For each β record:
    - per-domain PPL spread            (scripts/eval_domains.py)
    - ev0/median and effective_rank_ratio at k=11  (train_stack.py on the zoo)
    - wall clock and measured MFU

STEP 3 — THE GATE (RESEARCH_PLAN §4.3). Stop here and report.
  python scripts/eval_domains.py --arch gpt2_zoo_mini

  It exits 0 or 1 and prints a verdict. Two conditions:
    1. separation — each anchor's model is best-in-zoo on its own dominant domain
    2. signal>noise — between-anchor spread exceeds within-anchor spread

  IF IT FAILS: the answer is a LARGER β and a re-run. It is not a reframe, and it
  is not something to fix downstream. If β=0.60 still fails, STOP and report —
  that is a real finding about the trunk-branch design and I need to see it before
  you spend more compute.

  IF IT PASSES: fit the PCA and report the spectrum against the pre-registered
  prediction in §4.4 — effective_rank_ratio ≈ 0.2–0.3 and ev0/median ≫ 1. Gate on
  those BULK statistics, never on ev[0]/ev[k−1], which reads 12.09 on data whose
  true ratio is 1.01 (misstep 15b).

  A flat spectrum on a zoo that PASSED the gate is a real result and means we
  rebuild the paper around contribution 3. A flat spectrum on a zoo that FAILED
  the gate means nothing at all. Do not confuse them.

BEYOND (only after I confirm the gate result)
  Phase 2: full N=100 zoos at Mini / Small / Medium, 1-GPU arrays, ~219 GPU-hr
           total. Raise --time in zoo_trunk.sbatch and zoo_branch.sbatch for the
           bigger scales; the committed values are sized for Mini and will kill a
           Medium run mid-training.
  Phase 2.5: RESEARCH_PLAN Phase 0.1 (split family_idx into basis_key + cond)
           while the zoos train. Phase 3 needs it; Phase 2 does not.

RULES THAT WILL SAVE YOU TIME
- Read code_rms_ratio before believing any generative ΔPPL (misstep 19). A
  collapsed generator scores BETTER than an honest one on a mean-centred ensemble.
- Do not scale up the flow (misstep 21). Capacity × duration collapses it while
  training loss improves. Defaults are (64,128,64) at 500 epochs with
  --holdout_frac 0.15; keep all three.
- Do not edit scripts/eval_stack.py or scripts/report_stack.py while a dependency
  chain is queued — queued jobs pick up whatever is deployed at exec time.
- Print ${#HF_TOKEN}, never ${HF_TOKEN:-UNSET}; the latter expands to the value.

REPORT BACK WITH
  1. Which of steps 0a–0d needed fixing, and what you changed.
  2. Measured tokens/sec and MFU at Mini, and what that does to §4.6's estimates.
  3. The β table: per-domain PPL spread, ev0/median, effective_rank_ratio for each
     of the three β values.
  4. The gate verdict, and your recommended β.
  5. Anything you had to change in mixtures.py, with the reason.
```

---

## Notes for whoever is driving

The three things most likely to go wrong, in order:

1. **HF dataset ids** (`src/llmzoo/data/mixtures.py`). `--verify_domains` exists
   entirely for this. `uonlp/CulturaX` and `bigcode/*` are the gated-corpus risks.
2. **`train_zoo.py` has never executed.** It is syntax-clean and the APIs it calls
   were checked against their signatures, but the training loop, the
   `interleave_datasets` schema normalisation, and the autocast path have not run.
   Budget a debug cycle.
3. **`--time` in the zoo sbatch scripts is sized for Mini.** Small needs ~4× and
   Medium ~50×. This is called out in both scripts' headers but is exactly the kind
   of thing that gets missed and costs a 7-hour run.
