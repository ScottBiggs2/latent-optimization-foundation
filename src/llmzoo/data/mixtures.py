"""
Pretraining data mixtures for the zoo (RESEARCH_PLAN §6.3).

A "class" in this project is a pretraining data mixture `pi` -- a point on the
5-simplex over the domains below. The zoo is nested:

    20 anchors     5 mixtures x 4 branches   the within-mixture noise floor
    80 singletons  distinct Dirichlet draws  simplex coverage
    ------------------------------------------------------------------
    100 models, 85 distinct pi

Anchors sit on the vertices (one-hot) so held-out *vertex* extrapolation and
held-out *interior* interpolation are separable difficulties (§6.3).

--------------------------------------------------------------------------------
THE DATASET IDS BELOW ARE THE MOST LIKELY THING IN THIS REPO TO BE STALE.
--------------------------------------------------------------------------------
HF datasets get renamed, gated, and restructured. Run

    python scripts/train_zoo.py --verify_domains

before burning any GPU time -- it opens each stream and pulls one document, which
takes seconds and fails loudly with the exact id that moved. Swapping an id is a
one-line edit here and changes nothing else in the pipeline.
"""

from __future__ import annotations

import hashlib
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

# Order is load-bearing: it fixes which simplex coordinate means what, and that
# ordering is what the flow's conditioning vector is indexed by. Do not reorder
# without regenerating every zoo.
DOMAINS: Tuple[str, ...] = ("web", "code", "math", "books", "multilingual")

# (hf_path, hf_name/config, split, text_column)
#
# Every entry below was re-verified against datasets-server.huggingface.co on
# 2026-09-07: config exists, split exists, text column exists, repo is UNGATED and
# carries no loading script. Three of the five originals were broken; see the
# reason on each line. The constraint that produced these particular choices is
# that a job gets NO Hugging Face credentials -- slurm/aicr_env.sh sets HF_HOME to
# /scratch/$USER/hf_cache, which redirects the token lookup away from
# ~/.cache/huggingface/token -- so a `gated: auto` repo 401s inside the array.
DOMAIN_SOURCES: Dict[str, dict] = {
    "web": dict(path="HuggingFaceFW/fineweb-edu", name="sample-10BT",
                split="train", text_column="text"),
    # was codeparrot/github-code-clean, which ships a github-code-clean.py loading
    # script: removed outright in datasets>=4, and on 3.x it needs
    # trust_remote_code=True, which this repo never passes. codeparrot-clean is the
    # parquet-native predecessor -- deduplicated, filtered, Python only.
    "code": dict(path="codeparrot/codeparrot-clean", name=None,
                 split="train", text_column="content"),
    "math": dict(path="open-web-math/open-web-math", name=None,
                 split="train", text_column="text"),
    # name and split were swapped. This dataset has ONE config ("default") whose
    # splits are named by language, so the language belongs in `split`.
    "books": dict(path="manu/project_gutenberg", name=None,
                  split="en", text_column="text"),
    # was uonlp/CulturaX, which is `gated: auto` AND script-based -- two
    # independent blockers. fineweb-2 is the direct successor: same corpus type
    # (multilingual Common Crawl), ungated, parquet, per-language configs.
    "multilingual": dict(path="HuggingFaceFW/fineweb-2", name="fra_Latn",
                         split="train", text_column="text"),
}


# TRAIN / EVAL SPLIT. One definition, used by scripts/train_zoo.py (which must
# exclude these documents) and scripts/eval_domains.py (which must use only
# these). It lives here so the two cannot drift apart -- if they ever did, the
# §4.3 gate would be scoring models on their own training data and would pass for
# the wrong reason.
#
# A document is held out when a hash OF ITS TEXT falls in 1/HOLDOUT_EVERY of the
# hash space. Keyed on content, not on position, and that distinction is
# load-bearing -- see the third bullet.
#
# Why not the `skip(100_000)` this replaced. That was wrong in two independent
# ways, both measured 2026-09-07:
#
#   * `manu/project_gutenberg` split "en" has only 61,340 rows, so skip(100_000)
#     exhausted the entire split and raised "no evaluation text for domain
#     books" -- after streaming ~12 GB to discard it.
#   * For the other four, one anchor branch plus its share of the trunk consumes
#     420k-1.18M documents OF ITS OWN DOMAIN (doc sizes span 130x, 1.5 KB to
#     195 KB), so offset 100k sat well inside the training data. That biases the
#     gate in the dangerous direction: the anchor for a domain gets a
#     memorisation advantage on exactly the domain it is supposed to win, which
#     manufactures the separation §4.3 exists to detect.
#
# Why not a positional modulo either, which was the first fix attempted:
#
#   * The corpora contain EXACT DUPLICATE documents. Measured on books: of 126
#     consecutive training documents only 81 were unique, and 2 of 3 held-out
#     books appeared verbatim in the training split. A positional split puts
#     copies of one document on both sides; a content hash cannot, because
#     identical text always hashes to the same side.
#
# Content hashing also removes the ordering constraint, so the predicate is safe
# to apply anywhere in the pipeline. Cost is a sha1 over each document, which is
# microseconds against the tokenizer call on the same text.
HOLDOUT_EVERY = 64


def is_holdout(text: str) -> bool:
    """True if this document belongs to the evaluation split, by content hash."""
    if not text:
        return False
    digest = hashlib.sha1(text.encode("utf-8", "ignore")).digest()
    return int.from_bytes(digest[:8], "big") % HOLDOUT_EVERY == 0


# ---------------------------------------------------------------------------
# Opening a domain stream, with the retry the beta calibration never needed
# ---------------------------------------------------------------------------
#
# WHY THIS EXISTS, AND WHY PHASE 1 COULD NOT HAVE FOUND IT.
#
# A ONE-HOT anchor mixture takes `mixture_stream`'s `len(streams) == 1` fast
# path, so each of the beta calibration's 39 jobs resolved exactly ONE dataset.
# Every SINGLETON resolves FIVE, and a 32-wide branch array therefore issues
# ~160 near-simultaneous dataset-resolution calls from one cluster IP.
#
# Measured 2026-09-08: five sequential anonymous calls from a single `cpu` node
# were already enough to draw
#
#     429 Client Error: Too Many Requests ... We had to rate limit your IP
#     (192.69.103.196). To continue using our service, create a HF account or
#     login to your existing account
#
# Two independent mitigations, both wanted:
#   1. HF_TOKEN now reaches jobs via ~/.config/llmzoo/env (mode 600). An
#      authenticated request gets a far higher limit. Note this changes only the
#      auth header -- the corpora are the same UNGATED ids, so the beta
#      calibration stays valid. It is NOT licence to switch to a gated corpus
#      (RESEARCH_PLAN §6.3).
#   2. This retry. Exponential backoff with FULL JITTER, which also
#      de-synchronises an array whose tasks all started at once -- so the retry
#      *is* the stagger, and no job that would have succeeded pays a sleep.
# base_delay=12 with retries=6 gives a full-jitter budget of
# 12*(2^6-1) = 756 s worst case, ~6 min expected. That is sized to CROSS the
# quota window, and the window is the whole point:
#
#   authenticated 2026-09-08: "you hit the quota of 1000 api requests per
#   5 minutes period"
#
# The first version used base_delay=3.0, i.e. a 93 s worst-case budget -- all six
# attempts could land inside ONE exhausted 5-minute window and the job died
# anyway. That is exactly what happened to 8 of 153 Phase 2 branches. Anonymous
# rate limiting is per-IP and was survivable in seconds; authenticated limiting
# is per-USER with a fixed window, so the backoff has to outlast the window
# rather than merely spread the burst.
RETRYABLE_MARKERS = (
    "429", "too many requests", "rate limit", "ratelimit",
    "502", "503", "504", "timed out", "timeout",
    "connectionerror", "connectionreset", "incompleteread",
)


def _is_retryable(exc: BaseException) -> bool:
    blob = f"{type(exc).__name__} {exc}".lower()
    return any(m in blob for m in RETRYABLE_MARKERS)


def open_domain_stream(
    domain: str,
    *,
    retries: int = 6,
    base_delay: float = 12.0,
    log=print,
) -> Tuple[object, str]:
    """
    `load_dataset(..., streaming=True)` for one domain, retrying HF rate limits.

    Returns `(dataset, text_column)`. Raises the last exception if every attempt
    fails, because a persistent 429 is a real problem to surface rather than
    something to loop on forever.

    `datasets` is imported inside the call on purpose: `import
    llmzoo.data.mixtures` must stay cheap enough for the plan path and the tests,
    which never open a stream.
    """
    import random
    import time

    from datasets import load_dataset

    src = DOMAIN_SOURCES[domain]
    last: Optional[BaseException] = None
    for attempt in range(retries):
        try:
            return (load_dataset(src["path"], src["name"], split=src["split"],
                                 streaming=True),
                    src["text_column"])
        except Exception as e:                                # noqa: BLE001
            last = e
            if attempt == retries - 1 or not _is_retryable(e):
                raise
            # Full jitter: sleep ~U(0, base*2^attempt). Uniform-from-zero rather
            # than a fixed backoff is what actually spreads a synchronised array
            # instead of moving the whole thundering herd to a later instant.
            delay = random.uniform(0.0, base_delay * (2 ** attempt))
            log(f"  {domain}: {type(e).__name__} on attempt "
                f"{attempt + 1}/{retries}, retrying in {delay:.1f}s")
            time.sleep(delay)
    raise last                                                # unreachable


def n_domains() -> int:
    return len(DOMAINS)


def anchor_mixtures() -> List[np.ndarray]:
    """The five one-hot vertices, in DOMAINS order."""
    return [np.eye(len(DOMAINS), dtype=np.float64)[i] for i in range(len(DOMAINS))]


def sample_simplex(
    n: int,
    *,
    seed: int = 0,
    alpha: float = 1.0,
    avoid: Optional[Sequence[np.ndarray]] = None,
    min_l1_gap: float = 0.15,
    max_tries_per_point: int = 1000,
) -> List[np.ndarray]:
    """
    `n` distinct Dirichlet(alpha) draws on the simplex, rejecting any draw within
    `min_l1_gap` (L1) of an anchor or of a previously accepted draw.

    Rejection matters for two different reasons. Near-duplicate singletons waste a
    model on a point we already have, and a singleton sitting on top of an anchor
    silently converts that anchor's noise floor into a 5-branch estimate -- which
    would be fine, except nothing downstream would know.
    """
    rng = np.random.default_rng(seed)
    kept: List[np.ndarray] = []
    blocked = [np.asarray(a, dtype=np.float64) for a in (avoid or [])]
    for _ in range(n):
        for _try in range(max_tries_per_point):
            pi = rng.dirichlet([alpha] * len(DOMAINS))
            if all(np.abs(pi - b).sum() >= min_l1_gap for b in blocked + kept):
                kept.append(pi)
                break
        else:
            raise RuntimeError(
                f"could not place point {len(kept)+1}/{n} at min_l1_gap="
                f"{min_l1_gap} after {max_tries_per_point} tries. Lower the gap or "
                f"ask for fewer points.")
    return kept


def build_zoo_plan(
    n_members: int = 100,
    n_anchor_branches: int = 4,
    *,
    seed: int = 0,
    alpha: float = 1.0,
    min_l1_gap: float = 0.15,
) -> List[dict]:
    """
    The full member plan: `n_members` entries of
    {idx, pi, kind, mixture_id, branch}.

    Anchors come first so that a truncated run (say the N=12 beta calibration)
    still contains whole anchor groups rather than a ragged tail, which is what
    keeps the within-mixture noise floor estimable at small N.
    """
    anchors = anchor_mixtures()
    plan: List[dict] = []
    for a_i, pi in enumerate(anchors):
        for b in range(n_anchor_branches):
            plan.append(dict(idx=len(plan), pi=pi.tolist(), kind="anchor",
                             mixture_id=f"anchor_{DOMAINS[a_i]}", branch=b))
            if len(plan) >= n_members:
                return plan
    n_single = n_members - len(plan)
    for s_i, pi in enumerate(sample_simplex(n_single, seed=seed, alpha=alpha,
                                            avoid=anchors, min_l1_gap=min_l1_gap)):
        plan.append(dict(idx=len(plan), pi=pi.tolist(), kind="singleton",
                         mixture_id=f"dirichlet_{s_i:03d}", branch=0))
    return plan


def holdout_split(plan: List[dict], *, n_interior: int = 4,
                  holdout_vertex: str = "math") -> Dict[str, List[int]]:
    """
    Which members the flow never sees.

    Two different difficulties, reported separately (§6.3):
      * interior  -- singletons nearest the simplex barycentre: interpolation.
      * vertex    -- every branch of one anchor: extrapolation to a corner.

    Holding out a whole anchor group costs one of the five noise-floor estimates.
    That is the intended trade: a vertex nobody trained on is the harder and more
    convincing test, and four anchors still pool 12 dof.
    """
    centre = np.full(len(DOMAINS), 1.0 / len(DOMAINS))
    singles = [m for m in plan if m["kind"] == "singleton"]
    singles.sort(key=lambda m: np.abs(np.asarray(m["pi"]) - centre).sum())
    interior = [m["idx"] for m in singles[:n_interior]]
    vertex = [m["idx"] for m in plan
              if m["mixture_id"] == f"anchor_{holdout_vertex}"]
    held = set(interior) | set(vertex)
    return {
        "interior": interior,
        "vertex": vertex,
        "train": [m["idx"] for m in plan if m["idx"] not in held],
    }
