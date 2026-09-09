#!/usr/bin/env python
"""
train_zoo.py -- build a zoo of GPT-2 models by branching off a shared trunk.

RESEARCH_PLAN §4.1. One initialization, one partially-trained trunk, then N branches
that diverge on different data mixtures for the final fraction `beta` of training.

    trunk   : seed -> GPT-2 -> train (1-beta)*T tokens on the uniform mixture
              -> save weights AND optimizer state
    branch  : load trunk -> train beta*T tokens on mixture pi_i
              -> flatten to w_<i>.npy

Why the optimizer state is saved and reloaded: it is the entire reason §4.1 chose to
train our own trunk instead of branching a public checkpoint. Restarting Adam's
moments injects a transient at the branch point that is uniform across branches --
probably harmless, but an uncontrolled confound in a study whose dependent variable
is how far branches diverge.

Why the LR schedule is global: the branch resumes the cosine at step
(1-beta)*total_steps rather than restarting it. A fresh warmup at the branch point
would be a second, larger transient than the one above.

Usage
-----
    python scripts/train_zoo.py --verify_domains          # do this FIRST, it is free
    python scripts/train_zoo.py --mode plan  --arch gpt2_zoo_mini --n_members 12
    python scripts/train_zoo.py --mode trunk --arch gpt2_zoo_mini --beta 0.30
    python scripts/train_zoo.py --mode branch --arch gpt2_zoo_mini --beta 0.30 \
                                --member_idx $SLURM_ARRAY_TASK_ID

Submit branches as a 1-GPU array, never one multi-GPU job -- AICR nodes are shared
and 1-GPU jobs backfill immediately (RESEARCH_PLAN §4.6).
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from typing import List

import numpy as np
import torch

from llmzoo.artifacts.io import atomic_write_json
from llmzoo.data.mixtures import (
    DOMAIN_SOURCES, DOMAINS, build_zoo_plan, holdout_split, is_holdout,
    n_domains, open_domain_stream, _is_retryable,
)
from llmzoo.data.ensemble import ENSEMBLE_LAYOUT_VERSION
from llmzoo.models.registry import (
    build_zoo_model, get_layers, zoo_config, zoo_param_count,
)
from llmzoo.models.weight_extractor import extract_block_flat, extract_extra_flat
import llmzoo.wandb_utils as wb

TOKENIZER_ID = "gpt2"          # the published GPT-2 BPE, 50257 types


def ts() -> str:
    return time.strftime("%H:%M:%S")


def log(msg: str) -> None:
    print(f"[{ts()}] {msg}", flush=True)


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

def verify_domains(n_docs: int = 1) -> int:
    """
    Open every domain stream and pull `n_docs`. Seconds, no GPU, and it is the
    single cheapest way to find out that a dataset id moved or went gated before a
    zoo job discovers it 40 minutes in.
    """
    bad = []
    for dom in DOMAINS:
        src = DOMAIN_SOURCES[dom]
        try:
            # open_domain_stream, not a bare load_dataset: an HF 429 would
            # otherwise be reported as a dead dataset id, which is the wrong
            # diagnosis and invites the one edit RESEARCH_PLAN §6.3 forbids.
            ds, col = open_domain_stream(dom, log=log)
            it = iter(ds)
            for _ in range(n_docs):
                row = next(it)
            if col not in row:
                raise KeyError(
                    f"text_column={col!r} not in {sorted(row)[:8]}")
            log(f"  OK       {dom:<14} {src['path']}"
                f"{'/' + src['name'] if src['name'] else ''}")
        except Exception as e:                                # noqa: BLE001
            bad.append((dom, src, repr(e)[:300]))
            log(f"  FAILED   {dom:<14} {src['path']} -> {repr(e)[:200]}")
    if bad:
        log("")
        log(f"{len(bad)}/{len(DOMAINS)} domains unusable. Fix DOMAIN_SOURCES in "
            f"src/llmzoo/data/mixtures.py -- it is a one-line edit per domain and "
            f"nothing downstream depends on which corpus a domain maps to.")
        return 1
    log(f"all {len(DOMAINS)} domains stream cleanly")
    return 0


def verify_mixture(n_blocks: int = 2, n_ctx: int = 128) -> int:
    """
    Build one genuinely MIXED stream -- all five domains at equal weight -- and
    pull a couple of packed blocks.

    This exists because the beta calibration cannot reach that code path. At
    N <= 20 `build_zoo_plan` returns nothing but one-hot anchors, and a one-hot
    mixture takes the `len(streams) == 1` fast path in `mixture_stream`. So
    `interleave_datasets` and the schema normalisation feeding it are first
    exercised by the first *singleton* member of a full zoo -- i.e. 40 minutes
    into Phase 2, after the trunk has already been paid for.

    Seconds on the `cpu` partition, and it is the only cheap proof.
    """
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(TOKENIZER_ID)
    pi = np.full(len(DOMAINS), 1.0 / len(DOMAINS))
    try:
        gen = mixture_stream(pi, tok, n_ctx, seed=0)
        for i in range(n_blocks):
            blk = next(gen)
            if tuple(blk.shape) != (n_ctx + 1,):
                raise ValueError(f"block {i} has shape {tuple(blk.shape)}, "
                                 f"expected {(n_ctx + 1,)}")
    except Exception as e:                                    # noqa: BLE001
        log(f"  FAILED   {len(DOMAINS)}-way interleave -> {repr(e)[:400]}")
        log("")
        log("interleave_datasets could not merge the five domain streams. The "
            "usual cause is the schema normalisation in mixture_stream: the "
            "`.map` must pass remove_columns, or every stream keeps its own "
            "columns and the features cannot be aligned.")
        return 1
    log(f"  OK       {len(DOMAINS)}-way interleave, {n_blocks} blocks of "
        f"{n_ctx + 1} tokens")
    return 0


def mixture_stream(pi: np.ndarray, tokenizer, n_ctx: int, seed: int):
    """
    Yield packed (n_ctx+1,) int64 token blocks drawn from the domains with
    probabilities `pi`.

    Domains with zero weight are dropped rather than passed to interleave_datasets
    with p=0 -- some versions still open the stream, which for a gated corpus means
    an auth error on a mixture that does not use it.
    """
    from datasets import interleave_datasets

    keep = [i for i, w in enumerate(pi) if w > 1e-6]
    if not keep:
        raise ValueError("mixture is all zeros")
    probs = np.asarray([pi[i] for i in keep], dtype=np.float64)
    probs = probs / probs.sum()

    streams, cols = [], []
    for i in keep:
        # Retrying opener: a singleton mixture resolves FIVE datasets where an
        # anchor resolves one, and a 32-wide array makes ~160 calls at once.
        d, col = open_domain_stream(DOMAINS[i])
        # Drop the evaluation split. eval_domains.py keeps exactly the
        # complement, so the §4.3 gate never scores a model on text it trained
        # on. The predicate is a content hash, so duplicate documents -- which
        # these corpora do contain -- cannot land on both sides.
        d = d.filter(lambda r, _c=col: not is_holdout(r[_c]))
        streams.append(d.shuffle(seed=seed, buffer_size=10_000))
        cols.append(col)

    if len(streams) == 1:
        merged, col_of = streams[0], (lambda _row: cols[0])
    else:
        # Normalise the text column so interleave_datasets sees one schema.
        #
        # remove_columns is the load-bearing half, and it was missing. Without it
        # `.map` ADDS __text and leaves every original column in place, so the
        # streams still present five different schemas (fineweb-edu carries
        # text/id/dump/url/score/..., codeparrot-clean carries
        # content/repo_name/path/license/...) and interleave_datasets raises on
        # feature alignment.
        #
        # Note which runs can catch this: a one-hot anchor mixture takes the
        # single-stream path above, and build_zoo_plan emits nothing but anchors
        # up to N=20. So the beta calibration at N=12 provably cannot reach this
        # line, and without it the first failure would be the first *singleton*
        # member of a full zoo -- after the trunk is already paid for. Use
        # `--verify_mixture` to exercise it for free on the cpu partition.
        renamed = []
        for d, c in zip(streams, cols):
            drop = [x for x in (d.column_names or []) if x != "__text"]
            renamed.append(d.map(lambda r, _c=c: {"__text": r[_c]},
                                 remove_columns=drop))
        merged = interleave_datasets(
            renamed, probabilities=list(probs), seed=seed,
            stopping_strategy="all_exhausted")
        col_of = (lambda _row: "__text")

    buf: List[int] = []
    need = n_ctx + 1
    eos = tokenizer.eos_token_id
    for row in merged:
        text = row[col_of(row)]
        if not text:
            continue
        buf.extend(tokenizer(text, add_special_tokens=False)["input_ids"])
        buf.append(eos)
        while len(buf) >= need:
            yield torch.tensor(buf[:need], dtype=torch.long)
            buf = buf[need:]


def batches(gen, batch_size: int, device):
    chunk = []
    for blk in gen:
        chunk.append(blk)
        if len(chunk) == batch_size:
            x = torch.stack(chunk).to(device, non_blocking=True)
            chunk = []
            yield x[:, :-1].contiguous(), x[:, 1:].contiguous()


# ---------------------------------------------------------------------------
# Schedule
# ---------------------------------------------------------------------------

def lr_at(step: int, total: int, peak: float, warmup: int, floor_frac: float = 0.1):
    """Linear warmup then cosine to floor_frac*peak, over the GLOBAL step count."""
    if step < warmup:
        return peak * (step + 1) / max(warmup, 1)
    t = (step - warmup) / max(total - warmup, 1)
    t = min(max(t, 0.0), 1.0)
    return peak * (floor_frac + (1 - floor_frac) * 0.5 * (1 + math.cos(math.pi * t)))


def token_budget(arch: str, tokens_per_param: int) -> int:
    return int(tokens_per_param * zoo_param_count(arch)["total"])


def _mfu(tokens_per_sec: float, n_params, peak_tflops):
    """
    Model FLOP utilisation, 6*D FLOPs per token (fwd+bwd, dense).

    `peak_tflops` must be the bf16 DENSE peak of the device actually running the
    job. The --peak_tflops default is a B200; a job placed on rtx-batch needs it
    overridden or the reported percentage is meaningless (the ktok/s is not).
    """
    if not n_params or not peak_tflops:
        return None
    return (6.0 * n_params * tokens_per_sec) / (peak_tflops * 1e12)


def _steady(windows: List[float], seen: float, elapsed: float,
            tail: int = 5) -> float:
    """
    Median of the last `tail` per-interval rates, falling back to the average.

    Median rather than mean so one slow interval -- a shuffle-buffer refill, or a
    noisy neighbour on a shared node -- does not drag the sealed number down.
    """
    if windows:
        recent = sorted(windows[-tail:])
        return recent[len(recent) // 2]
    return max(seen, 0.0) / max(elapsed, 1e-9)


# ---------------------------------------------------------------------------
# Train
# ---------------------------------------------------------------------------

def train_span(model, opt, pi, *, tokenizer, device, n_ctx, batch_size,
               grad_accum, start_step, n_steps, total_steps, peak_lr, warmup,
               seed, log_every=50, amp_dtype=torch.bfloat16,
               n_params=None, peak_tflops=None, max_restarts=8):
    """
    Train `n_steps` optimizer steps on mixture `pi`, resuming the global LR
    schedule at `start_step`.

    Returns {"tokens", "seconds", "tokens_per_sec", "mfu", "final_loss"}. MFU is
    reported because every compute estimate in RESEARCH_PLAN §4.6 derives from an
    ASSUMED 30%, and nothing had ever measured it. The denominator is a CLI flag
    rather than a constant so the number stays auditable: 6*D per token is the
    standard fwd+bwd dense-FLOP count, and `peak_tflops` must be the bf16 DENSE
    peak of the device actually in use (B200 SXM: 2250).

    Caveat worth reading before believing a low number: the data path tokenizes
    one document at a time, synchronously, in this thread. There is no prefetch
    and no worker pool, so a low MFU here likely measures the tokenizer, not the
    GPU.
    """
    gen = mixture_stream(np.asarray(pi, dtype=np.float64), tokenizer, n_ctx, seed)
    bit = batches(gen, batch_size, device)
    model.train()
    t0, seen = time.time(), 0
    tot = float("nan")
    # STEADY-STATE CLOCK, separate from t0, and the distinction is not pedantic.
    #
    # HF's shuffle buffer must accumulate `buffer_size` documents before it
    # yields the first one, so the first next(bit) blocks while 10k docs per
    # stream are downloaded -- measured at 4m14s for the 5-stream uniform
    # mixture. Dividing `seen` by (now - t0) folds that stall into the rate and
    # reports an MFU near zero for a GPU that is running fine.
    #
    # So: t0..t_ready is startup, and throughput is measured from t_ready on.
    # Both get reported -- the startup cost is a real per-job overhead worth
    # knowing (it is paid 39 times across a 3-beta calibration), it is just not
    # a throughput.
    t_ready, seen_at_ready = None, 0
    # Stream rebuilds, whether from exhaustion or from a mid-stream shard error.
    # Also the data seed offset, so a rebuild does not re-read the same prefix.
    restarts = 0
    # Even after t_ready the run-to-date average keeps climbing for a while
    # (CUDA/cuBLAS warmup, and the shuffle buffers are still topping up over the
    # network while the first steps run). Measured on an RTX trunk: the average
    # read 123 ktok/s while every consecutive interval was ~150. So the SEALED
    # number is the median of the recent per-interval rates, not the average --
    # otherwise the headline MFU is quietly ~20% pessimistic.
    windows: List[float] = []
    last_t, last_seen = None, None
    for step in range(start_step, start_step + n_steps):
        lr = lr_at(step, total_steps, peak_lr, warmup)
        for g in opt.param_groups:
            g["lr"] = lr
        opt.zero_grad(set_to_none=True)
        tot = 0.0
        for _ in range(grad_accum):
            try:
                x, y = next(bit)
            except StopIteration:                     # corpus exhausted: restart
                restarts += 1
                gen = mixture_stream(np.asarray(pi, dtype=np.float64), tokenizer,
                                     n_ctx, seed + restarts)
                bit = batches(gen, batch_size, device)
                x, y = next(bit)
            except Exception as e:                    # noqa: BLE001
                # A SHARD FETCH CAN FAIL MID-STREAM, long after
                # open_domain_stream returned successfully. Seen in Phase 2:
                #
                #   FileNotFoundError: gzip://file-000000000010.json::hf://
                #   datasets/codeparrot/codeparrot-clean@35a59fb/...json.gz
                #
                # open_domain_stream's retry cannot help -- it wraps the OPEN,
                # and this is raised from inside the datasets iterator hundreds
                # of steps in. Note also that a bare FileNotFoundError is
                # deliberately NOT retryable at open time (it means a dead
                # dataset id), so it is classified here by the same predicate
                # but acted on differently: rebuild the stream and carry on,
                # because the alternative is losing a member 20 minutes in.
                if not _is_retryable(e) and not isinstance(e, OSError):
                    raise
                restarts += 1
                if restarts > max_restarts:
                    log(f"  stream failed {restarts} times, giving up: "
                        f"{type(e).__name__}: {e}")
                    raise
                log(f"  stream died at step {step} "
                    f"({type(e).__name__}: {str(e)[:120]}); rebuilding, "
                    f"restart {restarts}/{max_restarts}")
                gen = mixture_stream(np.asarray(pi, dtype=np.float64), tokenizer,
                                     n_ctx, seed + restarts)
                bit = batches(gen, batch_size, device)
                x, y = next(bit)
            if t_ready is None:
                # First batch in hand: the shuffle buffers are full. Everything
                # before this was startup, not throughput.
                t_ready = time.time()
                log(f"  stream ready after {t_ready - t0:.1f}s "
                    f"(shuffle buffers filled)")
            with torch.autocast("cuda", dtype=amp_dtype,
                                enabled=(device.type == "cuda")):
                loss = model(input_ids=x, labels=None).logits
                loss = torch.nn.functional.cross_entropy(
                    loss.view(-1, loss.size(-1)).float(), y.reshape(-1))
            (loss / grad_accum).backward()
            tot += loss.item() / grad_accum
            seen += x.numel()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if seen_at_ready == 0:
            seen_at_ready = seen          # tokens consumed by the first step
        if (step - start_step) % log_every == 0 or step == start_step + n_steps - 1:
            now = time.time()
            if last_t is not None and seen > last_seen:
                windows.append((seen - last_seen) / max(now - last_t, 1e-9))
            last_t, last_seen = now, seen
            tps = _steady(windows, seen - seen_at_ready,
                          now - (t_ready or t0))
            mfu = _mfu(tps, n_params, peak_tflops)
            log(f"  step {step:>6}/{total_steps}  loss {tot:7.4f}  lr {lr:.2e}  "
                f"tok {seen/1e6:8.2f}M  {tps/1e3:7.1f} ktok/s"
                + (f"  mfu {100*mfu:5.1f}%" if mfu is not None else ""))
            # No-ops when no run is active, so train_span behaves identically
            # whether it was reached from the CLI or from a wandb-enabled job.
            # `step` is the GLOBAL step, so a branch's curve continues the
            # trunk's x-axis instead of restarting at zero.
            wb.log({"train/loss": tot, "train/lr": lr,
                    "train/tokens": seen,
                    "train/tokens_per_sec": tps,
                    "train/ktok_per_sec": tps / 1e3,
                    **({"train/mfu": mfu} if mfu is not None else {}),
                    "train/step": step}, step=step)
    now = time.time()
    el = now - (t_ready or t0)
    avg = max(seen - seen_at_ready, 0) / max(el, 1e-9)
    tps = _steady(windows, seen - seen_at_ready, el)
    return {"tokens": int(seen),
            "seconds": round(now - t0, 3),              # wall, including startup
            "steady_seconds": round(el, 3),             # excluding startup
            "startup_seconds": round((t_ready or t0) - t0, 3),
            "tokens_per_sec": round(tps, 1),            # median recent interval
            "tokens_per_sec_avg": round(avg, 1),        # post-startup average
            "n_windows": len(windows),
            "stream_restarts": restarts,
            "mfu": _mfu(tps, n_params, peak_tflops),
            "peak_tflops": peak_tflops,
            "final_loss": float(tot)}


def flatten_model(model, arch: str, *, exclude_1d: bool, include_extra: bool):
    """Byte-identical to EnsembleDataset._extract, on purpose -- one layout, one
    implementation. Returns (w, meta_fragment)."""
    if include_extra:
        extra, extra_schema = extract_extra_flat(model, arch, exclude_1d=exclude_1d)
    else:
        extra, extra_schema = np.zeros(0, dtype=np.float32), []
    flats, schemas = [], []
    for layer in get_layers(model, arch):
        f, s = extract_block_flat(layer, exclude_1d=exclude_1d)
        flats.append(f)
        schemas.append(s)
    sizes = {len(f) for f in flats}
    if len(sizes) != 1:
        raise ValueError(f"{arch}: blocks not uniform ({sorted(sizes)})")
    w = np.concatenate([extra] + flats).astype(np.float32)
    return w, {
        "n_layers": len(flats),
        "block_size": sizes.pop(),
        "extra_size": int(extra.size),
        "n_params": int(w.size),
        "weight_std": float(w.std()),
        "schema": [[e.name, list(e.shape)] for e in schemas[0]],
        "extra_schema": [[e.name, list(e.shape)] for e in extra_schema],
    }


# ---------------------------------------------------------------------------

def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--mode", default="branch",
                   choices=["trunk", "branch", "plan", "verify"])
    p.add_argument("--verify_domains", action="store_true",
                   help="open all five domain streams and pull a doc each")
    p.add_argument("--verify_mixture", action="store_true",
                   help="build one 5-way interleaved stream and pull 2 blocks. "
                        "An all-anchor zoo (any N<=20) cannot reach this path, "
                        "so it is the only cheap check on it.")
    p.add_argument("--arch", default="gpt2_zoo_mini")
    p.add_argument("--zoo_dir", default=None,
                   help="default $ARTIFACT_DIR/zoo/<arch>")
    p.add_argument("--beta", type=float, default=0.30,
                   help="fraction of training done AFTER the branch point")
    p.add_argument("--n_members", type=int, default=100)
    p.add_argument("--n_anchor_branches", type=int, default=4)
    # Dirichlet concentration for the singleton draws, and the rejection floor.
    # Defaults reproduce the Phase 2 plan exactly -- do not change them for a real
    # zoo. They are exposed for the §5 diversity probe, where the point is to make
    # the singletons AS CLOSE as Phase 2's closest pair rather than typical.
    #
    # alpha is the knob that controls closeness; min_l1_gap only relaxes a
    # rejection test and cannot cluster draws. Measured: Phase 2's 80 singletons
    # (alpha=1, gap=0.15) have a closest pair at L1 0.173, and 6 draws at alpha=8
    # land at 0.169 -- matched. alpha=1 with only 6 draws gives ~0.60, 4x too easy.
    p.add_argument("--alpha", type=float, default=1.0,
                   help="Dirichlet concentration for singleton pi. 1.0 = uniform "
                        "on the simplex (the Phase 2 default); higher clusters "
                        "draws near the barycentre, which is the hard case.")
    p.add_argument("--min_l1_gap", type=float, default=0.15,
                   help="reject a singleton within this L1 of an anchor or an "
                        "already-accepted draw")
    p.add_argument("--member_idx", type=int, default=None)
    p.add_argument("--tokens_per_param", type=int, default=20)
    p.add_argument("--n_ctx", type=int, default=1024)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--grad_accum", type=int, default=8)
    p.add_argument("--peak_lr", type=float, default=6e-4)
    p.add_argument("--warmup_frac", type=float, default=0.01)
    p.add_argument("--trunk_seed", type=int, default=1234)
    p.add_argument("--plan_seed", type=int, default=0)
    # RESEARCH_PLAN §2.2 / Phase 0.6. GPT-2 has biases on every projection and a
    # gain on every LayerNorm. exclude_1d=True would omit them from D, so every
    # generated model would inherit member 0's -- fine for a noise ensemble where
    # all members share them by construction, wrong for a zoo where they differ.
    # Cost of including them is 13*d per layer, ~0.1% of D. Default flipped here.
    p.add_argument("--exclude_1d", type=lambda s: s.lower() == "true", default=False)
    p.add_argument("--include_extra", type=lambda s: s.lower() == "true", default=True)
    p.add_argument("--dropout", type=float, default=None)
    p.add_argument("--log_every", type=int, default=50)
    # bf16 DENSE peak of the device in use. B200 SXM = 2250 TFLOPS. A flag, not a
    # constant, so the MFU denominator is visible and correctable rather than
    # buried -- every estimate in RESEARCH_PLAN §4.6 rides on this number.
    # MEASURED achievable bf16 dense peak, not the spec sheet. A GEMM sweep on a
    # B200 tops out at 1662 TFLOPS (8192^3), where the datasheet says 2250 --
    # dividing by 2250 understated every MFU here by 26%. Override per device:
    # rtx-batch is a different number, and the ktok/s is device-independent
    # whereas this percentage is not.
    p.add_argument("--peak_tflops", type=float, default=1662.4)
    p.add_argument("--no_wandb", action="store_true",
                   help="disable W&B. Matches train_stack/train_flow/eval_stack.")
    wb.add_wandb_args(p)
    args = p.parse_args()

    if args.verify_domains or args.verify_mixture or args.mode == "verify":
        rc = 0
        if args.verify_domains or args.mode == "verify":
            rc = verify_domains()
        if args.verify_mixture or args.mode == "verify":
            if rc:
                log("skipping the interleave check: fix the domain ids first")
            else:
                rc |= verify_mixture()
        return rc

    art = os.environ.get("ARTIFACT_DIR", "./artifacts")
    zoo_dir = args.zoo_dir or os.path.join(art, "zoo", args.arch)
    os.makedirs(zoo_dir, exist_ok=True)

    plan = build_zoo_plan(args.n_members, args.n_anchor_branches,
                          seed=args.plan_seed, alpha=args.alpha,
                          min_l1_gap=args.min_l1_gap)
    counts = zoo_param_count(args.arch)
    T = token_budget(args.arch, args.tokens_per_param)
    tok_per_step = args.batch_size * args.grad_accum * args.n_ctx
    total_steps = max(int(T / tok_per_step), 1)
    trunk_steps = int(round((1.0 - args.beta) * total_steps))
    branch_steps = total_steps - trunk_steps
    warmup = max(int(args.warmup_frac * total_steps), 1)

    if args.mode == "plan":
        split = holdout_split(plan)
        out = {"arch": args.arch, "beta": args.beta, "n_members": args.n_members,
               "param_count": counts, "token_budget": T,
               "tokens_per_step": tok_per_step, "total_steps": total_steps,
               "trunk_steps": trunk_steps, "branch_steps": branch_steps,
               "domains": list(DOMAINS), "members": plan, "holdout": split}
        path = os.path.join(zoo_dir, "zoo_plan.json")
        atomic_write_json(path, out)
        log(f"{args.arch}: D={counts['total']:,} "
            f"({100*counts['embedding_fraction']:.1f}% embedding)  "
            f"T={T/1e9:.2f}B tok  steps={total_steps} "
            f"(trunk {trunk_steps} + branch {branch_steps})")
        log(f"holdout: {len(split['interior'])} interior + {len(split['vertex'])} "
            f"vertex, {len(split['train'])} train")
        log(f"plan -> {path}")
        return 0

    from transformers import AutoTokenizer
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_ID)
    trunk_path = os.path.join(zoo_dir, "trunk.pt")

    # One W&B group per (arch, beta), so a whole zoo -- its trunk, its 12
    # branches, its gate and its spectrum -- collapses into one expandable row
    # instead of 15 unrelated ones. Deliberately NOT the default group (the
    # Slurm job id): a beta arm spans many job ids, and the job id is what makes
    # the runs hard to relate.
    tag = f"b{round(args.beta * 100):03d}"
    # N is in the group name because a (arch, beta) pair spans MORE THAN ONE ZOO:
    # the Phase 1 calibration was N=12 and Phase 2 is N=100 at the same beta.
    # Without it the 100 Phase 2 branches land in the same expandable row as the
    # 12 calibration ones and the per-domain PPL charts mix two experiments.
    group = f"zoo_{args.arch}_{tag}_n{args.n_members}"

    if args.mode == "trunk":
        if os.path.exists(trunk_path):
            log(f"trunk already at {trunk_path}; nothing to do")
            return 0
        wb.init_run(job_type="zoo_trunk", config=vars(args),
                    tags=[args.arch, tag, "trunk"],
                    enabled=not args.no_wandb, artifact_dir=zoo_dir,
                    name_suffix=f"{tag}_trunk",
                    group=args.wandb_group or group,
                    project=args.wandb_project)
        wb.summary({"arch": args.arch, "beta": args.beta,
                    "n_params": counts["total"],
                    "embedding_fraction": counts["embedding_fraction"],
                    "token_budget": T, "total_steps": total_steps,
                    "trunk_steps": trunk_steps, "branch_steps": branch_steps})
        log(f"building {args.arch} from seed {args.trunk_seed}: "
            f"D={counts['total']:,}")
        model = build_zoo_model(args.arch, seed=args.trunk_seed,
                                dropout=args.dropout).to(device)
        opt = torch.optim.AdamW(model.parameters(), lr=args.peak_lr,
                                betas=(0.9, 0.95), weight_decay=0.1)
        uniform = np.full(n_domains(), 1.0 / n_domains())
        log(f"trunk: {trunk_steps} steps ({(1-args.beta):.0%} of {total_steps}) "
            f"on the uniform mixture")
        thr = train_span(model, opt, uniform, tokenizer=tokenizer, device=device,
                         n_ctx=args.n_ctx, batch_size=args.batch_size,
                         grad_accum=args.grad_accum, start_step=0,
                         n_steps=trunk_steps, total_steps=total_steps,
                         peak_lr=args.peak_lr, warmup=warmup,
                         seed=args.trunk_seed, log_every=args.log_every,
                         n_params=counts["total"], peak_tflops=args.peak_tflops)
        # Write via .tmp + os.replace. A plain torch.save that is killed at the
        # walltime leaves a TRUNCATED trunk.pt, and --mode trunk gates on
        # os.path.exists -- so it would then report "nothing to do" while every
        # branch fails to torch.load it.
        tmp = trunk_path + ".tmp"
        torch.save({"model": model.state_dict(), "optimizer": opt.state_dict(),
                    "step": trunk_steps, "total_steps": total_steps,
                    "arch": args.arch, "beta": args.beta,
                    "trunk_seed": args.trunk_seed,
                    "throughput": thr,
                    "config": zoo_config(args.arch, dropout=args.dropout).to_dict()},
                   tmp)
        os.replace(tmp, trunk_path)
        atomic_write_json(os.path.join(zoo_dir, "trunk_throughput.json"),
                          {"arch": args.arch, "beta": args.beta,
                           "n_params": counts["total"],
                           "peak_tflops": args.peak_tflops, **thr})
        log(f"trunk -> {trunk_path}")
        log(f"trunk throughput: {thr['tokens_per_sec']/1e3:.1f} ktok/s"
            + (f"  mfu {100*thr['mfu']:.1f}%" if thr["mfu"] is not None else ""))
        wb.summary({f"throughput/{k}": v for k, v in thr.items()
                    if v is not None})
        wb.finish()
        return 0

    # ---- branch ----
    if args.member_idx is None:
        log("--member_idx is required for --mode branch "
            "(use $SLURM_ARRAY_TASK_ID)")
        return 2
    m = plan[args.member_idx]
    out_path = os.path.join(zoo_dir, f"w_{args.member_idx}.npy")

    # THIS CHECK MUST COME BEFORE THE w_<i>.npy SKIP BELOW.
    #
    # zoo_dir carries no beta in its default path, and the skip returns 0. So
    # running beta=0.15 and then beta=0.30 into the same directory used to hand
    # back beta=0.15 weights labelled beta=0.30 -- silently, exit 0, every
    # member. That invalidates a whole calibration with no error anywhere. The
    # trunk's own beta check at :373 is too late to catch it, because the skip
    # returns before the trunk is ever loaded.
    #
    # slurm/zoo_{trunk,branch}.sbatch also give each beta its own ZOO_ROOT, but
    # that only protects the sbatch path; this protects the CLI too.
    meta_path = os.path.join(zoo_dir, "zoo_meta.json")
    if os.path.exists(meta_path):
        with open(meta_path) as f:
            prior = json.load(f)
        for key, mine in (("beta", args.beta), ("arch", args.arch),
                          ("n_members", args.n_members)):
            theirs = prior.get(key)
            stale = (abs(theirs - mine) > 1e-9
                     if isinstance(mine, float) else theirs != mine)
            if theirs is not None and stale:
                log(f"REFUSING: {zoo_dir} already holds a zoo with {key}="
                    f"{theirs!r}, this run asks for {key}={mine!r}.")
                log("Give each configuration its own directory "
                    "(ZOO_ROOT=$ARTIFACT_DIR/zoo_b030, etc). Mixing them would "
                    "silently relabel already-written members.")
                return 2

    if os.path.exists(out_path):
        log(f"member {args.member_idx} already at {out_path}; nothing to do")
        return 0
    if not os.path.exists(trunk_path):
        log(f"no trunk at {trunk_path}; run --mode trunk first")
        return 2

    ck = torch.load(trunk_path, map_location="cpu", weights_only=False)
    if ck["arch"] != args.arch or abs(ck["beta"] - args.beta) > 1e-9:
        log(f"trunk is arch={ck['arch']} beta={ck['beta']}, run asks for "
            f"{args.arch} / {args.beta}")
        return 2
    model = build_zoo_model(args.arch, seed=None, dropout=args.dropout)
    model.load_state_dict(ck["model"])
    model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.peak_lr,
                            betas=(0.9, 0.95), weight_decay=0.1)
    # The point of training our own trunk (§4.1): Adam's moments continue rather
    # than restarting, so the branch point injects no transient.
    opt.load_state_dict(ck["optimizer"])

    pi = np.asarray(m["pi"], dtype=np.float64)
    wb.init_run(job_type="zoo_branch", config={**vars(args), **m},
                tags=[args.arch, tag, m["kind"], m["mixture_id"]],
                enabled=not args.no_wandb, artifact_dir=zoo_dir,
                name_suffix=f"{tag}_member{args.member_idx:03d}",
                group=args.wandb_group or group,
                project=args.wandb_project)
    wb.summary({"arch": args.arch, "beta": args.beta,
                "member_idx": args.member_idx, "kind": m["kind"],
                "mixture_id": m["mixture_id"], "branch": m["branch"],
                "n_params": counts["total"],
                **{f"pi/{d}": float(pi[i]) for i, d in enumerate(DOMAINS)}})
    log(f"member {args.member_idx} ({m['kind']}, {m['mixture_id']}): "
        f"pi={np.round(pi, 3).tolist()}")
    log(f"branch: steps {ck['step']}..{total_steps} ({branch_steps} steps)")
    thr = train_span(model, opt, pi, tokenizer=tokenizer, device=device,
                     n_ctx=args.n_ctx, batch_size=args.batch_size,
                     grad_accum=args.grad_accum, start_step=int(ck["step"]),
                     n_steps=branch_steps, total_steps=total_steps,
                     peak_lr=args.peak_lr, warmup=warmup,
                     seed=10_000 + args.member_idx, log_every=args.log_every,
                     n_params=counts["total"], peak_tflops=args.peak_tflops)

    model.eval().to("cpu")
    w, frag = flatten_model(model, args.arch, exclude_1d=args.exclude_1d,
                            include_extra=args.include_extra)
    tmp = out_path + ".tmp.npy"
    np.save(tmp, w)
    os.replace(tmp, out_path)
    log(f"member {args.member_idx}: D={w.size:,}  std={w.std():.6g} -> {out_path}")

    # Each branch writes the shared meta. Identical content from every writer, and
    # atomic_write_json renames into place, so a concurrent array is safe.
    atomic_write_json(os.path.join(zoo_dir, "zoo_meta.json"), {
        "layout_version": ENSEMBLE_LAYOUT_VERSION,
        "arch": args.arch,
        "n_members": args.n_members,
        "beta": args.beta,
        "exclude_1d": args.exclude_1d,
        "include_extra": args.include_extra,
        "tokens_per_param": args.tokens_per_param,
        "plan_seed": args.plan_seed,
        "alpha": args.alpha,
        "min_l1_gap": args.min_l1_gap,
        "n_anchor_branches": args.n_anchor_branches,
        "total_steps": total_steps,
        "trunk_steps": trunk_steps,
        "trunk_seed": args.trunk_seed,
        "domains": list(DOMAINS),
        "members": plan,
        "holdout": holdout_split(plan),
        **frag,
    })
    # Per-member record, so a partially-complete zoo is still self-describing.
    # Throughput is sealed here as well as logged: /scratch is purged at 30 days
    # and the measured MFU is a reportable number, not a debugging aid.
    atomic_write_json(os.path.join(zoo_dir, f"member_{args.member_idx}.json"),
                      {**m, "weight_std": frag["weight_std"],
                       "n_params": frag["n_params"],
                       "peak_tflops": args.peak_tflops,
                       "throughput": thr})
    log(f"member {args.member_idx} throughput: {thr['tokens_per_sec']/1e3:.1f} "
        f"ktok/s" + (f"  mfu {100*thr['mfu']:.1f}%" if thr["mfu"] is not None
                     else ""))
    wb.summary({f"throughput/{k}": v for k, v in thr.items() if v is not None})
    wb.summary({"weight_std": frag["weight_std"], "D": frag["n_params"]})
    wb.finish()
    return 0


if __name__ == "__main__":
    sys.exit(main())
