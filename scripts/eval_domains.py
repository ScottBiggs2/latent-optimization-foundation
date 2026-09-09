#!/usr/bin/env python
"""
eval_domains.py -- the precondition that gates the whole zoo (RESEARCH_PLAN §4.3).

Measures per-domain perplexity for every member of a zoo and reports whether the
mixture axis is actually visible in the models.

Why this exists, and why it runs BEFORE any PCA
-----------------------------------------------
If beta is too small, every branch is essentially the trunk. Per-domain performance
barely moves, and then the spectrum (§4.4), the conditioning (§6.2) and the slope
figure (§6.5) ALL come out flat -- for a reason that has nothing to do with weight
space. This check costs minutes and can save the entire experiment.

The pass condition, from §4.3:

  1. SEPARATION -- for each anchor mixture, the model trained on it is best-in-zoo
     on its own dominant domain.
  2. SIGNAL > NOISE -- the between-anchor spread on a domain exceeds the
     within-anchor spread across that anchor's branches.

If it fails, the answer is a LARGER BETA and a re-run. It is not a reframe, and it
is not something to fix downstream.

  python scripts/eval_domains.py --arch gpt2_zoo_mini --n_docs 64
"""

from __future__ import annotations

import argparse
import json
import os
import socket
import sys
from collections import defaultdict
from typing import Dict, List

import numpy as np
import torch

from llmzoo.artifacts.io import atomic_write_json
from llmzoo.data.mixtures import DOMAINS, DOMAIN_SOURCES, is_holdout
from llmzoo.models.registry import build_zoo_model
from llmzoo.models.weight_extractor import build_stack_spec, write_stack_to_model
import llmzoo.wandb_utils as wb

TOKENIZER_ID = "gpt2"


def held_out_text(domain: str, tokenizer, n_docs: int, n_ctx: int, seed: int):
    """
    A fixed evaluation slice per domain, identical for every member and DISJOINT
    from what the zoo trained on.

    Disjointness is by construction, not by offset: `is_holdout` hashes each
    document's TEXT, train_zoo's `mixture_stream` drops the documents that hash
    into the holdout, and this keeps exactly those. Content hashing rather than
    position matters because these corpora contain exact duplicates -- measured
    on books, only 81 of 126 consecutive documents were unique. See
    mixtures.HOLDOUT_EVERY for why the original `skip(100_000)` was wrong in two
    further ways.

    No `.shuffle()` here, deliberately. The slice only has to be fixed and
    identical across members, which it already is. Shuffling would force the
    buffer to accumulate `buffer_size` HELD-OUT documents, i.e. scan ~64x that
    many raw ones -- on the books domain (195 KB/doc) a 2,000-doc buffer means
    streaming the whole corpus to pick 2 documents.
    """
    from datasets import load_dataset

    src = DOMAIN_SOURCES[domain]
    ds = load_dataset(src["path"], src["name"], split=src["split"], streaming=True)
    ds = ds.filter(lambda r, _c=src["text_column"]: is_holdout(r[_c]))
    buf: List[int] = []
    blocks = []
    need = n_ctx + 1
    for row in ds:
        t = row.get(src["text_column"])
        if not t:
            continue
        buf.extend(tokenizer(t, add_special_tokens=False)["input_ids"])
        buf.append(tokenizer.eos_token_id)
        while len(buf) >= need and len(blocks) < n_docs:
            blocks.append(torch.tensor(buf[:need], dtype=torch.long))
            buf = buf[need:]
        if len(blocks) >= n_docs:
            break
    if len(blocks) < n_docs:
        raise RuntimeError(
            f"domain {domain}: only {len(blocks)}/{n_docs} eval blocks -- the "
            f"held-out split is too small. Lower --n_docs or raise "
            f"HOLDOUT_EVERY.")
    return torch.stack(blocks)


def build_evalsets(tokenizer, n_docs: int, n_ctx: int, seed: int,
                   cache_dir: str):
    """
    The five held-out slices, cached to disk and reused.

    Two reasons this is not just a speedup. First, `held_out_text` calls
    `.skip(100_000)` on a STREAM, which really does pull and discard 100k
    documents per domain -- five times, on every invocation. Second, and more
    important: the beta calibration compares perplexities ACROSS zoos, so every
    beta must be scored on byte-identical text. Re-deriving the slice each time
    makes that a property of dataset-revision stability rather than something we
    control. Caching makes it exact.

    Keyed on (seed, n_docs, n_ctx) because those are the only inputs that change
    the contents.
    """
    path = os.path.join(cache_dir,
                        f"domain_evalset_s{seed}_n{n_docs}_c{n_ctx}.npz")
    if os.path.exists(path):
        z = np.load(path)
        missing = [d for d in DOMAINS if d not in z.files]
        if not missing:
            print(f"[eval_domains] reusing held-out slices from {path}")
            return {d: torch.from_numpy(z[d]).long() for d in DOMAINS}
        print(f"[eval_domains] cache at {path} lacks {missing}; rebuilding")

    print(f"[eval_domains] building held-out slices ({n_docs} blocks/domain); "
          f"this streams and discards 100k docs per domain, so it is the slow "
          f"part -- it happens once")
    sets = {d: held_out_text(d, tokenizer, n_docs, n_ctx, seed) for d in DOMAINS}
    os.makedirs(cache_dir, exist_ok=True)
    tmp = f"{path}.tmp.{socket.gethostname()}.{os.getpid()}.npz"
    np.savez(tmp, **{d: v.numpy() for d, v in sets.items()})
    os.replace(tmp, path)
    print(f"[eval_domains] cached -> {path}")
    return sets


@torch.no_grad()
def perplexity(model, blocks, device, batch_size=8) -> float:
    model.eval()
    tot_nll, tot_tok = 0.0, 0
    for i in range(0, len(blocks), batch_size):
        b = blocks[i:i + batch_size].to(device)
        x, y = b[:, :-1], b[:, 1:]
        with torch.autocast("cuda", dtype=torch.bfloat16,
                            enabled=(device.type == "cuda")):
            logits = model(input_ids=x).logits
        nll = torch.nn.functional.cross_entropy(
            logits.reshape(-1, logits.size(-1)).float(), y.reshape(-1),
            reduction="sum")
        tot_nll += float(nll)
        tot_tok += y.numel()
    return float(np.exp(tot_nll / max(tot_tok, 1)))


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--arch", default="gpt2_zoo_mini")
    p.add_argument("--zoo_dir", default=None)
    p.add_argument("--n_docs", type=int, default=64,
                   help="held-out blocks per domain; 64 x 1024 tok is plenty")
    p.add_argument("--n_ctx", type=int, default=1024)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--members", type=int, nargs="*", default=None)
    p.add_argument("--evalset_cache", default=None,
                   help="Where the held-out slices are cached. Defaults to "
                        "$ARTIFACT_DIR, deliberately NOT zoo_dir, so every beta "
                        "of the calibration is scored on identical text.")
    p.add_argument("--no_wandb", action="store_true")
    args = p.parse_args()

    from transformers import AutoTokenizer

    art = os.environ.get("ARTIFACT_DIR", "./artifacts")
    zoo_dir = args.zoo_dir or os.path.join(art, "zoo", args.arch)
    with open(os.path.join(zoo_dir, "zoo_meta.json")) as f:
        zmeta = json.load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_ID)

    # Same group as the trunk and branches of this zoo, so the gate lands in the
    # row it belongs to rather than as an orphan run.
    tag = f"b{round(float(zmeta['beta']) * 100):03d}"
    wb.init_run(job_type="zoo_gate", config={**vars(args), "beta": zmeta["beta"]},
                tags=[args.arch, tag, "gate"], enabled=not args.no_wandb,
                artifact_dir=zoo_dir, name_suffix=f"{tag}_gate",
                group=f"zoo_{args.arch}_{tag}")

    # Cache in the ARTIFACT_DIR root, NOT under zoo_dir: the whole point is that
    # every beta of the calibration scores on the same bytes, and each beta has
    # its own zoo_dir.
    evalsets = build_evalsets(tokenizer, args.n_docs, args.n_ctx, args.seed,
                              args.evalset_cache or art)

    # One model object, reused: write each member's flat vector into it rather than
    # constructing 100 GPT-2s. Same write-back path eval_stack.py uses.
    model = build_zoo_model(args.arch, seed=0).to(device)
    # build_stack_spec returns (flat_now, spec); the flat vector is this model's
    # current weights, which we are about to overwrite, so only the spec is kept.
    _flat0, spec = build_stack_spec(model, args.arch,
                                    exclude_1d=zmeta["exclude_1d"],
                                    include_extra=zmeta["include_extra"])
    if _flat0.size != int(zmeta["n_params"]):
        raise RuntimeError(
            f"spec gives D={_flat0.size:,} but the zoo was flattened at "
            f"D={int(zmeta['n_params']):,}. exclude_1d/include_extra disagree with "
            f"how train_zoo.py wrote these members.")

    idxs = args.members if args.members is not None else list(
        range(int(zmeta["n_members"])))
    members = {m["idx"]: m for m in zmeta["members"]}
    rows = []
    for i in idxs:
        wp = os.path.join(zoo_dir, f"w_{i}.npy")
        if not os.path.exists(wp):
            print(f"  member {i}: MISSING {wp}, skipping")
            continue
        w = np.load(wp)
        write_stack_to_model(w, model, args.arch, spec)
        ppl = {d: perplexity(model, evalsets[d], device, args.batch_size)
               for d in DOMAINS}
        m = members[i]
        rows.append({"idx": i, "kind": m["kind"], "mixture_id": m["mixture_id"],
                     "branch": m["branch"], "pi": m["pi"], "ppl": ppl})
        print(f"  member {i:>3} {m['mixture_id']:<20} "
              + "  ".join(f"{d[:4]}={ppl[d]:8.2f}" for d in DOMAINS), flush=True)
        # One history point per member, x-axis = member index, so the per-domain
        # PPL spread across the zoo is a chart rather than 12 log lines.
        wb.log({**{f"ppl/{d}": ppl[d] for d in DOMAINS},
                "member_idx": i}, step=i)

    # ---- the two gate conditions ----
    by_anchor: Dict[str, List[dict]] = defaultdict(list)
    for r in rows:
        if r["kind"] == "anchor":
            by_anchor[r["mixture_id"]].append(r)

    report = {"arch": args.arch, "n_evaluated": len(rows), "rows": rows}
    verdict = {"separation": None, "signal_over_noise": None}

    if by_anchor:
        wins, checks = 0, 0
        for dom in DOMAINS:
            key = f"anchor_{dom}"
            if key not in by_anchor:
                continue
            checks += 1
            own = float(np.mean([r["ppl"][dom] for r in by_anchor[key]]))
            others = [float(np.mean([r["ppl"][dom] for r in v]))
                      for k, v in by_anchor.items() if k != key]
            if others and own < min(others):
                wins += 1
            print(f"  [sep] {dom:<14} own-anchor ppl {own:8.2f}  "
                  f"best-other {min(others) if others else float('nan'):8.2f}  "
                  f"{'PASS' if others and own < min(others) else 'FAIL'}")
        verdict["separation"] = {"wins": wins, "checks": checks,
                                 "pass": checks > 0 and wins == checks}

        ratios = {}
        for dom in DOMAINS:
            means, within = [], []
            for k, v in by_anchor.items():
                vals = [r["ppl"][dom] for r in v]
                means.append(float(np.mean(vals)))
                if len(vals) > 1:
                    within.append(float(np.std(vals, ddof=1)))
            if len(means) > 1 and within:
                between = float(np.std(means, ddof=1))
                noise = float(np.mean(within))
                ratios[dom] = between / max(noise, 1e-12)
                print(f"  [snr] {dom:<14} between {between:8.3f}  "
                      f"within {noise:8.3f}  ratio {ratios[dom]:6.2f}")
        verdict["signal_over_noise"] = {
            "per_domain": ratios,
            "min_ratio": min(ratios.values()) if ratios else None,
            "pass": bool(ratios) and min(ratios.values()) > 1.0,
        }

    report["verdict"] = verdict
    out = os.path.join(zoo_dir, "domain_separation.json")
    atomic_write_json(out, report)
    print(f"\n[eval_domains] -> {out}")

    ok = all(v and v.get("pass") for v in verdict.values())

    # The gate verdict as run-summary columns, so the beta sweep is scannable in
    # the runs table without opening anything. Both sub-conditions are recorded
    # separately and the SNR is broken out per domain, because at N=12 only 3 of
    # 5 domains have an anchor: a failure on books or multilingual is a different
    # diagnosis from a failure on separation, and only the latter is about beta.
    sep, snr = verdict["separation"], verdict["signal_over_noise"]
    wb.summary({
        "gate/pass": ok,
        "gate/separation_pass": bool(sep and sep.get("pass")),
        "gate/separation_wins": (sep or {}).get("wins"),
        "gate/separation_checks": (sep or {}).get("checks"),
        "gate/snr_pass": bool(snr and snr.get("pass")),
        "gate/snr_min_ratio": (snr or {}).get("min_ratio"),
        **{f"gate/snr_{d}": v
           for d, v in ((snr or {}).get("per_domain") or {}).items()},
        "gate/n_evaluated": len(rows),
        "gate/n_anchor_groups": len(by_anchor),
    })
    wb.finish()
    print("\n" + ("=" * 72))
    if ok:
        print("GATE PASS -- the mixture axis is visible. Proceed to the PCA fit.")
    else:
        print("GATE FAIL -- mixture identity is NOT measurable in the models.")
        print("Per RESEARCH_PLAN §4.3 the response is a LARGER BETA and a re-run.")
        print("Do NOT proceed to the PCA fit and do NOT reframe: a flat spectrum")
        print("measured on a zoo that failed this gate says nothing about weight")
        print("space.")
    print("=" * 72)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
