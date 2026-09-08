"""
Whole-stack evaluation: PCA-only and PCA+VAE, at one or more ranks.

Every arm writes a full stack into a live model, measures it, and restores:

  pca_only         codes[i][:k] -> inverse_transform            reconstruction
  vae              codes[i][:k] -> VAE round trip -> inverse    the VAE's own cost
  generate         z ~ N(0,I)   -> VAE decode (CFG) -> inverse  VAE prior sample
  gauss_codes      z ~ N(mu_f, s_f) from code stats -> inverse  THE NULL MODEL
  flow_codes       flow sample in code space -> inverse         flow over codes
  flow_latent      flow sample in latent space -> VAE -> inverse flow over latents
  flow_rt_codes    codes -> reverse ODE -> forward ODE -> inverse  ODE consistency
  flow_rt_latent   same, in latent space

Four rules for reading the output
---------------------------------
1. `pca_only` at k = N-1 is a SELF-TEST, not a result. Centering removes one degree
   of freedom, so the rank bound makes it exact; cosine other than 1.000000 there is
   a bug (misstep 12).
2. Gate on dPPL, never on cosine. A measured 0.99939 cosine came with +72,468% PPL,
   so cosine below about 0.9999 tells you nothing (misstep 11).
3. Judge the flow arms against `gauss_codes`, not against `pca_only`. On a
   manufactured noise ensemble the per-family code distribution is near-Gaussian by
   construction, so a flow that merely reproduces the prior will look fine next to
   `pca_only` and identical to `gauss_codes`.
4. A NEGATIVE dPPL at low rank is not success. It means the target was a noisy
   ensemble member and truncation removed some of the noise (misstep 14).

Benchmark accuracy (--bench) is measured on the SAME in-memory reconstructed model,
between write-back and restore, so each arm pays one model load rather than one per
benchmark. It is opt-in because it is the expensive axis: 3 benchmarks x 200
questions x 4 choices is ~2,400 forward passes PER ARM PER ARCH PER RANK.

    python eval_stack.py --run_name perfam --k 99 50
    python eval_stack.py --run_name perfam --k 50 --arms pca_only vae generate
    python eval_stack.py --run_name perfam --arms pca_only vae --bench
"""

from __future__ import annotations

import argparse
import gc
import json
import os
from typing import Dict, List, Optional

import numpy as np
import torch

from artifact_io import read_json
from models.registry import get_arch_config, load_model, build_tiny_model
from models.weight_extractor import (
    read_stack_from_model, write_stack_to_model,
)
from run_bundle import load_run
import wandb_utils as wb


# ---------------------------------------------------------------------------
# Arms
# ---------------------------------------------------------------------------

ARMS = ("pca_only", "vae", "generate", "gauss_codes",
        "flow_codes", "flow_latent", "flow_rt_codes", "flow_rt_latent")

# Which artifacts each arm needs, so load_run can skip the expensive ones. Building
# the EnsembleDataset memmaps a ~1.2 GB mean per family, and --arms pca_only should
# not fail merely because vae_k50/ is absent.
_ARM_WANTS = {
    "pca_only":        ("dataset", "pca"),
    "vae":             ("dataset", "pca", "codes", "vae"),
    "generate":        ("dataset", "pca", "codes", "vae"),
    "gauss_codes":     ("dataset", "pca", "codes"),
    "flow_codes":      ("dataset", "pca", "codes", "flow"),
    "flow_latent":     ("dataset", "pca", "codes", "vae", "flow"),
    "flow_rt_codes":   ("dataset", "pca", "codes", "flow"),
    "flow_rt_latent":  ("dataset", "pca", "codes", "vae", "flow"),
}


def wants_for(arms) -> tuple:
    out = set()
    for a in arms:
        out.update(_ARM_WANTS.get(a, ("dataset", "pca")))
    return tuple(sorted(out))


def flow_spaces_for(arms) -> tuple:
    spaces = set()
    for a in arms:
        if a in ("flow_codes", "flow_rt_codes"):
            spaces.add("codes")
        if a in ("flow_latent", "flow_rt_latent"):
            spaces.add("latent")
    return tuple(sorted(spaces))


# ---------------------------------------------------------------------------
# Metrics
#
# write_stack_to_model / read_stack_from_model live in models/weight_extractor.py
# so the flow sampler and the benchmark path can reuse them without importing an
# eval script.
# ---------------------------------------------------------------------------

def stack_metrics(recon: np.ndarray, w0: np.ndarray) -> dict:
    r = np.asarray(recon, dtype=np.float64)
    w = np.asarray(w0, dtype=np.float64)
    denom = np.linalg.norm(r) * np.linalg.norm(w) + 1e-30
    return {"cosine_sim": float(np.dot(r, w) / denom),
            "mse": float(np.mean((r - w) ** 2)),
            "rel_l2": float(np.linalg.norm(r - w) / (np.linalg.norm(w) + 1e-30))}


# ---------------------------------------------------------------------------
# Multiple-choice benchmarks
#
# The scoring primitives live in eval_core.py, which takes any nn.Module and knows
# nothing about PCA / VAEs / bundles. That is what lets the measurement sit inside the
# arm loop on a model that has just had a reconstruction written into it.
# ---------------------------------------------------------------------------

ALL_BENCHMARKS = ("mmlu", "hellaswag", "gpqa")


def _synthetic_examples(n_questions: int, seed: int) -> list:
    """
    Fixed nonsense MCExamples, for smoke tests only.

    A random-init tiny model scores at chance on any real benchmark, so reporting
    an mmlu delta there is noise dressed as a measurement -- main() refuses that.
    But the plumbing still needs a cheap regression test, and this provides one
    with no download and no meaning.
    """
    import random
    from data.mc_loader import MCExample
    rng = random.Random(seed)
    vocab = ["alpha", "beta", "gamma", "delta", "epsilon", "zeta", "eta", "theta"]
    out = []
    for _ in range(n_questions):
        ctx = " ".join(rng.choice(vocab) for _ in range(8)) + " ?"
        out.append(MCExample(context=ctx,
                             choices=[" " + rng.choice(vocab) for _ in range(4)],
                             gold_idx=rng.randrange(4)))
    return out


class _SyntheticTokenizer:
    """
    Whitespace tokenizer whose ids are guaranteed to fit a tiny model's vocab.

    A tiny model has vocab_size=1000 while the real GPT-2 tokenizer emits ids up to
    50256, so scoring synthetic questions with the real tokenizer indexes the
    embedding out of range and dies on a CUDA device-side assert. score_choices only
    ever calls tokenizer.encode(text) -> list[int], so a stub is enough.

    crc32 rather than hash(): Python salts hash() per process, and a smoke test that
    tokenizes differently on every run is not a regression test.
    """

    def __init__(self, vocab_size: int):
        self.vocab_size = max(int(vocab_size), 2)

    def encode(self, text: str):
        # Whitespace splitting preserves the prefix property score_choices relies
        # on: encode(context + " word") starts with encode(context).
        import zlib
        return [1 + (zlib.crc32(w.encode()) % (self.vocab_size - 1))
                for w in text.split()]


def load_bench_examples(benchmarks, n_questions: int, hf_cache: Optional[str],
                        seed: int) -> Dict[str, list]:
    """
    Load each benchmark's MCExample list once per process.

    Called from main(), not from evaluate_arch: the questions do not depend on arch
    or rank, so loading them per-arch would re-read the datasets 6+ times.
    """
    from data.mc_loader import LOADERS
    out: Dict[str, list] = {}
    for name in benchmarks:
        if name == "synthetic":
            out[name] = _synthetic_examples(n_questions, seed)
        else:
            out[name] = LOADERS[name](n_questions=n_questions, seed=seed,
                                      cache_dir=hf_cache)
        print(f"  [bench] {name}: {len(out[name])} examples")
    return out


def measure_benchmarks(model, tokenizer, examples_by_bench: Dict[str, list],
                       device, max_length: int) -> Dict[str, dict]:
    """Score the model AS IT CURRENTLY IS. Returns {bench: {acc, acc_norm, n_examples}}."""
    from eval_core import compute_mc_accuracy
    out: Dict[str, dict] = {}
    for name, examples in examples_by_bench.items():
        out[name] = compute_mc_accuracy(model, tokenizer, examples, device,
                                        max_length=max_length)
    return out


def bench_deltas(original: Dict[str, dict], arm: Dict[str, dict]) -> Dict[str, dict]:
    """Pair an arm's scores with the baseline's. The delta is the signal, not the
    absolute score -- these are small base models near chance on MMLU/GPQA."""
    out: Dict[str, dict] = {}
    for name, a in arm.items():
        o = original.get(name, {})
        out[name] = {
            "acc": a["acc"], "acc_norm": a["acc_norm"],
            "acc_delta": a["acc"] - o.get("acc", float("nan")),
            "acc_norm_delta": a["acc_norm"] - o.get("acc_norm", float("nan")),
            "n_examples": a["n_examples"],
        }
    return out


# ---------------------------------------------------------------------------
# Per-arch evaluation
# ---------------------------------------------------------------------------

def evaluate_arch(
    arch: str,
    bundle,
    k: int,
    arms: List[str],
    seq_len: int,
    n_sequences: int,
    hf_cache: Optional[str],
    mode: str,
    guidance_scale: float,
    seed: int,
    sample_idx: int = 0,
    *,
    bench_examples: Optional[Dict[str, list]] = None,
    bench_baseline_cache: Optional[dict] = None,
    bench_max_length: Optional[int] = None,
    flow_steps: Optional[int] = None,
    flow_guidance_scale: Optional[float] = None,
    flow_rt_steps: Optional[List[int]] = None,
    dispersion_n: int = 64,
) -> dict:
    from eval_core import compute_perplexity, get_max_context_length

    ds = bundle.dataset
    pca = bundle.pcas.get(arch)
    vae = bundle.vae
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg = get_arch_config(arch)

    model = (build_tiny_model(arch) if mode == "tiny"
             else load_model(arch, cache_dir=hf_cache)).to(device)
    model.eval()

    tok = None
    if mode == "tiny":
        from data.val_loader import get_synthetic_loader
        vocab = cfg["tiny_config"].get("vocab_size",
                                       cfg["tiny_config"].get("n_positions", 1000))
        loader = get_synthetic_loader(vocab_size=vocab, seq_len=seq_len,
                                     n_sequences=n_sequences)
        if bench_examples:
            # NOT the real tokenizer: its ids overflow the tiny vocab and trip a
            # CUDA device-side assert in the embedding lookup.
            tok = _SyntheticTokenizer(vocab)
    else:
        from data.val_loader import get_wikitext2_loader
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(cfg["default_model_id"],
                                            cache_dir=hf_cache,
                                            trust_remote_code=True)
        loader = get_wikitext2_loader(tok, seq_len=seq_len,
                                     n_sequences=n_sequences, cache_dir=hf_cache)

    st = ds.stacks[arch]
    fidx = torch.tensor([st.family_idx], dtype=torch.long, device=device)

    # The target is ensemble member `sample_idx`. Member 0 is the real pretrained
    # model; any other member is w_0 plus augmentation noise, so it has to be
    # materialised and written into the model FIRST, or "original PPL" would be the
    # real model's while the reconstruction target is a different point.
    target = (np.asarray(ds.w0(arch)) if sample_idx == 0
              else ds.materialize_sample(arch, sample_idx, device=device))
    if sample_idx != 0:
        write_stack_to_model(target, model, arch, st)

    # Pristine copy AFTER installing the target, so restore() returns to the target.
    # This must be the WHOLE stack, not a list of block flats: with the extra
    # segment in play a per-block snapshot silently omits the embeddings, so
    # restore() would leave a mutated embedding behind and every arm after the
    # first would measure a corrupted baseline.
    pristine = read_stack_from_model(model, arch, st)

    print(f"  [{arch}] measuring PPL of ensemble member {sample_idx} …")
    base = compute_perplexity(model, loader, device)
    print(f"  [{arch}] member {sample_idx} PPL = {base['perplexity']:.4f}")

    def restore():
        write_stack_to_model(pristine, model, arch, st)

    out = {"arch": arch, "k": k, "n_layers": st.n_layers,
           "n_params": st.n_params, "extra_size": st.extra_size,
           "sample_idx": sample_idx,
           "original_ppl": base["perplexity"],
           "ce_original": base["ce_loss"], "arms": {}}

    # Baseline benchmark scores depend only on (arch, sample_idx) -- they are the
    # same at every rank -- but main() reloads the model for each (k, arch) pair.
    # Without this cache the most expensive measurement in the job is repeated once
    # per rank for nothing.
    base_bench = None
    if bench_examples:
        max_len = bench_max_length or get_max_context_length(model)
        ck = (arch, sample_idx)
        if bench_baseline_cache is not None and ck in bench_baseline_cache:
            base_bench = bench_baseline_cache[ck]
            print(f"  [{arch}] reusing cached baseline benchmark scores")
        else:
            print(f"  [{arch}] measuring baseline benchmarks "
                  f"({', '.join(bench_examples)}) …")
            base_bench = measure_benchmarks(model, tok, bench_examples, device,
                                            max_len)
            if bench_baseline_cache is not None:
                bench_baseline_cache[ck] = base_bench
        for bn, bv in base_bench.items():
            print(f"  [{arch}] baseline {bn:10s} acc={bv['acc']:.4f} "
                  f"acc_norm={bv['acc_norm']:.4f}  (n={bv['n_examples']})")
        out["bench_config"] = {"benchmarks": list(bench_examples),
                               "n_questions": max(len(v) for v in
                                                  bench_examples.values()),
                               "max_length": max_len}
        out["original_bench"] = base_bench

    codes_k = pca.codes(k)[sample_idx]
    codes_t = torch.from_numpy(codes_k.reshape(1, -1).copy()).float().to(device)

    def _decode(codes_1d) -> np.ndarray:
        """Codes -> full stack. Every arm funnels through here."""
        return bundle.decode_codes_to_stack(codes_1d, arch, k=k)

    def _need(obj, arm: str, what: str) -> bool:
        if obj is None:
            print(f"  [{arch}] skipping {arm!r} — no {what} loaded for k={k}")
            return False
        return True

    # --- dispersion reference (RESEARCH_NOTES misstep 19) --------------------
    #
    # dPPL alone CANNOT police a generative arm here. `mean(w) = w_0 +
    # O(s*sigma/sqrt(N))`, so the ensemble mean essentially IS the real pretrained
    # model, and a generator collapsed toward its family mean scores dPPL ~ 0 --
    # better than an honest sample -- while generating nothing. Measured on emb3:
    # flow_codes beat the gauss_codes null by 43x on dPPL while emitting codes at
    # 0.25x the correct RMS.
    #
    # `gauss_codes` cannot catch that, because a collapsed flow beats it by
    # construction. So report the spread of the codes each generative arm actually
    # produced. One k-dimensional draw gives its RMS to about 1/sqrt(2k) -- ~7% at
    # k=99 -- which is far tighter than the 4x collapse being detected, so this costs
    # no extra sampling.
    real_code_rms = None
    if bundle.code_stats is not None and bundle.code_stats.codes is not None:
        _cs = bundle.code_stats
        _rows = _cs.codes[_cs.family_idxs == int(fidx[0])]
        if _rows.shape[0] > 0:
            _rt = torch.from_numpy(_rows[:, :k].copy()).float().to(device)
            _rn = _cs.normalize(_rt, fidx.expand(_rt.shape[0]))
            real_code_rms = float(_rn.pow(2).mean().sqrt())

    def _dispersion(sampler, n: int) -> dict:
        """
        Spread of a generative arm's codes, pooled over `n` draws.

        A batch, not one draw. A single k-dimensional sample would estimate the RMS
        to 1/sqrt(2k) -- about 7% at k=99 -- ONLY if the arm's output were isotropic
        with stable per-sample norms. It is not: measured against a 256-draw
        reference, one draw read 0.14 where the pooled value was 0.249. The
        per-sample norm varies enough that one draw is not representative, so the
        statistic is pooled.

        Cheap either way. Sampling codes is microseconds; the expensive step in this
        function is `inverse_transform`, and that still runs exactly once per arm on
        the first row.
        """
        if bundle.code_stats is None or n < 1:
            return {}
        g = torch.Generator(device=device).manual_seed(seed)
        with torch.no_grad():
            batch = sampler(n, g)
        xn = bundle.code_stats.normalize(batch, fidx.expand(batch.shape[0]))
        rms = float(xn.pow(2).mean().sqrt())
        # Per-sample norms too: an arm that emits ONE off-centre point repeatedly has
        # a healthy pooled RMS and zero diversity.
        per = xn.pow(2).mean(dim=1).sqrt()
        out = {"code_rms": rms, "real_code_rms": real_code_rms,
               "code_rms_n_draws": int(n),
               "code_rms_per_sample_std": float(per.std()) if n > 1 else None}
        if real_code_rms:
            out["code_rms_ratio"] = rms / real_code_rms
        return out

    for arm in arms:
        extra_metrics: dict = {}

        if arm == "pca_only":
            recon = _decode(codes_k)

        elif arm == "vae":
            if not _need(vae, arm, "StackVAE checkpoint"):
                continue
            with torch.no_grad():
                # sample=False: reconstruction fidelity is a deterministic
                # measurement, so use the posterior mean.
                rc, _, _ = vae(codes_t, fidx, sample=False)
            recon = _decode(rc.cpu().numpy().reshape(-1))

        elif arm == "generate":
            if not _need(vae, arm, "StackVAE checkpoint"):
                continue
            def _gen_sampler(n, g, _v=vae):
                z = torch.randn(n, _v.latent_dim, device=device, generator=g)
                return _v.decode_cfg(z, fidx.expand(n),
                                     guidance_scale=guidance_scale)
            g = torch.Generator(device=device).manual_seed(seed)
            with torch.no_grad():
                gc_ = _gen_sampler(1, g)
            extra_metrics = _dispersion(_gen_sampler, dispersion_n)
            recon = _decode(gc_.cpu().numpy().reshape(-1))

        elif arm == "gauss_codes":
            # The NULL MODEL the flows have to beat. On a manufactured noise
            # ensemble the per-family code distribution is near-Gaussian by
            # construction, so if flow_codes does not beat this, the flow learned
            # only the prior and its ΔPPL means nothing on its own.
            if not _need(bundle.code_stats, arm, "code stats"):
                continue
            cs = bundle.code_stats

            def _gauss_sampler(n, g, _cs=cs):
                z = torch.randn(n, k, device=device, generator=g)
                return _cs.denormalize(z, fidx.expand(n))
            g = torch.Generator(device=device).manual_seed(seed)
            with torch.no_grad():
                gz = _gauss_sampler(1, g)
            extra_metrics = _dispersion(_gauss_sampler, dispersion_n)
            recon = _decode(gz.cpu().numpy().reshape(-1))

        elif arm in ("flow_codes", "flow_latent"):
            space = "codes" if arm == "flow_codes" else "latent"
            fm = bundle.flows.get(space)
            if not _need(fm, arm, f"{space}-space flow"):
                continue
            def _flow_sampler(n, g, _fm=fm):
                return _fm.sample_codes(fidx.expand(n), n_steps=flow_steps,
                                        guidance_scale=flow_guidance_scale,
                                        vae_guidance_scale=guidance_scale,
                                        generator=g)
            g = torch.Generator(device=device).manual_seed(seed)
            with torch.no_grad():
                oc = _flow_sampler(1, g)
            extra_metrics = {"flow_steps": flow_steps or fm.default_steps,
                             "flow_guidance_scale": (flow_guidance_scale
                                                     if flow_guidance_scale is not None
                                                     else fm.default_guidance),
                             "flow_trust": fm.trust}
            extra_metrics.update(_dispersion(_flow_sampler, dispersion_n))
            recon = _decode(oc.cpu().numpy().reshape(-1))

        elif arm in ("flow_rt_codes", "flow_rt_latent"):
            space = "codes" if arm == "flow_rt_codes" else "latent"
            fm = bundle.flows.get(space)
            if not _need(fm, arm, f"{space}-space flow"):
                continue
            # A sweep over n_steps is the only way to separate Euler
            # discretisation error (falls like 1/n_steps) from the model failing to
            # be a consistent vector field (does not). The reported arm uses the
            # largest step count; the sweep rides along as a diagnostic.
            steps_list = flow_rt_steps or [flow_steps or fm.default_steps]
            sweep = {}
            for ns in sorted(set(steps_list)):
                with torch.no_grad():
                    oc, diag = fm.round_trip_codes(codes_t, fidx, n_steps=ns)
                sweep[str(ns)] = diag
            best_ns = max(sweep, key=lambda x: int(x))
            extra_metrics = {"flow_rt_n_steps": int(best_ns),
                             "flow_rt_space_rel_l2": sweep[best_ns]["rel_l2_space"],
                             "flow_rt_x0_rms": sweep[best_ns]["x0_hat_rms"],
                             "flow_rt_sweep": sweep,
                             "flow_trust": fm.trust}
            recon = _decode(oc.cpu().numpy().reshape(-1))

        else:
            raise ValueError(f"unknown arm {arm!r} (expected one of {ARMS})")

        m = stack_metrics(recon, target)
        m.update(extra_metrics)
        write_stack_to_model(recon, model, arch, st)
        res = compute_perplexity(model, loader, device)
        # Benchmarks go here -- after write-back, before restore -- so every arm
        # added later inherits them without touching this code. The measurement is
        # keyed only on "the model currently holds a reconstruction".
        arm_bench = None
        if bench_examples:
            arm_bench = bench_deltas(
                base_bench,
                measure_benchmarks(model, tok, bench_examples, device,
                                   bench_max_length or get_max_context_length(model)))
        restore()

        delta = res["perplexity"] - base["perplexity"]
        pct = 100.0 * delta / max(base["perplexity"], 1e-9)
        out["arms"][arm] = {**m, "ppl": res["perplexity"], "ppl_delta": delta,
                            "ppl_delta_pct": pct, "ce": res["ce_loss"]}
        # Print the dispersion ratio INLINE for generative arms, with a COLLAPSED
        # tag. A low dPPL next to a low ratio is the misstep-19 trap, and it has to
        # be visible in the log where the number is read, not only in the JSON.
        disp = ""
        ratio = m.get("code_rms_ratio")
        if ratio is not None:
            tag = ("COLLAPSED" if ratio < 0.5 else
                   "shrunken" if ratio < 0.8 else
                   "over-disp" if ratio > 1.25 else "ok")
            disp = f"  rms={ratio:.2f}x[{tag}]"
        print(f"  [{arch}] {arm:9s} k={k:<4d} cos={m['cosine_sim']:.6f}  "
              f"relL2={m['rel_l2']:.4g}  PPL={res['perplexity']:12.4f}  "
              f"Δ={pct:+.3f}%{disp}")
        _wblog = {f"stack/{arch}/k{k}/{arm}/cosine_sim": m["cosine_sim"],
                  f"stack/{arch}/k{k}/{arm}/ppl_delta_pct": pct,
                  f"stack/{arch}/k{k}/{arm}/ppl": res["perplexity"]}
        if ratio is not None:
            _wblog[f"stack/{arch}/k{k}/{arm}/code_rms_ratio"] = ratio
        wb.log(_wblog)

        if arm_bench:
            out["arms"][arm]["bench"] = arm_bench
            parts = "   ".join(f"{bn} {bv['acc_delta']:+.4f}"
                               for bn, bv in arm_bench.items())
            print(f"  [{arch}]           {'':14s}└ bench Δacc  {parts}")
            wb.log({f"stack/{arch}/k{k}/{arm}/{bn}/{key}": bv[key]
                    for bn, bv in arm_bench.items()
                    for key in ("acc", "acc_norm", "acc_delta", "acc_norm_delta")})

    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return out


# ---------------------------------------------------------------------------

def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--artifact_dir", default="/scratch/biggs.s/llm_vae")
    p.add_argument("--run_name", default="perfam")
    p.add_argument("--arch_list", nargs="+", default=None)
    p.add_argument("--k", nargs="+", type=int, default=None,
                   help="Ranks to evaluate. Default: the fitted maximum and half it.")
    p.add_argument("--arms", nargs="+", default=["pca_only", "vae"],
                   choices=list(ARMS),
                   help="gauss_codes is the null model the flow arms must beat: on "
                        "a manufactured noise ensemble the per-family code "
                        "distribution is near-Gaussian by construction, so "
                        "flow_codes vs gauss_codes is the comparison that carries "
                        "information — not flow_codes vs pca_only.")
    p.add_argument("--mode", choices=["tiny", "full"], default="full")
    p.add_argument("--eval_seq_len", type=int, default=1024)
    p.add_argument("--eval_n_sequences", type=int, default=64)
    p.add_argument("--guidance_scale", type=float, default=1.0,
                   help="CFG on the VAE DECODER. Distinct from "
                        "--flow_guidance_scale, which acts on the velocity field. "
                        "flow_latent composes both; they are different mechanisms "
                        "and multiplying them together would mean nothing.")
    p.add_argument("--flow_steps", type=int, default=None,
                   help="Euler steps for the flow arms. Default: the value sealed "
                        "in flow_meta.json.")
    p.add_argument("--flow_guidance_scale", type=float, default=None,
                   help="CFG on the VELOCITY FIELD. Default: the sealed value.")
    p.add_argument("--flow_rt_steps", nargs="+", type=int, default=None,
                   help="Sweep n_steps in the flow_rt_* arms. Euler discretisation "
                        "error falls like 1/n_steps; the model failing to be a "
                        "consistent vector field does not, so the sweep is the only "
                        "way to tell the two apart.")
    p.add_argument("--allow_legacy_vae", action="store_true",
                   help="Adopt a pre-provenance vae_k*/ directory (vae_config.json "
                        "+ vae_best.pt). Nothing binds those weights to an ensemble "
                        "or to code statistics, so every metric derived from them "
                        "is marked trust=unverified-legacy and footnoted in the "
                        "report. A full re-train is ~10 minutes on one V100.")
    p.add_argument("--sample_idx", type=int, default=0,
                   help="Which ensemble member to reconstruct. 0 is the real "
                        "pretrained model, but it sits ~sqrt(N) nearer the ensemble "
                        "mean than a typical member, so a rank sweep on sample 0 "
                        "understates truncation loss. Use a nonzero index for a "
                        "meaningful k sweep.")
    p.add_argument("--bench", nargs="*", default=None,
                   choices=list(ALL_BENCHMARKS) + ["synthetic"],
                   help="Measure MC benchmark accuracy alongside ΔPPL, on the SAME "
                        "in-memory reconstructed model. Omit for none; bare --bench "
                        "means all three real benchmarks. This is the expensive "
                        "axis: 3 benchmarks x 200 questions x 4 choices is ~2,400 "
                        "forward passes PER ARM PER ARCH PER RANK.")
    p.add_argument("--bench_n_questions", type=int, default=200)
    p.add_argument("--bench_seed", type=int, default=0)
    p.add_argument("--bench_max_length", type=int, default=None,
                   help="Override the per-model context cap. Default: read from the "
                        "model config via eval_core.get_max_context_length.")
    p.add_argument("--dispersion_n", type=int, default=64,
                   help="Draws used to estimate each generative arm's code "
                        "dispersion (misstep 19). Costs microseconds -- the "
                        "expensive inverse_transform still runs once per arm. Set 0 "
                        "to skip, but then a collapsed arm's dPPL is unreadable.")
    p.add_argument("--out_suffix", default=None,
                   help="Extra tag in the results filename, after the _s<idx> part. "
                        "Use it to keep two eval jobs over the same run and target "
                        "from overwriting each other -- e.g. --out_suffix bench for "
                        "the benchmark job. report_stack.py picks up any suffix.")
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--no_wandb", action="store_true")
    args = p.parse_args()

    # Bare --bench means all three; --bench mmlu means just MMLU; no flag means off.
    bench_list = list(ALL_BENCHMARKS) if args.bench == [] else (args.bench or [])

    run_root = os.path.join(args.artifact_dir, "runs", args.run_name)
    ens_meta = os.path.join(run_root, "ensemble", "ensemble_meta.json")
    if not os.path.exists(ens_meta):
        raise SystemExit(f"No ensemble at {ens_meta}. Run train_stack.py first.")
    with open(ens_meta) as f:
        em = json.load(f)

    arch_list = args.arch_list or em["arch_list"]
    hf_cache = os.environ.get("HF_HOME", os.path.join(args.artifact_dir, "hf_cache"))

    # The fitted rank comes from the bases on disk, so it has to be read before the
    # rank list can be defaulted. load_run rebuilds the ensemble from the RECORDED
    # parameters, which is what guarantees inverse_transform streams the same
    # ensemble the PCA was fit on.
    fitted = min(
        (read_json(os.path.join(run_root, "pca", a, "gram_pca_meta.json")) or {}
         ).get("n_components", 0)
        for a in arch_list)
    if fitted <= 0:
        raise SystemExit(f"No fitted PCA under {run_root}/pca/. Run train_stack.py.")
    ks = args.k or sorted({fitted, max(2, fitted // 2)}, reverse=True)
    for k in ks:
        if k > fitted:
            raise SystemExit(f"--k {k} exceeds the fitted rank ({fitted}).")

    real_bench = [b for b in bench_list if b != "synthetic"]
    if real_bench and em["mode"] == "tiny":
        raise SystemExit(
            f"Refusing --bench {' '.join(real_bench)} against a tiny-mode ensemble. "
            f"Tiny models are random-init with a mismatched tokenizer, so every "
            f"accuracy sits at chance and the delta is noise dressed as a "
            f"measurement. Use --bench synthetic to exercise the benchmark plumbing "
            f"in a smoke test, or drop --bench.")
    if "gpqa" in bench_list and not os.environ.get("HF_TOKEN"):
        raise SystemExit(
            "Refusing --bench gpqa without HF_TOKEN. Idavidrein/gpqa is a gated "
            "dataset; without a token the job dies in a datasets traceback partway "
            "through instead of now. Export HF_TOKEN before sbatch, or drop gpqa "
            "from --bench.")

    wb.init_run(job_type="eval_stack",
                config={**vars(args), "ks": ks, "fitted_k": fitted,
                        "bench_list": bench_list},
                tags=["stack", args.run_name] + arch_list,
                enabled=not args.no_wandb, artifact_dir=args.artifact_dir,
                name_suffix=f"{args.run_name}_s{args.sample_idx}")

    print("=" * 74)
    print("Whole-stack evaluation")
    print(f"  run    : {args.run_name}")
    print(f"  archs  : {arch_list}")
    print(f"  N      : {em['n_samples']}  (rank bound {em['n_samples'] - 1}, "
          f"fitted {fitted})")
    print(f"  ranks  : {ks}")
    print(f"  arms   : {args.arms}")
    print(f"  bench  : {bench_list or 'off'}"
          + (f"  ({args.bench_n_questions} questions each)" if bench_list else ""))
    print("=" * 74)

    bench_examples = None
    if bench_list:
        print("\nloading benchmark questions …")
        bench_examples = load_bench_examples(bench_list, args.bench_n_questions,
                                             hf_cache, args.bench_seed)
    # Keyed (arch, sample_idx); the baseline is rank-independent, so this saves one
    # full benchmark pass per arch per extra rank.
    bench_baseline_cache: Dict[tuple, dict] = {}

    want = wants_for(args.arms)
    spaces = flow_spaces_for(args.arms)

    results: Dict[str, dict] = {}
    for k in ks:
        print(f"\n{'-' * 74}\nloading run at k={k}  (want={list(want)})")
        bundle = load_run(run_root, k, arch_list=arch_list, want=want,
                          flow_spaces=spaces,
                          allow_legacy_vae=args.allow_legacy_vae)
        if bundle.trust != "verified":
            print(f"  TRUST={bundle.trust} — every metric at k={k} inherits this "
                  f"label and is footnoted in the report.")

        for arch in arch_list:
            print(f"\n=== {arch}  (k={k}) ===")
            results[f"{arch}@k{k}"] = evaluate_arch(
                arch, bundle, k, args.arms,
                args.eval_seq_len, args.eval_n_sequences, hf_cache,
                em["mode"], args.guidance_scale, args.seed,
                sample_idx=args.sample_idx,
                bench_examples=bench_examples,
                bench_baseline_cache=bench_baseline_cache,
                bench_max_length=args.bench_max_length,
                flow_steps=args.flow_steps,
                flow_guidance_scale=args.flow_guidance_scale,
                flow_rt_steps=args.flow_rt_steps,
                dispersion_n=args.dispersion_n)
        del bundle
        gc.collect()

    res_dir = os.path.join(run_root, "results")
    os.makedirs(res_dir, exist_ok=True)
    # sample_idx and out_suffix are BOTH in the filename because the two axes are
    # independent: the all-arm dPPL job and the benchmark job are deliberately
    # separate sbatch jobs (see slurm_stack_bench.sh's header on why), and at
    # sample_idx 0 they would otherwise write the same path and clobber each other.
    # report_stack.py globs stack_eval_results*.json and renders each as its own
    # section, so any suffix slots in without a code change.
    suffix = "" if args.sample_idx == 0 else f"_s{args.sample_idx}"
    if args.out_suffix:
        suffix += "_" + args.out_suffix.lstrip("_")
    out_path = os.path.join(res_dir, f"stack_eval_results{suffix}.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)

    print("\n" + "=" * 74)
    print(f"{'arch@k':26s} {'arm':10s} {'cosine':>10s} {'PPL orig':>10s} "
          f"{'PPL new':>13s} {'Δ%':>12s}")
    for key, r in results.items():
        for arm, a in r["arms"].items():
            print(f"{key:26s} {arm:10s} {a['cosine_sim']:10.6f} "
                  f"{r['original_ppl']:10.3f} {a['ppl']:13.3f} "
                  f"{a['ppl_delta_pct']:+12.3f}")
            if "bench" in a:
                parts = "   ".join(f"{bn} {bv['acc_delta']:+.4f}"
                                   for bn, bv in a["bench"].items())
                print(f"{'':26s} {'└ Δacc':10s} {parts}")
    print(f"\nSaved → {out_path}")
    wb.finish()


if __name__ == "__main__":
    main()
