"""
Calibrate the ensemble augmentation noise scale.

The ensemble is w_i = w_0 + s * sigma_w * eps_i, so `s` sets everything downstream:

  * too small and the members are indistinguishable at float32 precision, the Gram
    matrix is roundoff-dominated, and the rank floor throws the basis away. The old
    default `--noise_scale 1e-7` is squarely in this regime -- 1e-7 IS float32's
    relative precision, so the perturbation is at the representable limit.
  * too large and the members are broken models, so "reconstruct the ensemble
    faithfully" stops meaning anything about language modelling.

This script measures the usable band directly: perturb the WHOLE stack at each
candidate `s` and report WikiText-2 perplexity. Pick the largest `s` whose PPL delta
is still small.

The stack must be the same object EnsembleDataset builds, or the calibration does
not transfer. With `--include_extra` (the default) that means embeddings, the final
norm and the untied LM head are perturbed too, and `weight_std` is the std over all
of it. Expect a given `s` to cost MORE perplexity than a decoder-blocks-only sweep
says: embeddings sit directly on the logit path, and where the LM head is untied
(pythia) a quarter of the stack is logit-facing. Re-run this whenever the layout
changes.

    python scripts/calibrate_noise.py --arch_list gpt2_medium smollm2_360m pythia_410m \
        --scales 1e-4 1e-3 3e-3 1e-2 3e-2
"""

from __future__ import annotations

import argparse
import gc
import json
import os
from typing import List

import numpy as np
import torch

from llmzoo.models.registry import get_arch_config, load_model
from llmzoo.models.weight_extractor import build_stack_spec, write_stack_to_model


def calibrate_arch(arch: str, scales: List[float], seq_len: int, n_sequences: int,
                   exclude_1d: bool, include_extra: bool,
                   hf_cache: str, seed: int) -> dict:
    from llmzoo.eval.core import compute_perplexity
    from llmzoo.data.val_loader import get_wikitext2_loader
    from transformers import AutoTokenizer

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg = get_arch_config(arch)

    model = load_model(arch, cache_dir=hf_cache).to(device)
    model.eval()
    tok = AutoTokenizer.from_pretrained(cfg["default_model_id"], cache_dir=hf_cache,
                                        trust_remote_code=True)
    loader = get_wikitext2_loader(tok, seq_len=seq_len, n_sequences=n_sequences,
                                  cache_dir=hf_cache)

    w0, spec = build_stack_spec(model, arch, exclude_1d=exclude_1d,
                                include_extra=include_extra)
    weight_std = spec.weight_std
    D = spec.n_params
    w0_norm = float(np.linalg.norm(w0))

    print(f"  [{arch}] L={spec.n_layers}  D={D:,}  "
          f"extra={spec.extra_size:,} "
          f"({100.0 * spec.extra_size / max(D, 1):.1f}%)  "
          f"weight_std={weight_std:.6g}")
    base = compute_perplexity(model, loader, device)
    print(f"  [{arch}] original PPL = {base['perplexity']:.4f}")

    rows = []
    rng = np.random.default_rng(seed)
    # One noise draw reused across scales, so the sweep varies only `s`. A fresh
    # draw per scale would fold sampling variation into the trend.
    eps = rng.standard_normal(D).astype(np.float32)

    for s in scales:
        write_stack_to_model((w0 + s * weight_std * eps).astype(np.float32),
                             model, arch, spec)
        res = compute_perplexity(model, loader, device)
        delta = res["perplexity"] - base["perplexity"]
        pct = 100.0 * delta / max(base["perplexity"], 1e-9)
        # Relative L2 of the perturbation, so the number is comparable across archs
        # regardless of their weight distribution.
        rel_l2 = s * weight_std * np.sqrt(D) / (w0_norm + 1e-30)
        rows.append({"noise_scale": s, "ppl": res["perplexity"],
                     "ppl_delta": delta, "ppl_delta_pct": pct,
                     "rel_l2_perturbation": float(rel_l2)})
        print(f"  [{arch}] s={s:<8g} PPL={res['perplexity']:12.4f}  "
              f"Δ={delta:+10.4f} ({pct:+8.3f}%)   ||δw||/||w||={rel_l2:.4g}")
        write_stack_to_model(w0, model, arch, spec)

    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return {"arch": arch, "n_layers": spec.n_layers, "n_params": D,
            "extra_size": spec.extra_size, "include_extra": include_extra,
            "exclude_1d": exclude_1d,
            "weight_std": weight_std, "original_ppl": base["perplexity"],
            "sweep": rows}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--arch_list", nargs="+",
                   default=["gpt2_medium", "smollm2_360m", "pythia_410m"])
    p.add_argument("--scales", nargs="+", type=float,
                   default=[1e-4, 1e-3, 3e-3, 1e-2, 3e-2])
    p.add_argument("--eval_seq_len", type=int, default=1024)
    p.add_argument("--eval_n_sequences", type=int, default=64)
    p.add_argument("--exclude_1d", action="store_true", default=True)
    p.add_argument("--no_exclude_1d", dest="exclude_1d", action="store_false")
    p.add_argument("--include_extra", action="store_true", default=True,
                   help="Perturb embeddings / final norm / untied LM head too. On "
                        "by default, matching EnsembleDataset.")
    p.add_argument("--no_include_extra", dest="include_extra",
                   action="store_false")
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--artifact_dir", default="/scratch/biggs.s/llm_vae")
    p.add_argument("--out", default=None)
    args = p.parse_args()

    hf_cache = os.environ.get("HF_HOME", os.path.join(args.artifact_dir, "hf_cache"))
    scales = sorted(args.scales)

    print("=" * 70)
    print("Ensemble noise-scale calibration")
    print(f"  archs      : {args.arch_list}")
    print(f"  scales     : {scales}")
    print(f"  exclude_1d : {args.exclude_1d}")
    print(f"  include_extra : {args.include_extra}")
    print(f"  eval       : WikiText-2, {args.eval_n_sequences} x {args.eval_seq_len}")
    print("=" * 70)

    results = {}
    for arch in args.arch_list:
        print(f"\n=== {arch} ===")
        results[arch] = calibrate_arch(arch, scales, args.eval_seq_len,
                                       args.eval_n_sequences, args.exclude_1d,
                                       args.include_extra,
                                       hf_cache, args.seed)

    print("\n" + "=" * 70)
    print("SUMMARY — PPL delta (%) by noise scale")
    print(f"{'arch':16s} {'orig PPL':>10s} " +
          " ".join(f"{s:>11g}" for s in scales))
    for arch, r in results.items():
        cells = " ".join(f"{row['ppl_delta_pct']:>+11.3f}" for row in r["sweep"])
        print(f"{arch:16s} {r['original_ppl']:>10.3f} {cells}")

    # A concrete recommendation, so the choice is not left to eyeballing.
    print("\nLargest scale keeping every family within a given PPL budget:")
    for budget in (0.5, 1.0, 5.0, 20.0):
        ok = [s for i, s in enumerate(scales)
              if all(r["sweep"][i]["ppl_delta_pct"] <= budget for r in results.values())]
        print(f"  <= {budget:5.1f}% : "
              f"{max(ok) if ok else 'none of the tested scales'}")

    print("\nNOTE the competing constraint: `s` must also be large enough that the "
          "ensemble members separate above float32 precision, or the Gram matrix is "
          "roundoff and the rank floor discards the basis. Anything at or below "
          "~1e-6 is unusable regardless of what PPL says here.")

    out = args.out or os.path.join(args.artifact_dir, "results",
                                    "noise_calibration.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w") as f:
        json.dump({"scales": scales, "exclude_1d": args.exclude_1d,
                   "include_extra": args.include_extra,
                   "seq_len": args.eval_seq_len,
                   "n_sequences": args.eval_n_sequences,
                   "results": results}, f, indent=2)
    print(f"\nSaved → {out}")


if __name__ == "__main__":
    main()
