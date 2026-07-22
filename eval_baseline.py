"""
Baseline evaluation on unmodified pretrained HuggingFace models.

No PCA / VAE required — this measures WikiText-2 perplexity and
MMLU / HellaSwag / GPQA accuracy directly on the stock checkpoints, so the
before/after deltas in eval_lm.py / eval_mc.py have a clean, independently
reproducible reference point.

Usage
-----
    python eval_baseline.py
    python eval_baseline.py --arch_list gpt2_medium smollm2_360m
    python eval_baseline.py --skip_mc                 # PPL only
    python eval_baseline.py --skip_ppl --mc_n_questions 200

Writes results/baseline_results.json (per-arch: ppl + per-benchmark acc/acc_norm),
in the same field names eval_lm.py / eval_mc.py use for "original_*", so
report.py can render a single baseline table alongside the reconstruction
tables if desired.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import time

import torch

from eval_lm import compute_perplexity
from eval_mc import compute_mc_accuracy, get_max_context_length
from data.mc_loader import LOADERS
from models.registry import get_arch_config, load_model

_HPC_SCRATCH  = "/scratch/biggs.s"
_HPC_HF_CACHE = os.path.join(_HPC_SCRATCH, "hf_cache")
_HPC_ARTIFACT = os.path.join(_HPC_SCRATCH, "llm_vae")

os.environ.setdefault("HF_HOME", _HPC_HF_CACHE)
os.environ.setdefault("HF_DATASETS_CACHE", _HPC_HF_CACHE)
os.environ.setdefault("TRITON_CACHE_DIR", os.path.join(_HPC_SCRATCH, "triton_cache"))


def ts() -> str:
    return time.strftime("[%H:%M:%S]")


def evaluate_baseline_family(
    arch: str,
    seq_len: int,
    n_sequences: int,
    mc_benchmarks: tuple[str, ...],
    mc_n_questions: int,
    hf_cache: str,
    skip_ppl: bool,
    skip_mc: bool,
) -> dict:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = load_model(arch, cache_dir=hf_cache)
    model = model.to(device)
    model.eval()

    cfg = get_arch_config(arch)
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        cfg["default_model_id"], cache_dir=hf_cache, trust_remote_code=True,
    )

    result: dict = {"arch": arch}

    if not skip_ppl:
        from data.val_loader import get_wikitext2_loader
        print(f"  [{arch}] Loading WikiText-2 …")
        dataloader = get_wikitext2_loader(
            tokenizer, seq_len=seq_len, n_sequences=n_sequences, cache_dir=hf_cache
        )
        print(f"  [{arch}] Measuring PPL …")
        ppl_result = compute_perplexity(model, dataloader, device)
        result["ppl"] = ppl_result["perplexity"]
        result["ce_loss"] = ppl_result["ce_loss"]
        result["n_tokens"] = ppl_result["n_tokens"]
        print(f"  [{arch}] PPL = {result['ppl']:.3f}")

    if not skip_mc:
        max_length = get_max_context_length(model)
        print(f"  [{arch}] Model max context length: {max_length}")
        for bench in mc_benchmarks:
            print(f"  [{arch}] Loading {bench} ({mc_n_questions} questions) …")
            examples = LOADERS[bench](n_questions=mc_n_questions, cache_dir=hf_cache)
            print(f"  [{arch}] Measuring {bench} accuracy …")
            acc = compute_mc_accuracy(model, tokenizer, examples, device, max_length=max_length)
            result[bench] = acc
            print(f"  [{arch}] {bench}: acc={acc['acc']:.3f}  acc_norm={acc['acc_norm']:.3f}")

    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return result


def main():
    p = argparse.ArgumentParser(description="Baseline eval on stock HuggingFace models (no PCA/VAE)")
    # Mirrors run_hpc.py's _DEFAULT_ARCHS (excludes gemma3_270m/opt_350m, which
    # need a HF_TOKEN / have known PCA issues respectively — pass explicitly
    # via --arch_list to include them).
    _DEFAULT_ARCHS = ["gpt2_medium", "smollm2_360m", "qwen3_0_6b",
                       "smollm2_135m", "pythia_160m", "pythia_410m"]
    p.add_argument("--arch_list", nargs="+", default=_DEFAULT_ARCHS)
    p.add_argument("--artifact_dir", type=str, default=_HPC_ARTIFACT)

    p.add_argument("--seq_len", type=int, default=512)
    p.add_argument("--n_sequences", type=int, default=32)

    p.add_argument("--mc_benchmarks", nargs="+", default=["mmlu", "hellaswag", "gpqa"])
    p.add_argument("--mc_n_questions", type=int, default=200)

    p.add_argument("--skip_ppl", action="store_true", default=False)
    p.add_argument("--skip_mc", action="store_true", default=False)

    args = p.parse_args()

    # Share the same cache as run_hpc.py / registry.py (HF_HOME, set above)
    # instead of a separate artifact_dir-local one — otherwise gated-model
    # downloads and access grants split across two disjoint cache dirs.
    hf_cache = os.environ["HF_HOME"]
    res_dir = os.path.join(args.artifact_dir, "results")
    os.makedirs(res_dir, exist_ok=True)

    print(f"{ts()} Baseline eval — arch_list={args.arch_list}")
    print(f"{ts()} skip_ppl={args.skip_ppl}  skip_mc={args.skip_mc}  "
          f"mc_benchmarks={args.mc_benchmarks if not args.skip_mc else '-'}")

    results = {}
    for arch in args.arch_list:
        print(f"\n{ts()} === {arch} ===")
        results[arch] = evaluate_baseline_family(
            arch,
            seq_len=args.seq_len,
            n_sequences=args.n_sequences,
            mc_benchmarks=tuple(args.mc_benchmarks),
            mc_n_questions=args.mc_n_questions,
            hf_cache=hf_cache,
            skip_ppl=args.skip_ppl,
            skip_mc=args.skip_mc,
        )

    out_path = os.path.join(res_dir, "baseline_results.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n{ts()} Saved -> {out_path}")


if __name__ == "__main__":
    main()
