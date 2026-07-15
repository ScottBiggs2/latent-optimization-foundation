"""
Multiple-choice benchmark evaluation for reconstructed models.

For each model family:
  1. Load the pretrained model
  2. Measure MMLU / HellaSwag / GPQA accuracy on the original weights
  3. Reconstruct each transformer block through PCA + VAE
  4. Re-measure accuracy on the reconstructed weights
  5. Compare original vs. reconstructed

All three benchmarks are scored with the same log-likelihood ranking
primitive (score_choices): no chat templates, since all registered
architectures are base pretrained models. See data/mc_loader.py for the
per-benchmark prompt construction.
"""

from __future__ import annotations

import gc
from typing import Iterable, Optional, Tuple

import torch
import torch.nn as nn

from data.block_dataset import BlockDataset
from data.mc_loader import LOADERS, MCExample
from dual_pca import BatchedCovariancePCA
from eval_lm import reconstruct_model_blocks
from models.registry import get_arch_config, load_model
from vae import ConditionedBlockVAE


# ---------------------------------------------------------------------------
# Log-likelihood choice scoring
# ---------------------------------------------------------------------------

def score_choices(
    model: nn.Module,
    tokenizer,
    context: str,
    choices: list[str],
    device: torch.device,
) -> Tuple[list[float], list[float]]:
    """
    Score each choice as a continuation of context by log-likelihood.

    context and context+choice are encoded jointly and the continuation
    token ids are recovered by slicing off the context-length prefix —
    this avoids BPE boundary mismatches from encoding pieces separately
    (the standard lm-eval-harness trick).

    Returns
    -------
    (sum_logprobs, mean_logprobs) — one value per choice. sum_logprobs
    gives raw accuracy; mean_logprobs (length-normalized) gives acc_norm.
    """
    context_ids = tokenizer.encode(context)
    context_len = len(context_ids)

    sum_logprobs: list[float] = []
    mean_logprobs: list[float] = []

    for choice in choices:
        full_ids = tokenizer.encode(context + choice)
        continuation_ids = full_ids[context_len:]
        if len(continuation_ids) == 0:
            sum_logprobs.append(float("-inf"))
            mean_logprobs.append(float("-inf"))
            continue

        input_ids = torch.tensor(full_ids, dtype=torch.long, device=device).unsqueeze(0)
        with torch.no_grad():
            logits = model(input_ids=input_ids).logits  # (1, T, V)

        log_probs = torch.log_softmax(logits[:, :-1, :].float(), dim=-1)  # predicts tokens[1:]
        target_ids = input_ids[:, 1:]  # (1, T-1)
        token_logprobs = log_probs.gather(-1, target_ids.unsqueeze(-1)).squeeze(-1).squeeze(0)  # (T-1,)

        cont_len = len(continuation_ids)
        cont_logprobs = token_logprobs[-cont_len:]
        total = float(cont_logprobs.sum().item())

        sum_logprobs.append(total)
        mean_logprobs.append(total / cont_len)

    return sum_logprobs, mean_logprobs


def compute_mc_accuracy(
    model: nn.Module,
    tokenizer,
    examples: Iterable[MCExample],
    device: torch.device,
) -> dict:
    """
    Score a set of MCExamples and return acc / acc_norm.

    acc      : argmax by raw summed log-likelihood
    acc_norm : argmax by length-normalized log-likelihood
    """
    n = 0
    n_correct = 0
    n_correct_norm = 0

    for ex in examples:
        sum_lp, mean_lp = score_choices(model, tokenizer, ex.context, ex.choices, device)
        pred = max(range(len(sum_lp)), key=lambda i: sum_lp[i])
        pred_norm = max(range(len(mean_lp)), key=lambda i: mean_lp[i])
        n += 1
        if pred == ex.gold_idx:
            n_correct += 1
        if pred_norm == ex.gold_idx:
            n_correct_norm += 1

    if n == 0:
        return {"acc": float("nan"), "acc_norm": float("nan"), "n_examples": 0}

    return {"acc": n_correct / n, "acc_norm": n_correct_norm / n, "n_examples": n}


# ---------------------------------------------------------------------------
# Per-family evaluation
# ---------------------------------------------------------------------------

def evaluate_family_mc(
    arch: str,
    pca: BatchedCovariancePCA,
    vae: ConditionedBlockVAE,
    max_block_size: int,
    benchmarks: tuple[str, ...] = ("mmlu", "hellaswag", "gpqa"),
    n_questions: int = 200,
    hf_cache: Optional[str] = None,
) -> dict:
    """
    Full before/after multiple-choice accuracy evaluation for one family.

    Returns a dict keyed by benchmark name, each holding original/
    reconstructed acc and acc_norm plus their deltas.
    """
    device = torch.device(
        next(vae.parameters()).device if hasattr(vae, "block_idx_emb") else "cpu"
    )

    model = load_model(arch, cache_dir=hf_cache)
    model = model.to(device)
    model.eval()

    cfg = get_arch_config(arch)
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        cfg["default_model_id"],
        cache_dir=hf_cache,
        trust_remote_code=True,
    )

    examples_by_bench: dict[str, list[MCExample]] = {}
    for bench in benchmarks:
        print(f"  [{arch}] Loading {bench} ({n_questions} questions) …")
        examples_by_bench[bench] = LOADERS[bench](n_questions=n_questions, cache_dir=hf_cache)

    original: dict[str, dict] = {}
    for bench, examples in examples_by_bench.items():
        print(f"  [{arch}] Measuring original {bench} accuracy …")
        original[bench] = compute_mc_accuracy(model, tokenizer, examples, device)
        print(f"  [{arch}] Original {bench}: acc={original[bench]['acc']:.3f}  "
              f"acc_norm={original[bench]['acc_norm']:.3f}")

    print(f"  [{arch}] Reconstructing blocks …")
    reconstruct_model_blocks(model, arch, pca, vae, max_block_size, device)

    reconstructed: dict[str, dict] = {}
    for bench, examples in examples_by_bench.items():
        print(f"  [{arch}] Measuring reconstructed {bench} accuracy …")
        reconstructed[bench] = compute_mc_accuracy(model, tokenizer, examples, device)
        print(f"  [{arch}] Reconstructed {bench}: acc={reconstructed[bench]['acc']:.3f}  "
              f"acc_norm={reconstructed[bench]['acc_norm']:.3f}")

    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    results = {"arch": arch}
    for bench in benchmarks:
        orig, recon = original[bench], reconstructed[bench]
        results[bench] = {
            "original_acc":          orig["acc"],
            "original_acc_norm":     orig["acc_norm"],
            "reconstructed_acc":     recon["acc"],
            "reconstructed_acc_norm": recon["acc_norm"],
            "acc_delta":             recon["acc"] - orig["acc"],
            "acc_norm_delta":        recon["acc_norm"] - orig["acc_norm"],
            "n_examples":            orig["n_examples"],
        }
    return results


# ---------------------------------------------------------------------------
# All families
# ---------------------------------------------------------------------------

def evaluate_all_families_mc(
    pca: BatchedCovariancePCA,
    vae: ConditionedBlockVAE,
    dataset: BlockDataset,
    benchmarks: tuple[str, ...] = ("mmlu", "hellaswag", "gpqa"),
    n_questions: int = 200,
    artifact_dir: str = "/scratch/biggs.s/llm_vae",
) -> dict:
    """Run MC benchmark eval for every arch in dataset.arch_list."""
    hf_cache = artifact_dir + "/hf_cache"
    results = {}
    for arch in dataset.arch_list:
        results[arch] = evaluate_family_mc(
            arch, pca, vae,
            max_block_size=dataset.max_block_size,
            benchmarks=benchmarks,
            n_questions=n_questions,
            hf_cache=hf_cache,
        )
    return results
