"""
Model-scoring primitives shared by every evaluation path.

Four functions, all of which take an already-constructed `nn.Module` and measure it.
They know nothing about PCA, VAEs, flows, ensembles or artifacts -- that separation is
the point. `eval_stack.py` and `calibrate_noise.py` both need them, and before this
module existed they reached into the block-era `eval_lm.py` / `eval_mc.py`, which
dragged the whole retired block pipeline (dual_pca, BlockDataset, ConditionedBlockVAE)
into every evaluation run.

  compute_perplexity      cross-entropy / PPL over a DataLoader
  get_max_context_length   read a model's context window out of its config
  score_choices            log-likelihood ranking primitive for multiple choice
  compute_mc_accuracy      accuracy over MCExamples, via score_choices

All three benchmarks (MMLU / HellaSwag / GPQA) are scored with the same
log-likelihood ranking primitive: no chat templates, since every registered
architecture is a base pretrained model. See data/mc_loader.py for the
per-benchmark prompt construction.
"""

from __future__ import annotations

import math
from typing import Iterable, Tuple

import torch
import torch.nn as nn

from data.mc_loader import MCExample


# ---------------------------------------------------------------------------
# Perplexity
# ---------------------------------------------------------------------------

def compute_perplexity(
    model: nn.Module,
    dataloader,
    device: torch.device,
) -> dict:
    """
    Measure cross-entropy and perplexity over a DataLoader.

    Returns
    -------
    dict with keys: perplexity, ce_loss, n_tokens
    """
    model.eval()
    total_loss = 0.0
    total_tokens = 0

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            # Shift labels: standard causal-LM cross-entropy
            labels = input_ids.clone()
            outputs = model(input_ids=input_ids, labels=labels)
            loss = outputs.loss
            if torch.isnan(loss) or torch.isinf(loss):
                continue
            # Number of predicted tokens: seq_len - 1 per sequence
            n_tok = (input_ids.shape[0] * (input_ids.shape[1] - 1))
            total_loss   += loss.item() * n_tok
            total_tokens += n_tok

    if total_tokens == 0:
        return {"perplexity": float("nan"), "ce_loss": float("nan"), "n_tokens": 0}

    ce = total_loss / total_tokens
    # Cap at e^20 to avoid overflow on degenerate reconstructions
    ppl = math.exp(min(ce, 20.0))
    return {"perplexity": ppl, "ce_loss": ce, "n_tokens": total_tokens}


# ---------------------------------------------------------------------------
# Context length
# ---------------------------------------------------------------------------

def get_max_context_length(model: nn.Module, default: int = 1024) -> int:
    """
    Read the model's max sequence length from its config.

    Different architectures name this differently: GPT-2 uses n_positions/
    n_ctx, Llama/Qwen/OPT-style configs use max_position_embeddings.
    """
    cfg = model.config
    for attr in ("max_position_embeddings", "n_positions", "n_ctx"):
        val = getattr(cfg, attr, None)
        if val:
            return int(val)
    return default


# ---------------------------------------------------------------------------
# Log-likelihood choice scoring
# ---------------------------------------------------------------------------

def score_choices(
    model: nn.Module,
    tokenizer,
    context: str,
    choices: list[str],
    device: torch.device,
    max_length: int = 1024,
) -> Tuple[list[float], list[float]]:
    """
    Score each choice as a continuation of context by log-likelihood.

    context and context+choice are encoded jointly and the continuation
    token ids are recovered by slicing off the context-length prefix —
    this avoids BPE boundary mismatches from encoding pieces separately
    (the standard lm-eval-harness trick).

    Few-shot contexts (MMLU/GPQA) can exceed a small model's max position
    embeddings (e.g. GPT-2-medium's 1024), which crashes the forward pass
    with an out-of-range position index. When the joint encoding would
    exceed max_length, the context is truncated from the left (dropping
    the oldest few-shot examples first) so the full continuation is always
    scored intact.

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

        if len(full_ids) > max_length:
            keep_context = max(0, max_length - len(continuation_ids))
            full_ids = context_ids[-keep_context:] + continuation_ids if keep_context > 0 else continuation_ids[-max_length:]

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


# ---------------------------------------------------------------------------
# Multiple-choice accuracy
# ---------------------------------------------------------------------------

def compute_mc_accuracy(
    model: nn.Module,
    tokenizer,
    examples: Iterable[MCExample],
    device: torch.device,
    max_length: int = 1024,
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
        sum_lp, mean_lp = score_choices(model, tokenizer, ex.context, ex.choices, device, max_length=max_length)
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
