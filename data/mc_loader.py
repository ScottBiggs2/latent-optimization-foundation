"""
Multiple-choice benchmark loaders for downstream LM evaluation.

Each loader returns a list of MCExample — a (context, choices, gold_idx)
triple — built for log-likelihood scoring (see eval_mc.score_choices).
No chat templates are used: all four registered architectures are base
pretrained models, so scoring follows the standard lm-eval-harness-style
protocol of ranking answer continuations by log-likelihood under the model.

  load_mmlu()      — cais/mmlu, 5-shot, letter continuations (" A".." D")
  load_hellaswag() — Rowan/hellaswag, 0-shot, full-sentence continuations
  load_gpqa()      — Idavidrein/gpqa (gated), few-shot, letter continuations
"""

from __future__ import annotations

import random
from typing import List, NamedTuple, Optional


class MCExample(NamedTuple):
    context: str          # prompt text shared by all choices
    choices: List[str]     # continuation strings to be scored
    gold_idx: int          # index into choices of the correct answer


# ---------------------------------------------------------------------------
# MMLU
# ---------------------------------------------------------------------------

_MMLU_LETTERS = ["A", "B", "C", "D"]


def _format_mmlu_question(question: str, options: List[str], answer_letter: Optional[str] = None) -> str:
    lines = [question.strip()]
    for letter, opt in zip(_MMLU_LETTERS, options):
        lines.append(f"{letter}. {opt}")
    lines.append("Answer:")
    text = "\n".join(lines)
    if answer_letter is not None:
        text += f" {answer_letter}\n\n"
    return text


def load_mmlu(
    n_questions: int = 200,
    n_shots: int = 5,
    seed: int = 0,
    cache_dir: Optional[str] = None,
) -> List[MCExample]:
    """
    Standard MMLU: 5-shot letter-probability scoring.

    Choices are always [" A", " B", " C", " D"] scored as continuations
    after "Answer:" — the conventional MMLU log-likelihood protocol.
    """
    from datasets import load_dataset

    test_ds = load_dataset("cais/mmlu", "all", split="test", cache_dir=cache_dir)
    dev_ds = load_dataset("cais/mmlu", "all", split="dev", cache_dir=cache_dir)

    # Group dev exemplars by subject for per-subject few-shot prompts.
    dev_by_subject: dict = {}
    for row in dev_ds:
        dev_by_subject.setdefault(row["subject"], []).append(row)

    rng = random.Random(seed)
    indices = list(range(len(test_ds)))
    rng.shuffle(indices)
    indices = indices[:n_questions]

    letter_choices = [f" {letter}" for letter in _MMLU_LETTERS]

    examples: List[MCExample] = []
    for i in indices:
        row = test_ds[i]
        subject = row["subject"]
        shots = dev_by_subject.get(subject, [])[:n_shots]

        prefix_parts = [f"The following are multiple choice questions (with answers) about {subject.replace('_', ' ')}.\n"]
        for shot in shots:
            shot_letter = _MMLU_LETTERS[shot["answer"]]
            prefix_parts.append(_format_mmlu_question(shot["question"], shot["choices"], shot_letter))
        prefix_parts.append(_format_mmlu_question(row["question"], row["choices"]))
        context = "\n".join(prefix_parts)

        examples.append(MCExample(context=context, choices=letter_choices, gold_idx=int(row["answer"])))

    return examples


# ---------------------------------------------------------------------------
# HellaSwag
# ---------------------------------------------------------------------------

def load_hellaswag(
    n_questions: int = 200,
    seed: int = 0,
    cache_dir: Optional[str] = None,
) -> List[MCExample]:
    """
    HellaSwag: 0-shot, full-sentence continuation scoring.

    Test split has no released labels, so validation is used (standard
    practice). Choices are the 4 full candidate endings — variable length,
    which is why acc_norm (length-normalized log-likelihood) is the
    conventional primary metric for this benchmark.
    """
    from datasets import load_dataset

    ds = load_dataset("Rowan/hellaswag", split="validation", cache_dir=cache_dir)

    rng = random.Random(seed)
    indices = list(range(len(ds)))
    rng.shuffle(indices)
    indices = indices[:n_questions]

    examples: List[MCExample] = []
    for i in indices:
        row = ds[i]
        context = row["ctx"]
        choices = list(row["endings"])
        gold_idx = int(row["label"])
        examples.append(MCExample(context=context, choices=choices, gold_idx=gold_idx))

    return examples


# ---------------------------------------------------------------------------
# GPQA
# ---------------------------------------------------------------------------

def load_gpqa(
    n_questions: int = 200,
    n_shots: int = 3,
    seed: int = 0,
    cache_dir: Optional[str] = None,
) -> List[MCExample]:
    """
    GPQA (gated dataset — requires HF_TOKEN + accepted terms at
    huggingface.co/datasets/Idavidrein/gpqa, same as gemma3_270m).

    Only split is "train" (448 questions total). A few questions are held
    out as few-shot exemplars; options are shuffled per-question with a
    fixed seed to avoid position bias (standard GPQA practice), then scored
    the same letter-continuation way as MMLU.
    """
    from datasets import load_dataset

    ds = load_dataset("Idavidrein/gpqa", "gpqa_main", split="train", cache_dir=cache_dir)

    rng = random.Random(seed)
    indices = list(range(len(ds)))
    rng.shuffle(indices)

    shot_indices = indices[:n_shots]
    eval_indices = indices[n_shots: n_shots + n_questions]

    def _shuffled_options(row, shuffle_rng):
        options = [
            row["Correct Answer"],
            row["Incorrect Answer 1"],
            row["Incorrect Answer 2"],
            row["Incorrect Answer 3"],
        ]
        order = list(range(4))
        shuffle_rng.shuffle(order)
        shuffled = [options[j] for j in order]
        gold_idx = order.index(0)
        return shuffled, gold_idx

    shots = []
    for i in shot_indices:
        row = ds[i]
        options, gold_idx = _shuffled_options(row, random.Random(seed + 1 + i))
        shots.append((row["Question"], options, gold_idx))

    examples: List[MCExample] = []
    for i in eval_indices:
        row = ds[i]
        options, gold_idx = _shuffled_options(row, random.Random(seed + 1 + i))

        prefix_parts = ["The following are multiple choice questions (with answers).\n"]
        for shot_question, shot_options, shot_gold in shots:
            shot_letter = _MMLU_LETTERS[shot_gold]
            prefix_parts.append(_format_mmlu_question(shot_question, shot_options, shot_letter))
        prefix_parts.append(_format_mmlu_question(row["Question"], options))
        context = "\n".join(prefix_parts)

        letter_choices = [f" {letter}" for letter in _MMLU_LETTERS]
        examples.append(MCExample(context=context, choices=letter_choices, gold_idx=gold_idx))

    return examples


LOADERS = {
    "mmlu": load_mmlu,
    "hellaswag": load_hellaswag,
    "gpqa": load_gpqa,
}
