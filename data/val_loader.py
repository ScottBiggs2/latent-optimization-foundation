"""
Validation data loader for language-model perplexity evaluation.

Default: WikiText-2 test split (~282K tokens).
The dataset is downloaded once by HuggingFace `datasets` and cached in
HF_HOME (defaults to ~/.cache/huggingface on a laptop; /scratch/biggs.s/hf_cache
on HPC via the HF_HOME env var).

Usage
-----
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B")
    loader = get_wikitext2_loader(tokenizer, seq_len=2048, n_sequences=64)
    for batch in loader:
        input_ids = batch["input_ids"]   # (batch_size, seq_len)
"""

from __future__ import annotations

from typing import Optional

import torch
from torch.utils.data import DataLoader, Dataset


class TokenChunkDataset(Dataset):
    """Fixed-length non-overlapping chunks cut from a flat token sequence."""

    def __init__(self, token_ids: list[int], seq_len: int, n_sequences: int):
        # Trim to exactly n_sequences complete chunks
        max_tokens = n_sequences * seq_len
        token_ids = token_ids[:max_tokens]
        self.chunks = [
            token_ids[i: i + seq_len]
            for i in range(0, len(token_ids), seq_len)
            if i + seq_len <= len(token_ids)
        ]

    def __len__(self):
        return len(self.chunks)

    def __getitem__(self, idx):
        return {"input_ids": torch.tensor(self.chunks[idx], dtype=torch.long)}


def get_wikitext2_loader(
    tokenizer,
    seq_len: int = 2048,
    n_sequences: int = 64,
    split: str = "test",
    batch_size: int = 1,
    cache_dir: Optional[str] = None,
) -> DataLoader:
    """
    Build a DataLoader over WikiText-2.

    Parameters
    ----------
    tokenizer:
        Any HuggingFace tokenizer with an ``encode`` method.
    seq_len:
        Token sequence length per sample.
    n_sequences:
        Number of fixed-length sequences to use (caps the dataset size).
        64 × 2048 = 131K tokens → ~2 min PPL eval on A100.
    split:
        HuggingFace split name ("test", "validation", "train").
    batch_size:
        DataLoader batch size (default 1 — safe for any model size).
    cache_dir:
        Override for HuggingFace dataset cache directory.

    Returns
    -------
    DataLoader yielding dicts with key "input_ids" of shape (batch, seq_len).
    """
    try:
        from datasets import load_dataset
    except ImportError:
        raise ImportError(
            "The 'datasets' package is required for LM evaluation.\n"
            "Install with: pip install datasets"
        )

    # datasets 3.x requires namespace/name format; "wikitext" alone is no longer valid.
    ds = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1",
                      split=split, cache_dir=cache_dir)

    # Concatenate all text, then tokenise in one shot for clean chunking
    raw_text = "\n\n".join(
        t for t in ds["text"] if t.strip()  # type: ignore[index]
    )
    token_ids = tokenizer.encode(raw_text)
    print(f"WikiText-2 ({split}): {len(token_ids):,} tokens total")

    dataset = TokenChunkDataset(token_ids, seq_len, n_sequences)
    print(f"  Using {len(dataset)} sequences × {seq_len} tokens = "
          f"{len(dataset) * seq_len:,} tokens for PPL eval")

    return DataLoader(dataset, batch_size=batch_size, shuffle=False, pin_memory=True)


class SyntheticTokenDataset(Dataset):
    """Random integer tokens in [0, vocab_size) — for tiny-model smoke tests."""

    def __init__(self, vocab_size: int, seq_len: int, n_sequences: int, seed: int = 0):
        rng = torch.Generator()
        rng.manual_seed(seed)
        self.data = torch.randint(0, vocab_size, (n_sequences, seq_len), generator=rng)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return {"input_ids": self.data[idx]}


def get_synthetic_loader(
    vocab_size: int,
    seq_len: int = 512,
    n_sequences: int = 16,
    batch_size: int = 1,
    seed: int = 0,
) -> DataLoader:
    """
    Synthetic DataLoader for tiny-model PPL smoke tests.

    Uses random token IDs in [0, vocab_size) — no real text, no tokenizer download.
    Validates the eval plumbing (before/after PPL should be nearly identical for a
    perfect reconstruction roundtrip) without requiring the real model's tokenizer.

    Parameters
    ----------
    vocab_size : must match the model being evaluated (e.g. tiny_config["vocab_size"])
    seq_len    : token sequence length per sample
    n_sequences: number of sequences (16 × 512 = 8K tokens; fast on CPU)
    """
    dataset = SyntheticTokenDataset(vocab_size, seq_len, n_sequences, seed=seed)
    print(f"Synthetic loader: {n_sequences} sequences × {seq_len} tokens "
          f"= {n_sequences * seq_len:,} tokens  (vocab_size={vocab_size})")
    return DataLoader(dataset, batch_size=batch_size, shuffle=False)
