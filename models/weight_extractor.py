"""
Block-wise weight extraction and reconstruction.

The key design choice: we use block.named_parameters() rather than
architecture-specific name lists. This gives a consistent, depth-first
parameter ordering that is stable for a given architecture and requires
no config changes when new architectures are added.

Public API
----------
  extract_block_flat(block)          → (flat_float32, schema)
  compute_max_block_size(archs)      → int
  pad_block(flat, target_size)       → (padded, mask)
  extract_all_blocks(model, arch, max_block_size)
                                     → (blocks, masks, block_idxs, family_idx)
  reconstruct_block(flat_unpadded, block, schema)
  make_block_loader(all_blocks)      → callable(start, end) for PCA
"""

from __future__ import annotations

from typing import Callable, List, NamedTuple, Tuple

import numpy as np
import torch
import torch.nn as nn

from .registry import get_arch_config, get_layers, build_tiny_model


# ---------------------------------------------------------------------------
# Schema type
# ---------------------------------------------------------------------------

class ParamEntry(NamedTuple):
    name: str    # relative dotted path within the block module
    shape: tuple


# ---------------------------------------------------------------------------
# Block flattening
# ---------------------------------------------------------------------------

def extract_block_flat(
    block: nn.Module,
) -> Tuple[np.ndarray, List[ParamEntry]]:
    """
    Flatten all parameters in one transformer decoder block.

    Uses block.named_parameters() which returns parameters in PyTorch's
    canonical registration order (depth-first, consistent across calls).

    Returns
    -------
    flat   : (n_params,) float32 numpy array
    schema : ordered list of (relative_name, shape) for reconstruction
    """
    schema: List[ParamEntry] = []
    parts: List[np.ndarray] = []
    for name, param in block.named_parameters():
        schema.append(ParamEntry(name, tuple(param.shape)))
        parts.append(param.detach().cpu().float().numpy().ravel())
    flat = np.concatenate(parts) if parts else np.array([], dtype=np.float32)
    return flat, schema


def count_block_params(block: nn.Module) -> int:
    return sum(p.numel() for p in block.parameters())


# ---------------------------------------------------------------------------
# Padding
# ---------------------------------------------------------------------------

def pad_block(
    flat: np.ndarray,
    target_size: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Zero-pad flat to target_size.

    Returns
    -------
    padded : (target_size,) float32 array
    mask   : (target_size,) uint8 array — 1 for real params, 0 for padding
    """
    n = len(flat)
    if n > target_size:
        raise ValueError(
            f"Block has {n:,} params, which exceeds target_size={target_size:,}. "
            "Re-run compute_max_block_size() to update the ceiling."
        )
    padded = np.zeros(target_size, dtype=np.float32)
    padded[:n] = flat
    mask = np.zeros(target_size, dtype=np.uint8)
    mask[:n] = 1
    return padded, mask


# ---------------------------------------------------------------------------
# Max block size discovery
# ---------------------------------------------------------------------------

def compute_max_block_size(arch_list: List[str]) -> int:
    """
    Load tiny random-init versions of each architecture and return the
    maximum number of parameters found in any single transformer block.

    Tiny models are used so no download is needed.  The block size scales
    with hidden_size, so the full-model max_block_size will be larger — but
    this function is only used when mode='tiny'.  For full-model runs,
    max_block_size is computed from the actually-loaded models.
    """
    max_size = 0
    for arch in arch_list:
        model = build_tiny_model(arch)
        layers = get_layers(model, arch)
        for layer in layers:
            size = count_block_params(layer)
            if size > max_size:
                max_size = size
        del model
    return max_size


# ---------------------------------------------------------------------------
# Full-model block extraction
# ---------------------------------------------------------------------------

def extract_all_blocks(
    model: nn.Module,
    arch: str,
    max_block_size: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    """
    Extract and pad all transformer blocks from one model.

    Parameters
    ----------
    model          : loaded HuggingFace causal LM (eval mode)
    arch           : architecture key (e.g. "gpt2_medium")
    max_block_size : target padding dimension (params per padded block)

    Returns
    -------
    blocks     : (n_layers, max_block_size) float32  — padded block flats
    masks      : (n_layers, max_block_size) uint8    — 1=real, 0=padding
    block_idxs : (n_layers,) int64                  — layer indices [0..L-1]
    family_idx : int                                 — model family index
    """
    cfg = get_arch_config(arch)
    layers = get_layers(model, arch)
    n_layers = len(layers)

    blocks = np.zeros((n_layers, max_block_size), dtype=np.float32)
    masks  = np.zeros((n_layers, max_block_size), dtype=np.uint8)

    for i, layer in enumerate(layers):
        flat, _schema = extract_block_flat(layer)
        padded, mask = pad_block(flat, max_block_size)
        blocks[i] = padded
        masks[i]  = mask

    block_idxs = np.arange(n_layers, dtype=np.int64)
    return blocks, masks, block_idxs, int(cfg["family_idx"])


# ---------------------------------------------------------------------------
# Reconstruction
# ---------------------------------------------------------------------------

def reconstruct_block(
    flat_unpadded: np.ndarray,
    block: nn.Module,
    schema: List[ParamEntry],
) -> None:
    """
    Load a flat weight vector back into block's parameters in-place.

    Parameters
    ----------
    flat_unpadded : (n_block_params,) float32 — must NOT include padding
    block         : the transformer block module (modified in-place)
    schema        : list of (name, shape) matching the original extraction order
    """
    param_dict = dict(block.named_parameters())
    offset = 0
    with torch.no_grad():
        for entry in schema:
            n = int(np.prod(entry.shape))
            chunk = flat_unpadded[offset: offset + n].reshape(entry.shape)
            tensor = torch.from_numpy(chunk.copy()).to(
                dtype=param_dict[entry.name].dtype,
                device=param_dict[entry.name].device,
            )
            param_dict[entry.name].copy_(tensor)
            offset += n

    if offset != len(flat_unpadded):
        raise ValueError(
            f"Schema total params ({offset}) ≠ flat_unpadded length ({len(flat_unpadded)})"
        )


# ---------------------------------------------------------------------------
# PCA loader interface
# ---------------------------------------------------------------------------

def make_block_loader(all_blocks: np.ndarray) -> Callable:
    """
    Build a loader callable suitable for BatchedCovariancePCA.fit().

    Parameters
    ----------
    all_blocks : (N, max_block_size) float32 array  [row = one block]

    Returns
    -------
    Callable(start, end) -> ndarray (max_block_size, batch_size)
    — column format expected by BatchedCovariancePCA
    """
    def loader(s: int, e: int) -> np.ndarray:
        batch = all_blocks[s:e, :]          # (batch, max_block_size)
        return np.ascontiguousarray(batch.T)  # (max_block_size, batch)
    return loader
