"""
BlockDataset — PyTorch Dataset over transformer decoder blocks.

Each sample is one transformer block extracted from one of the model families.
With ~24-32 layers per model and 4 models, the base dataset has ~100 samples.
On-the-fly noise augmentation smooths learning dynamics at negligible cost.

Memory layout
-------------
Blocks are stored as pre-allocated numpy arrays (in memory for tiny mode;
memory-mapped npy files for full mode so $HOME is never touched).

mode='tiny' : random-init small models — full pipeline, no download needed.
mode='full' : real pretrained weights — requires ~8-12 GB disk + RAM.

All artifact paths default to /scratch/biggs.s/llm_vae on HPC.
"""

from __future__ import annotations

import gc
import json
import os
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

from models.registry import (
    ARCH_CONFIGS, build_tiny_model, get_arch_config,
    get_layers, list_archs, load_model,
)
from models.weight_extractor import (
    extract_block_flat, extract_all_blocks, make_block_loader,
    ParamEntry,
)


class BlockDataset(Dataset):
    """
    Pre-extracts all transformer blocks from all specified model families.

    Parameters
    ----------
    arch_list    : list of architecture keys to include (default: all 4)
    max_block_size : pad target dimension; None → computed from data
    noise_scale  : on-the-fly additive noise std relative to block's own std
                   (0 = no augmentation)
    mode         : 'tiny' (random-init, no download) or 'full' (pretrained)
    artifact_dir : directory for memmap files (full mode) and dataset metadata
    augment      : whether to apply noise augmentation in __getitem__
    """

    def __init__(
        self,
        arch_list: Optional[List[str]] = None,
        max_block_size: Optional[int] = None,
        noise_scale: float = 1e-7,
        mode: str = "full",
        artifact_dir: str = "/scratch/biggs.s/llm_vae",
        augment: bool = True,
    ):
        if arch_list is None:
            arch_list = list_archs()

        self.arch_list = arch_list
        self.noise_scale = noise_scale
        self.mode = mode
        self.artifact_dir = artifact_dir
        self.augment = augment

        blocks_dir = os.path.join(artifact_dir, "blocks")
        os.makedirs(blocks_dir, exist_ok=True)

        # Per-block metadata lists (populated during extraction)
        self._block_idxs: List[int] = []
        self._family_idxs: List[int] = []
        self._arch_names: List[str] = []
        self._schemas: List[List[ParamEntry]] = []
        self._real_sizes: List[int] = []   # number of real (non-padded) params

        # Phase 1: collect real block sizes to determine max_block_size
        all_arch_blocks: Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray, int]] = {}
        discovered_max = 0

        for arch in arch_list:
            print(f"\n[BlockDataset] Loading {arch} ({mode}) …")
            if mode == "tiny":
                model = build_tiny_model(arch)
            else:
                model = load_model(arch)

            model.eval()
            layers = get_layers(model, arch)
            n_layers = len(layers)

            # Collect schemas and real sizes (don't pad yet)
            arch_schemas: List[List[ParamEntry]] = []
            arch_real_sizes: List[int] = []
            arch_flats: List[np.ndarray] = []

            for layer in layers:
                flat, schema = extract_block_flat(layer)
                arch_schemas.append(schema)
                arch_real_sizes.append(len(flat))
                arch_flats.append(flat)
                if len(flat) > discovered_max:
                    discovered_max = len(flat)

            all_arch_blocks[arch] = (arch_flats, arch_schemas, arch_real_sizes, n_layers)

            # Free the full model immediately
            del model
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            print(f"  {arch}: {n_layers} blocks, {arch_real_sizes[0]:,} params/block (layer 0)")

        # Phase 2: resolve max_block_size
        if max_block_size is None:
            max_block_size = discovered_max
        elif max_block_size < discovered_max:
            raise ValueError(
                f"Provided max_block_size={max_block_size:,} is smaller than the "
                f"largest block found ({discovered_max:,}). Increase max_block_size."
            )
        self.max_block_size = max_block_size
        print(f"\n[BlockDataset] max_block_size = {max_block_size:,} params")

        # Phase 3: allocate storage and fill
        # Determine total number of blocks
        total_blocks = sum(info[3] for info in all_arch_blocks.values())
        print(f"[BlockDataset] Total blocks: {total_blocks}")

        if mode == "tiny":
            # Keep everything in RAM for tiny mode
            self._blocks = np.zeros((total_blocks, max_block_size), dtype=np.float32)
            self._masks  = np.zeros((total_blocks, max_block_size), dtype=np.uint8)
        else:
            # Memory-mapped files on scratch for full mode
            blocks_path = os.path.join(blocks_dir, "all_blocks.npy")
            masks_path  = os.path.join(blocks_dir, "all_masks.npy")
            self._blocks = np.memmap(blocks_path, dtype=np.float32, mode="w+",
                                     shape=(total_blocks, max_block_size))
            self._masks  = np.memmap(masks_path,  dtype=np.uint8,  mode="w+",
                                     shape=(total_blocks, max_block_size))

        row = 0
        for arch, (arch_flats, arch_schemas, arch_real_sizes, n_layers) in all_arch_blocks.items():
            cfg = get_arch_config(arch)
            family_idx = int(cfg["family_idx"])
            for i, (flat, schema, real_size) in enumerate(
                zip(arch_flats, arch_schemas, arch_real_sizes)
            ):
                # Pad and write
                n = len(flat)
                self._blocks[row, :n] = flat
                self._blocks[row, n:] = 0.0
                self._masks[row, :n]  = 1
                self._masks[row, n:]  = 0
                # Metadata
                self._block_idxs.append(i)
                self._family_idxs.append(family_idx)
                self._arch_names.append(arch)
                self._schemas.append(schema)
                self._real_sizes.append(real_size)
                row += 1

        if mode != "tiny":
            self._blocks.flush()
            self._masks.flush()

        # Convert metadata lists to arrays for fast indexing
        self._block_idxs  = np.array(self._block_idxs,  dtype=np.int64)
        self._family_idxs = np.array(self._family_idxs, dtype=np.int64)
        self._real_sizes  = np.array(self._real_sizes,  dtype=np.int64)

        # Save dataset metadata alongside the memmaps
        meta = {
            "total_blocks": total_blocks,
            "max_block_size": max_block_size,
            "arch_list": arch_list,
            "mode": mode,
            "noise_scale": noise_scale,
            "blocks_per_arch": {
                arch: int(info[3]) for arch, info in all_arch_blocks.items()
            },
        }
        meta_path = os.path.join(blocks_dir, "dataset_meta.json")
        with open(meta_path, "w") as f:
            json.dump(meta, f, indent=2)

        print(f"[BlockDataset] Ready: {total_blocks} blocks × {max_block_size:,} params "
              f"(noise_scale={noise_scale}, augment={augment})")

    # ------------------------------------------------------------------
    # Dataset interface
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self._block_idxs)

    def __getitem__(
        self, idx: int
    ) -> Tuple[torch.Tensor, torch.Tensor, int, int]:
        """
        Returns
        -------
        block_flat  : (max_block_size,) float32 tensor  [padded, possibly noisy]
        mask        : (max_block_size,) bool tensor      [True = real param]
        block_idx   : int scalar tensor
        family_idx  : int scalar tensor
        """
        flat = torch.from_numpy(self._blocks[idx].copy()).float()
        mask = torch.from_numpy(self._masks[idx].copy()).bool()

        if self.augment and self.noise_scale > 0.0:
            # Apply noise only to real (non-padded) positions
            real = flat[mask]
            std = real.std()
            if std > 0:
                noise = torch.randn_like(flat) * (self.noise_scale * std)
                noise[~mask] = 0.0   # zero out padding positions
                flat = flat + noise

        block_idx  = int(self._block_idxs[idx])
        family_idx = int(self._family_idxs[idx])
        return flat, mask, block_idx, family_idx

    # ------------------------------------------------------------------
    # Utilities for downstream use
    # ------------------------------------------------------------------

    def get_schema(self, idx: int) -> List[ParamEntry]:
        """Return the parameter schema for block idx (for reconstruction)."""
        return self._schemas[idx]

    def get_arch(self, idx: int) -> str:
        """Return the architecture name for block idx."""
        return self._arch_names[idx]

    def get_real_size(self, idx: int) -> int:
        """Return the number of non-padded params for block idx."""
        return int(self._real_sizes[idx])

    def make_loader(self) -> "Callable":
        """
        Return a loader function for BatchedCovariancePCA.fit().

        loader(start, end) -> (max_block_size, batch_size) float32 array
        """
        return make_block_loader(np.array(self._blocks))

    def all_blocks_numpy(self) -> np.ndarray:
        """Return a contiguous (N, max_block_size) float32 copy of all blocks."""
        return np.array(self._blocks)
