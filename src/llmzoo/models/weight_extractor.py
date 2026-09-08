"""
Block-wise weight extraction and reconstruction.

The key design choice: we use block.named_parameters() rather than
architecture-specific name lists. This gives a consistent, depth-first
parameter ordering that is stable for a given architecture and requires
no config changes when new architectures are added.

1-D parameter exclusion (`exclude_1d`)
--------------------------------------
LayerNorm/RMSNorm gains and projection biases are 1-D tensors.  They are <0.1%
of a block's parameter count, so an L2 objective over the flat vector gives them
almost no weight — yet they are functionally critical, and their value
distribution is nothing like a weight matrix's, which drags the shared PCA basis
in unhelpful directions.  That mechanism is why opt_350m was dropped from the
default arch_list (see README "Known issues"); pythia_160m/pythia_410m carry the
same per-projection biases.

With exclude_1d=True, parameters with ndim < 2 are left out of the flat vector
AND out of the returned schema.  Because reconstruct_block() only writes the
parameters named in the schema, those tensors are simply never touched — they
keep whatever the freshly-loaded pretrained model had.  No side storage needed.

The "extra" segment (embeddings, final norm, LM head)
-----------------------------------------------------
A whole-stack sample is `[extra | block_0 … block_{L-1}]`.  The extra segment is
defined as the COMPLEMENT of the `layers_attr` subtree in
`model.named_parameters()` — not as a per-architecture list of attribute names.
Two things fall out of that choice for free:

  * it stays architecture-agnostic, the same property that makes block extraction
    need no config changes when a family is added;
  * weight tying is handled automatically.  `named_parameters()` defaults to
    `remove_duplicate=True`, so a tied `lm_head.weight` / `wte.weight` pair is
    yielded exactly once.  Nothing is double-counted on extraction or
    double-written on reconstruction.

Public API
----------
  extract_block_flat(block, exclude_1d=False)  → (flat_float32, schema)
  extract_extra_flat(model, arch, exclude_1d)  → (flat_float32, schema)
  reconstruct_block(flat_unpadded, block, schema)
  reconstruct_params(flat, module, schema)     → alias, for root-relative schemas
  build_stack_spec(model, arch, exclude_1d, include_extra)
                                     → (w0_float32, StackSpec)
  read_stack_from_model(model, arch, spec)     → (D,) float32
  write_stack_to_model(stack, model, arch, spec)

The padding helpers (pad_block, compute_max_block_size, extract_all_blocks,
make_block_loader) went with the block pipeline: a whole-stack sample needs no
padding, because every member of a family has identical shape.
"""

from __future__ import annotations

from dataclasses import dataclass, field as dataclasses_field
from typing import List, NamedTuple, Tuple

import numpy as np
import torch
import torch.nn as nn

from llmzoo.models.registry import get_arch_config, get_layers


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
    exclude_1d: bool = False,
) -> Tuple[np.ndarray, List[ParamEntry]]:
    """
    Flatten the parameters of one transformer decoder block.

    Uses block.named_parameters() which returns parameters in PyTorch's
    canonical registration order (depth-first, consistent across calls).

    Parameters
    ----------
    exclude_1d : skip parameters with ndim < 2 (norm gains, biases). They are
                 omitted from both the flat vector and the schema, so
                 reconstruct_block() leaves them untouched — see module docstring.

    Returns
    -------
    flat   : (n_params,) float32 numpy array
    schema : ordered list of (relative_name, shape) for reconstruction
    """
    schema: List[ParamEntry] = []
    parts: List[np.ndarray] = []
    for name, param in block.named_parameters():
        if exclude_1d and param.dim() < 2:
            continue
        schema.append(ParamEntry(name, tuple(param.shape)))
        parts.append(param.detach().cpu().float().numpy().ravel())
    flat = np.concatenate(parts) if parts else np.array([], dtype=np.float32)
    return flat, schema




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
# Whole-stack extraction / write-back
#
# A whole-stack sample is  [extra | block_0 ... block_{L-1}].  Everything below
# operates on that layout.  `spec` is duck-typed rather than imported so this
# module stays a leaf: llmzoo/data/ensemble.py imports FROM here, and its
# ArchStack is what gets passed in.  Required attributes:
#
#     n_layers, block_size, extra_size, schema, extra_schema
#     block_slice(i) -> (lo, hi)        extra_slice() -> (lo, hi)
# ---------------------------------------------------------------------------

def extract_extra_flat(
    model: nn.Module,
    arch: str,
    exclude_1d: bool = False,
) -> Tuple[np.ndarray, List[ParamEntry]]:
    """
    Flatten every parameter that is NOT inside the decoder-block subtree.

    For a causal LM that is the input embedding, any learned positional
    embedding, the final norm, and the LM head when it is untied.  Schema names
    are dotted paths relative to `model`, so reconstruct_params() can write them
    back by passing the root model.

    Weight tying needs no special handling: named_parameters() deduplicates
    shared tensors, so a tied lm_head/wte pair appears exactly once.

    Returns an empty array and an empty schema if the model has no parameters
    outside the block subtree, which is what a bare decoder stack would give.
    """
    prefix = get_arch_config(arch)["layers_attr"] + "."
    schema: List[ParamEntry] = []
    parts: List[np.ndarray] = []
    for name, param in model.named_parameters():
        if name.startswith(prefix):
            continue
        if exclude_1d and param.dim() < 2:
            continue
        schema.append(ParamEntry(name, tuple(param.shape)))
        parts.append(param.detach().cpu().float().numpy().ravel())
    flat = np.concatenate(parts) if parts else np.zeros(0, dtype=np.float32)
    return flat, schema


# reconstruct_block resolves names through dict(module.named_parameters()), so it
# already works verbatim on the root model with root-relative schema names. The
# alias exists so call sites reading the extra segment do not look like a bug.
reconstruct_params = reconstruct_block


def read_stack_from_model(model: nn.Module, arch: str, spec) -> np.ndarray:
    """
    Read a model's current weights out as one flat (D,) float32 stack vector.

    The inverse of write_stack_to_model, and the way to snapshot a pristine model
    before an evaluation arm mutates it. Returning a single array rather than a
    list of per-block flats matters now that the extra segment exists: a list of
    blocks silently omits the embeddings, so restore() would leave a mutated
    embedding behind and every arm after the first would measure a corrupted
    baseline.
    """
    parts: List[np.ndarray] = []
    if spec.extra_size:
        extra, schema = extract_extra_flat(model, arch, exclude_1d=_excludes_1d(spec))
        if len(extra) != spec.extra_size:
            raise RuntimeError(
                f"{arch}: extra segment is {len(extra):,} params but the ensemble "
                f"recorded {spec.extra_size:,}. The model or exclude_1d changed.")
        parts.append(extra)
    layers = get_layers(model, arch)
    if len(layers) != spec.n_layers:
        raise RuntimeError(f"{arch}: model has {len(layers)} layers but the ensemble "
                           f"recorded {spec.n_layers}")
    for layer in layers:
        parts.append(extract_block_flat(layer, exclude_1d=_excludes_1d(spec))[0])
    out = np.concatenate(parts).astype(np.float32, copy=False)
    if out.size != spec.n_params:
        raise RuntimeError(f"{arch}: read {out.size:,} params but the ensemble "
                           f"recorded D={spec.n_params:,}")
    return out


def write_stack_to_model(stack: np.ndarray, model: nn.Module, arch: str,
                         spec) -> None:
    """
    Slice a flat (D,) stack vector back into the model's parameters, in place.

    Every block has the same schema and the same size (EnsembleDataset enforces
    both at extraction), so block offsets are just extra_size + i * block_size.
    reconstruct_block writes only the parameters named in the schema, so with
    exclude_1d the norm gains and biases are never touched and keep their
    pretrained values.
    """
    if spec.extra_size:
        lo, hi = spec.extra_slice()
        reconstruct_params(np.ascontiguousarray(stack[lo:hi]), model,
                           spec.extra_schema)
    layers = get_layers(model, arch)
    if len(layers) != spec.n_layers:
        raise RuntimeError(f"{arch}: model has {len(layers)} layers but the ensemble "
                           f"recorded {spec.n_layers}")
    for i, layer in enumerate(layers):
        lo, hi = spec.block_slice(i)
        reconstruct_block(np.ascontiguousarray(stack[lo:hi]), layer, spec.schema)


def _excludes_1d(spec) -> bool:
    """
    Whether `spec` was built with exclude_1d.

    Read off the schema rather than stored: a spec whose schema contains no 1-D
    entry was extracted with exclude_1d=True. Deriving it keeps ArchStack from
    having to carry a flag that duplicates its own schema.
    """
    for entry in list(spec.extra_schema) + list(spec.schema):
        if len(entry.shape) < 2:
            return False
    return True


@dataclass
class StackSpec:
    """
    A standalone whole-stack layout, for callers that need the [extra | blocks]
    geometry without building an EnsembleDataset (which writes ~1.2 GB of w0 to
    scratch as a side effect).

    This is the reference implementation of the `spec` protocol that
    read_stack_from_model / write_stack_to_model consume. EnsembleDataset's
    ArchStack implements the same protocol and adds ensemble-specific fields.
    """
    n_layers: int
    block_size: int
    extra_size: int
    n_params: int
    weight_std: float
    schema: List[ParamEntry]
    extra_schema: List[ParamEntry] = dataclasses_field(default_factory=list)

    def extra_slice(self) -> Tuple[int, int]:
        return 0, self.extra_size

    def block_slice(self, block_idx: int) -> Tuple[int, int]:
        lo = self.extra_size + block_idx * self.block_size
        return lo, lo + self.block_size


def build_stack_spec(
    model: nn.Module,
    arch: str,
    exclude_1d: bool = True,
    include_extra: bool = True,
) -> Tuple[np.ndarray, StackSpec]:
    """
    Flatten a loaded model into (w0, spec) using the same layout and the same
    uniformity checks EnsembleDataset applies.

    Returns the (D,) float32 stack and its spec. `weight_std` is the std over the
    WHOLE stack, which is what the augmentation noise is scaled by — so with
    include_extra=True it now reflects the embeddings too, and a noise scale
    calibrated on decoder blocks alone no longer transfers.
    """
    if include_extra:
        extra, extra_schema = extract_extra_flat(model, arch, exclude_1d=exclude_1d)
    else:
        extra, extra_schema = np.zeros(0, dtype=np.float32), []

    layers = get_layers(model, arch)
    flats, schemas = [], []
    for layer in layers:
        flat, schema = extract_block_flat(layer, exclude_1d=exclude_1d)
        flats.append(flat)
        schemas.append(schema)

    sizes = {len(f) for f in flats}
    if len(sizes) != 1:
        raise ValueError(
            f"{arch}: decoder blocks are NOT uniform in size ({sorted(sizes)}). "
            f"A whole-stack sample requires a fixed length.")
    names0 = [(e.name, e.shape) for e in schemas[0]]
    for i, sch in enumerate(schemas[1:], start=1):
        if [(e.name, e.shape) for e in sch] != names0:
            raise ValueError(f"{arch}: block {i} schema differs from block 0")

    w0 = np.concatenate([extra] + flats).astype(np.float32)
    spec = StackSpec(
        n_layers=len(layers),
        block_size=sizes.pop(),
        extra_size=int(extra.size),
        n_params=int(w0.size),
        weight_std=float(w0.std()),
        schema=schemas[0],
        extra_schema=extra_schema,
    )
    return w0, spec
