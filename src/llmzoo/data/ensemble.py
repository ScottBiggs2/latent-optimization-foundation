"""
EnsembleDataset — per-architecture ensembles of WHOLE decoder stacks.

Why this exists (and how it differs from the block layout it replaced)
----------------------------------------------------------------------
The removed block pipeline treated **one transformer block** as one PCA sample,
zero-padded every block to a shared `max_block_size`, and fit ONE basis over all
families.

This module follows DeepWeightFlow (arXiv 2601.05052), where **one PCA sample is the
final weight vector of one complete network**:

    sample  = the whole network, flattened  (D_f = extra + n_layers x block_size)
    N       = the number of complete models in the ensemble
    k       = N - 1 (the rank bound) or N // 2
    basis   = one per architecture

Two consequences worth stating, because they are why this layout was chosen:

  * **Zero padding.** Every family has uniform block sizes (verified on real
    configs), so a stack is a fixed length and no mask is needed anywhere.
  * **Conditioning stops being a primary key.** The block layout gave every block a
    unique `(family_idx, block_idx)`, so a conditioned decoder could memorise the
    whole training set and ignore z — the measured cause of the 0.001 nats/sample
    posterior collapse. Here N samples share one `family_idx`, so it cannot.

The ensemble
------------
DeepWeightFlow trains ~100 networks from independent seeds. We have exactly ONE
pretrained model per family, so the ensemble is manufactured:

    sample 0      = w_0, the real pretrained stack (eps = 0 exactly)
    sample 1..N-1 = w_0 + noise_scale * weight_std * eps_i

Sample 0 being the genuine model matters: it makes the evaluation target an actual
ensemble member, so reconstruction at k = N-1 is exact by the rank bound.

**This is a machinery test, not evidence about weight-space structure.** An isotropic
ensemble spans injected-noise directions, so a generative model fit to it learns to
reproduce perturbations of one network. Real structure needs genuinely different
complete models — Pythia's ~143 checkpoint revisions are the cheap source at fixed
model scale.

Noise is never stored
---------------------
w_i = w_0 + s*sigma*eps_i, so eps_i is regenerated from a seed on every pass instead
of being written to scratch. At N=100 and D=3e8 that saves ~370 GB of I/O per family
and is far faster (GPU RNG rather than a Lustre read). Reproducibility across passes
requires a FIXED chunk grid, which this class owns — see `chunk_bounds`.
"""

from __future__ import annotations

import gc
import json
import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

from llmzoo.models.registry import (
    build_tiny_model, get_arch_config, get_layers, list_archs, load_model,
)
from llmzoo.models.weight_extractor import (
    ParamEntry, extract_block_flat, extract_extra_flat,
)

# Bumped whenever the on-disk layout under <artifact_dir>/ensemble/ changes.
#   1 -> 2 : a stack became [extra | block_0 .. block_{L-1}] instead of blocks
#            alone, so D and every code, basis and checkpoint changed meaning.
ENSEMBLE_LAYOUT_VERSION = 2

# Target bytes for one in-flight chunk across all samples. 512 MB keeps a 100-sample
# ensemble comfortably inside a 32 GB GPU alongside the Gram accumulator.
DEFAULT_CHUNK_BUDGET_BYTES = 512 * 1024 * 1024

# slurm/aicr_env.sh exports ARTIFACT_DIR (project space, persistent). The local
# fallback keeps a bare `python scripts/train_stack.py` from writing to somebody
# else's scratch path.
DEFAULT_ARTIFACT_DIR = os.environ.get("ARTIFACT_DIR", "./artifacts")

# Chunks below this many elements re-read filesystem pages and stop saturating the
# GPU on the RNG call; above it, larger is only marginally better.
MIN_CHUNK_ELEMS = 262_144
MAX_CHUNK_ELEMS = 8_388_608


# Where an ensemble's members come from. RESEARCH_PLAN Phase 0.3.
#
#   "noise" : members 1..N-1 are w_0 + s*sigma*eps_i, regenerated from a seed on
#             every pass and never written to disk. The machinery test, and the
#             beta -> 0 reference point of RESEARCH_PLAN §4.2.
#   "zoo"   : members are genuinely different trained models, one .npy each,
#             produced by scripts/train_zoo.py. No noise, no seed, no w_0 anywhere
#             -- member 0 is not privileged and must not be treated as such.
#
# `source` is already inside ensemble_fingerprint() (defaulting to "noise"), so
# switching it invalidates every downstream artifact automatically and a zoo PCA
# can never be paired with a noise-ensemble VAE.
ENSEMBLE_SOURCES = ("noise", "zoo")


class MemberSource:
    """
    Where the rows of one chunk come from.

    One method, deliberately: given a stack and a half-open element range, return
    (n_samples, r1-r0) float32. Everything above this -- the chunk grid, the Gram
    accumulation, PCA, the flow -- is indifferent to which subclass it got.
    """

    def load_rows(self, ds: "EnsembleDataset", arch: str, chunk_idx: int,
                  r0: int, r1: int, device, w0_mm) -> torch.Tensor:
        raise NotImplementedError

    def state(self) -> dict:
        raise NotImplementedError


class NoiseSource(MemberSource):
    """w_0 + s*sigma*eps_i, regenerated from (seed, chunk_idx) on every pass."""

    def load_rows(self, ds, arch, chunk_idx, r0, r1, device, w0_mm):
        st = ds.stacks[arch]
        L = r1 - r0
        # np.array (not ascontiguousarray) to force a writable copy: w0 is opened
        # read-only via mmap and torch.from_numpy warns on non-writable buffers.
        base = torch.from_numpy(np.array(w0_mm[r0:r1])).to(device)

        # Written in place so peak memory is one (N, L) block rather than two --
        # a repeat() plus a separate randn() would double the chunk budget.
        out = torch.empty((ds.n_samples, L), device=base.device, dtype=base.dtype)
        out[0] = base                                  # sample 0 is the real model
        if ds.n_samples > 1:
            if ds.noise_scale > 0.0:
                g = torch.Generator(device=out.device)
                # Same grid + same seed => same ensemble on every pass.
                g.manual_seed((ds.seed * 1_000_003 + chunk_idx) % (2**63 - 1))
                torch.randn((ds.n_samples - 1, L), generator=g,
                            device=out.device, dtype=out.dtype, out=out[1:])
                out[1:].mul_(ds.noise_scale * st.weight_std).add_(base)
            else:
                out[1:] = base
        return out

    def state(self) -> dict:
        return {"source": "noise"}


class ZooSource(MemberSource):
    """
    N genuinely different trained models, one `w_<i>.npy` per member.

    Memmaps are opened once per (arch, member) and cached, because a Gram fit walks
    every chunk of every member and reopening N files per chunk is the difference
    between minutes and hours on a shared filesystem.

    No member is privileged here. `w0(arch)` still returns member 0 because the rest
    of the pipeline needs *a* reference stack to write back into, but for a zoo it is
    an arbitrary member, not a centre -- so `--sample_idx 0` loses the special
    meaning it has for a noise ensemble (RESEARCH_NOTES misstep 9).
    """

    def __init__(self, zoo_dir: str):
        self.zoo_dir = zoo_dir
        self._mm: Dict[Tuple[str, int], np.ndarray] = {}

    def member_path(self, arch: str, i: int) -> str:
        return os.path.join(self.zoo_dir, arch, f"w_{i}.npy")

    def _member(self, arch: str, i: int) -> np.ndarray:
        key = (arch, i)
        if key not in self._mm:
            self._mm[key] = np.load(self.member_path(arch, i), mmap_mode="r")
        return self._mm[key]

    def load_rows(self, ds, arch, chunk_idx, r0, r1, device, w0_mm):
        out = torch.empty((ds.n_samples, r1 - r0), device=device, dtype=torch.float32)
        for i in range(ds.n_samples):
            out[i] = torch.from_numpy(np.array(self._member(arch, i)[r0:r1])).to(device)
        return out

    def state(self) -> dict:
        return {"source": "zoo", "zoo_dir": self.zoo_dir}


def build_source(source: str, *, zoo_dir: Optional[str] = None) -> MemberSource:
    if source not in ENSEMBLE_SOURCES:
        raise ValueError(f"source must be one of {ENSEMBLE_SOURCES}; got {source!r}")
    if source == "noise":
        return NoiseSource()
    if zoo_dir is None:
        raise ValueError("source='zoo' requires zoo_dir=")
    return ZooSource(zoo_dir)


@dataclass
class ArchStack:
    """
    One architecture's real weight stack plus everything needed to write it back.

    Layout is  [extra | block_0 .. block_{L-1}]  where `extra` is the embeddings,
    the final norm, and the LM head when untied. The extra segment leads rather
    than trails so that `block_slice` stays a single offset add and so a
    decoder-only stack (extra_size == 0) has byte-identical block offsets to the
    layout_version 1 ensembles.
    """
    arch: str
    family_idx: int
    n_layers: int
    block_size: int          # params per block (uniform within the family)
    n_params: int            # D = extra_size + n_layers * block_size
    weight_std: float
    schema: List[ParamEntry]  # per-block schema; identical for every block
    w0_path: str
    extra_size: int = 0
    extra_schema: List[ParamEntry] = field(default_factory=list)

    def extra_slice(self) -> Tuple[int, int]:
        """Offsets of the extra segment. It leads the stack, so it starts at 0."""
        return 0, self.extra_size

    def block_slice(self, block_idx: int) -> Tuple[int, int]:
        """Offsets of block `block_idx` within the flattened stack."""
        lo = self.extra_size + block_idx * self.block_size
        return lo, lo + self.block_size


class EnsembleDataset:
    """
    Per-architecture noise-augmented ensembles of whole decoder stacks.

    Parameters
    ----------
    arch_list     : architectures to include
    n_samples     : N, ensemble size per architecture (sample 0 is the real model)
    noise_scale   : augmentation std as a fraction of the stack's own weight std
    exclude_1d    : omit norm gains and biases from the flat vector, so they keep
                    their pretrained values on write-back
    include_extra : include the embeddings, final norm and LM head in the stack.
                    DeepWeightFlow flattens whole networks, and gpt2_medium's
                    embedding alone is ~17% of its parameters, so this is on by
                    default. Set False to reproduce a decoder-blocks-only run.
    mode          : 'full' (pretrained) or 'tiny' (random-init, for smoke tests)
    artifact_dir  : root for the ensemble/ subdirectory
    seed          : base seed for noise generation
    chunk_budget_bytes : target size of one in-flight (N, chunk) block
    source        : 'noise' (default, members regenerated from a seed) or 'zoo'
                    (members read from disk, one w_<i>.npy each). See
                    ENSEMBLE_SOURCES.
    zoo_dir       : required when source='zoo'. Expects <zoo_dir>/<arch>/w_<i>.npy
                    for i in 0..n_samples-1, all the same length.
    """

    def __init__(
        self,
        arch_list: Optional[List[str]] = None,
        n_samples: int = 100,
        noise_scale: float = 1e-2,
        exclude_1d: bool = True,
        include_extra: bool = True,
        mode: str = "full",
        artifact_dir: str = DEFAULT_ARTIFACT_DIR,
        seed: int = 42,
        chunk_budget_bytes: int = DEFAULT_CHUNK_BUDGET_BYTES,
        force_extract: bool = False,
        source: str = "noise",
        zoo_dir: Optional[str] = None,
    ):
        if arch_list is None:
            arch_list = list_archs()
        if n_samples < 3:
            raise ValueError(f"n_samples must be >= 3 to leave a usable rank "
                             f"(N-1 components); got {n_samples}")

        self.arch_list = list(arch_list)
        self.n_samples = int(n_samples)
        self.noise_scale = float(noise_scale)
        self.exclude_1d = bool(exclude_1d)
        self.include_extra = bool(include_extra)
        self.mode = mode
        self.artifact_dir = artifact_dir
        self.seed = int(seed)
        self.chunk_budget_bytes = int(chunk_budget_bytes)
        self.source = str(source)
        self.zoo_dir = zoo_dir
        self._src = build_source(self.source, zoo_dir=zoo_dir)

        if self.source == "zoo" and self.noise_scale != 0.0:
            raise ValueError(
                f"source='zoo' with noise_scale={self.noise_scale} would add "
                f"augmentation noise on top of genuinely different models, which is "
                f"not what any part of RESEARCH_PLAN §4 describes. Pass "
                f"noise_scale=0.0."
            )

        self.ens_dir = os.path.join(artifact_dir, "ensemble")
        os.makedirs(self.ens_dir, exist_ok=True)

        self.stacks: Dict[str, ArchStack] = {}
        meta_path = os.path.join(self.ens_dir, "ensemble_meta.json")

        if os.path.exists(meta_path) and not force_extract:
            self._load_meta(meta_path)
        else:
            self._extract(meta_path)

    # ------------------------------------------------------------------
    # Extraction / metadata
    # ------------------------------------------------------------------

    def _extract(self, meta_path: str) -> None:
        for arch in self.arch_list:
            print(f"\n[EnsembleDataset] Extracting {arch} ({self.mode}, "
                  f"source={self.source}) …")
            if self.source == "zoo":
                self._extract_zoo(arch)
                continue
            model = build_tiny_model(arch) if self.mode == "tiny" else load_model(arch)
            model.eval()
            cfg = get_arch_config(arch)
            layers = get_layers(model, arch)

            # The extra segment is the complement of the block subtree, so it
            # picks up embeddings / final norm / untied LM head without any
            # per-architecture configuration. See weight_extractor's docstring.
            if self.include_extra:
                extra, extra_schema = extract_extra_flat(
                    model, arch, exclude_1d=self.exclude_1d)
            else:
                extra, extra_schema = np.zeros(0, dtype=np.float32), []

            flats, schemas = [], []
            for layer in layers:
                flat, schema = extract_block_flat(layer, exclude_1d=self.exclude_1d)
                flats.append(flat)
                schemas.append(schema)

            sizes = {len(f) for f in flats}
            if len(sizes) != 1:
                raise ValueError(
                    f"{arch}: decoder blocks are NOT uniform in size ({sorted(sizes)}). "
                    f"A whole-stack sample requires a fixed length. This family needs "
                    f"per-depth handling, or exclude_1d changed the picture."
                )
            block_size = sizes.pop()

            # The schema must match across blocks too, or write-back would need one
            # schema per depth.
            names0 = [(e.name, e.shape) for e in schemas[0]]
            for i, sch in enumerate(schemas[1:], start=1):
                if [(e.name, e.shape) for e in sch] != names0:
                    raise ValueError(f"{arch}: block {i} schema differs from block 0")

            w0 = np.concatenate([extra] + flats).astype(np.float32)
            w0_path = os.path.join(self.ens_dir, f"{arch}_w0.npy")
            np.save(w0_path, w0)

            self.stacks[arch] = ArchStack(
                arch=arch,
                family_idx=int(cfg["family_idx"]),
                n_layers=len(layers),
                block_size=block_size,
                n_params=int(w0.size),
                weight_std=float(w0.std()),
                schema=schemas[0],
                w0_path=w0_path,
                extra_size=int(extra.size),
                extra_schema=extra_schema,
            )
            pct = 100.0 * extra.size / max(w0.size, 1)
            print(f"  {arch}: L={len(layers)}  block={block_size:,}  "
                  f"extra={extra.size:,} ({pct:.1f}%)  "
                  f"D={w0.size:,}  std={w0.std():.6g}")
            if self.include_extra and extra.size == 0:
                raise ValueError(
                    f"{arch}: include_extra=True but no parameters were found "
                    f"outside '{cfg['layers_attr']}'. That layers_attr is probably "
                    f"wrong, or exclude_1d removed everything.")

            del model, flats, schemas, w0, extra
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        self._save_meta(meta_path)

    def _extract_zoo(self, arch: str) -> None:
        """
        Adopt a pre-flattened zoo. No model is loaded and nothing is re-flattened:
        scripts/train_zoo.py already wrote one (D,) float32 per member plus the
        schema, and re-deriving it here would be a second implementation of the
        layout that could drift from the first.
        """
        adir = os.path.join(self.zoo_dir, arch)
        zmeta_path = os.path.join(adir, "zoo_meta.json")
        if not os.path.exists(zmeta_path):
            raise FileNotFoundError(
                f"source='zoo' but {zmeta_path} is missing. Run scripts/train_zoo.py "
                f"for {arch} first.")
        with open(zmeta_path) as f:
            z = json.load(f)

        if z.get("layout_version") != ENSEMBLE_LAYOUT_VERSION:
            raise RuntimeError(
                f"{zmeta_path}: layout_version={z.get('layout_version')!r} but this "
                f"code expects {ENSEMBLE_LAYOUT_VERSION}.")
        # The flattening flags are baked into the .npy files, so a mismatch here
        # means D means something different than this run thinks it does.
        for key, mine in (("exclude_1d", self.exclude_1d),
                          ("include_extra", self.include_extra)):
            if z.get(key) != mine:
                raise RuntimeError(
                    f"{zmeta_path}: zoo was flattened with {key}={z.get(key)!r} but "
                    f"this run asks for {mine!r}. Re-flatten or change the flag.")
        if int(z["n_members"]) < self.n_samples:
            raise RuntimeError(
                f"{arch}: zoo has {z['n_members']} members, run asks for "
                f"n_samples={self.n_samples}.")

        D = int(z["n_params"])
        for i in range(self.n_samples):
            p = self._src.member_path(arch, i)
            if not os.path.exists(p):
                raise FileNotFoundError(f"{arch}: member {i} missing at {p}")
            # Read the header only -- np.load(mmap_mode='r') does not read the body,
            # so this validates N files in milliseconds rather than reading ~D*N bytes.
            n = int(np.load(p, mmap_mode="r").shape[0])
            if n != D:
                raise ValueError(
                    f"{arch}: member {i} has D={n:,} but zoo_meta says {D:,}. A zoo "
                    f"with ragged members cannot be a fixed-length PCA sample.")

        cfg = get_arch_config(arch)
        self.stacks[arch] = ArchStack(
            arch=arch,
            family_idx=int(cfg["family_idx"]),
            n_layers=int(z["n_layers"]),
            block_size=int(z["block_size"]),
            n_params=D,
            # Informational only for a zoo: nothing scales noise by it, because
            # there is no noise. Kept so the metadata schema is uniform.
            weight_std=float(z.get("weight_std", 0.0)),
            schema=[ParamEntry(n, tuple(s)) for n, s in z["schema"]],
            # Member 0 is an ARBITRARY member, not a centre. It is the write-back
            # reference only. misstep 9's sample_idx warning does not apply here.
            w0_path=self._src.member_path(arch, 0),
            extra_size=int(z.get("extra_size", 0)),
            extra_schema=[ParamEntry(n, tuple(s))
                          for n, s in z.get("extra_schema", [])],
        )
        pct = 100.0 * int(z.get("extra_size", 0)) / max(D, 1)
        print(f"  {arch}: N={self.n_samples}  L={z['n_layers']}  "
              f"block={int(z['block_size']):,}  extra={int(z.get('extra_size', 0)):,} "
              f"({pct:.1f}%)  D={D:,}  [zoo]")

    def _save_meta(self, meta_path: str) -> None:
        meta = {
            "layout_version": ENSEMBLE_LAYOUT_VERSION,
            "arch_list": self.arch_list,
            "n_samples": self.n_samples,
            "noise_scale": self.noise_scale,
            "exclude_1d": self.exclude_1d,
            "include_extra": self.include_extra,
            "mode": self.mode,
            "seed": self.seed,
            "chunk_budget_bytes": self.chunk_budget_bytes,
            # Only written when non-default, so a source="noise" ensemble
            # fingerprints byte-identically to one written before Phase 0.3.
            **({} if self.source == "noise"
               else {"source": self.source, "zoo_dir": self.zoo_dir}),
            "stacks": {
                a: {
                    "family_idx": s.family_idx,
                    "n_layers": s.n_layers,
                    "block_size": s.block_size,
                    "n_params": s.n_params,
                    "weight_std": s.weight_std,
                    "schema": [[e.name, list(e.shape)] for e in s.schema],
                    "extra_size": s.extra_size,
                    "extra_schema": [[e.name, list(e.shape)] for e in s.extra_schema],
                    "w0_path": s.w0_path,
                }
                for a, s in self.stacks.items()
            },
        }
        with open(meta_path, "w") as f:
            json.dump(meta, f, indent=2)
        print(f"\n[EnsembleDataset] meta → {meta_path}")

    def _load_meta(self, meta_path: str) -> None:
        with open(meta_path) as f:
            meta = json.load(f)
        found = meta.get("layout_version")
        if found != ENSEMBLE_LAYOUT_VERSION:
            raise RuntimeError(
                f"Refusing to reuse {self.ens_dir}: layout_version={found!r} but this "
                f"code expects {ENSEMBLE_LAYOUT_VERSION}. Pass force_extract=True or "
                f"use a fresh --artifact_dir."
            )
        # Anything that changes the ensemble's contents must invalidate the cache,
        # otherwise a fit silently uses a different ensemble than the flags describe.
        for key, mine in (("n_samples", self.n_samples),
                          ("noise_scale", self.noise_scale),
                          ("exclude_1d", self.exclude_1d),
                          ("include_extra", self.include_extra),
                          ("seed", self.seed),
                          ("mode", self.mode),
                          # .get default matches _save_meta's omission of the
                          # default, so a pre-Phase-0.3 cache still loads.
                          ("source", self.source)):
            if key == "source":
                if meta.get("source", "noise") != mine:
                    raise RuntimeError(
                        f"Refusing to reuse {self.ens_dir}: cached "
                        f"source={meta.get('source', 'noise')!r} but this run asks "
                        f"for {mine!r}. These are different ensembles.")
                continue
            if meta.get(key) != mine:
                raise RuntimeError(
                    f"Refusing to reuse {self.ens_dir}: cached {key}={meta.get(key)!r} "
                    f"but this run asks for {mine!r}. The ensemble contents depend on "
                    f"it. Pass force_extract=True or use a fresh --artifact_dir."
                )
        missing = [a for a in self.arch_list if a not in meta["stacks"]]
        if missing:
            raise RuntimeError(f"Cached ensemble lacks {missing}; re-extract.")

        self.chunk_budget_bytes = int(meta.get("chunk_budget_bytes",
                                                self.chunk_budget_bytes))
        for arch in self.arch_list:
            d = meta["stacks"][arch]
            self.stacks[arch] = ArchStack(
                arch=arch,
                family_idx=int(d["family_idx"]),
                n_layers=int(d["n_layers"]),
                block_size=int(d["block_size"]),
                n_params=int(d["n_params"]),
                weight_std=float(d["weight_std"]),
                schema=[ParamEntry(n, tuple(s)) for n, s in d["schema"]],
                w0_path=d["w0_path"],
                extra_size=int(d.get("extra_size", 0)),
                extra_schema=[ParamEntry(n, tuple(sh))
                              for n, sh in d.get("extra_schema", [])],
            )
        print(f"[EnsembleDataset] reusing {self.ens_dir} "
              f"(N={self.n_samples}, noise_scale={self.noise_scale})")

    # ------------------------------------------------------------------
    # Chunk grid  (fixed, because the noise seeds are keyed to it)
    # ------------------------------------------------------------------

    def chunk_size(self, arch: str) -> int:
        """
        Elements per chunk. Derived from a byte budget rather than hardcoded: the
        in-flight block is (N, chunk) float32, so the right chunk depends on N.
        """
        raw = self.chunk_budget_bytes // (4 * self.n_samples)
        return int(np.clip(raw, MIN_CHUNK_ELEMS,
                            min(MAX_CHUNK_ELEMS, self.stacks[arch].n_params)))

    def chunk_bounds(self, arch: str) -> List[Tuple[int, int]]:
        """
        The FIXED chunk grid for this architecture.

        Noise is keyed by (seed, sample block, chunk index), so every pass must use
        this exact partition or the regenerated ensemble differs between passes.
        `load_chunk` takes a chunk index rather than raw offsets to make that
        impossible to get wrong.
        """
        D = self.stacks[arch].n_params
        cs = self.chunk_size(arch)
        return [(r0, min(r0 + cs, D)) for r0 in range(0, D, cs)]

    def w0(self, arch: str) -> np.ndarray:
        """Memory-mapped view of the real pretrained stack."""
        return np.load(self.stacks[arch].w0_path, mmap_mode="r")

    def load_chunk(
        self,
        arch: str,
        chunk_idx: int,
        device: torch.device | str = "cpu",
        w0_mm: Optional[np.ndarray] = None,
    ) -> torch.Tensor:
        """
        Return the ensemble restricted to one chunk: (n_samples, r1-r0) float32.

        Dispatches to `self._src` (NoiseSource or ZooSource). Everything above this
        method -- the Gram accumulation, PCA, the flow -- is indifferent to which.

        For source='noise': row 0 is the real model, rows 1.. are w0 + s*sigma*eps
        regenerated deterministically from (seed, chunk_idx).
        For source='zoo': row i is member i, read from disk. No row is privileged.

        w0_mm : pass the result of w0(arch) to avoid reopening the memmap per chunk.
                Unused by ZooSource, which caches its own memmaps.
        """
        bounds = self.chunk_bounds(arch)
        if not (0 <= chunk_idx < len(bounds)):
            raise IndexError(f"chunk_idx {chunk_idx} out of range "
                             f"(0..{len(bounds) - 1}) for {arch}")
        r0, r1 = bounds[chunk_idx]

        if w0_mm is None and self.source == "noise":
            w0_mm = self.w0(arch)
        return self._src.load_rows(self, arch, chunk_idx, r0, r1, device, w0_mm)

    # ------------------------------------------------------------------
    # Convenience
    # ------------------------------------------------------------------

    def materialize_sample(self, arch: str, idx: int,
                           device: torch.device | str = "cpu") -> np.ndarray:
        """
        Reconstruct ensemble member `idx` in full, (D,) float32. One streaming pass.

        For source='noise' only sample 0 is on disk and every other member exists
        only as a seed, and sample 0 sits ~sqrt(N) closer to the ensemble mean than a
        typical member -- so a rank sweep measured on sample 0 alone understates
        truncation loss badly (misstep 9).

        For source='zoo' every member is already a file, and no member is at a
        privileged centre, so that warning does not apply and any index is
        representative.
        """
        if not (0 <= idx < self.n_samples):
            raise IndexError(f"sample {idx} out of range (0..{self.n_samples - 1})")
        if self.source == "zoo":
            return np.array(np.load(self._src.member_path(arch, idx), mmap_mode="r"),
                            dtype=np.float32)
        out = np.empty(self.stacks[arch].n_params, dtype=np.float32)
        w0_mm = self.w0(arch)
        for ci, (r0, r1) in enumerate(self.chunk_bounds(arch)):
            B = self.load_chunk(arch, ci, device=device, w0_mm=w0_mm)
            out[r0:r1] = B[idx].float().cpu().numpy()
            del B
        return out

    def family_idxs(self) -> Dict[str, int]:
        return {a: s.family_idx for a, s in self.stacks.items()}

    def summary(self) -> str:
        lines = [f"EnsembleDataset  source={self.source}  N={self.n_samples}  "
                 f"noise_scale={self.noise_scale}  exclude_1d={self.exclude_1d}  "
                 f"include_extra={self.include_extra}"]
        for a in self.arch_list:
            s = self.stacks[a]
            nb = len(self.chunk_bounds(a))
            lines.append(f"  {a:16s} L={s.n_layers:3d}  block={s.block_size:>12,}  "
                         f"extra={s.extra_size:>12,}  "
                         f"D={s.n_params:>13,}  chunks={nb:>5d}  "
                         f"std={s.weight_std:.4g}")
        return "\n".join(lines)
