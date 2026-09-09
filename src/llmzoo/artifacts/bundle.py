"""
Reconstitute a whole-stack run from disk: ensemble, per-family PCA, code stats,
StackVAE, flows.

The problem this solves
----------------------
`DualGramPCA` deliberately never stores the (k, D) component matrix -- at k=99 and
D=302M it would be 119.6 GB per family. `inverse_transform` therefore STREAMS THE
WHOLE ENSEMBLE instead. A reloaded PCA is consequently NOT self-sufficient: decoding
any code back to weights needs a reconstructable `EnsembleDataset`, meaning the
`<arch>_w0.npy` files on disk plus ensemble parameters that match the ones the fit
used, bit for bit.

`EnsembleDataset` already refuses a parameter mismatch, so the invariant holds --
but only if every caller rebuilds the dataset from `ensemble_meta.json` rather than
from its own CLI flags. That was 61 hand-rolled lines in eval_stack.main, duplicated
nowhere else only because nothing else needed it yet. `load_run` makes it one call,
and adds the cross-artifact fingerprint checks that hand-rolled version had no way
to perform.

Ordering inside load_run is forced by the dependency chain:

    ensemble_meta.json -> EnsembleDataset -> DualGramPCA (needs the dataset to
    decode at all) -> CodeStats -> StackVAE (verified against CodeStats) ->
    latent-space flow (needs the VAE)

`want` exists because that chain is expensive at the front and often unnecessary:
`--arms pca_only` should not fail because `vae_k50/` is missing, and `train_flow.py`
needs only the 40 KB `codes_k<k>/` directory -- no memmapped mean, no ensemble.

The manifest is a derived index, not the source of truth
-------------------------------------------------------
Each artifact directory keeps its own authoritative `*_meta.json`; that is already
the pattern (`ensemble_meta.json`, `gram_pca_meta.json`). `run_manifest.json` is a
rebuildable index over them. One Slurm job invokes `train_stack.py` once per rank
and `train_flow.py` once per (rank, space), each a separate process, so a monolithic
read-modify-write manifest would race and drop sections. With a derived index a lost
update costs nothing -- `rebuild_manifest()` re-derives it by scanning -- and
`load_run` never REQUIRES the manifest to exist.
"""

from __future__ import annotations

import glob
import os
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

import numpy as np

from llmzoo.artifacts.io import (
    CODE_STATS_VERSION, MANIFEST_VERSION, atomic_write_json, ensemble_fingerprint,
    fingerprint, git_commit, pca_fingerprint, read_json, require_version,
)

WANT_ALL = ("dataset", "pca", "codes", "vae", "flow")


# ---------------------------------------------------------------------------
# Per-family code statistics
# ---------------------------------------------------------------------------

@dataclass
class CodeStats:
    """
    Per-family PCA-code mean and scale, promoted to a first-class artifact.

    Previously these lived only inside `StackVAE`'s `_code_mean` / `_code_std`
    buffers, recomputed from scratch on every run by `train_stack.stage_codes`. Three
    things follow from writing them out instead:

      1. `train_flow.py` can train on 40 KB -- codes, labels, statistics -- and never
         construct an `EnsembleDataset` or touch `DualGramPCA` at all. That is what
         makes the flow stack re-runnable on a different ensemble source (Pythia
         checkpoint revisions) without any change to the flow code: swap the source,
         re-run stages 1-3, and this artifact is the only interface.
      2. The statistics become inspectable and verifiable, so a `vae_k50/` left over
         from a run whose stage 3 differed can be DETECTED rather than silently used.
      3. `from_codes` becomes the single implementation of the reduction, instead of
         one copy in `stage_codes` and an implicit second one in `set_code_norm`.

    `stds` is one scalar PER FAMILY, not per dimension. PCA components are unit-norm
    and orthogonal, so MSE in raw code space is MSE in weight space up to a constant;
    per-dimension whitening would equalise PC 0 and PC k-1 and spend decoder capacity
    on directions carrying almost no weight-space energy.
    """
    k: int
    n_families: int
    means: np.ndarray            # (F, k) float32, per-dimension
    stds: np.ndarray             # (F, 1) float32, per-family scalar
    arch_to_family: Dict[str, int]
    meta: dict = field(default_factory=dict)
    codes: Optional[np.ndarray] = None        # (M, k) float32
    family_idxs: Optional[np.ndarray] = None  # (M,) int64

    STD_REDUCTION = "per_family_rms_over_dims"

    # -------- construction --------

    @classmethod
    def from_codes(cls, codes: np.ndarray, family_idxs: np.ndarray,
                   arch_to_family: Dict[str, int], n_families: int,
                   k: Optional[int] = None) -> "CodeStats":
        """
        The one implementation of the statistics, lifted from train_stack.stage_codes.

        Rows of `codes` belonging to family f give that family's per-dimension mean
        and a single scalar scale: the RMS over dimensions of the per-dimension
        standard deviations.
        """
        codes = np.asarray(codes, dtype=np.float32)
        family_idxs = np.asarray(family_idxs, dtype=np.int64)
        if codes.ndim != 2:
            raise ValueError(f"codes must be (M, k), got {codes.shape}")
        if codes.shape[0] != family_idxs.shape[0]:
            raise ValueError(f"codes has {codes.shape[0]} rows but family_idxs has "
                             f"{family_idxs.shape[0]}")
        kk = int(k or codes.shape[1])
        if kk > codes.shape[1]:
            raise ValueError(f"k={kk} exceeds code width {codes.shape[1]}")
        codes = codes[:, :kk]

        means = np.zeros((n_families, kk), dtype=np.float32)
        stds = np.ones((n_families, 1), dtype=np.float32)
        rows_per_family: Dict[str, int] = {}
        for fi in sorted(set(int(x) for x in family_idxs)):
            if not (0 <= fi < n_families):
                raise ValueError(f"family_idx {fi} is outside [0, {n_families})")
            block = codes[family_idxs == fi]
            means[fi] = block.mean(axis=0)
            # ddof=1 matches torch.std's default, which stage_codes used.
            per_dim = block.std(axis=0, ddof=1 if block.shape[0] > 1 else 0)
            per_dim = np.clip(per_dim, 1e-8, None)
            stds[fi, 0] = float(np.sqrt(np.mean(per_dim ** 2)))
            rows_per_family[str(fi)] = int(block.shape[0])

        return cls(k=kk, n_families=n_families, means=means, stds=stds,
                   arch_to_family=dict(arch_to_family),
                   codes=codes, family_idxs=family_idxs,
                   meta={"rows_per_family": rows_per_family})

    # -------- persistence --------

    def fingerprint(self) -> str:
        """
        Content hash over the statistics themselves.

        Rounded to 6 decimals: these are float32 reductions over a small matrix, so
        the last bits are not reproducible across BLAS versions, and a fingerprint
        that flips on a library upgrade is worse than no fingerprint.
        """
        return fingerprint({
            "layout_version": CODE_STATS_VERSION,
            "k": self.k,
            "n_families": self.n_families,
            "std_reduction": self.STD_REDUCTION,
            "arch_to_family": dict(sorted(self.arch_to_family.items())),
            "means": np.round(self.means.astype(np.float64), 6).tolist(),
            "stds": np.round(self.stds.astype(np.float64), 6).tolist(),
        })

    def save(self, save_dir: str, *, provenance: Optional[dict] = None) -> None:
        os.makedirs(save_dir, exist_ok=True)
        np.savez(os.path.join(save_dir, "code_stats.npz"),
                 means=self.means, stds=self.stds)
        if self.codes is not None:
            np.save(os.path.join(save_dir, "codes.npy"), self.codes)
        if self.family_idxs is not None:
            np.save(os.path.join(save_dir, "family_idxs.npy"), self.family_idxs)
        meta = {
            "layout_version": CODE_STATS_VERSION,
            "k": self.k,
            "n_families": self.n_families,
            "n_rows": int(0 if self.codes is None else self.codes.shape[0]),
            "arch_to_family": dict(sorted(self.arch_to_family.items())),
            "rows_per_family": self.meta.get("rows_per_family", {}),
            "std_reduction": self.STD_REDUCTION,
            "source": "train_stack.stage_codes",
            "fingerprint": self.fingerprint(),
            "provenance": provenance or {},
        }
        atomic_write_json(os.path.join(save_dir, "code_stats_meta.json"), meta)
        self.meta = meta
        print(f"[CodeStats] saved k={self.k} → {save_dir}  "
              f"(fingerprint {meta['fingerprint']})")

    @classmethod
    def load(cls, save_dir: str, *, expect_k: Optional[int] = None) -> "CodeStats":
        meta = read_json(os.path.join(save_dir, "code_stats_meta.json"))
        if meta is None:
            raise RuntimeError(
                f"Refusing to load {save_dir}: no code_stats_meta.json. This run "
                f"predates the codes artifact. Re-run train_stack.py for this rank.")
        require_version(meta, "layout_version", CODE_STATS_VERSION, save_dir)
        if expect_k is not None and int(meta["k"]) != int(expect_k):
            raise RuntimeError(
                f"Refusing to load {save_dir}: code stats are for k={meta['k']} but "
                f"k={expect_k} was requested.")

        z = np.load(os.path.join(save_dir, "code_stats.npz"))
        codes_p = os.path.join(save_dir, "codes.npy")
        fidx_p = os.path.join(save_dir, "family_idxs.npy")
        obj = cls(
            k=int(meta["k"]), n_families=int(meta["n_families"]),
            means=z["means"], stds=z["stds"],
            arch_to_family={a: int(f) for a, f in meta["arch_to_family"].items()},
            meta=meta,
            codes=np.load(codes_p) if os.path.exists(codes_p) else None,
            family_idxs=np.load(fidx_p) if os.path.exists(fidx_p) else None,
        )
        sealed = meta.get("fingerprint")
        got = obj.fingerprint()
        if sealed and sealed != got:
            raise RuntimeError(
                f"Refusing to load {save_dir}: code_stats.npz hashes to {got} but "
                f"code_stats_meta.json records {sealed}. One of the two files was "
                f"replaced without the other. Re-run train_stack.py for this rank.")
        return obj

    # -------- use --------

    def torch_means(self, device=None):
        import torch
        t = torch.from_numpy(np.ascontiguousarray(self.means)).float()
        return t.to(device) if device is not None else t

    def torch_stds(self, device=None):
        import torch
        t = torch.from_numpy(np.ascontiguousarray(self.stds)).float()
        return t.to(device) if device is not None else t

    def normalize(self, x, family_idx):
        """Match StackVAE._norm exactly, so the flow and the VAE agree on scale."""
        return (x - self.torch_means(x.device)[family_idx]) \
            / self.torch_stds(x.device)[family_idx]

    def denormalize(self, x, family_idx):
        return x * self.torch_stds(x.device)[family_idx] \
            + self.torch_means(x.device)[family_idx]


# ---------------------------------------------------------------------------
# The bundle
# ---------------------------------------------------------------------------

@dataclass
class RunBundle:
    """Everything a run needs at one rank, loaded and cross-checked."""
    run_root: str
    k: int
    arch_list: List[str]
    ens_meta: dict
    ensemble_fingerprint: str
    dataset: Optional[object] = None                  # EnsembleDataset
    pcas: Dict[str, object] = field(default_factory=dict)   # arch -> DualGramPCA
    code_stats: Optional[CodeStats] = None
    vae: Optional[object] = None                      # StackVAE
    flows: Dict[str, object] = field(default_factory=dict)  # space -> FlowModel
    manifest: dict = field(default_factory=dict)
    warnings: List[str] = field(default_factory=list)

    # -------- convenience --------

    @property
    def codes(self) -> Optional[np.ndarray]:
        return None if self.code_stats is None else self.code_stats.codes

    @property
    def family_idxs(self) -> Optional[np.ndarray]:
        return None if self.code_stats is None else self.code_stats.family_idxs

    @property
    def trust(self) -> str:
        """Worst trust level across the loaded artifacts. Propagates to the report."""
        levels = []
        for obj in (self.vae, *self.flows.values()):
            meta = getattr(obj, "loaded_meta", None) or getattr(obj, "meta", None)
            if isinstance(meta, dict):
                levels.append(meta.get("provenance", {}).get("trust", "verified"))
        return "unverified-legacy" if "unverified-legacy" in levels else "verified"

    def family_idx(self, arch: str) -> int:
        if self.dataset is not None:
            return int(self.dataset.stacks[arch].family_idx)
        return int(self.ens_meta["stacks"][arch]["family_idx"])

    def decode_codes_to_stack(self, codes_1d, arch: str,
                              k: Optional[int] = None) -> np.ndarray:
        """
        Codes -> full (D,) weight vector.

        The single place the (pca, dataset, arch, k) tuple is threaded together.
        Every generative arm ends here, which is why it is a method rather than
        four copies of the same call.
        """
        if self.dataset is None:
            raise RuntimeError(
                "decode_codes_to_stack needs the ensemble: DualGramPCA stores no "
                "basis, so inverse_transform streams the members. Call load_run "
                "with 'dataset' and 'pca' in `want`.")
        return self.pcas[arch].inverse_transform(
            np.asarray(codes_1d).reshape(-1), self.dataset, arch, k=k or self.k)


def _fail(bundle_warnings: List[str], strict: bool, msg: str) -> None:
    """Raise under strict, otherwise record. Triage tooling wants the second."""
    if strict:
        raise RuntimeError(msg)
    bundle_warnings.append(msg)
    print(f"[load_run] WARNING {msg}")


def load_run(
    run_root: str,
    k: int,
    *,
    arch_list: Optional[List[str]] = None,
    device: Optional[str] = None,
    want: Sequence[str] = WANT_ALL,
    flow_spaces: Sequence[str] = ("codes", "latent"),
    allow_legacy_vae: bool = False,
    strict: bool = True,
) -> RunBundle:
    """
    Reconstitute a stack run at rank `k`.

    Parameters
    ----------
    want
        Which artifacts to actually load, out of ("dataset", "pca", "codes", "vae",
        "flow"). Not cosmetic: constructing the dataset memmaps a ~1.2 GB mean per
        family, and `train_flow.py` needs none of it. Also lets `--arms pca_only`
        succeed on a run that has no VAE.
    strict
        True: any cross-artifact fingerprint mismatch raises. False: mismatches land
        in `bundle.warnings` and loading continues -- for triage only, never for
        producing a number you intend to report.
    """
    ens_meta_path = os.path.join(run_root, "ensemble", "ensemble_meta.json")
    ens_meta = read_json(ens_meta_path)
    if ens_meta is None:
        raise SystemExit(f"No ensemble at {ens_meta_path}. Run train_stack.py first.")

    archs = list(arch_list or ens_meta["arch_list"])
    missing = [a for a in archs if a not in ens_meta.get("stacks", {})]
    if missing:
        raise SystemExit(f"Ensemble at {run_root} has no stacks for {missing}.")

    ens_fp = ensemble_fingerprint(ens_meta)
    warnings: List[str] = []
    bundle = RunBundle(run_root=run_root, k=int(k), arch_list=archs,
                       ens_meta=ens_meta, ensemble_fingerprint=ens_fp,
                       manifest=read_json(os.path.join(run_root,
                                                       "run_manifest.json")) or {},
                       warnings=warnings)

    # --- ensemble ---------------------------------------------------------
    # Rebuilt from the RECORDED parameters, never from CLI flags. EnsembleDataset's
    # cache gate refuses a mismatch, so this guarantees inverse_transform streams
    # the same ensemble the PCA was fit on.
    if "dataset" in want:
        from llmzoo.data.ensemble import EnsembleDataset
        bundle.dataset = EnsembleDataset(
            arch_list=archs,
            n_samples=ens_meta["n_samples"],
            noise_scale=ens_meta["noise_scale"],
            exclude_1d=ens_meta["exclude_1d"],
            include_extra=ens_meta.get("include_extra", False),
            mode=ens_meta["mode"],
            artifact_dir=run_root,
            seed=ens_meta["seed"],
            chunk_budget_bytes=ens_meta["chunk_budget_bytes"],
            # Both default to a noise ensemble if omitted, and the cache gate
            # then refuses the run with "cached source='zoo' but this run asks
            # for 'noise'". Dropping them here made every downstream consumer --
            # train_flow, eval_stack, both diag scripts -- unable to open a zoo
            # run at all. Read from the recorded meta like everything else above.
            source=ens_meta.get("source", "noise"),
            zoo_dir=ens_meta.get("zoo_dir"),
        )

    # --- per-family PCA ---------------------------------------------------
    pca_fps: Dict[str, str] = {}
    if "pca" in want:
        from llmzoo.pca.gram import DualGramPCA
        for arch in archs:
            d = os.path.join(run_root, "pca", arch)
            bundle.pcas[arch] = DualGramPCA.load(d, device=device)
            pmeta = read_json(os.path.join(d, "gram_pca_meta.json")) or {}
            pca_fps[arch] = pca_fingerprint(pmeta)
            want_D = ens_meta["stacks"][arch]["n_params"]
            if int(pmeta.get("n_params", want_D)) != int(want_D):
                _fail(warnings, strict,
                      f"{arch}: PCA was fit on D={pmeta.get('n_params'):,} but the "
                      f"ensemble records D={want_D:,}. The stack layout changed "
                      f"(include_extra?). Re-fit with --force_pca.")
        fitted = min(p.n_components for p in bundle.pcas.values())
        if bundle.k > fitted:
            raise SystemExit(f"k={bundle.k} exceeds the fitted rank ({fitted}).")

    # --- code stats -------------------------------------------------------
    if "codes" in want:
        bundle.code_stats = CodeStats.load(
            os.path.join(run_root, f"codes_k{bundle.k}"), expect_k=bundle.k)
        sealed = bundle.code_stats.meta.get("provenance", {})
        if sealed.get("ensemble_fingerprint") not in (None, ens_fp):
            _fail(warnings, strict,
                  f"codes_k{bundle.k}/ was built from ensemble "
                  f"{sealed['ensemble_fingerprint']} but this run's ensemble hashes "
                  f"to {ens_fp}. Re-run train_stack.py for this rank.")

    # --- VAE --------------------------------------------------------------
    if "vae" in want:
        from llmzoo.gen.vae import StackVAE
        vdir = os.path.join(run_root, f"vae_k{bundle.k}")
        if os.path.exists(os.path.join(vdir, "vae_meta.json")) or \
                (allow_legacy_vae and
                 os.path.exists(os.path.join(vdir, "vae_config.json"))):
            bundle.vae = StackVAE.load(
                vdir, device=device, allow_legacy=allow_legacy_vae,
                expect_code_dim=bundle.k, code_stats=bundle.code_stats)
            vprov = bundle.vae.loaded_meta.get("provenance", {})
            if vprov.get("ensemble_fingerprint") not in (None, ens_fp):
                _fail(warnings, strict,
                      f"{vdir} was trained on ensemble "
                      f"{vprov['ensemble_fingerprint']} but this run's ensemble "
                      f"hashes to {ens_fp}. Re-train, or evaluate the run it "
                      f"belongs to.")
        else:
            print(f"[load_run] no StackVAE at {vdir} — VAE-dependent arms will be "
                  f"skipped. Run train_stack.py --k {bundle.k} first.")

    # --- flows ------------------------------------------------------------
    if "flow" in want:
        from llmzoo.gen.flow import load_flow
        for space in flow_spaces:
            fdir = os.path.join(run_root, f"flow_k{bundle.k}_{space}")
            if not os.path.exists(os.path.join(fdir, "flow_meta.json")):
                continue
            if space == "latent" and bundle.vae is None:
                _fail(warnings, strict,
                      f"{fdir} is a latent-space flow but no StackVAE was loaded at "
                      f"k={bundle.k}; it cannot be decoded. Include 'vae' in `want`.")
                continue
            bundle.flows[space] = load_flow(
                fdir, code_stats=bundle.code_stats, vae=bundle.vae, device=device)

    return bundle


# ---------------------------------------------------------------------------
# The manifest (a derived index)
# ---------------------------------------------------------------------------

def _k_from_dirname(name: str, prefix: str) -> Optional[int]:
    m = re.fullmatch(rf"{prefix}_?k(\d+)(?:_(\w+))?", name)
    return int(m.group(1)) if m else None


def rebuild_manifest(run_root: str) -> dict:
    """
    Re-derive run_manifest.json by scanning the run directory.

    Idempotent, and safe to call from any writer. Because this exists, a lost
    manifest update costs nothing -- which is why the manifest is allowed to be a
    best-effort index rather than a transactionally-maintained source of truth.
    """
    ens = read_json(os.path.join(run_root, "ensemble", "ensemble_meta.json")) or {}
    man: dict = {
        "manifest_version": MANIFEST_VERSION,
        "run_name": os.path.basename(os.path.normpath(run_root)),
        "run_root": os.path.abspath(run_root),
        "rebuilt": None,
        "git_commit": git_commit(),
        "ensemble": {}, "pca": {"per_arch": {}}, "codes": {}, "vae": {}, "flow": {},
        "results": [],
    }
    if ens:
        man["ensemble"] = {
            "dir": "ensemble",
            "layout_version": ens.get("layout_version"),
            "fingerprint": ensemble_fingerprint(ens),
            "arch_list": ens.get("arch_list", []),
            "n_samples": ens.get("n_samples"),
            "noise_scale": ens.get("noise_scale"),
            "exclude_1d": ens.get("exclude_1d"),
            "include_extra": ens.get("include_extra"),
            "mode": ens.get("mode"),
            "seed": ens.get("seed"),
            "chunk_budget_bytes": ens.get("chunk_budget_bytes"),
            "source": ens.get("source", "noise"),
        }

    for d in sorted(glob.glob(os.path.join(run_root, "pca", "*"))):
        pm = read_json(os.path.join(d, "gram_pca_meta.json"))
        if pm:
            man["pca"]["gram_layout_version"] = pm.get("layout_version")
            man["pca"]["per_arch"][os.path.basename(d)] = {
                "dir": os.path.relpath(d, run_root),
                "n_components": pm.get("n_components"),
                "n_samples": pm.get("n_samples"),
                "n_params": pm.get("n_params"),
                "fingerprint": pca_fingerprint(pm),
            }

    for d in sorted(glob.glob(os.path.join(run_root, "codes_k*"))):
        cm = read_json(os.path.join(d, "code_stats_meta.json"))
        if cm:
            man["codes"][str(cm["k"])] = {
                "dir": os.path.basename(d),
                "layout_version": cm.get("layout_version"),
                "code_dim": cm.get("k"),
                "n_rows": cm.get("n_rows"),
                "fingerprint": cm.get("fingerprint"),
            }

    for d in sorted(glob.glob(os.path.join(run_root, "vae_k*"))):
        vm = read_json(os.path.join(d, "vae_meta.json"))
        kk = _k_from_dirname(os.path.basename(d), "vae")
        if vm and kk is not None:
            man["vae"][str(kk)] = {
                "dir": os.path.basename(d),
                "layout_version": vm.get("layout_version"),
                "latent_dim": vm.get("config", {}).get("latent_dim"),
                "trust": vm.get("provenance", {}).get("trust"),
                "ensemble_fingerprint":
                    vm.get("provenance", {}).get("ensemble_fingerprint"),
                "code_stats_fingerprint":
                    vm.get("code_norm", {}).get("code_stats_fingerprint"),
            }

    for d in sorted(glob.glob(os.path.join(run_root, "flow_k*_*"))):
        fm = read_json(os.path.join(d, "flow_meta.json"))
        base = os.path.basename(d)
        m = re.fullmatch(r"flow_k(\d+)_(\w+)", base)
        if fm and m:
            kk, space = m.group(1), m.group(2)
            man["flow"].setdefault(kk, {})[space] = {
                "dir": base,
                "layout_version": fm.get("layout_version"),
                "dim": fm.get("dim"),
                "source_std": fm.get("flow", {}).get("source_std"),
                "n_steps_default": fm.get("defaults", {}).get("n_steps"),
                "trust": fm.get("provenance", {}).get("trust"),
                "spectrum": fm.get("train", {}).get("spectrum"),
            }

    man["results"] = sorted(
        os.path.relpath(p, run_root)
        for p in glob.glob(os.path.join(run_root, "results", "*.json")))

    import time
    man["rebuilt"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    atomic_write_json(os.path.join(run_root, "run_manifest.json"), man)
    return man


def update_manifest_section(run_root: str, section: str, key: str,
                            payload: dict) -> dict:
    """
    Read-modify-atomic-write one leaf of the index.

    Losing this write is harmless: rebuild_manifest() re-derives everything by
    scanning. That is the whole reason concurrent writers are tolerable here.
    """
    path = os.path.join(run_root, "run_manifest.json")
    man = read_json(path) or {"manifest_version": MANIFEST_VERSION}
    man.setdefault(section, {})[key] = payload
    atomic_write_json(path, man)
    return man
