"""
Dual / Graham iterative PCA for neural network weight spaces.

Faithfully adapted from DeepWeightFlow (NNeuralDynamics/DeepWeightFlow),
extended with fit_from_iterator() for direct numpy/iterator-based usage.

Algorithm overview
------------------
Rather than forming the enormous (n_params × n_params) covariance matrix,
the "dual" trick builds the (n_models × n_models) Gram matrix C = X^T X
where X has columns (w_i - mean).  For large models (e.g. LLaMA 3.1 8B,
~8B params) with a small ensemble (e.g. 50 models), C is only 50×50.

Four streaming passes over the data:
  Pass 1 – accumulate mean
  Pass 2 – build C block-by-block (outer loop stays in memory, inner loads/discards).
           NOTE this is O(n²/b) in disk reads — it re-reads the whole ensemble once
           per outer batch. Fine at n~150; see the param-chunked path for large n.
  Pass 3 – symmetric eigendecomposition of C (eigh — see fit())
  Pass 4 – vectorised back-projection to obtain full-space components

Memory footprint: O(micro_batch_size × n_params + n_models²)
"""

import gc
import tempfile
from typing import Callable, Iterator, Optional, Tuple

import numpy as np
import torch
from tqdm import tqdm


# Eigenvalues below this fraction of the leading eigenvalue are discarded as
# numerical noise.
#
# The dual trick forms C = Xc^T Xc, which squares the condition number: a singular
# value ratio of r shows up as an eigenvalue ratio of r^2. Block weights are
# extracted as float32 (eps ~ 1.2e-7), so singular values below ~1e-4 of the leading
# one are already at the input's noise level, which puts the corresponding
# EIGENVALUE floor at ~1e-8..1e-7. 1e-7 is the conservative end of that range.
#
# Setting this lower does not recover more signal — it admits eigenvectors whose
# directions are pure roundoff, which then get normalised to unit length in Pass 4
# and given equal footing with the real components in transform()/inverse_transform().
# That was the concrete failure mode of the old randomized_svd path.
DEFAULT_RANK_RTOL = 1e-7


class BatchedCovariancePCA:
    """
    GPU-accelerated incremental covariance-based PCA.

    Accepts a model_loader_func callable: (start_idx, end_idx) -> ndarray
    of shape (n_params, n_models_in_batch), enabling disk-streaming over
    arbitrarily large model collections without loading everything at once.
    """

    def __init__(self, n_components: int, seed: int = 42, device: str = "cuda"):
        self.n_components = n_components
        self.seed = seed
        self.device = device if torch.cuda.is_available() else "cpu"
        self.mean_ = None
        self.components_ = None
        self.explained_variance_ = None
        self.explained_variance_ratio_ = None
        # Total variance across ALL directions (not just the retained ones).
        # Needed to report a meaningful explained-variance ratio — see fit().
        self.total_variance_ = None
        self.n_models_ = None
        self.n_params_ = None

        if self.device == "cuda":
            print(f"GPU: {torch.cuda.get_device_name(0)}")
            mem_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
            print(f"GPU memory: {mem_gb:.1f} GB")
        else:
            print("CUDA unavailable — using CPU")

    # ------------------------------------------------------------------
    # Primary interface: callable loader
    # ------------------------------------------------------------------

    def fit(
        self,
        model_loader_func: Callable,
        n_models: int,
        batch_size: int = 20,
        use_fp16: bool = False,
        micro_batch_size: Optional[int] = None,
        rank_rtol: float = DEFAULT_RANK_RTOL,
    ) -> "BatchedCovariancePCA":
        """
        Fit PCA by building the Gram covariance matrix incrementally.

        Parameters
        ----------
        model_loader_func:
            Callable(start_idx, end_idx) -> ndarray (n_params, n_models_in_batch)
        n_models:
            Total number of models (= ensemble size / perturbation count)
        batch_size:
            Logical batch size for disk loading
        use_fp16:
            FP16 GPU computation (2× memory reduction, minor precision loss)
        micro_batch_size:
            Further subdivision for models with >1B parameters.
            If None, batch_size is used directly.
        rank_rtol:
            Eigenvalues below rank_rtol * max_eigenvalue are treated as numerical
            noise and dropped.  See DEFAULT_RANK_RTOL for why the default is what
            it is — this is not a knob to loosen casually.
        """
        print(f"Fitting PCA: {n_models} models, up to {self.n_components} components")
        self.n_models_ = n_models
        compute_dtype = torch.float16 if use_fp16 else torch.float32
        effective_bs = micro_batch_size if micro_batch_size else batch_size

        # ---- Pass 1: mean -------------------------------------------------
        print("Pass 1/4: computing mean …")
        mean_acc = None
        for s in tqdm(range(0, n_models, batch_size), desc="mean"):
            e = min(s + batch_size, n_models)
            batch = model_loader_func(s, e)
            if mean_acc is None:
                self.n_params_ = batch.shape[0]
                mean_acc = torch.zeros(self.n_params_, dtype=torch.float32, device=self.device)
            bt = torch.from_numpy(batch).to(self.device, dtype=compute_dtype)
            mean_acc += torch.sum(bt, dim=1, dtype=torch.float32)
            del batch, bt
            if self.device == "cuda":
                torch.cuda.empty_cache()
            gc.collect()

        self.mean_ = (mean_acc / n_models).cpu().numpy()
        del mean_acc
        if self.device == "cuda":
            torch.cuda.empty_cache()
        gc.collect()

        # ---- Pass 2: Gram matrix  C = X_c^T X_c  (n_models × n_models) ----
        print(f"Pass 2/4: building Gram matrix ({n_models}×{n_models}) …")
        # float64 accumulator. Entries of C are sums over n_params products; at
        # D ~ 3e8 the fp32 relative error is ~sqrt(D)*eps ~ 1e-3, which is the same
        # order as the structure the eigendecomposition has to resolve. The per-block
        # matmul below still runs in compute_dtype on the GPU; only the accumulation
        # is promoted.
        C = np.zeros((n_models, n_models), dtype=np.float64)
        mean_t = torch.from_numpy(self.mean_).to(self.device, dtype=compute_dtype)

        for i_s in tqdm(range(0, n_models, effective_bs), desc="Gram rows"):
            i_e = min(i_s + effective_bs, n_models)
            bi = model_loader_func(i_s, i_e)
            bi_t = torch.from_numpy(bi).to(self.device, dtype=compute_dtype)
            bi_c = bi_t - mean_t.unsqueeze(1)

            for j_s in range(0, n_models, effective_bs):
                j_e = min(j_s + effective_bs, n_models)
                bj = model_loader_func(j_s, j_e)
                bj_t = torch.from_numpy(bj).to(self.device, dtype=compute_dtype)
                bj_c = bj_t - mean_t.unsqueeze(1)

                # (batch_i, n_params)^T @ (n_params, batch_j)
                C[i_s:i_e, j_s:j_e] = (bi_c.T @ bj_c).cpu().double().numpy()

                del bj, bj_t, bj_c
                if self.device == "cuda":
                    torch.cuda.empty_cache()

            del bi, bi_t, bi_c
            if self.device == "cuda":
                torch.cuda.empty_cache()
            gc.collect()

        del mean_t
        if self.device == "cuda":
            torch.cuda.empty_cache()
        gc.collect()

        # Total variance over ALL n_models-1 directions.  tr(C) = Σ‖x_i − mean‖²,
        # and the eigenvalues of C/(n−1) are the per-direction variances, so
        # tr(C)/(n−1) is the total.  Captured BEFORE the eigendecomposition because
        # only the top n_comp eigenvalues are retained — normalising by their sum
        # (the original behaviour) makes explained_variance_ratio_ sum to exactly
        # 1.0 by construction, which says nothing about how much truncation discards.
        self.total_variance_ = float(np.trace(C)) / (n_models - 1)

        # ---- Pass 3: symmetric eigendecomposition of C ---------------------
        #
        # eigh, not randomized_svd.  C is a symmetric PSD Gram matrix of side
        # n_models, so eigh is exact and instant at these sizes.  randomized_svd was
        # actively harmful here for two reasons:
        #
        #   1. It returns |eigenvalue|, discarding sign.  Trailing directions that
        #      round negative come back POSITIVE with essentially arbitrary
        #      eigenvectors -- and the unit-normalisation further down then scales
        #      that noise to length 1 and gives it equal footing with the real
        #      components in transform()/inverse_transform().
        #   2. Its accuracy degrades exactly where we care most: at k -> n-1 on a
        #      steep spectrum.  Measured on a synthetic n=24 case with an fp32 Gram,
        #      orthogonality error reached 0.82 at k = n-1 and one component's
        #      direction was 100% wrong.
        #
        # eigh gives SIGNED eigenvalues, so the numerically-null tail can be detected
        # and dropped instead of silently poisoning the basis.
        print(f"Pass 3/4: symmetric eigendecomposition of {n_models}×{n_models} Gram matrix …")
        evals, evecs = np.linalg.eigh(C)              # ascending, C is symmetric
        order  = np.argsort(evals)[::-1]              # -> descending
        evals  = evals[order]
        evecs  = evecs[:, order]

        # Rank floor. The mathematical cap is n_models-1 (one degree of freedom goes
        # to the mean); anything below rank_rtol of the leading eigenvalue is
        # roundoff, not signal.
        rel_tol  = float(rank_rtol)
        max_eval = float(evals[0]) if evals.size else 0.0
        n_valid  = int(np.sum(evals > max(max_eval * rel_tol, 0.0)))
        n_comp   = min(self.n_components, n_models - 1, max(n_valid, 1))
        if n_valid < min(self.n_components, n_models - 1):
            print(f"  NOTE: only {n_valid} of {min(self.n_components, n_models - 1)} "
                  f"requested directions are above the {rel_tol:g} rank floor "
                  f"(lead eigenvalue {max_eval:.6g}). Keeping {n_comp}. A spectrum "
                  f"this rank-deficient usually means the ensemble members are too "
                  f"close together for the accumulator to separate them.")
        # Keep self.n_components truthful: transform() allocates its output with it
        # and save() records it as the codes width, so a stale value here silently
        # mis-shapes the codes file downstream.
        self.n_components = n_comp

        # Condition diagnostic. The dual formulation builds C = Xc^T Xc, which
        # SQUARES the condition number of the data: a singular-value ratio of r
        # becomes an eigenvalue ratio of r^2. So a spectrum that is merely steep in
        # singular values is unrecoverable in eigenvalues, and no choice of
        # rank_rtol fixes it — the information is gone before eigh ever runs.
        # A flat spectrum (e.g. an isotropic perturbation ensemble) is the benign
        # case; warn loudly on the other one.
        if n_comp > 1 and evals[0] > 0:
            spread = float(evals[0] / max(evals[n_comp - 1], np.finfo(np.float64).tiny))
            print(f"  spectrum: eigenvalue ratio ev[0]/ev[{n_comp - 1}] = {spread:.4g}")
            if spread > 1e6:
                print(f"  WARNING: the retained spectrum spans {spread:.3g}. Because the "
                      f"Gram matrix squares the data's condition number, the trailing "
                      f"components here are at or below the input precision and their "
                      f"DIRECTIONS are unreliable even though their magnitudes look "
                      f"plausible. Reduce n_components, or use a covariance-based PCA "
                      f"if the small directions genuinely matter.")

        U = np.ascontiguousarray(evecs[:, :n_comp])
        S = evals[:n_comp]
        eigenvalues = S / (n_models - 1)
        del C, evals, evecs
        gc.collect()

        # ---- Pass 4: back-project to get full-space components (vectorised) -
        print("Pass 4/4: computing principal components in parameter space …")
        components = torch.zeros((n_comp, self.n_params_), dtype=compute_dtype, device=self.device)
        U_t = torch.from_numpy(U).to(self.device, dtype=compute_dtype)
        mean_t = torch.from_numpy(self.mean_).to(self.device, dtype=compute_dtype)

        for s in tqdm(range(0, n_models, effective_bs), desc="components"):
            e = min(s + effective_bs, n_models)
            batch = model_loader_func(s, e)
            bt = torch.from_numpy(batch).to(self.device, dtype=compute_dtype)
            bc = bt - mean_t.unsqueeze(1)
            # (n_params, batch) @ (batch, n_comp)  -> (n_params, n_comp)
            components += (bc @ U_t[s:e, :]).T
            del batch, bt, bc
            if self.device == "cuda":
                torch.cuda.empty_cache()
            gc.collect()

        # Normalise to unit length.  This is load-bearing, not cosmetic:
        # inverse_transform() is only the adjoint of transform() when the rows are
        # orthonormal.  Xc @ U has column norms sqrt(S) exactly (since
        # ||Xc u_j||^2 = u_j^T C u_j = S_j), so the measured norms double as a free
        # consistency check on the whole Gram + eigendecomposition path.
        norms = torch.norm(components, dim=1, keepdim=True)
        measured = norms.squeeze(1).double().cpu().numpy()
        expected = np.sqrt(np.clip(S, 0.0, None))
        scale    = float(expected[0]) if expected.size and expected[0] > 0 else 1.0
        norm_err = float(np.max(np.abs(measured - expected)) / scale)
        if norm_err > 1e-3:
            print(f"  WARNING: component norms disagree with sqrt(eigenvalues) by "
                  f"{norm_err:.3g} (relative to the leading value). The Gram matrix "
                  f"and the streamed back-projection are inconsistent — suspect "
                  f"precision loss or a loader that is not returning the same data "
                  f"on every pass.")
        else:
            print(f"  norm check: max |‖component‖ − √eigenvalue| = {norm_err:.3g} (rel)")
        components = components / torch.clamp(norms, min=1e-10)

        self.components_ = components.cpu().float().numpy()
        self.explained_variance_ = eigenvalues[:n_comp]
        # Normalise by the TRUE total variance (all directions), not by the sum of
        # the retained eigenvalues — so this ratio actually measures what the
        # truncation to n_comp components keeps.
        total_var = self.total_variance_
        self.explained_variance_ratio_ = eigenvalues[:n_comp] / (total_var if total_var > 0 else 1.0)

        del components, U_t, mean_t
        if self.device == "cuda":
            torch.cuda.empty_cache()
        gc.collect()

        cum_var = np.sum(self.explained_variance_ratio_)
        print(f"Fitted {n_comp}/{n_models - 1} components  |  variance captured: {cum_var:.4%}"
              f"  (discarded: {max(0.0, 1.0 - cum_var):.4%})")
        return self

    # ------------------------------------------------------------------
    # Convenience: fit directly from a Python iterator of numpy arrays
    # ------------------------------------------------------------------

    def fit_from_iterator(
        self,
        weight_iter: Iterator[np.ndarray],
        n_models: int,
        batch_size: int = 20,
        use_fp16: bool = False,
    ) -> "BatchedCovariancePCA":
        """
        Fit PCA when weights arrive as a Python iterator of flat numpy arrays,
        each of shape (n_params,).  Arrays are buffered into a temp memmap and
        then handed to the standard fit() path so streaming guarantees hold.
        """
        # Buffer the iterator into a memmap (n_params, n_models)
        tmp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".npy")
        n_params = None
        store = None
        for i, w in enumerate(weight_iter):
            if store is None:
                n_params = w.shape[0]
                store = np.memmap(tmp_file.name, dtype=np.float32, mode="w+",
                                  shape=(n_params, n_models))
            store[:, i] = w.astype(np.float32)
        store.flush()

        def loader(s, e):
            return np.array(store[:, s:e])

        return self.fit(loader, n_models, batch_size=batch_size, use_fp16=use_fp16)

    # ------------------------------------------------------------------
    # Transform / inverse
    # ------------------------------------------------------------------

    def transform(
        self,
        model_loader_func: Callable,
        n_models: int,
        batch_size: int = 20,
        output_file: Optional[str] = None,
        use_fp16: bool = False,
    ) -> str:
        """
        Project models into PCA latent space.

        Returns path to memory-mapped file of shape (n_models, n_components).
        """
        if output_file is None:
            output_file = tempfile.NamedTemporaryFile(delete=False, suffix=".npy").name
        if not output_file.endswith(".npy"):
            raise ValueError(f"output_file must end in .npy (open_memmap writes a "
                             f"real npy header): {output_file}")

        # open_memmap writes a real .npy HEADER, unlike np.memmap. That matters:
        # a headerless raw memmap read back with an explicit shape SUCCEEDS SILENTLY
        # whenever the requested shape is smaller than the file, so a stale codes
        # file from a previous run is reinterpreted as the current one instead of
        # raising. Use load_codes() to read this back.
        projected = np.lib.format.open_memmap(
            output_file, mode="w+", dtype=np.float32,
            shape=(n_models, self.n_components))

        compute_dtype = torch.float16 if use_fp16 else torch.float32
        mean_t = torch.from_numpy(self.mean_).to(self.device, dtype=compute_dtype)
        comp_t = torch.from_numpy(self.components_).to(self.device, dtype=compute_dtype)

        idx = 0
        for s in tqdm(range(0, n_models, batch_size), desc="transform"):
            e = min(s + batch_size, n_models)
            batch = model_loader_func(s, e)
            bt = torch.from_numpy(batch).to(self.device, dtype=compute_dtype)
            bc = bt - mean_t.unsqueeze(1)
            # (batch, n_params) @ (n_components, n_params)^T
            proj = (bc.T @ comp_t.T).cpu().float().numpy()
            projected[idx: idx + proj.shape[0]] = proj
            idx += proj.shape[0]
            del batch, bt, bc, proj
            if self.device == "cuda":
                torch.cuda.empty_cache()
            gc.collect()

        projected.flush()
        del mean_t, comp_t
        if self.device == "cuda":
            torch.cuda.empty_cache()
        gc.collect()

        print(f"Latent projections saved → {output_file}")
        return output_file

    def inverse_transform(self, latent_vectors: np.ndarray) -> np.ndarray:
        """
        Reconstruct weight vectors from PCA coordinates.

        Parameters
        ----------
        latent_vectors: (n_samples, n_components)

        Returns
        -------
        (n_samples, n_params) float32 array
        """
        return (latent_vectors @ self.components_) + self.mean_

    def save(self, save_dir: str) -> None:
        """Save PCA artifacts to save_dir (components in float32)."""
        import json
        import os
        os.makedirs(save_dir, exist_ok=True)
        # float32, not float16.  Components are unit-norm over n_params, so a
        # typical entry is ~1/sqrt(D) -- 2.8e-4 at D=12.6M, 5.7e-5 at D=3e8. fp16
        # relative precision there is ~8.5e-4 and a large fraction of entries fall
        # into the subnormal range, which is a first-order error term against a
        # cosine >= 0.9999 fidelity bar. It also made train.py (in-memory fp32) and
        # eval_pca_only.py (loads from disk) measure two different bases.
        np.save(os.path.join(save_dir, "components.npy"),
                self.components_.astype(np.float32))
        np.save(os.path.join(save_dir, "mean.npy"), self.mean_)
        np.save(os.path.join(save_dir, "explained_variance.npy"), self.explained_variance_)
        np.save(os.path.join(save_dir, "explained_variance_ratio.npy"),
                self.explained_variance_ratio_)
        meta = {
            "n_components": int(self.n_components),
            "n_params": int(self.n_params_),
            "n_models": int(self.n_models_),
            "seed": self.seed,
            # Fraction of the TRUE total variance retained by the n_components
            # kept.  Before the trace fix this was always exactly 1.0 — see fit().
            "total_variance_captured": float(np.sum(self.explained_variance_ratio_)),
            "total_variance": (float(self.total_variance_)
                                if self.total_variance_ is not None else None),
            "max_components": int(self.n_models_ - 1) if self.n_models_ else None,
        }
        with open(os.path.join(save_dir, "pca_meta.json"), "w") as f:
            json.dump(meta, f, indent=2)
        print(f"PCA saved to {save_dir}  "
              f"(components: {self.components_.shape}, float32 on disk)")

    @classmethod
    def load(cls, save_dir: str, device: str = "cuda") -> "BatchedCovariancePCA":
        """Load PCA from save_dir. Components are cast to float32 regardless of on-disk dtype."""
        import json
        import os
        with open(os.path.join(save_dir, "pca_meta.json")) as f:
            meta = json.load(f)
        pca = cls(n_components=meta["n_components"], seed=meta["seed"], device=device)
        # astype is a no-op for float32 artifacts and an upcast for pre-fix
        # float16 ones, so old checkpoints still load (at their original precision).
        pca.components_ = np.load(
            os.path.join(save_dir, "components.npy")).astype(np.float32)
        pca.mean_ = np.load(os.path.join(save_dir, "mean.npy"))
        pca.explained_variance_ = np.load(
            os.path.join(save_dir, "explained_variance.npy"))
        pca.explained_variance_ratio_ = np.load(
            os.path.join(save_dir, "explained_variance_ratio.npy"))
        pca.n_models_ = meta["n_models"]
        pca.n_params_ = meta["n_params"]
        # Absent in checkpoints written before the trace fix in fit().
        pca.total_variance_ = meta.get("total_variance")
        print(f"PCA loaded from {save_dir}  "
              f"({pca.n_components} components, {pca.n_params_:,} params, "
              f"{meta['total_variance_captured']:.4%} variance captured)")
        if pca.total_variance_ is None:
            print("  NOTE: this PCA predates the explained-variance fix — its "
                  "'variance captured' figure is normalised by the retained "
                  "eigenvalues only and is always 1.0. Re-fit with --force_pca "
                  "for a meaningful number.")
        return pca


# ---------------------------------------------------------------------------
# Codes I/O
# ---------------------------------------------------------------------------

CODES_LAYOUT_VERSION = 2


def load_codes(
    path: str,
    n_models: Optional[int] = None,
    n_components: Optional[int] = None,
) -> np.ndarray:
    """
    Read a codes file written by BatchedCovariancePCA.transform().

    Always use this instead of np.memmap with an explicit shape.  transform() used
    to write a HEADERLESS raw memmap, and np.memmap(mode="r", shape=...) succeeds
    SILENTLY whenever the requested shape is smaller than the file on disk — it only
    raises when the shape is larger.  So a stale (150, 97) codes file read as
    (100, 50) returned the first 5,000 floats as if they were the current codes:
    plausible-looking numbers, silently wrong.

    Raises with an actionable message on a legacy headerless file rather than
    guessing its shape.
    """
    try:
        arr = np.load(path, mmap_mode="r")
    except ValueError as exc:
        raise ValueError(
            f"{path} is not a valid .npy file ({exc}). It was almost certainly "
            f"written by the pre-2026-08-25 code path as a headerless raw memmap, "
            f"whose shape cannot be recovered from the file. Re-run the encode "
            f"stage with --force_encode to regenerate it."
        ) from exc

    arr = np.array(arr)   # materialise; codes are (N, k) and tiny
    if arr.ndim != 2:
        raise ValueError(f"{path}: expected a 2-D codes array, got shape {arr.shape}")

    want = (n_models if n_models is not None else arr.shape[0],
            n_components if n_components is not None else arr.shape[1])
    if arr.shape != want:
        raise ValueError(
            f"{path}: codes shape {arr.shape} does not match the expected "
            f"{want}. This is a stale artifact from a different run — re-run the "
            f"encode stage with --force_encode. (Refusing to reinterpret it: the "
            f"old raw-memmap path would have accepted this silently.)"
        )
    return arr


# ---------------------------------------------------------------------------
# High-level wrapper (mirrors DeepWeightFlow API exactly)
# ---------------------------------------------------------------------------

def flatten_and_project_to_disk_maxcomponents(
    model_loader_func: Callable,
    n_models: int = 100,
    n_components: Optional[int] = None,
    target_variance: Optional[float] = None,
    batch_size: int = 20,
    output_file: Optional[str] = None,
    use_fp16: bool = True,
    micro_batch_size: Optional[int] = None,
    device: str = "cuda",
) -> Tuple[str, "BatchedCovariancePCA", dict]:
    """
    Fit PCA with automatic component selection, then transform to disk.

    Parameters
    ----------
    n_components:
        Fixed number of components.  Mutually exclusive with target_variance.
    target_variance:
        Automatically pick the minimum number of components that capture this
        fraction of total variance (e.g. 0.99).
    """
    max_comp = n_models - 1

    if n_components is None and target_variance is None:
        n_components = max_comp
        print(f"Using maximum components: {n_components}")
    elif target_variance is not None:
        print("Pre-fitting to determine component count for target variance …")
        tmp_pca = BatchedCovariancePCA(n_components=max_comp, device=device)
        tmp_pca.fit(model_loader_func, n_models, batch_size,
                    use_fp16=use_fp16, micro_batch_size=micro_batch_size)
        cum = np.cumsum(tmp_pca.explained_variance_ratio_)
        n_components = min(int(np.searchsorted(cum, target_variance)) + 1, max_comp)
        print(f"Auto-selected {n_components} components → {cum[n_components-1]:.4%} variance")

    pca = BatchedCovariancePCA(n_components=n_components, device=device)
    pca.fit(model_loader_func, n_models, batch_size,
            use_fp16=use_fp16, micro_batch_size=micro_batch_size)
    output_file = pca.transform(model_loader_func, n_models, batch_size,
                                output_file, use_fp16=use_fp16)

    cum_var = np.cumsum(pca.explained_variance_ratio_)
    info = {
        "n_components": n_components,
        "explained_variance": pca.explained_variance_,
        "explained_variance_ratio": pca.explained_variance_ratio_,
        "cumulative_variance": cum_var,
        "total_variance_captured": float(np.sum(pca.explained_variance_ratio_)),
    }

    print(f"\nDone — {n_components}/{n_models} components, "
          f"{info['total_variance_captured']:.4%} variance captured")
    print(f"Output: {output_file}  shape=({n_models}, {n_components})")
    return output_file, pca, info
