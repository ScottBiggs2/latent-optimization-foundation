"""
DualGramPCA — dual (Gram-matrix) PCA that never materialises the component basis.

Why the basis is never materialised
----------------------------------
The textbook formulation (and this repo's removed block-era `BatchedCovariancePCA`)
implements the same math but ends by building an explicit `(k, n_params)` component
matrix. That is fine when a sample is one transformer block (k=97, D=15.7M -> 6.1 GB)
and impossible when a sample is a whole decoder stack:

    k=99, D=302M  ->  99 * 302e6 * 4 B = 119.6 GB   per architecture

The dual formulation exists precisely so that matrix is never needed. Everything
downstream can be expressed through the n x n Gram eigendecomposition plus one
streaming pass over the ensemble.

The identities
--------------
Let Xc be the centered ensemble, shape (N, D), and C = Xc Xc^T its (N, N) Gram
matrix with eigendecomposition C = U S U^T (descending). The unit-norm principal
components are the rows of

    V_k = diag(S_k^-1/2) U_k^T Xc                                            (k, D)

which is never formed. Then:

  * **Codes come free.**  Z = Xc V_k^T = U_k S_k^1/2.  No pass over the data at all.
    (Verified against an explicit Xc @ components to 1.2e-15.)

  * **Reconstruction is a weighted sum of ensemble rows.**  For any code z,

        x_hat = mean + Xc^T a       with   a = U_k (z / sqrt(S_k))            (N,)

    so one streaming pass suffices. For a training row i this reduces to
    a = U_k U_k[i]^T, i.e. row i of the rank-k projector, as it must.

  * **Projecting an out-of-ensemble vector** x needs one pass to form
    t = Xc (x - mean), then z = (U_k^T t) / sqrt(S_k).

Passes over the data: **two** (mean, then centered Gram). The mean is needed before
the Gram can be centered. Accumulating an uncentered Gram and correcting afterwards
would need only one pass, but for an ensemble whose spread is ~1% of its magnitude
that correction is catastrophic cancellation, so it is not worth the saved pass.

Consequences of making sample 0 noise-free
------------------------------------------
EnsembleDataset sets sample 0 = w_0 exactly (no noise) so the evaluation target is a
genuine ensemble member. That is deliberate, and it has two measurable side effects
worth knowing before reading a spectrum or a residual:

  1. **The last eigen-direction is weakly determined.** Only N-1 members carry noise,
     and centering removes one degree of freedom, so ~N-2 directions are well
     conditioned and the final one is much smaller. Measured at N=40: ev[0]/ev[N-2]
     is flat (~1.2) while ev[0]/ev[N-1] is ~43. Reconstruction at k=N-1 is still
     exact; the weak direction is the one separating the real model from the noisy
     ones, which is signal, not roundoff.
  2. **Sample 0 sits near the ensemble mean.** mean ~= w_0 + O(s*sigma/sqrt(N)), so
     ||X[0] - mean|| is ~sqrt(N) smaller than for a typical member. Truncation
     residuals measured on sample 0 are correspondingly smaller than the
     s*sigma/sqrt(2) that applies to a typical member -- roughly s*sigma/sqrt(2N).
     Compare like with like when reading a rank sweep.

Numerics
--------
The Gram is accumulated in float64 and the eigendecomposition uses `eigh` (C is
symmetric), with a relative rank floor. See DEFAULT_RANK_RTOL below for why the
floor is where it is — briefly, forming C squares the condition number of the data,
so eigenvalues below ~fp32 epsilon are roundoff whose eigenVECTORS are meaningless
even when their magnitudes look plausible.
"""

from __future__ import annotations

import json
import os
from typing import Optional

import numpy as np
import torch

# Eigenvalues below DEFAULT_RANK_RTOL * max_eigenvalue are treated as numerical noise
# and dropped.  This is not a knob to loosen casually.
#
# The dual trick forms C = Xc^T Xc, which squares the condition number: a singular
# value ratio of r shows up as an eigenvalue ratio of r^2. Weights are extracted as
# float32 (eps ~ 1.2e-7), so singular values below ~1e-4 of the leading one are already
# at the input's noise level, which puts the corresponding EIGENVALUE floor at
# ~1e-8..1e-7. 1e-7 is the conservative end of that range.
#
# Setting this lower does not recover more signal -- it admits eigenvectors whose
# directions are pure roundoff, which then get normalised to unit length and given
# equal footing with the real components in transform()/inverse_transform(). That was
# the concrete failure mode of the old randomized_svd path.
DEFAULT_RANK_RTOL = 1e-7

GRAM_LAYOUT_VERSION = 1


# ---------------------------------------------------------------------------
# Bulk spectrum statistics
# ---------------------------------------------------------------------------
#
# Lifted out of scripts/train_stack.py so there is ONE implementation. It is
# needed by anything that computes a spectrum, and scripts/diag_block_spectrum.py
# computes one over a sub-range of D without fitting a PCA at all. NOT put in
# scripts/report_stack.py: that is pure stdlib by design and
# tests/test_report.py::test_stdlib_purity enforces it.
def spectrum_stats(evals: np.ndarray, k: int) -> dict:
    """
    Flatness of the retained spectrum, measured three ways.

    `ev0_over_evlast` is the ratio this repo reported first, and on its own it
    MISLEADS at k = N-1. Measured on a pure-noise ensemble (N=12): 12.09 at k=N-1 but
    1.006 at k=N/2, for the same data. At the rank bound `ev[k-1]` is the smallest
    numerically-marginal direction left standing after the rank floor, so the ratio
    describes the tail rather than the bulk and reads "steep" for an ensemble that is
    flat by construction. Same trap as `total_variance_captured` being vacuous at
    k = N-1 (misstep 15).

    So two bulk statistics are recorded alongside it, and they are what anything
    downstream should gate on:

    ev0_over_median
        The leading direction's variance as a multiple of the TYPICAL direction's.
        ~1 for isotropic noise, large when a few directions carry the energy.

    effective_rank_ratio
        (sum ev)^2 / (sum ev^2) / k -- the participation ratio, normalised to [0, 1].
        1.0 means every retained direction carries equal variance (perfectly flat);
        small means the variance is concentrated. Unlike any ratio of two individual
        eigenvalues, this cannot be moved by one marginal direction at the tail.
    """
    ev = np.asarray(evals, dtype=np.float64)[:max(int(k), 1)]
    ev = np.clip(ev, 1e-300, None)
    med = float(np.median(ev))
    eff = float((ev.sum() ** 2) / max(float((ev ** 2).sum()), 1e-300))
    return {
        "ev0_over_evlast": float(ev[0] / ev[-1]),
        "ev0_over_median": float(ev[0] / max(med, 1e-300)),
        "effective_rank_ratio": eff / max(len(ev), 1),
        "effective_rank": eff,
    }

class DualGramPCA:
    """
    Dual PCA over an EnsembleDataset architecture, with no explicit basis.

    Parameters
    ----------
    n_components : requested k. Capped at N-1 (the rank bound) and further reduced
                   if the spectrum is numerically rank-deficient.
    rank_rtol    : eigenvalues below this fraction of the leading one are dropped.
    """

    def __init__(
        self,
        n_components: int,
        rank_rtol: float = DEFAULT_RANK_RTOL,
        device: Optional[str] = None,
    ):
        self.n_components = int(n_components)
        self.rank_rtol = float(rank_rtol)
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        self.arch: Optional[str] = None
        self.n_samples_: Optional[int] = None
        self.n_params_: Optional[int] = None
        self.mean_: Optional[np.ndarray] = None        # (D,) float32
        self.gram_evals_: Optional[np.ndarray] = None  # (k,) float64, descending
        self.gram_evecs_: Optional[np.ndarray] = None  # (N, k) float64
        self.codes_: Optional[np.ndarray] = None       # (N, k) float32
        self.explained_variance_: Optional[np.ndarray] = None
        self.explained_variance_ratio_: Optional[np.ndarray] = None
        self.total_variance_: Optional[float] = None

    # ------------------------------------------------------------------
    # Fit
    # ------------------------------------------------------------------

    def fit(self, dataset, arch: str) -> "DualGramPCA":
        """Fit on `dataset`'s ensemble for `arch`. Two streaming passes."""
        st = dataset.stacks[arch]
        N, D = dataset.n_samples, st.n_params
        bounds = dataset.chunk_bounds(arch)
        dev = torch.device(self.device)
        w0_mm = dataset.w0(arch)

        print(f"[DualGramPCA] {arch}: N={N}, D={D:,}, {len(bounds)} chunks "
              f"of <= {dataset.chunk_size(arch):,} on {dev}")

        # ---- Pass 1: mean ------------------------------------------------
        mean = np.empty(D, dtype=np.float32)
        for ci, (r0, r1) in enumerate(bounds):
            B = dataset.load_chunk(arch, ci, device=dev, w0_mm=w0_mm)
            mean[r0:r1] = B.mean(dim=0).float().cpu().numpy()
            del B
            if ci % 50 == 0:
                print(f"  mean  chunk {ci + 1}/{len(bounds)}")
        if dev.type == "cuda":
            torch.cuda.empty_cache()

        # ---- Pass 2: centered Gram in float64 ----------------------------
        C = np.zeros((N, N), dtype=np.float64)
        for ci, (r0, r1) in enumerate(bounds):
            B = dataset.load_chunk(arch, ci, device=dev, w0_mm=w0_mm)
            m = torch.from_numpy(mean[r0:r1]).to(dev)
            Bc = (B - m.unsqueeze(0)).double()
            C += (Bc @ Bc.T).cpu().numpy()
            del B, Bc, m
            if ci % 50 == 0:
                print(f"  gram  chunk {ci + 1}/{len(bounds)}")
        if dev.type == "cuda":
            torch.cuda.empty_cache()

        self.total_variance_ = float(np.trace(C)) / (N - 1)

        # ---- Eigendecomposition -----------------------------------------
        evals, evecs = np.linalg.eigh(C)
        order = np.argsort(evals)[::-1]
        evals, evecs = evals[order], evecs[:, order]

        lead = float(evals[0]) if evals.size else 0.0
        n_valid = int(np.sum(evals > max(lead * self.rank_rtol, 0.0)))
        k = min(self.n_components, N - 1, max(n_valid, 1))
        if k < min(self.n_components, N - 1):
            print(f"  NOTE: only {n_valid} of {min(self.n_components, N - 1)} "
                  f"requested directions clear the {self.rank_rtol:g} rank floor "
                  f"(lead eigenvalue {lead:.6g}); keeping {k}. For a noise ensemble "
                  f"this usually means noise_scale is too small to separate the "
                  f"members above float32 precision.")
        self.n_components = k

        self.gram_evals_ = evals[:k].copy()
        self.gram_evecs_ = np.ascontiguousarray(evecs[:, :k])
        self.mean_ = mean
        self.arch = arch
        self.n_samples_ = N
        self.n_params_ = D

        # Codes for free — no pass over the data.
        self.codes_ = (self.gram_evecs_
                       * np.sqrt(np.clip(self.gram_evals_, 0.0, None))).astype(np.float32)

        self.explained_variance_ = self.gram_evals_ / (N - 1)
        tv = self.total_variance_ if self.total_variance_ else 1.0
        self.explained_variance_ratio_ = self.explained_variance_ / tv

        spread = float(self.gram_evals_[0] / max(self.gram_evals_[-1], 1e-300)) if k > 1 else 1.0
        print(f"  fitted k={k}/{N - 1}  "
              f"variance captured={np.sum(self.explained_variance_ratio_):.4%}  "
              f"ev[0]/ev[{k - 1}]={spread:.4g}")
        return self

    # ------------------------------------------------------------------
    # Transform / inverse
    # ------------------------------------------------------------------

    def _sqrt_S(self, k: int) -> np.ndarray:
        return np.sqrt(np.clip(self.gram_evals_[:k], 1e-300, None))

    def codes(self, k: Optional[int] = None) -> np.ndarray:
        """Training codes (N, k). A k-prefix IS the rank-k PCA (variance-ordered)."""
        k = self.n_components if k is None else min(k, self.n_components)
        return self.codes_[:, :k]

    def weights_for_code(self, z: np.ndarray, k: Optional[int] = None) -> np.ndarray:
        """
        Ensemble mixing weights a (N,) such that  x_hat = mean + Xc^T a.

        This is the whole trick: a rank-k reconstruction is a linear combination of
        the ensemble members, so no explicit basis is ever needed.
        """
        k = self.n_components if k is None else min(k, self.n_components)
        z = np.asarray(z, dtype=np.float64).reshape(-1)[:k]
        return self.gram_evecs_[:, :k] @ (z / self._sqrt_S(k))

    def inverse_transform(
        self,
        z: np.ndarray,
        dataset,
        arch: Optional[str] = None,
        k: Optional[int] = None,
        out: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        Reconstruct a full weight vector (D,) from a code. One streaming pass.

        x_hat = mean * (1 - sum(a)) + sum_j a_j * X_j,  a = weights_for_code(z)
        """
        arch = arch or self.arch
        a = self.weights_for_code(z, k=k)
        a_sum = float(a.sum())
        dev = torch.device(self.device)
        a_t = torch.from_numpy(a).to(dev, dtype=torch.float32)

        D = self.n_params_
        if out is None:
            out = np.empty(D, dtype=np.float32)
        w0_mm = dataset.w0(arch)

        for ci, (r0, r1) in enumerate(dataset.chunk_bounds(arch)):
            B = dataset.load_chunk(arch, ci, device=dev, w0_mm=w0_mm)
            m = torch.from_numpy(np.array(self.mean_[r0:r1])).to(dev)
            chunk = m * (1.0 - a_sum) + (a_t @ B)
            out[r0:r1] = chunk.float().cpu().numpy()
            del B, m, chunk
        if dev.type == "cuda":
            torch.cuda.empty_cache()
        return out

    def inverse_transform_many(
        self,
        Z: np.ndarray,
        dataset,
        arch: Optional[str] = None,
        k: Optional[int] = None,
        out: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        Reconstruct a WHOLE COHORT of codes (n, k) -> (n, D) in ONE streaming pass.

        Identical arithmetic to inverse_transform, with `a` promoted from (N,) to
        (n, N) so the per-chunk contraction becomes a matmul instead of a matvec.

        This is not a micro-optimisation. inverse_transform walks every member of the
        ensemble for every code it decodes -- at Mini that is 100 x 206 MB = 20.6 GB
        of reads PER GENERATED MODEL. A retrieval sweep decoding a few dozen models
        would move terabytes and look hung rather than failed. Batched, the whole
        cohort costs one pass, and the extra memory is only n x D x 4 bytes for the
        output plus one chunk of working set.

        Cap `n` by the caller's memory budget: at Mini, 32 rows of D=51.5M is 6.6 GB.
        """
        arch = arch or self.arch
        Z = np.atleast_2d(np.asarray(Z))
        A = np.stack([self.weights_for_code(z, k=k) for z in Z])   # (n, N)
        a_sum = A.sum(axis=1)                                      # (n,)
        dev = torch.device(self.device)
        A_t = torch.from_numpy(A).to(dev, dtype=torch.float32)
        s_t = torch.from_numpy(1.0 - a_sum).to(dev, dtype=torch.float32).reshape(-1, 1)

        n, D = A.shape[0], self.n_params_
        if out is None:
            out = np.empty((n, D), dtype=np.float32)
        w0_mm = dataset.w0(arch)

        for ci, (r0, r1) in enumerate(dataset.chunk_bounds(arch)):
            B = dataset.load_chunk(arch, ci, device=dev, w0_mm=w0_mm)   # (N, chunk)
            m = torch.from_numpy(np.array(self.mean_[r0:r1])).to(dev)   # (chunk,)
            chunk = s_t * m.reshape(1, -1) + (A_t @ B)                  # (n, chunk)
            out[:, r0:r1] = chunk.float().cpu().numpy()
            del B, m, chunk
        if dev.type == "cuda":
            torch.cuda.empty_cache()
        return out

    def transform_vector(
        self,
        x: np.ndarray,
        dataset,
        arch: Optional[str] = None,
        k: Optional[int] = None,
    ) -> np.ndarray:
        """
        Project an arbitrary (D,) vector into code space. One streaming pass.

        Only needed for weights that are NOT ensemble members — for member i the
        code is already codes()[i].
        """
        arch = arch or self.arch
        k = self.n_components if k is None else min(k, self.n_components)
        dev = torch.device(self.device)
        t = torch.zeros(self.n_samples_, dtype=torch.float64, device=dev)
        w0_mm = dataset.w0(arch)

        for ci, (r0, r1) in enumerate(dataset.chunk_bounds(arch)):
            B = dataset.load_chunk(arch, ci, device=dev, w0_mm=w0_mm)
            m = torch.from_numpy(np.array(self.mean_[r0:r1])).to(dev)
            xs = torch.from_numpy(np.ascontiguousarray(x[r0:r1])).to(dev)
            t += ((B - m.unsqueeze(0)).double() @ (xs - m).double())
            del B, m, xs
        if dev.type == "cuda":
            torch.cuda.empty_cache()

        t_np = t.cpu().numpy()
        return ((self.gram_evecs_[:, :k].T @ t_np) / self._sqrt_S(k)).astype(np.float32)

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, save_dir: str) -> None:
        os.makedirs(save_dir, exist_ok=True)
        np.save(os.path.join(save_dir, "mean.npy"), self.mean_)
        np.save(os.path.join(save_dir, "gram_evals.npy"), self.gram_evals_)
        np.save(os.path.join(save_dir, "gram_evecs.npy"), self.gram_evecs_)
        np.save(os.path.join(save_dir, "codes.npy"), self.codes_)
        meta = {
            "layout_version": GRAM_LAYOUT_VERSION,
            "arch": self.arch,
            "n_components": int(self.n_components),
            "n_samples": int(self.n_samples_),
            "n_params": int(self.n_params_),
            "rank_rtol": self.rank_rtol,
            "total_variance": self.total_variance_,
            "total_variance_captured": float(np.sum(self.explained_variance_ratio_)),
            # Explicitly recorded: at k = N-1 all non-null directions are retained, so
            # this ratio is ~1.0 by construction and says nothing about truncation.
            "variance_captured_is_vacuous_at_k_eq_N_minus_1":
                bool(self.n_components >= (self.n_samples_ or 1) - 1),
        }
        with open(os.path.join(save_dir, "gram_pca_meta.json"), "w") as f:
            json.dump(meta, f, indent=2)
        mb = self.mean_.nbytes / 1e6
        print(f"[DualGramPCA] saved {self.arch} → {save_dir}  "
              f"(mean {mb:.0f} MB + {self.gram_evecs_.shape} eigenbasis; "
              f"an explicit ({self.n_components}, {self.n_params_:,}) component "
              f"matrix would have been "
              f"{self.n_components * self.n_params_ * 4 / 1e9:.1f} GB)")

    @classmethod
    def load(cls, save_dir: str, device: Optional[str] = None) -> "DualGramPCA":
        with open(os.path.join(save_dir, "gram_pca_meta.json")) as f:
            meta = json.load(f)
        found = meta.get("layout_version")
        if found != GRAM_LAYOUT_VERSION:
            raise RuntimeError(
                f"Refusing to load {save_dir}: layout_version={found!r} but this code "
                f"expects {GRAM_LAYOUT_VERSION}. Re-fit."
            )
        obj = cls(n_components=meta["n_components"],
                  rank_rtol=meta.get("rank_rtol", DEFAULT_RANK_RTOL),
                  device=device)
        obj.arch = meta["arch"]
        obj.n_samples_ = meta["n_samples"]
        obj.n_params_ = meta["n_params"]
        obj.total_variance_ = meta.get("total_variance")
        obj.mean_ = np.load(os.path.join(save_dir, "mean.npy"), mmap_mode="r")
        obj.gram_evals_ = np.load(os.path.join(save_dir, "gram_evals.npy"))
        obj.gram_evecs_ = np.load(os.path.join(save_dir, "gram_evecs.npy"))
        obj.codes_ = np.load(os.path.join(save_dir, "codes.npy"))
        obj.explained_variance_ = obj.gram_evals_ / (obj.n_samples_ - 1)
        tv = obj.total_variance_ or 1.0
        obj.explained_variance_ratio_ = obj.explained_variance_ / tv
        print(f"[DualGramPCA] loaded {obj.arch} from {save_dir} "
              f"(k={obj.n_components}, N={obj.n_samples_}, D={obj.n_params_:,})")
        return obj
