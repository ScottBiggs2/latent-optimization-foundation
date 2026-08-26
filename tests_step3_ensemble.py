"""
Step 3 validation: EnsembleDataset + DualGramPCA.

The DualGramPCA checks use a small stand-in ensemble whose full data matrix fits in
memory, so every result can be compared against an explicit brute-force PCA. That is
the whole point: DualGramPCA never builds the (k, D) basis, so its correctness has to
be pinned against a reference that does.

    python tests_step3_ensemble.py            # math only, no model downloads
    python tests_step3_ensemble.py --tiny     # also exercise real EnsembleDataset
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
import tempfile
from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
import torch

from dual_gram_pca import DualGramPCA

FAILS: List[str] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f"  — {detail}" if detail else ""))
    if not ok:
        FAILS.append(name)


# ---------------------------------------------------------------------------
# A minimal stand-in with EnsembleDataset's loader interface
# ---------------------------------------------------------------------------

@dataclass
class _Stack:
    n_params: int
    n_layers: int = 2
    block_size: int = 0
    family_idx: int = 0
    weight_std: float = 1.0
    arch: str = "fake"


class _FakeEnsemble:
    """Holds an explicit (N, D) matrix and serves it through the chunk interface."""

    def __init__(self, X: np.ndarray, chunk: int, tmpdir: str, arch: str = "fake"):
        self.X = X.astype(np.float32)
        self.n_samples = X.shape[0]
        self._chunk = chunk
        self.arch_list = [arch]
        st = _Stack(n_params=X.shape[1], arch=arch)
        st.block_size = X.shape[1] // st.n_layers
        self.stacks = {arch: st}
        self._w0_path = os.path.join(tmpdir, f"{arch}_w0.npy")
        np.save(self._w0_path, self.X[0])

    def chunk_size(self, arch: str) -> int:
        return self._chunk

    def chunk_bounds(self, arch: str) -> List[Tuple[int, int]]:
        D = self.stacks[arch].n_params
        return [(r0, min(r0 + self._chunk, D)) for r0 in range(0, D, self._chunk)]

    def w0(self, arch: str) -> np.ndarray:
        return np.load(self._w0_path, mmap_mode="r")

    def load_chunk(self, arch, chunk_idx, device="cpu", w0_mm=None) -> torch.Tensor:
        r0, r1 = self.chunk_bounds(arch)[chunk_idx]
        return torch.from_numpy(np.ascontiguousarray(self.X[:, r0:r1])).to(device)


def brute_force(X: np.ndarray, k: int):
    """Explicit reference PCA. Returns (mean, components (k,D), codes (N,k))."""
    mean = X.mean(axis=0)
    Xc = X - mean
    C = Xc @ Xc.T
    ev, U = np.linalg.eigh(C)
    order = np.argsort(ev)[::-1]
    ev, U = ev[order], U[:, order]
    Uk, Sk = U[:, :k], ev[:k]
    comps = (Uk.T @ Xc) / np.sqrt(np.clip(Sk, 1e-300, None))[:, None]
    codes = Xc @ comps.T
    return mean, comps, codes


# ---------------------------------------------------------------------------
# DualGramPCA math
# ---------------------------------------------------------------------------

def test_gram_math(label: str, X: np.ndarray, k: int, chunk: int) -> None:
    print(f"\n--- {label}  (N={X.shape[0]}, D={X.shape[1]}, k={k}, chunk={chunk}) ---")
    N, D = X.shape
    tmp = tempfile.mkdtemp()
    try:
        ds = _FakeEnsemble(X, chunk, tmp)
        pca = DualGramPCA(n_components=k, device="cpu").fit(ds, "fake")
        kk = pca.n_components

        ref_mean, ref_comps, ref_codes = brute_force(X, kk)

        check(f"{label}: mean matches brute force",
              np.allclose(pca.mean_, ref_mean, rtol=1e-4, atol=1e-6),
              f"max|Δ|={np.max(np.abs(pca.mean_ - ref_mean)):.3g}")

        # Codes are defined up to a per-component sign.
        got = pca.codes(kk)
        cc = np.array([
            abs(np.dot(got[:, j], ref_codes[:, j]) /
                (np.linalg.norm(got[:, j]) * np.linalg.norm(ref_codes[:, j]) + 1e-30))
            for j in range(kk)
        ])
        check(f"{label}: codes == U*sqrt(S) matches explicit Xc @ V",
              np.all(cc > 0.9999), f"min|cos|={cc.min():.8f}")
        # Compare relative to the code SCALE. Codes are O(sqrt(S)), which is large
        # for raw weight data, so an absolute tolerance is meaningless here.
        scale = float(np.abs(ref_codes).max()) + 1e-30
        mag_err = float(np.max(np.abs(np.abs(got) - np.abs(ref_codes))) / scale)
        check(f"{label}: code magnitudes match", mag_err < 1e-4,
              f"max|Δ|/scale={mag_err:.3g} (scale={scale:.4g})")

        # Reconstruction of an ensemble member.
        i = min(3, N - 1)
        recon = pca.inverse_transform(pca.codes(kk)[i], ds, "fake", k=kk)
        ref_recon = ref_codes[i] @ ref_comps + ref_mean
        check(f"{label}: inverse_transform matches explicit basis reconstruction",
              np.allclose(recon, ref_recon, rtol=1e-3, atol=1e-5),
              f"max|Δ|={np.max(np.abs(recon - ref_recon)):.3g}")

        rel = np.linalg.norm(recon - X[i]) / (np.linalg.norm(X[i]) + 1e-30)
        if kk >= N - 1:
            check(f"{label}: exact round-trip at k=N-1", rel < 1e-5,
                  f"relative error={rel:.3g}")
        else:
            check(f"{label}: round-trip finite at k<N-1", np.isfinite(rel),
                  f"relative error={rel:.3g}")

        # Sample 0 is the real model, so its reconstruction is the one that matters.
        recon0 = pca.inverse_transform(pca.codes(kk)[0], ds, "fake", k=kk)
        rel0 = np.linalg.norm(recon0 - X[0]) / (np.linalg.norm(X[0]) + 1e-30)
        if kk >= N - 1:
            check(f"{label}: sample 0 (real model) exact at k=N-1", rel0 < 1e-5,
                  f"relative error={rel0:.3g}")

        # transform_vector on an ensemble member must recover that member's code.
        z0 = pca.transform_vector(X[0], ds, "fake", k=kk)
        check(f"{label}: transform_vector(X[0]) == codes[0]",
              np.allclose(z0, pca.codes(kk)[0], rtol=1e-2, atol=1e-3 * np.abs(pca.codes(kk)[0]).max()),
              f"max|Δ|={np.max(np.abs(z0 - pca.codes(kk)[0])):.3g}")

        # Chunk size must not change any result.
        ds2 = _FakeEnsemble(X, max(1, chunk // 3), tmp, arch="fake")
        pca2 = DualGramPCA(n_components=k, device="cpu").fit(ds2, "fake")
        check(f"{label}: result independent of chunk size",
              pca2.n_components == kk and
              np.allclose(pca2.gram_evals_, pca.gram_evals_, rtol=1e-6),
              f"k {pca2.n_components} vs {kk}")
    finally:
        shutil.rmtree(tmp)


def test_save_load() -> None:
    print("\n--- DualGramPCA save/load ---")
    rng = np.random.default_rng(5)
    N, D, k = 14, 900, 13
    X = rng.normal(0, 1, (N, D)).astype(np.float32)
    tmp = tempfile.mkdtemp()
    try:
        ds = _FakeEnsemble(X, 250, tmp)
        pca = DualGramPCA(n_components=k, device="cpu").fit(ds, "fake")
        sub = os.path.join(tmp, "saved")
        pca.save(sub)

        loaded = DualGramPCA.load(sub, device="cpu")
        check("k preserved", loaded.n_components == pca.n_components,
              f"{loaded.n_components} vs {pca.n_components}")
        check("codes preserved", np.array_equal(loaded.codes_, pca.codes_))
        check("eigenbasis preserved", np.allclose(loaded.gram_evecs_, pca.gram_evecs_))
        check("mean preserved", np.allclose(np.asarray(loaded.mean_), pca.mean_))
        r1 = pca.inverse_transform(pca.codes()[2], ds, "fake")
        r2 = loaded.inverse_transform(loaded.codes()[2], ds, "fake")
        check("reconstruction identical after reload", np.allclose(r1, r2, atol=1e-6),
              f"max|Δ|={np.max(np.abs(r1 - r2)):.3g}")

        # The whole point: no (k, D) basis on disk.
        check("no components matrix written",
              not os.path.exists(os.path.join(sub, "components.npy")))
    finally:
        shutil.rmtree(tmp)


def test_noise_ensemble_analytic() -> None:
    """The plan's four analytic predictions, on a w0 + noise ensemble."""
    print("\n--- noise ensemble: analytic predictions ---")
    rng = np.random.default_rng(0)
    N, D, s = 40, 30000, 1e-2
    w0 = rng.normal(0, 0.02, D).astype(np.float32)
    sigma = float(w0.std())
    X = np.empty((N, D), dtype=np.float32)
    X[0] = w0
    X[1:] = w0 + s * sigma * rng.normal(0, 1, (N - 1, D)).astype(np.float32)

    tmp = tempfile.mkdtemp()
    try:
        ds = _FakeEnsemble(X, 8000, tmp)
        pca = DualGramPCA(n_components=N - 1, device="cpu").fit(ds, "fake")
        k = pca.n_components
        check("(0) full rank retained", k == N - 1, f"k={k} of {N - 1}")

        # Flatness holds over the WELL-CONDITIONED directions. Sample 0 carries no
        # noise, so only N-1 members contribute noise and centering removes one dof:
        # ~N-2 directions are flat and the last is much weaker. See the
        # "Consequences of making sample 0 noise-free" note in dual_gram_pca.
        ev = pca.explained_variance_
        ratio_wc = float(ev[0] / ev[k - 2])
        check("(1) spectrum flat over well-conditioned directions", ratio_wc < 3.0,
              f"ev[0]/ev[{k - 2}]={ratio_wc:.3f}")
        check("(1b) final direction is the weak one (sample-0 artefact)",
              float(ev[0] / ev[-1]) > ratio_wc,
              f"ev[0]/ev[{k - 1}]={float(ev[0] / ev[-1]):.3f} vs "
              f"ev[0]/ev[{k - 2}]={ratio_wc:.3f}")

        evr = float(np.sum(pca.explained_variance_ratio_))
        check("(2a) EVR ~1.00 at k=N-1", abs(evr - 1.0) < 0.02, f"EVR={evr:.4f}")
        half = N // 2
        evr_h = float(np.sum(pca.explained_variance_ratio_[:half]))
        check("(2b) EVR ~0.50 at k=N/2", abs(evr_h - 0.5) < 0.08, f"EVR={evr_h:.4f}")

        r_full = pca.inverse_transform(pca.codes()[0], ds, "fake", k=k)
        rel = np.linalg.norm(r_full - w0) / np.linalg.norm(w0)
        check("(3) real model exact at k=N-1", rel < 1e-5, f"relative error={rel:.3g}")

        # (4) applies to a TYPICAL member. s*sigma/sqrt(2) is the residual left when
        # half the (isotropic) variance is discarded.
        j = 5
        r_typ = pca.inverse_transform(pca.codes()[j][:half], ds, "fake", k=half)
        resid_typ = float(np.std(r_typ - X[j]))
        predicted = s * sigma / np.sqrt(2.0)
        check("(4) k=N/2 residual std ~ s*sigma/sqrt(2) for a typical member",
              0.5 * predicted < resid_typ < 1.8 * predicted,
              f"measured={resid_typ:.5g}, predicted={predicted:.5g}")

        # Sample 0 sits ~sqrt(N) closer to the mean, so its residual is smaller by
        # about that factor. Checking it separately keeps the two straight.
        r_half0 = pca.inverse_transform(pca.codes()[0][:half], ds, "fake", k=half)
        resid0 = float(np.std(r_half0 - w0))
        predicted0 = s * sigma / np.sqrt(2.0 * N)
        check("(4b) sample 0 residual ~ s*sigma/sqrt(2N) (nearer the mean)",
              0.4 * predicted0 < resid0 < 2.5 * predicted0,
              f"measured={resid0:.5g}, predicted={predicted0:.5g}")
    finally:
        shutil.rmtree(tmp)


# ---------------------------------------------------------------------------
# The real EnsembleDataset (needs transformers)
# ---------------------------------------------------------------------------

def test_real_ensemble_dataset() -> None:
    print("\n--- EnsembleDataset (tiny mode) ---")
    from data.ensemble_dataset import EnsembleDataset

    tmp = tempfile.mkdtemp()
    try:
        ds = EnsembleDataset(
            arch_list=["gpt2_medium", "smollm2_360m", "pythia_410m"],
            n_samples=8, noise_scale=1e-2, exclude_1d=True,
            mode="tiny", artifact_dir=tmp, seed=7,
            chunk_budget_bytes=1 << 20,
        )
        print(ds.summary())

        for arch in ds.arch_list:
            st = ds.stacks[arch]
            check(f"{arch}: D == L * block_size",
                  st.n_params == st.n_layers * st.block_size,
                  f"{st.n_params} vs {st.n_layers}*{st.block_size}")

            bounds = ds.chunk_bounds(arch)
            covered = sum(r1 - r0 for r0, r1 in bounds)
            contiguous = all(bounds[i][1] == bounds[i + 1][0]
                             for i in range(len(bounds) - 1))
            check(f"{arch}: chunk grid tiles D exactly",
                  covered == st.n_params and contiguous and bounds[0][0] == 0,
                  f"covered {covered} of {st.n_params} in {len(bounds)} chunks")

            # Determinism across passes -- the property the whole no-disk scheme
            # rests on. If this fails, the Gram and mean passes disagree.
            a = ds.load_chunk(arch, 0)
            b = ds.load_chunk(arch, 0)
            check(f"{arch}: load_chunk deterministic across calls",
                  torch.equal(a, b))

            # Sample 0 must be the real model, bit-exactly.
            w0 = ds.w0(arch)
            r0, r1 = bounds[0]
            check(f"{arch}: sample 0 is w0 exactly",
                  np.array_equal(a[0].cpu().numpy(), np.asarray(w0[r0:r1])))

            # Noise magnitude should match the requested scale.
            spread = float((a[1:] - a[0]).std())
            want = 1e-2 * st.weight_std
            check(f"{arch}: noise std matches noise_scale * weight_std",
                  0.7 * want < spread < 1.4 * want,
                  f"measured={spread:.4g}, want={want:.4g}")

            # Block slicing must line up with the concatenation order.
            lo, hi = st.block_slice(1)
            check(f"{arch}: block_slice(1) is the second block",
                  (lo, hi) == (st.block_size, 2 * st.block_size),
                  f"({lo}, {hi})")

        # Cache-invalidation: reusing the dir with different ensemble params must
        # refuse rather than silently serve a different ensemble.
        raised = False
        try:
            EnsembleDataset(arch_list=["gpt2_medium"], n_samples=8,
                            noise_scale=5e-2, exclude_1d=True, mode="tiny",
                            artifact_dir=tmp, seed=7)
        except RuntimeError as exc:
            raised = "noise_scale" in str(exc)
        check("changing noise_scale invalidates the cached ensemble", raised)
    finally:
        shutil.rmtree(tmp)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tiny", action="store_true",
                    help="also run the real EnsembleDataset (imports transformers)")
    args = ap.parse_args()

    rng = np.random.default_rng(42)

    # Chunk count > 1 in every case, so the streaming paths are actually exercised.
    X = rng.normal(0, 1, (16, 2000)).astype(np.float32)
    test_gram_math("generic, k=N-1", X, 15, 512)
    test_gram_math("generic, k=N/2", X, 8, 512)
    test_gram_math("generic, D not a multiple of chunk", X, 15, 700)

    test_noise_ensemble_analytic()
    test_save_load()

    if args.tiny:
        test_real_ensemble_dataset()
    else:
        print("\n(skipping real EnsembleDataset — pass --tiny to include it)")

    print("\n" + "=" * 64)
    if FAILS:
        print(f"FAILED ({len(FAILS)}): " + "; ".join(FAILS))
        return 1
    print("ALL CHECKS PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
