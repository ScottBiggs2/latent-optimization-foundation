"""
Step 1 validation: dual_pca.py numerical correctness after the eigh / float64 /
fp32-storage changes.

Runs entirely on synthetic data — no model downloads, no GPU required. Every check
prints PASS/FAIL and the script exits non-zero if anything fails.

    python tests_step1_dual_pca.py
"""

from __future__ import annotations

import os
import shutil
import sys
import tempfile

import numpy as np

from dual_pca import BatchedCovariancePCA, load_codes

FAILS: list[str] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f"  — {detail}" if detail else ""))
    if not ok:
        FAILS.append(name)


def make_loader(X: np.ndarray):
    """X is (n_models, n_params). dual_pca wants (n_params, batch)."""
    def loader(s: int, e: int) -> np.ndarray:
        return np.ascontiguousarray(X[s:e, :].T)
    return loader


def exact_pca(X: np.ndarray, k: int):
    """Reference PCA via SVD of the centered data. Returns (mean, components, evals)."""
    mean = X.mean(axis=0)
    Xc = X - mean
    # Xc is (n, D); rows are samples
    U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
    return mean, Vt[:k], (S[:k] ** 2) / (X.shape[0] - 1)


# ---------------------------------------------------------------------------
# 1. Correctness against an exact reference, flat + steep spectra
# ---------------------------------------------------------------------------

def test_against_exact(label: str, X: np.ndarray, k: int) -> None:
    print(f"\n--- {label}  (n={X.shape[0]}, D={X.shape[1]}, k={k}) ---")
    n = X.shape[0]
    requested = k

    pca = BatchedCovariancePCA(n_components=k, device="cpu")
    pca.fit(make_loader(X), n_models=n, batch_size=max(2, n // 4))

    ref_mean, ref_comps, ref_evals = exact_pca(X, k)

    check(f"{label}: mean matches",
          np.allclose(pca.mean_, ref_mean, rtol=1e-4, atol=1e-6),
          f"max|Δ|={np.max(np.abs(pca.mean_ - ref_mean)):.3g}")

    # Components are only defined up to sign; compare subspaces via |cos|.
    got = pca.components_
    kk = got.shape[0]
    cos = np.abs(np.sum(got * ref_comps[:kk], axis=1))
    check(f"{label}: component directions match reference",
          np.all(cos > 0.999),
          f"min|cos|={cos.min():.6f} over {kk} components")

    check(f"{label}: eigenvalues match reference",
          np.allclose(pca.explained_variance_, ref_evals[:kk], rtol=1e-3),
          f"max rel Δ={np.max(np.abs(pca.explained_variance_ - ref_evals[:kk]) / (ref_evals[0] + 1e-30)):.3g}")

    # Orthonormality — this is what randomized_svd was breaking at k -> n-1.
    G = got @ got.T
    ortho_err = np.max(np.abs(G - np.eye(kk)))
    check(f"{label}: components orthonormal", ortho_err < 1e-3,
          f"max|GramG − I|={ortho_err:.3g}")

    # ---- the identity Step 3 depends on: codes == U * sqrt(S) ----
    codes_file = pca.transform(make_loader(X), n_models=n, batch_size=max(2, n // 4))
    codes = load_codes(codes_file, n_models=n, n_components=pca.n_components)
    os.unlink(codes_file)

    # Recover U, S from the Gram matrix the same way fit() does.
    Xc = X - pca.mean_
    C = Xc @ Xc.T
    ev, evec = np.linalg.eigh(C)
    order = np.argsort(ev)[::-1]
    ev, evec = ev[order], evec[:, order]
    codes_dual = evec[:, :kk] * np.sqrt(np.clip(ev[:kk], 0, None))

    # Also sign-invariant: compare column magnitudes and per-column correlation.
    col_cos = np.array([
        abs(np.dot(codes[:, j], codes_dual[:, j]) /
            (np.linalg.norm(codes[:, j]) * np.linalg.norm(codes_dual[:, j]) + 1e-30))
        for j in range(kk)
    ])
    check(f"{label}: codes == U*sqrt(S) (dual identity)",
          np.all(col_cos > 0.999),
          f"min|cos| per code dim={col_cos.min():.6f}")

    # ---- round-trip ----
    recon = pca.inverse_transform(codes)
    rel = np.linalg.norm(recon - X) / np.linalg.norm(X)
    if kk >= n - 1:
        check(f"{label}: round-trip exact at k=n-1", rel < 1e-4,
              f"relative error={rel:.3g}")
    else:
        check(f"{label}: round-trip finite at k<n-1", np.isfinite(rel),
              f"relative error={rel:.3g}")

    # ---- explained variance ratio ----
    # Two distinct reasons kk can be < n-1: we ASKED for fewer (real truncation, so
    # EVR must drop below 1) or the rank floor cut it (the discarded directions were
    # numerically null, so EVR legitimately stays ~1). Only the first is a
    # meaningful assertion.
    evr = float(np.sum(pca.explained_variance_ratio_))
    rank_limited = kk < requested
    if rank_limited:
        check(f"{label}: EVR ~1.0 (rank-limited, discarded directions were null)",
              evr > 0.98, f"EVR={evr:.4f}, kept {kk} of {requested} requested")
    elif kk >= n - 1:
        check(f"{label}: EVR ~1.0 at k=n-1", abs(evr - 1.0) < 0.02, f"EVR={evr:.4f}")
    else:
        check(f"{label}: EVR < 1.0 at k<n-1", evr < 0.999, f"EVR={evr:.4f}")

    return pca


# ---------------------------------------------------------------------------
# 2. The noise-ensemble predictions from the plan
# ---------------------------------------------------------------------------

def test_noise_ensemble() -> None:
    print("\n--- noise ensemble: analytic predictions (plan Verification 1-4) ---")
    rng = np.random.default_rng(0)
    n, D, s = 40, 20000, 1e-2
    w0 = rng.normal(0, 1.0, D).astype(np.float32)
    sigma = float(w0.std())
    X = (w0[None, :] + s * sigma * rng.normal(0, 1, (n, D))).astype(np.float32)

    k_full = n - 1
    pca = BatchedCovariancePCA(n_components=k_full, device="cpu")
    pca.fit(make_loader(X), n_models=n, batch_size=10)

    ev = pca.explained_variance_
    ratio = float(ev[0] / ev[-1])
    # For an isotropic Gaussian ensemble the Gram spectrum is Marchenko-Pastur-ish;
    # with D >> n it concentrates tightly around a constant.
    check("(1) spectrum is flat (not steep)", ratio < 3.0,
          f"ev[0]/ev[{k_full-1}]={ratio:.3f}")

    evr_full = float(np.sum(pca.explained_variance_ratio_))
    check("(2a) EVR ~1.00 at k=N-1", abs(evr_full - 1.0) < 0.02, f"EVR={evr_full:.4f}")

    k_half = n // 2
    evr_half = float(np.sum(pca.explained_variance_ratio_[:k_half]))
    check("(2b) EVR ~0.50 at k=N/2", abs(evr_half - 0.5) < 0.08, f"EVR={evr_half:.4f}")

    # (3) exact reconstruction at k=N-1
    codes_file = pca.transform(make_loader(X), n_models=n, batch_size=10)
    codes = load_codes(codes_file, n_models=n, n_components=pca.n_components)
    os.unlink(codes_file)
    recon_full = pca.inverse_transform(codes)
    rel_full = np.linalg.norm(recon_full - X) / np.linalg.norm(X)
    check("(3) k=N-1 reconstructs training samples exactly", rel_full < 1e-4,
          f"relative error={rel_full:.3g}")

    # (4) residual at k=N/2 is ~ s*sigma/sqrt(2)
    recon_half = (codes[:, :k_half] @ pca.components_[:k_half]) + pca.mean_
    resid = float(np.std(recon_half - X))
    predicted = s * sigma / np.sqrt(2.0)
    check("(4) k=N/2 residual std ~ s*sigma/sqrt(2)",
          0.6 * predicted < resid < 1.6 * predicted,
          f"measured={resid:.5g}, predicted={predicted:.5g}")


# ---------------------------------------------------------------------------
# 3. Storage precision + save/load round-trip
# ---------------------------------------------------------------------------

def test_save_load() -> None:
    print("\n--- save/load: fp32 storage, no precision loss ---")
    rng = np.random.default_rng(3)
    n, D = 12, 5000
    X = rng.normal(0, 1, (n, D)).astype(np.float32)
    pca = BatchedCovariancePCA(n_components=n - 1, device="cpu")
    pca.fit(make_loader(X), n_models=n, batch_size=4)

    tmp = tempfile.mkdtemp()
    try:
        pca.save(tmp)
        on_disk = np.load(os.path.join(tmp, "components.npy"))
        check("stored components are float32", on_disk.dtype == np.float32,
              f"dtype={on_disk.dtype}")
        check("stored components bit-identical to in-memory",
              np.array_equal(on_disk, pca.components_),
              f"max|Δ|={np.max(np.abs(on_disk - pca.components_)):.3g}")

        loaded = BatchedCovariancePCA.load(tmp, device="cpu")
        check("loaded n_components matches fitted",
              loaded.n_components == pca.n_components,
              f"{loaded.n_components} vs {pca.n_components}")
        check("loaded total_variance restored",
              loaded.total_variance_ is not None and
              abs(loaded.total_variance_ - pca.total_variance_) < 1e-3,
              f"{loaded.total_variance_} vs {pca.total_variance_}")
    finally:
        shutil.rmtree(tmp)


# ---------------------------------------------------------------------------
# 4. Rank-deficient input must be detected, not silently accepted
# ---------------------------------------------------------------------------

def test_rank_floor() -> None:
    print("\n--- rank floor: genuinely rank-deficient ensemble ---")
    rng = np.random.default_rng(7)
    n, D, true_rank = 20, 4000, 5
    basis = rng.normal(0, 1, (true_rank, D))
    coefs = rng.normal(0, 1, (n, true_rank))
    X = (coefs @ basis).astype(np.float32)   # rank <= 5, plus mean-centering

    pca = BatchedCovariancePCA(n_components=n - 1, device="cpu")
    pca.fit(make_loader(X), n_models=n, batch_size=5)
    check("rank floor caps components at the true rank",
          pca.n_components <= true_rank + 1,
          f"kept {pca.n_components}, true rank {true_rank}")
    check("retained components still orthonormal",
          np.max(np.abs(pca.components_ @ pca.components_.T
                        - np.eye(pca.n_components))) < 1e-3,
          f"max|G − I|={np.max(np.abs(pca.components_ @ pca.components_.T - np.eye(pca.n_components))):.3g}")


def test_pathological_spectrum(rng, n: int, D: int) -> None:
    print("\n--- pathologically steep spectrum: must truncate, not fabricate ---")
    decay = np.exp(-np.arange(n) * 0.7)          # ~1e-7 amplitude ratio -> 1e-14 in C
    X = ((rng.normal(0, 1, (n, n)) * decay) @ rng.normal(0, 1, (n, D))).astype(np.float32)

    pca = BatchedCovariancePCA(n_components=n - 1, device="cpu")
    pca.fit(make_loader(X), n_models=n, batch_size=6)
    kk = pca.n_components

    check("unresolvable directions are dropped", kk < n - 1,
          f"kept {kk} of {n - 1} requested")

    G = pca.components_ @ pca.components_.T
    ortho_err = float(np.max(np.abs(G - np.eye(kk))))
    # Bound is deliberately loose. On data this ill-conditioned the trailing
    # directions are at float32's noise level before the algorithm sees them, so
    # exact orthonormality is not attainable by any method operating on a Gram
    # matrix. The point of this check is the ORDER OF MAGNITUDE: the old
    # randomized_svd path produced 0.645-0.82 here, i.e. vectors that were barely
    # related to a basis at all.
    check("everything retained is near-orthonormal", ortho_err < 5e-2,
          f"max|G − I|={ortho_err:.3g} over {kk} components "
          f"(old randomized_svd path: 0.65-0.82)")

    # And the retained subspace must still capture essentially all the energy,
    # since what was dropped was numerically null.
    Xc = X - pca.mean_
    proj = (Xc @ pca.components_.T) @ pca.components_
    captured = 1.0 - (np.linalg.norm(Xc - proj) ** 2 / (np.linalg.norm(Xc) ** 2 + 1e-30))
    check("retained subspace captures ~all energy", captured > 0.999,
          f"captured={captured:.6f}")


def test_codes_file_is_self_describing() -> None:
    print("\n--- codes file: stale-shape reads must raise, not silently truncate ---")
    rng = np.random.default_rng(11)
    n, D, k = 16, 3000, 15
    X = rng.normal(0, 1, (n, D)).astype(np.float32)
    pca = BatchedCovariancePCA(n_components=k, device="cpu")
    pca.fit(make_loader(X), n_models=n, batch_size=4)

    tmpdir = tempfile.mkdtemp()
    try:
        path = os.path.join(tmpdir, "codes.npy")
        pca.transform(make_loader(X), n_models=n, batch_size=4, output_file=path)

        # A real .npy header is present.
        with open(path, "rb") as fh:
            magic = fh.read(6)
        check("codes file carries an npy header", magic == b"\x93NUMPY",
              f"magic={magic!r}")

        # Correct shape loads.
        ok_codes = load_codes(path, n_models=n, n_components=pca.n_components)
        check("correct shape loads", ok_codes.shape == (n, pca.n_components),
              f"shape={ok_codes.shape}")

        # The exact scenario that used to pass silently: asking for FEWER rows and
        # columns than the file holds. np.memmap would have returned the first
        # n_small*k_small floats reinterpreted as the current codes.
        raised = False
        try:
            load_codes(path, n_models=8, n_components=5)
        except ValueError:
            raised = True
        check("smaller-than-file shape is rejected", raised,
              "np.memmap would have accepted this and returned garbage")

        # And a legacy headerless file must produce an actionable error.
        legacy = os.path.join(tmpdir, "legacy_codes.npy")
        raw = np.memmap(legacy, dtype=np.float32, mode="w+", shape=(n, k))
        raw[:] = 1.0
        raw.flush(); del raw
        raised = False
        try:
            load_codes(legacy, n_models=n, n_components=k)
        except ValueError as exc:
            raised = "force_encode" in str(exc)
        check("legacy headerless file raises with actionable message", raised)
    finally:
        shutil.rmtree(tmpdir)


def main() -> int:
    rng = np.random.default_rng(42)

    # Flat-ish spectrum, k = n-1 (the regime randomized_svd handled worst)
    n, D = 24, 8000
    X_flat = rng.normal(0, 1, (n, D)).astype(np.float32)
    test_against_exact("flat spectrum, k=n-1", X_flat, n - 1)

    # Moderate decay — steep enough to be a real test, still inside what a Gram
    # matrix over float32 data can resolve.
    decay = np.exp(-np.arange(n) * 0.15)
    X_mod = ((rng.normal(0, 1, (n, n)) * decay) @ rng.normal(0, 1, (n, D))).astype(np.float32)
    test_against_exact("moderate decay, k=n-1", X_mod, n - 1)

    # Truncated
    test_against_exact("flat spectrum, k=n/2", X_flat, n // 2)

    # Pathologically steep. The contract here is NOT "reproduce the reference" --
    # forming C squares the condition number, so the trailing directions are below
    # float32 precision and genuinely unrecoverable. The contract is that the code
    # DETECTS this and drops them, rather than normalising roundoff to unit length
    # and passing it off as a component (which is what randomized_svd did).
    test_pathological_spectrum(rng, n, D)

    test_noise_ensemble()
    test_save_load()
    test_rank_floor()
    test_codes_file_is_self_describing()

    print("\n" + "=" * 64)
    if FAILS:
        print(f"FAILED ({len(FAILS)}): " + "; ".join(FAILS))
        return 1
    print("ALL CHECKS PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
