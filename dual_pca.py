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
  Pass 2 – build C block-by-block (outer loop stays in memory, inner loads/discards)
  Pass 3 – randomized SVD on C
  Pass 4 – vectorised back-projection to obtain full-space components

Memory footprint: O(micro_batch_size × n_params + n_models²)
"""

import gc
import tempfile
from typing import Callable, Iterator, Optional, Tuple

import numpy as np
import torch
from sklearn.utils.extmath import randomized_svd
from tqdm import tqdm


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
        C = np.zeros((n_models, n_models), dtype=np.float32)
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
                C[i_s:i_e, j_s:j_e] = (bi_c.T @ bj_c).cpu().float().numpy()

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

        # ---- Pass 3: randomised SVD on C ----------------------------------
        n_comp = min(self.n_components, n_models - 1)
        print(f"Pass 3/4: randomised SVD on {n_models}×{n_models} Gram matrix → {n_comp} components …")
        U, S, _ = randomized_svd(C, n_components=n_comp, n_iter=5, random_state=self.seed)
        eigenvalues = S / (n_models - 1)
        del C
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

        # Normalise to unit length
        norms = torch.norm(components, dim=1, keepdim=True)
        components = components / torch.clamp(norms, min=1e-10)

        self.components_ = components.cpu().float().numpy()
        self.explained_variance_ = eigenvalues[:n_comp]
        total_var = np.sum(eigenvalues)
        self.explained_variance_ratio_ = eigenvalues[:n_comp] / (total_var if total_var > 0 else 1.0)

        del components, U_t, mean_t
        if self.device == "cuda":
            torch.cuda.empty_cache()
        gc.collect()

        cum_var = np.sum(self.explained_variance_ratio_)
        print(f"Fitted {n_comp} components  |  cumulative variance: {cum_var:.4%}")
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

        projected = np.memmap(output_file, dtype=np.float32, mode="w+",
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
        """Save PCA artifacts to save_dir (components as float16 to halve disk usage)."""
        import json
        import os
        os.makedirs(save_dir, exist_ok=True)
        # Components saved as float16 to reduce disk footprint (~50%)
        np.save(os.path.join(save_dir, "components.npy"),
                self.components_.astype(np.float16))
        np.save(os.path.join(save_dir, "mean.npy"), self.mean_)
        np.save(os.path.join(save_dir, "explained_variance.npy"), self.explained_variance_)
        np.save(os.path.join(save_dir, "explained_variance_ratio.npy"),
                self.explained_variance_ratio_)
        meta = {
            "n_components": int(self.n_components),
            "n_params": int(self.n_params_),
            "n_models": int(self.n_models_),
            "seed": self.seed,
            "total_variance_captured": float(np.sum(self.explained_variance_ratio_)),
        }
        with open(os.path.join(save_dir, "pca_meta.json"), "w") as f:
            json.dump(meta, f, indent=2)
        print(f"PCA saved to {save_dir}  "
              f"(components: {self.components_.shape}, float16 on disk)")

    @classmethod
    def load(cls, save_dir: str, device: str = "cuda") -> "BatchedCovariancePCA":
        """Load PCA from save_dir. Components are stored as float16, loaded as float32."""
        import json
        import os
        with open(os.path.join(save_dir, "pca_meta.json")) as f:
            meta = json.load(f)
        pca = cls(n_components=meta["n_components"], seed=meta["seed"], device=device)
        # Load float16 components, cast back to float32 for computation
        pca.components_ = np.load(
            os.path.join(save_dir, "components.npy")).astype(np.float32)
        pca.mean_ = np.load(os.path.join(save_dir, "mean.npy"))
        pca.explained_variance_ = np.load(
            os.path.join(save_dir, "explained_variance.npy"))
        pca.explained_variance_ratio_ = np.load(
            os.path.join(save_dir, "explained_variance_ratio.npy"))
        pca.n_models_ = meta["n_models"]
        pca.n_params_ = meta["n_params"]
        print(f"PCA loaded from {save_dir}  "
              f"({pca.n_components} components, {pca.n_params_:,} params, "
              f"{meta['total_variance_captured']:.4%} variance captured)")
        return pca


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
