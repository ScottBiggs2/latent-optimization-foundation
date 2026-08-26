# Research Notes — read this before you write code

This file has one job. It stops the next agent from repeating work that already
happened, and from repeating mistakes that already happened.

Written 2026-08-26, after the whole-stack pipeline landed and ran.

---

## 1. Glossary

Define these once. The words are used the same way everywhere in this repo.

| Term | Meaning here |
|---|---|
| **sample** | One complete decoder stack, flattened. NOT one transformer block. |
| **stack** | All L decoder blocks of one model, concatenated. Length `D = L * block_size`. |
| **N** | The number of samples in one family's ensemble. NOT the layer count. |
| **k** | The number of PCA components kept. `k <= N - 1`. |
| **member** | One sample inside an ensemble. Member 0 is the real pretrained model. |
| **family** | One architecture, e.g. `pythia_410m`. Each family has its own PCA basis. |
| **arm** | One evaluation path: `pca_only`, `vae`, or `generate`. |
| **rank bound** | `k = N - 1`. Centering removes one degree of freedom, so the centered data cannot have higher rank. |
| **relL2** | `norm(reconstruction - target) / norm(target)`. |

---

## 2. Where the work stands

The machinery works. Every number it produces matches an analytic prediction.

**Verified**

1. Per-family PCA reconstructs exactly at `k = N-1`. Cosine is 1.000000 for all three
   families. The old shared block basis gave 0.388 for `pythia_410m`.
2. Padding is gone. Every family has uniform block sizes, so a stack is fixed-length.
3. Posterior collapse is gone. Total KL went from 0.00106 to 11.36 nats per sample.
   All 32 latent dimensions stay active. No tuning was needed.
4. The truncation residual formula holds to within 5%.
5. The full run costs 10 minutes and 20 GB of RAM on one V100.
6. 91 automated checks pass across two test modules.

**Not verified, because the current experiment cannot verify it**

The pipeline has never been asked a research question. See section 3.

---

## 3. The core problem: the ensemble is not real

DeepWeightFlow uses about 100 **independently trained** networks per task. This repo
has exactly one pretrained model per family. The ensemble is therefore manufactured:

```
member 0        = w_0                        (the real model, exactly)
member 1..N-1   = w_0 + s * sigma * eps_i    (isotropic noise)
```

Three consequences follow, and all three make the current results uninformative about
weight space.

1. **The basis spans injected noise.** The principal directions describe the
   perturbation, not anything about trained weights.
2. **The mean is the real model.** `mean = w_0 + O(s*sigma/sqrt(N))`. So
   `x_hat ~= mean ~= w_0` at any rank. Reconstruction of member 0 cannot fail.
3. **Truncation acts as a denoiser.** Discarding half the variance discards half the
   injected noise, so a truncated reconstruction of a noisy member is a BETTER model
   than the member. Measured: `pythia_410m` recovers exactly half the noise-induced
   perplexity penalty at `k = N/2`.

Point 3 is worth restating. On this ensemble, reconstruction fidelity and model
quality are **anti-correlated** at low rank. A rank sweep therefore measures the
augmentation, not the model.

### Permutation augmentation does not fix this

Do not reach for permutation augmentation. It is the wrong direction:

- Permuted copies of a network are **functionally identical** to the original.
- A generative model trained on them regenerates the same network.
- DeepWeightFlow's TransFusion exists to **remove** permutation variance so a flow can
  see genuine seed-to-seed variation. Generating permuted copies adds back exactly
  what TransFusion removes.

Permutation augmentation is a good sanity harness. Generated networks should score
identically to the original. It is not a source of structure.

---

## 4. Next experiments

### Experiment 1 — Pythia checkpoint revisions (do this first)

**Question.** Do complete models from one training run share low-dimensional
structure in weight space?

**Why this ensemble.** EleutherAI publishes about 143 intermediate revisions per
Pythia size. Every revision is a genuine complete model. Model scale is fixed. No
member sits at a privileged centre. This is the closest available thing to
DeepWeightFlow's ensemble.

**Start with `pythia_160m`, not `pythia_410m`.**

| | pythia_160m | pythia_410m |
|---|---|---|
| L | 12 | 24 |
| block (matrix only) | 7,077,888 | 12,582,912 |
| D | 84,934,656 | 301,989,888 |
| 143 members on disk | ~48 GB | ~173 GB |

Both fit on scratch. `pythia_160m` iterates about 3.5 times faster.

**Code change required.** `EnsembleDataset.load_chunk` currently regenerates members
from a seed. Real members must be read from disk instead. Add a second ensemble
source rather than replacing the first:

- Keep `source="noise"` for the machinery test.
- Add `source="revisions"`, which stores one `w_i.npy` per member and reads a chunk
  from each. `models/registry.py` needs a `revision=` argument threaded into
  `from_pretrained`.
- `chunk_bounds` must stay the fixed grid. `DualGramPCA` does not care where a chunk
  comes from, only that the chunk is the same on every pass.

**What proves the hypothesis.** A **steep** eigenvalue spectrum. Steep means the
first few components hold most of the variance, so training trajectories occupy a
low-dimensional subspace. Report `ev[0]/ev[k-1]` and the cumulative variance curve.

**What disproves it.** A **flat** spectrum, like the noise ensemble produces. Flat
means the checkpoints are near mutually orthogonal, so PCA does not compress and no
generative model over these codes will generalize.

**Read the spectrum warning.** `DualGramPCA` prints a warning when the retained
spectrum spans more than 1e6. That warning matters here. Forming the Gram matrix
squares the condition number, so a genuinely steep spectrum may exceed what float32
data supports. If the warning fires, lower `k` rather than lowering `rank_rtol`.

### Experiment 2 — Held-out revision

**Question.** Does the basis generalize to a checkpoint it never saw?

**Method.** Fit on revisions 0 to 119. Project revision 130 with
`DualGramPCA.transform_vector`. Reconstruct. Measure ΔPPL.

This is the first honest generalization test in the project. Every earlier number
measured in-sample reconstruction. `transform_vector` already exists and is tested.

**Gate.** ΔPPL, not cosine. Cosine below about 0.9999 tells you nothing.

### Experiment 3 — A flow over the codes

**Do not start this until Experiment 1 shows a steep spectrum.** A flow over codes
whose aggregate distribution is already Gaussian learns the prior and adds nothing.

**Run the cheap baseline first.** Fit a full-covariance Gaussian to the codes, per
family. Sample it. Push the samples through `DualGramPCA.inverse_transform`. Measure
ΔPPL. That is about 20 lines. If a Gaussian matches a flow, the flow is not earning
its complexity.

**If you do build it.** Rectified flow, conditional-OT path. Velocity field
`v(z_t, t, family, step)` as a 3 to 4 layer MLP with a sinusoidal time embedding.
About 50 Euler steps at sampling time. On 99-dimensional codes it trains in seconds.

`eval_stack.py` already has the seam: the `generate` arm decodes a latent and writes
the result back. Replace the sampler, keep everything else.

**Judge samples on ΔPPL.** Do not judge them on code-space or weight-space L2. See
misstep 9.

### Experiment 4 — Cross-family transfer (currently foreclosed)

Per-family PCA gives each family its own basis. Code dimension 0 of `gpt2_medium` and
code dimension 0 of `pythia_410m` are coefficients on unrelated directions. So
`family_emb` now means "which decoder to use", not "where in a shared space".

This forecloses generating weights for a family the model never saw. If cross-family
transfer is a research goal, it needs a different design, not a parameter change.
Decide this explicitly before building on top of per-family bases.

---

## 5. Recorded missteps

Every item below cost real time in this project. Do not repeat them.

### About the framing

**1. A PCA sample is a whole model, not a block.**
The repo's legacy code treats one block as one sample, so reading the code teaches the
wrong model. I got this wrong three times in one conversation. Check section 1 before
you reason about shapes.

**2. `N` is the ensemble size, not the layer count.**
`k = N-1` refers to the number of complete models. It has nothing to do with how many
layers a model has.

**3. Two pipelines coexist. Do not mix their artifacts.**
`blocks/` plus `pca/components.npy` belongs to the legacy pipeline. `ensemble/` plus
`pca/<arch>/gram_*.npy` belongs to the current one. Both write into
`<artifact_dir>`. The `layout_version` fields exist to stop a cross-read.

### About numerics

**4. Do not use `randomized_svd` on a symmetric Gram matrix.**
Use `np.linalg.eigh`. `randomized_svd` returns absolute eigenvalues, so a direction
that rounds negative comes back positive with an arbitrary eigenvector. Unit
normalisation then promotes that noise to a full-status component. Measured on a
synthetic case: orthogonality error 0.82 at `k = n-1`, one direction 100% wrong.

**5. The Gram matrix squares the condition number.**
A singular-value ratio of `r` appears as an eigenvalue ratio of `r^2`. So eigenvalues
below about float32 epsilon are unrecoverable. No choice of `rank_rtol` recovers them.
Lower `k` instead. `DEFAULT_RANK_RTOL = 1e-7` is the conservative end of the usable
range. Do not lower it to keep more components.

**6. Do not store components as float16.**
Unit-norm vectors over `D = 302M` have typical entries near 5.7e-5. Float16 is
subnormal there. Against a cosine target of 0.9999 that is a first-order error. This
also made two scripts silently measure two different bases: one used the in-memory
float32 object, the other loaded the float16 file.

**7. Accumulate the Gram matrix in float64.**
Entries are sums over hundreds of millions of products. Float32 carries about 1e-3
relative error at that length, which is the same size as the structure being resolved.

### About the ensemble

**8. `noise_scale = 1e-7` is unusable.**
1e-7 *is* float32's relative precision. The members sit at the representable limit,
the Gram matrix is roundoff, and the rank floor discards the basis. Use
`calibrate_noise.py`. The calibrated value is `1e-2`.

**9. Member 0 sits at the ensemble centre.**
`mean = w_0 + O(s*sigma/sqrt(N))`, so member 0's deviation from the mean is smaller by
`sqrt(N)`. Two different residual formulas follow, and mixing them up produces a wrong
prediction:

| target | residual at `k = N/2` |
|---|---|
| a typical member | `s*sigma/sqrt(2)` |
| member 0 | `s*sigma/sqrt(2N)` |

I documented this factor and then failed to apply it when writing a prediction into a
job header. Use `eval_stack.py --sample_idx` with a nonzero index for any rank sweep.

**10. Never change the chunk grid between passes.**
Noise is keyed to `(seed, chunk_idx)`. If the grid changes, pass 2 sees a different
ensemble than pass 1, and the Gram matrix is meaningless. `load_chunk` takes a chunk
index, not raw offsets, so that this cannot happen by accident. Keep it that way.

### About measurement

**11. Cosine above 0.999 is not a useful bar.**
A measured cosine of 0.99939 came with +72,468% perplexity. Cosine only becomes
informative above about 0.9999. Gate on ΔPPL.

**12. `pca_only` at `k = N-1` is a self-test, not a result.**
The rank bound makes it exact. Reporting "cosine 1.0" there proves the arithmetic
works and nothing else.

**13. Error magnitude does not predict functional damage.**
Three measurements on `pythia_410m`:

| error source | relL2 | ΔPPL |
|---|---|---|
| VAE round trip at `k=N-1` | 2.0e-3 | **+0.378%** |
| isotropic noise (calibration) | 2.0e-3 | ~+0.03% |
| PCA truncation at `k=N/2` | 7.4e-3 | **−0.524%** (improves) |

The VAE's error is structured. It sits in the leading principal directions, which are
the ones carrying function. Isotropic noise spreads over all 302M dimensions.
Truncation error points back toward `w_0`. So L2 alone ranks these wrongly, and by a
factor of 10.

**14. On a noise ensemble, truncation improves the model.**
Do not read a negative ΔPPL at low rank as success. It means the target was noisy and
truncation removed some of the noise.

**15. `total_variance_captured` is vacuous at `k = N-1`.**
All non-null directions are retained, so the ratio is 1.0 by construction. Report it
at a lower rank as well, or it misleads. `gram_pca_meta.json` carries an explicit
`variance_captured_is_vacuous_at_k_eq_N_minus_1` flag.

### About the code

**16. Do not read a codes file with `np.memmap` and an explicit shape.**
`np.memmap(mode="r", shape=...)` **succeeds silently** when the requested shape is
smaller than the file. It raises only when the shape is larger. So a stale
`(150, 97)` file read as `(100, 50)` returns the first 5,000 floats as if they were
the current codes. Use `dual_pca.load_codes`, which validates the shape.

**17. Do not gate a pipeline stage on `os.path.exists` alone.**
Every stage in the legacy pipeline did this, with no version check, so an artifact
directory from a different layout gets resumed instead of rejected. Add a
`layout_version` and hard-error on a mismatch.

---

## 6. Environment facts

1. **Never run torch or transformers on the user's laptop.** Use Explorer through
   `ssh explorer`. Submit every job with `sbatch`.
2. **Explorer login nodes kill heavy processes.** Even `conda activate` gets killed
   there. Submit even a small numpy test as a job. `slurm_tests.sh` exists for this.
3. **Write large files to `/scratch/biggs.s`.** Never write to `$HOME`. `/tmp` on
   Explorer refuses writes.
4. Partitions: `gpu` (8 h), `gpu-short` (2 h), `short` (CPU, 2 days).
5. Sync code with `rsync -az <files> explorer:/home/biggs.s/llm_vae/`. macOS ships an
   old rsync, so `--info=stats1` fails.
6. To pass arguments through `sbatch --export`, expand the variable unquoted in the
   script. `python -u "$TEST"` treats `"file.py --flag"` as one filename.
7. `HF_TOKEN` is only needed for `MC_EVAL=1`, which pulls the gated GPQA dataset.
   Never hardcode it. The file is tracked by git.

---

## 7. Open questions

1. Is a steep spectrum across Pythia revisions real, or do checkpoints drift into
   near-orthogonal directions? Experiment 1 answers this. Nothing else should be built
   until it does.
2. Should `D` include the embedding matrix and the LM head? The repo models decoder
   blocks only. DeepWeightFlow flattens whole networks. `gpt2_medium`'s embedding is
   51M parameters, about 17% of its stack.
3. Is cross-family transfer a goal? Per-family PCA forecloses it. See Experiment 4.
4. Does the VAE earn its place? At `k = N-1` the PCA is already exact, so the VAE only
   adds error. Its value has to come from producing a smooth, well-conditioned latent
   for a generative model. That claim is untested.
5. Is 32 latent dimensions right for 99-dimensional codes? The VAE currently
   compresses 99 to 32. Nobody has swept this.
