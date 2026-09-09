# Phase 2 results — the zoo at N=100, GPT-2 Mini

2026-09-09. Two zoos of 100 GPT-2 Mini models (51.5M), branched off one shared trunk
at β=0.15 and β=0.30. 20 anchors (5 one-hot mixtures × 4 branches) + 80 Dirichlet
singletons; 85 distinct π; k=99.

Figures: `reports/phase2_report.html`. Regenerate with
`python reports/build_phase2_figures.py && python reports/make_phase2_html.py` —
both pure stdlib, reading committed JSON under `reports/data/`.

**β = 0.15 is the decided value** (Scott, 2026-09-09), on the grounds below plus
~417 GPU-hr saved across Small and Medium.

---

## 1. The §4.3 gate passes on both arms

| | β=0.15 | β=0.30 |
|---|---|---|
| separation (anchor best on own domain) | 5/5 | 5/5 |
| min signal/noise | 18.37 | 10.86 |

Per-domain signal/noise, against Phase 1's N=12:

| domain | β=.15 N=12 | β=.15 N=100 | β=.30 N=12 | β=.30 N=100 |
|---|---|---|---|---|
| web | 72.32 | 29.76 | 52.07 | 31.40 |
| code | 34.84 | 28.22 | 54.66 | 26.49 |
| math | 80.55 | 23.52 | 29.89 | 21.92 |
| books | 5.96 | 21.60 | 18.66 | 23.43 |
| multilingual | 7.02 | 18.37 | 5.40 | 10.86 |

The N=100 gate is the stricter test. At N=12 only three domains had an anchor group,
so books and multilingual had no specialist model and their between-group spread was
small — they set the Phase 1 minimum. With five anchor groups every domain has a
specialist, separation is evaluated on 5 domains rather than 3, and the within-group
noise floor pools 15 degrees of freedom rather than 9. The minimum rises 5.96 → 18.37.
The three originally-anchored domains fall because their between-group spread is now
averaged over five groups instead of three.

books passes here (own-anchor PPL 38.89 against a best-other 68.01 at β=0.30) despite
being nearly flat in the singleton probe (§3). Both are true: a model trained *only* on
books is clearly distinguishable; π_books has little marginal effect *inside a blend*.

## 2. The spectrum: §4.4's prediction misses, and the band was under-specified

85 distinct π means the mean-centred rank is the full 99 — the N=12 design ceiling
(handoff §2) is gone. At k=99:

| arm | region | ev0/median | eff. rank | ratio | variance share |
|---|---|---|---|---|---|
| 0.15 | whole stack | 309.2 | 4.47 | 0.0451 | 100% |
| 0.15 | blocks only | 107.3 | 7.20 | 0.0727 | 13.4% |
| 0.15 | embeddings only | 400.9 | 4.03 | 0.0407 | 86.6% |
| 0.30 | whole stack | 325.5 | 4.48 | 0.0452 | 100% |
| 0.30 | blocks only | 118.9 | 7.31 | 0.0738 | 15.5% |
| 0.30 | embeddings only | 421.5 | 3.98 | 0.0403 | 84.5% |

§4.4 predicted `effective_rank_ratio ≈ 0.2–0.3`. Measured is 0.045 whole-stack and
0.073 block-only.

**The miss is in the opposite direction from the pre-registered falsifier.** §4.4's
disproof condition was a *flat* spectrum — ratio → 1.0, `ev0/median` → 1.0 — which would
have meant branches drift into near-orthogonal directions and no generative model over
these codes can generalize. `ev0/median` of 309–325 is the reverse: the structure is
present and more concentrated than predicted. The qualitative prediction holds; the
numeric band does not.

### 2a. `effective_rank_ratio` is not k-invariant, so the band needs a k

The 99 eigenvalues are stored, so any k is a re-read. β=0.15, the same eigenvalues:

| region | statistic | k=99 | k=50 | k=25 | k=10 |
|---|---|---|---|---|---|
| whole stack | effective rank | 4.47 | 4.00 | 3.72 | 3.43 |
| whole stack | ratio | 0.0451 | 0.0801 | 0.1488 | 0.3434 |
| blocks only | effective rank | 7.20 | 5.79 | 4.98 | 4.35 |
| blocks only | ratio | 0.0727 | 0.1157 | 0.1991 | 0.4352 |
| embeddings | effective rank | 4.03 | 3.68 | 3.46 | 3.21 |
| embeddings | ratio | 0.0407 | 0.0735 | 0.1384 | 0.3210 |

The ratio divides by k while the effective rank is nearly k-invariant, so the ratio
scales as roughly 1/k: it varies 7.6× across this sweep where the rank varies ~30%. At
k=25 the block ratio is 0.199 — inside §4.4's band. At k=99 it is 0.073 — well below.

§4.4 never states a k. This is the same defect as misstep 15b (`ev[0]/ev[k−1]` reading
12.09 at k=N−1 and 1.006 at k=N/2 on identical data), now in the statistic the repo
adopted *as the fix* for misstep 15b.

**Reporting rule going forward: quote the absolute `effective_rank`, or state k
alongside every ratio.** `scripts/diag_block_spectrum.py` now emits the sweep by
default so a single ratio cannot be quoted as a property of the data.

### 2b. The leading structure is ~4-dimensional, matching the mixture simplex

A class here is a point on the 5-domain simplex, which is **Δ⁴ — four-dimensional**.
However many distinct π are sampled, the between-mixture component of the weight
distribution can span at most 4 dimensions. Reaching `effective_rank_ratio` 0.2–0.3 at
k=99 would require 20–30 effective dimensions, which this conditioning variable cannot
supply.

The embeddings measure **4.03 and 3.98** — dim(Δ⁴) to within 0.05.

This is not a rank-4 claim. Cumulative variance at β=0.30:

| region | c1 | c2 | c3 | c4 | c5 | reaches 90% |
|---|---|---|---|---|---|---|
| whole stack | 39.9% | 59.8% | 73.7% | 79.1% | 83.2% | c19 |
| blocks only | 26.0% | 44.4% | 59.9% | 67.9% | 74.1% | > c24 |
| embeddings | 43.3% | 63.7% | 77.1% | 81.8% | 85.3% | c13 |

18% of embedding variance and 32% of block variance lie beyond component 4. The
participation ratio weights by squared eigenvalue, so it reports where the mass is, not
a cutoff. The defensible statement:

> The leading structure is approximately 4-dimensional and coincides with the dimension
> of the mixture simplex. The transformer blocks carry ~1.8× more effective dimensions
> than the embeddings, with a substantially heavier tail.

§6.3 and §11 both anticipated "between-mixture structure spans ≤4 dimensions" as a
hazard of the five-mixture fallback. It holds with 85 distinct mixtures too.

## 3. §4.5's embedding-share confound is dominant

The embedding block (`wte` + `wpe` + final LN) is **51.0% of D** and carries **86.6%**
of the centred variance at β=0.15. A whole-stack spectrum is therefore substantially an
embedding spectrum: the whole-stack ratio (0.045) tracks the embeddings' (0.041) while
the blocks sit at 0.073.

The vocab is fixed at 50257, so this share falls 51.0% → 31.6% → 14.8% from Mini to
Small to Medium. The composition of D changes along the §4.7 ladder, not only its size.
**Every ladder point must report both curves.** `scripts/diag_block_spectrum.py`
produces them in one streaming pass with no PCA, no GPU and no torch.

## 4. The singleton probe: mixture identity controls the model at Phase 2's hardest separation

Six singletons per arm at `--alpha 8.0`, minimum pairwise L1 = **0.164**, against Phase
2's actual closest pair of 0.1725. π identical across arms, so β is the only variable.

Statistic (§6.5): per domain *d*, regress `adv[i][d] = mean_j(ln ppl[j][d]) − ln ppl[i][d]`
on requested weight π_d. Log space because perplexity composes multiplicatively. A
*control* test, not a quality test — §6.5's reason for calling it immune to misstep 19's
collapse trap. The null is exact: all 6! = 720 permutations of the member↔π pairing,
enumerated, so the p floor is 1/720 = 0.0014.

| | β=0.15 | β=0.30 |
|---|---|---|
| pooled slope | +0.328 | +0.693 |
| exact p | 0.0014 (floor) | 0.0014 (floor) |
| domains with positive slope | 5/5 | 4/5 |

| domain | β=.15 slope | *r* | *p* | β=.30 slope | *r* | *p* |
|---|---|---|---|---|---|---|
| web | +0.370 | 0.757 | 0.050 | +0.552 | 0.564 | 0.083 |
| code | +0.158 | 0.550 | 0.161 | +0.414 | 0.471 | 0.185 |
| math | +0.495 | 0.888 | 0.011 | +1.202 | 0.945 | 0.008 |
| books | +0.038 | 0.563 | 0.128 | −0.072 | −0.535 | 0.864 |
| multilingual | +0.498 | 0.934 | 0.010 | +1.092 | 0.853 | 0.011 |

books is the outlier at both β and negative at 0.30. Its PPL spans 0.9% (β=0.15) and
2.0% (β=0.30) against 15–17% swings on math and multilingual. Three reasons, all
expected: Project Gutenberg is generic English so web and math training also improve it;
its ~195 KB documents mean the 64-block held-out slice comes from a handful of books; and
Phase 1 already had books as the min-SNR domain at β=0.15.

**§6.3's scope paragraph should read "four discriminative domains plus a near-flat book
axis"**, alongside the existing "four of five are English, the fifth is French, and code
is Python".

## 5. Weight-space geometry

| statistic | β=.15 N=12 | β=.15 probe | β=.15 N=100 | β=.30 N=12 | β=.30 probe | β=.30 N=100 |
|---|---|---|---|---|---|---|
| displacement from trunk | 16.5% | 5.0% | 7.73% | 28.7% | 12.2% | 16.20% |
| spread / displacement | 0.950 | 0.748 | **1.136** | 0.878 | 0.517 | 0.930 |
| centroid fraction | 0.741 | 0.646 | 0.682 | 0.738 | 0.646 | 0.689 |
| between / within | 4.075 | — | 3.963 | 3.975 | — | 3.925 |

`spread/displacement` at N=100 β=0.15 is 1.136, closer to the √2 = 1.414 of fully
independent movement than anything measured before. The probe's 0.748 was a lower bound
by construction: matching Phase 2's closest pair with only 6 draws required α=8, which
clusters the draws at the barycentre — and the barycentre is the uniform mixture the
trunk itself trained on.

`between/within` is now essentially identical across β (3.963 vs 3.925), where at N=12
it differed (4.075 vs 3.975). It no longer discriminates.

## 6. Why β = 0.15

| criterion | β=0.15 | β=0.30 | favours |
|---|---|---|---|
| gate separation | 5/5 | 5/5 | tie |
| gate min SNR | 18.37 | 10.86 | 0.15 |
| eff. rank, whole stack | 4.47 | 4.48 | tie |
| eff. rank, blocks | 7.20 | 7.31 | 0.30, marginal |
| ev0/median | 309.2 | 325.5 | 0.30, marginal |
| between/within | 3.963 | 3.925 | tie |
| spread/displacement | 1.136 | 0.930 | 0.15 |
| probe pooled slope | +0.328 | +0.693 | 0.30 |
| probe domains positive | 5/5 | 4/5 | 0.15 |
| Small + Medium cost | ~446 GPU-hr | ~863 | 0.15 |

The spectrum does not discriminate — 0.0451 vs 0.0452 whole-stack, 0.0727 vs 0.0738
block-only — which is the direct answer to the question the two arms were run to settle.
§4.3's rule ("the smallest β at which mixture identity is measurable") therefore selects
0.15, and the handoff's sole reason for overriding it — headroom for the singleton regime
N=12 could not test — has now been tested and passed 5/5.

The cost against that: β=0.30's probe slope is 2.1× stronger, which is real conditioning
signal for the flow. That trade was made explicitly in favour of 0.15.

## 7. What these results do and do not license

**Established.** The zoo is real and the mixture axis is measurable in the models (§4.3,
5/5 both arms). Requested π controls per-domain performance at Phase 2's hardest
separation (§6.5, exact p at the floor). The weight distribution is strongly structured
(`ev0/median` 309–325) with leading dimensionality ≈ dim(Δ⁴). The embedding share
confound is dominant and must be reported.

**Not established.** §4.4 as written. Any claim about how effective rank scales with D —
that needs Small and Medium (§4.7). Anything about generation: no flow has yet been
trained on a real zoo. And the memorisation question is now sharper, not softer: 100
members in a ~4.5-dimensional manifold is exactly the regime §11 warns a flow will
memorise, so §6.4's retrieval baseline — which is **not yet implemented** — is the
control that decides whether any generative result means anything.
