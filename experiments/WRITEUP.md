# Uncertainty in reference-relative novelty: experiment write-up

Branch `uncertainty-experiments`. Every headline number below was recomputed independently from the scripts' outputs or from the raw distance matrices. Scope: Li-containing compounds only.

## Bottom line

Novelty is always relative to a reference set, and reference sets are incomplete. The simplest defensible contribution is to measure how much an incomplete reference can change novelty calls, and to report novelty with that uncertainty attached:

1. **MINov's novelty calls hold up when the reference grows.** Novelty was scored against the experimental Li compounds known in 2000 and then against the full record. MINov's novel/not-novel call changed for **0.7–2.4%** of candidates. Nearest-neighbour novelty, the form most used in the field, changed for **2.5–10.4%**: about **15× more often** for real later discoveries under ElMD.
2. **The usual way of estimating uncertainty is overconfident.** Nominal 90% rank intervals from random reference subsampling contained the realized future rank only **11–86%** of the time. Fine ranks are not meaningful at scale: 88–91% of adjacent MINov ranks swap if a single reference material is removed. Report calls and coarse tiers, not ranks.
3. **Global and local novelty are different quantities, and kernel width decides which one you measure.** MINov's median cutoff gives a stable, global measure. Narrow kernels and nearest-neighbour distance are local, more sensitive and less stable. No single width gives both.
4. **Analogue vacancy characterises how a generator's novelty differs.** GNoME candidates sit in substitution families that experiment has not explored. They are about 100× more "vacant" than real later discoveries (AUC 0.96 vs MP). This holds with either reference, but vacancy does not predict where experiments go next.

Two ideas failed their forward tests and are dropped. Plausibility-weighted de-novelling ("P(stays novel)") was no better than the plain density margin. Removal fragility did not predict which ranks actually changed, because the future adds materials, and a removal analysis cannot see that direction.

## Data

- **Reference:** Materials Project v2023.11.1 (AWS Open Data, `materialsproject-build`), Li-containing, non-deprecated entries.
  - R23 = 2,095 experimental (ICSD-tagged) entries, all polymorphs.
  - Each entry gets a first-report year: the earliest literature year in MP provenance (median 1993, range 1916–2013).
- **Candidates:**
  - GNoME Li materials from the December 2023 release: a 3,000-sample for structure, all 16,762 formulas for formula-level analyses.
  - The 44 GNoME materials from the paper, used as a sanity check.
- **Year split:** reference = compounds first reported up to Y; "later" = those reported after Y. Y = 2000, with 1990 and 2005 as robustness checks.
  - MP itself added only 4 experimental Li entries between 2023 and 2025; nearly all its growth was GNoME imports tagged as theoretical. A comparison between MP releases was therefore not usable.
- **Distances:** LoStOP (structure) and ElMD (composition). The rebuilt matrices reproduce the published ones to 1e-7 and 1.6e-4 respectively.
- **Scores compared:**
  - MINov (exact, and the published grid)
  - the paper's KDE and kNN baselines, reimplemented to match `methods_analysis.ipynb`
  - Gaussian KDE and kNN width sweeps
  - 1-NN and 5-NN distance
  - LOF
  - a compressed MI kernel (cutoff at the q-quantile instead of the median)

A useful identity: for a symmetric distance matrix the MI profile equals the binary entropy of the fraction of reference pairs within distance t. The MI cutoff is therefore always the median pair distance, and MINov's weight 1 − H(F(d)) falls to half at the 11th percentile of pair distances.

## Results

### 1. How stable is novelty when the reference changes? (E1, E1b, E4, E7)

| Perturbation | MINov | Nearest-neighbour (1-NN) | Note |
|---|---|---|---|
| Random 90% removal: rank correlation with the full reference (E1) | 0.97–0.98 | 0.50–0.64 | Median-bandwidth Gaussian KDE: 0.99 |
| Additions needed to de-novel a novel candidate (E2) | median 42 (ElMD) | 1 | Closed form for MINov |
| Certified removals to flip a call (E4/E7, 3,000 candidates) | median 127 (ElMD) / 407 (LoStOP) | 36 / 42 | Greedy with refit; upper bounds |
| **Realized call flips, 2000 → full record (E7)** | **0.7–2.4%** | **2.5–10.4%** | GNoME: 1.1% vs 5.6% (ElMD), 0.8% vs 2.5% (LoStOP). Later discoveries: 0.7% vs 10.4%, 2.4% vs 5.6% |
| Nominal 90% rank-interval coverage of the realized rank (E7) | 0.35–0.86 | 0.42–0.55 | Median-bandwidth KDE: 0.11–0.41 |

![Figure 1: call flips and interval coverage in the forward test](results/report_fig1_forward.png)

*Figure 1. Left: share of candidates whose novel/not-novel call changed when the reference grew from the compounds known in 2000 to the full experimental record. Right: how often the nominal 90% rank interval from random reference subsampling contained the rank the candidate actually ended up with. The dashed line marks the nominal 90%.*

**Fine ranks are not robust:**
- **Swaps:** 88–91% of adjacent MINov rank pairs swap with a single removal, at 3,000 candidates.
- **Tiers:** of the two ways of grouping ranks into tiers, overlapping intervals chain into 4–5 blocks; walking down the ranking gives about 50 usable tiers.
- **What predicts movement:** the width of the random-subsampling interval does predict which ranks move (ρ ≈ 0.5). Removal fragility predicts it only weakly (ρ 0.07–0.26), because realized change comes from additions.

### 2. Global vs local novelty (E1b, E5, E9)

- **The trade-off:**
  - All scores lie on one stability–sensitivity frontier ordered by kernel width.
  - MINov sits at the stable, global end. It barely registers when a material's 20 nearest neighbours are removed (Δ ≈ 0.03 in novelty percentile).
  - Narrow kernels register such local holes, at a cost in stability.
- **Synthetic data with known density:**
  - Narrower kernels are more accurate (Spearman with the true density 0.97–0.99, vs 0.85–0.92 for MINov).
  - A compressed MI kernel (q ≈ 0.15) matches the best KDE there, but the advantage did not carry over to real ElMD data.
  - No ground-truth-free rule picks the width reliably.
- **Year split:**
  - Compounds reported after Y are scored as more novel than those already known by 1-NN and narrow kernels (AUC ≈ 0.70), but not by MINov (0.46–0.53).
  - Caution: "reported later" means newer, not chemically or structurally novel. This shows that later compounds sit slightly farther from their single nearest known neighbour, a local property. It does not show they are globally novel, and it is not evidence against MINov as a global measure.
- **Two-scale score:**
  - Rank-averaging MINov with a narrow MI kernel beats the stability–sensitivity frontier on both the original and the larger reference (+0.02 to +0.06 stability at equal sensitivity, CIs exclude 0, except 100-NN holes in ElMD).
  - It adds a second width and a mixing weight. Treat it as a diagnostic rather than a recommended method.

![Figure 2: stability against local sensitivity across kernel widths](results/report_fig2_tradeoff.png)

*Figure 2. Each point on the grey curve is an MI kernel of a different width, from narrow (left) to MINov's median cutoff (right). Top: sensitivity to a local gap (how much a compound's novelty rises when its 20 nearest neighbours are removed). Bottom: how well the score separates compounds reported after 2000 from those already known (0.5 = chance). This measures newness and local isolation, not global novelty. Wider kernels are more stable and less locally sensitive.*

### 3. Analogue vacancy (E6, E8)

V = log((E + 0.5)/(O + 0.5)). E is how many of a material's plausible substitution analogues (Hautier-style odds) the calibration expects to be present in the reference; O is how many actually are. Higher V means more vacant.

| Comparison | AUC [95% CI] |
|---|---|
| GNoME (16,762 formulas) vs MP, experimental reference | 0.959 [0.955, 0.962] |
| Same, reference includes theoretical MP entries | 0.954 [0.952, 0.957] |
| Compounds reported after Y vs before (Y = 1990 / 2000 / 2005) | 0.73 / 0.71 / 0.69 |
| GNoME vs real later discoveries | 0.92 |
| Vacancy at Y predicts an analogue being reported later | 0.35 (reversed: populated families fill, vacant families stay vacant) |

![Figure 3: analogue vacancy against MINov density](results/e6_vacancy_vs_density.png)

*Figure 3 (from E6, the 44 published GNoME candidates). Analogue vacancy against MINov density for MP compounds (grey, leave-one-out) and GNoME candidates (blue). Vacancy separates GNoME from MP along an axis density does not capture. Right: in the falsification test, MP split against itself (grey) sits at AUC 0.5, while GNoME (blue) sits near 0.93.*

- **Robustness:** vacancy survives cross-fitting, controls for analogue count and number of elements, removing f-block elements, and a falsification test (MP split against itself: AUC 0.50).
- **What it measures:** departure from explored chemical families. That is a useful way to characterise a generator's output. It is not a predictor of discovery.

### 4. What did not work

- **P(stays novel), from plausibility-weighted additions (E2, E7):**
  - Discriminated surviving novelty calls worse than the plain density margin in all 17 evaluable cells (e.g. AUC 0.72 vs 0.985).
  - The substitution-based existence calibration underpredicts survival for GNoME and overpredicts it for real discoveries.
- **Removal fragility as a forecast of future change (E7):** see section 1. It remains a useful measure of robustness, but not a forecast.
- **Coverage estimators (Good-Turing, Chao; E3):** predict random additions within ~2%, but underpredict the novelty of generated candidates 2–18×.
- **Near-duplicates (E5):** did not explain nearest-neighbour instability. The low intrinsic dimension of the data (1.6–2.4) likely does.

## Related work: novelty metrics in use and their exposure to reference uncertainty

A literature survey (mostly abstracts and partial full texts; items marked [UV] are unverified) found that common novelty metrics are binary exact-match checks or nearest-neighbour distances. None reports uncertainty.

| Metric / work | Definition | Reference set | Uncertainty reported | Exposure, given our results |
|---|---|---|---|---|
| CDVAE, DiffCSP, FlowMM benchmarks | Novel if pymatgen StructureMatcher finds no match in the training set | MP-20 / MPTS-52; tolerances [UV] | None | Exact match depends on reference coverage and matcher tolerance; a small reference inflates novelty. |
| MatterGen S.U.N. (Zeni et al., Nature 2025) | Stable, unique, novel; novel = no match under an ordered/disordered matcher | MP + Alexandria (+ ICSD) [UV] | No interval | A "novel" compound was a known disordered 1971 phase in the training data (Juelsholt, Mater. Horiz. 2026). |
| LeMat-GenBench (Betala et al., 2025) | Fraction not found in the reference | LeMat-Bulk (~5M) | One reference swap | MatterGen novelty 83.1% → 70.5% from MP-20 to LeMat-Bulk; the only reference-sensitivity result found, with no interval. |
| Continuous novelty (Negishi et al., 2025) | Mean minimum distance to the training set (Magpie, AMD) | MP-20 | None | A 1-NN score: the form our results show flips several times more often as the reference grows. |
| Substitution-based novelty (Negishi & Walsh, 2026) | Duplicate / substitution-derived / new | MP-20 | Not reported | 81–92% of generated crystals are duplicate or substitution-derived; consistent with our vacancy results. |
| GNoME (Merchant et al., Nature 2023) | New entries on the convex hull | MP, OQMD, WBM, ICSD snapshots [UV] | Snapshot only | Count depends on snapshot completeness; audit found few candidates that are novel, credible and useful (Cheetham & Seshadri, Chem. Mater. 2024). |
| A-Lab (Szymanski et al., 2023) | Absent from the reference | Not examined | None | About two-thirds argued to be known disordered phases (Leeman et al., PRX Energy 2024). |
| DiSCoVeR (Baird, Diep & Sparks, Digital Discovery 2022) | Composition density (ElMD + DensMAP) as a novelty proxy | MP | None | A wide-kernel density: a global measure, so it is stable but insensitive to local gaps (section 2). |
| kNN-distance OOD scores (e.g. MOOD) | Distance to the training set as an OOD flag | Training set | No reference perturbation | Same 1-NN fragility. |

No work was found that reports removal or addition fragility, rank intervals, matcher-tolerance sweeps or reference-version intervals for any of these metrics.

## Limitations

- **Scope:** one chemistry (Li). The first-report year is a literature proxy, and "later" means newer, not necessarily novel.
- **Ground truth:** there is none for global novelty in real data. The forward test validates robustness of calls, not whether they are correct.
- **Greedy counts:** removal counts are upper bounds.
- **Substitution model:** the analogue generator is element-level, ignores structure, and uses exact formula matching.
- **Coverage:** forward-test coverage is limited to MP's experimental record; GNoME's "future" there is narrow.

## Possible next step (optional)

An adaptive-bandwidth variant without a new free parameter: Abramson's square-root rule, using MINov itself as the pilot density. It could close part of the global/local gap. It can be tested on the existing E1b/E9 setup. A fully local version of MINov is degenerate: every point gets the same density, because its weights become rank-based. Any local adaptation therefore has to pool information across neighbours.

## Reproduce

1. `uv run python experiments/fetch_data.py` downloads ~3 GB of public data to `data/raw/` (git-ignored, no API key needed).
2. `uv run python experiments/build_li_sets.py` builds the Li sets and distance matrices in `data/raw/processed/` (~20 min).
3. Run the experiments with `uv run python experiments/<script>.py`:
   - on the original data: `e1_reference_stability`, `e1b_stability_sensitivity`, `e2_denovelling`, `e3_coverage`, `e4_rank_uncertainty`, `e5_kernel_shape`, `e6_analogue_vacancy`
   - on the rebuilt data: `e7_fragility_forward`, `e8_vacancy_forward`, `e9_two_scale_transfer`

Each takes 1–65 min. Results go to `experiments/results/`; figures are the `*.png` files there.

4. `uv run python experiments/make_report_figures.py` redraws Figures 1 and 2 from the E7 and E9 outputs.
