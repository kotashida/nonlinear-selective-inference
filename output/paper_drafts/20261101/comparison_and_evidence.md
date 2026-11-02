# Mathematical Audit, Source Evidence, and Publication Notes

Prepared November 1, 2026 for the Shapley paper. The manuscript concerns conditional tests of zero mean-response contribution for penalized-fit Shapley ranking, accounting jointly for ridge tuning and feature selection.

## Algorithm sources and verification

The authoritative source is [the RBF ridge/CV benchmark](../../../outputs/benchmarks/rbf_ridge_cv_combined_n25_50_75_20261101/README.md), together with its [manifest](../../../outputs/benchmarks/rbf_ridge_cv_combined_n25_50_75_20261101/combined_manifest.json) and [verification record](../../../outputs/benchmarks/rbf_ridge_cv_combined_n25_50_75_20261101/verification.json). Its merged dataset contains 1,692,000 test records, including paired and nested observations; these are not independent datasets.

The implemented method is defined by [rbf_ridge_cv.py](../../../benchmarks/rbf_ridge_cv.py) and [the study runner](../../../benchmarks/run_rbf_ridge_cv_study.py). The algorithm files match recorded provenance. Of the original run's 12 source entries, 11 match; the changed file is the plotting script, as already documented in the benchmark verification. See [the source-hash comparison](source_hash_comparison.json). Source integrity does not establish numerical replay stability.

The structure follows the supplied Matsukawa et al. paper, Section 2, visually inspected on local PDF pages 6-8. The local file is a 2024 preprint of the [AISTATS 2025 paper](https://proceedings.mlr.press/v258/matsukawa25a.html). Historical slides and [advisor notes](../../../docs/7-b4-shida.txt) provide context; recorded run settings take precedence.

Targeted checks passed for fold-local CV against direct regression, penalized coalition gains and positive semidefiniteness, and joint CV/top-k interval membership with chi quadrature. These checks do not resolve the recorded cross-platform replay discrepancies. The Markdown and LaTeX sources accompany the compiled manuscript PDF.

## Mathematical points affecting the contribution

### 1. Penalized objective improvement is essential

For $a=n\lambda>0$, the centered ridge smoother is $H_S=I-a(K_S+aI)^{-1}$, with $K_S=G_SG_S^\top$. Hence

$$
H_{S\cup\{j\}}-H_S
=a\{(K_S+aI)^{-1}-(K_S+G_jG_j^\top+aI)^{-1}\}\succeq0.
$$

This gives nonnegative coalition increments and a PSD Shapley matrix. The equality to minimized RSS plus penalty follows by substituting the ridge normal equations into the objective. It does not follow by taking the residual sum of squares alone. The October 26 OLS/RSS wording is superseded by the referenced ridge/CV run.

### 2. The zero-contribution null may be much broader than a partial effect

If $A=\sum_S w_S D_S$ with $D_S\succeq0$ and every $w_S>0$, then

$$
\ker A=\bigcap_S\ker D_S,\qquad
\operatorname{range}(A)=\sum_S\operatorname{range}(D_S).
$$

Proof: $v^\top Av=0$ is a sum of nonnegative terms, so it holds exactly when every $D_S^{1/2}v=0$. Orthogonal complements give the range statement. The ridge increment can introduce adjusted directions involving other feature blocks; there is no general reason for this range to equal the target feature's unadjusted span.

Consequently, the All-Features space can equal the full centered basis space. When it does, rejecting a feature-labeled hypothesis is rejecting that common basis-space signal null. It does not locate a uniquely active feature. The Selected-Features target can be narrower but has no universal feature-isolation guarantee either. In ideal arithmetic its range is contained in the corresponding All-Features range, since the former uses a subset of the latter's PSD increments. The code's separate eigenvalue cutoffs need not preserve exact containment of retained numerical spaces.

### 3. Positive Shapley weights do not by themselves make the radius test weight-sensitive

Changing strictly positive weights over the same coalition increments leaves their exact range unchanged. With the *ranking and selection event held fixed*, two target matrices with the same range produce the same projector, observed radius, radial path, conditioning event, and analytic $p$-value. Their raw attribution magnitudes can still differ. This is a mathematical consequence of the implemented test, not a newly observed empirical result.

Thus a proposed comparison between Shapley and uniform target weights cannot demonstrate a power difference for this radius procedure when their target spaces and selector remain identical. If the weights also change the selector, its event changes and the conclusion need not hold. If thresholding changes numerical ranks, a difference may reflect the tolerance rather than an intrinsic weighting effect. A procedure that retains more spectral information would be a separate methodological development.

### 4. CV is part of the response-dependent event

Fixing folds does not fix the penalty winner. The winner and top-$k$ comparisons jointly define the radial truncation region. Full-data basis compression does not justify projecting away the response components used in fold validation. This is why the draft retains CV comparisons in $\mathbb R^n$ and condition on the winner as well as the unordered selected set.

## Gaps to address before submission


1. **Fix the scientific claim and test-space definition.** State whether the goal is inference on allocated coalition contribution or identification of an individual active feature. Prove when the Shapley range is narrow or becomes the full universe space. Report ranks and overlap of the two test spaces. Treat the $10^{-10}$ SVD/eigenvalue thresholds as part of the implemented estimand until an exact-range equivalence or controlled approximation is demonstrated. Small positive eigenvalues cannot be dismissed as exact zeros solely for interpretability.
2. **Resolve numerical reproducibility.** The combined verification reports 18 differing tests among 276 original-source replays and 32 among 288 extension replays, with maximum absolute $p$-value discrepancy about 0.0045184. These replay sets overlap; their counts must not be added as distinct failures. No checked decision at $\alpha=0.01,0.05,0.10$ changed. Saved integrity checks passed, but this is not a numerical replay pass. Diagnose the roles of rank thresholds, ill-conditioned bases, CV score ties, interval roots, and tiny conditional masses rather than assigning an unverified cause. Add higher-precision or independent calculations for problematic cases before claiming stable exact computation. Source: [combined verification](../../../outputs/benchmarks/rbf_ridge_cv_combined_n25_50_75_20261101/verification.json).
3. **Write complete fixed-label proofs.** Establish PSD increments, Shapley efficiency, target-null equivalence, measurability and conditioning of selected spaces, quadratic event characterization with deterministic ties, and the conditional chi pivot. Separate exact-arithmetic theory from the thresholded implementation. The draft contains the central derivations, not a submission-ready theorem appendix.
4. **Add scientifically comparable baselines.** Compare inference methods on the same design, target, scale, and null. An unadjusted projection tail and a valid multiplicity or sample-splitting comparator can clarify the value of selection adjustment, but each comparator needs a justified target and error statement. A response-shuffling baseline requires exchangeability under the actual null; arbitrary nonconstant nuisance means do not generally permit it.
5. **Broaden validation of the null actually claimed.** Include designs with dependence, rank deficiencies, constant/discrete features, and nonzero nuisance means orthogonal to the specified target space. Account for fixed target labels when constructing selective-null simulations. Existing fresh-design global-null tests do not replace these checks.
6. **Keep real-data conclusions descriptive.** The pinned fixture uses full-dataset preprocessing; its preparation notebook standardizes responses and jitters some low-cardinality feature columns. The run preserves that fixture. $\sigma=1$ is a working assumption, not a noise estimate validated by response standardization. The observed-response rejection rates cannot be labeled calibrated Type I error or known power. A confirmatory analysis needs a defensible variance model and preprocessing account; the separate disjoint-pool datasets elsewhere in the repository must not be silently substituted for this benchmark's inputs.
7. **State the reporting and multiplicity policy.** One uniform target yields one test per procedure. Reporting both Shapley targets, multiple basis fractions, or multiple features as a collection of discoveries requires a separate policy. Pairing across methods and nested sample sizes creates dependence; pointwise intervals are not simultaneous comparisons. Prespecify the primary target universe and sensitivity settings.
8. **Establish a feasible computational scope.** Exhaustive coalition construction is exponential in $p$. Explain the intended feature-dimension range and measure runtime as well as calibration. Fixed-coalition or permutation approximations could be future work but would change the scoring rule and possibly the null; they are not implemented in the authoritative Shapley benchmark.
9. **Complete the novelty review.** The specific combination may be worthwhile, but a broad claim to first statistical inference for Shapley importance or first selective projection test is untenable. The focused references below establish relevant precedents; a systematic literature review remains necessary before a priority claim.

## Focused literature positioning


The following primary sources were checked for this assessment. Descriptions are limited to the claim supported by each source; this is not an exhaustive related-work section.

| Prior work | Consequence for positioning |
|---|---|
| [Matsukawa et al. (AISTATS 2025)](https://proceedings.mlr.press/v258/matsukawa25a.html) | Reference structure and saturated-mean viewpoint; its application is inference after automatic feature engineering, with coefficient targets. |
| [Loftus and Taylor (2015), group-variable selective inference](https://arxiv.org/abs/1511.01478) | Quadratic selection events and truncated chi/F tests already exist. The chi-radius reduction is a foundation to cite. |
| [Loftus (2015), selective inference after CV](https://arxiv.org/abs/1511.08866) | Representing squared-error CV comparisons by quadratic forms is established; the new claim must be more specific than conditioning on a CV winner. |
| [Rügamer, Baumann, and Greven, additive and mixed models](https://arxiv.org/abs/2007.07930) | Selective inference for additive models and repeatable outcome-dependent selection has precedents. The combined framing needs a clear distinction from this scope. |
| [Covert, Lundberg, and Lee (2020), SAGE](https://proceedings.neurips.cc/paper_files/paper/2020/hash/c7bf0b7c1a86d5eb3be2c722cf2cf746-Abstract.html) | Global loss-based Shapley attribution predates this project. The implemented refitted-coalition penalized training objective should not be equated with SAGE's predictive-power estimand. |
| [Williamson and Feng (ICML 2020), SPVIM](https://proceedings.mlr.press/v119/williamson20a/williamson20a.pdf) | Statistical inference for population Shapley variable importance already exists, using an asymptotic nonparametric approach. Differentiate the present fixed-design conditional null. |
| [Whitehouse, Sawarni, and Syrgkanis (2026), v2 October 17](https://arxiv.org/abs/2602.10532v2) | Recent semiparametric inference for global powers of SHAP reinforces the need to specify exactly which Shapley functional is being tested. It is not the fixed-design ridge contribution here. |
| [Kim et al., phi-Table, v3 May 3, 2026](https://arxiv.org/abs/2512.07578v3) | Current abstract describes SHAP screening and linear-surrogate explanation tables. Earlier search results call it phi-test and mention selective inference; those older descriptions must not be presented as the current method without inspecting the corresponding version. This is a related-work lead, not a verified equivalence or direct baseline. |

## Proposed paper framing

A defensible working title is **Selective Tests of Penalized-Fit Shapley Contributions after Cross-Validation**. A candidate contribution statement is:

> We formulate zero mean-response contribution hypotheses for selected- and all-feature Shapley allocations in additive ridge models, characterize the selection event induced jointly by cross-validation and Shapley ranking, and construct conditional projection tests using analytic radial integration. We distinguish these coalition-contribution hypotheses from individual partial-effect hypotheses and examine their dependence on the underlying test spaces.

This is proposed framing, contingent on completing the proofs and numerical work. It avoids a priority claim and does not imply inference on the magnitude of a Shapley value. The numerical threshold issue must be reconciled with any ideal zero-contribution language in the final paper.



## Experiment context

The primary basis fraction is $\rho=0.10$, with $0.05$ and $0.20$ sensitivity settings, and $n\in\{25,50,75\}$. Synthetic experiments use $p=4$, $k=2$, and known $\sigma=1$. The observed-response benchmarks retain their native feature counts and use $\sigma=1$ as a working scale. Their rejection frequencies are descriptive working-model results, not known Type I error or power. These design choices and their numerical audits belong in the experimental section; they do not broaden the assumptions of (20).
