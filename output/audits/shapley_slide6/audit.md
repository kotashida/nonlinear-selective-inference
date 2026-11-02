# Audit of the RSS Shapley selection and inference algorithm

Audit date: October 8, 2026. Scope: the current working-tree implementation and printed slide 6 of presentation/261005_si_rd/presentation.pdf, which is PDF page 7 because the title page is unnumbered. I inspected the rendered PDF as well as its LaTeX source. No implementation files or existing experiment results were changed.

**Overall finding.** The slide correctly describes the intended RSS Shapley score, its zero-contribution null, the conditional chi-radius construction, and the AIS ratio used for the matched experiment. The ideal selective test is mathematically valid under its explicit assumptions. However, this is a test of Shapley mean contribution, which can be a global predictor-space signal null. It generally cannot certify that a selected feature has a nonzero partial coefficient. AIS is an approximation to the valid ideal p-value, not itself a finite-sample-valid p-value. I also found a concrete centering/normalization bug for constant decimal predictors.

**Evidence and status.** “Verified” below means checked against source, reproduced numerically, or established by the stated derivation. Assumptions about a real data-generating process are not established by tests. Reproductions and numeric results are in verify.py and verification.json beside this report.

## 1. Code-to-slide equivalence

The relevant implementation is **LinearShapleyStatistic** and **LinearShapleySelector**, not **ShapSelector**. The latter is a different random-forest/Tree-SHAP selector. The public API also permits other selector/statistic combinations, so the mere use of “SHAP” elsewhere in this repository does not identify this algorithm.

| Slide operation | Source | Assessment |
|---|---|---|
| Fixed X; Gaussian spherical noise; known sigma | src/si_shap/api.py:61, :145 | Assumptions are documented; positive sigma is required. The code cannot establish Gaussianity or that supplied sigma is the true noise SD. |
| RSS improvement with an intercept | src/si_shap/linear_shapley.py:50, :63 | Correct algebra on regular designs; column centering implements the intercept. Constant-decimal bug below is an exception. |
| Exact weights | src/si_shap/linear_shapley.py:75 | Code uses 1/[p binom(p-1,s)], algebraically identical to slide factorial weights. |
| Fixed 1,000-permutation approximation | src/si_shap/linear_shapley.py:82 | Matches the p=20 experiment, with seed 42. General auto mode uses exact enumeration when p<=10. |
| Top k=2 and one uniform target | src/si_shap/selection.py:94, :124, :661; power.py:495; null_calibration.py:455 | Matches the experiment: top-k ties go to smaller index, then a fresh uniform auxiliary draw picks from the sorted selected set. |
| Range projection and numerical rank | src/si_shap/linear_shapley.py:106; api.py:420 | Matches in exact arithmetic; floating-point eigenvalue thresholding changes numerical support near singularity. |
| Condition on complement and direction | src/si_shap/api.py:420-455; inference.py:150 | Matches, including sigma scaling. The local callback argument named z is a radius; the orthogonal component is stored as orthogonal. |
| Exact-set event | src/si_shap/selection.py:153 | Compares frozensets. It fixes membership, not the order among selected features. |
| Chi-radius AIS ratio and ESS failure | src/si_shap/inference.py:1681, :2046 | Same integral target and weighted ratio. Actual proposal construction is more detailed than the slide: pilot adaptation and deterministic-mixture sampling. Insufficient ESS produces NaN. |

The slide source is presentation/261005_si_rd/presentation.tex:173-220.

**Configuration differences from API defaults, not contradictions in the experiment.** Calling selective_inference without event/target/method options defaults to same_target, uniform_from_selected, and conditional_mc. Supplying exact_set without a target rule defaults to all_selected. To reproduce slide 6, explicitly use:

~~~python
selector = LinearShapleySelector(
    mode="monte_carlo",
    n_permutations=1000,
    random_state=42,
    fit_intercept=True,
)
result = selective_inference(
    X, y, k_select=2, sigma=known_sigma,
    selector=selector,
    test_statistic="linear_shapley",
    selection_event="exact_set",
    target_rule="uniform_from_selected",
    auxiliary_u=fresh_independent_uniform_draw,
    inference_method="ais",
    ais_strategy="a_robust_draw_floor",
    pilot_iters=10, pilot_samples=320,
    final_batch_size=800, max_final_samples=51200,
    min_denominator_ess=80, min_tail_ess=15,
    stop_when_ess_met=True,
)
~~~

The experiment manifest and saved settings support these choices. In ordinary API calls, reusing the same target_seed generates the same auxiliary quantile each time; fresh random targeting across repeated datasets requires fresh auxiliary draws or seeds. The experiment does this.

**Notation discrepancy.** The slide defines exact A_j and then reuses A_j for the fixed approximation. Write A_j for the exhaustive matrix and \(\widehat A_j\) for the finite-permutation matrix. The algorithm actually selects and tests the latter in this experiment. Their ranks and null sets need not coincide.

**Historical limitation.** The saved source hashes match the current linear_shapley.py, selection.py, and api.py. The hashes for inference.py, power.py, and null_calibration.py differ. Thus I verified current logic, saved configurations, and saved outputs; I cannot certify byte-for-byte identity of the entire historical execution with today's working tree. The AIS/exact data-file hashes do match their comparison manifest.

## 2. First principles: least squares as projection

Fix \(X=[x_1,\ldots,x_p]\in\mathbb R^{n\times p}\). For a coalition \(S\subseteq[p]\), define

\[
D_S=[\mathbf1,X_S],\qquad
\mathcal L_S=\operatorname{col}(D_S),\qquad
P_S=D_SD_S^+.
\]

Here \(+\) denotes the Moore-Penrose pseudoinverse. \(P_S\) is the orthogonal projector onto the fitted-value space, even if coefficients are not identifiable. The least-squares fitted response is \(P_Sy\), so

\[
\operatorname{RSS}_y(S)
=\|(I-P_S)y\|^2
=y^\top(I-P_S)y,
\]

because \(I-P_S\) is symmetric and idempotent. The empty coalition still has an intercept: \(P_\varnothing=P_{\mathbf1}\).

For \(j\notin S\), residualize its predictor against the coalition:

\[
v_{j|S}=(I-P_S)x_j.
\]

If \(v_{j|S}\ne0\), then

\[
\mathcal L_{S\cup\{j\}}
=\mathcal L_S\oplus\operatorname{span}(v_{j|S}),
\]

with an orthogonal direct sum. Consequently,

\[
P_{S\cup\{j\}}-P_S
=\frac{v_{j|S}v_{j|S}^\top}{v_{j|S}^\top v_{j|S}}
=q_{j|S}q_{j|S}^\top,\quad
q_{j|S}=\frac{v_{j|S}}{\|v_{j|S}\|}.
\]

If the residual is zero, the projector difference is zero. Therefore,

\[
\boxed{
\Delta_j(S;y)
=\operatorname{RSS}_y(S)-\operatorname{RSS}_y(S\cup\{j\})
=(q_{j|S}^\top y)^2\ge0
}
\]

for nonzero residuals, and zero otherwise.

Intuitively, each coalition asks how much of the response lies along the part of feature j that this particular coalition cannot explain. Different coalitions give different residual directions.

In the code, **coalition_basis(mask)** constructs an orthonormal coalition basis. **add(j, mask, weight)** subtracts the coalition projection from feature j, repeats the subtraction to reduce roundoff, normalizes the residual, and adds its weighted outer product. This directly implements the displayed projector increment.

## 3. Why these are exactly Shapley weights

For a uniform random ordering \(\pi\) of the p features, let \(S_j^\pi\) be j's predecessors. Exactly \(s!(p-s-1)!\) orderings have a specified predecessor set S of size s: order S first, place j next, and order the remaining features afterward. Hence

\[
\Pr(S_j^\pi=S)
=\frac{s!(p-s-1)!}{p!}
=\frac{1}{p\binom{p-1}{s}}
=:w(S).
\]

The weights sum to one:

\[
\sum_{S\subseteq-j}w(S)
=\sum_{s=0}^{p-1}\binom{p-1}{s}
 \frac{1}{p\binom{p-1}{s}}
=1.
\]

Define the game

\[
g_y(S)=\operatorname{RSS}_y(\varnothing)-\operatorname{RSS}_y(S).
\]

Its Shapley value for j is

\[
T_j(y)=\sum_{S\subseteq-j}w(S)
[g_y(S\cup\{j\})-g_y(S)]
=y^\top A_jy,
\]

where

\[
\boxed{A_j=\sum_{S\subseteq-j}w(S)
(P_{S\cup\{j\}}-P_S).}
\]

The sign and normalization on slide 6 are correct.

For L sampled permutations the implemented matrix is

\[
\widehat A_j=\frac1L\sum_{b=1}^L
(P_{S_j^{\pi_b}\cup\{j\}}-P_{S_j^{\pi_b}}).
\]

It is unbiased over the permutation sampling: \(E_\pi[\widehat A_j]=A_j\). This does not make a realized approximation exact, nor make its realized top-k ranking equal to the exhaustive ranking.

Permutations are fixed independently of the response and reused for every candidate. This is essential: resampling them for each candidate would change the selection map being conditioned on.

## 4. Matrix structure and what rank means

Every increment is positive semidefinite (PSD). Thus A_j and \(\widehat A_j\) are PSD. For the exact matrix,

\[
h^\top A_jh
=\sum_Sw(S)(q_{j|S}^\top h)^2.
\]

With all weights positive,

\[
\ker(A_j)=\bigcap_{\text{nonzero }v_{j|S}}v_{j|S}^{\perp},
\qquad
\operatorname{range}(A_j)
=\operatorname{span}\{v_{j|S}:S\subseteq-j\}.
\]

The second identity follows by taking orthogonal complements of the first. Equivalently, A_j is \(BB^\top\), where B has columns \(\sqrt{w(S)}q_{j|S}\).

Let \(C=I-P_{\mathbf1}\) and \(V=\operatorname{col}(CX)\). All residual directions lie in V, so \(A_j\mathbf1=0\) and \(\operatorname{range}(A_j)\subseteq V\).

If \([\mathbf1,X]\) has full column rank, every increment adds one dimension and has trace one. Therefore \(\operatorname{tr}(A_j)=1\). Rank can nevertheless exceed one: averages of differently oriented rank-one projections are usually not themselves projections.

For each permutation the increments telescope:

\[
\sum_j(P_{S_j^\pi\cup\{j\}}-P_{S_j^\pi})
=P_{[p]}-P_\varnothing=P_V.
\]

Averaging gives

\[
\boxed{\sum_jA_j=P_V,\qquad \sum_j\widehat A_j=P_V.}
\]

Thus the observed scores divide the full-model explained sum of squares:

\[
\sum_jT_j(y)=y^\top P_Vy
=\operatorname{RSS}_y(\varnothing)-\operatorname{RSS}_y([p]).
\]

It also follows that \(0\preceq A_j\preceq P_V\preceq I\). For rank-deficient designs, some increments vanish and individual traces can be below one. Duplicates can share credit even though they add no direction to a coalition that already contains their alias.

The code stores reduced matrices \(B_j=H^\top A_jH\) in an orthonormal basis H of V. It evaluates

\[
T_j(y)=(H^\top y)^\top B_j(H^\top y).
\]

This is an algebraically faithful storage optimization. Score evaluation costs \(O(nr+pr^2)\), with storage \(O(nr+pr^2)\), where \(r=\dim V\). Exact construction remains exponential in p.

## 5. Observed attribution is not a coefficient

Under the model

\[
Y=\mu+\varepsilon,\qquad \varepsilon\sim N_n(0,\sigma^2I),
\]

define the noiseless attribution

\[
\theta_j(\mu)=\mu^\top A_j\mu.
\]

Expanding the observed score gives

\[
T_j(Y)=\mu^\top A_j\mu+
2\mu^\top A_j\varepsilon+\varepsilon^\top A_j\varepsilon.
\]

Since \(E\varepsilon=0\) and \(E(\varepsilon\varepsilon^\top)=\sigma^2I\),

\[
\boxed{E_\mu[T_j(Y)]
=\theta_j(\mu)+\sigma^2\operatorname{tr}(A_j).}
\]

Gaussian quadratic-form calculations also give

\[
\operatorname{Var}_\mu[T_j(Y)]
=2\sigma^4\operatorname{tr}(A_j^2)
+4\sigma^2\mu^\top A_j^2\mu.
\]

For the variance identity, diagonalize A_j, write independent normal coordinates, use \(\operatorname{Var}(G^2)=2\), and note that the centered quadratic and linear terms have zero covariance by vanishing odd Gaussian moments.

Under full column rank, the noise offset is sigma squared for each feature. Subtracting this common known offset preserves the ranking, but does not convert attribution into a partial effect. Dividing by the centered total sum of squares, when nonzero, gives a Shapley decomposition of training R-squared; it does not establish out-of-sample predictive performance.

Top-k Shapley selection is a well-defined feature-ranking rule. It distributes shared explanatory credit; it does not prove support recovery, optimal subset prediction, unique conditional relevance, or causal relevance. It always selects k features, including under no signal, and can select redundant proxies. Those are limitations of its interpretation, not mistakes in the Shapley formula.

This refitted training-RSS game is also distinct from SAGE's restricted fixed-model expected-loss game. The SAGE paper discusses Shapley Net Effects with refitted linear subsets as a related approach. [Covert, Lundberg, and Lee (2020)](https://papers.nips.cc/paper/2020/file/c7bf0b7c1a86d5eb3be2c722cf2cf746-Paper.pdf).

## 6. The actual null hypothesis

PSD is the crucial assumption behind the slide's equivalences:

\[
\theta_j(\mu)=\mu^\top A_j\mu
=\|A_j^{1/2}\mu\|^2.
\]

Let \(W_j=\operatorname{range}(A_j)\) and \(Q_j=P_{W_j}\). Then

\[
\boxed{
\theta_j(\mu)=0
\iff A_j\mu=0
\iff Q_j\mu=0.
}
\]

This is correct. It is a simultaneous zero-contribution null over every positively weighted residual direction. Positive and negative coalition contributions cannot cancel because every contribution is a square.

For exact Shapley weights, the null requires zero signal along every coalition-specific direction. For the permutation approximation it requires this only for sampled directions. In exact arithmetic,

\[
\operatorname{range}(\widehat A_j)
\subseteq\operatorname{range}(A_j),\qquad
\ker(A_j)\subseteq\ker(\widehat A_j).
\]

Thus the approximate zero-contribution null can be larger. One sampled permutation gives a rank-one matrix for each nonredundant feature, whereas exhaustive averaging can give full rank. In the independent reproduction, all four one-permutation matrices had rank one.

Under a full-rank linear mean model \(\mu=\beta_0\mathbf1+X\beta\), the exact Shapley null implies \(\beta_j=0\), since it includes the full-adjustment coalition \(-j\). The converse generally fails. Rejecting the stronger Shapley null therefore does not reject the broader coefficient null.

For arbitrary mu, which the range test permits, the full-model projection coefficient is the relevant coefficient analogue. There is no need to assume mu is linear to justify the range test.

## 7. Complete two-feature counterexample

Take centered unit vectors x_1,x_2 with \(x_1^\top x_2=\rho\), \(|\rho|<1\). For feature 2,

\[
A_2=\frac12x_2x_2^\top+
\frac12\frac{(x_2-\rho x_1)(x_2-\rho x_1)^\top}{1-\rho^2}.
\]

If \(\mu=\beta_0\mathbf1+\beta_1x_1+\beta_2x_2\), then

\[
\boxed{
\theta_2(\mu)=
\frac12(\rho\beta_1+\beta_2)^2+
\frac12(1-\rho^2)\beta_2^2.
}
\]

The first term is the marginal coalition; the second is the coalition adjusting for feature 1. If \(\beta_2=0\), the attribution is \(\rho^2\beta_1^2/2\), positive whenever \(\rho\beta_1\ne0\).

Let \(e=(x_2-\rho x_1)/\sqrt{1-\rho^2}\). In the orthonormal basis \((x_1,e)\),

\[
A_2|_V=\frac12
\begin{pmatrix}
\rho^2 & \rho\sqrt{1-\rho^2}\\
\rho\sqrt{1-\rho^2} & 2-\rho^2
\end{pmatrix}.
\]

Its trace is one and determinant is \(\rho^2/4\). Its eigenvalues are

\[
\lambda_\pm=\frac{1\pm\sqrt{1-\rho^2}}2.
\]

For \(0<|\rho|<1\), both are positive. Thus W_2=V and

\[
\theta_2(\mu)=0\iff\beta_1=\beta_2=0.
\]

The same holds for feature 1. Different attributions can have exactly the same zero set. For example,

\[
T_1(y)-T_2(y)=(x_1^\top y)^2-(x_2^\top y)^2,
\]

so their rankings can differ even while their nulls coincide.

The slide's example \(\rho=0.7,\beta_1=6,\beta_2=0\) gives \(\theta_2=8.82\), independently reproduced as 8.819999999999991. A high rejection rate for feature 2 here detects predictor-space signal; it is not a false positive for the Shapley null, but would be a false positive if mislabeled as a coefficient test.

At rho=0, A_2 is rank one and its null is \(\beta_2=0\). At \(|\rho|=1\), full rank fails and separate partial coefficients are not identifiable.

## 8. Why global nulls arise even with independent population predictors

Write \(\tilde x_j=Cx_j\). The empty and singleton coalitions give

\[
v_{j|\varnothing}=\tilde x_j,\quad
v_{j|\{\ell\}}=
\tilde x_j-
\frac{\tilde x_\ell^\top\tilde x_j}{\|\tilde x_\ell\|^2}\tilde x_\ell.
\]

If \(\tilde x_\ell^\top\tilde x_j\ne0\), their difference is a nonzero multiple of \(\tilde x_\ell\). Both residuals lie in W_j, so \(\tilde x_\ell\in W_j\). Consequently, if j has nonzero empirical inner product with every other nonconstant feature,

\[
W_j=V.
\]

This is sufficient, not necessary. It is not a claim that any amount of correlation in any design makes every W_j equal V.

For independent continuous Gaussian columns, empirical centered correlations are almost surely nonzero. Thus, with n sufficiently large for full rank, the exact Shapley ranges are almost surely full predictor ranges, even when population feature correlation is zero. Population independence is not finite-sample orthogonality.

The finite-permutation approximation needs its own rank check. Reconstructing the saved fixed null design from seed 3405814020 with the fixed 1,000 permutations gave rank 20 for all 20 matrices, with smallest reduced eigenvalue approximately \(6.126\times10^{-5}\). This is far above the support threshold, approximately \(2.22\times10^{-14}\). The recorded target ranks are also 20 in all 1,500 saved trials.

Under W_j=V,

\[
H_{0,j}\iff P_V\mu=0.
\]

Under a full-rank linear mean model, this means beta=0, with the intercept unrestricted. For arbitrary mu, mean signal orthogonal to V remains possible. Therefore “global no signal” should be qualified as “no signal in the centered linear predictor space,” not “mu is zero” or “no relationship of any kind.”

## 9. Deriving the chi radius without assuming a chi-square score

Choose an orthonormal basis H_j of W_j, with dimension d_j>0. Under \(H_{0,j}:Q_j\mu=0\),

\[
G=H_j^\top Y/\sigma\sim N_{d_j}(0,I).
\]

The orthogonal component \(Z=(I-Q_j)Y\) is independent of G because their Gaussian cross-covariance is zero.

Write \(G=RD\), with R its norm and D its unit direction. In polar coordinates the standard normal density becomes

\[
(2\pi)^{-d_j/2}\exp(-r^2/2)\,
r^{d_j-1}\,dr\,d\omega.
\]

The Jacobian \(r^{d_j-1}\) accounts for spherical surface area. This factorization shows that direction is uniform on the sphere and independent of radius. Integrating over the sphere yields

\[
\boxed{
f_{\chi_{d_j}}(r)=
\frac{2^{1-d_j/2}}{\Gamma(d_j/2)}
r^{d_j-1}e^{-r^2/2},\quad r>0.
}
\]

The response-space direction is \(U=H_jD=Q_jY/\|Q_jY\|\). Thus R is independent of both U and Z under the null.

Conditioning on their observed values leaves exactly the ray

\[
Y(r)=z+\sigma ru,\quad r\ge0,\quad
r_{\mathrm{obs}}=\|Q_jy_{\mathrm{obs}}\|/\sigma.
\]

Since A_j z=0,

\[
T_j(Y(r))=\sigma^2r^2u^\top A_ju.
\]

A_j is positive definite on its range, so \(u^\top A_ju>0\). Therefore comparing target scores is equivalent to comparing nonnegative radii.

However, the unconditioned score distribution is

\[
T_j(Y)/\sigma^2\overset d=
\sum_{a=1}^{d_j}\lambda_{ja}G_a^2,
\]

a weighted chi-square, not generally \(\chi^2_{d_j}\). It is the radius that has the ordinary chi law. The slide is correct on this distinction.

Under alternatives, the unconditioned radius has a noncentral chi law. Conditional on direction its density need not be noncentral chi, since radius and direction need not remain independent. The central-null factorization is what justifies this procedure.

This conditional projection-radius construction is a specialization of the truncated-chi framework in Loftus and Taylor, Theorem 3.1. [Loftus and Taylor (2015)](https://arxiv.org/html/1511.01478).

## 10. Deriving selection conditioning and the ideal p-value

Let \(M(y)=\operatorname{Top}_k(T_1(y),\ldots,T_p(y))\), with deterministic ties, and m=M(y_obs). Condition on the sampled permutations and auxiliary target randomization as well as z,u,m. For exact-set conditioning define

\[
\mathcal C_j=\{r\ge0:M(z+\sigma ru)=m\}.
\]

Selecting a target uniformly from the sorted set can be expressed as

\[
J(m,a)=m_{(\lfloor ka\rfloor+1)},\qquad a\sim U(0,1).
\]

Once m and a are fixed, j is determined. Exact-set candidate checks therefore need no extra target indicator. If instead only the same_target event is conditioned on, the candidate event must reproduce the full selector-to-target map, which the code implements separately.

Bayes' rule gives

\[
f(r\mid z,u,m,H_{0,j})
=
\frac{f_{\chi_{d_j}}(r)\mathbf1_{\mathcal C_j}(r)}
{\int_{\mathcal C_j} f_{\chi_{d_j}}(s)\,ds}.
\]

The ideal selective upper-tail p-value is

\[
\boxed{
p_j=
\frac{\int_{\mathcal C_j\cap[r_{\mathrm{obs}},\infty)}
 f_{\chi_{d_j}}(r)\,dr}
{\int_{\mathcal C_j} f_{\chi_{d_j}}(r)\,dr}.
}
\]

For almost every conditioning value with positive denominator, this is uniform under the null by the conditional probability integral transform. Integrating over conditioning variables preserves the level guarantee. The distribution has no atoms when d_j>0 and the truncation set has positive probability; tied Monte Carlo ranks require the conservative convention discussed below.

The selective-inference adjustment addresses response-dependent selection. It does not change the null into a coefficient or causal hypothesis.

## 11. Selection regions are quadratic and can be integrated analytically

For selected a in m and unselected b outside m, top-k membership requires \(T_a(y)\ge T_b(y)\), with strictness determined by the tie rule. Write \(D_{ab}=A_a-A_b\). On the ray,

\[
T_a(Y(r))-T_b(Y(r))
=\alpha_{ab}r^2+\beta_{ab}r+\gamma_{ab},
\]

where

\[
\alpha_{ab}=\sigma^2u^\top D_{ab}u,\quad
\beta_{ab}=2\sigma u^\top D_{ab}z,\quad
\gamma_{ab}=z^\top D_{ab}z.
\]

Intersect these quadratic-inequality solution sets with \([0,\infty)\). The result is a finite union of intervals. Persistent ties, such as identical score matrices, need index-aware tie handling; one cannot simply dismiss every tie as a probability-zero event.

If \(\mathcal C_j=\bigcup_\ell[a_\ell,b_\ell]\), the denominator is

\[
\sum_\ell[F_{\chi_{d_j}}(b_\ell)-F_{\chi_{d_j}}(a_\ell)]
\]

and the numerator is the same interval mass above r_obs, omitting intervals below it. For extreme tails, survival-function differences are preferable to subtracting CDFs close to one.

This gives an analytic route for general fixed quadratic Shapley selection. The current API's exact_spline engine is restricted to spline screening and does not implement this general Shapley interval solver.

As a simple nontrivial example, take orthogonal centered unit predictors, p=2,k=1, and suppose feature 1 is selected. Conditioning on the complement fixes \(x_2^\top y\). Then

\[
\mathcal C_1=[|x_2^\top y|/\sigma,\infty),
\quad
p_1=
\frac{\Pr(\chi_1\ge|x_1^\top y|/\sigma)}
{\Pr(\chi_1\ge|x_2^\top y|/\sigma)}.
\]

For radii 2 and 1, respectively, the exact p-value is 0.1433935. The independent 100,000-proposal conditional-MC check returned 0.1419239. Here selection adjustment is material.

## 12. The full-range case: the selective correction disappears

A stronger and useful observation is that it is sufficient for the **target** range to equal V; the other matrices need only have their usual ranges contained in V.

If W_j=V, then z is orthogonal to V. For every feature h,

\[
T_h(z+\sigma ru)=\sigma^2r^2u^\top A_hu.
\]

For r>0 all scores receive the same positive factor. Their complete ranking, top-k set, and fixed-auxiliary target therefore remain constant. Radius zero has zero probability under a nondegenerate chi law. Hence

\[
\mathcal C_j=(0,\infty)\quad\text{up to null-probability boundaries},
\]

and

\[
\boxed{
p_j=\Pr(\chi_r\ge\|P_Vy\|/\sigma)
=\Pr(\chi_r^2\ge\|P_Vy\|^2/\sigma^2),
\quad r=\dim V.
}
\]

Every target with full range uses the same exact global p-value. Different AIS estimates for such targets would be Monte Carlo noise, not different underlying evidence.

This is why selecting and testing with the same quadratic score is not automatically invalid. Under this null the selected set depends only on direction, while the radius is independent of direction. Once direction is conditioned on, there is no further radial restriction.

The saved data have rank 20 for every tested target and all proposals retained their selection event. The exact-analysis builder computes chi2.sf(radial_t_obs**2, df=20), consistent with this derivation. All 1,500 saved exact p-values agree with this formula within \(1.11\times10^{-16}\).

“Exact” in those outputs refers to analytic inference conditional on the fixed permutation selector. It does not mean exhaustive Shapley coalition averaging.

## 13. AIS: correct integration target, approximate statistical calibration

For a frozen proposal q with positive density wherever the selected null density has mass, write

\[
D=\int f_\chi(r)\mathbf1_{\mathcal C_j}(r)\,dr,\quad
N=\int f_\chi(r)\mathbf1_{\mathcal C_j}(r)
\mathbf1\{r\ge r_{\mathrm{obs}}\}\,dr.
\]

If \(R_b\) are independent from q, importance sampling gives unbiased estimates of N and D separately:

\[
\widehat N=B^{-1}\sum_b
\frac{f_\chi(R_b)}{q(R_b)}
\mathbf1_{\mathcal C_j}(R_b)\mathbf1\{R_b\ge r_{\mathrm{obs}}\},
\]

and the analogous denominator. Their ratio is the slide's \(\widehat p_j\). A ratio of unbiased estimates is not generally unbiased, nor is its null distribution automatically super-uniform. With a frozen, adequately supported proposal and ordinary integrability assumptions, consistency follows from laws of large numbers.

The actual robust strategy uses pilot draws to fit a defensive mixture of chi and truncated-normal components, freezes the components, and schedules component counts deterministically for the declared total budget. Its denominator q is the aggregate mixture density, not merely the component that generated a particular draw. This is legitimate deterministic-mixture importance sampling at a fixed completed allocation. Because the component labels are shuffled, the sample also has a useful mixture-marginal interpretation; it is not literally an iid q sample as the slide shorthand might suggest.

ESS stopping makes the final sample size data dependent. An ordinary fixed-budget consistency or variance argument is not a finite-sample guarantee for that stopped ratio. The code appropriately labels AIS finite_sample_valid=False.

The reported diagnostics are

\[
\operatorname{ESS}_D=
\frac{(\sum_bW_b)^2}{\sum_bW_b^2},\quad
\operatorname{ESS}_{\mathrm{tail}}=
\frac{(\sum_{b:R_b\ge r_{\mathrm{obs}}}W_b)^2}
{\sum_{b:R_b\ge r_{\mathrm{obs}}}W_b^2},
\]

where selected-event weights W_b are used. Passing ESS thresholds does not prove all important regions were sampled, certify a numerical error bound, or establish Type I control. The code's normal-approximation Monte Carlo interval is likewise not a rigorous selective confidence bound. Small-tail ESS requirements can make very small p-values computationally expensive to resolve.

**Measured approximation error.** The verified paired outputs contain 1,500 distinct trials and 3,000 inference records, not 3,000 independent trials. Twelve decisions changed at alpha=0.05 between AIS and the exact reference. The largest absolute p-value difference was approximately 0.0634809. This verifies consequential numerical error on some trials, but does not by itself establish systematic anticonservatism or prove AIS has a particular Type I error rate.

In this experiment the full-range simplification removes any need for AIS for the Shapley test. The analytic chi-square tail is preferable.

## 14. Other engines: what their validity does and does not imply

**Conditional MC.** The implementation draws a fixed number of iid chi radii and retains those reproducing the event. Conditional on the retained count K, retained radii are iid from the truncated reference, independently of the observed radius under the selective null. Consequently,

\[
p_{\mathrm{MC}}=
\frac{1+\sum_{b=1}^K\mathbf1\{R_b\ge r_{\mathrm{obs}}\}}{K+1}
\]

is a conservative finite-sample rank p-value. The plus one includes the observation. With K=0, p=1 is valid but uninformative. The guarantee assumes correct event evaluation and genuinely independent sampling in the probability model; floating-point arithmetic introduces the usual implementation approximations.

**Parallel reversible MCMC rank.** The kernel combines a symmetric reflected Gaussian proposal with Metropolis acceptance and a global chi proposal accepted when selected. Both are reversible for the selected chi density. The code first draws a hub H from \(K^L(r_{\mathrm{obs}},\cdot)\), then draws independent replicas from \(K^L(H,\cdot)\). Detailed balance implies that, under stationarity, the observed radius conditional on H has that same replica law. The observation and replicas are exchangeable conditional on H, so the plus-one upper rank is valid without requiring rapid mixing. Mixing affects power and resolution.

The MCMC diagnostic binomial interval estimates a hub-conditional tail, not necessarily the ideal stationary selective tail; the exchangeability proof validates the rank p-value, not an assertion that each short-run replica is an independent stationary draw.

Neither engine validates mislabeled coefficient hypotheses or incorrect data-model assumptions.

## 15. Concrete implementation bug: decimal constants become signal directions

The sequence in linear_shapley.py:50-52 is

~~~python
design = X - X.mean(axis=0)
norms = np.linalg.norm(design, axis=0)
design = np.divide(design, norms, out=np.zeros_like(design),
                   where=norms > 0)
~~~

A nonzero centering residual is normalized even when it is entirely roundoff.

Reproduction:

~~~python
X = np.full((100, 1), 0.1)
y = np.ones(100)
s = LinearShapleyStatistic(X)
s.scores(y)             # about [100.0]
s.support_basis(0).shape[1]  # 1
~~~

The computed mean is 0.09999999999999998. Subtracting it leaves a small positive constant vector, which normalization turns into the unit intercept direction. Mathematically, X adds nothing to the intercept-only model, so every RSS improvement, matrix, score, and rank should be zero.

This is a substantive departure from the slide's algorithm. For a constant nonzero response mean, the intended no-feature-signal null permits that intercept, whereas the erroneous direction tests it. Calibration for the intended null can therefore fail badly; a global-zero-mean simulation would not expose this interpretation error.

The existing tests use exactly representable constants such as 1 and do not catch the decimal-constant case. A repair should detect constant columns before centering/scaling, zero their centered columns exactly, and use a documented scale-aware treatment for nearly constant columns. Merely relying on later SVD tolerance is inadequate because the noise has already been scaled to unit magnitude. Large offsets and very small variation also deserve numerical checks.

No repair was made because this task requested an analysis. The reproduction is included in verify.py.

## 16. Numerical-rank and statistical limitations

**Numerical support differs from exact support.** support_basis keeps eigenvalues above

\[
\max(n,p)\epsilon_{\mathrm{mach}}
\max(1,\lambda_{\max}).
\]

For the two-feature example, \(\lambda_-\sim\rho^2/4\) as rho approaches zero. The mathematical rank is two for every nonzero rho, but the numerical support switches to rank one once the small eigenvalue crosses the threshold. The reproduction shows rank two at rho=1e-5 and rank one at rho=1e-8.

The score matrix itself is not spectrally truncated. If \(\widetilde Q_j\) projects onto only retained eigenvectors, then generally \(A_j(I-\widetilde Q_j)\ne0\). Thus the slide's literal zero-intercept identity becomes, in ideal arithmetic for that spectral split,

\[
T_j(z+\sigma ru)=z^\top A_jz+
\sigma^2r^2u^\top A_ju.
\]

The cross term vanishes because retained and discarded eigenspaces are orthogonal invariant subspaces. The score remains increasing in radius, but the claimed exact range and null equivalences no longer literally describe the numerical projector.

If the retained space is truly a subspace of the exact intended range, calibration under the stronger intended null remains valid; dropping a small direction does not automatically make the test anticonservative. It does make the tested subspace, power, and claimed equality of nulls sensitive to tolerance. The decimal-constant bug is more serious because it creates a direction outside the intended centered design span.

**Model assumptions.** Spherical Gaussian noise and externally known sigma are required by this derivation. Heteroscedasticity, dependence, nonnormal noise, or plugging in an estimated sigma invalidate the exact chi derivation unless a new calibration is supplied. An independently estimated sigma still has uncertainty and cannot simply be treated as known without adjustment. Random X is permissible through conditioning on X when the conditional Gaussian model is correct.

**Preprocessing and tuning.** X-only preprocessing can be fixed by conditioning on X. Response-dependent transformations, tuning, choice of k, inspection-based method choice, or choosing the “best” permutation seed must be included in the selection map or handled by independent data. Merely fixing a response-dependent choice after seeing data does not remove selection.

**Feature-space limitation.** Even an arbitrary nonlinear mean is allowed by the fixed-design Gaussian model. But the test sees only its projection onto the linear design span. A nonlinear component orthogonal to V is invisible. The simulation effect \(x+0.5(x^2-1)\), standardized empirically, includes a linear component; success there does not establish sensitivity to arbitrary nonlinear effects.

**Multiplicity.** Slide 6 tests one randomized target, giving one selective decision. Testing all selected targets requires a separate simultaneous-error argument. Under common exact-set conditioning, valid featurewise p-values permit Bonferroni/Holm control. These corrections cannot make approximate AIS p-values exact. When all hypotheses coincide with the same global null, reporting many “significant features” still supplies only evidence for that common null.

**Calibration plots.** At alpha=0.05 the saved 300 null trials have 15 rejections for both inference methods, with a Wilson interval approximately [0.0305,0.0808]. This is compatible with nominal calibration but cannot prove exact control, diagnose all alternatives, or validate assumptions on real data. Identical empirical rejection counts can conceal different individual decisions.

**Power definitions.** With a fresh uniform target from k=2 features,

\[
\Pr(J=1)=\tfrac12\Pr(1\in M(Y))\le\tfrac12,
\]

and

\[
\Pr(J=1,\mathcal R)=\Pr(J=1)\Pr(\mathcal R\mid J=1).
\]

The slide's factorization and 50% targeting ceiling are correct. In the full-range case, \(\mathcal R\) is global predictor-space rejection. These metrics separate targeting the generating feature from detecting global signal; they do not establish that the selected feature has a nonzero partial effect. Under alternatives, targeting and radius need not be independent, so target-conditional global power need not equal unconditional global power.

## 17. How to obtain individual-feature inference while retaining this selector

Selection and inferential target need not use the same statistic. If the scientific question is an individual full-model coefficient, retain Shapley top-k selection but use the fully adjusted direction

\[
e_j=\frac{(I-P_{-j})x_j}{\|(I-P_{-j})x_j\|}.
\]

Under full column rank and \(\mu=\beta_0\mathbf1+X\beta\),

\[
e_j^\top\mu=\beta_j\|(I-P_{-j})x_j\|.
\]

Thus \(H_0:\beta_j=0\) is equivalent to \(e_j^\top\mu=0\). Conditional on \(z_j=(I-e_je_j^\top)Y\), the signed coordinate

\[
t=e_j^\top Y/\sigma\sim N(0,1)
\]

under that null, and candidates are \(z_j+\sigma t e_j\). Reevaluate the same Shapley selector and restrict t to the selected event. A conditional absolute-value tail,

\[
\frac{\int_{\mathcal D_j}\phi(t)\mathbf1\{|t|\ge|t_{\mathrm{obs}}|\}\,dt}
{\int_{\mathcal D_j}\phi(t)\,dt},
\]

is a valid two-sided selected test, where \(\mathcal D_j\) is the event-preserving subset of the signed line. Under arbitrary mu, the analogous target is the full-model projection coefficient. This would be a different inferential procedure; it is not achieved by relabeling the current chi-radius null.

## 18. Verification record and recommended priorities

All 52 existing checks passed: 17 in tests/test_linear_shapley.py, 14 in tests/test_inference.py, and 21 in tests/test_api.py. Passing these checks did not rule out the independently reproduced decimal-constant bug. Independent checks compared the implemented matrices to explicit coalition projectors with maximum entrywise error \(2.22\times10^{-16}\). They also verified permutation efficiency, reproduced full ranges on the actual saved fixed null design, checked invariant selected sets on its rays, reproduced the correlated-feature example, demonstrated finite-permutation range differences, and checked nontrivial orthogonal selection truncation.

The saved data hashes match their manifest; all 1,500 exact reference values reproduce the chi-square formula; all recorded AIS proposals preserve selection; and the 12 changed decisions were independently counted. These are source and numerical checks, not proofs of real-data assumptions or every historical execution detail.

Priorities:

1. Repair the decimal-constant centering/scaling bug and add a regression check.
2. Use analytic chi-square tails whenever the target range is V.
3. Distinguish exact A_j from fixed permutation \(\widehat A_j\), and document numerical support.
4. Describe the method as RSS Shapley ranking followed by selective testing of zero mean contribution; label full-range results as global predictor-space tests.
5. Use exact integration or valid rank engines when a formal level guarantee is needed. Treat AIS results and ESS-based decisions as numerical approximations.
6. If individual partial coefficients are the scientific target, derive and implement the different signed-coordinate test while retaining Shapley selection if desired.

The central mathematical construction is sound. Its useful scope is narrower than “selected features are individually significant,” and the implementation needs the specific numerical repair identified above.
