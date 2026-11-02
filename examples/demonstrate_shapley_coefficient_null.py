"""Reproduce item 1 on slide numbered 7 of the 2026-10-05 discussion.

Run from the repository root with .venv/Scripts/python.exe
examples/demonstrate_shapley_coefficient_null.py. No fitted variance or AIS
approximation is needed: sigma is known and p=k=2 makes selection automatic.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import chi2, norm

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from si_shap.linear_shapley import LinearShapleyStatistic


def design(n, seed):
    rng = np.random.default_rng(seed)
    z = rng.normal(size=(n, 2))
    z -= z.mean(axis=0)
    return np.linalg.qr(z)[0]


def rss(y, x):
    """Independent OLS calculation, including the intercept."""
    d = np.column_stack((np.ones(len(y)), x))
    residual = y - d @ np.linalg.lstsq(d, y, rcond=None)[0]
    return float(residual @ residual)


def wilson(count, total):
    z = norm.ppf(0.975)
    p = count / total
    center = (p + z*z/(2*total)) / (1 + z*z/total)
    half = z*np.sqrt(p*(1-p)/total + z*z/(4*total*total))/(1+z*z/total)
    return center-half, center+half


def write_csv(path, rows):
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repetitions", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=20261002)
    parser.add_argument("--output-dir", type=Path,
                        default=Path("outputs/shapley_coefficient_null_demo"))
    args = parser.parse_args()
    if args.repetitions < 2:
        parser.error("--repetitions must be at least 2")
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    n, sigma, alpha, beta_demo = 100, 1., .05, 6.
    q = design(n, args.seed)
    # Shared noise across settings reduces noise in comparisons.
    errors = np.random.default_rng(args.seed + 1).normal(size=(n, args.repetitions))
    correlations = [-.9, -.7, -.4, 0., .4, .7, .9]
    scores_rows, rate_rows, raw = [], [], {}
    max_identity_error = 0.
    for rho in correlations:
        x = np.column_stack((q[:, 0], rho*q[:, 0] + np.sqrt(1-rho*rho)*q[:, 1]))
        stat = LinearShapleyStatistic(x, mode="exact")
        mu = beta_demo*x[:, 0]
        delta_empty = rss(mu, np.empty((n, 0))) - rss(mu, x[:, [1]])
        delta_given1 = rss(mu, x[:, [0]]) - rss(mu, x)
        target = stat.scores(mu)[1]
        theory = .5*beta_demo**2*rho**2
        np.testing.assert_allclose(x.T@x, [[1, rho], [rho, 1]], atol=1e-12)
        np.testing.assert_allclose(x.mean(axis=0), 0, atol=1e-12)
        np.testing.assert_allclose(target, theory, atol=1e-10)
        np.testing.assert_allclose(target, .5*(delta_empty+delta_given1), atol=1e-10)
        np.testing.assert_allclose(np.linalg.lstsq(x, mu, rcond=None)[0], [beta_demo, 0], atol=1e-10)
        max_identity_error = max(max_identity_error, abs(target-theory))
        y = mu[:, None]+sigma*errors
        coordinates = stat.basis.T@y
        noisy_scores = np.einsum("ai,ab,bi->i", coordinates, stat.reduced_matrices[1], coordinates)
        noise_baseline = sigma**2*np.trace(stat.reduced_matrices[1])
        corrected = noisy_scores-noise_baseline
        scores_rows.append(dict(rho=rho, beta1=beta_demo, beta2=0.,
                               delta2_empty=delta_empty, delta2_given_feature1=delta_given1,
                               shapley_signal=target, theoretical_signal=theory,
                               noise_baseline=noise_baseline, mean_noise_corrected_score=float(corrected.mean()),
                               mc_se=float(corrected.std(ddof=1)/np.sqrt(args.repetitions))))
        if rho not in (0., .7):
            continue
        support = stat.support_basis(1)
        rank = support.shape[1]
        assert rank == (1 if rho == 0 else 2)
        # The radial selective p-value is chi.sf(r/sigma, rank).
        # With both features selected the exact-set event imposes no truncation.
        # On each fixed direction T_2 grows with r^2, so this is the pipeline's
        # exact score-tail test, rather than the unconditional tail of Y'A_2Y.
        inverse_gram = np.linalg.inv(x.T@x)
        coef2_se = sigma*np.sqrt(inverse_gram[1, 1])
        for beta in [0., 2., 4., 6., 8., 10.]:
            y = beta*x[:, 0, None]+sigma*errors
            radius_squared = np.sum((support.T@y)**2, axis=0)/sigma**2
            shapley_p = chi2.sf(radius_squared, rank)
            coef2 = (inverse_gram@x.T@y)[1]
            coef_p = 2*norm.sf(np.abs(coef2/coef2_se))
            for name, pvalues in [("Shapley radial test", shapley_p), ("Full-model coefficient test", coef_p)]:
                count = int(np.count_nonzero(pvalues <= alpha))
                low, high = wilson(count, args.repetitions)
                rate_rows.append(dict(rho=rho, beta1=beta, beta2=0., method=name,
                                      support_rank=rank, rejections=count, repetitions=args.repetitions,
                                      rejection_rate=count/args.repetitions, ci95_low=low, ci95_high=high))
                raw[f"rho{rho}_beta{beta}_{'shapley' if name.startswith('Shapley') else 'coefficient'}"] = pvalues
    write_csv(out/"signal_decomposition.csv", scores_rows)
    write_csv(out/"rejection_rates.csv", rate_rows)
    np.savez_compressed(out/"pvalues.npz", **raw)
    plt.rcParams.update({"font.size": 11, "axes.spines.top": False,
                         "axes.spines.right": False, "pdf.fonttype": 42})
    blue, orange, green = "#3B36C4", "#F68B1F", "#258568"
    fig, axes = plt.subplots(1, 2, figsize=(11.8, 4.6), layout="constrained")
    fig.suptitle(r"Feature 2 has zero coefficient, but positive Shapley signal", fontsize=17)
    rho_grid = np.linspace(-.95, .95, 201)
    axes[0].plot(rho_grid, .5*beta_demo**2*rho_grid**2, color=blue, label=r"Theory: $\beta_1^2\rho^2/2$")
    axes[0].errorbar(correlations, [r["mean_noise_corrected_score"] for r in scores_rows],
                     yerr=[1.96*r["mc_se"] for r in scores_rows], fmt="o", color=orange,
                     capsize=4, label=r"Simulation: mean $T_2(Y)-\sigma^2\mathrm{tr}(A_2)$")
    axes[0].set(xlabel=r"Correlation $\rho$", ylabel="Feature 2 Shapley signal", title=r"$n=100$, unit-length columns, $\mu=6x_1$")
    axes[0].legend(fontsize=9, loc="upper center")
    row = next(r for r in scores_rows if r["rho"] == .7)
    # RSS differences of order 1e-29 are floating-point zero; avoid placing
    # a negative-zero bar label below the axis into the tick text.
    values = [max(0., row["delta2_empty"]), max(0., row["delta2_given_feature1"]), row["shapley_signal"]]
    bars = axes[1].bar([0, 1, 2], values, color=[orange, green, blue], width=.65)
    axes[1].set_xticks([0, 1, 2], ["Add feature 2\nto intercept only", "Add feature 2\nafter feature 1", "Shapley score\n(average)"])
    axes[1].set(ylabel="RSS improvement on the noiseless mean", title=r"Why it happens: $\rho=0.7$, $\beta_1=6$, $\beta_2=0$", ylim=(0, 22))
    axes[1].bar_label(bars, labels=[f"{v:.2f}" for v in values], padding=5)
    for ax in axes:
        ax.grid(axis="y", alpha=.15)
        ax.set_axisbelow(True)
    fig.savefig(out/"signal_decomposition.png", dpi=200)
    fig.savefig(out/"signal_decomposition.pdf")
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(11.8, 4.6), layout="constrained", sharey=True)
    fig.suptitle(r"Different null hypotheses give different rejection behavior ($\beta_2=0$ throughout)", fontsize=15)
    for ax, rho in zip(axes, [0., .7]):
        for name, color, marker in [("Shapley radial test", blue, "o"), ("Full-model coefficient test", orange, "s")]:
            rows = [r for r in rate_rows if r["rho"] == rho and r["method"] == name]
            rates = np.array([r["rejection_rate"] for r in rows])
            ax.errorbar([r["beta1"] for r in rows], rates,
                        yerr=[rates-np.array([r["ci95_low"] for r in rows]), np.array([r["ci95_high"] for r in rows])-rates],
                        color=color, marker=marker, capsize=3, label=name)
        ax.axhline(alpha, color="gray", linestyle="--", linewidth=1, label="5% reference")
        ax.set(xlabel=r"Signal strength $\beta_1=\|\mu\|$", ylim=(0, 1.04),
               title="Orthogonal features ($\\rho=0$)" if rho == 0 else "Correlated features ($\\rho=0.7$)")
        ax.grid(alpha=.15)
    axes[0].set_ylabel("Rejection probability for feature 2")
    axes[0].legend(loc="upper left", fontsize=9)
    fig.savefig(out/"rejection_comparison.png", dpi=200)
    fig.savefig(out/"rejection_comparison.pdf")
    plt.close(fig)
    metadata = dict(seed=args.seed, n=n, p=2, k=2, sigma=sigma, alpha=alpha,
                    repetitions=args.repetitions, beta_demo=beta_demo,
                    max_formula_error=max_identity_error,
                    selection="exact set; automatic because k=p=2",
                    confidence_intervals="pointwise 95% Wilson intervals; shared noise across settings")
    (out/"settings.json").write_text(json.dumps(metadata, indent=2)+"\n", encoding="utf-8")
    example = [r for r in rate_rows if r["rho"] == .7 and r["beta1"] == 6.]
    lines = ["# Zero coefficient versus the Shapley null", "",
             "This experiment illustrates item 1 on the slide numbered 7 (physical PDF page 8).", "",
             f"Fixed centered unit-length design: n=100, p=k=2, sigma=1; {args.repetitions:,} independent Gaussian responses per setting; seed={args.seed}.", "",
             "The response is Y = beta1*x1 + epsilon. The true full-model coefficient beta2 is exactly zero at every setting. We use exact Shapley matrices, not sampled permutations.", "",
             "At rho=0.7 and beta1=6, adding feature 2 to the intercept-only model improves noiseless RSS by 17.64; adding it after feature 1 improves RSS by zero. Their Shapley-weighted average is 8.82, matching beta1^2*rho^2/2.", "",
             "The first plot compares the formula with simulated noise-corrected scores. Raw noisy scores have a positive noise baseline sigma^2*trace(A2), equal to 1 here, even when the Shapley null holds. Error bars are 1.96 Monte Carlo standard errors.", "",
             "The second plot compares two known-sigma tests. The coefficient test is a two-sided normal test of beta2=0 in the full model. The Shapley test uses the pipeline's radial null: chi-square survival of ||P_range(A2)Y||^2/sigma^2 with rank(A2) degrees of freedom. Conditioning on the orthogonal component and projection direction leaves this exact radial distribution under the Shapley null. Both features are always selected, so there is no selection truncation or AIS estimation error.", "",
             "For nonzero correlation the support of A2 is the full two-feature span, so this radial test detects signal from feature 1 too. At rho=0 the support is only feature 2 and both tests remain calibrated. This design isolates the null mismatch; it does not benchmark nontrivial feature selection.", "",
             "At rho=0.7, beta1=6:", ""]
    for r in example:
        lines.append(f"- {r['method']}: {r['rejection_rate']:.2%} rejected (95% CI {r['ci95_low']:.2%}-{r['ci95_high']:.2%}).")
    lines += ["", "These Shapley rejections are power against its own false null, not Type I errors for the Shapley hypothesis. Interpreting them as evidence that beta2 is nonzero would be incorrect.", "",
              "Outputs: signal_decomposition.png/pdf, rejection_comparison.png/pdf, both summary CSVs, settings.json, and pvalues.npz containing all simulated p-values.", "",
              "Reproduce from the repository root:", "", "```powershell", ".venv/Scripts/python.exe examples/demonstrate_shapley_coefficient_null.py", "```", ""]
    (out/"README.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({"output_dir": str(out), "formula_error": max_identity_error, "example": example}, indent=2))


if __name__ == "__main__":
    main()
