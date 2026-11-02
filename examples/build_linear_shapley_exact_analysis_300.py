"""Plot exact radial inference for the saved rank-20 Linear RSS Shapley run.

Run with .venv/Scripts/python.exe examples/build_linear_shapley_exact_analysis_300.py.
The selector still uses its original 1,000 fixed permutations. Only inference
is exact; the tested null is absence of signal in the centered design space.
"""
from pathlib import Path
import hashlib
import html
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "outputs/selection_method_validation/exact_set/linear_shapley_selected_five_matched_300_20260928"
OUT = SOURCE.with_name("linear_shapley_selected_five_matched_300_exact_20261002")
BETAS = [0.3, 0.5, 0.75, 1.0]
COLOR = "#6b6ecf"
FIGURES = []
NOTE = "Exact inference for the global linear-signal null; fixed permutation Shapley selector"


def require(condition, message):
    if not condition:
        raise ValueError(message)


def wilson(k, n):
    z = stats.norm.ppf(0.975)
    fraction = k / n
    denominator = 1 + z * z / n
    center = (fraction + z * z / (2 * n)) / denominator
    half = z * np.sqrt(fraction * (1 - fraction) / n + z * z / (4 * n * n)) / denominator
    return max(0.0, center - half), min(1.0, center + half)


def save(fig, name, caption):
    fig.text(0.5, 0.018, NOTE, ha="center", fontsize=9, color="#555555")
    fig.tight_layout(rect=(0, 0.06, 1, 0.94))
    for extension in ("png", "pdf"):
        fig.savefig(OUT / f"{name}.{extension}", dpi=180, bbox_inches="tight")
    plt.close(fig)
    FIGURES.append((name, caption))


def load():
    path = SOURCE / "unified_tests.csv"
    source = pd.read_csv(path)
    require(len(source) == 1500, "Expected 1,500 saved tests")
    require(source.selection_method.eq("linear_shapley").all(), "Unexpected selector")
    require(source.selection_event.eq("exact_set").all(), "Unexpected selection event")
    require(source.test_rank.eq(20).all(), "The full-design simplification requires rank 20")
    require(source.test_statistic.eq("linear_shapley").all(), "Unexpected statistic")
    require(source.selected_samples.eq(source.proposals).all(), "Saved draws did not all preserve selection")
    # The unified power rows omit sigma and multiplicity; recover them from
    # the per-condition settings rather than treating missing values as facts.
    folders = ["null_fresh", *[f"power_feature_0_beta_{str(b).replace('.', 'p')}" for b in BETAS]]
    for shard in range(32):
        for folder in folders:
            settings = json.loads((SOURCE / "runs/linear_shapley" / f"shard_{shard}" / "linear_shapley/baseline" / folder / "settings.json").read_text())
            require(settings["sigma"] == 1.0 and settings["multiplicity"] == "none", "Unexpected variance or multiplicity")
            require(settings["n_features"] == 20 and settings["selector_settings"]["fit_intercept"], "Unexpected design")
    require(source.sigma.dropna().eq(1.0).all(), "Saved sigma mismatch")
    require(source.multiplicity_method.dropna().eq("none").all(), "Saved multiplicity mismatch")
    require(not source.duplicated(["scope", "validation_shard", "iteration", "signal_strength"]).any(), "Duplicate trials")
    require(source.scope.eq("null_calibration").sum() == 300, "Null trial count drift")
    power = source[source.scope.eq("power")]
    require(power.groupby("signal_strength").size().to_dict() == {b: 300 for b in BETAS}, "Power trial count drift")
    columns = ["scope", "validation_shard", "iteration", "selection_method", "selection_event",
               "target_feature", "observed_selected_features", "auxiliary_u", "target_seed",
               "signal_strength", "target_is_signal", "test_rank", "radial_t_obs", "sigma", "importance_score"]
    data = source[columns].copy()
    data["sigma"] = 1.0
    data["p_value"] = stats.chi2.sf(data.radial_t_obs.to_numpy() ** 2, df=20)
    require(np.isfinite(data.p_value).all(), "Nonfinite exact p-value")
    np.testing.assert_allclose(data.p_value, source.unadjusted_p_value, atol=1e-14, rtol=1e-12)
    data["source_ais_p_value"] = source.p_value
    data["source_ais_total_proposals"] = source.total_proposals
    data["inference"] = "exact_global_radial"
    data["p_value_method"] = "chi_square_full_centered_design"
    data["null_hypothesis"] = "P_centered_X mu = 0"
    data["sampling_proposals"] = 0
    data["rejected"] = data.p_value.lt(0.05)
    data["source_ais_rejected"] = source.p_value.lt(0.05)
    data["decision_changed"] = data.rejected.ne(data.source_ais_rejected)
    return data, path


def summarize(data):
    power_rows = []
    for beta, group in data[data.scope.eq("power")].groupby("signal_strength"):
        signal = group[group.target_is_signal.eq(True)]
        k = int(signal.rejected.sum())
        lo, hi = wilson(k, len(signal))
        total = int(group.rejected.sum())
        glo, ghi = wilson(total, len(group))
        power_rows.append(dict(beta=beta, n=len(group), signal_targets=len(signal), detections=k,
                               conditional_power=k / len(signal), ci_lower=lo, ci_upper=hi,
                               marginal_power=k / len(group), global_rejections=total,
                               global_power=total / len(group), global_ci_lower=glo, global_ci_upper=ghi))
    null = data[data.scope.eq("null_calibration")]
    calibration = []
    for alpha in (0.01, 0.05, 0.1):
        k = int(null.p_value.lt(alpha).sum())
        lo, hi = wilson(k, len(null))
        calibration.append(dict(alpha=alpha, n=len(null), rejections=k, fpr=k / len(null), ci_lower=lo, ci_upper=hi))
    return pd.DataFrame(power_rows), pd.DataFrame(calibration)


def plots(data, power, calibration):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6))
    for ax, values, title in zip(axes, [power.signal_targets / power.n, power.conditional_power, power.marginal_power],
                                 ["Probability of targeting the signal", "Rejection rate given a signal target", "Signal target and rejection"]):
        ax.plot(power.beta, values, "o-", color=COLOR)
        ax.set(title=title, xlabel="Signal strength (beta)", xticks=BETAS, ylim=(0, 1.04))
        ax.yaxis.set_major_formatter(PercentFormatter(1))
        ax.grid(alpha=0.2)
    axes[1].fill_between(power.beta, power.ci_lower, power.ci_upper, color=COLOR, alpha=0.14)
    fig.suptitle("Linear RSS Shapley: exact inference on the original 300-trial conditions")
    save(fig, "01_power_curves", "Original target-based reporting definitions. Shading is a pointwise 95% Wilson interval. A rejection tests global linear signal, not the selected feature coefficient.")

    fig, axes = plt.subplots(1, 2, figsize=(13, 3.8))
    for ax, metric, title in zip(axes, ["conditional_power", "ci_lower"], ["Rejection rate given a signal target", "Pointwise 95% lower bound"]):
        values = power[metric].to_numpy()[None, :]
        ax.imshow(values, vmin=0, vmax=1, cmap="YlGnBu", aspect="auto")
        for j, row in enumerate(power.itertuples()):
            value = getattr(row, metric)
            extra = f"{row.detections}/{row.signal_targets}" if metric == "conditional_power" else ("PASS" if row.ci_lower >= 0.8 and row.signal_targets >= 30 else "BELOW 80%")
            ax.text(j, 0, f"{value:.1%}\n{extra}", ha="center", va="center", color="white" if value > 0.65 else "black")
        ax.set(title=title, xticks=range(4), xticklabels=[f"{b:.2f}" for b in BETAS], yticks=[], xlabel="Signal strength (beta)")
    fig.suptitle("Exact inference: target-based power summaries")
    save(fig, "02_power_target_heatmap", "Counts are rejections / signal targets. PASS requires at least 30 signal targets and a pointwise 95% lower bound of at least 80%. This summarizes global-null rejection within the signal-target subset.")

    null = data[data.scope.eq("null_calibration")]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8))
    axes[0].errorbar(calibration.alpha, calibration.fpr,
                    yerr=np.maximum(0, np.vstack([calibration.fpr - calibration.ci_lower, calibration.ci_upper - calibration.fpr])),
                    fmt="o", color=COLOR, capsize=4)
    axes[0].plot([0, 0.12], [0, 0.12], "--", color="gray")
    axes[0].set(title="Null rejection rates with 95% Wilson intervals", xlabel="Nominal alpha", ylabel="Observed false-positive rate", xticks=[0.01, 0.05, 0.1], ylim=(0, 0.15))
    values = np.sort(null.p_value)
    axes[1].step(np.r_[0, values, 1], np.r_[0, np.arange(1, 301) / 300, 1], where="post", color=COLOR)
    axes[1].plot([0, 1], [0, 1], "--", color="gray")
    axes[1].set(title="Null p-value empirical distribution", xlabel="Exact p-value", ylabel="Fraction at or below p", xlim=(0, 1), ylim=(0, 1))
    for ax in axes:
        ax.grid(alpha=0.2)
    fig.suptitle("Exact global-signal test: 300 null trials")
    save(fig, "03_null_calibration", "Exact analytic p-values under the Gaussian model with known sigma. Empirical rejection rates still have uncertainty from the 300 simulated datasets.")

    fig, ax = plt.subplots(figsize=(8, 4.8))
    x = np.arange(4)
    ax.bar(x, power.conditional_power, color="#258568", label="Global null rejected")
    ax.bar(x, 1 - power.conditional_power, bottom=power.conditional_power, color="#bcc8d1", label="Global null not rejected")
    for i, row in enumerate(power.itertuples()):
        ax.text(i, 1.025, f"n={row.signal_targets}", ha="center", fontsize=9)
    ax.set(title="Outcomes among signal targets", xticks=x, xticklabels=BETAS, xlabel="Signal strength (beta)", ylim=(0, 1.14))
    ax.yaxis.set_major_formatter(PercentFormatter(1))
    ax.legend(loc="lower right")
    save(fig, "04_signal_outcomes", "All exact p-values were evaluated successfully. Bars use the original signal-target subsets and distinguish global-null rejection from non-rejection.")

    fig, ax = plt.subplots(figsize=(8, 4.8))
    ax.plot(power.beta, power.global_power, "o-", color=COLOR)
    ax.fill_between(power.beta, power.global_ci_lower, power.global_ci_upper, color=COLOR, alpha=0.14)
    for row in power.itertuples():
        ax.annotate(f"{row.global_rejections}/{row.n}", (row.beta, row.global_power), xytext=(0, -20), textcoords="offset points", ha="center")
    ax.set(title="Global-signal detection across all 300 trials per beta", xlabel="Signal strength (beta)", ylabel="Global-null rejection rate", xticks=BETAS, ylim=(0, 1.08))
    ax.yaxis.set_major_formatter(PercentFormatter(1))
    ax.grid(alpha=0.2)
    save(fig, "05_global_signal_power", "All power trials contribute, regardless of whether the chosen target is the generating signal feature. Shading shows pointwise 95% Wilson intervals.")

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    for scope, color, label in [("null_calibration", COLOR, "Null trials"), ("power", "#258568", "Power trials")]:
        group = data[data.scope.eq(scope)]
        axes[0].scatter(group.p_value, group.source_ais_p_value, s=12, alpha=0.4, color=color, label=label)
    axes[0].plot([0, 1], [0, 1], "--", color="gray")
    axes[0].set(title="Paired p-values on identical trials", xlabel="Exact p-value", ylabel="Original AIS p-value", xlim=(0, 1), ylim=(0, 1))
    axes[0].legend()
    changed = data[data.decision_changed]
    axes[1].scatter(changed.p_value, changed.source_ais_p_value, color="#c44e52", s=35)
    axes[1].axvline(0.05, ls="--", color="gray")
    axes[1].axhline(0.05, ls="--", color="gray")
    axes[1].plot([0, 0.1], [0, 0.1], ":", color="gray")
    axes[1].set(title=f"{len(changed)} decisions changed at alpha = 0.05", xlabel="Exact p-value", ylabel="Original AIS p-value", xlim=(0, 0.1), ylim=(0, 0.1))
    for ax in axes:
        ax.grid(alpha=0.2)
    fig.suptitle("Exact inference versus the original AIS estimates")
    save(fig, "06_exact_vs_ais", "Identical datasets, selected sets, and targets. Differences arise from replacing AIS estimates with the analytic chi-square tail; there is no new simulation run.")


def main():
    data, source_path = load()
    OUT.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    power, calibration = summarize(data)
    data.to_csv(OUT / "unified_tests.csv", index=False)
    power.to_csv(OUT / "power_summary.csv", index=False)
    calibration.to_csv(OUT / "calibration_summary.csv", index=False)
    data[data.decision_changed].to_csv(OUT / "changed_decisions.csv", index=False)
    manifest = dict(source=source_path.relative_to(ROOT).as_posix(), source_sha256=hashlib.sha256(source_path.read_bytes()).hexdigest(),
                    formula="scipy.stats.chi2.sf(radial_t_obs**2, df=20)", null_hypothesis="P_centered_X mu = 0",
                    interpretation="Global linear-signal test, not a test of beta_j = 0",
                    selector="Original fixed 1,000-permutation RSS Shapley approximation; not exact coalition averaging",
                    exact_inference_sampling_proposals=0, total_tests=len(data), trials_per_condition=300,
                    decisions_changed_at_005=int(data.decision_changed.sum()),
                    maximum_absolute_ais_error=float((data.p_value - data.source_ais_p_value).abs().max()),
                    assumptions=["Fixed design; Gaussian noise with known sigma", "Target support equals the full rank-20 centered design space", "Condition on projected direction and orthogonal complement", "All scores homogeneous quadratic forms in that design space"])
    (OUT / "source_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    FIGURES.clear()
    plots(data, power, calibration)
    intro = ("Exact chi-square inference recomputed from the original 1,500 saved tests: 300 null trials and 300 trials at each beta. "
             "The null is absence of signal in the full centered linear design space, not a zero coefficient for the selected feature. "
             "The original fixed 1,000-permutation Shapley selector, selected sets, and targets are retained. "
             "Exact refers to inference, not exact Shapley coalition averaging. No new datasets or sampling proposals were generated. "
             "Intervals describe pointwise simulation uncertainty; the trials reuse the original study seeds.")
    markdown = "# Linear RSS Shapley: exact global-signal inference\n\n" + intro + "\n\n"
    gallery = []
    for name, caption in FIGURES:
        title = name[3:].replace("_", " ").title()
        markdown += f"## {title}\n\n{caption}\n\n![{title}]({name}.png)\n\n[PDF]({name}.pdf)\n\n"
        gallery.append(f'<section><h2>{html.escape(title)}</h2><p>{html.escape(caption)}</p><a href="{name}.pdf">PDF</a><img src="{name}.png" alt="{html.escape(title)}"></section>')
    markdown += "## Data and reproduction\n\nunified_tests.csv contains exact p-values and explicitly prefixed source AIS values. power_summary.csv includes both target-based metrics and all-trial global power. changed_decisions.csv lists the paired changes at alpha=0.05. source_manifest.json records the formula, assumptions, and source hash.\n\nRun `.venv/Scripts/python.exe examples/build_linear_shapley_exact_analysis_300.py` from the repository root.\n"
    (OUT / "README.md").write_text(markdown, encoding="utf-8")
    document = '<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Linear RSS Shapley: exact inference</title><style>body{max-width:1200px;margin:40px auto;padding:0 24px;font:16px/1.55 system-ui;color:#263238}img{width:100%;height:auto}section{margin-top:40px}a{color:#575ab2}</style><h1>Linear RSS Shapley: exact global-signal inference</h1><p>' + html.escape(intro) + '</p>' + ''.join(gallery) + '</html>'
    (OUT / "index.html").write_text(document, encoding="utf-8")
    print(power.to_string(index=False))
    print(calibration.to_string(index=False))
    print(f"Created {len(FIGURES)} PNG/PDF plot pairs in {OUT}")


if __name__ == "__main__":
    main()
