"""Combine the matched Linear Shapley AIS/exact folders and plot three views."""
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

import build_linear_shapley_exact_analysis_300 as exact

ROOT = Path(__file__).resolve().parents[1]
AIS = exact.SOURCE
EXACT = exact.OUT
OUT = AIS.with_name("linear_shapley_selected_five_matched_300_ais_exact_20261002")
LABELS = {"ais": "Linear RSS Shapley (AIS)", "exact": "Linear RSS Shapley (exact inference)"}
COLORS = {"ais": "#6b6ecf", "exact": "#258568"}
BETAS = exact.BETAS
KEYS = ["scope", "validation_shard", "iteration", "selection_event", "signal_strength"]
INTRO = ("Matched AIS and exact inference for the same 1,500 trials, selected sets, and targets. "
         "Both test absence of signal in the full centered linear design space, not a zero coefficient for the selected feature. "
         "The selector retains its fixed 1,000-permutation Shapley approximation. Exact refers to inference. "
         "Combined plots compare the two inference methods on paired trials; confidence intervals are computed separately for each method.")


def load():
    ais = pd.read_csv(AIS / "unified_tests.csv")
    analytic = pd.read_csv(EXACT / "unified_tests.csv")
    manifest = json.loads((EXACT / "source_manifest.json").read_text())
    exact.require(hashlib.sha256((AIS / "unified_tests.csv").read_bytes()).hexdigest() == manifest["source_sha256"], "Exact folder references a different AIS dataset")
    for frame in (ais, analytic):
        exact.require(len(frame) == 1500 and not frame.duplicated(KEYS).any(), "Unexpected trial count or duplicate keys")
    paired = ais.merge(analytic, on=KEYS, suffixes=("_ais", "_exact"), validate="one_to_one")
    exact.require(len(paired) == 1500, "Unmatched trials")
    for column in ["target_feature", "observed_selected_features", "target_seed", "auxiliary_u", "test_rank", "radial_t_obs", "target_is_signal"]:
        pd.testing.assert_series_equal(paired[f"{column}_ais"], paired[f"{column}_exact"], check_names=False)
    np.testing.assert_allclose(paired.p_value_ais, paired.source_ais_p_value, rtol=1e-14)
    exact.require(not ais.failed.any(), "Unexpected failed AIS tests")
    ais = ais.copy()
    ais["inference"] = "ais"
    ais["sampling_proposals"] = ais.total_proposals
    ais["sigma"] = ais.sigma.fillna(1.0)
    ais["rejected"] = ais.p_value.lt(0.05)
    ais["null_hypothesis"] = "P_centered_X mu = 0"
    analytic["inference"] = "exact"
    analytic["failed"] = False
    analytic["finite_sample_valid"] = True
    analytic["status"] = "ok"
    # AIS-only diagnostics remain missing in exact rows, rather than describing
    # the source sampler as though it was used for analytic inference.
    analytic["sampling_proposals"] = 0
    data = pd.concat([ais, analytic], ignore_index=True, sort=False)
    exact.require(len(data) == 3000 and not data.duplicated(["inference", *KEYS]).any(), "Combined dataset count drift")
    paired = paired[KEYS + ["target_feature_ais", "p_value_ais", "p_value_exact"]].copy()
    paired["ais_rejected"] = paired.p_value_ais.lt(0.05)
    paired["exact_rejected"] = paired.p_value_exact.lt(0.05)
    paired["decision_changed"] = paired.ais_rejected.ne(paired.exact_rejected)
    return data, paired


def summaries(data):
    powers, calibrations = [], []
    for inference, group in data.groupby("inference", sort=False):
        power, calibration = exact.summarize(group)
        power["inference"] = calibration["inference"] = inference
        powers.append(power)
        calibrations.append(calibration)
    return pd.concat(powers, ignore_index=True), pd.concat(calibrations, ignore_index=True)


def gallery(folder, title, figures):
    cards, markdown = [], f"# {title}\n\n{INTRO}\n\n"
    for name, caption in figures:
        label = name[3:].replace("_", " ").title()
        cards.append(f'<section><h2>{html.escape(label)}</h2><p>{html.escape(caption)}</p><a href="{name}.pdf">PDF</a><img src="{name}.png" alt="{html.escape(label)}"></section>')
        markdown += f"## {label}\n\n{caption}\n\n![{label}]({name}.png)\n\n[PDF]({name}.pdf)\n\n"
    page = '<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>' + html.escape(title) + '</title><style>body{max-width:1250px;margin:40px auto;padding:0 24px;font:16px/1.55 system-ui;color:#263238}img{width:100%;height:auto}section{margin-top:40px}a{color:#575ab2}</style><p><a href="../index.html">All views</a></p><h1>' + html.escape(title) + '</h1><p>' + html.escape(INTRO) + '</p>' + ''.join(cards) + '</html>'
    (folder / "index.html").write_text(page, encoding="utf-8")
    (folder / "README.md").write_text(markdown, encoding="utf-8")


def plot_view(data, power, calibration, modes, name, paired):
    folder = OUT / name
    folder.mkdir(parents=True, exist_ok=True)
    data = data[data.inference.isin(modes)]
    power = power[power.inference.isin(modes)]
    calibration = calibration[calibration.inference.isin(modes)]
    data.to_csv(folder / "unified_tests.csv", index=False)
    power.to_csv(folder / "power_summary.csv", index=False)
    calibration.to_csv(folder / "calibration_summary.csv", index=False)
    figures = []
    title = "Linear RSS Shapley: " + {"ais_only": "AIS only", "exact_only": "exact inference only", "combined": "AIS and exact inference"}[name]

    def save(fig, filename, caption):
        fig.suptitle(title, fontsize=13)
        fig.text(0.5, 0.016, "Global linear-signal null; fixed permutation Shapley selector; pointwise confidence intervals", ha="center", fontsize=9, color="#555555")
        fig.tight_layout(rect=(0, 0.06, 1, 0.94))
        for extension in ("png", "pdf"):
            fig.savefig(folder / f"{filename}.{extension}", dpi=180, bbox_inches="tight")
        plt.close(fig)
        figures.append((filename, caption))

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8))
    for mode in modes:
        group = power[power.inference.eq(mode)]
        for ax, values in zip(axes, [group.signal_targets / group.n, group.conditional_power, group.marginal_power]):
            ax.plot(group.beta, values, "o-" if mode == "ais" else "s--", color=COLORS[mode], label=LABELS[mode])
        axes[1].fill_between(group.beta, group.ci_lower, group.ci_upper, color=COLORS[mode], alpha=0.1)
    for ax, panel_title in zip(axes, ["Probability of targeting the signal", "Rejection rate given a signal target", "Signal target and rejection"]):
        ax.set(title=panel_title, xlabel="Signal strength (beta)", xticks=BETAS, ylim=(0, 1.04))
        ax.yaxis.set_major_formatter(PercentFormatter(1))
        ax.grid(alpha=0.2)
    axes[0].legend(fontsize=8)
    save(fig, "01_power_curves", "Original target-based definitions, with 95% Wilson intervals for the rejection rate among signal targets. Curves use 300 trials per beta per inference method.")

    fig, axes = plt.subplots(1, 2, figsize=(14, 3.8 + 0.6 * len(modes)))
    for ax, metric, panel_title in zip(axes, ["conditional_power", "ci_lower"], ["Rejection rate given a signal target", "Pointwise 95% lower confidence bound"]):
        matrix = power.pivot(index="inference", columns="beta", values=metric).reindex(modes)[BETAS]
        ax.imshow(matrix, vmin=0, vmax=1, cmap="YlGnBu", aspect="auto")
        for i, mode in enumerate(modes):
            for j, beta in enumerate(BETAS):
                row = power[power.inference.eq(mode) & power.beta.eq(beta)].iloc[0]
                value = row[metric]
                extra = f"{int(row.detections)}/{int(row.signal_targets)}" if metric == "conditional_power" else ("PASS" if row.ci_lower >= 0.8 and row.signal_targets >= 30 else "BELOW 80%")
                ax.text(j, i, f"{value:.1%}\n{extra}", ha="center", va="center", color="white" if value > 0.65 else "black", fontsize=9)
        ax.set(title=panel_title, xticks=range(4), xticklabels=BETAS, yticks=range(len(modes)), yticklabels=["AIS" if m == "ais" else "Exact" for m in modes], xlabel="Signal strength (beta)")
    save(fig, "02_power_target_heatmap", "Counts are global-null rejections / signal targets. PASS requires at least 30 signal targets and a pointwise 95% lower bound of at least 80%.")

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    for i, mode in enumerate(modes):
        group = calibration[calibration.inference.eq(mode)]
        offset = (i - (len(modes) - 1) / 2) * 0.002
        axes[0].errorbar(group.alpha + offset, group.fpr, yerr=np.maximum(0, np.vstack([group.fpr - group.ci_lower, group.ci_upper - group.fpr])), fmt="o", color=COLORS[mode], capsize=4, label=LABELS[mode])
        values = np.sort(data[data.scope.eq("null_calibration") & data.inference.eq(mode)].p_value)
        axes[1].step(np.r_[0, values, 1], np.r_[0, np.arange(1, len(values) + 1) / len(values), 1], where="post", color=COLORS[mode], ls="-" if mode == "ais" else "--")
    axes[0].plot([0, 0.12], [0, 0.12], "--", color="gray")
    axes[0].set(title="Null rejection rates and 95% Wilson intervals", xlabel="Nominal alpha (points offset in comparison)", ylabel="Observed false-positive rate", xticks=[0.01, 0.05, 0.1], ylim=(0, 0.15))
    axes[0].legend(fontsize=8)
    axes[1].plot([0, 1], [0, 1], ":", color="gray")
    axes[1].set(title="Null p-value empirical distribution", xlabel="p-value", ylabel="Fraction at or below p", xlim=(0, 1), ylim=(0, 1))
    for ax in axes:
        ax.grid(alpha=0.2)
    save(fig, "03_null_calibration", "300 paired null trials per method. Exact inference uses an analytic chi-square tail; AIS uses the saved importance-sampling estimates. Combined views retain separate method estimates.")

    fig, axes = plt.subplots(1, len(modes), figsize=(7 * len(modes), 4.8), squeeze=False)
    outcomes = []
    for ax, mode in zip(axes.flat, modes):
        group = power[power.inference.eq(mode)]
        x = np.arange(4)
        ax.bar(x, group.conditional_power, color=COLORS[mode], label="Global null rejected")
        ax.bar(x, 1 - group.conditional_power, bottom=group.conditional_power, color="#bcc8d1", label="Global null not rejected")
        for i, row in enumerate(group.itertuples()):
            ax.text(i, 1.02, f"n={row.signal_targets}", ha="center", fontsize=9)
            outcomes.append(dict(inference=mode, beta=row.beta, signal_targets=row.signal_targets, detected=row.detections, valid_not_detected=row.signal_targets-row.detections, failed=0))
        ax.set(title=LABELS[mode], xticks=x, xticklabels=BETAS, xlabel="Signal strength (beta)", ylim=(0, 1.14))
        ax.yaxis.set_major_formatter(PercentFormatter(1))
        ax.legend(loc="lower right", fontsize=8)
    save(fig, "04_signal_outcomes", "Outcomes among signal targets. All 1,500 evaluations per method succeeded; rejection concerns the global signal null.")
    pd.DataFrame(outcomes).to_csv(folder / "signal_outcomes.csv", index=False)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    costs = []
    for mode in modes:
        trials = data[data.inference.eq(mode) & data.scope.eq("power")]
        means = trials.groupby("signal_strength").sampling_proposals.mean().reindex(BETAS)
        axes[0].plot(BETAS, means, "o-", color=COLORS[mode], label=LABELS[mode])
        values = np.sort(trials.sampling_proposals)
        axes[1].step(values, np.arange(1, len(values)+1)/len(values), where="post", color=COLORS[mode], label=LABELS[mode])
        for beta, group in trials.groupby("signal_strength"):
            costs.append(dict(inference=mode, beta=beta, mean=group.sampling_proposals.mean(), median=group.sampling_proposals.median(), maximum=group.sampling_proposals.max()))
    axes[0].set(title="Mean inference sampling cost", xlabel="Signal strength (beta)", ylabel="Proposals including AIS pilot", xticks=BETAS)
    axes[1].set(title="Sampling cost distribution", xlabel="Inference sampling proposals", ylabel="Fraction at or below cost", ylim=(0, 1.04))
    axes[0].set_ylim(bottom=0)
    axes[1].set_xlim(left=0)
    if modes == ["exact"]:
        axes[0].set_ylim(0, 1)
        axes[1].set_xlim(0, 1)
        axes[1].text(0.5, 0.5, "All 1,200 power trials:\n0 sampling proposals", ha="center", va="center", transform=axes[1].transAxes)
    for ax in axes:
        ax.legend(fontsize=8)
        ax.grid(alpha=0.2)
    save(fig, "05_sampling_cost", "AIS counts include pilot and final proposals. Exact inference uses zero sampling proposals. These counts exclude selector preparation and are not wall-clock measurements.")
    pd.DataFrame(costs).to_csv(folder / "sampling_cost_summary.csv", index=False)

    fig, axes = plt.subplots(1, len(modes), figsize=(6 * len(modes), 4.8), squeeze=False)
    for ax, mode in zip(axes.flat, modes):
        if mode == "exact":
            ax.text(0.5, 0.5, "Not applicable\nAnalytic chi-square inference\nNo importance samples", ha="center", va="center", transform=ax.transAxes, fontsize=13)
            ax.set(title=LABELS[mode], xticks=[], yticks=[])
        else:
            trials = data[data.inference.eq(mode) & data.scope.eq("power")]
            ax.scatter(trials.denominator_ess, trials.tail_ess, s=12, alpha=0.3, color=COLORS[mode])
            ax.axvline(80, ls="--", color="gray")
            ax.axhline(15, ls="--", color="gray")
            ax.set(title=LABELS[mode], xlabel="Denominator ESS", ylabel="Tail ESS", xscale="log", yscale="log")
            ax.grid(alpha=0.2)
    save(fig, "06_ess_diagnostics", "AIS power-trial ESS values with the original thresholds of 80 and 15. ESS does not apply to exact inference.")

    fig, ax = plt.subplots(figsize=(9, 4.8))
    for mode in modes:
        row = power[power.inference.eq(mode) & power.beta.eq(1.0)].iloc[0]
        cost = data[data.inference.eq(mode) & data.scope.eq("power") & data.signal_strength.eq(1.0)].sampling_proposals.mean()
        ax.errorbar(cost, row.conditional_power, yerr=[[row.conditional_power-row.ci_lower], [row.ci_upper-row.conditional_power]], fmt="o", capsize=4, color=COLORS[mode], label=LABELS[mode])
        ax.annotate(f"{row.conditional_power:.1%}", (cost, row.conditional_power), xytext=(8, -10), textcoords="offset points")
    ax.axhline(0.8, ls="--", color="gray")
    ax.set(title="Signal-target rejection at beta = 1", xlabel="Mean inference sampling proposals", ylabel="Rejection rate given a signal target", ylim=(0.75, 1.02), xlim=(-300, 6000))
    ax.yaxis.set_major_formatter(PercentFormatter(1))
    ax.legend(loc="lower left", fontsize=9)
    ax.grid(alpha=0.2)
    save(fig, "07_power_cost_tradeoff", "Pointwise 95% Wilson intervals at beta=1. Exact inference has zero sampling cost; both methods retain the same selector construction cost.")

    fig, ax = plt.subplots(figsize=(9, 4.8))
    for mode in modes:
        group = power[power.inference.eq(mode)]
        ax.plot(group.beta, group.global_power, "o-" if mode == "ais" else "s--", color=COLORS[mode], label=LABELS[mode])
        ax.fill_between(group.beta, group.global_ci_lower, group.global_ci_upper, color=COLORS[mode], alpha=0.1)
    ax.set(title="Global-signal detection across all power trials", xlabel="Signal strength (beta)", ylabel="Global-null rejection rate", xticks=BETAS, ylim=(0, 1.04))
    ax.yaxis.set_major_formatter(PercentFormatter(1))
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(alpha=0.2)
    save(fig, "08_global_signal_power", "All 300 power trials per beta contribute regardless of the chosen target. Shading shows separate pointwise 95% Wilson intervals.")

    if len(modes) == 2:
        fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
        axes[0].scatter(paired.p_value_exact, paired.p_value_ais, s=12, alpha=0.35, color=COLORS["ais"])
        axes[0].plot([0, 1], [0, 1], "--", color="gray")
        axes[0].set(title="Paired p-values: 1,500 trials", xlabel="Exact p-value", ylabel="AIS p-value", xlim=(0, 1), ylim=(0, 1))
        changed = paired[paired.decision_changed]
        axes[1].scatter(changed.p_value_exact, changed.p_value_ais, s=35, color="#c44e52")
        axes[1].axhline(0.05, ls="--", color="gray")
        axes[1].axvline(0.05, ls="--", color="gray")
        axes[1].set(title=f"{len(changed)} decisions differ at alpha = 0.05", xlabel="Exact p-value", ylabel="AIS p-value", xlim=(0, 0.1), ylim=(0, 0.1))
        save(fig, "09_paired_exact_vs_ais", "Paired inference comparisons on identical trials, targets, and selection events. The CSV audit records all paired outcomes.")
    gallery(folder, title, figures)
    return len(figures)


def main():
    data, paired = load()
    OUT.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    power, calibration = summaries(data)
    data.to_csv(OUT / "unified_tests.csv", index=False)
    power.to_csv(OUT / "power_summary.csv", index=False)
    calibration.to_csv(OUT / "calibration_summary.csv", index=False)
    paired.to_csv(OUT / "paired_inference_audit.csv", index=False)
    paired[paired.decision_changed].to_csv(OUT / "changed_decisions.csv", index=False)
    manifest = dict(sources={mode: dict(path=folder.relative_to(ROOT).as_posix(), data_sha256=hashlib.sha256((folder / "unified_tests.csv").read_bytes()).hexdigest()) for mode, folder in [("ais", AIS), ("exact", EXACT)]},
                    inference_records=3000, unique_paired_trials=1500, trials_per_condition_per_inference=300,
                    combined_definition="Overlay or compare AIS and exact; do not pool paired records as independent trials",
                    pairing_verified=["selected set", "target feature", "target seed", "auxiliary draw", "radial statistic", "test rank", "signal target indicator"],
                    decisions_changed_at_005=int(paired.decision_changed.sum()), reproduction=".venv/Scripts/python.exe examples/build_linear_shapley_ais_exact_comparison_300.py")
    (OUT / "source_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    counts = {}
    for name, modes in [("ais_only", ["ais"]), ("exact_only", ["exact"]), ("combined", ["ais", "exact"])]:
        counts[name] = plot_view(data, power, calibration, modes, name, paired)
    links = ''.join(f'<li><a href="{name}/index.html">{name.replace("_", " ").title()}</a> — {count} plots, PNG and PDF</li>' for name, count in counts.items())
    (OUT / "index.html").write_text('<!doctype html><html lang="en"><meta charset="utf-8"><title>Linear RSS Shapley: AIS and exact</title><style>body{max-width:950px;margin:40px auto;padding:0 24px;font:17px/1.6 system-ui;color:#263238}</style><h1>Linear RSS Shapley: AIS and exact inference</h1><p>' + html.escape(INTRO) + '</p><ul>' + links + '</ul><p>3,000 inference records represent 1,500 paired trials. <a href="paired_inference_audit.csv">Pairing audit</a> · <a href="power_summary.csv">Power summaries</a> · <a href="calibration_summary.csv">Calibration summaries</a></p></html>', encoding="utf-8")
    (OUT / "README.md").write_text("# Linear RSS Shapley: AIS and exact inference\n\n" + INTRO + "\n\n" + ''.join(f"- [{name.replace('_', ' ').title()}]({name}/index.html): {count} plots in PNG/PDF.\n" for name, count in counts.items()) + "\nThe root unified_tests.csv contains 3,000 inference records from 1,500 paired trials. Each view includes its own data and summaries. Source folders are retained. Reproduce with `.venv/Scripts/python.exe examples/build_linear_shapley_ais_exact_comparison_300.py`.\n", encoding="utf-8")
    print(json.dumps(dict(output=str(OUT), plots_per_view=counts, paired_trials=len(paired), records=len(data)), indent=2))


if __name__ == "__main__":
    main()
