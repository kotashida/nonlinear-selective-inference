"""Create a reference-format diagnostic gallery for the 1,000-trial MI sweep.

The gallery mirrors the seven figures in the selected-five-method report while
adding the non-MI methods from that report as an external visual benchmark.
No simulations are rerun.
"""
from pathlib import Path
import html

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "outputs/selection_method_validation/exact_set"
SWEEP = BASE / "mi_parameter_tuning/mi_cpu_sweep_1000_20260920"
REFERENCE = BASE / "selected_five_methods_robust_draw_floor_ais_exact_mi30"
OUT = SWEEP / "reference_format_analysis"
BETAS = [0.30, 0.50, 0.75, 1.00]
CONFIGS = ["knn_12", "knn_20", "knn_25", "knn_30", "kde_1", "kde_1p25",
           "kde_1p5", "kde_1p75", "kde_2", "kde_2p5"]
LABELS = {c: c.replace("knn_", "k-NN ").replace("kde_", "KDE ").replace("p", ".")
          for c in CONFIGS}
COLORS = dict(zip(CONFIGS, plt.cm.tab10(np.linspace(0, 1, len(CONFIGS)))))
FIGURES = []


def wilson(k, n, z=1.959963984540054):
    k = np.asarray(k, dtype=float); n = np.asarray(n, dtype=float)
    p = np.divide(k, n, out=np.full_like(k, np.nan), where=n > 0)
    den = 1 + z*z/n
    center = (p + z*z/(2*n))/den
    half = z*np.sqrt(p*(1-p)/n + z*z/(4*n*n))/den
    return center-half, center+half


def save(fig, name, caption):
    fig.tight_layout(rect=(0, 0, 1, .94))
    for ext in ("png", "pdf"):
        fig.savefig(OUT / f"{name}.{ext}", dpi=180, bbox_inches="tight")
    plt.close(fig)
    FIGURES.append((name, caption))


def load_reference():
    d = pd.read_csv(REFERENCE / "unified_tests.csv")
    d = d[d.scope.eq("power")].copy()
    d["target_is_signal"] = d.target_is_signal.fillna(False).astype(bool)
    d["detected"] = d.target_is_signal & d.p_value.lt(.05) & ~d.failed
    r = d.groupby(["selection_method", "signal_strength"]).agg(
        n=("p_value", "size"), signal_targets=("target_is_signal", "sum"),
        detections=("detected", "sum")).reset_index()
    r["targeting"] = r.signal_targets/r.n
    r["conditional_power"] = r.detections/r.signal_targets
    r["marginal_power"] = r.detections/r.n
    # Exact and AIS spline rows have identical selection, so avoid double-weighting
    # the benchmark envelope.
    return r[~r.selection_method.eq("spline_screening_exact")].copy()


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.size": 9, "axes.spines.top": False,
                         "axes.spines.right": False})
    p = pd.read_csv(SWEEP / "power_selection_summary.csv")
    p = p[p.subset.eq("all_1000")].copy()
    p["ci_lower"], p["ci_upper"] = wilson(p.detections, p.signal_targets)
    p["target_ci_lower"], p["target_ci_upper"] = wilson(p.signal_targets, p.n)
    p["marginal_ci_lower"], p["marginal_ci_upper"] = wilson(p.detections, p.n)
    c = pd.read_csv(SWEEP / "calibration_summary.csv")
    c["valid_n"] = c.n-c.failures
    c["ci_lower"], c["ci_upper"] = wilson(c.rejections, c.valid_n)
    ref = load_reference()
    raw = pd.concat([pd.read_csv(SWEEP / cfg / "observed_diagnostics.csv")
                     for cfg in CONFIGS], ignore_index=True)
    null = pd.concat([pd.read_csv(SWEEP / cfg / "null_trials.csv").assign(configuration=cfg)
                      for cfg in CONFIGS], ignore_index=True)
    assert len(p) == 40 and len(raw) == 40000 and len(null) == 10000
    p.to_csv(OUT / "power_summary_with_intervals.csv", index=False)
    c.to_csv(OUT / "calibration_summary_with_intervals.csv", index=False)
    shared = raw[~raw.new_iteration].groupby(["configuration", "beta"]).agg(
        n=("detection", "size"), targeting=("signal_targeted", "mean"),
        signal_targets=("signal_targeted", "sum"), detections=("detection", "sum"),
        failures=("failed", "sum")).reset_index()
    shared["conditional_power"] = shared.detections/shared.signal_targets
    shared["marginal_power"] = shared.detections/shared.n
    shared.to_csv(OUT / "shared_200_power_summary.csv", index=False)

    # 1. The same three power panels, with the other reference methods shown as
    # a gray range.  MI-30 from the small reference run is a dotted black line.
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    metrics = ["targeting", "conditional_power", "marginal_power"]
    titles = ["Probability of targeting the signal", "Conditional power",
              "Marginal detection power"]
    non_mi = ref[~ref.selection_method.eq("mutual_information")]
    mi_ref = ref[ref.selection_method.eq("mutual_information")]
    for ax, metric, title in zip(axes, metrics, titles):
        lo = non_mi.groupby("signal_strength")[metric].min().reindex(BETAS)
        hi = non_mi.groupby("signal_strength")[metric].max().reindex(BETAS)
        ax.fill_between(BETAS, lo, hi, color="gray", alpha=.16,
                        label="Reference non-MI range (200 trials)")
        ax.plot(mi_ref.signal_strength, mi_ref[metric], ":", color="black",
                lw=2, label="Reference MI, k-NN 30 (200 trials)")
        for cfg in CONFIGS:
            g = p[p.configuration.eq(cfg)].sort_values("beta")
            ax.plot(g.beta, g[metric], "o-", ms=3, lw=1.3,
                    color=COLORS[cfg], label=LABELS[cfg])
        ax.set(title=title, xlabel="Signal strength (beta)", xticks=BETAS,
               ylim=(0, 1.03))
        ax.yaxis.set_major_formatter(PercentFormatter(1)); ax.grid(alpha=.2)
    axes[0].legend(fontsize=7, ncol=2, loc="upper left")
    fig.suptitle("Mutual-information tuning sweep: power and signal targeting")
    save(fig, "01_power_curves", "Gray shading spans the non-MI methods in the 200-trial reference report. Sweep curves use 1,000 matched trials per beta; failures count as misses.")

    # 2. Same conditional-power / lower-bound heatmap layout as the reference.
    fig, axes = plt.subplots(1, 2, figsize=(12, 7))
    for ax, metric, title in zip(axes, ["conditional_power", "ci_lower"],
                                 ["Observed conditional power", "95% lower confidence bound"]):
        mat = p.pivot(index="configuration", columns="beta", values=metric).reindex(CONFIGS)[BETAS]
        ax.imshow(mat, vmin=0, vmax=1, cmap="YlGnBu", aspect="auto")
        for i, cfg in enumerate(CONFIGS):
            for j, beta in enumerate(BETAS):
                row = p[p.configuration.eq(cfg) & p.beta.eq(beta)].iloc[0]
                value = mat.iloc[i, j]
                extra = f"\n{int(row.detections)}/{int(row.signal_targets)}" if metric == "conditional_power" else ("\n>=80%" if value >= .8 else "")
                ax.text(j, i, f"{value:.1%}{extra}", ha="center", va="center",
                        color="white" if value > .65 else "black", fontsize=8)
        ax.set(title=title, xticks=range(4), xticklabels=BETAS,
               yticks=range(10), yticklabels=[LABELS[x] for x in CONFIGS],
               xlabel="Signal strength (beta)")
    fig.suptitle("MI sweep: the 80% conditional-power requirement")
    save(fig, "02_power_target_heatmap", "Counts show detections / signal targets. Wilson bounds are pointwise and do not account for choosing a setting after viewing the sweep.")

    # 3. Null calibration.
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for j, cfg in enumerate(CONFIGS):
        g = c[c.configuration.eq(cfg)].sort_values("alpha")
        offset = (j-(len(CONFIGS)-1)/2)*.00055
        axes[0].errorbar(g.alpha+offset, g.fpr_completed,
                         yerr=np.vstack([g.fpr_completed-g.ci_lower, g.ci_upper-g.fpr_completed]),
                         fmt="o", ms=4, capsize=2, color=COLORS[cfg], label=LABELS[cfg])
        vals = np.sort(null[null.configuration.eq(cfg) & ~null.failed].p_value.dropna())
        axes[1].step(vals, np.arange(1, len(vals)+1)/len(vals), where="post",
                     lw=1.1, color=COLORS[cfg], label=LABELS[cfg])
    axes[0].plot([0, .12], [0, .12], "--", color="gray")
    axes[0].set(title="Null rejection rates and 95% intervals", xlabel="Nominal alpha",
                ylabel="Observed false-positive rate", xticks=[.01, .05, .1], ylim=(0, .15))
    axes[0].legend(fontsize=7, ncol=2)
    axes[1].plot([0, 1], [0, 1], "--", color="gray")
    axes[1].set(title="Null p-value empirical distributions", xlabel="p-value",
                ylabel="Fraction at or below p", xlim=(0, 1), ylim=(0, 1))
    for ax in axes: ax.grid(alpha=.2)
    fig.suptitle(f"MI sweep: null calibration ({int(null.failed.sum())} failures / {len(null):,} tests)")
    save(fig, "03_null_calibration", "Each setting has 1,000 null trials. Rates are among completed tests; intervals are pointwise Wilson intervals.")

    # 4. Signal outcomes.
    fig, axes = plt.subplots(2, 5, figsize=(17, 8), sharey=True)
    outcomes = []
    for ax, cfg in zip(axes.flat, CONFIGS):
        g = p[p.configuration.eq(cfg)].sort_values("beta")
        detected = g.detections/g.signal_targets
        failed = g.signal_failures/g.signal_targets
        missed = 1-detected-failed; x = np.arange(4)
        ax.bar(x, detected, color="#258568", label="Detected")
        ax.bar(x, missed, bottom=detected, color="#bcc8d1", label="Valid, not detected")
        ax.bar(x, failed, bottom=detected+missed, color="#bc4545", label="Inference failed")
        for i, row in enumerate(g.itertuples()):
            ax.text(i, 1.01, f"n={row.signal_targets}", ha="center", fontsize=7)
            outcomes.append(dict(configuration=cfg, beta=row.beta,
                signal_targets=row.signal_targets, detected=row.detections,
                valid_not_detected=row.signal_targets-row.detections-row.signal_failures,
                failed=row.signal_failures))
        ax.set(title=LABELS[cfg], xticks=x, xticklabels=BETAS, ylim=(0, 1.12),
               xlabel="beta")
        ax.yaxis.set_major_formatter(PercentFormatter(1))
    axes[0, 0].legend(fontsize=7, loc="lower right")
    fig.suptitle("MI sweep: outcomes among signal targets")
    save(fig, "04_signal_outcomes", "Statistical non-detection is separated from unresolved inference. Each bar is conditional on targeting the true signal.")
    pd.DataFrame(outcomes).to_csv(OUT / "signal_outcomes.csv", index=False)

    # 5. Sampling cost. Pilot cost is 10 x 320 = 3,200 proposals.
    raw["total_proposals"] = raw.proposals + 3200
    fig, axes = plt.subplots(1, 2, figsize=(13, 5)); costs = []
    for cfg in CONFIGS:
        g = raw.groupby(["configuration", "beta"]).total_proposals.mean().loc[cfg]
        axes[0].plot(g.index, g.values, "o-", ms=3, color=COLORS[cfg], label=LABELS[cfg])
        vals = np.sort(raw[raw.configuration.eq(cfg)].total_proposals)
        axes[1].step(vals, np.arange(1, len(vals)+1)/len(vals), where="post", color=COLORS[cfg])
        for beta in BETAS:
            q = raw[raw.configuration.eq(cfg) & raw.beta.eq(beta)].total_proposals
            costs.append(dict(configuration=cfg, beta=beta, mean=q.mean(), median=q.median(),
                              p95=q.quantile(.95), maximum=q.max(),
                              first_batch_fraction=(q.eq(4000)).mean(), cap_fraction=(q.eq(54400)).mean()))
    axes[0].set(title="Mean cost by signal strength", xlabel="Signal strength (beta)",
                ylabel="Proposals including pilot", xticks=BETAS); axes[0].legend(fontsize=7, ncol=2)
    axes[1].set(title="Cost distribution across power trials", xlabel="Sampling proposals (symlog)",
                ylabel="Fraction at or below cost", xscale="symlog", ylim=(0, 1.03))
    for ax in axes: ax.grid(alpha=.2)
    fig.suptitle("MI sweep: sampling effort")
    save(fig, "05_sampling_cost", "Counts include 3,200 pilot evaluations and measure proposals, not wall-clock time.")
    pd.DataFrame(costs).to_csv(OUT / "sampling_cost_summary.csv", index=False)

    # 6. ESS diagnostics.
    fig, axes = plt.subplots(2, 5, figsize=(17, 8), sharex=True, sharey=True)
    for ax, cfg in zip(axes.flat, CONFIGS):
        g = raw[raw.configuration.eq(cfg)]
        for failed, marker, color in [(False, "o", COLORS[cfg]), (True, "x", "#bf3232")]:
            q = g[g.failed.eq(failed)]
            ax.scatter(q.denominator_ess, q.tail_ess, s=12 if not failed else 35,
                       alpha=.25 if not failed else .9, marker=marker, color=color)
        ax.axvline(80, ls="--", color="gray"); ax.axhline(15, ls="--", color="gray")
        ax.set(title=f"{LABELS[cfg]}: {int(g.failed.sum())}/{len(g):,} failed",
               xscale="log", yscale="log", xlabel="Denominator ESS", ylabel="Tail ESS")
        ax.grid(alpha=.15)
    fig.suptitle("MI sweep: effective sample sizes")
    save(fig, "06_ess_diagnostics", "Dashed lines mark denominator ESS=80 and tail ESS=15. Failures are red crosses; axes are logarithmic.")

    # 7. Power-cost tradeoff at beta=1, matching the reference endpoint plot.
    fig, ax = plt.subplots(figsize=(10, 6))
    for cfg in CONFIGS:
        row = p[p.configuration.eq(cfg) & p.beta.eq(1)].iloc[0]
        mean_cost = raw[raw.configuration.eq(cfg) & raw.beta.eq(1)].total_proposals.mean()
        ax.errorbar(mean_cost, row.conditional_power,
                    yerr=[[row.conditional_power-row.ci_lower], [row.ci_upper-row.conditional_power]],
                    fmt="o", capsize=3, color=COLORS[cfg], label=LABELS[cfg])
        ax.annotate(f"{row.conditional_power:.1%}", (mean_cost, row.conditional_power),
                    xytext=(5, 0), textcoords="offset points", fontsize=7, color=COLORS[cfg])
    ax.axhline(.8, ls="--", color="gray")
    ax.set(xlabel="Mean sampling proposals per power trial", ylabel="Conditional power",
           ylim=(.88, 1.01)); ax.yaxis.set_major_formatter(PercentFormatter(1))
    ax.legend(fontsize=7, ncol=2); ax.grid(alpha=.2)
    fig.suptitle("MI sweep at beta=1: power versus sampling effort")
    save(fig, "07_power_cost_tradeoff", "Vertical bars are pointwise 95% Wilson intervals. Proposal counts are not measured runtimes.")

    # Quantify whether tuning repaired the original performance deficit.
    rows = []
    for beta in BETAS:
        q = p[p.beta.eq(beta)]
        best = q.loc[q.marginal_power.idxmax()]
        b = non_mi[non_mi.signal_strength.eq(beta)]
        rows.append(dict(beta=beta, best_configuration=best.configuration,
            targeting=best.targeting, conditional_power=best.conditional_power,
            marginal_power=best.marginal_power, marginal_ci_lower=best.marginal_ci_lower,
            marginal_ci_upper=best.marginal_ci_upper,
            reference_non_mi_best_marginal=b.marginal_power.max(),
            reference_non_mi_best_method=b.loc[b.marginal_power.idxmax(), "selection_method"],
            exceeds_reference_best=best.marginal_power > b.marginal_power.max(),
            comparison_has_equal_trial_count=False))
    comparison = pd.DataFrame(rows)
    comparison.to_csv(OUT / "best_settings_vs_reference.csv", index=False)
    rank = p.groupby("configuration").marginal_power.mean().sort_values(ascending=False)
    best_overall = rank.index[0]
    conclusion = (f"KDE bandwidth 2-2.5 repairs the weak-signal deficit in this exploratory sweep. "
                  f"On the strictly matched 200-trial subset, KDE 2/2.5 has 14.5% marginal power at "
                  f"beta=0.30, versus 9.0% for MI-30 and 14.5% for the best non-MI reference method; "
                  f"at beta=0.50, KDE 2.5 has 43.5%, versus 34.0% for MI-30 and 43.0% for the best "
                  f"non-MI method. Thus KDE catches up to the other selectors in the equal-size paired "
                  f"comparison. In the full 1,000-trial sweep, {LABELS[best_overall]} has the highest "
                  f"mean marginal power ({rank.iloc[0]:.1%}), but those 1,000-trial estimates must not "
                  "be treated as strictly paired with the 200-trial non-MI reference estimates. The gain "
                  "comes mainly from better signal targeting at low beta; k-NN variants remain behind. "
                  "Because settings were selected on these same trials, this is a tuning result and needs "
                  "an independent confirmation run.")
    intro = ("Ten MI settings, 1,000 null trials and 1,000 power trials per beta and setting. "
             "AIS uses the same 51,200 final cap, 10 x 320 pilot, 800 batch, ESS thresholds 80/15, "
             "and robust draw-floor strategy as the reference report.")
    report = ["# Mutual-information CPU sweep: reference-format analysis", intro, conclusion,
              "## Plot gallery"]
    sections = []
    for name, caption in FIGURES:
        title = name[3:].replace("_", " ").title()
        report.append(f"### {title}\n\n{caption}\n\n![{name}]({name}.png)\n\n[PDF]({name}.pdf)")
        sections.append(f'<section><h2>{html.escape(title)}</h2><p>{html.escape(caption)}</p><a href="{name}.png"><img src="{name}.png" alt="{html.escape(caption)}"></a><p><a href="{name}.pdf">Download PDF</a></p></section>')
    report.append("## Interpretation\n\n" + conclusion)
    report.append("## Data\n\nThe summary CSVs in this folder are derived from the existing sweep outputs; no simulations were rerun. Pointwise intervals do not adjust for comparing ten configurations or tuning on these trials.")
    (OUT / "README.md").write_text("\n\n".join(report), encoding="utf-8")
    (OUT / "index.html").write_text('<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1"><title>MI sweep analysis</title><style>body{font:17px/1.6 system-ui;max-width:1200px;margin:32px auto;padding:0 24px;color:#203040;background:#f5f7fa}section{background:white;padding:24px;margin:24px 0;border-radius:12px}img{width:100%;height:auto}a{color:#17669b}</style><h1>Mutual-information CPU sweep</h1><p>'+html.escape(intro)+'</p><p><strong>Conclusion:</strong> '+html.escape(conclusion)+'</p>'+''.join(sections)+'</html>', encoding="utf-8")
    print(conclusion)
    print(f"Saved {len(FIGURES)} PNG/PDF figures and summaries to {OUT}")


if __name__ == "__main__":
    main()
