"""Compare all available LIME choices against matched adaptive AIS benchmarks."""
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import analyze_unified_lime_comparison as common

OUT = common.BASE / "lime_parameter_tuning/analysis/explanation_comparison"
NEW = {
    "explanations_40": "LIME 40 explanations",
    "explanations_60": "LIME 60 explanations",
    "explanations_80": "LIME 80 explanations",
    "explanations_40_ridge_3": "LIME 40 + ridge 3",
    "explanations_40_kernel_sqrt_p": "LIME 40 + kernel sqrt(p)",
}
ORDER = ["LIME baseline", "LIME ridge 30", "LIME wider kernel", *NEW.values(), "SHAP", "MI (k=12)", "Spline AIS"]


def save(fig, stem):
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(OUT / f"{stem}.{ext}", dpi=180, bbox_inches="tight")
    plt.close(fig)


def main():
    oldroot = common.BASE / "lime_parameter_tuning/initial_400_trials"
    newroot = common.BASE / "lime_parameter_tuning/explanations_200_trials"
    runs = [(oldroot / k, common.EXTRAS[k]) for k in ("ridge_30", "kernel_1p5_sqrt_p")]
    runs += [(newroot / k, v) for k, v in NEW.items()]
    common.ORDER = ORDER
    n, t, f, manifest = common.load(runs)
    # Avoid double counting the previous 40-explanation experiment. Verify its
    # matched prefix reproduces the new 200-trial run, including p-values.
    previous = oldroot / "analysis"
    checks = []
    for filename, fresh, keys in [
        ("null_combined.csv", n, ["iteration"]),
        ("power_features_combined.csv", f, ["iteration", "signal_strength"])]:
        old = pd.read_csv(previous / filename)
        old = old[old.variant.eq("explanations_40")].copy()
        old["validation_shard"] = "shard_" + old.shard.astype(str)
        keys = ["validation_shard", *keys]
        current = fresh[fresh.option.eq("LIME 40 explanations")]
        pair = current.merge(old, on=keys, suffixes=("_new", "_old"), validate="one_to_one")
        assert len(pair) == len(current)
        assert np.allclose(pair.p_value_new, pair.p_value_old, equal_nan=True, rtol=1e-12, atol=1e-14)
        assert np.array_equal(pair.failed_new, pair.failed_old)
        checks.append(f"{filename}: {len(pair)} matching 40-explanation results")
    OUT.mkdir(exist_ok=True)
    n.to_csv(OUT / "null_tests.csv", index=False)
    t.to_csv(OUT / "power_targets.csv", index=False)
    f.to_csv(OUT / "power_features.csv", index=False)
    unified = pd.concat([n.assign(scope="null"), f.assign(scope="power")], ignore_index=True)
    unified.to_csv(OUT / "unified_tests.csv", index=False)
    unified[unified.matched_200].to_csv(OUT / "unified_tests_matched_200.csv", index=False)
    c, p = common.summarize(n, t, f)
    p["point_estimate_at_least_80pct"] = p.conditional_failures_as_misses.ge(.8)
    p["ci_lower_at_least_80pct"] = p.conditional_lower.ge(.8)
    c.to_csv(OUT / "calibration_summary.csv", index=False)
    p.to_csv(OUT / "power_and_diagnostics_summary.csv", index=False)
    (OUT / "source_manifest.json").write_text(json.dumps(manifest, indent=2))
    m = p[p.view.eq("matched_200")]
    calibration = c[c.view.eq("matched_200") & c.alpha.eq(.05)].set_index("option")
    ranking = m.groupby("option").agg(
        mean_power_all_strengths=("conditional_failures_as_misses", "mean"),
        minimum_power_all_strengths=("conditional_failures_as_misses", "min"),
        strengths_at_least_80pct=("point_estimate_at_least_80pct", "sum"),
        failed_power_tests=("n_failed", "sum"),
        mean_final_proposals=("mean_proposals", "mean"))
    strong = m[m.beta.ge(.5)].groupby("option").conditional_failures_as_misses
    ranking["minimum_power_beta_ge_0p5"] = strong.min()
    ranking["mean_power_beta_ge_0p5"] = strong.mean()
    ranking["null_failures"] = calibration.failed
    ranking["fpr_at_0p05"] = calibration.fpr
    ranking.sort_values("minimum_power_beta_ge_0p5", ascending=False).to_csv(OUT / "ranking_summary.csv")
    fig, axes = plt.subplots(1, 2, figsize=(13, 8))
    for ax, metric, title in zip(axes,
        ["conditional_failures_as_misses", "signal_failure_rate"],
        ["Conditional power (failed signal tests = misses)", "Signal-target inference failure rate"]):
        matrix = m.pivot(index="option", columns="beta", values=metric).reindex(ORDER)
        ax.imshow(matrix.values, vmin=0, vmax=1, cmap="YlGnBu" if metric.startswith("conditional") else "YlOrRd", aspect="auto")
        ax.set(xticks=range(4), xticklabels=[.3, .5, .75, 1.], yticks=range(len(ORDER)), yticklabels=ORDER,
               xlabel="Signal strength", title=title)
        for i in range(len(ORDER)):
            for j in range(4):
                v = matrix.iloc[i, j]
                ax.text(j, i, f"{100*v:.1f}%", ha="center", va="center", color="white" if v>.65 else "black")
    fig.suptitle("Same AIS settings and matched 200 simulation trials per configuration")
    save(fig, "all_options_power_and_failures")
    selected = [*NEW.values(), "SHAP", "MI (k=12)", "Spline AIS"]
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    for i, label in enumerate(selected):
        g = m[m.option.eq(label)].sort_values("beta")
        for ax, metric in zip(axes.flat, ["conditional_failures_as_misses", "marginal_power", "signal_failure_rate", "mean_proposals"]):
            ax.plot(g.beta, g[metric], marker="o", ms=4, linestyle="-" if label.startswith("LIME") else "--", label=label, color=f"C{i}")
    for ax, title in zip(axes.flat, ["Conditional power (failures = misses)", "Marginal detection power", "Signal-target failure rate", "Mean final AIS proposals (pilot excluded)"]):
        ax.set(title=title, xlabel="Signal strength", xticks=[.3,.5,.75,1.]); ax.grid(alpha=.2)
    for ax in list(axes.flat)[:3]: ax.set_ylim(-.02,1.02)
    axes[0,0].axhline(.8,color="gray",linestyle=":")
    handles, labels = axes[0,0].get_legend_handles_labels()
    fig.legend(handles,labels,loc="lower center",ncol=3,fontsize=9)
    fig.tight_layout(rect=(0,.13,1,1))
    for ext in ("png","pdf"): fig.savefig(OUT/f"new_lime_vs_benchmarks.{ext}",dpi=180,bbox_inches="tight")
    plt.close(fig)
    fig, axes = plt.subplots(1,2,figsize=(13,7))
    for i,label in enumerate(ORDER):
        g=calibration.loc[label]
        axes[0].errorbar(g.fpr,i,xerr=[[g.fpr-g.ci_lower],[g.ci_upper-g.fpr]],fmt="o",color="C0",capsize=3)
        axes[0].text(.13,i,f"{int(g.failed)}/200 failed",va="center",fontsize=8)
        q=m[m.option.eq(label)&m.beta.eq(1)].iloc[0]
        v=q.conditional_failures_as_misses
        axes[1].errorbar(v,i,xerr=[[v-q.conditional_lower],[q.conditional_upper-v]],fmt="o",color="C1",capsize=3)
    for ax in axes:
        ax.set(yticks=range(len(ORDER)),yticklabels=ORDER);ax.invert_yaxis();ax.grid(axis="x",alpha=.2)
    axes[0].set(title="FPR at nominal 5% (successful tests)",xlabel="Rate and pointwise 95% Wilson interval",xlim=(0,.19))
    axes[0].axvline(.05,color="gray",linestyle="--")
    axes[1].set(title="Conditional power at signal strength 1",xlabel="Failures = misses; pointwise 95% Wilson interval",xlim=(.45,1.02))
    axes[1].axvline(.8,color="gray",linestyle="--")
    save(fig,"calibration_and_strong_signal_intervals")
    report=["# LIME tuning results versus adaptive AIS benchmarks", "",
        "All five new variants completed 32 shards and 200 null trials plus 200 trials at each of four signal strengths. Shared AIS settings, runtime metadata, per-shard simulation settings, target seeds and auxiliary draws passed the inherited audit. The original 40-explanation results reproduce on matched trials: "+'; '.join(checks)+'.',
        "The primary comparison includes eight LIME choices and SHAP, MI k=12 and spline screening with AIS. The two earlier 400-trial choices are restricted to the matching 200 trials in the primary plots; full-data summaries remain available. The old 40-explanation run is used only for reproduction checks, not pooled as independent evidence. The first five-setting sweep remains unavailable locally.",
        "## Findings",
        "No tested LIME option meets 80% conditional power at all four signal strengths, or even at all three strengths 0.5, 0.75 and 1.0. At beta=1, 40 explanations remains the best observed LIME choice at 79.81% (83/104 signal targets detected), with a pointwise 95% Wilson interval of 71.10%-86.40%. One additional detection would give 84/104=80.77%; this small point-estimate shortfall should not be confused with precise evidence about true power.",
        "There is no uniformly best option. Forty explanations with kernel width sqrt(p) is best among tested LIME variants at beta=0.75 (90.10%), but falls to 75.96% at beta=1. Sixty explanations is best at beta=0.5 (86.60%), but falls to 66.35% at beta=1 and has two failed null tests. Eighty explanations reaches 89.11% at beta=0.75 but only 69.23% at beta=1, with one failed null test. Increasing the number of explanations does not monotonically improve this pipeline.",
        "For stability across beta=0.5, 0.75 and 1, 40 explanations has the highest minimum conditional power among tested LIME options. This descriptive criterion was chosen after examining the pilot; it is not a pre-registered winner or independent confirmation. At beta=1 it still trails SHAP and MI (85.58%) and spline AIS (96.15%). Only spline AIS has an 80%-exceeding lower Wilson bound at beta=1 in this matched comparison.",
        "At nominal 5%, the 40-point reference has 10/200 rejections (5%) with no failures. Its ridge-3 variant has 8/200 (4%), and its wider-kernel variant 10/200 (5%), both with no failures. The 60- and 80-point variants have 2 and 1 null failures. Observed rates alone do not establish exact calibration. FPR excludes failed tests, whose counts are explicitly reported.",
        "The new results argue against adopting 60 or 80 points solely to improve reliability. Forty points remains a reasonable provisional choice for broad performance; 40 plus a wider kernel is a tradeoff favoring beta=0.75. None can currently be claimed to satisfy the 80% objective across the tested moderate/strong strengths. Do not lower ESS thresholds or discard failed tests to manufacture the target.",
        "These are pilot comparisons on reused seeds and selected hyperparameters. Confidence intervals are pointwise, not simultaneous or paired-difference intervals. Conditional power can compare different selected signal-target subsets; marginal power includes targeting differences. Proposal counts exclude the AIS pilot and do not measure runtime: more explanation points increase work per proposal.",
        "## Matched power and diagnostics",common.table(m,["option","beta","signal_targets","failed_signal","conditional_failures_as_misses","conditional_lower","conditional_upper","marginal_power","mean_proposals"]),
        "## Matched calibration at alpha=0.05",common.table(calibration.reset_index(),["option","n","failed","fpr","ci_lower","ci_upper"])]
    (OUT/"analysis_report.md").write_text('\n\n'.join(report),encoding="utf-8")
    (OUT/"README.md").write_text(
        "# Data guide\n\nunified_tests.csv contains all available tests for the 11 options, excluding the duplicate old 40-point run. "
        "unified_tests_matched_200.csv restricts every option to the original 200 trial IDs per configuration. "
        "Options share simulated datasets; these rows are not all independent. "
        "Key: scope, option, validation_shard, iteration, selection_event and (power only) signal_strength. "
        "Source provenance is in source_run and source_manifest.json. Raw null, target and feature tables are also saved. "
        "Summary views all_available and matched_200 overlap; do not add them together. "
        "The primary power metric conditional_failures_as_misses includes failed signal tests in the denominator. "
        "ranking_summary.csv gives several descriptive criteria; no single ranking establishes a statistically superior option. "
        "Plots have PNG and vector PDF versions. See analysis_report.md for conclusions.\n",encoding="utf-8")
    print(ranking.to_string())
    print(f"Saved {len(unified)} full records and {int(unified.matched_200.sum())} matched records to {OUT}")


if __name__=="__main__": main()
