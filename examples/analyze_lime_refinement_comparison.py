"""Compare the LIME refinement sweep against its reference and adaptive AIS benchmarks."""
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import analyze_unified_lime_comparison as common

OUT = common.BASE / "lime_parameter_tuning/analysis/refinement_comparison"
NEW = {
    "explanations_30": "LIME 30 explanations",
    "explanations_35": "LIME 35 explanations",
    "explanations_45": "LIME 45 explanations",
    "explanations_40_ridge_0p3": "LIME 40 + ridge 0.3",
    "explanations_40_kernel_0p6_sqrt_p": "LIME 40 + narrower kernel",
}
ORDER = ["LIME baseline", "LIME 40 explanations", *NEW.values(), "SHAP", "MI (k=12)", "Spline AIS"]



def save(fig, stem):
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(OUT / f"{stem}.{ext}", dpi=180, bbox_inches="tight")
    plt.close(fig)


def main():
    reference = common.BASE / "lime_parameter_tuning/explanations_200_trials/explanations_40"
    newroot = common.BASE / "lime_parameter_tuning/refinement_200_trials"
    runs = [(reference, "LIME 40 explanations")]
    runs += [(newroot / k, v) for k, v in NEW.items()]
    common.ORDER = ORDER
    n, t, f, manifest = common.load(runs)
    assert n.groupby("option").size().eq(200).all()
    assert t.groupby(["option", "signal_strength"]).size().eq(200).all()
    assert n.matched_200.all() and t.matched_200.all() and f.matched_200.all()
    assert len(n) == 2000 and len(t) == len(f) == 8000
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
    selected = ["LIME 40 explanations", *NEW.values(), "SHAP", "MI (k=12)", "Spline AIS"]
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
    # Paired marginal detections remain comparable when selected targets differ.
    paired = []
    for beta in [.3, .5, .75, 1.]:
        ref = t[t.option.eq("LIME 40 explanations") & t.signal_strength.eq(beta)]
        for option in ORDER:
            q = t[t.option.eq(option) & t.signal_strength.eq(beta)]
            g = ref.merge(q, on=["validation_shard", "iteration", "signal_strength", "selection_event"],
                          suffixes=("_ref", "_option"), validate="one_to_one")
            a, b = g.successful_detection_ref, g.successful_detection_option
            paired.append(dict(option=option, beta=beta, paired_trials=len(g),
                gained_detections=int((~a & b).sum()), lost_detections=int((a & ~b).sum()),
                marginal_power_difference=float(b.mean()-a.mean())))
    pd.DataFrame(paired).to_csv(OUT / "paired_detection_changes_vs_lime40.csv", index=False)
    strong = m[m.beta.eq(1)].set_index("option").reindex(ORDER).reset_index()
    summary = strong[["option","signal_targets","failed_signal","conditional_failures_as_misses",
                      "conditional_lower","conditional_upper","marginal_power"]]
    best_new = strong[strong.option.isin(NEW.values())].sort_values("conditional_failures_as_misses", ascending=False).iloc[0]
    report = ["# LIME refinement versus matched adaptive AIS benchmarks", "",
        "All ten options have 200 null trials and 200 trials per signal strength. Shared AIS settings, runtime metadata, per-shard simulation settings, target seeds and auxiliary draws passed the inherited audit. LIME parameters differ; RF, LIME neighborhood size and random state remain fixed. Raw counts and pooled calibration/power summaries were cross-checked. All 10,000 records are in the matched comparison, but methods share simulation trials, so these are not 10,000 independent datasets.",
        "## Findings",
        f"None of the five new options reaches 80% conditional power at beta=1 when failed signal tests count as no detection. The best new option there is {best_new.option}: {100*best_new.conditional_failures_as_misses:.2f}%. The previous 40-explanation reference remains better at 79.81%, although pilot differences do not establish statistical superiority.",
        "At beta=1, the reference detects 83/104 signal targets, versus 79 for 30 points, 70 for 35 points, 78 for 45 points, 75 for ridge 0.3 and 76 for the narrower kernel. New variants have 23, 33, 25, 27 and 27 failed signal tests respectively, versus 19 for the reference. All variants target the same number of signals here; paired marginal gains/losses are saved separately.",
        "Forty-five explanations is best among the new settings at beta=0.5 (88.66%) and beta=0.75 (89.11%), but falls to 75% at beta=1. The original 40-point reference has the highest minimum LIME power across beta=0.5, 0.75 and 1.0 in this comparison. No LIME setting reaches 80% at all these strengths; all methods are well below 80% at beta=0.3.",
        "At beta=1, SHAP and MI achieve 85.58%, while spline screening AIS achieves 96.15%. Their failed signal-test counts are 13, 9 and 0 respectively. The SS benchmark uses AIS, not exact spline inference. MI is weaker at beta=0.5 (68.29%); SHAP and SS both exceed 80% in point estimates at all strengths from 0.5 to 1.0.",
        "At nominal 5%, the new 30-point, 35-point, 45-point, ridge-0.3 and narrower-kernel options have successful-test FPRs of 3.00%, 3.02%, 4.00%, 3.50% and 3.52%, respectively. The 35-point and narrower-kernel variants each have one failed null test; the other three have none. The 40-point reference has 5% FPR and no null failures. These small pilots do not prove exact calibration.",
        "The refinements do not support replacing the 40-point reference for the beta=1 objective. Forty-five points is a tradeoff favoring moderate signals. The present evidence does not establish that any tested LIME configuration meets the 80% target at beta=1; repeated selection using the same seeds is tuning, not independent validation.",
        "Conditional power counts failed signal tests as misses; marginal detection includes the probability of targeting a signal. Intervals are pointwise 95% Wilson Monte Carlo intervals, not simultaneous or paired-difference intervals. More explanation points increase work per proposal, so proposal counts alone are not runtime measurements. source_manifest.json records actual settings and source locations.",
        "## Strong-signal comparison", common.table(summary, list(summary.columns)),
        "## All power results", common.table(m,["option","beta","signal_targets","failed_signal","conditional_failures_as_misses","conditional_lower","conditional_upper","marginal_power","mean_proposals"]),
        "## Null calibration at 5%", common.table(calibration.reset_index(),["option","n","failed","fpr","ci_lower","ci_upper"])]
    (OUT/"analysis_report.md").write_text('\n\n'.join(report),encoding="utf-8")
    (OUT/"README.md").write_text(
        "# Dataset guide\n\nThe unified CSV has 2,000 null and 8,000 power records for ten options. "
        "Each option has 200 trials per configuration, all matching the original shard/iteration IDs. "
        "The full and matched CSVs are identical here; do not add them together. "
        "Key: scope, option, validation_shard, iteration, selection_event and signal_strength (power only). "
        "source_run identifies provenance. null_tests.csv, power_targets.csv and power_features.csv retain separate raw tables. "
        "Summary views overlap and must not be pooled. paired_detection_changes_vs_lime40.csv records paired gains and losses in marginal detection. "
        "ranking_summary.csv reports multiple descriptive criteria. Use the analysis report for interpretation. "
        "All three plots are saved as PNG and vector PDF. Earlier distinct LIME settings remain in previous comparison directories.\n",encoding="utf-8")
    print(ranking.to_string())
    print(f"Saved {len(unified)} full records and {int(unified.matched_200.sum())} matched records to {OUT}")


if __name__=="__main__": main()
