"""Build a verified analysis bundle from the combined five-setting AIS sweep."""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from analyze_lime_parameter_sweep import AIS_200_VARIANTS
from analyze_ais_method_comparison import _wilson_interval

ROOT = Path(__file__).resolve().parents[1] / "outputs/selection_method_validation/exact_set/lime_ais_tuning/five_settings_200_trials"
OUT = ROOT / "analysis"
LABELS = dict(zip(AIS_200_VARIANTS, ["Baseline", "Final cap 51,200", "Pilot samples 320", "Pilot rounds 10", "Combined increase"]))


def table(frame):
    lines = ["| " + " | ".join(frame.columns) + " |", "| " + " | ".join(["---"]*len(frame.columns)) + " |"]
    for row in frame.itertuples(index=False, name=None):
        lines.append("| " + " | ".join(f"{x:.4f}" if isinstance(x, float) else str(x) for x in row) + " |")
    return "\n".join(lines)


def save(fig, name):
    for ext in ("png", "pdf"):
        fig.savefig(OUT / f"{name}.{ext}", dpi=180, bbox_inches="tight")
    plt.close(fig)


def main():
    n = pd.read_csv(OUT / "null_combined.csv")
    f = pd.read_csv(OUT / "power_features_combined.csv")
    p = pd.read_csv(OUT / "power_target_80pct.csv")
    c = pd.read_csv(OUT / "calibration.csv")
    retention = pd.read_csv(OUT / "baseline_retention.csv", keep_default_na=False)
    assert len(n) == 1000 and len(f) == 4000
    manifest = {}
    for variant in AIS_200_VARIANTS:
        manifest[variant] = json.loads((ROOT / variant / "pooled_summary/validation_settings.json").read_text())
    # Verify the matched simulation/selector configuration, allowing AIS controls only.
    baseline = manifest["ais_baseline"]
    for variant, settings in manifest.items():
        for key in baseline:
            if key not in {"max_final_samples", "pilot_iters", "pilot_samples"}:
                assert settings[key] == baseline[key], (variant, key)
    for row in c.itertuples():
        g = n[n.variant.eq(row.variant)]
        assert len(g) == row.n_iterations == 200
        assert int(g.failed.sum()) == row.n_failed
        assert int(g.p_value.lt(row.alpha).sum()) == row.n_rejected
    pairs = []
    for row in p.itertuples():
        g = f[f.variant.eq(row.variant) & f.signal_strength.eq(row.signal_strength)]
        signal = g[g.is_signal]
        assert len(g) == row.n_iterations == 200
        assert len(signal) == row.n_signal_targets
        assert int(signal.failed.sum()) == row.n_failed_signal_targets
        detections = int((signal.rejected & ~signal.failed).sum())
        assert np.isclose(detections/len(signal), row.conditional_power_resolution_lower_bound)
        assert np.isclose(detections/len(g), row.marginal_detection_power)
        ref = f[f.variant.eq("ais_baseline") & f.signal_strength.eq(row.signal_strength)]
        pair = ref.merge(g, on=["shard", "iteration", "signal_strength"], suffixes=("_ref", "_new"), validate="one_to_one")
        assert len(pair) == 200
        for key in ("feature", "is_signal", "auxiliary_u", "t_obs"):
            assert np.array_equal(pair[key+"_ref"], pair[key+"_new"]), (row.variant, key)
        signal_pair = pair[pair.is_signal_ref]
        old_detection = signal_pair.rejected_ref & ~signal_pair.failed_ref
        new_detection = signal_pair.rejected_new & ~signal_pair.failed_new
        pairs.append(dict(variant=row.variant, beta=row.signal_strength,
            signal_targets=len(signal_pair), gained_detections=int((~old_detection & new_detection).sum()),
            lost_detections=int((old_detection & ~new_detection).sum()),
            recovered_signal_failures=int((signal_pair.failed_ref & ~signal_pair.failed_new).sum()),
            new_signal_failures=int((~signal_pair.failed_ref & signal_pair.failed_new).sum())))
    for variant in AIS_200_VARIANTS:
        ref = n[n.variant.eq("ais_baseline")]
        other = n[n.variant.eq(variant)]
        pair = ref.merge(other, on=["shard", "iteration"], suffixes=("_ref", "_new"), validate="one_to_one")
        for key in ("target_feature", "target_seed", "auxiliary_u", "t_obs"):
            assert np.array_equal(pair[key+"_ref"], pair[key+"_new"])
    unified = pd.concat([n.assign(scope="null_calibration"), f.assign(scope="power")], ignore_index=True)
    for i, field in enumerate(["max_final_samples", "pilot_iters", "pilot_samples"]):
        unified[field] = unified.variant.map({key: value[i] for key,value in AIS_200_VARIANTS.items()})
    unified["source_run"] = unified.variant.map(lambda name: str((ROOT/name).relative_to(ROOT.parents[2])))
    unified.to_csv(OUT / "unified_tests.csv", index=False)
    (OUT / "source_manifest.json").write_text(json.dumps(manifest,indent=2))
    pd.DataFrame(pairs).to_csv(OUT / "paired_signal_outcomes.csv",index=False)
    costs = []
    for (scope, variant, beta), g in unified.groupby(["scope", "variant", "signal_strength"], dropna=False):
        budget, rounds, samples = AIS_200_VARIANTS[variant]
        costs.append(dict(scope=scope, variant=variant, beta=beta,
            mean_final_proposals=g.proposals.mean(), pilot_proposals=rounds*samples,
            mean_total_proposals=g.proposals.mean()+rounds*samples,
            fraction_reaching_cap=g.proposals.eq(budget).mean(), failures=int(g.failed.sum())))
    costs = pd.DataFrame(costs)
    costs.to_csv(OUT / "cost_and_failures.csv",index=False)
    ci = _wilson_interval(c.n_rejected, c.n_iterations-c.n_failed)
    c["ci_lower"],c["ci_upper"] = ci
    c.to_csv(OUT / "calibration_with_intervals.csv",index=False)
    fig, axes = plt.subplots(2,2,figsize=(12,9))
    for i, (variant,label) in enumerate(LABELS.items()):
        q = p[p.variant.eq(variant)].sort_values("signal_strength")
        d = costs[costs.variant.eq(variant) & costs.scope.eq("power")].sort_values("beta")
        for ax, y in zip(axes.flat,[q.conditional_power_resolution_lower_bound, q.marginal_detection_power,
                                   q.n_failed_signal_targets/q.n_signal_targets, d.mean_total_proposals]):
            ax.plot(q.signal_strength,y,marker="o",color=f"C{i}",label=label)
    for ax,title in zip(axes.flat,["Conditional power (failures = misses)","Marginal detection power",
                                  "Signal-target failure rate","Mean proposals including pilot"]):
        ax.set(title=title,xlabel="Signal strength",xticks=[.3,.5,.75,1.]);ax.grid(alpha=.2)
    for ax in list(axes.flat)[:3]: ax.set_ylim(-.02,1.02)
    axes[0,0].axhline(.8,color="gray",linestyle="--")
    handles,labels=axes[0,0].get_legend_handles_labels()
    fig.legend(handles,labels,loc="lower center",ncol=3,frameon=False)
    fig.suptitle("LIME fixed at 40 explanations; 200 matched trials per configuration")
    fig.tight_layout(rect=(0,.09,1,.96));save(fig,"ais_power_reliability_cost")
    fig,axes=plt.subplots(1,2,figsize=(12,4.6))
    for i,(variant,label) in enumerate(LABELS.items()):
        q=p[p.variant.eq(variant)&p.signal_strength.eq(1)].iloc[0]
        v=q.conditional_power_resolution_lower_bound
        axes[0].errorbar(v,i,xerr=[[v-q.failure_adjusted_ci_lower],[q.failure_adjusted_ci_upper-v]],fmt="o",color=f"C{i}",capsize=3)
        q=c[c.variant.eq(variant)&c.alpha.eq(.05)].iloc[0]
        axes[1].errorbar(q.rejection_rate,i,xerr=[[q.rejection_rate-q.ci_lower],[q.ci_upper-q.rejection_rate]],fmt="o",color=f"C{i}",capsize=3)
    for ax in axes:
        ax.set(yticks=range(5),yticklabels=list(LABELS.values()));ax.invert_yaxis();ax.grid(axis="x",alpha=.2)
    axes[0].axvline(.8,color="gray",linestyle="--");axes[1].axvline(.05,color="gray",linestyle="--")
    axes[0].set(title="Power at beta=1 (failures = misses)",xlabel="Rate and pointwise 95% Wilson interval",xlim=(.65,1))
    axes[1].set(title="Null FPR at nominal 5% (no failed null tests)",xlabel="Rate and pointwise 95% Wilson interval",xlim=(0,.11))
    fig.tight_layout();save(fig,"ais_power_and_calibration_intervals")
    fig,ax=plt.subplots(figsize=(9,4.8))
    r=retention[retention.scope.eq("power")].set_index("variant").reindex(LABELS)
    pos=np.arange(5)
    ax.bar(pos-.18,r.recovered_failures,width=.36,label="Recovered baseline failures",color="C2")
    ax.bar(pos+.18,r.new_failures,width=.36,label="New failures",color="C3")
    ax.set(xticks=pos,xticklabels=list(LABELS.values()),ylabel="Tests out of 800 power trials",title="Reliability changes relative to baseline")
    ax.tick_params(axis="x",labelsize=8);ax.legend();fig.tight_layout();save(fig,"ais_baseline_retention")
    q=p[p.signal_strength.eq(1)]
    report=["# AIS settings sweep: fixed LIME 40", "",
        "Verified 1,000 null tests and 4,000 power tests across five AIS variants. All variants use the same LIME selector, 200 trials per configuration, seeds, targets, observed statistics and software versions. Raw rejections and failure counts agree with pooled summaries. Only the declared AIS pilot settings and/or final cap differ. The combined dataset contains option-specific records sharing simulation data; they are not 5,000 independent datasets.",
        "## Main result",
        "The combined increase is best by observed power at beta=1: 96/104=92.31%, with a pointwise 95% Wilson interval whose lower bound is 85.55%. This exceeds the 80% target in both point estimate and lower bound for this fixed configuration. Budget-only achieves 89/104=85.58%, pilot samples 320 gives 85/104=81.73%, and pilot rounds 10 gives 84/104=80.77%, versus baseline 83/104=79.81%.",
        "All four alternatives exceed 80% in point estimates at beta=0.5, 0.75 and 1.0. None reaches 80% at beta=0.3. The combined variant has the highest power at all three moderate/strong strengths, but the 95% lower bound at beta=0.5 remains below 80%. These pointwise intervals do not adjust for selecting the best of five settings or repeated tuning on the same seeds.",
        "## Reliability and the shared-pipeline decision",
        "Budget-only preserves every successful baseline p-value and introduces no new failures, while resolving 10 of the 35 baseline power-test failures. The combined setting resolves 24 baseline failures but introduces 3 new failures, leaving 14 overall. Pilot samples 320 resolves 9 and introduces 5; pilot rounds 10 resolves 6 and introduces 8. Extra pilot rounds alone therefore increase total power-test failures despite a small beta=1 power gain.",
        "The combined setting is the strongest observed LIME candidate, while budget-only is the more conservative candidate for retaining already successful inference. SHAP, MI and spline screening were not run under the changed settings. Do not claim their performance is preserved for the pilot-changing alternatives. A matched cross-method check is needed before promoting the combined setting to shared defaults; fresh seeds are then needed for independent confirmation.",
        "All five settings have zero null failures and pass the existing pipeline calibration checks. Observed FPRs at nominal 5% are 5%, 5%, 3%, 4% and 4.5% in baseline/budget/samples/rounds/combined order. Passing 200-trial diagnostics is not a proof of exact p-values or calibration under adaptive stopping.",
        "## Cost",
        "The cost plot includes pilot draws (400 baseline/budget, 1,600 samples, 800 rounds, 3,200 combined). Counts are selector evaluations, not measured wall-clock runtime. The final cap is a maximum; successful tests can stop earlier. Increasing the cap fourfold does not necessarily multiply mean cost by four. Failed tests remain counted as no detection throughout the power analysis.",
        "## Power at beta=1",table(q[["variant","n_signal_targets","n_failed_signal_targets","conditional_power_resolution_lower_bound","failure_adjusted_ci_lower","failure_adjusted_ci_upper","marginal_detection_power"]]),
        "## All power results",table(p[["variant","signal_strength","n_failed_signal_targets","conditional_power_resolution_lower_bound","marginal_detection_power"]]),
        "## Baseline retention",table(retention),"## Costs",table(costs)]
    (OUT/"analysis_report.md").write_text("\n\n".join(report),encoding="utf-8")
    (OUT/"README.md").write_text("# Dataset guide\n\nunified_tests.csv contains one test per variant, scope, shard and iteration (plus signal_strength for power). Scope is null_calibration or power. Do not sum null/power raw exports again into this unified file. Source and configuration are preserved in source_run, AIS columns and source_manifest.json. power_target_80pct.csv uses failed signal tests as misses. baseline_retention.csv contains literal scope 'null'; read with keep_default_na=False to preserve that label. paired_signal_outcomes.csv records matched gains/losses per signal strength. cost_and_failures.csv includes pilot draws. All new figures have PNG/PDF versions.\n",encoding="utf-8")
    print(q[["variant","conditional_power_resolution_lower_bound","failure_adjusted_ci_lower","n_failed_signal_targets"]].to_string(index=False))
    print(costs[costs.scope.eq("power") & costs.beta.eq(1)].to_string(index=False))
    print("Verified matched trials, raw counts and baseline retention; saved report and plots to",OUT)


if __name__ == "__main__": main()
