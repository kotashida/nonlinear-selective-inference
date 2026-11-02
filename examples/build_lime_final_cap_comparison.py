"""Audit and plot the selected methods plus matched LIME AIS cap follow-up."""
from __future__ import annotations

import html
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
import numpy as np
import pandas as pd

from analyze_ais_method_comparison import _read_raw, _wilson_interval


ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "outputs/selection_method_validation/exact_set"
SOURCE = BASE / "selected_four_methods_combined_ais_mi30"
LIME_51_RUN = BASE / "lime_ais_tuning/five_settings_200_trials/ais_combined"
LIME_102_RUN = BASE / "lime_ais_followup_200_20260913/cap102400"
OUT = BASE / "selected_four_methods_lime_cap_comparison_20260913"
BETAS = [0.3, 0.5, 0.75, 1.0]
METHODS = {
    "shap": "Tree-SHAP",
    "lime_cap_51200": "LIME (cap=51,200)",
    "lime_cap_102400": "LIME (cap=102,400)",
    "spline_screening": "Spline screening",
    "mutual_information": "Mutual information (30 neighbors)",
}
COLORS = dict(zip(METHODS, ["#2674a6", "#d58122", "#d94c4c", "#258568", "#9256a0"]))
FIGURES: list[tuple[str, str]] = []


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def save(fig, name: str, caption: str) -> None:
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    for ext in ("png", "pdf"):
        fig.savefig(OUT / f"{name}.{ext}", dpi=180, bbox_inches="tight")
    plt.close(fig)
    FIGURES.append((name, caption))


def audit_configurations() -> tuple[dict, dict]:
    old_manifest = read_json(LIME_51_RUN / "pooled_summary/validation_settings.json")
    new_manifest = read_json(LIME_102_RUN / "pooled_summary/validation_settings.json")
    differing = sorted(k for k in old_manifest.keys() | new_manifest.keys()
                       if old_manifest.get(k) != new_manifest.get(k))
    assert differing == ["max_final_samples"], differing
    assert old_manifest["max_final_samples"] == 51200
    assert new_manifest["max_final_samples"] == 102400

    rows = []
    conditions = ["null_fresh", *[f"power_feature_0_beta_{str(b).replace('.', 'p')}" for b in BETAS]]
    for shard in range(32):
        for condition in conditions:
            suffix = Path(f"shard_{shard}/lime/baseline/{condition}/settings.json")
            old, new = read_json(LIME_51_RUN / suffix), read_json(LIME_102_RUN / suffix)
            keys = sorted(old.keys() | new.keys())
            diff = [key for key in keys if old.get(key) != new.get(key)]
            assert diff == ["max_final_samples"], (shard, condition, diff)
            rows.append({"shard": shard, "condition": condition,
                         "differing_keys": "max_final_samples", "all_other_settings_match": True,
                         "cap_51200": old["max_final_samples"], "cap_102400": new["max_final_samples"]})
        old_design = np.load(LIME_51_RUN / f"shard_{shard}/lime/baseline/null_fresh/fixed_design.npy")
        new_design = np.load(LIME_102_RUN / f"shard_{shard}/lime/baseline/null_fresh/fixed_design.npy")
        np.testing.assert_array_equal(old_design, new_design)
    pd.DataFrame(rows).to_csv(OUT / "configuration_audit.csv", index=False)

    categories = [
        ("DGP/design", "Gaussian features; rho=0; nonlinear x + 0.5(x^2-1), centered/scaled; Gaussian noise sigma=1"),
        ("Dimensions", "n_samples=100; n_features=20; k_select=2"),
        ("Signal/power", "signal feature 0; beta=0.3,0.5,0.75,1.0; alpha=0.05; 200 trials per beta"),
        ("Null", "global null; fixed design per shard; fresh auxiliary randomization; alpha=0.01,0.05,0.10; 200 trials"),
        ("AIS", "pilot_iters=10; pilot_samples=320; batch=800; denominator ESS=80; tail ESS=15; early stop=true; MCMC steps=20"),
        ("Selection event", "exact_set; selection_decimals=10"),
        ("Inference", "AIS self-normalized importance sampling; uniform target from selected set; no multiplicity adjustment; known sigma"),
        ("LIME selector", "40 explanation rows; 64 neighborhood samples; ridge alpha=1; default width 0.75*sqrt(p); RF(50 trees,max_depth=5,random_state=42); mean absolute standardized coefficient"),
        ("Seeds/design realization", "base shard seeds 20260908..20260939; design_seed=314159; fixed null-design arrays exactly equal across all 32 shards; matched target/data/Monte Carlo seeds"),
    ]
    pd.DataFrame([{"category": c, "common_setting": v, "match": True} for c, v in categories]).to_csv(
        OUT / "configuration_summary.csv", index=False)
    return old_manifest, new_manifest


def load_data(new_manifest: dict) -> pd.DataFrame:
    base = pd.read_csv(SOURCE / "unified_tests.csv")
    base["source_selection_method"] = base["selection_method"]
    base.loc[base.selection_method.eq("lime"), "selection_method"] = "lime_cap_51200"
    base["analysis_label"] = base.selection_method.map(METHODS)

    n, t, f = _read_raw(LIME_102_RUN, "ais", new_manifest, 32)
    keys = ["selection_method", "validation_shard", "iteration", "selection_event", "signal_strength"]
    t = t.merge(f[keys + ["feature", "t_obs", "proposals", "denominator_ess", "tail_ess"]],
                on=keys, validate="one_to_one")
    for frame, scope in ((n, "null_calibration"), (t, "power")):
        frame["scope"] = scope
        frame["ais_setting"] = "combined"
        frame["source_run"] = "lime_ais_followup_200_20260913/cap102400"
        frame["max_final_samples"] = 102400
        frame["pilot_iters"] = 10
        frame["pilot_samples"] = 320
        frame["total_proposals"] = frame.proposals + 3200
        frame["source_selection_method"] = "lime"
        frame["selection_method"] = "lime_cap_102400"
        frame["analysis_label"] = METHODS["lime_cap_102400"]
        frame["option"] = METHODS["lime_cap_102400"]
        frame["mi_neighbors"] = np.nan
    follow = pd.concat([n, t], ignore_index=True, sort=False)
    data = pd.concat([base, follow], ignore_index=True, sort=False)
    assert len(data) == 5000
    assert data.groupby(["selection_method", "scope"], dropna=False).size().to_dict() == {
        **{(m, "null_calibration"): 200 for m in METHODS},
        **{(m, "power"): 800 for m in METHODS},
    }
    duplicate_keys = ["selection_method", "scope", "validation_shard", "iteration", "selection_event", "signal_strength"]
    assert not data.duplicated(duplicate_keys).any()

    trial_rows = []
    for scope in ("null_calibration", "power"):
        extra = ["signal_strength"] if scope == "power" else []
        old = data[data.selection_method.eq("lime_cap_51200") & data.scope.eq(scope)]
        new = data[data.selection_method.eq("lime_cap_102400") & data.scope.eq(scope)]
        pair = old.merge(new, on=["validation_shard", "iteration", "selection_event", *extra],
                         suffixes=("_old", "_new"), validate="one_to_one")
        assert len(pair) == (800 if scope == "power" else 200)
        checked = ["target_feature", "target_seed", "auxiliary_u", "t_obs"]
        if scope == "null_calibration":
            checked.append("observed_selected_features")
        else:
            checked.append("target_is_signal")
        for key in checked:
            if key in {"auxiliary_u", "t_obs"}:
                np.testing.assert_allclose(pair[f"{key}_old"], pair[f"{key}_new"], rtol=0, atol=1e-14)
            else:
                np.testing.assert_array_equal(pair[f"{key}_old"], pair[f"{key}_new"])
        trial_rows.append({"scope": scope, "matched_trials": len(pair),
                           "matched_fields": ";".join(checked), "all_match": True})
    pd.DataFrame(trial_rows).to_csv(OUT / "trial_audit.csv", index=False)
    return data


def summarize(data: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    power, calibration = [], []
    for method in METHODS:
        null = data[data.selection_method.eq(method) & data.scope.eq("null_calibration")]
        valid = ~null.failed & null.p_value.notna()
        for alpha in (0.01, 0.05, 0.10):
            k = int((valid & null.p_value.lt(alpha)).sum())
            lo, hi = _wilson_interval(k, int(valid.sum()))
            calibration.append(dict(method=method, ais="combined", alpha=alpha, n=len(null),
                                    failures=int((~valid).sum()), rejections=k, fpr=k / valid.sum(),
                                    ci_lower=float(lo), ci_upper=float(hi),
                                    mean_total_proposals=null.total_proposals.mean()))
        for beta in BETAS:
            g = data[data.selection_method.eq(method) & data.scope.eq("power") & data.signal_strength.eq(beta)]
            sig = g[g.target_is_signal.astype(bool)]
            k = int((sig.p_value.lt(0.05) & ~sig.failed).sum())
            lo, hi = _wilson_interval(k, len(sig))
            power.append(dict(method=method, ais="combined", beta=beta, n=len(g), signal_targets=len(sig),
                              detections=k, conditional_power=k / len(sig), ci_lower=float(lo), ci_upper=float(hi),
                              marginal_power=k / len(g), failures=int(g.failed.sum()),
                              signal_failures=int(sig.failed.sum()), signal_failure_rate=sig.failed.mean(),
                              mean_total_proposals=g.total_proposals.mean(), mean_final_proposals=g.proposals.mean()))
    p, c = pd.DataFrame(power), pd.DataFrame(calibration)
    p.to_csv(OUT / "power_summary.csv", index=False)
    c.to_csv(OUT / "calibration_summary.csv", index=False)
    return p, c


def make_plots(data: pd.DataFrame, p: pd.DataFrame, c: pd.DataFrame) -> None:
    n, t = data[data.scope.eq("null_calibration")], data[data.scope.eq("power")]
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    for method, label in METHODS.items():
        g, color = p[p.method.eq(method)].sort_values("beta"), COLORS[method]
        values = [g.signal_targets / 200, g.conditional_power, g.marginal_power]
        for ax, value in zip(axes, values):
            ax.plot(g.beta, value, "o-", color=color, label=label)
        axes[1].fill_between(g.beta, g.ci_lower, g.ci_upper, color=color, alpha=0.10)
    titles = ["Probability of targeting the signal", "Conditional power", "Marginal detection power"]
    for ax, title in zip(axes, titles):
        ax.set(title=title, xlabel="Signal strength (beta)", xticks=BETAS, ylim=(0, 1.03))
        ax.yaxis.set_major_formatter(PercentFormatter(1)); ax.grid(alpha=0.2)
    axes[1].axhline(0.8, color="gray", ls="--"); axes[0].legend(fontsize=8)
    fig.suptitle("Combined AIS: power and signal targeting")
    save(fig, "01_power_curves", "Left-to-right: signal-target probability, conditional power, and marginal detection power. Failed signal tests count as misses; shading is the pointwise 95% Wilson interval for conditional power.")

    fig, axes = plt.subplots(1, 2, figsize=(12, 5.5))
    for ax, metric, title in zip(axes, ["conditional_power", "ci_lower"], ["Observed conditional power", "95% lower confidence bound"]):
        mat = p.pivot(index="method", columns="beta", values=metric).reindex(METHODS)[BETAS]
        ax.imshow(mat, vmin=0, vmax=1, cmap="YlGnBu", aspect="auto")
        for i, method in enumerate(METHODS):
            for j, beta in enumerate(BETAS):
                row, value = p[p.method.eq(method) & p.beta.eq(beta)].iloc[0], mat.iloc[i, j]
                extra = f"\n{int(row.detections)}/{int(row.signal_targets)}" if metric == "conditional_power" else ("\n>= 80%" if value >= 0.8 else "")
                ax.text(j, i, f"{value:.1%}{extra}", ha="center", va="center", color="white" if value > 0.65 else "black", fontsize=9)
        ax.set(title=title, xticks=range(4), xticklabels=[f"{b:.2f}" for b in BETAS],
               yticks=range(5), yticklabels=list(METHODS.values()), xlabel="Signal strength (beta)")
    fig.suptitle("Combined AIS: the 80% power requirement")
    save(fig, "02_power_target_heatmap", "Counts show signal detections / signal targets, with failures retained in the denominator. Confidence bounds are pointwise.")

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for j, (method, label) in enumerate(METHODS.items()):
        g = c[c.method.eq(method)].sort_values("alpha")
        axes[0].errorbar(g.alpha + (j - 2) * 0.0015, g.fpr,
                         yerr=np.maximum(0, np.vstack([g.fpr-g.ci_lower, g.ci_upper-g.fpr])),
                         fmt="o", capsize=3, color=COLORS[method], label=label)
        vals = np.sort(n[n.selection_method.eq(method)].p_value)
        axes[1].step(vals, np.arange(1, len(vals)+1)/len(vals), where="post", color=COLORS[method], label=label)
    axes[0].plot([0, .12], [0, .12], "--", color="gray")
    axes[0].set(title="Null rejection rates and 95% intervals", xlabel="Nominal alpha (points offset for readability)", ylabel="Observed false-positive rate", xticks=[.01,.05,.1], ylim=(0,.18)); axes[0].legend(fontsize=8)
    axes[1].plot([0,1],[0,1],"--",color="gray"); axes[1].set(title="Null p-value empirical distribution", xlabel="p-value", ylabel="Fraction at or below p", xlim=(0,1), ylim=(0,1))
    for ax in axes: ax.grid(alpha=.2)
    fig.suptitle("Combined AIS: null calibration (200 trials per method)")
    save(fig, "03_null_calibration", "The diagonal is the nominal/uniform reference. Small departures in 200 trials do not establish calibration or miscalibration.")

    fig, axes = plt.subplots(2, 3, figsize=(14, 8)); outcomes=[]
    for ax, (method, label) in zip(axes.flat, METHODS.items()):
        g=p[p.method.eq(method)].sort_values("beta"); det=g.detections/g.signal_targets; fail=g.signal_failures/g.signal_targets; miss=1-det-fail; x=np.arange(4)
        ax.bar(x,det,color="#258568",label="Detected"); ax.bar(x,miss,bottom=det,color="#bcc8d1",label="Valid, not detected"); ax.bar(x,fail,bottom=det+miss,color="#bc4545",label="Inference failed")
        for i,row in enumerate(g.itertuples()):
            ax.text(i,1.02,f"n={row.signal_targets}\nfailed={row.signal_failures}",ha="center",fontsize=8)
            outcomes.append(dict(method=method,beta=row.beta,signal_targets=row.signal_targets,detected=row.detections,valid_not_detected=row.signal_targets-row.detections-row.signal_failures,failed=row.signal_failures))
        ax.set(title=label,xticks=x,xticklabels=BETAS,ylim=(0,1.18),xlabel="Signal strength (beta)"); ax.yaxis.set_major_formatter(PercentFormatter(1))
    axes.flat[5].axis("off"); axes[0,0].legend(fontsize=8,loc="lower right"); fig.suptitle("Combined AIS: outcomes among signal targets")
    save(fig,"04_signal_outcomes","Separates statistical non-detection from unresolved inference. Each bar is conditional on the target being a true signal.")
    pd.DataFrame(outcomes).to_csv(OUT/"signal_outcomes.csv",index=False)

    fig,axes=plt.subplots(1,2,figsize=(12,5)); costs=[]
    for method,label in METHODS.items():
        g=p[p.method.eq(method)].sort_values("beta"); axes[0].plot(g.beta,g.mean_total_proposals,"o-",color=COLORS[method],label=label)
        vals=np.sort(t[t.selection_method.eq(method)].total_proposals); axes[1].step(vals,np.arange(1,len(vals)+1)/len(vals),where="post",color=COLORS[method])
        cap=int(data[data.selection_method.eq(method)].max_final_samples.iloc[0])
        for beta in BETAS:
            raw=t[t.selection_method.eq(method)&t.signal_strength.eq(beta)]
            costs.append(dict(method=method,beta=beta,mean=raw.total_proposals.mean(),median=raw.total_proposals.median(),p95=raw.total_proposals.quantile(.95),maximum=raw.total_proposals.max(),first_batch_fraction=raw.proposals.eq(800).mean(),cap_fraction=raw.proposals.eq(cap).mean()))
    axes[0].set(title="Mean cost by signal strength",xlabel="Signal strength (beta)",ylabel="Proposals including pilot",xticks=BETAS);axes[0].legend(fontsize=8)
    axes[1].set(title="Cost distribution across 800 power trials per method",xlabel="Proposals including pilot (log scale)",ylabel="Fraction of trials at or below cost",xscale="log",ylim=(0,1.03))
    for ax in axes:ax.grid(alpha=.2)
    fig.suptitle("Combined AIS: sampling effort (3,200 pilot + final proposals)")
    save(fig,"05_sampling_cost","Counts measure sampling effort, not wall-clock time. Selector evaluation costs differ between methods.")
    pd.DataFrame(costs).to_csv(OUT/"sampling_cost_summary.csv",index=False)

    fig,axes=plt.subplots(2,3,figsize=(14,8),sharex=True,sharey=True)
    for ax,(method,label) in zip(axes.flat,METHODS.items()):
        g=t[t.selection_method.eq(method)]
        for failed,marker,color in [(False,"o",COLORS[method]),(True,"x","#bf3232")]:
            q=g[g.failed.eq(failed)]; ax.scatter(q.denominator_ess,q.tail_ess,s=15 if not failed else 40,alpha=.3 if not failed else .9,marker=marker,color=color,label="Failed" if failed else "Successful")
        ax.axvline(80,ls="--",color="gray");ax.axhline(15,ls="--",color="gray");ax.set(title=f"{label}: {int(g.failed.sum())}/800 failures",xlabel="Denominator ESS",ylabel="Tail ESS",xscale="log",yscale="log");ax.grid(alpha=.15)
    axes.flat[5].axis("off");axes[0,0].legend(fontsize=8);fig.suptitle("Combined AIS: final effective sample sizes across all power trials")
    save(fig,"06_ess_diagnostics","Dashed lines mark denominator ESS=80 and tail ESS=15 thresholds. Failed inference is marked in red; both axes are logarithmic.")

    fig,ax=plt.subplots(figsize=(9,5))
    for method,label in METHODS.items():
        g=p[p.method.eq(method)&p.beta.eq(1)].iloc[0]
        ax.errorbar(g.mean_total_proposals,g.conditional_power,yerr=[[g.conditional_power-g.ci_lower],[g.ci_upper-g.conditional_power]],fmt="o",capsize=4,color=COLORS[method],label=label)
        ax.annotate(f"{g.conditional_power:.1%}",(g.mean_total_proposals,g.conditional_power),xytext=(7,0),textcoords="offset points",color=COLORS[method])
    ax.axhline(.8,ls="--",color="gray"); xmax=p[p.beta.eq(1)].mean_total_proposals.max(); xmin=p[p.beta.eq(1)].mean_total_proposals.min(); pad=max(300,(xmax-xmin)*.12)
    ax.set(xlabel="Mean proposals per power trial, including pilot",ylabel="Conditional power",ylim=(.77,1.015),xlim=(max(0,xmin-pad),xmax+pad));ax.yaxis.set_major_formatter(PercentFormatter(1));ax.legend(loc="lower left",fontsize=8);ax.grid(alpha=.2)
    fig.suptitle("Combined AIS at beta=1: power versus sampling effort")
    save(fig,"07_power_cost_tradeoff","Vertical bars show pointwise 95% power intervals. Sampling counts are not measured runtimes.")


def write_reports(data: pd.DataFrame, p: pd.DataFrame, old: dict, new: dict) -> None:
    selected = {
        "comparison": list(METHODS.values()),
        "common": {"n_samples":100,"n_features":20,"k_select":2,"feature_correlation":0.0,"sigma":1.0,
                   "signal_features":[0],"signal_strengths":BETAS,"selection_event":"exact_set","target_rule":"uniform_from_selected",
                   "inference_method":"ais","pilot_iters":10,"pilot_samples":320,"final_batch_size":800,
                   "min_denominator_ess":80,"min_tail_ess":15,"stop_when_ess_met":True},
        "lime": {"num_explanations":40,"num_samples":64,"ridge_alpha":1.0,"caps":[51200,102400]},
        "audit_result":"Only max_final_samples differs between the two LIME runs."
    }
    (OUT/"selected_configuration.json").write_text(json.dumps(selected,indent=2),encoding="utf-8")
    (OUT/"source_manifest.json").write_text(json.dumps({"selected_four_method_sources":read_json(SOURCE/"source_manifest.json"),"lime_cap_102400":new},indent=2),encoding="utf-8")
    data.to_csv(OUT/"unified_tests.csv",index=False)
    strongest=p[p.beta.eq(1)].set_index("method")
    notes="At beta=1: "+", ".join(f"{METHODS[m]} {strongest.loc[m,'conditional_power']:.1%}" for m in METHODS)+"."
    intro="Five matched configurations, each with 200 null trials and 200 power trials per beta. All settings match between the two LIME arms except max_final_samples (51,200 versus 102,400); both use a 3,200-draw pilot and 800-draw final batches."
    report=["# LIME final-sample-cap comparison",intro,notes,
            "## Configuration audit","The pooled manifests and all 160 pairs of per-shard/per-condition settings were compared. The only differing key is `max_final_samples`. The records also match target feature, target seed, auxiliary draw, observed statistic, signal-target status, and (for null trials) selected set. See `configuration_summary.csv`, `configuration_audit.csv`, and `trial_audit.csv`.",
            "## Plot gallery"]
    sections=[]
    for name,caption in FIGURES:
        title=name[3:].replace("_"," ").title();report.append(f"### {title}\n\n{caption}\n\n![{name}]({name}.png)\n\n[PDF]({name}.pdf)")
        sections.append(f'<section><h2>{html.escape(title)}</h2><p>{html.escape(caption)}</p><a href="{name}.png"><img src="{name}.png" alt="{html.escape(caption)}"></a><p><a href="{name}.pdf">Download PDF</a></p></section>')
    report.append("## Data\n\n`unified_tests.csv` contains 5,000 records: 1,000 per displayed configuration. The two LIME arms reuse matched simulation seeds and differ only in AIS final-sample cap; they are paired, not independent experiments. Proposal counts include the 3,200 pilot draws.")
    (OUT/"README.md").write_text("\n\n".join(report),encoding="utf-8")
    page=('<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1"><title>LIME cap comparison</title><style>body{font:17px/1.6 system-ui;max-width:1200px;margin:32px auto;padding:0 24px;color:#203040;background:#f5f7fa}section{background:white;padding:24px;margin:24px 0;border-radius:12px}img{width:100%;height:auto}a{color:#17669b}</style><h1>LIME final-sample-cap comparison</h1><p>'
          +html.escape(intro)+'</p><p>'+html.escape(notes)+'</p>'+''.join(sections)+'</html>')
    (OUT/"index.html").write_text(page,encoding="utf-8")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.size":10,"axes.spines.top":False,"axes.spines.right":False})
    old,new=audit_configurations()
    data=load_data(new)
    p,c=summarize(data)
    make_plots(data,p,c)
    write_reports(data,p,old,new)
    print(f"Verified configurations, matched trials, and saved {len(FIGURES)} PNG/PDF figures plus data to {OUT}")
    print(p[p.beta.eq(1)][["method","signal_targets","detections","conditional_power","signal_failures","mean_total_proposals"]].to_string(index=False))


if __name__ == "__main__":
    main()
