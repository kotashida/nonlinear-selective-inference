"""Plot the five matched AIS-fix settings in the style of the selected-method report."""
from pathlib import Path
import html

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
import numpy as np
import pandas as pd

from analyze_ais_method_comparison import _wilson_interval

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs/selection_method_validation/exact_set/ais_fix_tests"
CONDITIONS = {
    "baseline_reduced": "Baseline (reduced)",
    "multimodal_qadapt": "Multimodal q-adaptation",
    "raised_sd_floor": "Raised SD floor",
    "adaptive_mixture_weights": "Adaptive mixture weights",
    "reallocated_pilot_exploration": "Reallocated pilot exploration",
}
COLORS = dict(zip(CONDITIONS, ["#2674a6", "#d58122", "#258568", "#9256a0", "#bc4545"]))
BETAS = [0.50, 0.75, 1.00]
N = 100
FIGURES = []


def save(fig, name, caption):
    fig.tight_layout(rect=(0, 0, 1, .94))
    for ext in ("png", "pdf"):
        fig.savefig(OUT / f"{name}.{ext}", dpi=180, bbox_inches="tight")
    plt.close(fig)
    FIGURES.append((name, caption))


def summarize_power(t):
    rows = []
    for (condition, beta), g in t.groupby(["condition", "signal_strength"], sort=False):
        sig = g[g.is_signal]
        detected = int((sig.p_value.lt(.05) & ~sig.failed).sum())
        lo, hi = _wilson_interval(detected, len(sig))
        rows.append(dict(condition=condition, beta=beta, trials=len(g), signal_targets=len(sig),
                         detections=detected, signal_failures=int(sig.failed.sum()),
                         conditional_power=detected / len(sig), ci_lower=lo, ci_upper=hi,
                         marginal_power=detected / len(g), mean_aj_evaluations=g.aj_evaluations.mean()))
    return pd.DataFrame(rows)


def main():
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    data = pd.read_csv(OUT / "all_target_diagnostics.csv")
    # Older pooled files left the null-scope label blank; signal strength is the
    # authoritative discriminator and is populated for every power record.
    data["scope"] = np.where(data.signal_strength.notna(), "power", "null")
    data["failed"] = data.failed.astype(bool)
    data["is_signal"] = data.is_signal.fillna(False).astype(bool)
    assert set(data.condition) == set(CONDITIONS) and len(data) == 5 * 400
    assert (data.groupby(["condition", "scope"]).size().unstack()[["null", "power"]] == [100, 300]).all().all()
    n, t = data[data.scope.eq("null")].copy(), data[data.scope.eq("power")].copy()
    p = summarize_power(t)
    p.to_csv(OUT / "power_summary.csv", index=False)

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    for condition, label in CONDITIONS.items():
        g = p[p.condition.eq(condition)].sort_values("beta"); color = COLORS[condition]
        for ax, vals in zip(axes, [g.signal_targets / N, g.conditional_power, g.marginal_power]):
            ax.plot(g.beta, vals, "o-", color=color, label=label)
        axes[1].fill_between(g.beta, g.ci_lower, g.ci_upper, color=color, alpha=.10)
    for ax, title in zip(axes, ["Probability of targeting the signal", "Conditional power", "Marginal detection power"]):
        ax.set(title=title, xlabel="Signal strength (beta)", xticks=BETAS, ylim=(0, 1.03)); ax.yaxis.set_major_formatter(PercentFormatter(1)); ax.grid(alpha=.2)
    axes[0].axhline(.5, color="gray", ls="--"); axes[2].axhline(.5, color="gray", ls="--"); axes[0].legend(fontsize=8)
    fig.suptitle("AIS fix tests: power and signal targeting")
    save(fig, "01_power_curves", "Failed signal tests count as misses. Shading is the pointwise 95% Wilson interval for conditional power. Marginal power also accounts for signal targeting.")

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for ax, metric, title in zip(axes, ["conditional_power", "ci_lower"], ["Observed conditional power", "95% lower confidence bound"]):
        mat = p.pivot(index="condition", columns="beta", values=metric).reindex(CONDITIONS)[BETAS]
        ax.imshow(mat, vmin=0, vmax=1, cmap="YlGnBu", aspect="auto")
        for i, condition in enumerate(CONDITIONS):
            for j, beta in enumerate(BETAS):
                r = p[p.condition.eq(condition) & p.beta.eq(beta)].iloc[0]; v = mat.iloc[i, j]
                extra = f"\n{r.detections:.0f}/{r.signal_targets:.0f}" if metric == "conditional_power" else ("\n>= 80%" if v >= .8 else "")
                ax.text(j, i, f"{v:.1%}{extra}", ha="center", va="center", color="white" if v > .65 else "black")
        ax.set(title=title, xticks=range(3), xticklabels=[f"{b:.2f}" for b in BETAS], yticks=range(5), yticklabels=CONDITIONS.values(), xlabel="Signal strength (beta)")
    fig.suptitle("AIS fix tests: the 80% power requirement")
    save(fig, "02_power_target_heatmap", "Counts show signal detections / signal targets, with failures retained in the denominator. Confidence bounds are pointwise.")

    cal = []
    for condition, g in n.groupby("condition", sort=False):
        for alpha in (.01, .05, .10):
            k = int((g.p_value < alpha).sum()); lo, hi = _wilson_interval(k, len(g))
            cal.append(dict(condition=condition, alpha=alpha, rejections=k, trials=len(g), fpr=k/len(g), ci_lower=lo, ci_upper=hi))
    c = pd.DataFrame(cal); c.to_csv(OUT / "calibration_summary.csv", index=False)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for j, (condition, label) in enumerate(CONDITIONS.items()):
        g = c[c.condition.eq(condition)].sort_values("alpha"); color = COLORS[condition]
        yerr = np.maximum(0, np.vstack([g.fpr-g.ci_lower, g.ci_upper-g.fpr]))
        axes[0].errorbar(g.alpha + (j-2)*.0012, g.fpr, yerr=yerr, fmt="o", capsize=3, color=color, label=label)
        vals = np.sort(n[n.condition.eq(condition)].p_value); axes[1].step(vals, np.arange(1, len(vals)+1)/len(vals), where="post", color=color)
    axes[0].plot([0, .12], [0, .12], "--", color="gray"); axes[0].set(title="Null rejection rates and 95% intervals", xlabel="Nominal alpha (points offset)", ylabel="Observed false-positive rate", xticks=[.01,.05,.1], ylim=(0,.22)); axes[0].legend(fontsize=7)
    axes[1].plot([0,1],[0,1],"--",color="gray"); axes[1].set(title="Null p-value empirical distribution", xlabel="p-value", ylabel="Fraction at or below p", xlim=(0,1), ylim=(0,1))
    for ax in axes: ax.grid(alpha=.2)
    fig.suptitle("AIS fix tests: null calibration (100 trials per setting)")
    save(fig, "03_null_calibration", "The diagonal is the nominal/uniform reference. Small departures in 100 trials do not prove either miscalibration or exact calibration.")

    fig, axes = plt.subplots(2, 3, figsize=(15, 8), squeeze=False); outcomes=[]
    for ax, (condition, label) in zip(axes.flat, CONDITIONS.items()):
        g=p[p.condition.eq(condition)].sort_values("beta"); det=g.detections/g.signal_targets; fail=g.signal_failures/g.signal_targets; miss=1-det-fail; x=np.arange(3)
        ax.bar(x,det,color="#258568",label="Detected"); ax.bar(x,miss,bottom=det,color="#bcc8d1",label="Valid, not detected"); ax.bar(x,fail,bottom=det+miss,color="#bc4545",label="Inference failed")
        for i,r in enumerate(g.itertuples()):
            ax.text(i,1.02,f"n={r.signal_targets}\nfailed={r.signal_failures}",ha="center",fontsize=8)
            outcomes.append(dict(condition=condition,beta=r.beta,signal_targets=r.signal_targets,detected=r.detections,valid_not_detected=r.signal_targets-r.detections-r.signal_failures,failed=r.signal_failures))
        ax.set(title=label,xticks=x,xticklabels=BETAS,ylim=(0,1.18),xlabel="Signal strength (beta)"); ax.yaxis.set_major_formatter(PercentFormatter(1))
    axes.flat[-1].remove(); axes[0,0].legend(fontsize=8,loc="lower right"); fig.suptitle("AIS fix tests: outcomes among signal targets")
    save(fig,"04_signal_outcomes","Separates statistical non-detection from unresolved inference. Each bar is conditional on the target being a true signal.")
    pd.DataFrame(outcomes).to_csv(OUT/"signal_outcomes.csv",index=False)

    fig,axes=plt.subplots(1,2,figsize=(12,5)); costs=[]
    for condition,label in CONDITIONS.items():
        g=p[p.condition.eq(condition)].sort_values("beta"); axes[0].plot(g.beta,g.mean_aj_evaluations,"o-",color=COLORS[condition],label=label)
        vals=np.sort(t[t.condition.eq(condition)].aj_evaluations); axes[1].step(vals,np.arange(1,len(vals)+1)/len(vals),where="post",color=COLORS[condition])
        for beta in BETAS:
            raw=t[t.condition.eq(condition)&t.signal_strength.eq(beta)]; costs.append(dict(condition=condition,beta=beta,mean=raw.aj_evaluations.mean(),median=raw.aj_evaluations.median(),p95=raw.aj_evaluations.quantile(.95),maximum=raw.aj_evaluations.max(),first_batch_fraction=raw.proposals.eq(800).mean(),cap_fraction=raw.proposals.eq(51200).mean()))
    axes[0].set(title="Mean cost by signal strength",xlabel="Signal strength (beta)",ylabel="Total AIS evaluations",xticks=BETAS); axes[0].legend(fontsize=7)
    axes[1].set(title="Cost distribution across 300 power trials per setting",xlabel="Total AIS evaluations",ylabel="Fraction of trials at or below cost",ylim=(0,1.03))
    for ax in axes: ax.grid(alpha=.2)
    fig.suptitle("AIS fix tests: sampling effort"); save(fig,"05_sampling_cost","Total AIS evaluations include pilot and final proposals; counts are not wall-clock time.")
    pd.DataFrame(costs).to_csv(OUT/"sampling_cost_summary.csv",index=False)

    fig,axes=plt.subplots(2,3,figsize=(15,8),squeeze=False)
    for ax,(condition,label) in zip(axes.flat,CONDITIONS.items()):
        g=t[t.condition.eq(condition)]
        for failed,marker,color in [(False,"o",COLORS[condition]),(True,"x","#bf3232")]:
            q=g[g.failed.eq(failed)&g.denominator_ess.gt(0)&g.tail_ess.gt(0)]; ax.scatter(q.denominator_ess,q.tail_ess,s=15 if not failed else 40,alpha=.3 if not failed else .9,marker=marker,color=color,label="Failed" if failed else "Successful")
        ax.axvline(80,ls="--",color="gray"); ax.axhline(15,ls="--",color="gray"); ax.set(title=f"{label}: {int(g.failed.sum())}/300 failures",xlabel="Denominator ESS",ylabel="Tail ESS",xscale="log",yscale="log"); ax.grid(alpha=.15)
    axes.flat[-1].remove(); axes[0,0].legend(fontsize=8); fig.suptitle("AIS fix tests: effective sample sizes")
    save(fig,"06_ess_diagnostics","Dashed lines mark denominator ESS=80 and tail ESS=15 thresholds. Failed inference is marked in red; both axes are logarithmic.")

    fig,ax=plt.subplots(figsize=(10,5))
    for condition,label in CONDITIONS.items():
        g=p[p.condition.eq(condition)&p.beta.eq(1)].iloc[0]; ax.errorbar(g.mean_aj_evaluations,g.conditional_power,yerr=[[g.conditional_power-g.ci_lower],[g.ci_upper-g.conditional_power]],fmt="o",capsize=4,color=COLORS[condition],label=label); ax.annotate(f"{g.conditional_power:.1%}",(g.mean_aj_evaluations,g.conditional_power),xytext=(6,0),textcoords="offset points",color=COLORS[condition])
    ax.axhline(.8,ls="--",color="gray"); ax.set(xlabel="Mean total AIS evaluations per power trial",ylabel="Conditional power"); ax.yaxis.set_major_formatter(PercentFormatter(1)); ax.legend(fontsize=8); ax.grid(alpha=.2)
    fig.suptitle("AIS fix tests at beta=1: power versus sampling effort"); save(fig,"07_power_cost_tradeoff","Vertical bars show pointwise 95% power intervals. Sampling counts are not measured runtimes.")

    intro="Five matched LIME AIS configurations, each with 100 null trials and 100 power trials per beta. LIME uses 40 explanations; AIS uses a 51,200 final cap, 10 x 320 pilot draws, batch size 800, and ESS thresholds 80/15."
    report=["# AIS fix tests",intro,"## Plot gallery"]
    sections=[]
    for name,caption in FIGURES:
        title=name[3:].replace("_"," ").title(); report.append(f"### {title}\n\n{caption}\n\n![{name}]({name}.png)\n\n[PDF]({name}.pdf)"); sections.append(f'<section><h2>{html.escape(title)}</h2><p>{html.escape(caption)}</p><a href="{name}.png"><img src="{name}.png" alt="{html.escape(caption)}"></a><p><a href="{name}.pdf">Download PDF</a></p></section>')
    report.append("## Data\n\nPlots are derived from `all_target_diagnostics.csv`. Summary tables were regenerated from those 2,000 target-level records. Total AIS evaluations are used as sampling cost; no wall-clock timing is inferred.")
    (OUT/"README.md").write_text("\n\n".join(report),encoding="utf-8")
    (OUT/"index.html").write_text('<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1"><title>AIS fix tests</title><style>body{font:17px/1.6 system-ui;max-width:1200px;margin:32px auto;padding:0 24px;color:#203040;background:#f5f7fa}section{background:white;padding:24px;margin:24px 0;border-radius:12px}img{width:100%;height:auto}a{color:#17669b}</style><h1>AIS fix tests</h1><p>'+html.escape(intro)+'</p>'+''.join(sections)+'</html>',encoding="utf-8")
    print(f"Saved {len(FIGURES)} PNG/PDF figures and summaries to {OUT}")


if __name__ == "__main__":
    main()
