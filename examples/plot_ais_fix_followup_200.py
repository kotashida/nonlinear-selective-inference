"""Plot the three-setting, 200-iteration AIS follow-up report."""
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
OUT = ROOT / "outputs/selection_method_validation/exact_set/ais_fix_followup_200"
CONDITIONS = {
    "baseline_control": "Baseline control",
    "multimodal_qadapt": "Multimodal q-adaptation",
    "reallocated_pilot_exploration": "Reallocated pilot exploration",
}
COLORS = dict(zip(CONDITIONS, ["#2674a6", "#d58122", "#258568"]))
N = 200
FIGURES = []


def as_bool(s):
    return s.fillna(False).astype(str).str.lower().eq("true")


def save(fig, name, caption):
    fig.tight_layout(rect=(0, 0, 1, .93))
    for ext in ("png", "pdf"):
        fig.savefig(OUT / f"{name}.{ext}", dpi=180, bbox_inches="tight")
    plt.close(fig)
    FIGURES.append((name, caption))


def main():
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    # Preserve the literal scope label "null" instead of treating it as NA.
    data = pd.read_csv(OUT / "all_target_diagnostics.csv", keep_default_na=False)
    data["failed"] = as_bool(data.failed)
    data["is_signal"] = as_bool(data.is_signal)
    for column in ("p_value", "aj_evaluations", "proposals", "denominator_ess", "tail_ess"):
        data[column] = pd.to_numeric(data[column], errors="coerce")
    assert set(data.condition) == set(CONDITIONS) and len(data) == 3 * 2 * N
    counts = data.groupby(["condition", "scope"]).size().unstack()
    assert (counts[["null", "power"]] == N).all().all()
    n, t = data[data.scope.eq("null")].copy(), data[data.scope.eq("power")].copy()

    rows = []
    for condition, g in t.groupby("condition", sort=False):
        sig = g[g.is_signal]
        detected = int((sig.p_value.lt(.05) & ~sig.failed).sum())
        lo, hi = _wilson_interval(detected, len(sig))
        rows.append(dict(condition=condition, beta=1.0, trials=len(g), signal_targets=len(sig),
                         detections=detected, signal_failures=int(sig.failed.sum()),
                         target_probability=len(sig)/len(g), conditional_power=detected/len(sig),
                         ci_lower=lo, ci_upper=hi, marginal_power=detected/len(g),
                         mean_aj_evaluations=g.aj_evaluations.mean()))
    p = pd.DataFrame(rows).set_index("condition").reindex(CONDITIONS).reset_index()
    p.to_csv(OUT / "power_summary.csv", index=False)

    # 1. The same three power concepts as the reference, shown as grouped bars
    # because this follow-up contains only beta=1.
    metrics = [("target_probability", "Probability of targeting the signal"),
               ("conditional_power", "Conditional power"),
               ("marginal_power", "Marginal detection power")]
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    x = np.arange(len(CONDITIONS))
    for ax, (metric, title) in zip(axes, metrics):
        vals = p[metric].to_numpy()
        ax.bar(x, vals, color=[COLORS[c] for c in CONDITIONS])
        for i, v in enumerate(vals): ax.text(i, v + .025, f"{v:.1%}", ha="center")
        ax.set(title=title, xticks=x, xticklabels=list(CONDITIONS.values()), ylim=(0, 1.08))
        ax.tick_params(axis="x", rotation=18); ax.yaxis.set_major_formatter(PercentFormatter(1)); ax.grid(axis="y", alpha=.2)
    axes[0].axhline(.5, color="gray", ls="--"); axes[2].axhline(.5, color="gray", ls="--")
    fig.suptitle("AIS fix follow-up at beta=1: power and signal targeting")
    save(fig, "01_power_curves", "The reference power panels are retained as beta=1 grouped bars. Failed signal tests count as misses; marginal power also accounts for signal targeting.")

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, metric, title in zip(axes, ["conditional_power", "ci_lower"], ["Observed conditional power", "95% lower confidence bound"]):
        mat = p.set_index("condition")[[metric]].reindex(CONDITIONS)
        ax.imshow(mat, vmin=0, vmax=1, cmap="YlGnBu", aspect="auto")
        for i, condition in enumerate(CONDITIONS):
            r = p[p.condition.eq(condition)].iloc[0]; v = r[metric]
            extra = f"\n{int(r.detections)}/{int(r.signal_targets)}" if metric == "conditional_power" else ("\n>= 80%" if v >= .8 else "")
            ax.text(0, i, f"{v:.1%}{extra}", ha="center", va="center", color="white" if v > .65 else "black")
        ax.set(title=title, xticks=[0], xticklabels=["1.00"], yticks=range(3), yticklabels=list(CONDITIONS.values()), xlabel="Signal strength (beta)")
    fig.suptitle("AIS fix follow-up: the 80% power requirement")
    save(fig, "02_power_target_heatmap", "Counts show detections / signal targets, with failures retained in the denominator. Confidence bounds are pointwise.")

    cal = []
    for condition, g in n.groupby("condition", sort=False):
        for alpha in (.01, .05, .10):
            k = int((g.p_value < alpha).sum()); lo, hi = _wilson_interval(k, len(g))
            cal.append(dict(condition=condition, alpha=alpha, rejections=k, trials=len(g), fpr=k/len(g), ci_lower=lo, ci_upper=hi))
    c = pd.DataFrame(cal); c.to_csv(OUT / "calibration_summary.csv", index=False)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for j, (condition, label) in enumerate(CONDITIONS.items()):
        g = c[c.condition.eq(condition)].sort_values("alpha"); color = COLORS[condition]
        axes[0].errorbar(g.alpha+(j-1)*.0015, g.fpr, yerr=np.maximum(0, np.vstack([g.fpr-g.ci_lower, g.ci_upper-g.fpr])), fmt="o", capsize=3, color=color, label=label)
        vals = np.sort(n[n.condition.eq(condition)].p_value); axes[1].step(vals, np.arange(1, len(vals)+1)/len(vals), where="post", color=color, label=label)
    axes[0].plot([0,.12],[0,.12],"--",color="gray"); axes[0].set(title="Null rejection rates and 95% intervals", xlabel="Nominal alpha (points offset)", ylabel="Observed false-positive rate", xticks=[.01,.05,.1], ylim=(0,.18)); axes[0].legend(fontsize=8)
    axes[1].plot([0,1],[0,1],"--",color="gray"); axes[1].set(title="Null p-value empirical distribution", xlabel="p-value", ylabel="Fraction at or below p", xlim=(0,1), ylim=(0,1))
    for ax in axes: ax.grid(alpha=.2)
    fig.suptitle("AIS fix follow-up: null calibration (200 trials per setting)")
    save(fig, "03_null_calibration", "The diagonal is the nominal/uniform reference. Small departures in 200 trials do not establish calibration or miscalibration.")

    fig, axes = plt.subplots(1, 3, figsize=(15, 5)); outcomes=[]
    for ax, (condition, label) in zip(axes, CONDITIONS.items()):
        r=p[p.condition.eq(condition)].iloc[0]; det=r.detections/r.signal_targets; fail=r.signal_failures/r.signal_targets; miss=1-det-fail
        ax.bar([0],[det],color="#258568",label="Detected"); ax.bar([0],[miss],bottom=[det],color="#bcc8d1",label="Valid, not detected"); ax.bar([0],[fail],bottom=[det+miss],color="#bc4545",label="Inference failed")
        ax.text(0,1.02,f"n={int(r.signal_targets)}\nfailed={int(r.signal_failures)}",ha="center",fontsize=9)
        ax.set(title=label,xticks=[0],xticklabels=["1.00"],ylim=(0,1.18),xlabel="Signal strength (beta)"); ax.yaxis.set_major_formatter(PercentFormatter(1))
        outcomes.append(dict(condition=condition,beta=1.0,signal_targets=int(r.signal_targets),detected=int(r.detections),valid_not_detected=int(r.signal_targets-r.detections-r.signal_failures),failed=int(r.signal_failures)))
    axes[0].legend(fontsize=8,loc="lower right"); fig.suptitle("AIS fix follow-up: outcomes among signal targets")
    save(fig,"04_signal_outcomes","Separates statistical non-detection from unresolved inference; each bar is conditional on targeting the true signal.")
    pd.DataFrame(outcomes).to_csv(OUT/"signal_outcomes.csv",index=False)

    fig,axes=plt.subplots(1,2,figsize=(12,5)); costs=[]
    for condition,label in CONDITIONS.items():
        raw=t[t.condition.eq(condition)]; color=COLORS[condition]
        axes[0].bar(label,raw.aj_evaluations.mean(),color=color)
        vals=np.sort(raw.aj_evaluations); axes[1].step(vals,np.arange(1,len(vals)+1)/len(vals),where="post",color=color,label=label)
        costs.append(dict(condition=condition,beta=1.0,mean=raw.aj_evaluations.mean(),median=raw.aj_evaluations.median(),p95=raw.aj_evaluations.quantile(.95),maximum=raw.aj_evaluations.max(),first_batch_fraction=raw.proposals.eq(800).mean(),cap_fraction=raw.proposals.eq(51200).mean()))
    axes[0].set(title="Mean cost at beta=1",ylabel="Total AIS evaluations"); axes[0].tick_params(axis="x",rotation=18)
    axes[1].set(title="Cost distribution across 200 power trials per setting",xlabel="Total AIS evaluations",ylabel="Fraction at or below cost",ylim=(0,1.03)); axes[1].legend(fontsize=8)
    for ax in axes: ax.grid(alpha=.2)
    fig.suptitle("AIS fix follow-up: sampling effort"); save(fig,"05_sampling_cost","Total AIS evaluations include pilot and final proposals; counts are not wall-clock time.")
    pd.DataFrame(costs).to_csv(OUT/"sampling_cost_summary.csv",index=False)

    fig,axes=plt.subplots(1,3,figsize=(15,5))
    for ax,(condition,label) in zip(axes,CONDITIONS.items()):
        g=t[t.condition.eq(condition)]
        for failed,marker,color in [(False,"o",COLORS[condition]),(True,"x","#bf3232")]:
            q=g[g.failed.eq(failed)&g.denominator_ess.gt(0)&g.tail_ess.gt(0)]; ax.scatter(q.denominator_ess,q.tail_ess,s=15 if not failed else 40,alpha=.3 if not failed else .9,marker=marker,color=color,label="Failed" if failed else "Successful")
        ax.axvline(80,ls="--",color="gray"); ax.axhline(15,ls="--",color="gray"); ax.set(title=f"{label}: {int(g.failed.sum())}/200 failures",xlabel="Denominator ESS",ylabel="Tail ESS",xscale="log",yscale="log"); ax.grid(alpha=.15)
    axes[0].legend(fontsize=8); fig.suptitle("AIS fix follow-up: effective sample sizes")
    save(fig,"06_ess_diagnostics","Dashed lines mark denominator ESS=80 and tail ESS=15 thresholds. Failed inference is red; both axes are logarithmic.")

    fig,ax=plt.subplots(figsize=(9,5))
    for condition,label in CONDITIONS.items():
        r=p[p.condition.eq(condition)].iloc[0]; ax.errorbar(r.mean_aj_evaluations,r.conditional_power,yerr=[[r.conditional_power-r.ci_lower],[r.ci_upper-r.conditional_power]],fmt="o",capsize=4,color=COLORS[condition],label=label); ax.annotate(f"{r.conditional_power:.1%}",(r.mean_aj_evaluations,r.conditional_power),xytext=(7,0),textcoords="offset points",color=COLORS[condition])
    ax.axhline(.8,ls="--",color="gray"); ax.set(xlabel="Mean total AIS evaluations per power trial",ylabel="Conditional power"); ax.yaxis.set_major_formatter(PercentFormatter(1)); ax.legend(fontsize=8); ax.grid(alpha=.2)
    fig.suptitle("AIS fix follow-up at beta=1: power versus sampling effort"); save(fig,"07_power_cost_tradeoff","Vertical bars show pointwise 95% power intervals. Sampling counts are not measured runtimes.")

    intro="Three matched LIME AIS settings, each with 200 null and 200 beta=1 power trials. LIME uses 40 explanations; AIS uses a 51,200 final cap, 10 x 320 pilot draws, batch size 800, and ESS thresholds 80/15."
    report=["# AIS fix follow-up (200 iterations)",intro,"## Plot gallery"]; sections=[]
    for name,caption in FIGURES:
        title=name[3:].replace("_"," ").title(); report.append(f"### {title}\n\n{caption}\n\n![{name}]({name}.png)\n\n[PDF]({name}.pdf)"); sections.append(f'<section><h2>{html.escape(title)}</h2><p>{html.escape(caption)}</p><a href="{name}.png"><img src="{name}.png" alt="{html.escape(caption)}"></a><p><a href="{name}.pdf">Download PDF</a></p></section>')
    report.append("## Data\n\nPlots and regenerated summaries derive from `all_target_diagnostics.csv` (1,200 matched target-level records). Sampling cost is measured by total AIS evaluations; no wall-clock timing is inferred.")
    (OUT/"README.md").write_text("\n\n".join(report),encoding="utf-8")
    page = ('<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">'
            '<title>AIS fix follow-up</title><style>body{font:17px/1.6 system-ui;max-width:1200px;margin:32px auto;padding:0 24px;color:#203040;background:#f5f7fa}'
            'section{background:white;padding:24px;margin:24px 0;border-radius:12px}img{width:100%;height:auto}a{color:#17669b}</style>'
            '<h1>AIS fix follow-up (200 iterations)</h1><p>' + html.escape(intro) + '</p>' + ''.join(sections) + '</html>')
    (OUT/"index.html").write_text(page, encoding="utf-8")
    print(f"Saved {len(FIGURES)} PNG/PDF figures and summaries to {OUT}")


if __name__ == "__main__":
    main()
