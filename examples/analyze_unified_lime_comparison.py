"""Audit and compare downloaded adaptive AIS benchmarks and LIME variants."""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from analyze_ais_method_comparison import _read_raw, _wilson_interval

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "outputs/selection_method_validation/exact_set"
OUT = BASE / "lime_parameter_tuning/analysis/initial_comparison"
LABELS = {"lime": "LIME baseline", "shap": "SHAP",
          "spline_screening": "Spline AIS", "mutual_information": "MI (k=12)"}
EXTRAS = {"ridge_30": "LIME ridge 30", "kernel_1p5_sqrt_p": "LIME wider kernel",
          "explanations_40": "LIME 40 explanations"}
ORDER = ["LIME baseline", *EXTRAS.values(), "SHAP", "Spline AIS", "MI (k=12)"]
COLORS = dict(zip(ORDER, ["#555555", "#c44e52", "#dd8452", "#8172b3", "#55a868", "#4c72b0", "#937860"]))
COMMON = ["alpha_levels", "comparison_events", "design_seed", "designs", "final_batch_size",
          "fixed_auxiliary_values", "inference_method", "max_final_samples", "min_denominator_ess",
          "min_tail_ess", "pilot_iters", "pilot_samples", "rf_jobs", "shard_root_seeds",
          "signal_positions", "signal_strengths", "stop_when_ess_met", "runtime_metadata"]


def audit_low_level(reference, candidate):
    # Check actual simulation/inference configuration, allowing only selector differences.
    ignored = {"n_iters", "selection_method", "selector_settings", "rf_params"}
    for key in reference.keys() | candidate.keys():
        if key not in ignored and reference.get(key) != candidate.get(key):
            raise ValueError(f"Simulation/inference mismatch: {key}")


def load(extra_runs=None):
    original = BASE / "ais_comparison/baseline_four_methods_lime10/ais_all_methods"
    extra = BASE / "lime_parameter_tuning/initial_400_trials"
    specs = [(original, None)] + (extra_runs if extra_runs is not None else [(extra / variant, label) for variant, label in EXTRAS.items()])
    reference = json.loads((original / "pooled_summary/validation_settings.json").read_text())
    nulls, targets, features, manifest = [], [], [], []
    for run, label in specs:
        settings = json.loads((run / "pooled_summary/validation_settings.json").read_text())
        for key in COMMON:
            if settings[key] != reference[key]:
                raise ValueError(f"Mismatch in {run.name}: {key}")
        for shard in range(32):
            for folder in ["null_fresh", *[f"power_feature_0_beta_{str(b).replace('.', 'p')}" for b in reference["signal_strengths"]]]:
                suffix = Path(f"shard_{shard}/lime/baseline/{folder}/settings.json")
                old = json.loads((original / suffix).read_text())
                new = json.loads((run / suffix).read_text())
                audit_low_level(old, new)
                a, b = old["selector_settings"], new["selector_settings"]
                for key in a:
                    if key not in {"ridge_alpha", "kernel_width", "num_explanations"} and a[key] != b[key]:
                        raise ValueError(f"Unexpected LIME change: {key}")
        n, t, f = _read_raw(run, "ais", settings, 32)
        for frame in (n, t, f):
            frame["option"] = label if label else frame.selection_method.map(LABELS)
            frame["source_run"] = str(run.relative_to(BASE))
            frame.drop(columns="analysis_label", inplace=True)
        assert len(n) == settings["n_null_iters"] * len(settings["methods"])
        assert len(t) == len(f) == settings["n_power_iters"] * 4 * len(settings["methods"])
        nulls.append(n); targets.append(t); features.append(f)
        manifest.append({"source": str(run), "settings": settings})
        print(f"Loaded {run.name}: {len(n)} null / {len(t)} power", flush=True)
    n, t, f = [pd.concat(frames, ignore_index=True) for frames in (nulls, targets, features)]
    # The original 200 trials are a per-shard prefix of each 400-trial run.
    for frame, extra_keys in [(n, []), (t, ["signal_strength"])]:
        keys = ["validation_shard", "iteration", "selection_event", *extra_keys]
        baseline = frame[frame.option.eq("LIME baseline")]
        assert not frame.duplicated(["option", *keys]).any()
        reference_keys = pd.MultiIndex.from_frame(baseline[keys])
        frame["matched_200"] = pd.MultiIndex.from_frame(frame[keys]).isin(reference_keys)
        for option in ORDER:
            other = frame[frame.option.eq(option) & frame.matched_200]
            pair = baseline.merge(other, on=keys, suffixes=("_baseline", "_option"), validate="one_to_one")
            assert len(pair) == len(baseline)
            for key in ["target_seed", "auxiliary_u"]:
                assert np.array_equal(pair[key + "_baseline"], pair[key + "_option"]), (option, key)
    keys = ["option", "validation_shard", "iteration", "signal_strength", "selection_event"]
    f = f.merge(t[keys + ["matched_200", "target_seed", "successful_detection"]], on=keys, validate="one_to_one")
    # Verify published pooled rates against raw data rather than trusting stale summaries.
    for run, label in specs:
        c = pd.read_csv(run / "pooled_summary/calibration_decisions.csv")
        p = pd.read_csv(run / "pooled_summary/power_decisions.csv")
        for row in c.itertuples():
            group = n[n.option.eq(label or LABELS[row.selection_method])]
            assert len(group) == row.n_iterations
            assert group.failed.sum() == row.n_failed
            assert group.p_value.lt(row.alpha).sum() == row.n_rejected
        for row in p.itertuples():
            group = t[t.option.eq(label or LABELS[row.selection_method]) & t.signal_strength.eq(row.signal_strength)]
            assert len(group) == row.n_iterations
            assert group.target_is_signal.sum() == row.n_signal_targets
            assert np.isclose(group.successful_detection.mean(), row.marginal_detection_power)
    return n, t, f, manifest


def interval(k, count):
    if not count:
        return np.nan, np.nan
    return tuple(float(x) for x in _wilson_interval(k, count))


def summarize(n, t, f):
    calibration, power = [], []
    for view in ["all_available", "matched_200"]:
        nn, tt, ff = [(x if view == "all_available" else x[x.matched_200]) for x in (n, t, f)]
        for option, g in nn.groupby("option"):
            valid = ~g.failed & g.p_value.notna()
            for alpha in [0.01, 0.05, 0.1]:
                rejected = int((valid & g.p_value.lt(alpha)).sum())
                lo, hi = interval(rejected, valid.sum())
                calibration.append(dict(view=view, option=option, alpha=alpha, n=len(g),
                    failed=int((~valid).sum()), rejected=rejected, fpr=rejected / valid.sum(),
                    ci_lower=lo, ci_upper=hi, failure_rate=float((~valid).mean())))
        for (option, beta), g in tt.groupby(["option", "signal_strength"]):
            sig = g[g.target_is_signal]
            valid_sig = sig[~sig.failed]
            detections = int(g.successful_detection.sum())
            diag = ff[ff.option.eq(option) & ff.signal_strength.eq(beta)]
            lo, hi = interval(detections, len(sig))
            ml, mh = interval(detections, len(g))
            power.append(dict(view=view, option=option, beta=beta, n=len(g),
                signal_targets=len(sig), failed_signal=int(sig.failed.sum()),
                signal_failure_rate=float(sig.failed.mean()), target_rate=len(sig) / len(g),
                conditional_complete_case=detections / len(valid_sig) if len(valid_sig) else np.nan,
                conditional_failures_as_misses=detections / len(sig) if len(sig) else np.nan,
                conditional_lower=lo, conditional_upper=hi,
                marginal_power=detections / len(g), marginal_lower=ml, marginal_upper=mh,
                mean_proposals=diag.proposals.mean(), cap_rate=diag.proposals.eq(12800).mean(),
                n_failed=int(diag.failed.sum()), denominator_fail=int(diag.denominator_ess.lt(80).sum()),
                tail_fail=int(diag.tail_ess.lt(15).sum())))
    return pd.DataFrame(calibration), pd.DataFrame(power)


def save(fig, name):
    fig.tight_layout(rect=(0, 0.09, 1, 0.96))
    for ext in ["png", "pdf"]:
        fig.savefig(OUT / f"{name}.{ext}", dpi=180, bbox_inches="tight")
    plt.close(fig)


def plots(c, p):
    plt.rcParams.update({"font.size": 10})
    matched = p[p.view.eq("matched_200")]
    for stem, metrics in [
        ("matched_power_and_reliability", [("conditional_failures_as_misses", "Conditional power: failures count as misses"),
            ("marginal_power", "Marginal signal-detection power"), ("signal_failure_rate", "Signal-target inference failure rate"),
            ("target_rate", "Probability of targeting the signal")]),
        ("matched_sampling_effort", [("mean_proposals", "Mean final AIS proposals (pilot excluded)"),
            ("cap_rate", "Fraction reaching the 12,800 proposal cap")])]:
        fig, axes = plt.subplots(len(metrics)//2, 2, figsize=(13, 5 if len(metrics)==2 else 9), squeeze=False)
        for ax, (metric, title) in zip(axes.flat, metrics):
            for option in ORDER:
                g = matched[matched.option.eq(option)].sort_values("beta")
                ax.plot(g.beta, g[metric], marker="o", ms=4, color=COLORS[option],
                        linestyle="-" if option.startswith("LIME") else "--", label=option)
            ax.set(title=title, xlabel="Signal strength", xticks=[0.3, 0.5, 0.75, 1])
            if metric != "mean_proposals": ax.set_ylim(-0.02, 1.02)
            ax.grid(alpha=0.2)
        handles, labels = axes.flat[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="lower center", ncol=4, frameon=False)
        fig.suptitle("Identical AIS settings; matched 200 trials per configuration")
        save(fig, stem)
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for ax, view in zip(axes, ["matched_200", "all_available"]):
        for i, option in enumerate(ORDER):
            g = c[c.view.eq(view) & c.option.eq(option) & c.alpha.eq(0.05)].iloc[0]
            ax.errorbar(g.fpr, i, xerr=[[g.fpr-g.ci_lower], [g.ci_upper-g.fpr]], fmt="o", color=COLORS[option], capsize=3)
            ax.text(0.14, i, f"{int(g.failed)}/{int(g.n)} failed", va="center", fontsize=8)
        ax.axvline(.05, color="0.5", linestyle="--")
        ax.set(yticks=range(len(ORDER)), yticklabels=ORDER, xlim=(0, .20),
               xlabel="Null rejection rate among successful tests (95% Wilson CI)", title=view.replace("_", " "))
        ax.invert_yaxis(); ax.grid(axis="x", alpha=.2)
    save(fig, "null_calibration")
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for ax, beta in zip(axes, [.75, 1.]):
        for i, option in enumerate(ORDER):
            g = matched[matched.option.eq(option) & matched.beta.eq(beta)].iloc[0]
            v = g.conditional_failures_as_misses
            ax.errorbar(v, i, xerr=[[v-g.conditional_lower], [g.conditional_upper-v]], fmt="o", color=COLORS[option], capsize=3)
        ax.set(yticks=range(len(ORDER)), yticklabels=ORDER, xlim=(0.45, 1.02), title=f"Signal strength {beta:g}",
               xlabel="Conditional power, failures = misses (95% Wilson CI)")
        ax.invert_yaxis(); ax.grid(axis="x", alpha=.2)
    save(fig, "matched_power_intervals")
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    full = p[p.view.eq("all_available")]
    for ax, metric, title in zip(axes,
            ["conditional_failures_as_misses", "signal_failure_rate"],
            ["Conditional power: failures count as misses", "Signal-target inference failure rate"]):
        for option in ORDER:
            g = full[full.option.eq(option)].sort_values("beta")
            ax.plot(g.beta, g[metric], marker="o", ms=4, color=COLORS[option],
                    linestyle="-" if option.startswith("LIME") else "--",
                    label=f"{option} (n={int(g.n.iloc[0])})")
        ax.set(title=title, xlabel="Signal strength", xticks=[.3, .5, .75, 1], ylim=(-.02, 1.02))
        ax.grid(alpha=.2)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, frameon=False, fontsize=9)
    fig.suptitle("All available trials; unequal repetition counts shown in legend")
    save(fig, "all_available_power_and_reliability")


def table(frame, columns):
    lines = ["| " + " | ".join(columns) + " |", "| " + " | ".join(["---"]*len(columns)) + " |"]
    for row in frame[columns].itertuples(index=False, name=None):
        lines.append("| " + " | ".join(f"{x:.4f}" if isinstance(x, float) else str(x) for x in row) + " |")
    return "\n".join(lines)


def main():
    OUT.mkdir(exist_ok=True)
    n, t, f, manifest = load()
    n.to_csv(OUT / "null_tests.csv", index=False)
    t.to_csv(OUT / "power_targets.csv", index=False)
    f.to_csv(OUT / "power_features.csv", index=False)
    pd.concat([n.assign(scope="null"), f.assign(scope="power")], ignore_index=True).to_csv(OUT / "unified_tests.csv", index=False)
    c, p = summarize(n, t, f)
    c.to_csv(OUT / "calibration_summary.csv", index=False)
    p.to_csv(OUT / "power_and_diagnostics_summary.csv", index=False)
    (OUT / "source_manifest.json").write_text(json.dumps(manifest, indent=2))
    plots(c, p)
    report = ["# Unified LIME / adaptive AIS comparison", "",
        "All seven options use the same recorded AIS parameters, data-generating settings and software versions. Low-level settings were checked for all 32 shards and all five null/power configurations. Only LIME parameters and repetition counts differ between the LIME runs.",
        "The original methods have 200 trials per configuration; the three extra LIME variants have 400. The matched_200 view uses the original shard/iteration IDs, with exact equality of target seeds and auxiliary draws verified. Trial seeds generate the same data independently of iteration count. Features selected and targets can change with the selector.",
        "Available: baseline LIME, ridge 30, wider kernel (1.5 sqrt(20)), 40 explanations, SHAP, spline AIS and MI k=12. The first five-setting sweep is not present locally. Exact spline is not included because the adaptive reference failed; spline here means AIS.",
        "Rates are exploratory Monte Carlo estimates. Wilson intervals are pointwise, not simultaneous or paired difference intervals. Different selectors can target different trials, so conditional power is not a comparison on a fixed signal-target subset; marginal power includes that selection effect. Failures count as misses for the primary power metric. More explanations increase LIME prediction work per AIS proposal; proposal counts alone are not wall-clock runtime.", "",
        "## Null calibration at 5%", table(c[c.alpha.eq(.05)], ["view", "option", "n", "failed", "fpr", "ci_lower", "ci_upper"])]
    for view in ["matched_200", "all_available"]:
        report += [f"\n## Power: {view}", table(p[p.view.eq(view)], ["option", "beta", "n", "signal_targets", "failed_signal", "conditional_complete_case", "conditional_failures_as_misses", "marginal_power", "mean_proposals"])]
    report += ["## Findings",
        "On the matched 200-trial comparison, 40 explanation points is the most promising available LIME change at beta=1: failures among 104 signal targets fall from 35 to 19, power counting failures as misses rises from 66.35% to 79.81%, and marginal detection rises from 34.5% to 41.5%. Ridge 30 and a wider kernel achieve only 69.23% and 71.15% conditional power there. These are observed pilot differences, not a claim of statistical superiority.",
        "The benefit is not uniform: at beta=0.75, 40 explanations gives 85.15% versus baseline 84.16%, with the same 14 failed signal tests. At beta=0.5 it improves to 84.54% versus 78.13%. At beta=0.3 marginal detection is unchanged at 13.5%; conditional rates differ partly because different signal trials are targeted. Ridge 30 and wider-kernel results at beta=0.75 are slightly worse than baseline.",
        "Forty explanations still trails the matched strong-signal benchmarks: beta=1 conditional power is 85.58% for SHAP and MI, 96.15% for spline AIS. Signal-target failure rates are 18.27% for 40-explanation LIME, 12.5% SHAP, 8.65% MI, and 0% spline AIS. The remaining 19 failures for 40-explanation LIME all miss the denominator-ESS threshold.",
        "The full 400-trial results support the same qualitative ranking among the new variants at beta=1: 78.54% for 40 explanations versus 68.78% for ridge 30 and the wider kernel. These full-data estimates must not be treated as a paired 400-trial comparison against the original methods, which only have 200 trials.",
        "Calibration remains a qualification. At nominal 5%, 40 explanations gives 5.0% with 0/200 failures in the matched subset, and 6.02% with 1/400 failures in the full data (pointwise Wilson interval 4.08%-8.79%). This does not establish excess FPR, but it does not prove calibration either. Every new variant has one failed null test over 400 and therefore fails the pipeline's zero-failure requirement. The wider kernel's failure is already in the matched subset.",
        "Forty explanations uses fewer final AIS proposals at beta=1 (2,956 versus baseline 3,820), but evaluates four times as many explanation neighborhoods per proposal. This is not evidence of lower runtime. Confirm the chosen LIME setting on fresh seeds with the common AIS settings fixed; the missing first sweep may contain a better cost/performance choice."]
    (OUT / "analysis_report.md").write_text("\n\n".join(report), encoding="utf-8")
    (OUT / "README.md").write_text(
        "# Dataset guide\n\n"
        "unified_tests.csv contains 10,000 option-specific inference records: 2,000 null and 8,000 power tests. "
        "These are not 10,000 independent datasets; options share simulation seeds. "
        "Use scope, option, validation_shard, iteration, selection_event and (for power) signal_strength as the trial key. "
        "source_run identifies provenance; matched_200 selects trials common to all options. "
        "Do not pool different options or count the matched subset again as additional data.\n\n"
        "null_tests.csv, power_targets.csv and power_features.csv preserve the separate raw tables. "
        "calibration_summary.csv and power_and_diagnostics_summary.csv contain both all_available and matched_200 views. "
        "Their rows overlap and must not be added together. Conditional power's primary metric counts failed signal tests as misses. "
        "FPR uses only successful null tests; failures are reported separately. "
        "source_manifest.json records original settings. analysis_report.md explains findings and limitations. "
        "All PNG plots have matching vector PDF versions.\n", encoding="utf-8")
    print(p[p.view.eq("matched_200") & p.beta.eq(1)].to_string(index=False))
    print(f"Saved {len(n)} null tests + {len(t)} power tests to {OUT}")


if __name__ == "__main__":
    main()
