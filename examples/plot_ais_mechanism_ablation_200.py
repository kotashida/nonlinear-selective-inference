"""Audit and plot the paired exact-set AIS mechanism ablation."""

from __future__ import annotations

import html
import json
import os
from pathlib import Path

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["POLARS_MAX_THREADS"] = "1"

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
import numpy as np
import pandas as pd

from analyze_ais_method_comparison import _wilson_interval
from analyze_ais_mechanism_ablation_200 import validate_matched_scenarios


ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "outputs/selection_method_validation/exact_set"
REFERENCE = BASE / "ais_fix_followup_200" / "baseline_control"
OUT = BASE / "ais_mechanism_ablation_200"
CONDITIONS = {
    "baseline_control": "Baseline",
    "a_multimodal_retention": "A: multimodal retention",
    "b_denominator_targeting": "B: denominator targeting",
    "c_stratified_exploration": "C: stratified exploration",
    "ab_multimodal_denominator": "A + B",
    "ac_multimodal_stratified": "A + C",
    "bc_denominator_stratified": "B + C",
    "abc_all_mechanisms": "A + B + C",
}
STRATEGIES = {
    "baseline_control": "baseline",
    "a_multimodal_retention": "ablation_a",
    "b_denominator_targeting": "ablation_b",
    "c_stratified_exploration": "ablation_c",
    "ab_multimodal_denominator": "ablation_ab",
    "ac_multimodal_stratified": "ablation_ac",
    "bc_denominator_stratified": "ablation_bc",
    "abc_all_mechanisms": "ablation_abc",
}
COLORS = dict(
    zip(
        CONDITIONS,
        [
            "#2674a6",
            "#d58122",
            "#258568",
            "#9256a0",
            "#bb4f70",
            "#6f8f2f",
            "#4f7f91",
            "#7b6045",
        ],
    )
)
N = 200
FIGURES = []


def as_bool(values):
    if pd.api.types.is_bool_dtype(values):
        return values.fillna(False)
    return values.fillna(False).astype(str).str.lower().eq("true")


def read_config(path):
    result = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        key, separator, value = line.partition("=")
        if separator:
            result[key] = value
    return result


def without_strategy(settings):
    return {key: value for key, value in settings.items() if key != "ais_strategy"}


def audit_settings():
    """Prove that the seven new arms differ only in AIS strategy."""
    reference_config = read_config(REFERENCE / "run_config.txt")
    reference_strategy = reference_config.pop("AIS_STRATEGY")
    assert reference_strategy == "baseline"
    rows = []
    for condition in tuple(CONDITIONS)[1:]:
        arm = OUT / condition
        config = read_config(arm / "run_config.txt")
        strategy = config.pop("AIS_STRATEGY")
        assert strategy == STRATEGIES[condition]
        assert config == reference_config, condition
        rows.append(
            {
                "condition": condition,
                "scope": "run_config",
                "shard": np.nan,
                "settings_match_except_ais_strategy": True,
                "reference_strategy": "baseline",
                "arm_strategy": strategy,
            }
        )
        for shard in range(48):
            reference_shard = REFERENCE / f"shard_{shard}"
            arm_shard = arm / f"shard_{shard}"
            reference_validation = json.loads(
                (reference_shard / "validation_settings.json").read_text(
                    encoding="utf-8"
                )
            )
            arm_validation = json.loads(
                (arm_shard / "validation_settings.json").read_text(encoding="utf-8")
            )
            assert arm_validation["ais_strategy"] == STRATEGIES[condition]
            assert without_strategy(arm_validation) == without_strategy(
                reference_validation
            ), (condition, shard, "validation")
            rows.append(
                {
                    "condition": condition,
                    "scope": "validation",
                    "shard": shard,
                    "settings_match_except_ais_strategy": True,
                    "reference_strategy": reference_validation["ais_strategy"],
                    "arm_strategy": arm_validation["ais_strategy"],
                }
            )
            for scope, relative in (
                ("null", Path("lime/baseline/null_fresh/settings.json")),
                (
                    "power",
                    Path("lime/baseline/power_feature_0_beta_1p0/settings.json"),
                ),
            ):
                reference_settings = json.loads(
                    (reference_shard / relative).read_text(encoding="utf-8")
                )
                arm_settings = json.loads(
                    (arm_shard / relative).read_text(encoding="utf-8")
                )
                assert arm_settings["ais_strategy"] == STRATEGIES[condition]
                assert without_strategy(arm_settings) == without_strategy(
                    reference_settings
                ), (condition, shard, scope)
                rows.append(
                    {
                        "condition": condition,
                        "scope": scope,
                        "shard": shard,
                        "settings_match_except_ais_strategy": True,
                        "reference_strategy": reference_settings["ais_strategy"],
                        "arm_strategy": arm_settings["ais_strategy"],
                    }
                )
    audit = pd.DataFrame(rows)
    audit.to_csv(OUT / "configuration_audit.csv", index=False)
    return audit


def save(fig, name, caption):
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    for extension in ("png", "pdf"):
        fig.savefig(OUT / f"{name}.{extension}", dpi=180, bbox_inches="tight")
    plt.close(fig)
    FIGURES.append((name, caption))


def power_summary(power):
    rows = []
    for condition in CONDITIONS:
        group = power.loc[power["condition"].eq(condition)]
        signal = group.loc[group["is_signal"]]
        detections = int((signal["p_value"].lt(0.05) & ~signal["failed"]).sum())
        lower, upper = _wilson_interval(detections, len(signal))
        rows.append(
            {
                "condition": condition,
                "beta": 1.0,
                "trials": len(group),
                "signal_targets": len(signal),
                "detections": detections,
                "signal_failures": int(signal["failed"].sum()),
                "target_probability": len(signal) / len(group),
                "conditional_power": detections / len(signal),
                "ci_lower": lower,
                "ci_upper": upper,
                "marginal_power": detections / len(group),
                "mean_aj_evaluations": group["aj_evaluations"].mean(),
            }
        )
    return pd.DataFrame(rows)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update(
        {
            "font.size": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )
    audit = audit_settings()
    data = pd.read_csv(OUT / "all_target_diagnostics.csv", keep_default_na=False)
    data = data.loc[data["condition"].isin(CONDITIONS)].copy()
    # The carried analyzer read the literal label "null" with pandas' default
    # NA parsing, so historical null scopes were serialized as empty strings.
    data["scope"] = data["scope"].replace("", "null")
    data["failed"] = as_bool(data["failed"])
    data["is_signal"] = as_bool(data["is_signal"])
    for column in (
        "p_value",
        "aj_evaluations",
        "proposals",
        "denominator_ess",
        "tail_ess",
        "t_obs",
        "auxiliary_u",
    ):
        data[column] = pd.to_numeric(data[column], errors="coerce")
    assert len(data) == len(CONDITIONS) * 2 * N
    assert set(data["condition"]) == set(CONDITIONS)
    counts = data.groupby(["condition", "scope"]).size().unstack()
    assert (counts[["null", "power"]] == N).all().all()
    validate_matched_scenarios(data)
    data.to_csv(OUT / "plotted_target_diagnostics.csv", index=False)

    null = data.loc[data["scope"].eq("null")].copy()
    power = data.loc[data["scope"].eq("power")].copy()
    assert not null["failed"].any()
    summary = power_summary(power)
    summary.to_csv(OUT / "power_summary.csv", index=False)

    # 1. The reference three-panel power decomposition, adapted to beta=1.
    metrics = (
        ("target_probability", "Probability of targeting the signal"),
        ("conditional_power", "Conditional power"),
        ("marginal_power", "Marginal detection power"),
    )
    fig, axes = plt.subplots(1, 3, figsize=(16, 7), sharey=True)
    y = np.arange(len(CONDITIONS))
    for ax, (metric, title) in zip(axes, metrics):
        values = summary.set_index("condition").loc[list(CONDITIONS), metric]
        ax.barh(y, values, color=[COLORS[item] for item in CONDITIONS])
        for index, value in enumerate(values):
            ax.text(value + 0.012, index, f"{value:.1%}", va="center")
        ax.set(title=title, xlabel="Rate", xlim=(0, 1.08), yticks=y)
        ax.xaxis.set_major_formatter(PercentFormatter(1))
        ax.grid(axis="x", alpha=0.2)
    axes[0].set_yticklabels(CONDITIONS.values())
    axes[0].invert_yaxis()
    axes[0].axvline(0.5, color="gray", linestyle="--")
    axes[2].axvline(0.5, color="gray", linestyle="--")
    fig.suptitle("AIS mechanism ablation at beta=1: power and signal targeting")
    save(
        fig,
        "01_power_curves",
        "The reference power decomposition is retained for the single beta=1 condition. Failed signal tests count as misses; marginal power also accounts for signal targeting.",
    )

    # 2. The same observed-power and lower-bound heatmaps.
    fig, axes = plt.subplots(1, 2, figsize=(12, 7))
    for ax, metric, title in zip(
        axes,
        ("conditional_power", "ci_lower"),
        ("Observed conditional power", "95% lower confidence bound"),
    ):
        values = summary.set_index("condition").loc[list(CONDITIONS), metric]
        ax.imshow(values.to_numpy()[:, None], vmin=0, vmax=1, cmap="YlGnBu", aspect="auto")
        for index, condition in enumerate(CONDITIONS):
            row = summary.loc[summary["condition"].eq(condition)].iloc[0]
            value = row[metric]
            extra = (
                f"\n{int(row.detections)}/{int(row.signal_targets)}"
                if metric == "conditional_power"
                else ("\n>= 80%" if value >= 0.8 else "")
            )
            ax.text(
                0,
                index,
                f"{value:.1%}{extra}",
                ha="center",
                va="center",
                color="white" if value > 0.65 else "black",
            )
        ax.set(
            title=title,
            xticks=[0],
            xticklabels=["1.00"],
            yticks=range(len(CONDITIONS)),
            yticklabels=CONDITIONS.values(),
            xlabel="Signal strength (beta)",
        )
    fig.suptitle("AIS mechanism ablation: the 80% power requirement")
    save(
        fig,
        "02_power_target_heatmap",
        "Counts show signal detections / signal targets, with failures retained in the denominator. Confidence bounds are pointwise.",
    )

    # 3. Null calibration.
    calibration = []
    for condition in CONDITIONS:
        group = null.loc[null["condition"].eq(condition)]
        for alpha in (0.01, 0.05, 0.10):
            rejections = int(group["p_value"].lt(alpha).sum())
            lower, upper = _wilson_interval(rejections, len(group))
            calibration.append(
                {
                    "condition": condition,
                    "alpha": alpha,
                    "rejections": rejections,
                    "trials": len(group),
                    "fpr": rejections / len(group),
                    "ci_lower": lower,
                    "ci_upper": upper,
                }
            )
    calibration = pd.DataFrame(calibration)
    calibration.to_csv(OUT / "calibration_summary.csv", index=False)
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    offsets = np.linspace(-0.004, 0.004, len(CONDITIONS))
    for offset, (condition, label) in zip(offsets, CONDITIONS.items()):
        group = calibration.loc[calibration["condition"].eq(condition)].sort_values("alpha")
        axes[0].errorbar(
            group["alpha"] + offset,
            group["fpr"],
            yerr=np.maximum(
                0,
                np.vstack(
                    [group["fpr"] - group["ci_lower"], group["ci_upper"] - group["fpr"]]
                ),
            ),
            fmt="o",
            capsize=2,
            color=COLORS[condition],
            label=label,
        )
        values = np.sort(null.loc[null["condition"].eq(condition), "p_value"])
        axes[1].step(
            values,
            np.arange(1, len(values) + 1) / len(values),
            where="post",
            color=COLORS[condition],
            label=label,
        )
    axes[0].plot([0, 0.12], [0, 0.12], "--", color="gray")
    axes[0].set(
        title="Null rejection rates and 95% intervals",
        xlabel="Nominal alpha (points offset)",
        ylabel="Observed false-positive rate",
        xticks=[0.01, 0.05, 0.10],
        ylim=(0, 0.18),
    )
    axes[0].legend(fontsize=7, ncol=2)
    axes[1].plot([0, 1], [0, 1], "--", color="gray")
    axes[1].set(
        title="Null p-value empirical distribution",
        xlabel="p-value",
        ylabel="Fraction at or below p",
        xlim=(0, 1),
        ylim=(0, 1),
    )
    for ax in axes:
        ax.grid(alpha=0.2)
    fig.suptitle("AIS mechanism ablation: null calibration (200 trials per arm)")
    save(
        fig,
        "03_null_calibration",
        "The diagonal is the nominal/uniform reference. Small departures in 200 trials do not establish calibration or miscalibration.",
    )

    # 4. Signal outcomes.
    fig, axes = plt.subplots(2, 4, figsize=(16, 9), squeeze=False)
    outcomes = []
    for ax, (condition, label) in zip(axes.flat, CONDITIONS.items()):
        row = summary.loc[summary["condition"].eq(condition)].iloc[0]
        detected = row["detections"] / row["signal_targets"]
        failed = row["signal_failures"] / row["signal_targets"]
        missed = 1.0 - detected - failed
        ax.bar([0], [detected], color="#258568", label="Detected")
        ax.bar([0], [missed], bottom=[detected], color="#bcc8d1", label="Valid, not detected")
        ax.bar([0], [failed], bottom=[detected + missed], color="#bc4545", label="Inference failed")
        ax.text(
            0,
            1.02,
            f"n={int(row.signal_targets)}\nfailed={int(row.signal_failures)}",
            ha="center",
            fontsize=9,
        )
        ax.set(title=label, xticks=[0], xticklabels=["1.00"], ylim=(0, 1.18), xlabel="Signal strength (beta)")
        ax.yaxis.set_major_formatter(PercentFormatter(1))
        outcomes.append(
            {
                "condition": condition,
                "beta": 1.0,
                "signal_targets": int(row["signal_targets"]),
                "detected": int(row["detections"]),
                "valid_not_detected": int(row["signal_targets"] - row["detections"] - row["signal_failures"]),
                "failed": int(row["signal_failures"]),
            }
        )
    axes[0, 0].legend(fontsize=8, loc="lower right")
    fig.suptitle("AIS mechanism ablation: outcomes among signal targets")
    save(
        fig,
        "04_signal_outcomes",
        "Separates statistical non-detection from unresolved inference. Each bar is conditional on the target being a true signal.",
    )
    pd.DataFrame(outcomes).to_csv(OUT / "signal_outcomes.csv", index=False)

    # 5. Sampling cost.
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    costs = []
    means = []
    for condition, label in CONDITIONS.items():
        raw = power.loc[power["condition"].eq(condition)]
        means.append(raw["aj_evaluations"].mean())
        values = np.sort(raw["aj_evaluations"])
        axes[1].step(
            values,
            np.arange(1, len(values) + 1) / len(values),
            where="post",
            color=COLORS[condition],
            label=label,
        )
        costs.append(
            {
                "condition": condition,
                "beta": 1.0,
                "mean": raw["aj_evaluations"].mean(),
                "median": raw["aj_evaluations"].median(),
                "p95": raw["aj_evaluations"].quantile(0.95),
                "maximum": raw["aj_evaluations"].max(),
                "first_batch_fraction": raw["proposals"].eq(800).mean(),
                "cap_fraction": raw["proposals"].eq(51200).mean(),
            }
        )
    axes[0].barh(y, means, color=[COLORS[item] for item in CONDITIONS])
    axes[0].set(title="Mean cost at beta=1", xlabel="Total AIS evaluations", yticks=y, yticklabels=CONDITIONS.values())
    axes[0].invert_yaxis()
    axes[1].set(
        title="Cost distribution across 200 power trials per arm",
        xlabel="Total AIS evaluations",
        ylabel="Fraction at or below cost",
        ylim=(0, 1.03),
    )
    axes[1].legend(fontsize=7, ncol=2)
    for ax in axes:
        ax.grid(alpha=0.2)
    fig.suptitle("AIS mechanism ablation: sampling effort")
    save(
        fig,
        "05_sampling_cost",
        "Total AIS evaluations include pilot and final proposals; counts are not wall-clock time.",
    )
    pd.DataFrame(costs).to_csv(OUT / "sampling_cost_summary.csv", index=False)

    # 6. ESS diagnostics.
    fig, axes = plt.subplots(2, 4, figsize=(16, 9), squeeze=False)
    for ax, (condition, label) in zip(axes.flat, CONDITIONS.items()):
        group = power.loc[power["condition"].eq(condition)]
        for failed, marker, color in (
            (False, "o", COLORS[condition]),
            (True, "x", "#bf3232"),
        ):
            selected = group.loc[
                group["failed"].eq(failed)
                & group["denominator_ess"].gt(0)
                & group["tail_ess"].gt(0)
            ]
            ax.scatter(
                selected["denominator_ess"],
                selected["tail_ess"],
                s=15 if not failed else 40,
                alpha=0.3 if not failed else 0.9,
                marker=marker,
                color=color,
                label="Failed" if failed else "Successful",
            )
        ax.axvline(80, linestyle="--", color="gray")
        ax.axhline(15, linestyle="--", color="gray")
        ax.set(
            title=f"{label}: {int(group.failed.sum())}/200 failures",
            xlabel="Denominator ESS",
            ylabel="Tail ESS",
            xscale="log",
            yscale="log",
        )
        ax.grid(alpha=0.15)
    axes[0, 0].legend(fontsize=8)
    fig.suptitle("AIS mechanism ablation: effective sample sizes")
    save(
        fig,
        "06_ess_diagnostics",
        "Dashed lines mark denominator ESS=80 and tail ESS=15. Failed inference is red; both axes are logarithmic.",
    )

    # 7. Power-cost tradeoff.
    fig, ax = plt.subplots(figsize=(10, 6))
    annotation_offsets = {
        "baseline_control": (8, 2, "left"),
        "a_multimodal_retention": (16, -18, "left"),
        "b_denominator_targeting": (8, 0, "left"),
        "c_stratified_exploration": (8, 0, "left"),
        "ab_multimodal_denominator": (16, 14, "left"),
        "ac_multimodal_stratified": (0, 18, "center"),
        "bc_denominator_stratified": (8, 0, "left"),
        "abc_all_mechanisms": (-12, -14, "right"),
    }
    for condition, label in CONDITIONS.items():
        row = summary.loc[summary["condition"].eq(condition)].iloc[0]
        ax.errorbar(
            row["mean_aj_evaluations"],
            row["conditional_power"],
            yerr=[
                [row["conditional_power"] - row["ci_lower"]],
                [row["ci_upper"] - row["conditional_power"]],
            ],
            fmt="o",
            capsize=4,
            color=COLORS[condition],
            label=label,
        )
        x_offset, y_offset, alignment = annotation_offsets[condition]
        ax.annotate(
            f"{row.conditional_power:.1%}",
            (row["mean_aj_evaluations"], row["conditional_power"]),
            xytext=(x_offset, y_offset),
            textcoords="offset points",
            color=COLORS[condition],
            ha=alignment,
            va="center",
        )
    ax.axhline(0.8, linestyle="--", color="gray")
    ax.set(
        xlabel="Mean total AIS evaluations per power trial",
        ylabel="Conditional power",
        ylim=(0.75, 1.02),
    )
    ax.yaxis.set_major_formatter(PercentFormatter(1))
    ax.legend(fontsize=8, ncol=2, loc="lower left")
    ax.grid(alpha=0.2)
    fig.suptitle("AIS mechanism ablation at beta=1: power versus sampling effort")
    save(
        fig,
        "07_power_cost_tradeoff",
        "Vertical bars show pointwise 95% power intervals. Sampling counts are not measured runtimes.",
    )

    total_failures = power.groupby("condition")["failed"].sum()
    notes = "Power-trial failures: " + ", ".join(
        f"{CONDITIONS[condition]} {int(total_failures[condition])}/200"
        for condition in CONDITIONS
    ) + ". Failed signal tests count as misses; intervals are pointwise."
    intro = (
        "Eight paired LIME AIS arms: the historical baseline and seven A/B/C "
        "mechanism combinations. Every arm uses exact-set conditioning, 200 null "
        "and 200 beta=1 power trials, the same scenario seeds, 10 x 320 pilot "
        "evaluations, a 51,200 final cap, batch size 800, and ESS thresholds 80/15. "
        "The configuration audit confirms that AIS strategy is the only settings difference."
    )
    report = ["# AIS mechanism ablation (200 paired trials)", intro, notes, "## Plot gallery"]
    sections = []
    for name, caption in FIGURES:
        title = name[3:].replace("_", " ").title()
        report.append(
            f"### {title}\n\n{caption}\n\n![{name}]({name}.png)\n\n[PDF]({name}.pdf)"
        )
        sections.append(
            f'<section><h2>{html.escape(title)}</h2><p>{html.escape(caption)}</p>'
            f'<a href="{name}.png"><img src="{name}.png" alt="{html.escape(caption)}"></a>'
            f'<p><a href="{name}.pdf">Download PDF</a></p></section>'
        )
    report.append(
        "## Data\n\n`plotted_target_diagnostics.csv` contains the 3,200 plotted "
        "target-level records. `configuration_audit.csv` contains "
        f"{len(audit):,} successful cross-run settings comparisons. The historical "
        "reallocated-pilot arm remains in `all_target_diagnostics.csv` but is excluded "
        "from these requested baseline-plus-seven plots."
    )
    (OUT / "README.md").write_text("\n\n".join(report), encoding="utf-8")
    page = (
        '<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" '
        'content="width=device-width, initial-scale=1"><title>AIS mechanism ablation</title>'
        '<style>body{font:17px/1.6 system-ui;max-width:1200px;margin:32px auto;padding:0 24px;color:#203040;background:#f5f7fa}'
        'section{background:white;padding:24px;margin:24px 0;border-radius:12px}img{width:100%;height:auto}a{color:#17669b}</style>'
        f"<h1>AIS mechanism ablation</h1><p>{html.escape(intro)}</p><p>{html.escape(notes)}</p>"
        + "".join(sections)
        + "</html>"
    )
    (OUT / "index.html").write_text(page, encoding="utf-8")
    print(
        f"Verified {len(audit):,} settings comparisons and saved "
        f"{len(FIGURES)} PNG/PDF figures to {OUT}."
    )


if __name__ == "__main__":
    main()
