"""Build unified datasets and diagnostic plots from validation shard snapshots.

Unlike the strict pooling command, this analyzer accepts an incomplete snapshot.
Every output carries explicit completeness information and uses observed row counts
as denominators. Raw experiment files are read only and never modified.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from itertools import combinations
from pathlib import Path

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["POLARS_MAX_THREADS"] = "1"

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages
from scipy import stats


METHOD_LABELS = {
    "shap": "SHAP",
    "spline_screening": "Spline screening",
    "mutual_information": "Mutual information",
}
METHOD_COLORS = {
    "shap": "#4C78A8",
    "spline_screening": "#F58518",
    "mutual_information": "#54A24B",
}
ALPHAS = (0.01, 0.05, 0.10)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_dir", type=Path)
    parser.add_argument("--output-dir", type=Path)
    return parser.parse_args(argv)


def read_json(path: Path) -> dict:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def proportion_interval(successes: int, total: int) -> tuple[float, float]:
    if total <= 0:
        return np.nan, np.nan
    return tuple(float(x) for x in stats.binomtest(successes, total).proportion_ci())


def ordered_shards(root: Path) -> list[Path]:
    def key(path: Path):
        suffix = path.name.removeprefix("shard_")
        return (0, int(suffix)) if suffix.isdigit() else (1, suffix)

    return sorted((p for p in root.glob("shard_*") if p.is_dir()), key=key)


def configuration_inventory(root: Path, validation: dict) -> pd.DataFrame:
    methods = validation["methods"]
    designs = validation["designs"]
    strengths = [float(x) for x in validation["signal_strengths"]]
    shards = ordered_shards(root)
    records = []
    for shard in shards:
        shard_index = int(shard.name.removeprefix("shard_"))
        for method in methods:
            for design in designs:
                configurations = [("null_calibration", np.nan, "null_fresh")]
                configurations += [
                    ("power", strength, f"power_feature_0_beta_{str(strength).replace('.', 'p')}")
                    for strength in strengths
                ]
                for analysis_type, strength, directory in configurations:
                    base = shard / method / design / directory
                    result_name = "p_value_results.csv" if analysis_type == "null_calibration" else "target_results.csv"
                    result_path = base / result_name
                    records.append(
                        {
                            "shard": shard.name,
                            "shard_index": shard_index,
                            "selection_method": method,
                            "design": design,
                            "analysis_type": analysis_type,
                            "signal_strength": strength,
                            "scenario_directory": directory,
                            "expected_result_file": result_name,
                            "present": result_path.is_file() and result_path.stat().st_size > 0,
                            "settings_present": (base / "settings.json").is_file(),
                            "source_path": result_path.relative_to(root).as_posix(),
                        }
                    )
    return pd.DataFrame.from_records(records)


def load_observations(root: Path, inventory: pd.DataFrame, validation: dict) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    per_shard_null = int(validation["n_null_iters"])
    per_shard_power = int(validation["n_power_iters"])
    method_inference = validation.get("method_inference_methods", {})
    for item in inventory[inventory["present"]].itertuples(index=False):
        source = root / item.source_path
        frame = pd.read_csv(source)
        if item.analysis_type == "power":
            feature_path = source.with_name("feature_results.csv")
            if feature_path.is_file() and feature_path.stat().st_size > 0:
                detail = pd.read_csv(feature_path)
                keys = ["iteration", "selection_event"]
                drop = [c for c in detail if c in frame.columns and c not in keys]
                detail = detail.drop(columns=drop)
                frame = frame.merge(detail, on=keys, how="left", validate="one_to_one")
            frame["feature"] = frame["target_feature"]
            frame["is_signal"] = frame["target_is_signal"]
        else:
            frame["target_is_signal"] = False
            frame["successful_detection"] = False
            frame["failed_signal_test"] = False
            frame["resolution_limited"] = frame.get("resolution_status", "").eq("resolution_limited")

        n_per_shard = per_shard_null if item.analysis_type == "null_calibration" else per_shard_power
        metadata = {
            "run_name": root.name,
            "shard": item.shard,
            "shard_index": item.shard_index,
            "global_iteration": item.shard_index * n_per_shard + frame["iteration"].astype(int),
            "selection_method": item.selection_method,
            "inference_method": method_inference.get(item.selection_method),
            "design": item.design,
            "analysis_type": item.analysis_type,
            "signal_strength": item.signal_strength,
            "signal_feature": 0 if item.analysis_type == "power" else np.nan,
            "scenario_directory": item.scenario_directory,
            "source_file": item.source_path,
        }
        for position, (name, value) in enumerate(metadata.items()):
            frame.insert(position, name, value)
        frames.append(frame)

    if not frames:
        raise FileNotFoundError("No non-empty result CSVs were found.")
    unified = pd.concat(frames, ignore_index=True, sort=False)
    for column in ("failed", "rejected", "target_is_signal", "successful_detection", "failed_signal_test", "resolution_limited"):
        if column in unified:
            unified[column] = unified[column].fillna(False).astype(bool)
    return unified


def summarize_configurations(observations: pd.DataFrame, inventory: pd.DataFrame) -> pd.DataFrame:
    rows = []
    keys = ["selection_method", "design", "analysis_type", "signal_strength", "selection_event"]
    for key, frame in observations.groupby(keys, dropna=False, sort=True):
        method, design, analysis_type, strength, event = key
        p = pd.to_numeric(frame["p_value"], errors="coerce")
        valid = p.notna() & ~frame["failed"]
        base = {
            "selection_method": method,
            "design": design,
            "analysis_type": analysis_type,
            "signal_strength": strength,
            "selection_event": event,
            "shards_present": int(frame["shard"].nunique()),
            "n_iterations": int(len(frame)),
            "n_valid_p_values": int(valid.sum()),
            "n_failed": int(frame["failed"].sum()),
            "failure_rate": float(frame["failed"].mean()),
            "mean_p_value": float(p[valid].mean()) if valid.any() else np.nan,
            "median_p_value": float(p[valid].median()) if valid.any() else np.nan,
            "mean_denominator_ess": float(pd.to_numeric(frame.get("denominator_ess"), errors="coerce").mean()),
            "mean_tail_ess": float(pd.to_numeric(frame.get("tail_ess"), errors="coerce").mean()),
            "mean_mc_se": float(pd.to_numeric(frame.get("mc_se"), errors="coerce").mean()),
            "mean_mcmc_hub_acceptance_rate": float(pd.to_numeric(frame.get("mcmc_hub_acceptance_rate"), errors="coerce").mean()),
            "mean_mcmc_replicate_acceptance_rate": float(pd.to_numeric(frame.get("mcmc_replicate_acceptance_rate"), errors="coerce").mean()),
        }
        if analysis_type == "null_calibration":
            base["uniform_ks_statistic"] = float(stats.kstest(p[valid], "uniform").statistic) if valid.any() else np.nan
            base["uniform_ks_p_value"] = float(stats.kstest(p[valid], "uniform").pvalue) if valid.any() else np.nan
            for alpha in ALPHAS:
                successes = int((p[valid] < alpha).sum())
                lower, upper = proportion_interval(successes, int(valid.sum()))
                suffix = f"{alpha:g}"
                base[f"rejections_alpha_{suffix}"] = successes
                base[f"rejection_rate_alpha_{suffix}"] = successes / int(valid.sum()) if valid.any() else np.nan
                base[f"rejection_ci_95_lower_alpha_{suffix}"] = lower
                base[f"rejection_ci_95_upper_alpha_{suffix}"] = upper
        else:
            signal = frame[frame["target_is_signal"] & ~frame["failed"]]
            marginal_success = int(frame["successful_detection"].sum())
            conditional_success = int(signal["rejected"].sum())
            marginal_ci = proportion_interval(marginal_success, len(frame))
            conditional_ci = proportion_interval(conditional_success, len(signal))
            selection_ci = proportion_interval(int(frame["target_is_signal"].sum()), len(frame))
            base.update(
                {
                    "n_signal_targets": int(frame["target_is_signal"].sum()),
                    "target_signal_rate": float(frame["target_is_signal"].mean()),
                    "target_signal_rate_ci_95_lower": selection_ci[0],
                    "target_signal_rate_ci_95_upper": selection_ci[1],
                    "marginal_detection_power": marginal_success / len(frame),
                    "marginal_power_ci_95_lower": marginal_ci[0],
                    "marginal_power_ci_95_upper": marginal_ci[1],
                    "conditional_power_given_signal_target": conditional_success / len(signal) if len(signal) else np.nan,
                    "conditional_power_ci_95_lower": conditional_ci[0],
                    "conditional_power_ci_95_upper": conditional_ci[1],
                    "resolution_limited_rate": float(frame["resolution_limited"].mean()),
                    "signal_target_failure_rate": float(frame.loc[frame["target_is_signal"], "failed"].mean()) if frame["target_is_signal"].any() else np.nan,
                }
            )
        rows.append(base)
    summary = pd.DataFrame.from_records(rows)
    expected = inventory.groupby(["selection_method", "design", "analysis_type", "signal_strength"], dropna=False)["present"].agg(["sum", "count"]).reset_index()
    expected = expected.rename(columns={"sum": "configuration_shards_present", "count": "configuration_shards_expected"})
    summary = summary.merge(expected, on=["selection_method", "design", "analysis_type", "signal_strength"], how="left")
    summary["configuration_complete"] = summary["configuration_shards_present"] == summary["configuration_shards_expected"]
    return summary


def summarize_shards(observations: pd.DataFrame) -> pd.DataFrame:
    rows = []
    keys = ["shard", "shard_index", "selection_method", "analysis_type", "signal_strength", "selection_event"]
    for key, frame in observations.groupby(keys, dropna=False, sort=True):
        shard, shard_index, method, analysis_type, strength, event = key
        valid = ~frame["failed"] & frame["p_value"].notna()
        row = {
            "shard": shard,
            "shard_index": shard_index,
            "selection_method": method,
            "analysis_type": analysis_type,
            "signal_strength": strength,
            "selection_event": event,
            "n_iterations": len(frame),
            "failure_rate": float(frame["failed"].mean()),
        }
        if analysis_type == "null_calibration":
            row["rejection_rate_alpha_0.05"] = float((frame.loc[valid, "p_value"] < 0.05).mean()) if valid.any() else np.nan
            row["uniform_ks_statistic"] = float(stats.kstest(frame.loc[valid, "p_value"], "uniform").statistic) if valid.any() else np.nan
        else:
            signal = frame[frame["target_is_signal"] & ~frame["failed"]]
            row["marginal_detection_power"] = float(frame["successful_detection"].mean())
            row["conditional_power_given_signal_target"] = float(signal["rejected"].mean()) if len(signal) else np.nan
            row["target_signal_rate"] = float(frame["target_is_signal"].mean())
        rows.append(row)
    return pd.DataFrame.from_records(rows)


def paired_method_comparisons(observations: pd.DataFrame) -> pd.DataFrame:
    rows = []
    pair_keys = ["shard_index", "iteration", "selection_event"]
    for analysis_type, strength_frame in observations.groupby("analysis_type", sort=True):
        strengths = [np.nan] if analysis_type == "null_calibration" else sorted(strength_frame["signal_strength"].dropna().unique())
        for strength in strengths:
            frame = strength_frame if math.isnan(strength) else strength_frame[strength_frame["signal_strength"] == strength]
            methods = sorted(frame["selection_method"].unique())
            for first, second in combinations(methods, 2):
                left = frame[frame["selection_method"] == first].copy()
                right = frame[frame["selection_method"] == second].copy()
                if analysis_type == "null_calibration":
                    outcomes = [(f"rejection_alpha_{a:g}", left["p_value"] < a, right["p_value"] < a) for a in ALPHAS]
                else:
                    outcomes = [("marginal_detection", left["successful_detection"], right["successful_detection"])]
                for outcome, left_values, right_values in outcomes:
                    l = left[pair_keys].copy(); l["value_first"] = left_values.astype(bool).to_numpy()
                    r = right[pair_keys].copy(); r["value_second"] = right_values.astype(bool).to_numpy()
                    paired = l.merge(r, on=pair_keys, how="inner", validate="one_to_one")
                    n = len(paired)
                    first_only = int((paired["value_first"] & ~paired["value_second"]).sum())
                    second_only = int((~paired["value_first"] & paired["value_second"]).sum())
                    diff = (second_only - first_only) / n if n else np.nan
                    se = np.std(paired["value_second"].astype(int) - paired["value_first"].astype(int), ddof=1) / np.sqrt(n) if n > 1 else np.nan
                    discordant = first_only + second_only
                    p_value = float(stats.binomtest(second_only, discordant, 0.5).pvalue) if discordant else 1.0
                    rows.append({
                        "analysis_type": analysis_type,
                        "signal_strength": strength,
                        "outcome": outcome,
                        "first_method": first,
                        "second_method": second,
                        "n_complete_pairs": n,
                        "first_rate": float(paired["value_first"].mean()) if n else np.nan,
                        "second_rate": float(paired["value_second"].mean()) if n else np.nan,
                        "difference_second_minus_first": diff,
                        "ci_95_lower": max(-1.0, diff - 1.96 * se) if np.isfinite(se) else np.nan,
                        "ci_95_upper": min(1.0, diff + 1.96 * se) if np.isfinite(se) else np.nan,
                        "first_only": first_only,
                        "second_only": second_only,
                        "mcnemar_exact_p_value": p_value,
                    })
    return pd.DataFrame.from_records(rows)


def annotate_snapshot(fig, coverage: str):
    title = fig._suptitle.get_text() if fig._suptitle is not None else ""
    fig.suptitle(f"{title}\n{coverage}", fontsize=14, fontweight="bold", color="#222222")


def plot_overview(summary: pd.DataFrame, coverage: str):
    fig, axes = plt.subplots(2, 2, figsize=(13, 9), constrained_layout=True)
    null = summary[summary["analysis_type"] == "null_calibration"]
    power = summary[summary["analysis_type"] == "power"]
    ax = axes[0, 0]
    offsets = np.linspace(-0.018, 0.018, max(1, null["selection_method"].nunique()))
    for offset, method in zip(offsets, sorted(null["selection_method"].unique())):
        row = null[null["selection_method"] == method].iloc[0]
        rates = np.array([row[f"rejection_rate_alpha_{a:g}"] for a in ALPHAS])
        lows = np.array([row[f"rejection_ci_95_lower_alpha_{a:g}"] for a in ALPHAS])
        highs = np.array([row[f"rejection_ci_95_upper_alpha_{a:g}"] for a in ALPHAS])
        x = np.array(ALPHAS) + offset
        ax.errorbar(x, rates, yerr=[rates-lows, highs-rates], marker="o", capsize=3,
                    label=METHOD_LABELS.get(method, method), color=METHOD_COLORS.get(method))
    ax.plot(ALPHAS, ALPHAS, "--", color="#333333", linewidth=1, label="Nominal")
    ax.set(title="Null calibration with exact 95% intervals", xlabel="Nominal alpha", ylabel="Rejection rate", xlim=(0, .115), ylim=(0, .14))
    ax.legend(fontsize=8)

    metrics = [
        ("marginal_detection_power", "Marginal detection power", axes[0, 1]),
        ("conditional_power_given_signal_target", "Conditional power given signal target", axes[1, 0]),
        ("target_signal_rate", "Signal selected as inference target", axes[1, 1]),
    ]
    for metric, title, ax in metrics:
        for method, frame in power.groupby("selection_method"):
            frame = frame.sort_values("signal_strength")
            ax.plot(frame["signal_strength"], frame[metric], marker="o", linewidth=2,
                    label=METHOD_LABELS.get(method, method), color=METHOD_COLORS.get(method))
            low_col = metric.replace("given_signal_target", "") + "_ci_95_lower"
            high_col = metric.replace("given_signal_target", "") + "_ci_95_upper"
            if metric == "marginal_detection_power":
                low_col, high_col = "marginal_power_ci_95_lower", "marginal_power_ci_95_upper"
            elif metric == "conditional_power_given_signal_target":
                low_col, high_col = "conditional_power_ci_95_lower", "conditional_power_ci_95_upper"
            if low_col in frame and high_col in frame:
                ax.fill_between(frame["signal_strength"], frame[low_col], frame[high_col], color=METHOD_COLORS.get(method), alpha=.12)
        if metric == "conditional_power_given_signal_target":
            ax.axhline(.80, linestyle="--", color="#777777", linewidth=1, label="80% target")
        ax.set(title=title, xlabel="Signal strength", ylabel="Rate", ylim=(0, 1.02))
        ax.legend(fontsize=8)
    fig.suptitle("Selection-method validation overview — exact-set conditioning", fontsize=16, fontweight="bold")
    annotate_snapshot(fig, coverage)
    return fig


def plot_null_diagnostics(observations: pd.DataFrame, summary: pd.DataFrame, coverage: str):
    null = observations[observations["analysis_type"] == "null_calibration"]
    fig, axes = plt.subplots(2, 2, figsize=(13, 9), constrained_layout=True)
    ax = axes[0, 0]
    for method, frame in null.groupby("selection_method"):
        values = np.sort(frame.loc[~frame["failed"], "p_value"].dropna().to_numpy())
        expected = (np.arange(len(values)) + .5) / len(values)
        ax.plot(expected, values, label=METHOD_LABELS.get(method, method), color=METHOD_COLORS.get(method))
    ax.plot([0, 1], [0, 1], "--", color="#333333", linewidth=1)
    ax.set(title="Uniform Q–Q plot", xlabel="Uniform quantile", ylabel="Observed p-value", aspect="equal")
    ax.legend(fontsize=8)

    ax = axes[0, 1]
    grid = np.linspace(0, 1, 201)
    for method, frame in null.groupby("selection_method"):
        values = np.sort(frame.loc[~frame["failed"], "p_value"].dropna().to_numpy())
        ecdf = np.searchsorted(values, grid, side="right") / len(values)
        ax.plot(grid, ecdf-grid, label=METHOD_LABELS.get(method, method), color=METHOD_COLORS.get(method))
    ax.axhline(0, color="#333333", linestyle="--", linewidth=1)
    ax.set(title="ECDF minus Uniform CDF", xlabel="p-value threshold", ylabel="Empirical excess")
    ax.legend(fontsize=8)

    ax = axes[1, 0]
    mcmc = null[null["mcmc_replicate_acceptance_rate"].notna()]
    methods = sorted(mcmc["selection_method"].unique())
    if methods:
        ax.boxplot([mcmc.loc[mcmc["selection_method"] == m, "mcmc_replicate_acceptance_rate"] for m in methods], tick_labels=[METHOD_LABELS.get(m, m) for m in methods], showfliers=False)
    ax.set(title="MCMC replicate acceptance", ylabel="Acceptance rate", ylim=(0, 1))

    ax = axes[1, 1]
    ess_values = pd.to_numeric(null["denominator_ess"], errors="coerce")
    finite_ess = null[np.isfinite(ess_values)].copy()
    finite_ess["denominator_ess"] = ess_values[np.isfinite(ess_values)]
    methods = sorted(finite_ess["selection_method"].unique())
    ess = [finite_ess.loc[finite_ess["selection_method"] == m, "denominator_ess"] for m in methods]
    ax.boxplot(ess, tick_labels=[METHOD_LABELS.get(m, m) for m in methods], showfliers=False)
    ax.set(title="Finite denominator effective sample size", ylabel="ESS")
    fig.suptitle("Null-calibration and Monte Carlo diagnostics", fontsize=16, fontweight="bold")
    annotate_snapshot(fig, coverage)
    return fig


def plot_shards_and_coverage(shards: pd.DataFrame, inventory: pd.DataFrame, coverage: str):
    fig, axes = plt.subplots(2, 2, figsize=(13, 9), constrained_layout=True)
    null = shards[shards["analysis_type"] == "null_calibration"]
    for method, frame in null.groupby("selection_method"):
        axes[0, 0].plot(frame["shard_index"], frame["rejection_rate_alpha_0.05"], marker="o", label=METHOD_LABELS.get(method, method), color=METHOD_COLORS.get(method))
        axes[0, 1].plot(frame["shard_index"], frame["uniform_ks_statistic"], marker="o", label=METHOD_LABELS.get(method, method), color=METHOD_COLORS.get(method))
    axes[0, 0].axhline(.05, color="#333333", linestyle="--", linewidth=1)
    axes[0, 0].set(title="Null rejection rate by shard (alpha=.05)", xlabel="Shard", ylabel="Rate", ylim=(0, .16))
    axes[0, 1].set(title="Uniform K–S statistic by shard", xlabel="Shard", ylabel="K–S statistic")
    axes[0, 0].legend(fontsize=8); axes[0, 1].legend(fontsize=8)

    power = shards[shards["analysis_type"] == "power"]
    for method, frame in power.groupby("selection_method"):
        grouped = frame.groupby("signal_strength")["marginal_detection_power"].agg(["mean", "std"])
        axes[1, 0].errorbar(grouped.index, grouped["mean"], yerr=grouped["std"], marker="o", capsize=3, label=METHOD_LABELS.get(method, method), color=METHOD_COLORS.get(method))
    axes[1, 0].set(title="Between-shard power variation (mean ± SD)", xlabel="Signal strength", ylabel="Marginal power", ylim=(0, 1.02))
    axes[1, 0].legend(fontsize=8)

    inv = inventory.copy()
    inv["config"] = np.where(inv["analysis_type"].eq("null_calibration"), "Null", "β=" + inv["signal_strength"].astype(str))
    pivot = inv.pivot_table(index=["selection_method", "config"], columns="shard_index", values="present", aggfunc="max", fill_value=False)
    axes[1, 1].imshow(pivot.to_numpy(dtype=float), aspect="auto", cmap="RdYlGn", vmin=0, vmax=1)
    axes[1, 1].set_xticks(range(len(pivot.columns)), pivot.columns)
    axes[1, 1].set_yticks(range(len(pivot.index)), [f"{METHOD_LABELS.get(m,m)} | {c}" for m,c in pivot.index], fontsize=8)
    axes[1, 1].set(title="Result-file coverage (red = missing)", xlabel="Shard")
    fig.suptitle("Shard stability and snapshot completeness", fontsize=16, fontweight="bold")
    annotate_snapshot(fig, coverage)
    return fig


def plot_method_comparisons(comparisons: pd.DataFrame, summary: pd.DataFrame, coverage: str):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5), constrained_layout=True)
    null = comparisons[(comparisons["analysis_type"] == "null_calibration") & (comparisons["outcome"] == "rejection_alpha_0.05")]
    labels = [f"{METHOD_LABELS.get(r.second_method,r.second_method)} −\n{METHOD_LABELS.get(r.first_method,r.first_method)}" for r in null.itertuples()]
    values = null["difference_second_minus_first"].to_numpy()
    errors = np.vstack([values-null["ci_95_lower"].to_numpy(), null["ci_95_upper"].to_numpy()-values]) if len(null) else None
    axes[0].errorbar(np.arange(len(null)), values, yerr=errors, fmt="o", capsize=4, color="#4C78A8")
    axes[0].axhline(0, color="#333333", linestyle="--", linewidth=1)
    axes[0].set_xticks(np.arange(len(null)), labels, fontsize=8)
    axes[0].set(title="Paired null rejection-rate differences", ylabel="Second − first at alpha=.05")

    power = comparisons[(comparisons["analysis_type"] == "power") & (comparisons["outcome"] == "marginal_detection")]
    for pair, frame in power.groupby(["first_method", "second_method"]):
        frame = frame.sort_values("signal_strength")
        label = f"{METHOD_LABELS.get(pair[1],pair[1])} − {METHOD_LABELS.get(pair[0],pair[0])}"
        axes[1].plot(frame["signal_strength"], frame["difference_second_minus_first"], marker="o", label=label)
        axes[1].fill_between(frame["signal_strength"], frame["ci_95_lower"], frame["ci_95_upper"], alpha=.10)
    axes[1].axhline(0, color="#333333", linestyle="--", linewidth=1)
    axes[1].set(title="Paired marginal-power differences", xlabel="Signal strength", ylabel="Second − first")
    axes[1].legend(fontsize=8)
    fig.suptitle("Cross-method paired comparisons on shared replicates", fontsize=16, fontweight="bold")
    annotate_snapshot(fig, coverage)
    return fig


def write_readme(output_dir: Path, root: Path, inventory: pd.DataFrame, observations: pd.DataFrame, summary: pd.DataFrame):
    present_null = int(inventory.query("analysis_type == 'null_calibration'")["present"].sum())
    expected_null = int((inventory["analysis_type"] == "null_calibration").sum())
    present_power = int(inventory.query("analysis_type == 'power'")["present"].sum())
    expected_power = int((inventory["analysis_type"] == "power").sum())
    missing = inventory[~inventory["present"]][["shard", "selection_method", "analysis_type", "signal_strength", "source_path"]]
    is_complete = present_null == expected_null and present_power == expected_power
    lines = [
        "# Selection-method validation snapshot analysis",
        "",
        f"Source: `{root}`",
        f"Observation rows: {len(observations):,}",
        f"Null configuration-shards: {present_null}/{expected_null}",
        f"Power configuration-shards: {present_power}/{expected_power}",
        "",
        (
            "All expected configuration-shards are present; rates and intervals use the full "
            "1,500 iterations per method/configuration."
            if is_complete
            else "Rates and intervals use observed snapshot rows; this is not a completed "
            "1,500-iteration pooled analysis."
        ),
        "",
        "## Files",
        "",
        "- `unified_dataset.csv`: one row per target-level simulation result, with provenance and diagnostics.",
        "- `configuration_summary.csv`: pooled snapshot metrics and exact binomial intervals.",
        "- `shard_summary.csv`: per-shard calibration, power, and selection metrics.",
        "- `method_pair_comparisons.csv`: paired differences using shared shard/iteration IDs.",
        "- `configuration_inventory.csv`: expected versus present result files.",
        "- `plots/*.png`: individual high-resolution figures.",
        "- `comprehensive_analysis_plots.pdf`: all figures in one PDF.",
        "",
        "## Missing result files",
        "",
    ]
    if missing.empty:
        lines.append("None.")
    else:
        lines += ["```", missing.to_string(index=False), "```"]
    lines += ["", "## Configuration-level snapshot summary", "", "```", summary.to_string(index=False), "```", ""]
    (output_dir / "README.md").write_text("\n".join(lines), encoding="utf-8")


def main(argv=None):
    args = parse_args(argv)
    root = args.input_dir.resolve()
    output_dir = (args.output_dir or root / "analysis_snapshot").resolve()
    validation_files = [p for p in root.glob("shard_*/validation_settings.json") if p.is_file()]
    if not validation_files:
        raise FileNotFoundError("No shard validation_settings.json was found.")
    validation = read_json(sorted(validation_files)[0])
    inventory = configuration_inventory(root, validation)
    observations = load_observations(root, inventory, validation)
    summary = summarize_configurations(observations, inventory)
    shards = summarize_shards(observations)
    comparisons = paired_method_comparisons(observations)

    output_dir.mkdir(parents=True, exist_ok=True)
    plot_dir = output_dir / "plots"
    plot_dir.mkdir(exist_ok=True)
    observations.to_csv(output_dir / "unified_dataset.csv", index=False)
    inventory.to_csv(output_dir / "configuration_inventory.csv", index=False)
    summary.to_csv(output_dir / "configuration_summary.csv", index=False)
    shards.to_csv(output_dir / "shard_summary.csv", index=False)
    comparisons.to_csv(output_dir / "method_pair_comparisons.csv", index=False)

    present_null = int(inventory.query("analysis_type == 'null_calibration'")["present"].sum())
    expected_null = int((inventory["analysis_type"] == "null_calibration").sum())
    present_power = int(inventory.query("analysis_type == 'power'")["present"].sum())
    expected_power = int((inventory["analysis_type"] == "power").sum())
    status = "COMPLETE RUN" if present_null == expected_null and present_power == expected_power else "INCOMPLETE SNAPSHOT"
    coverage = f"{status} | null {present_null}/{expected_null} configuration-shards | power {present_power}/{expected_power}"
    figures = [
        ("01_overview", plot_overview(summary, coverage)),
        ("02_null_diagnostics", plot_null_diagnostics(observations, summary, coverage)),
        ("03_shards_and_coverage", plot_shards_and_coverage(shards, inventory, coverage)),
        ("04_paired_method_comparisons", plot_method_comparisons(comparisons, summary, coverage)),
    ]
    with PdfPages(output_dir / "comprehensive_analysis_plots.pdf") as pdf:
        for name, figure in figures:
            figure.savefig(plot_dir / f"{name}.png", dpi=220, bbox_inches="tight")
            pdf.savefig(figure, bbox_inches="tight")
            plt.close(figure)
    write_readme(output_dir, root, inventory, observations, summary)
    print(f"Wrote {len(observations):,} unified rows to {output_dir}")
    print(coverage)
    print(inventory.groupby(["selection_method", "analysis_type"])["present"].agg(["sum", "count"]).to_string())


if __name__ == "__main__":
    main()
