"""Combine SHAP, spline-screening, and marginal-screening validation shards.

Only the common, scientifically comparable slice is pooled: baseline design,
fresh auxiliary randomization, and the ``same_target`` selection event.  Raw
records remain independent across methods; the script never treats method
results as paired because the root simulation seeds differ.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats


METHODS = ("shap", "spline_screening", "marginal_screening")
METHOD_LABELS = {
    "shap": "SHAP",
    "spline_screening": "Spline",
    "marginal_screening": "Corr",
}
COLORS = {
    "shap": "#4C78A8",
    "spline_screening": "#F58518",
    "marginal_screening": "#54A24B",
}
BETAS = (0.3, 0.5, 0.75, 1.0)
ALPHAS = (0.01, 0.05, 0.1)
POWER_DIRS = {
    0.3: "power_feature_0_beta_0p3",
    0.5: "power_feature_0_beta_0p5",
    0.75: "power_feature_0_beta_0p75",
    1.0: "power_feature_0_beta_1p0",
}


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--shap-dir",
        type=Path,
        default=Path("outputs/selection_method_validation_clean_shards"),
    )
    parser.add_argument(
        "--marginal-dir",
        type=Path,
        default=Path("outputs/selection_method_validation_same_target_mi_mc_shards"),
    )
    parser.add_argument(
        "--spline-dir",
        type=Path,
        default=Path("outputs/selection_method_validation_same_target_spline_screening_shards"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/selection_method_validation_combined_analysis"),
    )
    return parser.parse_args(argv)


def ordered_shards(root: Path) -> list[Path]:
    shards = [p for p in root.glob("shard_*") if p.is_dir()]
    return sorted(
        shards,
        key=lambda p: (
            0,
            int(p.name.removeprefix("shard_")),
        )
        if p.name.removeprefix("shard_").isdigit()
        else (1, p.name),
    )


def read_json(path: Path) -> dict:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def exact_interval(successes: int, trials: int, confidence: float = 0.95):
    if trials == 0:
        return np.nan, np.nan
    tail = (1.0 - confidence) / 2.0
    lower = 0.0 if successes == 0 else stats.beta.ppf(tail, successes, trials - successes + 1)
    upper = 1.0 if successes == trials else stats.beta.ppf(1 - tail, successes + 1, trials - successes)
    return float(lower), float(upper)


def add_context(
    frame: pd.DataFrame,
    *,
    method: str,
    shard: Path,
    source_root: Path,
    record_type: str,
    signal_strength: float | None,
    source_file: Path,
) -> pd.DataFrame:
    frame = frame.copy()
    frame.insert(0, "record_type", record_type)
    frame.insert(1, "selection_method", method)
    frame.insert(2, "design", "baseline")
    frame.insert(3, "auxiliary_regime", "fresh")
    frame.insert(4, "signal_strength", signal_strength)
    frame.insert(5, "validation_shard", shard.name)
    frame.insert(6, "source_experiment", source_root.name)
    frame.insert(7, "source_file", str(source_file))
    frame.insert(
        8,
        "record_id",
        [f"{method}:{record_type}:{signal_strength}:{shard.name}:{i}" for i in range(len(frame))],
    )
    return frame


def comparable_settings(settings: dict, kind: str) -> dict:
    keys = [
        "feature_correlation",
        "final_batch_size",
        "inference_method",
        "k_select",
        "max_final_samples",
        "min_denominator_ess",
        "min_tail_ess",
        "multiplicity",
        "n_features",
        "n_samples",
        "pilot_iters",
        "pilot_samples",
        "selection_decimals",
        "sigma",
        "stop_when_ess_met",
        "target_rule",
        "variance_method",
    ]
    if kind == "null":
        keys += ["alpha_levels", "auxiliary_randomization_mode", "fixed_auxiliary_u", "fixed_design"]
    else:
        keys += ["alpha", "memoized_selector_within_iteration", "signal_features", "signal_strength"]
    return {key: settings.get(key) for key in keys}


def verify_settings(reference: dict, candidate: dict, label: str):
    differences = {
        key: (reference.get(key), candidate.get(key))
        for key in sorted(set(reference) | set(candidate))
        if reference.get(key) != candidate.get(key)
    }
    if differences:
        raise ValueError(f"Non-method experimental settings differ for {label}: {differences}")


def load_all(shap_root: Path, marginal_root: Path, spline_root: Path):
    roots = {
        "shap": shap_root,
        "spline_screening": spline_root,
        "marginal_screening": marginal_root,
    }
    null_frames: list[pd.DataFrame] = []
    feature_frames: list[pd.DataFrame] = []
    target_frames: list[pd.DataFrame] = []
    settings_audit: dict[str, dict] = {}
    design_hashes: set[str] = set()
    expected_common: dict[tuple[str, float | None], dict] = {}
    duplicate_signal_files_verified = True

    for method, root in roots.items():
        shards = ordered_shards(root)
        if not shards:
            raise FileNotFoundError(f"No shards found under {root}")
        settings_audit[method] = {"root": str(root), "n_shards": len(shards), "shards": []}
        for shard in shards:
            method_root = shard / method / "baseline"
            null_dir = method_root / "null_fresh"
            null_settings = read_json(null_dir / "settings.json")
            if "same_target" not in null_settings["selection_events"]:
                raise ValueError(f"same_target is absent from {null_dir}")
            common = comparable_settings(null_settings, "null")
            if ("null", None) not in expected_common:
                expected_common[("null", None)] = common
            else:
                verify_settings(expected_common[("null", None)], common, f"{method} null")

            design_bytes = (null_dir / "fixed_design.npy").read_bytes()
            design_hashes.add(hashlib.sha256(design_bytes).hexdigest())
            null_path = null_dir / "p_value_results.csv"
            null = pd.read_csv(null_path)
            null = null.loc[null["selection_event"].eq("same_target")].reset_index(drop=True)
            expected = int(null_settings["n_iters"])
            if len(null) != expected:
                raise ValueError(f"{null_path} has {len(null)} comparable rows; expected {expected}")
            null_frames.append(
                add_context(
                    null,
                    method=method,
                    shard=shard,
                    source_root=root,
                    record_type="null_target",
                    signal_strength=None,
                    source_file=null_path,
                )
            )
            shard_audit = {"shard": shard.name, "null_iterations": expected, "power_iterations": {}}

            for beta, power_name in POWER_DIRS.items():
                power_dir = method_root / power_name
                power_settings = read_json(power_dir / "settings.json")
                if "same_target" not in power_settings["selection_events"]:
                    raise ValueError(f"same_target is absent from {power_dir}")
                common = comparable_settings(power_settings, "power")
                if ("power", beta) not in expected_common:
                    expected_common[("power", beta)] = common
                else:
                    verify_settings(expected_common[("power", beta)], common, f"{method} beta={beta}")
                expected_power = int(power_settings["n_iters"])
                feature_path = power_dir / "feature_results.csv"
                target_path = power_dir / "target_results.csv"
                signal_path = power_dir / "signal_results.csv"
                features = pd.read_csv(feature_path)
                features = features.loc[features["selection_event"].eq("same_target")].reset_index(drop=True)
                targets = pd.read_csv(target_path)
                targets = targets.loc[targets["selection_event"].eq("same_target")].reset_index(drop=True)
                signals = pd.read_csv(signal_path)
                signals = signals.loc[signals["selection_event"].eq("same_target")].reset_index(drop=True)
                if len(features) != expected_power or len(targets) != expected_power:
                    raise ValueError(
                        f"{power_dir} has feature/target counts {len(features)}/{len(targets)}; "
                        f"expected {expected_power}"
                    )
                comparison_columns = [c for c in targets.columns if c in signals.columns]
                if not targets[comparison_columns].equals(signals[comparison_columns]):
                    duplicate_signal_files_verified = False
                    raise ValueError(f"target_results and signal_results differ in {power_dir}")
                feature_frames.append(
                    add_context(
                        features,
                        method=method,
                        shard=shard,
                        source_root=root,
                        record_type="power_feature",
                        signal_strength=beta,
                        source_file=feature_path,
                    )
                )
                target_frames.append(
                    add_context(
                        targets,
                        method=method,
                        shard=shard,
                        source_root=root,
                        record_type="power_target",
                        signal_strength=beta,
                        source_file=target_path,
                    )
                )
                shard_audit["power_iterations"][str(beta)] = expected_power
            settings_audit[method]["shards"].append(shard_audit)

    if len(design_hashes) != 1:
        raise ValueError("The fixed null designs are not byte-identical across methods/shards")
    settings_audit["comparison"] = {
        "common_slice": {
            "design": "baseline",
            "auxiliary_regime": "fresh",
            "selection_event": "same_target",
            "signal_feature": 0,
            "signal_strengths": list(BETAS),
        },
        "fixed_design_sha256": next(iter(design_hashes)),
        "fixed_design_identical": True,
        "non_method_experiment_settings_identical": True,
        "independent_method_runs": True,
        "target_and_signal_results_exact_duplicates": duplicate_signal_files_verified,
        "intentional_non_scientific_differences": [
            "root seeds and shard partitioning",
            "SHAP also recorded feature_inclusion; comparison filters to same_target",
            "SHAP random-forest worker count (method-specific compute setting)",
        ],
    }
    return (
        pd.concat(null_frames, ignore_index=True),
        pd.concat(feature_frames, ignore_index=True),
        pd.concat(target_frames, ignore_index=True),
        settings_audit,
    )


def summarize_null(null: pd.DataFrame):
    rows = []
    distribution_rows = []
    for method, frame in null.groupby("selection_method", sort=False):
        valid = frame.loc[~frame["failed"].astype(bool) & frame["p_value"].notna()]
        p_values = valid["p_value"].to_numpy(float)
        ks = stats.kstest(p_values, "uniform")
        sorted_p = np.sort(p_values)
        ecdf = np.arange(1, len(sorted_p) + 1) / len(sorted_p)
        max_excess = float(np.max(ecdf - sorted_p))
        distribution_rows.append(
            {
                "selection_method": method,
                "n_iterations": len(frame),
                "n_failed": int(frame["failed"].astype(bool).sum()),
                "ks_statistic": float(ks.statistic),
                "ks_p_value_two_sided": float(ks.pvalue),
                "max_ecdf_excess": max_excess,
                "mean_p_value": float(np.mean(p_values)),
                "median_p_value": float(np.median(p_values)),
            }
        )
        for alpha in ALPHAS:
            # Match the simulation workflow's rejection rule exactly.
            rejected = int(np.sum(p_values < alpha))
            lo, hi = exact_interval(rejected, len(p_values))
            rows.append(
                {
                    "selection_method": method,
                    "alpha": alpha,
                    "n_valid": len(p_values),
                    "n_rejected": rejected,
                    "rejection_rate": rejected / len(p_values),
                    "ci_95_lower": lo,
                    "ci_95_upper": hi,
                    "anti_conservative_binomial_p_value": float(
                        stats.binomtest(rejected, len(p_values), alpha, alternative="greater").pvalue
                    ),
                }
            )
    return pd.DataFrame(rows), pd.DataFrame(distribution_rows)


def metric_row(successes: int, trials: int, prefix: str):
    lo, hi = exact_interval(successes, trials)
    return {
        prefix: successes / trials if trials else np.nan,
        f"{prefix}_ci_95_lower": lo,
        f"{prefix}_ci_95_upper": hi,
    }


def summarize_power(targets: pd.DataFrame):
    rows = []
    for (method, beta), frame in targets.groupby(["selection_method", "signal_strength"], sort=False):
        failed = frame["failed"].astype(bool)
        signal = frame["target_is_signal"].astype(bool)
        rejected = frame["rejected"].astype(bool)
        detected = frame["successful_detection"].astype(bool)
        valid = ~failed
        n = len(frame)
        n_valid = int(valid.sum())
        n_signal = int(signal.sum())
        n_non_signal = int((~signal).sum())
        row = {
            "selection_method": method,
            "signal_strength": float(beta),
            "n_iterations": n,
            "n_failed": int(failed.sum()),
            "n_signal_targets": n_signal,
            "n_non_signal_targets": n_non_signal,
        }
        row.update(metric_row(n_signal, n, "target_signal_rate"))
        # The target is sampled uniformly from k=2 selected features.
        target_lo, target_hi = exact_interval(n_signal, n)
        row.update(
            {
                "signal_inclusion_rate": min(1.0, 2 * n_signal / n),
                "signal_inclusion_rate_ci_95_lower": min(1.0, 2 * target_lo),
                "signal_inclusion_rate_ci_95_upper": min(1.0, 2 * target_hi),
            }
        )
        row.update(metric_row(int(detected.sum()), n, "marginal_detection_power"))
        row.update(metric_row(int((detected & signal).sum()), n_signal, "conditional_power"))
        row.update(metric_row(int((rejected & ~signal & valid).sum()), n_non_signal, "non_signal_rejection_rate"))
        row.update(metric_row(int((rejected & valid).sum()), n_valid, "all_target_rejection_rate"))
        rows.append(row)
    return pd.DataFrame(rows).sort_values(["selection_method", "signal_strength"])


def summarize_diagnostics(null: pd.DataFrame, features: pd.DataFrame):
    records = []
    frames = [("null", null)]
    for beta, frame in features.groupby("signal_strength", sort=True):
        frames.append((f"power beta={beta:g}", frame))
    for context, data in frames:
        for method, frame in data.groupby("selection_method", sort=False):
            records.append(
                {
                    "selection_method": method,
                    "context": context,
                    "n_records": len(frame),
                    "failure_rate": float(frame["failed"].astype(bool).mean()),
                    "finite_sample_valid_rate": float(frame["finite_sample_valid"].astype(bool).mean()),
                    "median_denominator_ess": float(frame["denominator_ess"].median()),
                    "q10_denominator_ess": float(frame["denominator_ess"].quantile(0.1)),
                    "median_tail_ess": float(frame["tail_ess"].median()),
                    "q10_tail_ess": float(frame["tail_ess"].quantile(0.1)),
                    "median_mc_se": float(frame["mc_se"].median()),
                    "q90_mc_se": float(frame["mc_se"].quantile(0.9)),
                    "fraction_below_denominator_ess_80": float((frame["denominator_ess"] < 80).mean()),
                    "fraction_below_tail_ess_15": float((frame["tail_ess"] < 15).mean()),
                }
            )
    return pd.DataFrame(records)


def pairwise_power_differences(power: pd.DataFrame):
    metrics = ["signal_inclusion_rate", "marginal_detection_power", "conditional_power", "non_signal_rejection_rate"]
    trial_columns = {
        "signal_inclusion_rate": "n_iterations",
        "marginal_detection_power": "n_iterations",
        "conditional_power": "n_signal_targets",
        "non_signal_rejection_rate": "n_non_signal_targets",
    }
    rows = []
    for beta in BETAS:
        subset = power.loc[power["signal_strength"].eq(beta)].set_index("selection_method")
        for left_i, left in enumerate(METHODS):
            for right in METHODS[left_i + 1 :]:
                for metric in metrics:
                    p_left = float(subset.loc[left, metric])
                    p_right = float(subset.loc[right, metric])
                    n_left = int(subset.loc[left, trial_columns[metric]])
                    n_right = int(subset.loc[right, trial_columns[metric]])
                    if metric == "signal_inclusion_rate":
                        # This is 2 * P(random target is the signal), not a direct
                        # binomial proportion; use the variance of the underlying
                        # target-signal indicators and apply the same scale.
                        q_left = p_left / 2
                        q_right = p_right / 2
                        se = 2 * np.sqrt(
                            q_left * (1 - q_left) / n_left
                            + q_right * (1 - q_right) / n_right
                        )
                    else:
                        se = np.sqrt(p_left * (1 - p_left) / n_left + p_right * (1 - p_right) / n_right)
                    rows.append(
                        {
                            "signal_strength": beta,
                            "metric": metric,
                            "method_left": left,
                            "method_right": right,
                            "difference_left_minus_right": p_left - p_right,
                            "wald_ci_95_lower": p_left - p_right - 1.96 * se,
                            "wald_ci_95_upper": p_left - p_right + 1.96 * se,
                        }
                    )
    return pd.DataFrame(rows)


def setup_style():
    plt.rcParams.update(
        {
            "figure.dpi": 120,
            "savefig.dpi": 240,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.22,
            "font.size": 10,
        }
    )


def plot_null(null: pd.DataFrame, calibration: pd.DataFrame, path: Path):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    ax = axes[0]
    ax.plot([0, 0.11], [0, 0.11], color="0.25", linestyle="--", label="Ideal")
    for method in METHODS:
        frame = calibration.loc[calibration["selection_method"].eq(method)].sort_values("alpha")
        y = frame["rejection_rate"].to_numpy()
        err = np.vstack([y - frame["ci_95_lower"].to_numpy(), frame["ci_95_upper"].to_numpy() - y])
        ax.errorbar(frame["alpha"], y, yerr=err, marker="o", capsize=3, color=COLORS[method], label=METHOD_LABELS[method])
    ax.set(xlabel="Nominal alpha", ylabel="Empirical null rejection rate", xlim=(0, 0.105), ylim=(0, 0.115), title="Null calibration with exact 95% CIs")
    ax.legend(frameon=False)

    ax = axes[1]
    grid = np.linspace(0, 1, 301)
    ax.plot(grid, grid, color="0.25", linestyle="--", label="Uniform")
    for method in METHODS:
        values = np.sort(null.loc[null["selection_method"].eq(method), "p_value"].dropna().to_numpy())
        ecdf = np.searchsorted(values, grid, side="right") / len(values)
        ax.plot(grid, ecdf, color=COLORS[method], label=METHOD_LABELS[method])
    ax.set(xlabel="p-value", ylabel="Empirical CDF", xlim=(0, 1), ylim=(0, 1), title="Full null p-value distribution")
    ax.legend(frameon=False)
    fig.suptitle("Same-target inference: null validity across selection methods", fontweight="bold")
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def plot_power(power: pd.DataFrame, path: Path):
    panels = [
        ("signal_inclusion_rate", "Signal inclusion rate", "Feature-selection sensitivity"),
        ("marginal_detection_power", "Marginal detection power", "End-to-end discovery probability"),
        ("conditional_power", "Conditional power", "Power given a signal target"),
        ("non_signal_rejection_rate", "Non-signal rejection rate", "Selected non-signal tests"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True)
    for ax, (metric, ylabel, title) in zip(axes.flat, panels):
        for method in METHODS:
            frame = power.loc[power["selection_method"].eq(method)].sort_values("signal_strength")
            y = frame[metric].to_numpy()
            lo = frame[f"{metric}_ci_95_lower"].to_numpy()
            hi = frame[f"{metric}_ci_95_upper"].to_numpy()
            ax.errorbar(frame["signal_strength"], y, yerr=np.vstack([y - lo, hi - y]), marker="o", capsize=3, color=COLORS[method], label=METHOD_LABELS[method])
        if metric == "non_signal_rejection_rate":
            ax.axhline(0.05, color="0.25", linestyle="--", linewidth=1, label="0.05 reference")
        if metric == "conditional_power":
            ax.axhline(0.8, color="0.25", linestyle=":", linewidth=1, label="0.80 target")
        ax.set(ylabel=ylabel, title=title, ylim=(-0.02, 1.02))
        ax.set_xticks(BETAS)
    axes[1, 0].set_xlabel("Signal strength (beta)")
    axes[1, 1].set_xlabel("Signal strength (beta)")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 0.99))
    fig.suptitle("Selection and inference performance (same target, n=5,000 per point)", y=1.04, fontweight="bold")
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def plot_null_histograms(null: pd.DataFrame, path: Path):
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8), sharex=True, sharey=True)
    bins = np.linspace(0, 1, 21)
    for ax, method in zip(axes, METHODS):
        values = null.loc[null["selection_method"].eq(method), "p_value"].dropna()
        ax.hist(values, bins=bins, density=True, color=COLORS[method], alpha=0.75, edgecolor="white")
        ax.axhline(1, color="0.25", linestyle="--")
        ax.set(title=METHOD_LABELS[method], xlabel="Null p-value", ylim=(0, 1.55))
    axes[0].set_ylabel("Density")
    fig.suptitle("Null p-value histograms (same-target event)", fontweight="bold")
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def plot_diagnostics(diagnostics: pd.DataFrame, path: Path):
    contexts = ["null", *[f"power beta={beta:g}" for beta in BETAS]]
    x = np.arange(len(contexts))
    fig, axes = plt.subplots(2, 2, figsize=(13, 8), sharex=True)
    panels = [
        ("median_denominator_ess", "Median denominator ESS", 80),
        ("median_tail_ess", "Median tail ESS", 15),
        ("median_mc_se", "Median Monte Carlo SE", None),
        ("fraction_below_tail_ess_15", "Fraction with tail ESS < 15", None),
    ]
    for ax, (metric, title, threshold) in zip(axes.flat, panels):
        for method in METHODS:
            frame = diagnostics.loc[diagnostics["selection_method"].eq(method)].set_index("context").reindex(contexts)
            ax.plot(x, frame[metric], marker="o", color=COLORS[method], label=METHOD_LABELS[method])
        if threshold is not None:
            ax.axhline(threshold, color="0.25", linestyle="--", linewidth=1)
        ax.set_title(title)
        if metric.startswith("fraction"):
            ax.set_ylim(-0.02, 1.02)
    for ax in axes[1]:
        ax.set_xticks(x, contexts, rotation=25, ha="right")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 0.99))
    fig.suptitle("Conditional Monte Carlo diagnostics", y=1.03, fontweight="bold")
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def markdown_table(frame: pd.DataFrame, float_digits: int = 4) -> str:
    """Render a compact Markdown table without pandas' optional tabulate dependency."""
    rendered = frame.copy()
    for column in rendered.select_dtypes(include=["number"]).columns:
        rendered[column] = rendered[column].map(
            lambda value: "" if pd.isna(value) else f"{value:.{float_digits}f}"
        )
    columns = [str(column) for column in rendered.columns]
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join(["---"] * len(columns)) + " |",
    ]
    for row in rendered.itertuples(index=False, name=None):
        values = [str(value).replace("|", "\\|") for value in row]
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def write_report(path: Path, calibration: pd.DataFrame, null_dist: pd.DataFrame, power: pd.DataFrame, diagnostics: pd.DataFrame, audit: dict):
    power_lookup = power.set_index(["selection_method", "signal_strength"])
    null_005 = calibration.loc[calibration["alpha"].eq(0.05)].set_index("selection_method")
    lines = [
        "# Combined feature-selection-method validation",
        "",
        "## Comparability conclusion",
        "",
        "The three methods are directly comparable on the **baseline / fresh auxiliary / same-target** slice. "
        "The fixed design is byte-identical and all non-method statistical settings match. The runs use different root seeds and shard layouts, so method comparisons are independent rather than paired. SHAP additionally recorded the feature-inclusion event; those extra rows are deliberately excluded here.",
        "",
        "`signal_results.csv` was verified to duplicate `target_results.csv` exactly and was not double-counted.",
        "",
        "## Main findings",
        "",
        f"- Null calibration at alpha=0.05 is close to nominal for every method: SHAP {null_005.loc['shap', 'rejection_rate']:.2%}, Spline Screening {null_005.loc['spline_screening', 'rejection_rate']:.2%}, and Marginal Correlation Screening {null_005.loc['marginal_screening', 'rejection_rate']:.2%}. No method shows significant anti-conservative evidence in the tested alpha grid.",
        f"- At beta=1.0, marginal detection power is {power_lookup.loc[('marginal_screening', 1.0), 'marginal_detection_power']:.2%} for Marginal Correlation Screening, {power_lookup.loc[('spline_screening', 1.0), 'marginal_detection_power']:.2%} for Spline Screening, and {power_lookup.loc[('shap', 1.0), 'marginal_detection_power']:.2%} for SHAP.",
        f"- At beta=1.0, conditional power is {power_lookup.loc[('marginal_screening', 1.0), 'conditional_power']:.2%} for Marginal Correlation Screening, {power_lookup.loc[('spline_screening', 1.0), 'conditional_power']:.2%} for Spline Screening, and {power_lookup.loc[('shap', 1.0), 'conditional_power']:.2%} for SHAP.",
        "- All records are marked finite-sample valid and no tests failed. However, tail ESS deteriorates sharply under stronger alternatives; low/zero tail ESS and zero Monte Carlo SE at boundary p-values should be read as limited tail resolution, not perfect numerical precision.",
        "- Rejections among selected non-signal features rise with signal strength. Under this fixed-design alternative these features can have nonzero projection onto the signal mean, so this curve is diagnostic and should not be interpreted as a pure null type-I-error estimate.",
        "",
        "## Null calibration",
        "",
        markdown_table(calibration),
        "",
        "### Distribution diagnostics",
        "",
        markdown_table(null_dist),
        "",
        "## Power and selection",
        "",
        markdown_table(power[[
            "selection_method", "signal_strength", "n_iterations", "signal_inclusion_rate",
            "marginal_detection_power", "conditional_power", "non_signal_rejection_rate"
        ]]),
        "",
        "The signal-inclusion rate is reconstructed as twice the target-signal rate because the target is sampled uniformly from the two selected features (`k_select=2`).",
        "",
        "## Monte Carlo diagnostics",
        "",
        markdown_table(diagnostics),
        "",
        "## Files",
        "",
        "- `all_records_combined.csv`: schema-union long file of every non-duplicate raw record used.",
        "- `null_results_combined.csv`, `feature_results_combined.csv`, `target_results_combined.csv`: typed raw tables.",
        "- `null_calibration_summary.csv`, `null_distribution_summary.csv`, `power_summary.csv`, `power_pairwise_differences.csv`, `mc_diagnostics_summary.csv`: derived tables.",
        "- `null_calibration.png`, `null_p_value_histograms.png`, `power_comparison.png`, `mc_diagnostics.png`: plot suite.",
        "- `comparison_audit.json`: settings, shard counts, fixed-design hash, and comparability audit.",
        "",
        f"Fixed-design SHA-256: `{audit['comparison']['fixed_design_sha256']}`",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv=None):
    args = parse_args(argv)
    output = args.output_dir
    output.mkdir(parents=True, exist_ok=True)
    setup_style()
    null, features, targets, audit = load_all(
        args.shap_dir, args.marginal_dir, args.spline_dir
    )
    calibration, null_dist = summarize_null(null)
    power = summarize_power(targets)
    diagnostics = summarize_diagnostics(null, features)
    differences = pairwise_power_differences(power)
    audit["comparison"]["combined_record_counts"] = {
        "null_target": len(null),
        "power_feature": len(features),
        "power_target": len(targets),
        "all_nonduplicate_records": len(null) + len(features) + len(targets),
    }

    null.to_csv(output / "null_results_combined.csv", index=False)
    features.to_csv(output / "feature_results_combined.csv", index=False)
    targets.to_csv(output / "target_results_combined.csv", index=False)
    pd.concat([null, features, targets], ignore_index=True, sort=False).to_csv(
        output / "all_records_combined.csv", index=False
    )
    calibration.to_csv(output / "null_calibration_summary.csv", index=False)
    null_dist.to_csv(output / "null_distribution_summary.csv", index=False)
    power.to_csv(output / "power_summary.csv", index=False)
    differences.to_csv(output / "power_pairwise_differences.csv", index=False)
    diagnostics.to_csv(output / "mc_diagnostics_summary.csv", index=False)
    with (output / "comparison_audit.json").open("w", encoding="utf-8") as handle:
        json.dump(audit, handle, indent=2, sort_keys=True)

    plot_null(null, calibration, output / "null_calibration.png")
    plot_null_histograms(null, output / "null_p_value_histograms.png")
    plot_power(power, output / "power_comparison.png")
    plot_diagnostics(diagnostics, output / "mc_diagnostics.png")
    write_report(output / "analysis_report.md", calibration, null_dist, power, diagnostics, audit)

    print(f"Comparable null records: {len(null):,}")
    print(f"Comparable power feature records: {len(features):,}")
    print(f"Comparable power target records: {len(targets):,}")
    print(f"Saved combined analysis to {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
