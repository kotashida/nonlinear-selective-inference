"""Combine sharded AIS method-comparison results and create diagnostic plots."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


METHOD_LABELS = {
    "shap": "SHAP (AIS)",
    "lime": "LIME (AIS)",
    "spline_screening": "Spline screening (AIS)",
    "mutual_information": "Mutual information (AIS)",
}
ORDER = [
    "Spline screening (exact)",
    "Spline screening (AIS)",
    "SHAP (AIS)",
    "LIME (AIS)",
    "Mutual information k=12 (AIS)",
]
MARKERS = ["s", "D", "^", "v", "o"]
COLORS = {
    "Spline screening (exact)": "tab:blue",
    "Spline screening (AIS)": "tab:orange",
    "SHAP (AIS)": "tab:green",
    "LIME (AIS)": "tab:red",
    "Mutual information k=12 (AIS)": "tab:purple",
}


def _method_label(method: str, *, exact: bool, settings: dict) -> str:
    """Return a label that reflects the inference and active selector settings."""
    if exact:
        return "Spline screening (exact)"
    if method != "mutual_information":
        return METHOD_LABELS[method]
    params = settings.get("mi_params", {})
    if params.get("estimator") == "kde":
        bandwidth = params.get("bandwidth")
        suffix = "" if bandwidth is None else f" {float(bandwidth):g}"
        return f"Mutual information (KDE{suffix}, AIS)"
    neighbors = settings.get("mi_neighbors")
    suffix = "" if neighbors is None else f" k={int(neighbors)}"
    return f"Mutual information{suffix} (AIS)"


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_dir", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--expected-shards", type=int, default=32)
    parser.add_argument("--ais-only", action="store_true", help="Analyze completed AIS results without an exact reference.")
    return parser.parse_args(argv)


def _ordered_shards(run_dir: Path, expected: int) -> list[Path]:
    def key(path: Path):
        suffix = path.name.removeprefix("shard_")
        return (0, int(suffix)) if suffix.isdigit() else (1, suffix)

    shards = sorted(
        (
            path
            for path in run_dir.glob("shard_*")
            if path.is_dir() and (path / "validation_settings.json").is_file()
        ),
        key=key,
    )
    if len(shards) != expected:
        raise ValueError(
            f"{run_dir} has {len(shards)} completed shards; expected {expected}."
        )
    return shards


def _wilson_interval(successes, trials, z=1.959963984540054):
    successes = np.asarray(successes, dtype=float)
    trials = np.asarray(trials, dtype=float)
    proportion = successes / trials
    denominator = 1.0 + z**2 / trials
    center = (proportion + z**2 / (2.0 * trials)) / denominator
    half = (
        z
        * np.sqrt(proportion * (1.0 - proportion) / trials + z**2 / (4 * trials**2))
        / denominator
    )
    return center - half, center + half


def _read_pooled(run_dir: Path, run_kind: str) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    pooled = run_dir / "pooled_summary"
    calibration = pd.read_csv(pooled / "calibration_decisions.csv")
    power = pd.read_csv(pooled / "power_decisions.csv")
    with (pooled / "validation_settings.json").open(encoding="utf-8") as file:
        settings = json.load(file)
    exact = run_kind in {"exact", "exact_spline"}
    for frame in (calibration, power):
        frame.insert(0, "inference", "exact" if exact else "ais")
        frame.insert(0, "analysis_label", "Spline screening (exact)" if exact else "")
        if not exact:
            frame["analysis_label"] = frame["selection_method"].map(
                lambda method: _method_label(method, exact=False, settings=settings)
            )
    return calibration, power, settings


def _read_raw(
    run_dir: Path,
    run_kind: str,
    settings: dict,
    expected_shards: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    null_frames: list[pd.DataFrame] = []
    target_frames: list[pd.DataFrame] = []
    feature_frames: list[pd.DataFrame] = []
    exact = run_kind in {"exact", "exact_spline"}
    for shard in _ordered_shards(run_dir, expected_shards):
        for method in settings["methods"]:
            label = _method_label(method, exact=exact, settings=settings)
            prefix = {
                "analysis_label": label,
                "inference": "exact" if exact else "ais",
                "selection_method": method,
                "validation_shard": shard.name,
            }
            null_path = shard / method / "baseline" / "null_fresh" / "p_value_results.csv"
            null = pd.read_csv(null_path)
            for name, value in reversed(prefix.items()):
                null.insert(0, name, value)
            null_frames.append(null)

            for strength in settings["signal_strengths"]:
                strength_label = str(strength).replace(".", "p")
                result_dir = (
                    shard
                    / method
                    / "baseline"
                    / f"power_feature_0_beta_{strength_label}"
                )
                target = pd.read_csv(result_dir / "target_results.csv")
                feature = pd.read_csv(result_dir / "feature_results.csv")
                for frame in (target, feature):
                    frame.insert(0, "signal_strength", float(strength))
                    for name, value in reversed(prefix.items()):
                        frame.insert(0, name, value)
                target_frames.append(target)
                feature_frames.append(feature)

    return (
        pd.concat(null_frames, ignore_index=True, sort=False),
        pd.concat(target_frames, ignore_index=True, sort=False),
        pd.concat(feature_frames, ignore_index=True, sort=False),
    )


def _save_figure(fig: plt.Figure, output_dir: Path, stem: str) -> None:
    fig.tight_layout()
    fig.savefig(output_dir / f"{stem}.png", dpi=190, bbox_inches="tight")
    fig.savefig(output_dir / f"{stem}.pdf", bbox_inches="tight")
    plt.close(fig)


def _plot_type1(calibration: pd.DataFrame, output_dir: Path) -> None:
    frame = calibration.copy()
    trials = frame["n_iterations"] - frame["n_failed"]
    frame["ci_lower"], frame["ci_upper"] = _wilson_interval(
        frame["n_rejected"], trials
    )
    fig, ax = plt.subplots(figsize=(8.8, 6.2))
    for label, marker in zip(ORDER, MARKERS):
        group = frame.loc[frame["analysis_label"].eq(label)].sort_values("alpha")
        if group.empty:
            continue
        ax.errorbar(
            group["alpha"],
            group["rejection_rate"],
            yerr=np.vstack(
                [
                    group["rejection_rate"] - group["ci_lower"],
                    group["ci_upper"] - group["rejection_rate"],
                ]
            ),
            marker=marker,
            color=COLORS[label],
            linewidth=1.8,
            capsize=3,
            label=label,
        )
    limits = (0.0, 0.13)
    ax.plot(limits, limits, color="0.35", linestyle="--", linewidth=1.2, label="Nominal")
    ax.set(
        xlabel="Nominal type-I error rate",
        ylabel="Observed null rejection rate",
        xlim=limits,
        ylim=(0.0, max(0.13, float(frame["ci_upper"].max()) * 1.08)),
        xticks=[0.01, 0.05, 0.10],
    )
    ax.grid(alpha=0.22)
    ax.legend(frameon=False, fontsize=8.5, ncol=2)
    ax.set_title("Type-I error regulation (95% Wilson intervals)")
    _save_figure(fig, output_dir, "type1-error-comparison")


def _plot_power(power: pd.DataFrame, output_dir: Path) -> None:
    metrics = [
        (
            "conditional_power_given_signal_target",
            "conditional_power_ci_95_lower",
            "conditional_power_ci_95_upper",
            "Complete-case estimate",
        ),
        (
            "conditional_power_resolution_lower_bound",
            None,
            None,
            "Conservative lower bound (failures = no detection)",
        ),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(13.2, 5.4), sharex=True, sharey=True)
    for ax, (metric, lower, upper, title) in zip(axes, metrics):
        for label, marker in zip(ORDER, MARKERS):
            group = power.loc[power["analysis_label"].eq(label)].sort_values(
                "signal_strength"
            )
            if group.empty:
                continue
            if lower:
                ax.errorbar(
                    group["signal_strength"],
                    group[metric],
                    yerr=np.vstack(
                        [group[metric] - group[lower], group[upper] - group[metric]]
                    ),
                    marker=marker,
                    color=COLORS[label],
                    linewidth=1.8,
                    capsize=3,
                    label=label,
                )
            else:
                ax.plot(
                    group["signal_strength"],
                    group[metric],
                    marker=marker,
                    color=COLORS[label],
                    linewidth=1.8,
                    label=label,
                )
        ax.axhline(0.8, color="0.35", linestyle="--", linewidth=1.1)
        ax.set(
            xlabel="Signal strength",
            ylim=(-0.02, 1.04),
            xticks=sorted(power["signal_strength"].unique()),
            title=title,
        )
        ax.grid(alpha=0.22)
    axes[0].set_ylabel("Conditional power given signal target")
    axes[1].legend(frameon=False, fontsize=8.2, loc="lower right")
    fig.suptitle("Conditional power by selection method")
    _save_figure(fig, output_dir, "conditional-power-comparison")


def _plot_marginal_power(power: pd.DataFrame, output_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(9.2, 5.8))
    for label, marker in zip(ORDER, MARKERS):
        group = power.loc[power["analysis_label"].eq(label)].sort_values(
            "signal_strength"
        )
        if group.empty:
            continue
        ax.errorbar(
            group["signal_strength"],
            group["marginal_detection_power"],
            yerr=np.vstack(
                [
                    group["marginal_detection_power"]
                    - group["marginal_power_ci_95_lower"],
                    group["marginal_power_ci_95_upper"]
                    - group["marginal_detection_power"],
                ]
            ),
            marker=marker,
            color=COLORS[label],
            linewidth=1.8,
            capsize=3,
            label=label,
        )
    ax.set(
        xlabel="Signal strength",
        ylabel="Marginal detection power",
        ylim=(-0.02, 1.02),
        xticks=sorted(power["signal_strength"].unique()),
    )
    ax.grid(alpha=0.22)
    ax.legend(frameon=False, fontsize=8.5, ncol=2)
    ax.set_title("Marginal power (selection and rejection; 95% Wilson intervals)")
    _save_figure(fig, output_dir, "marginal-power-comparison")


def _diagnostic_summary(
    null: pd.DataFrame, feature: pd.DataFrame, settings: dict
) -> pd.DataFrame:
    ais_null = null.loc[null["inference"].eq("ais")].copy()
    ais_null["scope"] = "null_calibration"
    ais_null["signal_strength"] = np.nan
    ais_power = feature.loc[feature["inference"].eq("ais")].copy()
    ais_power["scope"] = "power"
    data = pd.concat([ais_null, ais_power], ignore_index=True, sort=False)
    denominator_threshold = float(settings["min_denominator_ess"])
    tail_threshold = float(settings["min_tail_ess"])

    rows = []
    for keys, group in data.groupby(
        ["scope", "analysis_label", "signal_strength"], dropna=False, sort=False
    ):
        denominator = pd.to_numeric(group["denominator_ess"], errors="coerce")
        tail = pd.to_numeric(group["tail_ess"], errors="coerce")
        status = group["status"].fillna("missing").astype(str)
        failed = group["failed"].fillna(True).astype(bool)
        if keys[0] == "power":
            signal_target = group["is_signal"].fillna(False).astype(bool)
            n_signal_targets = int(signal_target.sum())
            n_failed_signal_targets = int((failed & signal_target).sum())
            signal_failure_rate = (
                n_failed_signal_targets / n_signal_targets
                if n_signal_targets
                else np.nan
            )
        else:
            n_signal_targets = 0
            n_failed_signal_targets = 0
            signal_failure_rate = np.nan
        rows.append(
            {
                "scope": keys[0],
                "analysis_label": keys[1],
                "signal_strength": keys[2],
                "n_tests": len(group),
                "n_failed": int(failed.sum()),
                "failure_rate": float(failed.mean()),
                "n_signal_targets": n_signal_targets,
                "n_failed_signal_targets": n_failed_signal_targets,
                "signal_target_failure_rate": signal_failure_rate,
                "n_insufficient_ess": int(status.eq("insufficient_ess").sum()),
                "denominator_ess_median": denominator.median(),
                "denominator_ess_q10": denominator.quantile(0.10),
                "denominator_ess_q25": denominator.quantile(0.25),
                "denominator_ess_q75": denominator.quantile(0.75),
                "tail_ess_median": tail.median(),
                "tail_ess_q10": tail.quantile(0.10),
                "tail_ess_q25": tail.quantile(0.25),
                "tail_ess_q75": tail.quantile(0.75),
                "denominator_ess_pass_rate": float((denominator >= denominator_threshold).mean()),
                "tail_ess_pass_rate": float((tail >= tail_threshold).mean()),
            }
        )
    return pd.DataFrame.from_records(rows)


def _plot_ais_diagnostics(summary: pd.DataFrame, settings: dict, output_dir: Path) -> None:
    power = summary.loc[summary["scope"].eq("power")].copy()
    ais_order = [label for label in ORDER if label.endswith("(AIS)")]
    fig, axes = plt.subplots(1, 2, figsize=(13.2, 5.4), sharex=True, sharey=True)
    failure_metrics = [
        ("failure_rate", "All randomly targeted selected features"),
        ("signal_target_failure_rate", "Signal targets only"),
    ]
    for ax, (metric, title) in zip(axes, failure_metrics):
        for label, marker in zip(ais_order, MARKERS[1:]):
            group = power.loc[power["analysis_label"].eq(label)].sort_values(
                "signal_strength"
            )
            ax.plot(
                group["signal_strength"],
                group[metric],
                marker=marker,
                color=COLORS[label],
                linewidth=1.8,
                label=label,
            )
        ax.set(
            xlabel="Signal strength",
            ylabel="AIS inference failure rate",
            ylim=(-0.02, 1.02),
            xticks=sorted(power["signal_strength"].unique()),
            title=title,
        )
        ax.grid(alpha=0.22)
    axes[1].legend(frameon=False, fontsize=8.2, loc="upper left")
    fig.suptitle("Adaptive AIS failures" if settings.get("stop_when_ess_met") else "Fixed-budget AIS failures")
    _save_figure(fig, output_dir, "ais-failure-rate")

    fig, axes = plt.subplots(1, 2, figsize=(13.2, 5.4), sharex=True)
    definitions = [
        ("denominator_ess", float(settings["min_denominator_ess"]), "Denominator ESS"),
        ("tail_ess", float(settings["min_tail_ess"]), "Tail ESS"),
    ]
    for ax, (prefix, threshold, title) in zip(axes, definitions):
        for label, marker in zip(ais_order, MARKERS[1:]):
            group = power.loc[power["analysis_label"].eq(label)].sort_values(
                "signal_strength"
            )
            median = group[f"{prefix}_median"]
            ax.errorbar(
                group["signal_strength"],
                median,
                yerr=np.vstack(
                    [median - group[f"{prefix}_q25"], group[f"{prefix}_q75"] - median]
                ),
                marker=marker,
                color=COLORS[label],
                linewidth=1.8,
                capsize=3,
                label=label,
            )
        ax.axhline(threshold, color="0.35", linestyle="--", linewidth=1.1)
        ax.set(
            xlabel="Signal strength",
            ylabel=f"{title} (median and IQR)",
            yscale="log",
            xticks=sorted(power["signal_strength"].unique()),
            title=f"{title}; required minimum = {threshold:g}",
        )
        ax.grid(alpha=0.22)
    axes[1].legend(frameon=True, framealpha=1.0, fontsize=8.2)
    fig.suptitle("AIS effective sample size diagnostics")
    _save_figure(fig, output_dir, "ais-ess-diagnostics")


def _write_report(
    output_dir: Path,
    calibration: pd.DataFrame,
    power: pd.DataFrame,
    diagnostics: pd.DataFrame,
) -> None:
    alpha05 = calibration.loc[np.isclose(calibration["alpha"], 0.05)].copy()
    beta075 = power.loc[np.isclose(power["signal_strength"], 0.75)].copy()
    beta075_diag = diagnostics.loc[
        diagnostics["scope"].eq("power")
        & np.isclose(diagnostics["signal_strength"], 0.75)
    ].copy()
    lines = [
        "# AIS method-comparison analysis",
        "",
        "This is a 200-repetition diagnostic pilot. AIS p-values are self-normalized importance-sampling estimates and are not finite-sample exact.",
        "",
        "## Type-I error at alpha = 0.05",
        "",
        "| Method | Rate | Failures | Formal pipeline pass |",
        "|---|---:|---:|:---:|",
    ]
    for label in ORDER:
        row = alpha05.loc[alpha05["analysis_label"].eq(label)]
        if row.empty:
            continue
        item = row.iloc[0]
        lines.append(
            f"| {label} | {item.rejection_rate:.3f} | {int(item.n_failed)}/{int(item.n_iterations)} | {'yes' if item.calibration_pass else 'no'} |"
        )
    lines.extend(
        [
            "",
            "## Power at beta = 0.75",
            "",
            "| Method | Target rate | Conditional (complete case) | Conservative lower bound | Marginal | Failed signal tests |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for label in ORDER:
        row = beta075.loc[beta075["analysis_label"].eq(label)]
        if row.empty:
            continue
        item = row.iloc[0]
        lines.append(
            f"| {label} | {item.target_signal_rate:.3f} | {item.conditional_power_given_signal_target:.3f} | {item.conditional_power_resolution_lower_bound:.3f} | {item.marginal_detection_power:.3f} | {int(item.n_failed_signal_targets)}/{int(item.n_signal_targets)} |"
        )
    lines.extend(
        [
            "",
            "## AIS reliability at beta = 0.75",
            "",
            "| Method | All-target failure rate | Signal-target failure rate | Median denominator ESS | Median tail ESS |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for label in ORDER:
        row = beta075_diag.loc[beta075_diag["analysis_label"].eq(label)]
        if row.empty:
            continue
        item = row.iloc[0]
        lines.append(
            f"| {label} | {item.failure_rate:.3f} | {item.signal_target_failure_rate:.3f} | {item.denominator_ess_median:.1f} | {item.tail_ess_median:.1f} |"
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "Complete-case conditional power excludes failed signal tests. Compare it with the conservative lower bound and marginal power, and inspect failure rates and ESS before interpreting differences. This pilot has only 200 repetitions per method and signal strength; it does not establish exact calibration of AIS p-values.",
            "",
        ]
    )
    if not calibration["inference"].eq("exact").any():
        lines.extend([
            "## Missing exact reference", "",
            "The exact-spline reference is excluded: it has no completed shards. Its server log reports `ValueError: stop_when_ess_met requires inference_method='ais'.` The reference must be rerun with AIS_STOP_WHEN_ESS_MET=false before an exact-versus-AIS comparison is available.", "",
        ])
    (output_dir / "analysis_report.md").write_text("\n".join(lines), encoding="utf-8")


def main(argv=None) -> None:
    args = parse_args(argv)
    input_dir = args.input_dir.resolve()
    output_dir = (args.output_dir or input_dir / "analysis").resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    run_specs = [
        (input_dir / "ais_all_methods", "ais"),
    ]
    if not args.ais_only:
        run_specs.append((input_dir / "spline_exact_reference", "exact"))
    calibration_frames = []
    power_frames = []
    null_frames = []
    target_frames = []
    feature_frames = []
    ais_settings = None
    for run_dir, run_kind in run_specs:
        calibration, power, settings = _read_pooled(run_dir, run_kind)
        null, target, feature = _read_raw(
            run_dir, run_kind, settings, args.expected_shards
        )
        calibration_frames.append(calibration)
        power_frames.append(power)
        null_frames.append(null)
        target_frames.append(target)
        feature_frames.append(feature)
        if run_kind == "ais":
            ais_settings = settings

    calibration = pd.concat(calibration_frames, ignore_index=True, sort=False)
    power = pd.concat(power_frames, ignore_index=True, sort=False)
    null = pd.concat(null_frames, ignore_index=True, sort=False)
    target = pd.concat(target_frames, ignore_index=True, sort=False)
    feature = pd.concat(feature_frames, ignore_index=True, sort=False)
    assert ais_settings is not None

    calibration.to_csv(output_dir / "calibration_comparison.csv", index=False)
    power.to_csv(output_dir / "power_comparison.csv", index=False)
    null.to_csv(output_dir / "null_results_combined.csv", index=False)
    target.to_csv(output_dir / "power_target_results_combined.csv", index=False)
    feature.to_csv(output_dir / "power_feature_results_combined.csv", index=False)

    diagnostics = _diagnostic_summary(null, feature, ais_settings)
    diagnostics.to_csv(output_dir / "ais_diagnostics_summary.csv", index=False)
    _plot_type1(calibration, output_dir)
    _plot_power(power, output_dir)
    _plot_marginal_power(power, output_dir)
    _plot_ais_diagnostics(diagnostics, ais_settings, output_dir)
    _write_report(output_dir, calibration, power, diagnostics)

    print(f"Combined {len(null):,} null rows and {len(target):,} power rows.")
    print(f"Saved analysis to {output_dir}")


if __name__ == "__main__":
    main()
