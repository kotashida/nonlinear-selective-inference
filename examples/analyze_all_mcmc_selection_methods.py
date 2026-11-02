"""Combine SHAP, mutual-information, and spline-screening MCMC validations.

The source experiments remain immutable. The collection records the source run
and source file for every observation, validates matching configurations and
shared replicate seeds, then creates unified tables and comparison plots.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["POLARS_MAX_THREADS"] = "1"

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from examples.analyze_selection_method_validation_snapshot import (
    configuration_inventory,
    load_observations,
    paired_method_comparisons,
    plot_method_comparisons,
    plot_null_diagnostics,
    plot_overview,
    plot_shards_and_coverage,
    read_json,
    summarize_configurations,
    summarize_shards,
)


METHODS = ("shap", "mutual_information", "spline_screening")
COMPATIBLE_FIELDS = (
    "preset",
    "designs",
    "n_null_iters",
    "n_power_iters",
    "max_final_samples",
    "mcmc_steps",
    "fixed_auxiliary_values",
    "signal_strengths",
    "signal_positions",
    "minimum_conditional_power",
    "minimum_calibration_iterations",
    "minimum_signal_targets",
    "primary_selection_event",
    "comparison_events",
    "alpha_levels",
    "design_seed",
)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shap-source", required=True, type=Path)
    parser.add_argument("--mutual-information-source", required=True, type=Path)
    parser.add_argument("--spline-screening-source", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    return parser.parse_args(argv)


def validation_settings(root: Path) -> dict:
    pooled = root / "pooled_summary" / "validation_settings.json"
    if pooled.is_file():
        return read_json(pooled)
    shard = root / "shard_0" / "validation_settings.json"
    if not shard.is_file():
        raise FileNotFoundError(f"No validation settings under {root}")
    return read_json(shard)


def validate_compatible(settings_by_method: dict[str, dict]) -> None:
    reference_method = METHODS[0]
    reference = settings_by_method[reference_method]
    for method in METHODS[1:]:
        candidate = settings_by_method[method]
        mismatches = [field for field in COMPATIBLE_FIELDS if candidate.get(field) != reference.get(field)]
        if mismatches:
            raise ValueError(f"{method} has incompatible settings: {', '.join(mismatches)}")


def validate_scenario_settings(root: Path, inventory: pd.DataFrame, method: str) -> None:
    expected_iters = {"null_calibration": 150, "power": 150}
    for item in inventory.itertuples(index=False):
        settings_path = (root / item.source_path).with_name("settings.json")
        settings = read_json(settings_path)
        failures = []
        if settings.get("selection_method") != method:
            failures.append("selection_method")
        if settings.get("inference_method") != "mcmc_rank":
            failures.append("inference_method")
        if settings.get("mcmc_steps") != 8:
            failures.append("mcmc_steps")
        if settings.get("max_final_samples") != 100:
            failures.append("max_final_samples")
        if settings.get("selection_events") != ["exact_set"]:
            failures.append("selection_events")
        if settings.get("n_iters") != expected_iters[item.analysis_type]:
            failures.append("n_iters")
        if failures:
            raise ValueError(f"{settings_path} differs on: {', '.join(failures)}")


def validate_shared_replicates(observations: pd.DataFrame) -> None:
    keys = ["analysis_type", "signal_strength", "shard_index", "iteration", "selection_event"]
    grouped = observations.groupby(keys, dropna=False)
    method_counts = grouped["selection_method"].nunique()
    if not method_counts.eq(len(METHODS)).all():
        raise ValueError("Not every replicate is present for all three methods.")
    seed_counts = grouped["target_seed"].nunique(dropna=False)
    if not seed_counts.eq(1).all():
        raise ValueError("Target seeds are not shared across methods.")
    auxiliary_range = grouped["auxiliary_u"].agg(lambda values: float(values.max() - values.min()))
    if not np.allclose(auxiliary_range, 0.0, equal_nan=True):
        raise ValueError("Auxiliary target randomization is not shared across methods.")


def write_readme(
    output_dir: Path,
    sources: dict[str, Path],
    observations: pd.DataFrame,
    summary: pd.DataFrame,
) -> None:
    lines = [
        "# All-method MCMC validation collection",
        "",
        "This folder compares SHAP, mutual information, and spline screening with",
        "the same `mcmc_rank` selective-inference sampler (100 replicas x 8 steps).",
        "All source experiments use exact-set conditioning, the same root/design",
        "seeds, and 1,500 iterations per configuration.",
        "",
        f"Unified observation rows: {len(observations):,}",
        "Configuration-shards present: 150/150",
        "Failed tests: 0",
        "",
        "## Source runs",
        "",
    ]
    for method in METHODS:
        lines.append(f"- `{method}`: `{sources[method]}`")
    lines += [
        "",
        "## Deliverables",
        "",
        "- `unified_mcmc_dataset.csv`: target-level null and power observations.",
        "- `configuration_summary.csv`: pooled calibration, power, and MCMC diagnostics.",
        "- `shard_summary.csv`: per-shard stability metrics.",
        "- `paired_method_comparisons.csv`: paired comparisons on shared replicates.",
        "- `configuration_inventory.csv`: expected files, presence, and provenance.",
        "- `source_manifest.csv`: method-to-source-run mapping.",
        "- `comprehensive_mcmc_analysis.pdf`: consolidated four-page plot report.",
        "- `plots/*.png`: high-resolution copies of individual figures.",
        "",
        "## Configuration summary",
        "",
        "```",
        summary.to_string(index=False),
        "```",
        "",
    ]
    (output_dir / "README.md").write_text("\n".join(lines), encoding="utf-8")


def main(argv=None):
    args = parse_args(argv)
    sources = {
        "shap": args.shap_source.resolve(),
        "mutual_information": args.mutual_information_source.resolve(),
        "spline_screening": args.spline_screening_source.resolve(),
    }
    output_dir = args.output_dir.resolve()
    settings_by_method = {method: validation_settings(root) for method, root in sources.items()}
    validate_compatible(settings_by_method)

    inventories = []
    observation_frames = []
    for method, root in sources.items():
        inventory = configuration_inventory(root, settings_by_method[method])
        inventory = inventory[inventory["selection_method"] == method].copy()
        inventory.insert(0, "source_run", root.name)
        inventory.insert(1, "source_root", str(root))
        if len(inventory) != 50 or not inventory["present"].all():
            missing = inventory.loc[~inventory["present"], "source_path"].tolist()
            raise ValueError(f"{method} inventory is incomplete: {missing}")
        validate_scenario_settings(root, inventory, method)
        observations = load_observations(root, inventory, settings_by_method[method])
        observations = observations[observations["selection_method"] == method].copy()
        observations.rename(columns={"run_name": "source_run"}, inplace=True)
        observations.insert(0, "collection", output_dir.name)
        inventories.append(inventory)
        observation_frames.append(observations)

    inventory = pd.concat(inventories, ignore_index=True)
    observations = pd.concat(observation_frames, ignore_index=True, sort=False)
    validate_shared_replicates(observations)
    summary = summarize_configurations(observations, inventory)
    shard_summary = summarize_shards(observations)
    comparisons = paired_method_comparisons(observations)

    output_dir.mkdir(parents=True, exist_ok=True)
    plot_dir = output_dir / "plots"
    plot_dir.mkdir(exist_ok=True)
    observations.to_csv(output_dir / "unified_mcmc_dataset.csv", index=False)
    summary.to_csv(output_dir / "configuration_summary.csv", index=False)
    shard_summary.to_csv(output_dir / "shard_summary.csv", index=False)
    comparisons.to_csv(output_dir / "paired_method_comparisons.csv", index=False)
    inventory.to_csv(output_dir / "configuration_inventory.csv", index=False)
    pd.DataFrame(
        [
            {
                "selection_method": method,
                "sampling_method": "mcmc_rank",
                "source_run": root.name,
                "source_root": str(root),
            }
            for method, root in sources.items()
        ]
    ).to_csv(output_dir / "source_manifest.csv", index=False)
    manifest = {
        "collection": output_dir.name,
        "methods": list(METHODS),
        "sampling_method": "mcmc_rank",
        "mcmc_replicates": 100,
        "mcmc_steps": 8,
        "selection_event": "exact_set",
        "iterations_per_configuration": 1500,
        "source_runs": {method: str(root) for method, root in sources.items()},
    }
    (output_dir / "collection_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )

    coverage = "COMPLETE ALL-MCMC COMPARISON | null 30/30 configuration-shards | power 120/120"
    figures = [
        ("01_overview", plot_overview(summary, coverage)),
        ("02_null_and_mcmc_diagnostics", plot_null_diagnostics(observations, summary, coverage)),
        ("03_shards_and_coverage", plot_shards_and_coverage(shard_summary, inventory, coverage)),
        ("04_paired_method_comparisons", plot_method_comparisons(comparisons, summary, coverage)),
    ]
    with PdfPages(output_dir / "comprehensive_mcmc_analysis.pdf") as pdf:
        for name, figure in figures:
            figure.savefig(plot_dir / f"{name}.png", dpi=220, bbox_inches="tight")
            pdf.savefig(figure, bbox_inches="tight")
            plt.close(figure)

    write_readme(output_dir, sources, observations, summary)
    print(f"Wrote {len(observations):,} rows to {output_dir}")
    print(coverage)


if __name__ == "__main__":
    main()
