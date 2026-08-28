"""Combine pooled same-target versus exact-set summaries across selectors."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["POLARS_MAX_THREADS"] = "1"

import pandas as pd


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_dir", type=Path)
    parser.add_argument(
        "--methods",
        nargs="+",
        default=("shap", "mutual_information"),
    )
    parser.add_argument("--output-dir", type=Path)
    return parser.parse_args(argv)


def _read_method_table(input_dir: Path, method: str, filename: str):
    path = input_dir / method / "pooled_summary" / filename
    if not path.is_file():
        raise FileNotFoundError(f"Missing pooled method result: {path}")
    frame = pd.read_csv(path)
    frame.insert(0, "selection_method", method)
    return frame


def main(argv=None):
    args = parse_args(argv)
    if not args.methods or len(set(args.methods)) != len(args.methods):
        raise ValueError("--methods must contain unique method names.")
    output_dir = args.output_dir or args.input_dir / "pooled_summary"
    output_dir.mkdir(parents=True, exist_ok=True)
    comparison = pd.concat(
        [
            _read_method_table(
                args.input_dir, method, "comparison_summary.csv"
            )
            for method in args.methods
        ],
        ignore_index=True,
    )
    paired = pd.concat(
        [
            _read_method_table(
                args.input_dir, method, "paired_comparison_summary.csv"
            )
            for method in args.methods
        ],
        ignore_index=True,
    )
    comparison.to_csv(output_dir / "method_comparison_summary.csv", index=False)
    paired.to_csv(output_dir / "method_paired_comparison_summary.csv", index=False)
    print(comparison.to_string(index=False))
    print("\nPaired comparisons by selection method:")
    print(paired.to_string(index=False))
    return comparison, paired


if __name__ == "__main__":
    main()
