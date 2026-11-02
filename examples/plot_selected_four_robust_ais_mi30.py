"""Create reference-format four-AIS-method galleries for both robust settings."""
import json
from pathlib import Path

import pandas as pd

import plot_combined_ais_analysis as plotting


BASE = Path(__file__).resolve().parents[1] / "outputs/selection_method_validation/exact_set"
SETTINGS = {
    "robust_draw_floor": "Robust draw floor AIS",
    "robust_floor_multimodal_denominator": "Robust floor + multimodal denominator AIS",
}
METHODS = {
    "shap": "SHAP",
    "lime": "LIME (40 explanations)",
    "spline_screening": "Spline screening (AIS)",
    "mutual_information": "Mutual information (30 neighbors)",
}


def main():
    for setting, label in SETTINGS.items():
        source = BASE / f"selected_five_methods_{setting}_ais_exact_mi30"
        output = source / "four_methods_plots"
        data = pd.read_csv(source / "unified_tests.csv")
        data = data[data.selection_method.isin(METHODS)].copy()
        if len(data) != 4000 or set(data.selection_method) != set(METHODS):
            raise ValueError(f"Incomplete four-method dataset: {source}")
        counts = data.groupby(["selection_method", "scope", "signal_strength"], dropna=False).size()
        if len(counts) != 20 or not counts.eq(200).all():
            raise ValueError(f"Unexpected per-condition counts: {source}")
        output.mkdir(exist_ok=True)
        data.to_csv(output / "unified_tests.csv", index=False)
        for filename in ("power_summary.csv", "calibration_summary.csv"):
            summary = pd.read_csv(source / filename)
            summary[summary.method.isin(METHODS)].to_csv(output / filename, index=False)
        manifest = json.loads((source / "source_manifest.json").read_text(encoding="utf-8"))
        used = {name: manifest[name] for name in data.source_run.unique()}
        (output / "source_manifest.json").write_text(json.dumps(used, indent=2), encoding="utf-8")
        selected = json.loads((source / "selected_configuration.json").read_text(encoding="utf-8"))
        selected.update(methods=list(METHODS), plot_source=str(source),
                        plot_reference="selected_four_methods_combined_ais_mi30",
                        plot_note="Four AIS methods from the completed five-method run; no new simulations.")
        (output / "selected_configuration.json").write_text(json.dumps(selected, indent=2), encoding="utf-8")
        plotting.SOURCE = plotting.OUT = output
        plotting.METHODS = METHODS.copy()
        plotting.COLORS = dict(zip(METHODS, ["#2674a6", "#d58122", "#258568", "#9256a0"]))
        plotting.DATA_FILENAME = "unified_tests.csv"
        plotting.REPORT_LABEL = label
        plotting.REPORT_INTRO = None
        # Null failures are valid outcomes for these new strategies; the
        # explicit counts above enforce complete data independently.
        plotting.ALLOW_INCOMPLETE = True
        plotting.FIGURES.clear()
        plotting.main()
        with (output / "README.md").open("a", encoding="utf-8") as file:
            file.write("\n\n## Plot configuration\n\nUses the same seven plot types, method order, colors, panel layouts, axes, and confidence intervals as `selected_four_methods_combined_ais_mi30`. The four AIS methods are taken from the completed parent dataset; the parent five-method gallery includes the exact-spline reference. No new simulations were run.\n\nRegenerate both galleries with `python examples/plot_selected_four_robust_ais_mi30.py`.\n")
        gallery = output / "index.html"
        html = gallery.read_text(encoding="utf-8")
        html = html.replace("<title>Combined AIS analysis</title>", f"<title>{label} analysis</title>")
        html = html.replace("<h1>Combined AIS analysis</h1>", f"<h1>{label}: four-method analysis</h1>")
        gallery.write_text(html, encoding="utf-8")


if __name__ == "__main__":
    main()
