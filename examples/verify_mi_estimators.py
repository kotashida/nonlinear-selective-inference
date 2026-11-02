"""Check the two estimator runs, including raw counts and selector manifests."""
import csv
import json
from pathlib import Path
import sys


def check_settings(path, expected):
    settings = json.loads(path.read_text())
    for key, value in expected.items():
        if settings.get(key) != value:
            raise ValueError(f"{path}: unexpected {key}: {settings.get(key)!r}; expected {value!r}")
    return settings


def verify(output_base):
    expected = json.loads(Path(__file__).with_name("mi_neighbors_sweep_expected.json").read_text())
    expected["mi_neighbors"] = 30  # Inactive for the two alternatives.
    for estimator in ("histogram", "kde"):
        expected["mi_params"] = dict(estimator=estimator, n_bins=5, bandwidth=1.0)
        root = Path(output_base) / estimator
        if len(list(root.glob("shard_*/validation_settings.json"))) != 32:
            raise ValueError(f"Expected exactly 32 completed shards: {root}")
        for i in range(32):
            count = 7 if i < 8 else 6
            shard = root / f"shard_{i}"
            check_settings(shard / "validation_settings.json", {
                **expected, "seed": 20260908 + i, "n_null_iters": count, "n_power_iters": count})
            for condition in ("null_fresh", "power_feature_0_beta_0p3", "power_feature_0_beta_0p5",
                              "power_feature_0_beta_0p75", "power_feature_0_beta_1p0"):
                directory = shard / "mutual_information" / "baseline" / condition
                raw_settings = check_settings(directory / "settings.json", {
                    "inference_method": "ais", "max_final_samples": 51200,
                    "pilot_iters": 10, "pilot_samples": 320, "final_batch_size": 800,
                    "min_denominator_ess": 80.0, "min_tail_ess": 15.0,
                    "stop_when_ess_met": True, "n_samples": 100, "n_features": 20,
                    "k_select": 2, "n_iters": count, "selection_events": ["exact_set"]})
                selector = raw_settings["selector_settings"]
                if selector.get("estimator") != estimator:
                    raise ValueError(f"Wrong raw selector: {directory}")
                if estimator == "histogram" and selector.get("n_bins") != 5:
                    raise ValueError(f"Wrong bin count: {directory}")
                if estimator == "kde" and (selector.get("bandwidth_multiplier") != 1.0 or
                                            selector.get("density_evaluation") != "leave_one_out"):
                    raise ValueError(f"Wrong KDE settings: {directory}")
                filename = "p_value_results.csv" if condition == "null_fresh" else "target_results.csv"
                with (directory / filename).open(newline="") as file:
                    rows = list(csv.DictReader(file))
                if len(rows) != count or len({row["iteration"] for row in rows}) != count:
                    raise ValueError(f"Wrong raw count or duplicate iterations: {directory}")
                if any(row["selection_event"] != "exact_set" for row in rows):
                    raise ValueError(f"Wrong selection event: {directory}")
        check_settings(root / "pooled_summary/validation_settings.json", {
            **expected, "n_shards": 32, "n_null_iters": 200, "n_power_iters": 200,
            "shard_root_seeds": list(range(20260908, 20260940))})
        for name in ("calibration_decisions.csv", "power_decisions.csv"):
            print((root / "pooled_summary" / name).read_text())
        print(f"Verified {estimator}: 200 null + 800 power trials, combined AIS.")


if __name__ == "__main__":
    verify(sys.argv[1])
