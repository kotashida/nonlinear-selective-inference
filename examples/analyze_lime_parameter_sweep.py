"""Compare completed LIME configurations without changing inference."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

VARIANTS = {
    "baseline": (1.0, None, 10),
    "ridge_3": (3.0, None, 10),
    "ridge_10": (10.0, None, 10),
    "kernel_sqrt_p": (1.0, 4.47213595499958, 10),
    "explanations_20": (1.0, None, 20),
}


ADDITIONAL_VARIANTS = {
    "ridge_30": (30.0, None, 10),
    "kernel_1p5_sqrt_p": (1.0, 6.708203932499369, 10),
    "explanations_40": (1.0, None, 40),
}


FOCUSED_200_VARIANTS = {
    "explanations_40": (1.0, None, 40),
    "explanations_60": (1.0, None, 60),
    "explanations_80": (1.0, None, 80),
    "explanations_40_ridge_3": (3.0, None, 40),
    "explanations_40_kernel_sqrt_p": (1.0, 4.47213595499958, 40),
}


REFINEMENT_200_VARIANTS = {
    "explanations_30": (1.0, None, 30),
    "explanations_35": (1.0, None, 35),
    "explanations_45": (1.0, None, 45),
    "explanations_40_ridge_0p3": (0.3, None, 40),
    "explanations_40_kernel_0p6_sqrt_p": (1.0, 2.6832815729997477, 40),
}


AIS_200_VARIANTS = {
    "ais_baseline": (12800, 5, 80),
    "ais_budget_51200": (51200, 5, 80),
    "ais_pilot_samples_320": (12800, 5, 320),
    "ais_pilot_rounds_10": (12800, 10, 80),
    "ais_combined": (51200, 10, 320),
}


def analyze(root, *, additional=False, focused_200=False, refinement_200=False, ais_200=False):
    if sum([additional, focused_200, refinement_200, ais_200]) > 1:
        raise ValueError("Select only one sweep family")
    variants = FOCUSED_200_VARIANTS if focused_200 else (ADDITIONAL_VARIANTS if additional else VARIANTS)
    if refinement_200:
        variants = REFINEMENT_200_VARIANTS
    if ais_200:
        variants = {name: (1.0, None, 40) for name in AIS_200_VARIANTS}
    iterations = 200 if focused_200 or refinement_200 or ais_200 else 400
    calibration, power, raw_null, raw_power = [], [], [], []
    for variant, (ridge, width, explanations) in variants.items():
        run = root / variant
        pooled = run / "pooled_summary"
        settings = json.loads((pooled / "validation_settings.json").read_text())
        expected = {"ridge_alpha": ridge, "kernel_width": width,
                    "num_explanations": explanations}
        if settings["lime_params"] != expected:
            raise ValueError(f"Wrong LIME parameters in {variant}")
        budget, rounds, samples = AIS_200_VARIANTS[variant] if ais_200 else (12800, 5, 80)
        for key, value in {"methods": ["lime"], "n_shards": 32,
                           "n_null_iters": iterations, "n_power_iters": iterations,
                           "max_final_samples": budget, "pilot_iters": rounds,
                           "pilot_samples": samples, "final_batch_size": 800,
                           "min_denominator_ess": 80, "min_tail_ess": 15,
                           "stop_when_ess_met": True,
                           "inference_method": "ais", "designs": ["baseline"],
                           "design_seed": 314159, "rf_jobs": 1,
                           "shard_root_seeds": list(range(20260908, 20260940)),
                           "minimum_conditional_power": 0.8,
                           "comparison_events": ["exact_set"],
                           "signal_strengths": [0.3, 0.5, 0.75, 1.0]}.items():
            if settings[key] != value:
                raise ValueError(f"Unexpected {key} in {variant}")
        calibration.append(pd.read_csv(pooled / "calibration_decisions.csv").assign(variant=variant))
        power.append(pd.read_csv(pooled / "power_decisions.csv").assign(variant=variant))
        for shard in range(32):
            base = run / f"shard_{shard}" / "lime" / "baseline"
            raw_null.append(pd.read_csv(base / "null_fresh/p_value_results.csv").assign(variant=variant, shard=shard))
            for beta in settings["signal_strengths"]:
                folder = base / f"power_feature_0_beta_{str(beta).replace('.', 'p')}"
                raw_power.append(pd.read_csv(folder / "feature_results.csv").assign(variant=variant, shard=shard, signal_strength=beta))
    c, p, n, f = [pd.concat(frames, ignore_index=True) for frames in (calibration, power, raw_null, raw_power)]
    if len(n) != iterations * len(variants) or len(f) != 4 * iterations * len(variants):
        raise ValueError(f"Expected {iterations * len(variants)} null tests and {4 * iterations * len(variants)} power tests")
    if not n.groupby("variant").size().eq(iterations).all():
        raise ValueError("Unexpected per-variant null counts")
    if not f.groupby(["variant", "signal_strength"]).size().eq(iterations).all():
        raise ValueError("Unexpected per-configuration power counts")
    for frame, keys in [(n, ["variant", "shard", "iteration"]), (f, ["variant", "shard", "iteration", "signal_strength"])]:
        if frame.duplicated(keys).any():
            raise ValueError("Duplicate trial identifiers")
    out = root / "analysis"
    out.mkdir(exist_ok=True)
    for frame, name in [(c, "calibration"), (p, "power"), (n, "null_combined"), (f, "power_features_combined")]:
        frame.to_csv(out / f"{name}.csv", index=False)
    rows = []
    for (variant, beta), group in f.groupby(["variant", "signal_strength"]):
        signal = group[group.is_signal]
        rows.append(dict(variant=variant, signal_strength=beta,
                         signal_failure_rate=signal.failed.mean(),
                         mean_final_proposals=group.proposals.mean(),
                         n_failed=int(group.failed.sum())))
    diagnostics = pd.DataFrame(rows)
    diagnostics.to_csv(out / "diagnostics.csv", index=False)
    if ais_200:
        import numpy as np
        retention = []
        for scope, frame, keys in [
            ("null", n, ["shard", "iteration"]),
            ("power", f, ["shard", "iteration", "signal_strength"])]:
            baseline = frame[frame.variant.eq("ais_baseline")]
            for variant in AIS_200_VARIANTS:
                candidate = frame[frame.variant.eq(variant)]
                pair = baseline.merge(candidate, on=keys, suffixes=("_baseline", "_candidate"), validate="one_to_one")
                if len(pair) != len(baseline):
                    raise ValueError("Unmatched AIS comparison trials")
                if not np.array_equal(pair.feature_baseline, pair.feature_candidate):
                    raise ValueError("LIME selected targets changed across AIS configurations")
                ok = ~pair.failed_baseline
                same = np.isclose(pair.p_value_baseline, pair.p_value_candidate, rtol=1e-12, atol=1e-14, equal_nan=True)
                if variant == "ais_budget_51200" and (pair.loc[ok, "failed_candidate"].any() or not same[ok].all()):
                    raise ValueError("Budget-only change did not preserve baseline successful p-values")
                retention.append(dict(scope=scope, variant=variant,
                    n_trials=len(pair), recovered_failures=int((~ok & ~pair.failed_candidate).sum()),
                    new_failures=int((ok & pair.failed_candidate).sum()),
                    changed_successful_pvalues=int((ok & ~same).sum())))
        pd.DataFrame(retention).to_csv(out / "baseline_retention.csv", index=False)
        diagnostics["pilot_draws_per_test"] = diagnostics.variant.map(
            {name: rounds*samples for name, (_, rounds, samples) in AIS_200_VARIANTS.items()})
        diagnostics["mean_total_proposals_including_pilot"] = diagnostics.mean_final_proposals + diagnostics.pilot_draws_per_test
        diagnostics.to_csv(out / "diagnostics.csv", index=False)
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    for variant in variants:
        q = p[p.variant.eq(variant)].sort_values("signal_strength")
        d = diagnostics[diagnostics.variant.eq(variant)].sort_values("signal_strength")
        for ax, values in [(axes[0, 0], q.conditional_power_resolution_lower_bound),
                           (axes[0, 1], q.marginal_detection_power),
                           (axes[1, 0], d.signal_failure_rate),
                           (axes[1, 1], d.mean_final_proposals)]:
            ax.plot(q.signal_strength, values, marker="o", label=variant)
    for ax, title in zip(axes.flat, ["Conditional power (failures = no detection)",
                                   "Marginal detection power", "Signal-target failure rate",
                                   "Mean final AIS proposals (excludes pilot)"]):
        ax.set(title=title, xlabel="Signal strength", xticks=[0.3, 0.5, 0.75, 1.0])
        ax.grid(alpha=0.2)
    for ax in [axes[0, 0], axes[0, 1], axes[1, 0]]:
        ax.set_ylim(-0.02, 1.02)
    axes[0, 0].axhline(0.8, color="0.4", linestyle="--", label="80% target")
    axes[0, 0].legend(fontsize=8)
    fig.tight_layout()
    for extension in ("png", "pdf"):
        fig.savefig(out / f"lime_parameter_comparison.{extension}", dpi=180)
    plt.close(fig)
    # Evaluate the 80% target including failed signal tests as no detection.
    from numpy import sqrt
    targets = p.copy()
    trials = targets.n_signal_targets
    rate = targets.conditional_power_resolution_lower_bound
    z = 1.959963984540054
    center = (rate + z*z/(2*trials)) / (1 + z*z/trials)
    half = z * sqrt(rate*(1-rate)/trials + z*z/(4*trials*trials)) / (1 + z*z/trials)
    targets["failure_adjusted_ci_lower"] = center - half
    targets["failure_adjusted_ci_upper"] = center + half
    targets["point_estimate_at_least_80pct"] = rate.ge(0.8)
    targets["ci_lower_at_least_80pct"] = targets.failure_adjusted_ci_lower.ge(0.8)
    targets.to_csv(out / "power_target_80pct.csv", index=False)
    print(c[c.alpha.eq(0.05)].to_string(index=False))
    print(p.to_string(index=False))
    print(f"Validated {len(variants)} configurations; saved comparison to {out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_dir", type=Path)
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--additional", action="store_true")
    group.add_argument("--focused-200", action="store_true")
    group.add_argument("--refinement-200", action="store_true")
    group.add_argument("--ais-200", action="store_true")
    args = parser.parse_args()
    analyze(args.input_dir, additional=args.additional, focused_200=args.focused_200, refinement_200=args.refinement_200, ais_200=args.ais_200)
