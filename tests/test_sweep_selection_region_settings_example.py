from examples.sweep_selection_region_settings import parse_args, resolve_experiments


def test_quick_preset_uses_small_comparison_matrix():
    args = parse_args(["101", "202"])

    assert resolve_experiments(args) == (
        (1, 2),
        ("baseline", "medium"),
    )


def test_explicit_settings_override_preset():
    args = parse_args(
        [
            "101",
            "--preset",
            "recommended",
            "--k-select",
            "2",
            "5",
            "--rf-config",
            "shallow",
            "flexible",
        ]
    )

    assert resolve_experiments(args) == (
        (2, 5),
        ("shallow", "flexible"),
    )
