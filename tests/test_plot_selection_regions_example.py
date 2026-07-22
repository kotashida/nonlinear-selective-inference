import argparse

import pytest

from examples.plot_selection_regions import _parse_rf_parameter, parse_args


@pytest.mark.parametrize(
    ("argument", "expected"),
    [
        ("n_estimators=100", ("n_estimators", 100)),
        ("max_depth=null", ("max_depth", None)),
        ("bootstrap=false", ("bootstrap", False)),
        ("criterion=squared_error", ("criterion", "squared_error")),
    ],
)
def test_parse_rf_parameter_preserves_value_types(argument, expected):
    assert _parse_rf_parameter(argument) == expected


def test_parse_rf_parameter_requires_name_value_syntax():
    with pytest.raises(argparse.ArgumentTypeError, match="NAME=VALUE"):
        _parse_rf_parameter("n_estimators")


def test_parse_args_accepts_repeated_rf_parameters():
    args = parse_args(
        [
            "101",
            "202",
            "--rf-param",
            "n_estimators=100",
            "--rf-param",
            "max_depth=8",
        ]
    )

    assert args.dataset_seeds == [101, 202]
    assert dict(args.rf_param) == {"n_estimators": 100, "max_depth": 8}
