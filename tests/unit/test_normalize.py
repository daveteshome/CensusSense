import json
import pathlib

import pytest

from metadata.normalize import (
    is_estimate_column,
    moe_column_for,
    parse_column,
    to_census_variable_id,
)

DATA_DIR = pathlib.Path(__file__).resolve().parent.parent.parent / "data"


@pytest.mark.parametrize(
    "column,group,suffix,line",
    [
        ("B01001e1", "B01001", "e", 1),
        ("B01001m1", "B01001", "m", 1),
        ("B01001e15", "B01001", "e", 15),
        ("B01001e18", "B01001", "e", 18),
        ("B01001e38", "B01001", "e", 38),
        ("B01002Fe1", "B01002F", "e", 1),
        ("B01002Ge1", "B01002G", "e", 1),
        ("B01001m23", "B01001", "m", 23),
    ],
)
def test_parse_column_real_examples(column, group, suffix, line):
    parsed = parse_column(column)
    assert parsed is not None
    assert parsed.group == group
    assert parsed.suffix == suffix
    assert parsed.line == line


def test_parse_column_rejects_non_matching_names():
    assert parse_column("CENSUS_BLOCK_GROUP") is None
    assert parse_column("Total: Renter-occupied housing units") is None
    assert parse_column("STATE_FIPS") is None


def test_is_estimate_column():
    assert is_estimate_column("B01001e1") is True
    assert is_estimate_column("B01001m1") is False
    assert is_estimate_column("not_a_census_column") is False


def test_moe_column_for():
    assert moe_column_for("B01001e15") == "B01001m15"
    assert moe_column_for("B01002Fe1") == "B01002Fm1"
    # MOE columns themselves have no paired MOE column
    assert moe_column_for("B01001m1") is None


@pytest.mark.parametrize(
    "column,expected_id",
    [
        ("B01001e1", "B01001_001E"),
        ("B01001e15", "B01001_015E"),
        ("B01001m15", "B01001_015M"),
        ("B01002Fe1", "B01002F_001E"),
    ],
)
def test_to_census_variable_id(column, expected_id):
    assert to_census_variable_id(column) == expected_id


@pytest.mark.parametrize("year", ["2019", "2020"])
@pytest.mark.parametrize(
    "column",
    ["B01001e1", "B01001e15", "B01002Fe1", "B01002Ge1"],
)
def test_generated_ids_exist_in_real_census_dictionary(year, column):
    """Round-trip against the real, live-fetched ACS dictionary, not a
    hand-written fixture, so this catches a wrong transform, not just a
    self-consistent one."""
    dict_path = DATA_DIR / f"acs_variables_{year}.json"
    if not dict_path.exists():
        pytest.skip(f"{dict_path} not present; run scripts/fetch_census_dictionary.py")
    variables = json.loads(dict_path.read_text())
    variable_id = to_census_variable_id(column)
    assert variable_id in variables, f"{variable_id} (from {column}) not found in {year} ACS dictionary"
