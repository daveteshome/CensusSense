import pathlib

import pytest

from metadata.metadata_store import load_store

FIXTURES = pathlib.Path(__file__).resolve().parent.parent / "fixtures"
REAL_DATA = pathlib.Path(__file__).resolve().parent.parent.parent / "data"


@pytest.fixture
def store():
    return load_store(
        metadata_path=FIXTURES / "sample_metadata.json",
        geography_path=FIXTURES / "sample_geography.json",
    )


def test_find_by_table_and_column(store):
    entry = store.find("2019_CBG_B01", "B01001e1")
    assert entry is not None
    assert entry.aggregation_type == "SUM"
    assert entry.moe_column == "B01001m1"


def test_find_returns_none_for_unknown_column(store):
    assert store.find("2019_CBG_B01", "NOT_A_REAL_COLUMN") is None


def test_available_years(store):
    assert store.available_years == ["2019", "2020"]


def test_entries_for_year(store):
    entries_2020 = store.entries_for_year("2020")
    assert {e.column for e in entries_2020} == {"B01001e1", "B01003e1"}


def test_resolve_state_by_name_and_abbrev(store):
    by_name = store.resolve_state("Texas")
    by_abbrev = store.resolve_state("tx")
    assert by_name == by_abbrev
    assert by_name["fips"] == "48"


def test_resolve_state_unknown_returns_none(store):
    assert store.resolve_state("Narnia") is None


def test_counties_for_state(store):
    counties = store.counties_for_state("48")
    assert {c["name"] for c in counties} == {"Travis County", "Harris County"}


def test_all_state_names_deduplicated(store):
    names = store.all_state_names()
    assert names.count("Texas") == 1
    assert "California" in names


class TestRealCommittedData:
    """Sanity checks against the actual committed data/metadata.json and
    data/geography_fips.json, built from the live Snowflake trial. Skips
    cleanly if those files haven't been generated yet."""

    @pytest.fixture
    def real_store(self):
        metadata_path = REAL_DATA / "metadata.json"
        geography_path = REAL_DATA / "geography_fips.json"
        if not metadata_path.exists() or not geography_path.exists():
            pytest.skip("data/metadata.json or geography_fips.json not built yet")
        return load_store(metadata_path=metadata_path, geography_path=geography_path)

    def test_both_vintages_present(self, real_store):
        assert set(real_store.available_years) == {"2019", "2020"}

    def test_total_population_column_is_summable(self, real_store):
        entry = real_store.find("2019_CBG_B01", "B01001e1")
        assert entry is not None
        assert entry.aggregation_type == "SUM"

    def test_median_income_column_is_flagged_median(self, real_store):
        entry = real_store.find("2019_CBG_B19", "B19013e1")
        assert entry is not None
        assert entry.aggregation_type == "MEDIAN"

    def test_texas_and_california_resolve(self, real_store):
        assert real_store.resolve_state("Texas")["fips"] == "48"
        assert real_store.resolve_state("California")["fips"] == "06"

    def test_texas_has_254_counties(self, real_store):
        assert len(real_store.counties_for_state("48")) == 254
