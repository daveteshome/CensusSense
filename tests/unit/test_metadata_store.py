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
        concept_catalog_path=FIXTURES / "sample_concept_catalog.json",
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


def test_concepts_for_year_resolves_source_pointers(store):
    concepts_2019 = store.concepts_for_year("2019")
    by_id = {c.concept: c for c in concepts_2019}
    assert by_id["population"].entry.column == "B01003e1"
    assert by_id["median_household_income"].entry.column == "B19013e1"


def test_concepts_for_year_only_includes_concepts_with_a_source_for_that_year(store):
    # The fixture catalog's income/rent concepts only have a 2019 source.
    concepts_2020 = store.concepts_for_year("2020")
    ids_2020 = {c.concept for c in concepts_2020}
    assert "population" in ids_2020
    assert "median_household_income" not in ids_2020


def test_concepts_for_year_skips_a_pointer_that_does_not_resolve(store, tmp_path):
    import json

    catalog = json.loads((FIXTURES / "sample_concept_catalog.json").read_text())
    catalog.append(
        {
            "concept": "drifted",
            "label": "Drifted Concept",
            "aliases": ["drifted concept"],
            "sources": {"2019": {"table": "2019_CBG_B01", "column": "NOT_A_REAL_COLUMN"}},
        }
    )
    drifted_path = tmp_path / "drifted_catalog.json"
    drifted_path.write_text(json.dumps(catalog))
    drifted_store = load_store(
        metadata_path=FIXTURES / "sample_metadata.json",
        geography_path=FIXTURES / "sample_geography.json",
        concept_catalog_path=drifted_path,
    )
    ids = {c.concept for c in drifted_store.concepts_for_year("2019")}
    assert "drifted" not in ids  # skipped, not a crash


def test_concept_ranker_for_year_matches_a_known_alias(store):
    ranker = store.concept_ranker_for_year("2019")
    results = ranker.query("population", top_n=1)
    assert results  # at least one match with non-zero cosine


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


def test_state_name_by_fips(store):
    assert store.state_name_by_fips("48") == "Texas"
    assert store.state_name_by_fips("06") == "California"


def test_state_name_by_fips_unknown_returns_none(store):
    assert store.state_name_by_fips("99") is None


def test_county_name_by_fips(store):
    assert store.county_name_by_fips("48", "453") == "Travis County"
    assert store.county_name_by_fips("48", "201") == "Harris County"


def test_county_name_by_fips_unknown_returns_none(store):
    assert store.county_name_by_fips("48", "999") is None


class TestRealCommittedData:
    """Sanity checks against the actual committed data/metadata.json and
    data/geography_fips.json, built from the live Snowflake trial. Skips
    cleanly if those files haven't been generated yet."""

    @pytest.fixture
    def real_store(self):
        metadata_path = REAL_DATA / "metadata.json"
        geography_path = REAL_DATA / "geography_fips.json"
        concept_catalog_path = REAL_DATA / "concept_catalog.json"
        if not metadata_path.exists() or not geography_path.exists():
            pytest.skip("data/metadata.json or geography_fips.json not built yet")
        if not concept_catalog_path.exists():
            pytest.skip("data/concept_catalog.json not built yet")
        return load_store(
            metadata_path=metadata_path,
            geography_path=geography_path,
            concept_catalog_path=concept_catalog_path,
        )

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

    def test_population_concept_prefers_redistricting_for_2020(self, real_store):
        # Verified override: the real decennial count, not the ACS 5-year
        # estimate, since a genuine population figure should be exact for
        # 2020, not an estimate -- see metadata/build_concept_catalog.py.
        concepts_2020 = {c.concept: c for c in real_store.concepts_for_year("2020")}
        assert concepts_2020["b01003"].entry.table == "2020_REDISTRICTING_CBG_DATA"
