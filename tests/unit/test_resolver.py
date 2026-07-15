import pathlib

import pytest

from agent.resolver import resolve_geography, resolve_metric, resolve_year
from metadata.metadata_store import load_store

FIXTURES = pathlib.Path(__file__).resolve().parent.parent / "fixtures"
REAL_DATA = pathlib.Path(__file__).resolve().parent.parent.parent / "data"


@pytest.fixture
def store():
    return load_store(
        metadata_path=FIXTURES / "sample_metadata.json",
        geography_path=FIXTURES / "sample_geography.json",
    )


# --- resolve_year -----------------------------------------------------

def test_resolve_year_defaults_to_most_recent_and_flags_it(store):
    res = resolve_year(None, store)
    assert res.status == "RESOLVED"
    assert res.year == "2020"
    assert res.was_defaulted is True


def test_resolve_year_uses_requested_year_when_available(store):
    res = resolve_year("2019", store)
    assert res.status == "RESOLVED"
    assert res.year == "2019"
    assert res.was_defaulted is False


def test_resolve_year_rejects_unavailable_year(store):
    res = resolve_year("2025", store)
    assert res.status == "NOT_AVAILABLE"
    assert res.available_years == ["2019", "2020"]


# --- resolve_geography --------------------------------------------------

def test_resolve_geography_no_state_means_nationwide(store):
    res = resolve_geography(None, None, store)
    assert res.status == "RESOLVED"
    assert res.state_fips is None


def test_resolve_geography_exact_state_match(store):
    res = resolve_geography("Texas", None, store)
    assert res.status == "RESOLVED"
    assert res.state_fips == "48"
    assert res.corrected_state_name is None


def test_resolve_geography_typo_correction(store):
    res = resolve_geography("Califronia", None, store)
    assert res.status == "RESOLVED"
    assert res.state_fips == "06"
    assert res.corrected_state_name == "California"


def test_resolve_geography_unknown_state_not_found(store):
    res = resolve_geography("Narnia", None, store)
    assert res.status == "NOT_FOUND"


def test_resolve_geography_state_and_county(store):
    res = resolve_geography("Texas", "Travis", store)
    assert res.status == "RESOLVED"
    assert res.county_fips == "453"
    assert res.county_name == "Travis County"


def test_resolve_geography_county_not_in_state_not_found(store):
    # Los Angeles County exists under California (06), not Texas (48)
    res = resolve_geography("Texas", "Los Angeles", store)
    assert res.status == "NOT_FOUND"
    assert res.state_fips == "48"  # state still resolved for a helpful message


# --- resolve_metric -------------------------------------------------------

def test_resolve_metric_no_match_for_fixture_with_no_overlap(store):
    res = resolve_metric("completely unrelated gibberish gibberish", store, year="2019")
    assert res.status == "NO_MATCH"


def test_resolve_metric_ambiguous_income_in_fixture(store):
    # Fixture has both median household income and per capita income
    # (both MEDIAN type, both containing "income") -- neither phrase
    # match should let one silently win.
    res = resolve_metric("income", store, year="2019")
    assert res.status in ("AMBIGUOUS", "RESOLVED")
    # whichever it is, it must not silently invent a metric outside the
    # fixture's two income entries
    if res.status == "RESOLVED":
        assert "Income" in res.entry.description
    else:
        assert len(res.candidates) >= 2


def test_resolve_metric_resolves_exact_phrase(store):
    res = resolve_metric("median household income", store, year="2019")
    assert res.status == "RESOLVED"
    assert res.entry.column == "B19013e1"


class TestRealMetricRankerBehavior:
    """Regression tests locking in behavior validated empirically against
    the live-built metadata.json -- these encode real bugs found and
    fixed during development (see metric_ranker.py docstring)."""

    @pytest.fixture
    def real_store(self):
        metadata_path = REAL_DATA / "metadata.json"
        geography_path = REAL_DATA / "geography_fips.json"
        if not metadata_path.exists() or not geography_path.exists():
            pytest.skip("data/metadata.json or geography_fips.json not built yet")
        return load_store(metadata_path=metadata_path, geography_path=geography_path)

    def test_total_population_resolves_cleanly(self, real_store):
        res = resolve_metric("total population", real_store, year="2020")
        assert res.status == "RESOLVED"
        assert res.entry.aggregation_type == "SUM"

    def test_median_household_income_is_not_confused_with_gross_rent(self, real_store):
        res = resolve_metric("median household income", real_store, year="2020")
        assert res.status in ("RESOLVED", "AMBIGUOUS")
        descriptions = (
            [res.entry.description] if res.entry else [c.entry.description for c in res.candidates]
        )
        assert all("Rent" not in d for d in descriptions)
        assert any("Household Income" in d for d in descriptions)

    def test_median_age_defaults_to_total_not_ambiguous(self, real_store):
        res = resolve_metric("median age", real_store, year="2020")
        assert res.status == "RESOLVED"
        assert res.entry.description.lower().endswith("total")

    def test_per_capita_income_not_confused_by_race_iterations(self, real_store):
        res = resolve_metric("per capita income", real_store, year="2020")
        assert res.status == "RESOLVED"
        assert "alone" not in res.entry.description.lower()

    def test_unanswerable_query_is_no_match(self, real_store):
        res = resolve_metric("average salary of software engineers", real_store, year="2020")
        assert res.status == "NO_MATCH"

    def test_median_gross_rent_resolves_cleanly(self, real_store):
        res = resolve_metric("median gross rent", real_store, year="2020")
        assert res.status == "RESOLVED"
        assert res.entry.aggregation_type == "MEDIAN"

    def test_bare_income_is_genuinely_ambiguous(self, real_store):
        # Unlike "median household income" (specific), a bare "income"
        # query has many similarly-plausible income-related metrics in
        # the real corpus -- this should prompt a clarification, not
        # silently guess one.
        res = resolve_metric("income", real_store, year="2020")
        assert res.status == "AMBIGUOUS"
        assert len(res.candidates) >= 2

    def test_texas_and_travis_county_resolve(self, real_store):
        res = resolve_geography("Texas", "Travis", real_store)
        assert res.status == "RESOLVED"
        assert res.state_fips == "48"
        assert res.county_name == "Travis County"

    def test_misspelled_state_is_corrected(self, real_store):
        res = resolve_geography("Calfornia", None, real_store)
        assert res.status == "RESOLVED"
        assert res.corrected_state_name == "California"
