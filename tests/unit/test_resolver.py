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
        concept_catalog_path=FIXTURES / "sample_concept_catalog.json",
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


def test_resolve_geography_unknown_state_never_silently_resolved(store):
    # "Narnia" scores in the NEEDS_CONFIRMATION band against this tiny
    # 2-state fixture (California, via fuzz.ratio) -- the safety guarantee
    # that matters is it's never silently RESOLVED as if it were a real,
    # confident match.
    res = resolve_geography("Narnia", None, store)
    assert res.status in ("NOT_FOUND", "NEEDS_CONFIRMATION")


def test_resolve_geography_needs_confirmation_for_mid_confidence_match(store):
    res = resolve_geography("Narnia", None, store)
    assert res.status == "NEEDS_CONFIRMATION"
    assert res.suggested_state_name == "California"


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


def test_resolve_geography_county_without_state_resolves_when_unique(store):
    # Regression test for a real bug found via live testing: a county
    # mentioned with no state used to be silently dropped entirely,
    # falling through to nationwide -- "median household income in Travis
    # County" (no "Texas" in the question) answered with the nationwide
    # figure while the response text still said "Travis County". In this
    # fixture "Travis County" only exists under Texas, so it should now
    # resolve directly instead of going nationwide.
    res = resolve_geography(None, "Travis County", store)
    assert res.status == "RESOLVED"
    assert res.state_fips == "48"
    assert res.county_name == "Travis County"


def test_resolve_geography_county_without_state_not_found_is_not_nationwide(store):
    # A county name that matches nothing anywhere must decline, not
    # silently fall back to a nationwide (unfiltered) answer.
    res = resolve_geography(None, "Nonexistent County", store)
    assert res.status == "NOT_FOUND"
    assert res.state_fips is None


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


def test_resolve_metric_exact_alias_match_short_circuits_fuzzy_ranking(store):
    # "rent" is a curated alias of median_gross_rent in the fixture --
    # an exact match should resolve directly, bypassing the fuzzy ranker.
    res = resolve_metric("rent", store, year="2019")
    assert res.status == "RESOLVED"
    assert res.entry.column == "B25064e1"


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

    def test_total_population_resolves_cleanly(self, real_store):
        res = resolve_metric("total population", real_store, year="2020")
        assert res.status == "RESOLVED"
        assert res.entry.aggregation_type == "SUM"

    def test_total_population_resolves_to_population_not_housing_units(self, real_store):
        # Regression test for a real bug found via live testing: "total
        # population" (an exact, curated alias of the "Population"
        # concept) was scoring a *lower* fuzzy cosine than an unrelated
        # auto-derived concept ("Total Population In Occupied Housing
        # Units By Tenure") whose entire label happens to also contain
        # those same two words -- a short, narrowly-focused document
        # naturally aligns more tightly with a short query than a longer
        # one with more (legitimately useful) alias diversity, even at
        # equal coverage. Fixed with an exact alias/label match
        # short-circuit that bypasses the fuzzy ranker entirely.
        res = resolve_metric("total population", real_store, year="2020")
        assert res.status == "RESOLVED"
        assert "Occupied Housing Units" not in res.entry.description

    def test_bare_population_word_resolves_to_population_not_housing_units(self, real_store):
        res = resolve_metric("population", real_store, year="2020")
        assert res.status == "RESOLVED"
        assert "Occupied Housing Units" not in res.entry.description

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

    def test_typo_d_metric_word_resolves_same_as_correct_spelling(self, real_store):
        # Found via manual testing: "populaton" (missing an 'i') returned
        # NO_MATCH before per-word typo correction was added, unlike
        # geography which already had it. Also a regression test for a
        # follow-on bug: once "population" got an exact-match short-
        # circuit (see test_total_population_resolves_to_population_not_
        # housing_units above), the *typo'd* spelling stopped benefiting
        # from it (it doesn't match anything verbatim) and fell back to
        # the same wrong-concept fuzzy-ranking bug -- fixed by also
        # exact-matching the ranker's typo-corrected text, not just the
        # raw query.
        typo = resolve_metric("populaton", real_store, year="2020")
        clean = resolve_metric("population", real_store, year="2020")
        assert typo.status == clean.status == "RESOLVED"
        assert typo.entry.column == clean.entry.column
        assert "Occupied Housing Units" not in typo.entry.description

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

    def test_bare_income_resolves_to_the_general_headline_concept(self, real_store):
        # Under the old full-corpus TF-IDF matcher, bare "income" tied
        # closely between several similarly-worded income metrics and had
        # to ask for clarification. The curated concept catalog changes
        # this deliberately: "income" is a literal, hand-written alias
        # only on the median-household-income concept (the general,
        # commonly-meant one), not on per-capita/retirement/self-
        # employment income -- so it now resolves cleanly and confidently
        # instead of asking, which is the intended, defensible behavior
        # for the catalog's curated aliases, not a regression.
        res = resolve_metric("income", real_store, year="2020")
        assert res.status == "RESOLVED"
        assert "Household Income" in res.entry.description

    def test_bare_ranking_word_never_silently_resolves_to_an_unrelated_concept(self, real_store):
        # Regression test for a real bug: "least" (a real, correctly-
        # spelled word, not a typo) scores 88.9 against "last" via
        # fuzz.ratio -- above the metric typo-correction threshold -- and
        # "last" appears in several real concept labels ("...In The Last
        # 12 Months..."), so a stray "least" was silently resolving to an
        # unrelated income metric instead of declining. The Gatekeeper is
        # instructed to never pass ranking words into concept (they go in
        # ranking_direction instead), but this guards the resolver itself
        # too, in case one ever leaks through.
        for word in ("least", "most", "highest", "lowest"):
            res = resolve_metric(word, real_store, year="2020")
            assert res.status == "NO_MATCH", f"{word!r} should decline, got {res.status}"

    def test_texas_and_travis_county_resolve(self, real_store):
        res = resolve_geography("Texas", "Travis", real_store)
        assert res.status == "RESOLVED"
        assert res.state_fips == "48"
        assert res.county_name == "Travis County"

    def test_misspelled_state_is_corrected(self, real_store):
        res = resolve_geography("Calfornia", None, real_store)
        assert res.status == "RESOLVED"
        assert res.corrected_state_name == "California"

    def test_short_state_typo_is_corrected(self, real_store):
        # Found via manual testing: "texs" (missing the 'a') sits right at
        # the edge of what the old threshold/scorer combo could catch.
        res = resolve_geography("texs", None, real_store)
        assert res.status == "RESOLVED"
        assert res.corrected_state_name == "Texas"

    @pytest.mark.parametrize(
        "nonsense",
        ["Narnia", "chicago", "miami", "seattle", "atlanta", "boston", "portland", "austin"],
    )
    def test_nonsense_or_city_names_never_silently_resolved_as_a_state(self, real_store, nonsense):
        # Found via manual testing: fuzz.WRatio scored "Narnia" at 72
        # against "California" -- a wholly fictional word "correcting" to
        # a real state is a worse failure than asking or saying not-found.
        # Since the NEEDS_CONFIRMATION tier was added, several of these
        # (Narnia, chicago, atlanta, boston, portland, austin) now land in
        # the "did you mean X?" band rather than a flat rejection -- the
        # guarantee that actually matters is that none of them are ever
        # silently RESOLVED as if confidently matched.
        res = resolve_geography(nonsense, None, real_store)
        assert res.status in ("NOT_FOUND", "NEEDS_CONFIRMATION")

    def test_confirmation_band_never_auto_resolves(self, real_store):
        # Real example from this band: "atlanta" (a real city, not a
        # state) scores in the confirmation range against "Montana".
        res = resolve_geography("atlanta", None, real_store)
        assert res.status == "NEEDS_CONFIRMATION"
        assert res.state_fips is None  # not silently treated as resolved
        assert res.suggested_state_name is not None

    def test_travis_county_without_state_resolves_to_texas_not_nationwide(self, real_store):
        # The exact real bug, reproduced against the real dataset: before
        # the fix, resolve_geography(None, "Travis County", ...) returned
        # a bare RESOLVED with no geography at all (nationwide), silently
        # discarding the county. "Travis County" is nationally unique, so
        # it should now resolve directly to Texas.
        res = resolve_geography(None, "Travis County", real_store)
        assert res.status == "RESOLVED"
        assert res.state_name == "Texas"
        assert res.county_name == "Travis County"

    def test_washington_county_without_state_is_ambiguous_not_a_silent_guess(self, real_store):
        # "Washington County" exists in dozens of states -- resolving it
        # to any single one without the user having said which would be a
        # confident, silent wrong answer. Must ask instead of guess.
        res = resolve_geography(None, "Washington County", real_store)
        assert res.status == "AMBIGUOUS_COUNTY"
        assert len(res.candidate_state_names) > 5
        assert "Alabama" in res.candidate_state_names

    def test_nonsense_county_without_state_is_not_found_not_nationwide(self, real_store):
        res = resolve_geography(None, "Xyzzyplex County", real_store)
        assert res.status == "NOT_FOUND"
        assert res.state_fips is None
