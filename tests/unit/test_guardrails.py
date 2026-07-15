import pytest

from agent.guardrails import (
    GLOSSARY,
    RateLimiter,
    build_message,
    glossary_message,
    greeting_message,
    is_greeting,
    out_of_scope_message,
)
from agent.gatekeeper import GatekeeperResult
from agent.resolver import GeographyResolution, MetricCandidate, MetricResolution, YearResolution
from metadata.metadata_store import MetricEntry


@pytest.mark.parametrize("text", ["hello", "Hi", "  hey  ", "Hello!", "thanks", "thank you", "Good morning"])
def test_is_greeting_matches_common_greetings(text):
    assert is_greeting(text) is True


@pytest.mark.parametrize("text", ["population of texas", "hi, what's the population of Texas?", ""])
def test_is_greeting_rejects_real_questions(text):
    assert is_greeting(text) is False


def test_out_of_scope_message_without_note_is_generic():
    msg = out_of_scope_message()
    assert "only answer questions about US population" in msg


def test_out_of_scope_message_with_note_is_short_not_the_full_boilerplate():
    msg = out_of_scope_message("This dataset has no city-level geography.")
    assert msg.startswith("This dataset has no city-level geography.")
    # The full generic paragraph should NOT also be appended -- that's
    # exactly the "robotic repeated disclaimer" behavior being fixed.
    assert "only answer questions about US population" not in msg


def _gatekeeper(**overrides):
    defaults = dict(
        status="IN_SCOPE", concept="", state="", county="", year="",
        clarification_reason="", scope_note="", glossary_term="",
    )
    defaults.update(overrides)
    return GatekeeperResult(**defaults)


@pytest.mark.parametrize("term", sorted(GLOSSARY.keys()))
def test_glossary_message_covers_every_key(term):
    assert glossary_message(term) == GLOSSARY[term]


def test_glossary_message_is_case_and_whitespace_insensitive():
    assert glossary_message("  Country  ") == GLOSSARY["country"]


def test_glossary_message_miss_returns_none():
    assert glossary_message("not a real term") is None


def test_build_message_glossary_term_wins_over_generic_out_of_scope():
    state = {
        "question": "what is a country",
        "gatekeeper": _gatekeeper(status="OUT_OF_SCOPE", glossary_term="country"),
    }
    assert build_message(state) == GLOSSARY["country"]


def test_build_message_unknown_glossary_term_falls_back_to_out_of_scope():
    # Defensive: if the Gatekeeper ever returns a term outside the known
    # list (it's instructed not to, but never trust the LLM blindly),
    # fall back to the normal decline rather than crashing or showing
    # nothing.
    state = {
        "question": "what is a widget",
        "gatekeeper": _gatekeeper(status="OUT_OF_SCOPE", glossary_term="widget"),
    }
    msg = build_message(state)
    assert msg == out_of_scope_message()


def test_build_message_greeting_gets_friendly_welcome():
    state = {"question": "hello", "gatekeeper": _gatekeeper(status="OUT_OF_SCOPE")}
    assert build_message(state) == greeting_message()


def test_build_message_out_of_scope_with_scope_note_is_specific():
    state = {
        "question": "what is the most populated city",
        "gatekeeper": _gatekeeper(status="OUT_OF_SCOPE", scope_note="This dataset has no city-level geography."),
    }
    msg = build_message(state)
    assert msg.startswith("This dataset has no city-level geography.")


def test_build_message_plain_out_of_scope_not_treated_as_greeting():
    state = {"question": "who won the super bowl?", "gatekeeper": _gatekeeper(status="OUT_OF_SCOPE")}
    msg = build_message(state)
    assert msg == out_of_scope_message()


def test_build_message_unhandled_error_wins_over_everything():
    state = {"error": "boom", "question": "hello"}
    assert "went wrong" in build_message(state).lower()


def test_build_message_geography_needs_confirmation_asks_did_you_mean():
    state = {
        "question": "population of begas",
        "gatekeeper": _gatekeeper(status="IN_SCOPE", concept="population", state="begas"),
        "year_resolution": YearResolution(status="RESOLVED", year="2020"),
        "geography_resolution": GeographyResolution(status="NEEDS_CONFIRMATION", suggested_state_name="Texas"),
    }
    msg = build_message(state)
    assert "begas" in msg
    assert "Did you mean Texas?" in msg


def test_build_message_ambiguous_county_asks_which_state():
    state = {
        "question": "population of Washington County",
        "gatekeeper": _gatekeeper(status="IN_SCOPE", concept="population", county="Washington County"),
        "year_resolution": YearResolution(status="RESOLVED", year="2020"),
        "geography_resolution": GeographyResolution(
            status="AMBIGUOUS_COUNTY", candidate_state_names=["Alabama", "Arkansas", "Colorado"]
        ),
    }
    msg = build_message(state)
    assert "Washington County" in msg
    assert "Alabama" in msg
    assert "which state" in msg.lower()


def test_build_message_county_not_found_without_state_does_not_claim_nationwide():
    # Regression test: before the fix this path fell through to
    # geography_not_found_message(gatekeeper.state), which is empty here
    # ('""') and would have produced a confusing 'called ""' message.
    state = {
        "question": "population of Xyzzyplex County",
        "gatekeeper": _gatekeeper(status="IN_SCOPE", concept="population", county="Xyzzyplex County"),
        "year_resolution": YearResolution(status="RESOLVED", year="2020"),
        "geography_resolution": GeographyResolution(status="NOT_FOUND"),
    }
    msg = build_message(state)
    assert "Xyzzyplex County" in msg
    assert '""' not in msg


def test_build_message_unsupported_year():
    state = {
        "question": "population in 2025",
        "gatekeeper": _gatekeeper(status="IN_SCOPE"),
        "year_resolution": YearResolution(status="NOT_AVAILABLE", available_years=["2019", "2020"]),
    }
    msg = build_message(state)
    assert "2019" in msg and "2020" in msg


def test_build_message_empty_result():
    entry = MetricEntry(
        table="t", column="c", year="2020", description="d", table_topics=None,
        universe=None, aggregation_type="SUM", moe_column=None,
    )
    state = {
        "question": "population of texas",
        "gatekeeper": _gatekeeper(status="IN_SCOPE"),
        "year_resolution": YearResolution(status="RESOLVED", year="2020"),
        "geography_resolution": GeographyResolution(status="RESOLVED", state_fips="48", state_name="Texas"),
        "metric_resolution": MetricResolution(status="RESOLVED", entry=entry),
        "execution_result": type("R", (), {"rows": [{"VALUE": None}]})(),
    }
    assert "couldn't find any Census records" in build_message(state)


def test_build_message_ranking_unsupported_metric():
    entry = MetricEntry(
        table="t", column="c", year="2020", description="Median Household Income: Total",
        table_topics=None, universe=None, aggregation_type="MEDIAN", moe_column=None,
    )
    state = {
        "question": "most expensive state by median income",
        "gatekeeper": _gatekeeper(status="IN_SCOPE", operation="RANKING", sort="DESC", limit=1, ranking_scope="STATE"),
        "year_resolution": YearResolution(status="RESOLVED", year="2020"),
        "geography_resolution": GeographyResolution(status="RESOLVED"),
        "metric_resolution": MetricResolution(status="RESOLVED", entry=entry),
        "ranking_unsupported_metric": True,
    }
    msg = build_message(state)
    assert "Ranking isn't supported" in msg
    assert "Median Household Income" in msg


def test_rate_limiter_allows_up_to_max_then_blocks():
    limiter = RateLimiter(max_requests=2, window_seconds=60)
    assert limiter.allow() is True
    assert limiter.allow() is True
    assert limiter.allow() is False
