"""Canned, deterministic copy for every degrade-gracefully case the
pipeline can land in, plus a simple per-session rate limiter. Kept
separate from the LLM layer so these messages are never themselves
subject to hallucination -- they're the fallback *for* LLM failures too.
"""

import re
import time
from typing import Any

_GREETING_RE = re.compile(
    r"^\s*(hi|hello|hey|yo|sup|good\s*(morning|afternoon|evening)|greetings|howdy|thanks|thank\s*you)\W*\s*$",
    re.IGNORECASE,
)


def is_greeting(text: str) -> bool:
    return bool(_GREETING_RE.match(text))


def greeting_message() -> str:
    return (
        "Hi! I can answer questions about US population and demographics using "
        "Census Bureau data (2019/2020 ACS 5-year estimates and 2020 decennial "
        'counts). Try asking something like "What\'s the population of Texas?" '
        'or "What\'s the median household income in Travis County?"'
    )


def out_of_scope_message(scope_note: str = "") -> str:
    if scope_note:
        # A specific reason is already a complete, helpful answer on its
        # own -- re-appending the full boilerplate on top of it every
        # time is what made repeated declines feel like a canned loop.
        return f"{scope_note} Feel free to ask about a specific US state, county, or metric instead."
    return (
        "I can only answer questions about US population and demographics using "
        "Census Bureau data (2019/2020 ACS 5-year estimates and 2020 decennial "
        "counts). Try asking about population, income, age, housing, or similar "
        "topics for a US state, county, or the nation as a whole."
    )


# Hand-written, reviewed once, and testable -- never LLM-improvised.
# Each entry follows Acknowledge -> Contextualize -> Redirect: give the
# real definition, tie it to what this agent actually covers, then nudge
# toward a real query. Deliberately bounded to terms either specific to
# this dataset (where an ungrounded LLM answer could describe our own
# implementation inconsistently) or basic geography vocabulary this
# agent's own responses use ("the nation as a whole") and should be able
# to clarify. Terms outside this list fall back to the normal decline --
# an honest, documented limitation rather than open-ended generation.
GLOSSARY: dict[str, str] = {
    "block group": (
        "A Census Block Group is the smallest geography this dataset reports data for, "
        "typically a few hundred to a few thousand people. I combine block groups to "
        "answer state and county level questions. Want a number for a specific state or county?"
    ),
    "decennial census": (
        "The decennial census is the official US population count conducted every 10 years "
        "(most recently 2020) -- an exact headcount, not a statistical estimate. I have the "
        "2020 decennial counts alongside 2019/2020 ACS estimates. Want a population figure "
        "for a specific state or county?"
    ),
    "acs": (
        "ACS stands for the American Community Survey: the Census Bureau's ongoing survey "
        "(not a full headcount) used to estimate detailed statistics like income, housing, "
        "and age between decennial censuses. I have the 2019 and 2020 5-year ACS estimates. "
        "Want to look one up for a state or county?"
    ),
    "margin of error": (
        "Margin of error reflects the statistical uncertainty in an ACS estimate (since it's "
        "based on a survey sample, not a full count) -- smaller geographies tend to have "
        "wider margins. I don't currently surface margin of error alongside answers, only "
        "the estimate itself. Want me to look one up?"
    ),
    "fips code": (
        "A FIPS code is the standard numeric ID the Census Bureau uses for a state or county "
        "(e.g. Texas is 48). I use these internally to filter data by geography. Just give me "
        "a state or county name and I'll handle the rest."
    ),
    "median vs average": (
        "A median is the middle value when everything is sorted low to high; an average "
        "(mean) is the sum divided by the count -- medians resist being skewed by outliers, "
        "which is why the Census Bureau reports median household income rather than average. "
        "Want a median or total figure for a specific area?"
    ),
    "country": (
        "A country (or nation) is a sovereign state with its own government. In my data, "
        "the United States is the top-level geography -- I can give you nationwide totals, "
        "or narrow down to a specific state or county. Which would you like?"
    ),
    "nation": (
        "\"The nation\" and \"country\" mean the same thing here: the United States as a "
        "whole, the top-level geography I cover above state and county. Want a nationwide "
        "figure, or one for a specific state or county?"
    ),
    "state": (
        "A US state is the geography level between the nation and counties (there are 50, "
        "plus DC and territories). I can look up most metrics at the state level directly. "
        "Which state are you interested in?"
    ),
    "county": (
        "A county is a subdivision of a US state (Texas alone has 254). I can look up most "
        "metrics at the county level. Which state and county are you interested in?"
    ),
}


def glossary_message(term: str) -> str | None:
    return GLOSSARY.get(term.strip().lower())


def inappropriate_message() -> str:
    return "I can't help with that. I'm only able to answer factual questions about US Census demographic data."


def unsupported_year_message(available_years: list[str]) -> str:
    years = " or ".join(available_years)
    return f'This dataset only contains data for {years}. Could you ask about one of those years instead?'


def no_match_metric_message(query: str) -> str:
    return (
        f'I couldn\'t match "{query}" to a Census metric this version supports. This version '
        "covers headline Census metrics (population, income, age, housing, employment, and "
        'similar topics), not fine-grained demographic breakdowns. Try a term like '
        '"population", "median household income", or "median age".'
    )


def ambiguous_metric_message(candidates: list[Any]) -> str:
    options = []
    seen = set()
    for c in candidates:
        label = c.entry.description.split(":")[0].strip()
        if label not in seen:
            options.append(label)
            seen.add(label)
    joined = ", ".join(options)
    return f"There are a few different metrics that could match your question: {joined}. Which one did you mean?"


def geography_not_found_message(query: str) -> str:
    return (
        f'I couldn\'t find a US state called "{query}" in this dataset. This dataset only '
        "supports state and county level geography (not individual cities). Could you check "
        "the spelling or try a state/county name?"
    )


def geography_confirmation_message(query: str, suggested_name: str) -> str:
    return f'I couldn\'t find a US state called "{query}". Did you mean {suggested_name}?'


def county_not_found_message(county_query: str, state_name: str) -> str:
    return f'I couldn\'t find a county called "{county_query}" in {state_name}. Could you check the spelling?'


def county_ambiguous_message(county_query: str, candidate_state_names: list[str]) -> str:
    if len(candidate_state_names) > 6:
        shown = candidate_state_names[:6]
        options = ", ".join(shown) + f", and {len(candidate_state_names) - 6} other states"
    else:
        options = ", ".join(candidate_state_names)
    return (
        f'There\'s a "{county_query}" in more than one state ({options}). '
        "Which state did you mean?"
    )


def empty_result_message() -> str:
    return "I couldn't find any Census records matching your request. Please verify the geography or try a broader region."


def clarification_message(reason: str) -> str:
    return f"Your question has some conflicting details ({reason}). Could you clarify which one you meant?"


def unhandled_error_message() -> str:
    return "Something went wrong on my end while processing that. Please try rephrasing your question, or try again in a moment."


def ranking_unsupported_metric_message(metric_description: str) -> str:
    label = metric_description.split(":")[0].strip()
    return (
        f'Ranking isn\'t supported for "{label}" since it\'s a median/rate-based metric -- '
        "comparing many already-approximate numbers against each other could give a "
        "misleading \"winner\". I can look it up for a specific state or county instead, "
        "or rank a count-based metric like population."
    )


def build_message(state: dict) -> str:
    """Inspects pipeline state to pick the single most relevant guardrail
    message. Order matters: hard errors first, then the earliest stage
    that didn't resolve cleanly."""
    if state.get("error"):
        return unhandled_error_message()

    gatekeeper = state.get("gatekeeper")
    if gatekeeper is not None:
        if gatekeeper.status == "INAPPROPRIATE":
            return inappropriate_message()
        if gatekeeper.status == "OUT_OF_SCOPE":
            if gatekeeper.glossary_term:
                definition = glossary_message(gatekeeper.glossary_term)
                if definition:
                    return definition
            if not gatekeeper.scope_note and is_greeting(state.get("question", "")):
                return greeting_message()
            return out_of_scope_message(gatekeeper.scope_note)
        if gatekeeper.status == "NEEDS_CLARIFICATION":
            return clarification_message(gatekeeper.clarification_reason or "some details conflict")

    year_resolution = state.get("year_resolution")
    if year_resolution is not None and year_resolution.status == "NOT_AVAILABLE":
        return unsupported_year_message(year_resolution.available_years)

    geography_resolution = state.get("geography_resolution")
    if geography_resolution is not None and geography_resolution.status == "NEEDS_CONFIRMATION":
        query = gatekeeper.state if gatekeeper is not None else ""
        return geography_confirmation_message(query, geography_resolution.suggested_state_name)
    if geography_resolution is not None and geography_resolution.status == "AMBIGUOUS_COUNTY":
        query = gatekeeper.county if gatekeeper is not None else ""
        return county_ambiguous_message(query, geography_resolution.candidate_state_names)
    if geography_resolution is not None and geography_resolution.status == "NOT_FOUND":
        if geography_resolution.state_name and gatekeeper is not None and gatekeeper.county:
            return county_not_found_message(gatekeeper.county, geography_resolution.state_name)
        if gatekeeper is not None and gatekeeper.county and not gatekeeper.state:
            return county_not_found_message(gatekeeper.county, "any state")
        query = gatekeeper.state if gatekeeper is not None else ""
        return geography_not_found_message(query)

    metric_resolution = state.get("metric_resolution")
    if metric_resolution is not None:
        if metric_resolution.status == "AMBIGUOUS":
            return ambiguous_metric_message(metric_resolution.candidates)
        if metric_resolution.status == "NO_MATCH":
            query = gatekeeper.concept if gatekeeper is not None else ""
            return no_match_metric_message(query)

    if state.get("ranking_unsupported_metric") and metric_resolution is not None:
        return ranking_unsupported_metric_message(metric_resolution.entry.description)

    execution_result = state.get("execution_result")
    if execution_result is not None:
        rows = execution_result.rows
        if not rows or rows[0].get("VALUE") is None:
            return empty_result_message()

    return unhandled_error_message()


class RateLimiter:
    """Simple per-session token bucket -- bounds LLM/Snowflake cost
    exposure on a public, potentially unauthenticated-beyond-shared-
    password URL. One instance per Streamlit session, not shared
    globally, so one heavy user can't starve another's session."""

    def __init__(self, max_requests: int = 20, window_seconds: float = 60.0):
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self._timestamps: list[float] = []

    def allow(self) -> bool:
        now = time.monotonic()
        cutoff = now - self.window_seconds
        self._timestamps = [t for t in self._timestamps if t > cutoff]
        if len(self._timestamps) >= self.max_requests:
            return False
        self._timestamps.append(now)
        return True
