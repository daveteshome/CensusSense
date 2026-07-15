"""Canned, deterministic copy for every degrade-gracefully case the
pipeline can land in, plus a simple per-session rate limiter. Kept
separate from the LLM layer so these messages are never themselves
subject to hallucination -- they're the fallback *for* LLM failures too.
"""

import time
from typing import Any


def out_of_scope_message() -> str:
    return (
        "I can only answer questions about US population and demographics using "
        "Census Bureau data (2019/2020 ACS 5-year estimates and 2020 decennial "
        "counts). Try asking about population, income, age, housing, or similar "
        "topics for a US state, county, or the nation as a whole."
    )


def inappropriate_message() -> str:
    return "I can't help with that. I'm only able to answer factual questions about US Census demographic data."


def unsupported_year_message(available_years: list[str]) -> str:
    years = " or ".join(available_years)
    return f'This dataset only contains data for {years}. Could you ask about one of those years instead?'


def no_match_metric_message(query: str) -> str:
    return (
        f'I couldn\'t find a Census metric matching "{query}" in this dataset. It may not be '
        "something the Census Bureau tracks, or it may need to be phrased differently. Try "
        'a term like "population", "median household income", or "median age".'
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


def county_not_found_message(county_query: str, state_name: str) -> str:
    return f'I couldn\'t find a county called "{county_query}" in {state_name}. Could you check the spelling?'


def empty_result_message() -> str:
    return "I couldn't find any Census records matching your request. Please verify the geography or try a broader region."


def clarification_message(reason: str) -> str:
    return f"Your question has some conflicting details ({reason}). Could you clarify which one you meant?"


def unhandled_error_message() -> str:
    return "Something went wrong on my end while processing that. Please try rephrasing your question, or try again in a moment."


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
            return out_of_scope_message()
        if gatekeeper.status == "NEEDS_CLARIFICATION":
            return clarification_message(gatekeeper.clarification_reason or "some details conflict")

    year_resolution = state.get("year_resolution")
    if year_resolution is not None and year_resolution.status == "NOT_AVAILABLE":
        return unsupported_year_message(year_resolution.available_years)

    geography_resolution = state.get("geography_resolution")
    if geography_resolution is not None and geography_resolution.status == "NOT_FOUND":
        if geography_resolution.state_name and gatekeeper is not None and gatekeeper.county:
            return county_not_found_message(gatekeeper.county, geography_resolution.state_name)
        query = gatekeeper.state if gatekeeper is not None else ""
        return geography_not_found_message(query)

    metric_resolution = state.get("metric_resolution")
    if metric_resolution is not None:
        if metric_resolution.status == "AMBIGUOUS":
            return ambiguous_metric_message(metric_resolution.candidates)
        if metric_resolution.status == "NO_MATCH":
            query = gatekeeper.metric_phrase if gatekeeper is not None else ""
            return no_match_metric_message(query)

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
