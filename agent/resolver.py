"""Turns Gatekeeper-extracted phrases (metric, geography, year) into
verified metadata/geography/year, or a clear reason why it can't be
resolved. Nothing downstream of this module ever sees free text again --
sql_builder.py only accepts what comes out of here.
"""

from dataclasses import dataclass, field
from typing import Optional

from rapidfuzz import fuzz, process

from metadata.metadata_store import MetadataStore, MetricEntry
from metadata.metric_ranker import MetricRanker

# Below this fraction of the query's distinctive (non-stopword) terms
# found in a candidate description, treat it as no real match rather
# than a low-confidence guess -- empirically, unanswerable queries like
# "average salary of software engineers" top out around 0.25 coverage,
# while genuine matches ("median household income", "median gross
# rent") reach 1.0.
METRIC_MIN_COVERAGE = 0.4
METRIC_AMBIGUOUS_COSINE_GAP = 0.05
METRIC_TOP_N = 8

# When surfacing an AMBIGUOUS clarification, only offer candidates that
# are genuinely close to the top score -- not a blind top-3 slice, which
# can pad the list with an irrelevant low-score straggler that never
# factored into the ambiguity decision itself.
AMBIGUOUS_DISPLAY_BAND = 0.3

STATE_MATCH_THRESHOLD = 75
COUNTY_MATCH_THRESHOLD = 75

# Race/ethnicity ACS table iterations consistently use "Alone" (White
# Alone, Asian Alone, ...) or "Hispanic"/"Latino" in their title -- unlike
# a generic "(" check, this doesn't also match the unrelated "(In 2020
# Inflation-Adjusted Dollars)" parenthetical present on every dollar
# metric regardless of race breakdown.
_RACE_QUALIFIER_MARKERS = ("alone", "hispanic", "latino", "races")


def _has_race_qualifier(description: str) -> bool:
    lowered = description.lower()
    return any(marker in lowered for marker in _RACE_QUALIFIER_MARKERS)


@dataclass(frozen=True)
class MetricCandidate:
    entry: MetricEntry
    score: float


@dataclass(frozen=True)
class MetricResolution:
    status: str  # "RESOLVED" | "AMBIGUOUS" | "NO_MATCH"
    entry: Optional[MetricEntry] = None
    candidates: list[MetricCandidate] = field(default_factory=list)


@dataclass(frozen=True)
class YearResolution:
    status: str  # "RESOLVED" | "NOT_AVAILABLE"
    year: Optional[str] = None
    was_defaulted: bool = False
    available_years: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class GeographyResolution:
    status: str  # "RESOLVED" | "NOT_FOUND"
    state_fips: Optional[str] = None
    state_name: Optional[str] = None
    county_fips: Optional[str] = None
    county_name: Optional[str] = None
    corrected_state_name: Optional[str] = None  # set when typo-corrected


def resolve_year(requested: Optional[str], store: MetadataStore) -> YearResolution:
    if requested is None:
        return YearResolution(
            status="RESOLVED",
            year=max(store.available_years),
            was_defaulted=True,
            available_years=store.available_years,
        )
    if requested in store.available_years:
        return YearResolution(status="RESOLVED", year=requested, available_years=store.available_years)
    return YearResolution(status="NOT_AVAILABLE", available_years=store.available_years)


def resolve_metric(query: str, store: MetadataStore, year: str) -> MetricResolution:
    entries = store.entries_for_year(year)
    if not entries:
        return MetricResolution(status="NO_MATCH")

    ranker = store.ranker_for_year(year)
    results = ranker.query(query, top_n=METRIC_TOP_N)
    if not results:
        return MetricResolution(status="NO_MATCH")

    if results[0][1] < METRIC_MIN_COVERAGE:
        return MetricResolution(status="NO_MATCH")

    candidates = [
        MetricCandidate(entry=entries[idx], score=cosine)
        for idx, coverage, cosine in results
        if coverage >= METRIC_MIN_COVERAGE
    ]

    # Dedupe by description text: the same metric can legitimately appear
    # more than once (e.g. matched via different phrasing), that's not
    # ambiguity -- ambiguity is multiple *distinct* metrics scoring close
    # to the top (e.g. "income" matching median household income, per
    # capita income, and family income all at once).
    distinct: list[MetricCandidate] = []
    seen = set()
    for c in candidates:
        if c.entry.description not in seen:
            distinct.append(c)
            seen.add(c.entry.description)

    # Race/ethnicity ACS iterations always carry a parenthetical
    # qualifier. If the best match is unqualified, a query that never
    # mentioned a race shouldn't be treated as ambiguous just because a
    # same-metric race iteration also scored close to the top.
    if distinct and not _has_race_qualifier(distinct[0].entry.description):
        unqualified = [c for c in distinct if not _has_race_qualifier(c.entry.description)]
        distinct = unqualified or distinct[:1]

    # Similarly, a table title containing " By <dimension>" (e.g.
    # "Median Household Income ... By Age Of Householder") signals an
    # extra cross-tabulation the query never asked for. Only applies
    # when the top candidate itself has no such breakdown -- tables
    # whose canonical name legitimately includes "By" (e.g. "Median Age
    # By Sex") are unaffected, since the filter only fires when the top
    # candidate lacks "by" too.
    if distinct and " by " not in distinct[0].entry.description.lower():
        no_breakdown = [c for c in distinct if " by " not in c.entry.description.lower()]
        distinct = no_breakdown or distinct[:1]

    if len(distinct) == 1:
        return MetricResolution(status="RESOLVED", entry=distinct[0].entry, candidates=distinct)

    # "Total" is the natural default aggregate row (all sexes/groups
    # combined); don't ask the user to disambiguate "Total" vs "Male" vs
    # "Female" when the query never asked for a specific subgroup and
    # "Total" is already the top-ranked candidate.
    top_last_segment = distinct[0].entry.description.rsplit(":", 1)[-1].strip().lower()
    if top_last_segment == "total":
        return MetricResolution(status="RESOLVED", entry=distinct[0].entry, candidates=distinct)

    if (distinct[0].score - distinct[1].score) < METRIC_AMBIGUOUS_COSINE_GAP:
        top_score = distinct[0].score
        close = [c for c in distinct if top_score - c.score < AMBIGUOUS_DISPLAY_BAND][:3]
        return MetricResolution(status="AMBIGUOUS", candidates=close)

    return MetricResolution(status="RESOLVED", entry=distinct[0].entry, candidates=distinct)


def resolve_geography(
    state_query: Optional[str],
    county_query: Optional[str],
    store: MetadataStore,
) -> GeographyResolution:
    if state_query is None:
        return GeographyResolution(status="RESOLVED")  # nationwide: no geography filter

    direct = store.resolve_state(state_query)
    corrected_name = None
    if direct is not None:
        state_match = direct
    else:
        names = store.all_state_names()
        best = process.extractOne(state_query, names, scorer=fuzz.WRatio)
        if best is None or best[1] < STATE_MATCH_THRESHOLD:
            return GeographyResolution(status="NOT_FOUND")
        corrected_name = best[0]
        state_match = store.resolve_state(corrected_name)

    county_fips = None
    county_name = None
    if county_query:
        counties = store.counties_for_state(state_match["fips"])
        names = [c["name"] for c in counties]
        best = process.extractOne(county_query, names, scorer=fuzz.WRatio)
        if best is None or best[1] < COUNTY_MATCH_THRESHOLD:
            return GeographyResolution(
                status="NOT_FOUND",
                state_fips=state_match["fips"],
                state_name=state_match["name"],
                corrected_state_name=corrected_name,
            )
        matched = next(c for c in counties if c["name"] == best[0])
        county_fips, county_name = matched["fips"], matched["name"]

    return GeographyResolution(
        status="RESOLVED",
        state_fips=state_match["fips"],
        state_name=state_match["name"],
        county_fips=county_fips,
        county_name=county_name,
        corrected_state_name=corrected_name,
    )
