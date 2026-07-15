"""Turns Gatekeeper-extracted phrases (metric, geography, year) into
verified metadata/geography/year, or a clear reason why it can't be
resolved. Nothing downstream of this module ever sees free text again --
sql_builder.py only accepts what comes out of here.
"""

from dataclasses import dataclass, field
from typing import Optional

from rapidfuzz import fuzz, process

from metadata.metadata_store import MetadataStore, MetricEntry

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

# NOTE: the old per-word rapidfuzz typo-correction / race-qualifier /
# breakdown-filtering logic that used to live here was retired along with
# the 8,276-entry raw-corpus matching it existed to patch around -- see
# metadata/build_concept_catalog.py. resolve_metric now matches against a
# small (~200-entry), pre-curated concept catalog instead.

# Defense in depth: the Gatekeeper is instructed to never put a ranking/
# superlative word in `concept` (that's ranking_direction's job), but if
# one ever leaks through anyway, metric_ranker.py's typo-corrector can
# still turn it into a real, wrong concept match -- verified concretely:
# "least" (not a typo) scores 88.9 against "last" via fuzz.ratio, above
# the 82 typo-correction threshold, and "last" appears in several real
# concept labels ("...In The Last 12 Months..."), so "least" alone was
# resolving to an unrelated income metric instead of declining. Stripped
# here, before matching, rather than raising the shared typo threshold
# (which would just re-break genuine typo correction elsewhere -- this
# collision is between two real, correctly-spelled English words, not
# fixable by threshold-tuning).
_RANKING_OPERATION_WORDS = {
    "most", "least", "highest", "lowest", "greatest", "smallest", "biggest", "top", "bottom",
}


def _strip_ranking_words(query: str) -> str:
    words = [w for w in query.split() if w.lower() not in _RANKING_OPERATION_WORDS]
    return " ".join(words)

# Tuned empirically, not guessed -- and the scorer choice matters as much
# as the threshold. fuzz.WRatio's partial/token-reorder matching produces
# false positives even for a wholly fictional word ("Narnia" scored 72
# against "California"). Plain fuzz.ratio (pure Levenshtein similarity)
# gives a clean separation instead: every tested false-positive risk
# (Miami, Atlanta, Portland, Narnia, Boston, ...) tops out at 62.5, while
# every genuine state typo ("texs"->Texas, "Marylnd"->Maryland, ...)
# scores 65-94. 65 sits in that gap.
STATE_MATCH_THRESHOLD = 65
COUNTY_MATCH_THRESHOLD = 75

# Below STATE_MATCH_THRESHOLD but above this, the best candidate is
# plausible enough to ask about rather than silently reject outright --
# "did you mean California?" rather than either guessing or a dead-end
# "not found". Below this floor, nothing plausible enough to suggest
# (empirically, real-city-name false positives like Miami/DC and
# Seattle/Delaware score 40-44, well clear of this floor; genuine typos
# scored 65+, so this band only ever catches the ambiguous middle).
STATE_CONFIRMATION_THRESHOLD = 45

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
    status: str  # "RESOLVED" | "NOT_FOUND" | "NEEDS_CONFIRMATION"
    state_fips: Optional[str] = None
    state_name: Optional[str] = None
    county_fips: Optional[str] = None
    county_name: Optional[str] = None
    corrected_state_name: Optional[str] = None  # set when typo-corrected
    suggested_state_name: Optional[str] = None  # set only when NEEDS_CONFIRMATION


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
    concepts = store.concepts_for_year(year)
    if not concepts:
        return MetricResolution(status="NO_MATCH")

    query = _strip_ranking_words(query)
    if not query.strip():
        return MetricResolution(status="NO_MATCH")

    # An exact (case-insensitive) match against a concept's own label or
    # a curated alias is the strongest possible signal -- stronger than
    # the fuzzy ranker's cosine score, which can be misled by document
    # length: verified a real case where "total population" (an exact,
    # curated alias of the "Population" concept) scored a *lower* cosine
    # than an unrelated, auto-derived concept ("Total Population In
    # Occupied Housing Units By Tenure") whose entire label happens to
    # also contain those same two words -- a short, narrowly-focused
    # document naturally aligns more tightly with a short query than a
    # longer document with more (legitimately useful) alias diversity,
    # even at equal coverage. Skip the fuzzy ranker entirely when exactly
    # one concept matches this way.
    def _exact_match(text: str):
        normalized = text.strip().lower()
        return [
            c for c in concepts
            if normalized == c.label.lower() or normalized in (a.lower() for a in c.aliases)
        ]

    ranker = store.concept_ranker_for_year(year)

    exact_matches = _exact_match(query)
    if len(exact_matches) != 1:
        # Also try after typo correction -- a bare misspelled metric word
        # ("populaton") won't match anything verbatim, but should still
        # get the same strong exact-match treatment once corrected,
        # rather than falling through to fuzzy cosine ranking, which (at
        # equal coverage) can favor a short, narrowly-focused unrelated
        # concept over the correct one -- the same class of bug the raw
        # exact-match check above exists to avoid.
        exact_matches = _exact_match(ranker.correct_text(query))
    if len(exact_matches) == 1:
        entry = exact_matches[0].entry
        return MetricResolution(status="RESOLVED", entry=entry, candidates=[MetricCandidate(entry=entry, score=1.0)])

    results = ranker.query(query, top_n=METRIC_TOP_N)
    if not results or results[0][1] < METRIC_MIN_COVERAGE:
        return MetricResolution(status="NO_MATCH")

    # Dedupe by concept id, not description text: the concept catalog is
    # already pre-folded to one canonical row per real-world concept, so
    # the only way the same concept appears twice here is via a different
    # matching alias -- not genuine ambiguity. Genuine ambiguity is two
    # *distinct* concepts both scoring close to the top (e.g. "income"
    # matching both median household income and per capita income).
    distinct: list[MetricCandidate] = []
    seen_concepts = set()
    for idx, coverage, cosine in results:
        if coverage < METRIC_MIN_COVERAGE:
            continue
        concept = concepts[idx]
        if concept.concept in seen_concepts:
            continue
        seen_concepts.add(concept.concept)
        distinct.append(MetricCandidate(entry=concept.entry, score=cosine))

    if len(distinct) == 1:
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
        best = process.extractOne(state_query, names, scorer=fuzz.ratio)
        if best is None or best[1] < STATE_CONFIRMATION_THRESHOLD:
            return GeographyResolution(status="NOT_FOUND")
        if best[1] < STATE_MATCH_THRESHOLD:
            # Plausible but not confident enough to silently auto-correct
            # -- ask rather than guess or dead-end.
            return GeographyResolution(status="NEEDS_CONFIRMATION", suggested_state_name=best[0])
        corrected_name = best[0]
        state_match = store.resolve_state(corrected_name)

    county_fips = None
    county_name = None
    if county_query:
        counties = store.counties_for_state(state_match["fips"])
        names = [c["name"] for c in counties]
        # WRatio (not plain ratio) here deliberately: county names almost
        # always carry a "County"/"Parish"/"Borough" suffix, and WRatio's
        # partial-match handling correctly scores "Travis" highly against
        # "Travis County" (90.0) where plain ratio penalizes the length
        # difference (63.2, a false rejection). The state-level false-
        # positive problem above doesn't apply here: it was specifically
        # about WRatio being too lenient for bare state names with no
        # common suffix pattern to legitimately match against.
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
