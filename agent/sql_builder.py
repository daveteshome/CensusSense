"""Deterministic SQL construction. Only ever accepts already-verified
metadata (a MetricEntry straight out of MetadataStore) plus FIPS codes
that were themselves resolved against our own geography crosswalk —
never raw natural language. This is what makes hallucinated SQL
structurally impossible: there is no code path from user text to a
table/column name that doesn't pass through the metadata allow-list.

Column identifiers in this dataset are mixed-case (e.g. "B19013e1", not
"B19013E1") and Snowflake folds unquoted identifiers to uppercase, so
every identifier here is double-quoted to preserve exact case — verified
against the live schema; an unquoted reference to a real column silently
fails to resolve.
"""

import re
from dataclasses import dataclass
from typing import Optional

from metadata.metadata_store import MetricEntry

_FIPS_RE = re.compile(r"^\d+$")

APPROXIMATE_AGGREGATE_CAVEAT = (
    "This dataset only reports this metric per Census block group. The value "
    "below is an unweighted average across block groups, not a recomputed "
    "true median/rate for the requested area, so treat it as an approximation."
)


class SqlBuildError(Exception):
    pass


@dataclass(frozen=True)
class BuiltQuery:
    sql: str
    is_approximate: bool
    caveat: Optional[str]


def build_query(
    entry: MetricEntry,
    state_fips: Optional[str] = None,
    county_fips: Optional[str] = None,
) -> BuiltQuery:
    if state_fips is not None and not _FIPS_RE.match(state_fips):
        raise SqlBuildError(f"invalid state_fips: {state_fips!r}")
    if county_fips is not None and not _FIPS_RE.match(county_fips):
        raise SqlBuildError(f"invalid county_fips: {county_fips!r}")
    if county_fips is not None and state_fips is None:
        raise SqlBuildError("county_fips given without state_fips")

    if entry.aggregation_type == "SUM":
        agg_func = "SUM"
        is_approximate = False
        caveat = None
    else:
        # MEDIAN / RATE: never silently SUM a non-additive metric. AVG is
        # itself an approximation (unweighted across block groups), so it
        # is always surfaced to the user via the caveat, never presented
        # as an exact figure.
        agg_func = "AVG"
        is_approximate = True
        caveat = APPROXIMATE_AGGREGATE_CAVEAT

    where_clauses = []
    if state_fips is not None:
        where_clauses.append(f"SUBSTR(\"CENSUS_BLOCK_GROUP\", 1, 2) = '{state_fips}'")
    if county_fips is not None:
        where_clauses.append(f"SUBSTR(\"CENSUS_BLOCK_GROUP\", 3, 3) = '{county_fips}'")
    where_sql = f" WHERE {' AND '.join(where_clauses)}" if where_clauses else ""

    sql = (
        f'SELECT {agg_func}("{entry.column}") AS VALUE, COUNT(*) AS BLOCK_GROUP_COUNT '
        f'FROM "{entry.table}"'
        f"{where_sql} "
        f"LIMIT 1"
    )
    return BuiltQuery(sql=sql, is_approximate=is_approximate, caveat=caveat)


RANKING_SORTS = ("ASC", "DESC")
RANKING_SCOPES = ("STATE", "COUNTY")

# Verified against the real geography crosswalk: 56 distinct state-level
# FIPS groups (50 states + DC + 5 territories), so this must exceed that
# or "all states ordered by population" would silently truncate. A
# nationwide county ranking (~3,143 real counties) legitimately exceeds
# this cap -- that's surfaced to the user as a disclosed truncation note
# (see agent/pipeline.py's respond_ranking_node), not a silent drop.
MAX_RANKING_LIMIT = 60


@dataclass(frozen=True)
class BuiltRankingQuery:
    sql: str


def build_ranking_query(
    entry: MetricEntry,
    sort: str,
    scope: str,
    limit: int,
    state_fips: Optional[str] = None,
) -> BuiltRankingQuery:
    """One query, not fifty (or a loop): GROUP BY the FIPS prefix already
    embedded in CENSUS_BLOCK_GROUP, ORDER BY the aggregate, LIMIT N --
    finds the top/bottom N geographies without a join or a loop. N is a
    validated, capped parameter (the Gatekeeper LLM only ever picks a
    sort direction and a row count -- it never sees or influences the
    table/column/SQL itself). Deliberately restricted to SUM-type
    metrics: ranking already-approximate AVG-of-block-group-medians
    numbers against each other would compound one approximation on top
    of another in a potentially misleading way, so that gate is enforced
    here, not just hoped for upstream."""
    if entry.aggregation_type != "SUM":
        raise SqlBuildError(
            f"ranking is only supported for summable metrics, got aggregation_type={entry.aggregation_type!r}"
        )
    if sort not in RANKING_SORTS:
        raise SqlBuildError(f"invalid ranking sort: {sort!r}")
    if scope not in RANKING_SCOPES:
        raise SqlBuildError(f"invalid ranking scope: {scope!r}")
    # bool is a subclass of int in Python -- guard against True/False
    # slipping through as 1/0.
    if isinstance(limit, bool) or not isinstance(limit, int) or not (1 <= limit <= MAX_RANKING_LIMIT):
        raise SqlBuildError(f"ranking limit out of range: {limit!r}")
    if state_fips is not None and not _FIPS_RE.match(state_fips):
        raise SqlBuildError(f"invalid state_fips: {state_fips!r}")

    if scope == "STATE":
        group_expr = 'SUBSTR("CENSUS_BLOCK_GROUP", 1, 2)'
        where_sql = ""
    else:  # COUNTY -- state_fips is optional: None means rank nationwide
        # across every US county, not just within one named state.
        group_expr = 'SUBSTR("CENSUS_BLOCK_GROUP", 1, 5)'
        where_sql = "" if state_fips is None else f" WHERE SUBSTR(\"CENSUS_BLOCK_GROUP\", 1, 2) = '{state_fips}'"

    sql = (
        f'SELECT {group_expr} AS GROUP_FIPS, SUM("{entry.column}") AS VALUE '
        f'FROM "{entry.table}"'
        f"{where_sql} "
        f"GROUP BY {group_expr} "
        # NULLS LAST regardless of direction: Snowflake defaults to NULLS
        # FIRST for DESC, which would let a NULL-valued group (e.g. a
        # geography with zero matching block groups) rank #1 in a "most
        # populated" list -- latent even with the old LIMIT 1, but only
        # became an obvious, visible bug once N rows are surfaced.
        f"ORDER BY VALUE {sort} NULLS LAST "
        f"LIMIT {limit}"
    )
    return BuiltRankingQuery(sql=sql)
