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
