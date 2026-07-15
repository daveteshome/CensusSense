"""Transforms between Census Bureau ACS variable IDs and this dataset's
Snowflake column naming.

Census Bureau: "B01001_001E" (group, underscore, 3-digit zero-padded
line number, E=estimate/M=margin-of-error).
Snowflake:     "B01001e1"    (group, lowercase e/m, no zero-padding).

Verified against the live schema: columns look like B01001e1, B01001m1,
B01001e15, B01002Fe1, B01002Ge1 (race-iteration groups end in a letter).
"""

import re
from dataclasses import dataclass
from typing import Optional

_COLUMN_RE = re.compile(r"^([A-Z]\d{5}[A-Za-z]?)([em])(\d+)$")


@dataclass(frozen=True)
class ParsedColumn:
    group: str  # e.g. "B01001" or "B01002F"
    suffix: str  # "e" (estimate) or "m" (margin of error)
    line: int  # e.g. 15


def parse_column(column: str) -> Optional[ParsedColumn]:
    match = _COLUMN_RE.match(column)
    if not match:
        return None
    group, suffix, line = match.groups()
    return ParsedColumn(group=group, suffix=suffix, line=int(line))


def is_estimate_column(column: str) -> bool:
    parsed = parse_column(column)
    return parsed is not None and parsed.suffix == "e"


def moe_column_for(column: str) -> Optional[str]:
    """Given an estimate column, return its paired margin-of-error column."""
    parsed = parse_column(column)
    if parsed is None or parsed.suffix != "e":
        return None
    return f"{parsed.group}m{parsed.line}"


def to_census_variable_id(column: str) -> Optional[str]:
    """B01001e15 -> B01001_015E"""
    parsed = parse_column(column)
    if parsed is None:
        return None
    return f"{parsed.group}_{parsed.line:03d}{parsed.suffix.upper()}"
