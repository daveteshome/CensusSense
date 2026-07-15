"""Builds data/metadata.json: the single in-memory metadata store the
resolver and SQL builder rely on. Never touches Snowflake directly —
consumes the cached exports from scripts/export_schema.py,
scripts/export_field_descriptions.py so metadata rebuilds are fast,
offline, and reproducible.

Run:
    python -m metadata.build_metadata
"""

import json
import pathlib
import re

from metadata.normalize import moe_column_for, parse_column

DATA_DIR = pathlib.Path(__file__).resolve().parent.parent / "data"

RAW_ACS_TABLE_RE = re.compile(r"^(2019|2020)_CBG_[A-Z]\d+$")
REDISTRICTING_TABLE = "2020_REDISTRICTING_CBG_DATA"
CURATED_TABLES = {
    "2019_TOTAL_RENTAL_GEO": "2019",
    "2019_RENT_PERCENTAGE_HOUSEHOLD_INCOME": "2019",
}
# Non-metric columns present on every fact table (join keys / coordinates),
# never candidates for the metric index.
NON_METRIC_COLUMNS = {"CENSUS_BLOCK_GROUP", "LATITUDE", "LONGITUDE"}


def _load_json(filename: str):
    return json.loads((DATA_DIR / filename).read_text())


def _aggregation_type(table_title: str) -> str:
    title_upper = (table_title or "").upper()
    if title_upper.startswith("MEDIAN"):
        return "MEDIAN"
    if "PERCENT" in title_upper or "RATIO" in title_upper:
        return "RATE"
    return "SUM"


def _normalize_for_dedup(text: str) -> str:
    return re.sub(r"[^a-z0-9]", "", text.lower())


def _build_description(field_desc: dict) -> str:
    """Joins TABLE_TITLE with the FIELD_LEVEL breadcrumb, skipping any
    level that's a near-duplicate of text already included. Single-line
    tables (e.g. most MEDIAN metrics) commonly repeat the table title
    almost verbatim in FIELD_LEVEL_4 -- without dedup that inflates the
    description's length and dilutes its TF-IDF weight enough that an
    unrelated, shorter description can outrank the actual match (verified
    empirically: "median household income" was failing to surface
    B19013e1 in the top 5 before this fix)."""
    title = field_desc.get("TABLE_TITLE")
    parts = [title] if title else []
    seen = {_normalize_for_dedup(title)} if title else set()
    for level in range(4, 11):
        key = "FIELD_LEVELl_9" if level == 9 else f"FIELD_LEVEL_{level}"
        value = field_desc.get(key)
        if not value:
            continue
        normalized = _normalize_for_dedup(value)
        if normalized in seen:
            continue
        parts.append(value)
        seen.add(normalized)
    return ": ".join(parts)


def _build_acs_entries(schema_rows, field_desc_by_year) -> list[dict]:
    entries = []
    for row in schema_rows:
        table = row["TABLE_NAME"]
        column = row["COLUMN_NAME"]
        match = RAW_ACS_TABLE_RE.match(table)
        if not match:
            continue
        year = match.group(1)

        parsed = parse_column(column)
        if parsed is None or parsed.suffix != "e":
            continue  # skip MOE columns and anything that doesn't parse

        field_desc = field_desc_by_year[year].get(column)
        if field_desc is None:
            continue  # no description available in this dataset; exclude rather than guess

        entries.append({
            "table": table,
            "column": column,
            "year": year,
            "description": _build_description(field_desc),
            "table_topics": field_desc.get("TABLE_TOPICS"),
            "universe": field_desc.get("TABLE_UNIVERSE"),
            "aggregation_type": _aggregation_type(field_desc.get("TABLE_TITLE")),
            "moe_column": moe_column_for(column),
        })
    return entries


def _build_redistricting_entries(schema_rows, redistricting_desc_by_id) -> list[dict]:
    entries = []
    for row in schema_rows:
        if row["TABLE_NAME"] != REDISTRICTING_TABLE:
            continue
        column = row["COLUMN_NAME"]
        if column in NON_METRIC_COLUMNS:
            continue
        desc = redistricting_desc_by_id.get(column)
        if desc is None:
            continue
        description = f"{desc.get('COLUMN_TOPIC', '')}: {desc.get('FIELD_NAME', '')}".strip(": ")
        entries.append({
            "table": REDISTRICTING_TABLE,
            "column": column,
            "year": "2020",
            "description": description,
            "table_topics": desc.get("COLUMN_TOPIC"),
            "universe": desc.get("COLUMN_UNIVERSE"),
            "aggregation_type": "SUM",  # decennial counts are additive
            "moe_column": None,  # no margin of error on decennial counts
        })
    return entries


def _build_curated_entries(schema_rows) -> list[dict]:
    entries = []
    for row in schema_rows:
        table = row["TABLE_NAME"]
        if table not in CURATED_TABLES:
            continue
        column = row["COLUMN_NAME"]
        if column in NON_METRIC_COLUMNS:
            continue
        entries.append({
            "table": table,
            "column": column,
            "year": CURATED_TABLES[table],
            "description": column,  # already human-readable, e.g. "Total: Renter-occupied housing units"
            "table_topics": None,
            "universe": None,
            "aggregation_type": "SUM",
            "moe_column": None,
        })
    return entries


def build() -> list[dict]:
    schema_rows = _load_json("schema_columns.json")
    field_desc_by_year = {
        "2019": {r["TABLE_ID"]: r for r in _load_json("field_descriptions_2019.json")},
        "2020": {r["TABLE_ID"]: r for r in _load_json("field_descriptions_2020.json")},
    }
    redistricting_desc_by_id = {r["COLUMN_ID"]: r for r in _load_json("redistricting_field_descriptions.json")}

    entries = (
        _build_acs_entries(schema_rows, field_desc_by_year)
        + _build_redistricting_entries(schema_rows, redistricting_desc_by_id)
        + _build_curated_entries(schema_rows)
    )
    return entries


def main() -> None:
    entries = build()
    out_path = DATA_DIR / "metadata.json"
    out_path.write_text(json.dumps(entries, indent=2))
    by_agg = {}
    for e in entries:
        by_agg[e["aggregation_type"]] = by_agg.get(e["aggregation_type"], 0) + 1
    print(f"Wrote {out_path}: {len(entries)} metric entries {by_agg}")


if __name__ == "__main__":
    main()
