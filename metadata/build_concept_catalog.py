"""Builds data/concept_catalog.json: a small (~150-200 entry), curated
index of headline Census concepts, derived offline from the already-
committed data/metadata.json (8,276 raw column entries). This replaces
matching free-form questions against the full raw corpus (fragile --
see agent/resolver.py's retired typo-correction machinery) with matching
against a much smaller, cleaner set of real, human-recognizable concepts.

Each concept maps to one canonical (table, column) per year -- the
"headline" row for that concept (e.g. "Total Population: Total", not one
of the hundreds of age/sex/race-sliced breakdowns underneath it).

Run:
    python -m metadata.build_concept_catalog
"""

import json
import pathlib
import re
from collections import defaultdict

from metadata.normalize import parse_column

DATA_DIR = pathlib.Path(__file__).resolve().parent.parent / "data"

REDISTRICTING_TABLE = "2020_REDISTRICTING_CBG_DATA"
CURATED_TABLE = "2019_TOTAL_RENTAL_GEO"
# 2019_RENT_PERCENTAGE_HOUSEHOLD_INCOME is deliberately excluded entirely:
# its only rows are a percentage-bucket distribution (rent as % of
# household income, bucketed), not a single standalone headline figure --
# including "one of the buckets" as if it were a concept would be a
# misleading number, not a reasonable scope cut. Don't resurrect this by
# trying to force-fit it into the general clustering rule below.
EXCLUDED_CURATED_TABLE = "2019_RENT_PERCENTAGE_HOUSEHOLD_INCOME"

# ACS race/ethnicity-iteration tables append a single letter to the base
# variable group (B01002 -> B01002A..I for White/Black/Asian/.../Hispanic
# iterations -- see metadata/normalize.py's docstring). This structural
# signal is more precise than matching description text for a race-related
# word: text matching on e.g. "alone" false-positives on unrelated phrases
# like "Household Type (Including Living Alone)" (a household-composition
# category, not a race qualifier) -- verified this collision is real by
# testing against B11001's actual descriptions before settling on the
# structural check instead.
_RACE_ITERATION_GROUP_RE = re.compile(r"^[A-Z]\d{5}[A-Za-z]$")


def _is_race_iteration_group(group: str) -> bool:
    return bool(_RACE_ITERATION_GROUP_RE.match(group))


def _normalize_for_dedup(text: str) -> str:
    return re.sub(r"[^a-z0-9]", "", text.lower())


def _last_segment_is_total(description: str) -> bool:
    return description.rsplit(":", 1)[-1].strip().lower() == "total"


def _derive_label(description: str) -> str:
    """Strips trailing ': Total' segments (there can be more than one,
    e.g. "Median Age By Sex: Median age: Total") and a final segment
    that's just a near-duplicate restatement of the first, leaving a
    clean human-readable concept label."""
    parts = [p.strip() for p in description.split(":")]
    while len(parts) > 1 and parts[-1].lower() == "total":
        parts.pop()
    if len(parts) > 1 and _normalize_for_dedup(parts[-1]) == _normalize_for_dedup(parts[0]):
        parts.pop()
    return ": ".join(parts)


def _raw_group(entry: dict) -> str:
    return parse_column(entry["column"]).group


def _base_group(entry: dict) -> str:
    group = _raw_group(entry)
    if _is_race_iteration_group(group):
        group = group[:-1]
    return group


def _cluster_acs_entries(entries: list[dict]) -> tuple[list[dict], list[str]]:
    """Groups ACS rows by (base_group, year) and resolves each cluster to
    one canonical row, or excludes it with a logged reason. Returns
    (canonical_rows, exclusion_log_lines)."""
    clusters: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for e in entries:
        clusters[(_base_group(e), e["year"])].append(e)

    canonical = []
    log = []
    for (base_group, year), rows in sorted(clusters.items()):
        if len(rows) == 1:
            canonical.append(rows[0])
            continue

        unqualified = [r for r in rows if not _is_race_iteration_group(_raw_group(r))]
        candidates = unqualified or rows
        if len(candidates) == 1:
            canonical.append(candidates[0])
            continue

        totals = [r for r in candidates if _last_segment_is_total(r["description"])]
        if len(totals) == 1:
            canonical.append(totals[0])
            continue

        log.append(
            f"skipped {base_group} ({year}): no clean default row "
            f"({len(rows)} rows, {len(candidates)} unqualified, {len(totals)} unqualified totals)"
        )
    return canonical, log


def _build_acs_concepts(metadata: list[dict]) -> tuple[dict[str, dict], list[str]]:
    acs_entries = [
        e for e in metadata
        if e["table"] not in (REDISTRICTING_TABLE, CURATED_TABLE, EXCLUDED_CURATED_TABLE)
        and parse_column(e["column"]) is not None
        and not _base_group(e).startswith("B99")  # allocation/imputation flag tables, not real metrics
    ]
    canonical_rows, log = _cluster_acs_entries(acs_entries)

    concepts: dict[str, dict] = {}
    for row in canonical_rows:
        base_group = _base_group(row)
        concept_id = base_group.lower()
        label = _derive_label(row["description"])
        concept = concepts.setdefault(
            concept_id,
            {"concept": concept_id, "label": label, "aliases": [label.lower()], "sources": {}},
        )
        concept["sources"][row["year"]] = {"table": row["table"], "column": row["column"]}
    return concepts, log


# P001's "Total" row (P0010001) is the same real total-population count
# already used as the "population" concept's 2020 source (see
# CURATED_CONCEPTS["b01003"]) -- P001's own table title is "RACE" (it's
# the base row of the race-breakdown table, not a race breakdown itself),
# so auto-deriving a separate "RACE"-labeled concept from it here would
# just be a confusing, redundant duplicate of "population" that also
# collides with the real ACS Race concept (b02001)'s "race" alias.
_REDISTRICTING_PREFIXES_TO_SKIP = {"P001"}


def _build_redistricting_concepts(metadata: list[dict]) -> dict[str, dict]:
    redist_entries = [e for e in metadata if e["table"] == REDISTRICTING_TABLE]
    by_prefix: dict[str, list[dict]] = defaultdict(list)
    for e in redist_entries:
        prefix = e["column"][:4]  # e.g. "P001"
        by_prefix[prefix].append(e)

    concepts: dict[str, dict] = {}
    for prefix, rows in sorted(by_prefix.items()):
        if prefix in _REDISTRICTING_PREFIXES_TO_SKIP:
            continue
        totals = [r for r in rows if _last_segment_is_total(r["description"])]
        if len(totals) != 1:
            continue  # verified all 5 real groups have exactly one; don't guess if this ever changes
        row = totals[0]
        concept_id = f"redistricting_{prefix.lower()}"
        label = _derive_label(row["description"])
        concepts[concept_id] = {
            "concept": concept_id,
            "label": label,
            "aliases": [label.lower()],
            "sources": {row["year"]: {"table": row["table"], "column": row["column"]}},
        }
    return concepts


def _build_curated_concepts(metadata: list[dict]) -> dict[str, dict]:
    rows = [e for e in metadata if e["table"] == CURATED_TABLE]
    if not rows:
        return {}
    row = rows[0]  # single-column curated table
    return {
        "renter_occupied_housing_units": {
            "concept": "renter_occupied_housing_units",
            "label": "Renter-Occupied Housing Units",
            "aliases": ["renter-occupied housing units", "renters", "renter occupied units"],
            "sources": {row["year"]: {"table": row["table"], "column": row["column"]}},
        }
    }


# Hand-curated overrides for well-known concepts, keyed by the
# *auto-derived* concept id they refine. Applied as a final pass:
# supplies a cleaner label + hand-written aliases, and (population only)
# a source override -- see the module-level note on the ACS-vs-decennial
# collision for 2020 population, verified directly against the real data
# (2020_CBG_B01/B01003e1 "Total Population: Total" and
# 2020_REDISTRICTING_CBG_DATA/P0010001 "RACE: Total" are mechanically
# unrelated candidates for the same real-world concept; nothing about
# their table numbers or topic labels would let an automatic rule unify
# them -- the decennial count is the correct source for a real 2020
# population figure, not the ACS 5-year estimate).
CURATED_CONCEPTS: dict[str, dict] = {
    "b01003": {
        "label": "Population",
        "aliases": [
            "population", "total population", "residents", "people",
            "how many people", "number of people", "inhabitants",
        ],
        "source_overrides": {"2020": {"table": REDISTRICTING_TABLE, "column": "P0010001"}},
    },
    "b19013": {
        "label": "Median Household Income",
        "aliases": ["median household income", "household income", "income"],
    },
    "b19301": {
        "label": "Per Capita Income",
        "aliases": ["per capita income", "average income per person"],
    },
    "b01002": {
        "label": "Median Age",
        "aliases": ["median age", "average age", "age"],
    },
    "b17021": {
        "label": "Poverty Status",
        "aliases": ["poverty", "people in poverty", "poverty rate", "poverty status"],
    },
    "b23025": {
        "label": "Employment Status",
        "aliases": ["employment", "unemployment", "labor force", "jobs"],
    },
    "b25001": {
        "label": "Housing Units",
        "aliases": ["housing units", "number of houses", "homes"],
    },
    "b25077": {
        "label": "Median Home Value",
        "aliases": ["median home value", "home value", "house prices", "home prices"],
    },
    "b25064": {
        "label": "Median Gross Rent",
        "aliases": ["median gross rent", "rent", "average rent"],
    },
    "b15003": {
        "label": "Educational Attainment",
        "aliases": ["education", "educational attainment", "degrees"],
    },
    "b11001": {
        "label": "Households",
        "aliases": ["households", "number of households"],
    },
    "b21001": {
        "label": "Veteran Status",
        "aliases": ["veterans", "veteran population"],
    },
    # No standalone "disability status" concept exists in this dataset --
    # verified directly: every disability-related table here is a compound
    # cross-tabulation (food stamps by disability, poverty by disability by
    # employment, age by veteran by poverty by disability), with no clean
    # "number of people with a disability" headline row. Deliberately not
    # curating a "disability" alias onto one of these compound tables,
    # since that would silently answer a different, narrower question than
    # what was asked -- an honest NO_MATCH is correct here, not a curated
    # override onto a misleading proxy.
    "b27010": {
        "label": "Health Insurance Coverage",
        "aliases": ["health insurance", "insurance coverage", "health insurance coverage"],
    },
    "b02001": {
        "label": "Race",
        "aliases": ["race", "race breakdown"],
    },
    "b03003": {
        "label": "Hispanic or Latino Origin",
        "aliases": ["ethnicity", "hispanic or latino origin", "hispanic population"],
    },
}


def _apply_overrides(concepts: dict[str, dict]) -> list[str]:
    log = []
    for concept_id, override in CURATED_CONCEPTS.items():
        if concept_id not in concepts:
            log.append(f"WARNING: curated override for {concept_id!r} has no matching auto-derived concept")
            continue
        entry = concepts[concept_id]
        entry["label"] = override["label"]
        entry["aliases"] = override["aliases"]
        for year, source in override.get("source_overrides", {}).items():
            entry["sources"][year] = source
    return log


def _validate(concepts: list[dict], metadata_by_key: dict[tuple[str, str], dict]) -> None:
    for concept in concepts:
        for year, source in concept["sources"].items():
            key = (source["table"], source["column"])
            if key not in metadata_by_key:
                raise RuntimeError(
                    f"concept {concept['concept']!r} ({year}) points at {key}, "
                    "which doesn't exist in data/metadata.json -- catalog and metadata have drifted"
                )


def build() -> list[dict]:
    metadata = json.loads((DATA_DIR / "metadata.json").read_text())
    metadata_by_key = {(e["table"], e["column"]): e for e in metadata}

    acs_concepts, exclusion_log = _build_acs_concepts(metadata)
    override_log = _apply_overrides(acs_concepts)
    redistricting_concepts = _build_redistricting_concepts(metadata)
    curated_concepts = _build_curated_concepts(metadata)

    all_concepts = {**acs_concepts, **redistricting_concepts, **curated_concepts}
    result = sorted(all_concepts.values(), key=lambda c: c["concept"])

    _validate(result, metadata_by_key)

    for line in exclusion_log + override_log:
        print(line)

    return result


def main() -> None:
    concepts = build()
    out_path = DATA_DIR / "concept_catalog.json"
    out_path.write_text(json.dumps(concepts, indent=2))
    curated_count = sum(1 for c in concepts if c["concept"] in CURATED_CONCEPTS)
    print(
        f"Wrote {out_path}: {len(concepts)} concepts "
        f"({curated_count} curated, {len(concepts) - curated_count} auto-only)"
    )


if __name__ == "__main__":
    main()
