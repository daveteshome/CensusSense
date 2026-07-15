import json
import pathlib
from dataclasses import dataclass
from typing import Optional

from metadata.metric_ranker import MetricRanker

DATA_DIR = pathlib.Path(__file__).resolve().parent.parent / "data"


@dataclass(frozen=True)
class MetricEntry:
    table: str
    column: str
    year: str
    description: str
    table_topics: Optional[str]
    universe: Optional[str]
    aggregation_type: str  # "SUM" | "MEDIAN" | "RATE"
    moe_column: Optional[str]


@dataclass(frozen=True)
class ConceptEntry:
    concept: str
    label: str
    aliases: list[str]
    entry: MetricEntry


def _concept_document(concept: ConceptEntry) -> str:
    """Builds the text the concept ranker matches against. Drops any
    alias that's just a case-insensitive duplicate of the label (auto-
    generated concepts default to `aliases = [label.lower()]`) -- purely
    redundant text, not a meaningfully distinct phrasing. Note: this is
    NOT what protects against a short auto-derived label outscoring a
    longer curated one on TF-IDF cosine -- L2 normalization makes
    uniformly duplicating a whole document's text a mathematical no-op
    for cosine similarity, verified directly. The actual fix for that
    class of bug is the exact alias/label match short-circuit in
    agent/resolver.py's resolve_metric."""
    extra_aliases = [a for a in concept.aliases if a.lower() != concept.label.lower()]
    return " ".join([concept.label] + extra_aliases)


class MetadataStore:
    """Loaded once at process startup; the resolver and SQL builder never
    touch Snowflake or the filesystem again after this."""

    def __init__(self, entries: list[MetricEntry], geography: dict, concept_catalog: Optional[list[dict]] = None):
        self.entries = entries
        self.geography = geography
        self.concept_catalog = concept_catalog or []
        self.available_years = sorted({e.year for e in entries})
        self._by_table_column = {(e.table, e.column): e for e in entries}
        self._concepts_by_year: dict[str, list[ConceptEntry]] = {}
        self._concept_ranker_by_year: dict[str, MetricRanker] = {}
        self._name_by_state_fips = {
            entry["fips"]: entry["name"] for entry in geography["states"].values()
        }

    def find(self, table: str, column: str) -> Optional[MetricEntry]:
        return self._by_table_column.get((table, column))

    def concepts_for_year(self, year: str) -> list[ConceptEntry]:
        """Resolves each concept catalog entry's per-year source pointer
        against the real metadata index. A pointer that doesn't resolve is
        skipped, not a crash -- a drift guard between concept_catalog.json
        and metadata.json (should never happen if both are rebuilt
        together, but a stale/corrupted catalog shouldn't take down the
        whole store)."""
        if year not in self._concepts_by_year:
            resolved = []
            for c in self.concept_catalog:
                source = c["sources"].get(year)
                if source is None:
                    continue
                entry = self.find(source["table"], source["column"])
                if entry is None:
                    continue
                resolved.append(ConceptEntry(concept=c["concept"], label=c["label"], aliases=c["aliases"], entry=entry))
            self._concepts_by_year[year] = resolved
        return self._concepts_by_year[year]

    def concept_ranker_for_year(self, year: str) -> MetricRanker:
        """Built lazily and cached, same pattern as the old per-year
        metric ranker -- but now over the small (~200-entry) concept
        catalog's label+aliases text, not the full ~4,000-8,000-entry raw
        corpus, which is what makes the string matching itself reliable."""
        if year not in self._concept_ranker_by_year:
            concepts = self.concepts_for_year(year)
            documents = [_concept_document(c) for c in concepts]
            self._concept_ranker_by_year[year] = MetricRanker(documents)
        return self._concept_ranker_by_year[year]

    def resolve_state(self, name_or_abbrev: str) -> Optional[dict]:
        return self.geography["states"].get(name_or_abbrev.strip().lower())

    def counties_for_state(self, state_fips: str) -> list[dict]:
        return self.geography["counties"].get(state_fips, [])

    def all_state_names(self) -> list[str]:
        """Canonical display names, deduplicated (the raw lookup dict has
        both abbreviation and full-name keys pointing at the same state)."""
        seen = {}
        for entry in self.geography["states"].values():
            seen[entry["fips"]] = entry["name"]
        return sorted(seen.values())

    def state_name_by_fips(self, fips: str) -> Optional[str]:
        """Reverse of resolve_state -- needed for ranking queries, which
        get a winning FIPS code back from Snowflake and need to turn it
        into a human name for the answer."""
        return self._name_by_state_fips.get(fips)

    def county_name_by_fips(self, state_fips: str, county_fips: str) -> Optional[str]:
        for county in self.counties_for_state(state_fips):
            if county["fips"] == county_fips:
                return county["name"]
        return None


def load_metadata_entries(path: pathlib.Path = None) -> list[MetricEntry]:
    path = path or DATA_DIR / "metadata.json"
    raw = json.loads(path.read_text())
    return [MetricEntry(**r) for r in raw]


def load_geography(path: pathlib.Path = None) -> dict:
    path = path or DATA_DIR / "geography_fips.json"
    return json.loads(path.read_text())


def load_concept_catalog(path: pathlib.Path = None) -> list[dict]:
    path = path or DATA_DIR / "concept_catalog.json"
    return json.loads(path.read_text())


def load_store(
    metadata_path: pathlib.Path = None,
    geography_path: pathlib.Path = None,
    concept_catalog_path: pathlib.Path = None,
) -> MetadataStore:
    return MetadataStore(
        load_metadata_entries(metadata_path),
        load_geography(geography_path),
        load_concept_catalog(concept_catalog_path),
    )
