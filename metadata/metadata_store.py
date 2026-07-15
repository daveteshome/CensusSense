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


class MetadataStore:
    """Loaded once at process startup; the resolver and SQL builder never
    touch Snowflake or the filesystem again after this."""

    def __init__(self, entries: list[MetricEntry], geography: dict):
        self.entries = entries
        self.geography = geography
        self.available_years = sorted({e.year for e in entries})
        self._by_table_column = {(e.table, e.column): e for e in entries}
        self._entries_by_year: dict[str, list[MetricEntry]] = {}
        self._ranker_by_year: dict[str, MetricRanker] = {}

    def find(self, table: str, column: str) -> Optional[MetricEntry]:
        return self._by_table_column.get((table, column))

    def entries_for_year(self, year: str) -> list[MetricEntry]:
        if year not in self._entries_by_year:
            self._entries_by_year[year] = [e for e in self.entries if e.year == year]
        return self._entries_by_year[year]

    def ranker_for_year(self, year: str) -> MetricRanker:
        """Built lazily and cached: constructing the TF-IDF index over
        ~4,000 descriptions takes ~100ms, not worth redoing per query."""
        if year not in self._ranker_by_year:
            entries = self.entries_for_year(year)
            self._ranker_by_year[year] = MetricRanker([e.description for e in entries])
        return self._ranker_by_year[year]

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


def load_metadata_entries(path: pathlib.Path = None) -> list[MetricEntry]:
    path = path or DATA_DIR / "metadata.json"
    raw = json.loads(path.read_text())
    return [MetricEntry(**r) for r in raw]


def load_geography(path: pathlib.Path = None) -> dict:
    path = path or DATA_DIR / "geography_fips.json"
    return json.loads(path.read_text())


def load_store(metadata_path: pathlib.Path = None, geography_path: pathlib.Path = None) -> MetadataStore:
    return MetadataStore(load_metadata_entries(metadata_path), load_geography(geography_path))
