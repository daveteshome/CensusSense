"""One-off script: export the in-dataset field-description tables.

Discovered while exploring the live schema: 2019_METADATA_CBG_FIELD_DESCRIPTIONS
and 2020_METADATA_CBG_FIELD_DESCRIPTIONS are keyed directly by the exact
Snowflake column name (e.g. TABLE_ID='B01001e17') and give a human-readable
breadcrumb (TABLE_TITLE + FIELD_LEVEL_1..10). This is a better metadata
source than bridging through the external Census Bureau API, since it's
already aligned to this dataset's exact column names with no transform
needed. 2020_REDISTRICTING_METADATA_CBG_FIELD_DESCRIPTIONS is a similar,
flatter table for the 2020 Redistricting (decennial) counts.

Run manually when the underlying Snowflake schema changes:
    python scripts/export_field_descriptions.py
"""

import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from config import load_config
from snowflake_client import fetch_all

DATA_DIR = pathlib.Path(__file__).resolve().parent.parent / "data"

TABLES = {
    "2019_METADATA_CBG_FIELD_DESCRIPTIONS": "field_descriptions_2019.json",
    "2020_METADATA_CBG_FIELD_DESCRIPTIONS": "field_descriptions_2020.json",
    "2020_REDISTRICTING_METADATA_CBG_FIELD_DESCRIPTIONS": "redistricting_field_descriptions.json",
}


def main() -> None:
    cfg = load_config()
    for table, filename in TABLES.items():
        rows = fetch_all(cfg, f'SELECT * FROM "{table}"')
        out_path = DATA_DIR / filename
        out_path.write_text(json.dumps(rows, indent=2, default=str))
        print(f"{table}: {len(rows)} rows -> {out_path}")


if __name__ == "__main__":
    main()
