"""One-off script: export the real INFORMATION_SCHEMA.COLUMNS for the
census database, so build_metadata.py never needs a live Snowflake
connection to rebuild metadata.json.

Run manually when the underlying Snowflake schema changes:
    python scripts/export_schema.py
"""

import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from config import load_config
from snowflake_client import fetch_all

DATA_DIR = pathlib.Path(__file__).resolve().parent.parent / "data"

QUERY = """
    SELECT TABLE_NAME, COLUMN_NAME, DATA_TYPE, COMMENT
    FROM INFORMATION_SCHEMA.COLUMNS
    WHERE TABLE_SCHEMA = %s
    ORDER BY TABLE_NAME, ORDINAL_POSITION
"""


def main() -> None:
    cfg = load_config()
    rows = fetch_all(cfg, QUERY, (cfg.snowflake_schema,))
    print(f"Fetched {len(rows)} columns across "
          f"{len({r['TABLE_NAME'] for r in rows})} tables from "
          f"{cfg.snowflake_database}.{cfg.snowflake_schema}")

    out_path = DATA_DIR / "schema_columns.json"
    out_path.write_text(json.dumps(rows, indent=2, default=str))
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
