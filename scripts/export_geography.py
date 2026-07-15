"""One-off script: export the state/county FIPS crosswalk from Snowflake
into a small JSON file the app loads entirely into memory at startup, so
geography resolution never needs a runtime SQL join.

Also prints the real data types of STATE_FIPS/COUNTY_FIPS so we can
confirm (rather than assume) whether the leading-zero padding is already
correct on this table.

Run manually when the underlying Snowflake schema changes:
    python scripts/export_geography.py
"""

import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from config import load_config
from metadata.us_states import STATE_ABBREV_TO_NAME, TERRITORY_FIPS_TO_NAME
from snowflake_client import fetch_all

DATA_DIR = pathlib.Path(__file__).resolve().parent.parent / "data"
FIPS_TABLE = "2020_METADATA_CBG_FIPS_CODES"

TYPE_CHECK_QUERY = f"""
    SELECT COLUMN_NAME, DATA_TYPE
    FROM INFORMATION_SCHEMA.COLUMNS
    WHERE TABLE_NAME = '{FIPS_TABLE}'
      AND COLUMN_NAME IN ('STATE_FIPS', 'COUNTY_FIPS')
"""

CROSSWALK_QUERY = f"""
    SELECT DISTINCT STATE, STATE_FIPS, COUNTY_FIPS, COUNTY
    FROM "{FIPS_TABLE}"
    ORDER BY STATE_FIPS, COUNTY_FIPS
"""


def main() -> None:
    cfg = load_config()

    type_rows = fetch_all(cfg, TYPE_CHECK_QUERY)
    print("STATE_FIPS / COUNTY_FIPS data types (verify zero-padding assumption):")
    for row in type_rows:
        print(f"  {row['COLUMN_NAME']}: {row['DATA_TYPE']}")

    rows = fetch_all(cfg, CROSSWALK_QUERY)
    print(f"Fetched {len(rows)} state/county rows from {FIPS_TABLE}")

    states: dict[str, dict] = {}
    counties: dict[str, list[dict]] = {}

    skipped = 0
    for row in rows:
        state_fips = str(row["STATE_FIPS"]).strip().zfill(2)
        county_fips_raw = row["COUNTY_FIPS"]
        county_name = row["COUNTY"]
        raw_abbrev = row["STATE"]

        if county_fips_raw is None or county_name is None:
            skipped += 1
            continue
        county_fips = str(county_fips_raw).strip().zfill(3)

        if raw_abbrev is not None:
            abbrev = raw_abbrev.strip().upper()
            full_name = STATE_ABBREV_TO_NAME.get(abbrev, abbrev)
            states[abbrev.lower()] = {"abbrev": abbrev, "fips": state_fips, "name": full_name}
        else:
            # Territory rows have no STATE abbreviation in this table.
            full_name = TERRITORY_FIPS_TO_NAME.get(state_fips, f"FIPS {state_fips}")

        states[full_name.lower()] = {"abbrev": raw_abbrev, "fips": state_fips, "name": full_name}
        counties.setdefault(state_fips, []).append({"name": county_name, "fips": county_fips})

    if skipped:
        print(f"Skipped {skipped} rows with missing county fips/name")

    payload = {"states": states, "counties": counties}
    out_path = DATA_DIR / "geography_fips.json"
    out_path.write_text(json.dumps(payload, indent=2, sort_keys=True))
    print(f"Wrote {out_path} ({len(states)} state lookup keys, "
          f"{sum(len(v) for v in counties.values())} counties)")


if __name__ == "__main__":
    main()
