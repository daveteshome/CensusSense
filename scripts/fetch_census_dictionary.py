"""One-off script: cache the Census Bureau's ACS 5-year variable dictionaries.

Run manually when the dictionary needs refreshing:
    python scripts/fetch_census_dictionary.py

Output is committed to the repo so `build_metadata.py` never depends on
api.census.gov being reachable at build/deploy time.
"""

import json
import pathlib

import requests

DATA_DIR = pathlib.Path(__file__).resolve().parent.parent / "data"
YEARS = ["2019", "2020"]
URL_TEMPLATE = "https://api.census.gov/data/{year}/acs/acs5/variables.json"


def fetch_year(year: str) -> dict:
    url = URL_TEMPLATE.format(year=year)
    resp = requests.get(url, timeout=30)
    resp.raise_for_status()
    return resp.json()


def main() -> None:
    DATA_DIR.mkdir(exist_ok=True)
    for year in YEARS:
        print(f"Fetching ACS 5-year variable dictionary for {year}...")
        payload = fetch_year(year)
        variables = payload.get("variables", payload)
        out_path = DATA_DIR / f"acs_variables_{year}.json"
        out_path.write_text(json.dumps(variables, indent=2, sort_keys=True))
        print(f"  wrote {out_path} ({len(variables)} variables)")


if __name__ == "__main__":
    main()
