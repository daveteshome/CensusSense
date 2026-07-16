import os
from dataclasses import dataclass

from dotenv import load_dotenv

load_dotenv()


def _require(name: str) -> str:
    value = os.environ.get(name)
    if not value or not value.strip():
        raise RuntimeError(f"Missing required environment variable: {name}")
    return value.strip()


@dataclass(frozen=True)
class Config:
    groq_api_key: str

    snowflake_account: str
    snowflake_user: str
    snowflake_password: str
    snowflake_role: str
    snowflake_warehouse: str
    snowflake_database: str
    snowflake_schema: str

    demo_accounts: dict[str, str]

    run_live_snowflake_tests: bool

    default_year: str = "2020"
    conversations_db_path: str = "data/conversations.db"


def _parse_demo_accounts(raw: str) -> dict[str, str]:
    accounts: dict[str, str] = {}
    for pair in raw.split(","):
        pair = pair.strip()
        if not pair:
            continue
        username, sep, password = pair.partition(":")
        if not sep or not username or not password:
            raise RuntimeError(
                f"Malformed DEMO_ACCOUNTS entry {pair!r} -- expected user:pass"
            )
        accounts[username] = password
    if not accounts:
        raise RuntimeError("DEMO_ACCOUNTS must contain at least one user:pass entry")
    return accounts


def load_config() -> Config:
    return Config(
        groq_api_key=_require("GROQ_API_KEY"),
        snowflake_account=_require("SNOWFLAKE_ACCOUNT"),
        snowflake_user=_require("SNOWFLAKE_USER"),
        snowflake_password=_require("SNOWFLAKE_PASSWORD"),
        snowflake_role=os.environ.get("SNOWFLAKE_ROLE", "CENSUSSENSE_APP_ROLE").strip(),
        snowflake_warehouse=os.environ.get("SNOWFLAKE_WAREHOUSE", "CENSUSSENSE_WH").strip(),
        snowflake_database=_require("SNOWFLAKE_DATABASE"),
        snowflake_schema=os.environ.get("SNOWFLAKE_SCHEMA", "PUBLIC").strip(),
        demo_accounts=_parse_demo_accounts(_require("DEMO_ACCOUNTS")),
        run_live_snowflake_tests=os.environ.get("RUN_LIVE_SNOWFLAKE_TESTS", "0") == "1",
        conversations_db_path=os.environ.get("CONVERSATIONS_DB_PATH", "data/conversations.db"),
    )
