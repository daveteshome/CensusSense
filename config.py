import os
from dataclasses import dataclass

from dotenv import load_dotenv

load_dotenv()


def _require(name: str) -> str:
    value = os.environ.get(name)
    if not value:
        raise RuntimeError(f"Missing required environment variable: {name}")
    return value


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

    demo_username: str
    demo_password: str

    run_live_snowflake_tests: bool

    default_year: str = "2020"


def load_config() -> Config:
    return Config(
        groq_api_key=_require("GROQ_API_KEY"),
        snowflake_account=_require("SNOWFLAKE_ACCOUNT"),
        snowflake_user=_require("SNOWFLAKE_USER"),
        snowflake_password=_require("SNOWFLAKE_PASSWORD"),
        snowflake_role=os.environ.get("SNOWFLAKE_ROLE", "CENSUSSENSE_APP_ROLE"),
        snowflake_warehouse=os.environ.get("SNOWFLAKE_WAREHOUSE", "CENSUSSENSE_WH"),
        snowflake_database=_require("SNOWFLAKE_DATABASE"),
        snowflake_schema=os.environ.get("SNOWFLAKE_SCHEMA", "PUBLIC"),
        demo_username=_require("DEMO_USERNAME"),
        demo_password=_require("DEMO_PASSWORD"),
        run_live_snowflake_tests=os.environ.get("RUN_LIVE_SNOWFLAKE_TESTS", "0") == "1",
    )
