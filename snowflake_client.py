import contextlib
from typing import Any, Iterator

import snowflake.connector

from config import Config


@contextlib.contextmanager
def get_connection(cfg: Config) -> Iterator[snowflake.connector.SnowflakeConnection]:
    conn = snowflake.connector.connect(
        account=cfg.snowflake_account,
        user=cfg.snowflake_user,
        password=cfg.snowflake_password,
        role=cfg.snowflake_role,
        warehouse=cfg.snowflake_warehouse,
        database=cfg.snowflake_database,
        schema=cfg.snowflake_schema,
        client_session_keep_alive=False,
    )
    try:
        yield conn
    finally:
        conn.close()


def fetch_all(cfg: Config, sql: str, params: tuple = ()) -> list[dict[str, Any]]:
    with get_connection(cfg) as conn:
        cur = conn.cursor(snowflake.connector.DictCursor)
        try:
            cur.execute(sql, params)
            return cur.fetchall()
        finally:
            cur.close()
