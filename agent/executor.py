"""Pipeline-facing execution layer: runs already-validated SQL against
Snowflake with a bounded timeout and a single retry, and captures the
timing/row-count data the structured logs and latency budget rely on.

A dedicated, least-privilege connection (see sql/setup_role.sql) is
reused across calls rather than reconnecting per request, with lazy
reconnect if the cached connection has gone stale.
"""

import logging
import time
from dataclasses import dataclass
from typing import Any, Optional

import snowflake.connector

from config import Config

logger = logging.getLogger(__name__)

STATEMENT_TIMEOUT_SECONDS = 25


class ExecutorError(Exception):
    pass


@dataclass(frozen=True)
class ExecutionResult:
    rows: list[dict[str, Any]]
    row_count: int
    exec_time_seconds: float


_cached_connection: Optional[snowflake.connector.SnowflakeConnection] = None


def _new_connection(cfg: Config) -> snowflake.connector.SnowflakeConnection:
    return snowflake.connector.connect(
        account=cfg.snowflake_account,
        user=cfg.snowflake_user,
        password=cfg.snowflake_password,
        role=cfg.snowflake_role,
        warehouse=cfg.snowflake_warehouse,
        database=cfg.snowflake_database,
        schema=cfg.snowflake_schema,
        client_session_keep_alive=False,
        session_parameters={"STATEMENT_TIMEOUT_IN_SECONDS": STATEMENT_TIMEOUT_SECONDS},
    )


def _get_connection(cfg: Config) -> snowflake.connector.SnowflakeConnection:
    global _cached_connection
    if _cached_connection is None or _cached_connection.is_closed():
        _cached_connection = _new_connection(cfg)
    return _cached_connection


def _reset_connection() -> None:
    global _cached_connection
    if _cached_connection is not None:
        try:
            _cached_connection.close()
        except Exception:
            pass
    _cached_connection = None


def execute(cfg: Config, sql: str) -> ExecutionResult:
    """Runs `sql` (already built + validated) against Snowflake. Retries
    once, with a fresh connection, on any Snowflake-side error (covers a
    dropped/suspended-warehouse-resume edge case), then surfaces a clear
    ExecutorError instead of letting a raw connector exception escape to
    the pipeline."""
    last_exc: Optional[Exception] = None
    for attempt in (1, 2):
        conn = _get_connection(cfg)
        cursor = conn.cursor(snowflake.connector.DictCursor)
        start = time.perf_counter()
        try:
            cursor.execute(sql)
            rows = cursor.fetchall()
            elapsed = time.perf_counter() - start
            return ExecutionResult(rows=rows, row_count=len(rows), exec_time_seconds=elapsed)
        except Exception as exc:  # noqa: BLE001 - deliberately broad, see docstring
            last_exc = exc
            logger.warning("snowflake query attempt %d failed: %s", attempt, exc)
            _reset_connection()
        finally:
            cursor.close()

    raise ExecutorError(f"query failed after retry: {last_exc}") from last_exc
