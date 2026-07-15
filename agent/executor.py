"""Pipeline-facing execution layer: runs already-validated SQL against
Snowflake with a bounded timeout and a single retry, and captures the
timing/row-count data the structured logs and latency budget rely on.

A fresh connection is opened per call and always closed when done -- it
is never cached or shared across calls. A single shared, cached
connection was tried first and found to be unsafe under real deployed
conditions: Streamlit serves multiple accounts/sessions from the same
process, and two calls reaching a shared connection object close
together (not necessarily truly simultaneous -- different accounts
active a few seconds apart is enough) collide on the same underlying
Snowflake session. This surfaces as the connector logging "Session with
id X is already connected! Connecting to a new session." and a failed
query for whichever call lost the race -- confirmed live against the
Railway deployment with two different demo accounts. Opening a
dedicated connection per call costs a bit of latency (sub-second) but
removes the entire bug class, which matters more than the small
performance cost for a human-paced chat app.
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


def execute(cfg: Config, sql: str) -> ExecutionResult:
    """Runs `sql` (already built + validated) against Snowflake. Opens a
    dedicated connection for this call only -- never shared or cached
    across calls, so two requests (different accounts, or anything else
    that overlaps) can never collide on the same underlying session.
    Retries once on any Snowflake-side error (covers a dropped connection
    or a warehouse still resuming from auto-suspend), each attempt with
    its own fresh connection, then surfaces a clear ExecutorError instead
    of letting a raw connector exception escape to the pipeline.

    Explicitly issues `USE WAREHOUSE` before the real query rather than
    relying solely on `connect(warehouse=...)` -- live-verified this
    matters: with `AUTO_SUSPEND=60` on the warehouse and human-paced chat
    (each query on its own fresh connection now, per the fix above), a
    warehouse that suspended between turns produced "No active warehouse
    selected in the current session" on the very first query of a new
    session, on *both* retry attempts, even though `warehouse=` was
    correctly set in `connect()`. The connect-time warehouse argument's
    implicit activation isn't guaranteed to finish before the first query
    is dispatched when the warehouse needs to resume from suspend; a
    separate, synchronous `USE WAREHOUSE` statement blocks until the
    warehouse is actually active before the real query ever runs."""
    last_exc: Optional[Exception] = None
    for attempt in (1, 2):
        conn = _new_connection(cfg)
        try:
            cursor = conn.cursor(snowflake.connector.DictCursor)
            start = time.perf_counter()
            try:
                cursor.execute(f'USE WAREHOUSE "{cfg.snowflake_warehouse}"')
                cursor.execute(sql)
                rows = cursor.fetchall()
                elapsed = time.perf_counter() - start
                return ExecutionResult(rows=rows, row_count=len(rows), exec_time_seconds=elapsed)
            except Exception as exc:
                last_exc = exc
                logger.warning("snowflake query attempt %d failed: %s", attempt, exc)
            finally:
                cursor.close()
        finally:
            try:
                conn.close()
            except Exception:
                pass

    raise ExecutorError(f"query failed after retry: {last_exc}") from last_exc
