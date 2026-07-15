from dataclasses import dataclass
from unittest.mock import MagicMock

import pytest

import agent.executor as executor_module
from agent.executor import ExecutorError, execute


@dataclass
class _FakeConfig:
    snowflake_warehouse: str = "TEST_WH"


def _fake_conn(cursor: MagicMock) -> MagicMock:
    conn = MagicMock()
    conn.cursor.return_value = cursor
    return conn


def test_execute_returns_rows_and_timing(monkeypatch):
    cursor = MagicMock()
    cursor.fetchall.return_value = [{"VALUE": 123}]
    monkeypatch.setattr(executor_module, "_new_connection", lambda cfg: _fake_conn(cursor))

    result = execute(cfg=_FakeConfig(), sql="SELECT 1")

    assert result.rows == [{"VALUE": 123}]
    assert result.row_count == 1
    assert result.exec_time_seconds >= 0
    cursor.execute.assert_any_call("SELECT 1")


def test_execute_retries_once_then_succeeds(monkeypatch):
    failing_cursor = MagicMock()
    failing_cursor.execute.side_effect = RuntimeError("connection reset")

    succeeding_cursor = MagicMock()
    succeeding_cursor.fetchall.return_value = [{"VALUE": 42}]

    connections = [_fake_conn(failing_cursor), _fake_conn(succeeding_cursor)]
    monkeypatch.setattr(executor_module, "_new_connection", lambda cfg: connections.pop(0))

    result = execute(cfg=_FakeConfig(), sql="SELECT 1")

    assert result.rows == [{"VALUE": 42}]


def test_execute_raises_executor_error_after_retry_exhausted(monkeypatch):
    cursor = MagicMock()
    cursor.execute.side_effect = RuntimeError("still broken")
    monkeypatch.setattr(executor_module, "_new_connection", lambda cfg: _fake_conn(cursor))

    with pytest.raises(ExecutorError):
        execute(cfg=_FakeConfig(), sql="SELECT 1")


def test_execute_opens_a_fresh_connection_per_call(monkeypatch):
    # Regression test for a real, live-verified bug: a single connection
    # shared/cached across calls collided when two different accounts hit
    # the deployed app within a short window of each other (Snowflake
    # logged "Session with id X is already connected!" and the losing
    # call failed). Every call to execute() must open its own connection
    # -- never reuse one across calls.
    cursor_factory_calls = []

    def new_cursor():
        cursor = MagicMock()
        cursor.fetchall.return_value = [{"VALUE": 1}]
        return cursor

    new_connection_calls = []

    def fake_new_connection(cfg):
        new_connection_calls.append(1)
        return _fake_conn(new_cursor())

    monkeypatch.setattr(executor_module, "_new_connection", fake_new_connection)

    execute(cfg=_FakeConfig(), sql="SELECT 1")
    execute(cfg=_FakeConfig(), sql="SELECT 1")

    assert len(new_connection_calls) == 2


def test_execute_closes_connection_after_success(monkeypatch):
    cursor = MagicMock()
    cursor.fetchall.return_value = [{"VALUE": 1}]
    conn = _fake_conn(cursor)
    monkeypatch.setattr(executor_module, "_new_connection", lambda cfg: conn)

    execute(cfg=_FakeConfig(), sql="SELECT 1")

    conn.close.assert_called_once()


def test_execute_closes_every_connection_after_failure(monkeypatch):
    cursor = MagicMock()
    cursor.execute.side_effect = RuntimeError("still broken")
    created_connections = []

    def fake_new_connection(cfg):
        conn = _fake_conn(cursor)
        created_connections.append(conn)
        return conn

    monkeypatch.setattr(executor_module, "_new_connection", fake_new_connection)

    with pytest.raises(ExecutorError):
        execute(cfg=_FakeConfig(), sql="SELECT 1")

    # Both attempts' connections must be closed, not just the last one --
    # a leaked connection on the failing attempt would defeat the point
    # of not caching/sharing them.
    assert len(created_connections) == 2
    for conn in created_connections:
        conn.close.assert_called_once()
