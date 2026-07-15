from unittest.mock import MagicMock

import pytest

import agent.executor as executor_module
from agent.executor import ExecutorError, execute


@pytest.fixture(autouse=True)
def reset_cached_connection():
    executor_module._cached_connection = None
    yield
    executor_module._cached_connection = None


def _fake_conn(cursor: MagicMock) -> MagicMock:
    conn = MagicMock()
    conn.is_closed.return_value = False
    conn.cursor.return_value = cursor
    return conn


def test_execute_returns_rows_and_timing(monkeypatch):
    cursor = MagicMock()
    cursor.fetchall.return_value = [{"VALUE": 123}]
    monkeypatch.setattr(executor_module, "_new_connection", lambda cfg: _fake_conn(cursor))

    result = execute(cfg=object(), sql="SELECT 1")

    assert result.rows == [{"VALUE": 123}]
    assert result.row_count == 1
    assert result.exec_time_seconds >= 0
    cursor.execute.assert_called_once_with("SELECT 1")


def test_execute_retries_once_then_succeeds(monkeypatch):
    failing_cursor = MagicMock()
    failing_cursor.execute.side_effect = RuntimeError("connection reset")

    succeeding_cursor = MagicMock()
    succeeding_cursor.fetchall.return_value = [{"VALUE": 42}]

    connections = [_fake_conn(failing_cursor), _fake_conn(succeeding_cursor)]
    monkeypatch.setattr(executor_module, "_new_connection", lambda cfg: connections.pop(0))

    result = execute(cfg=object(), sql="SELECT 1")

    assert result.rows == [{"VALUE": 42}]


def test_execute_raises_executor_error_after_retry_exhausted(monkeypatch):
    cursor = MagicMock()
    cursor.execute.side_effect = RuntimeError("still broken")
    monkeypatch.setattr(executor_module, "_new_connection", lambda cfg: _fake_conn(cursor))

    with pytest.raises(ExecutorError):
        execute(cfg=object(), sql="SELECT 1")


def test_execute_reuses_cached_connection_across_calls(monkeypatch):
    cursor = MagicMock()
    cursor.fetchall.return_value = [{"VALUE": 1}]
    conn = _fake_conn(cursor)
    new_connection_calls = []
    monkeypatch.setattr(
        executor_module,
        "_new_connection",
        lambda cfg: (new_connection_calls.append(1), conn)[1],
    )

    execute(cfg=object(), sql="SELECT 1")
    execute(cfg=object(), sql="SELECT 1")

    assert len(new_connection_calls) == 1
