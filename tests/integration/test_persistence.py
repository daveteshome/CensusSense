"""Verifies conversation history actually survives a process restart when
backed by the SQLite checkpointer, keyed by a stable per-account
thread_id -- this is the behavior the persistent per-account chat
history feature depends on. LLM/Snowflake calls are mocked, same as
test_pipeline_e2e.py; only the checkpointer backend itself is real."""

import pathlib
import sqlite3

import pytest
from langgraph.checkpoint.sqlite import SqliteSaver

import agent.pipeline as pipeline_module
from agent.executor import ExecutionResult
from agent.gatekeeper import GatekeeperResult
from agent.pipeline import build_checkpoint_serde, build_pipeline, run_turn
from metadata.metadata_store import load_store

FIXTURES = pathlib.Path(__file__).resolve().parent.parent / "fixtures"


@pytest.fixture
def store():
    return load_store(
        metadata_path=FIXTURES / "sample_metadata.json",
        geography_path=FIXTURES / "sample_geography.json",
        concept_catalog_path=FIXTURES / "sample_concept_catalog.json",
    )


def _gatekeeper_result(**overrides) -> GatekeeperResult:
    defaults = dict(
        status="IN_SCOPE",
        concept="total population",
        state="Texas",
        county="",
        year="",
        clarification_reason="",
    )
    defaults.update(overrides)
    return GatekeeperResult(**defaults)


def _build_sqlite_pipeline(store, db_path):
    # A fresh connection each call, same file -- simulates a new process
    # attaching to the same persisted database, not just reusing a live
    # in-memory object.
    conn = sqlite3.connect(str(db_path), check_same_thread=False)
    checkpointer = SqliteSaver(conn, serde=build_checkpoint_serde())
    checkpointer.setup()
    return build_pipeline(store, cfg=object(), checkpointer=checkpointer)


def _patch_llm_and_executor(monkeypatch, gatekeeper_result):
    monkeypatch.setattr(pipeline_module, "classify", lambda question, history, cfg: gatekeeper_result)
    monkeypatch.setattr(
        pipeline_module, "generate_answer", lambda payload, history, cfg: f"ANSWER value={payload.value}"
    )
    monkeypatch.setattr(
        pipeline_module,
        "execute",
        lambda cfg, sql: ExecutionResult(rows=[{"VALUE": 100, "BLOCK_GROUP_COUNT": 1}], row_count=1, exec_time_seconds=0.1),
    )


def test_history_survives_rebuilding_the_pipeline_against_the_same_db_file(store, monkeypatch, tmp_path):
    db_path = tmp_path / "conversations.db"

    pipeline_a = _build_sqlite_pipeline(store, db_path)
    _patch_llm_and_executor(monkeypatch, _gatekeeper_result())
    run_turn(pipeline_a, "population of Texas", "account-alice")

    # A brand-new pipeline instance (and a brand-new sqlite connection),
    # same underlying file -- this is the "process restarted, user signs
    # back in" scenario, not just re-reading from the same live object.
    pipeline_b = _build_sqlite_pipeline(store, db_path)
    snapshot = pipeline_b.get_state({"configurable": {"thread_id": "account-alice"}})
    history = snapshot.values["history"]

    assert len(history) == 1
    assert history[0].question == "population of Texas"
    assert history[0].answer == "ANSWER value=100"


def test_different_accounts_have_isolated_persisted_history(store, monkeypatch, tmp_path):
    db_path = tmp_path / "conversations.db"

    pipeline = _build_sqlite_pipeline(store, db_path)
    _patch_llm_and_executor(monkeypatch, _gatekeeper_result())
    run_turn(pipeline, "population of Texas", "account-alice")

    bob_snapshot = pipeline.get_state({"configurable": {"thread_id": "account-bob"}})
    bob_history = (bob_snapshot.values or {}).get("history", [])

    assert bob_history == []  # bob's thread is untouched by alice's turn


def test_second_turn_on_same_account_sees_first_turns_history_after_restart(store, monkeypatch, tmp_path):
    db_path = tmp_path / "conversations.db"

    pipeline_a = _build_sqlite_pipeline(store, db_path)
    captured_histories = []

    def fake_classify(question, history, cfg):
        captured_histories.append(list(history))
        return _gatekeeper_result()

    monkeypatch.setattr(pipeline_module, "classify", fake_classify)
    monkeypatch.setattr(pipeline_module, "generate_answer", lambda payload, history, cfg: "ANSWER")
    monkeypatch.setattr(
        pipeline_module,
        "execute",
        lambda cfg, sql: ExecutionResult(rows=[{"VALUE": 100, "BLOCK_GROUP_COUNT": 1}], row_count=1, exec_time_seconds=0.1),
    )

    run_turn(pipeline_a, "population of Texas", "account-alice")

    # Simulate a restart: new pipeline/connection, same file, same thread_id.
    pipeline_b = _build_sqlite_pipeline(store, db_path)
    monkeypatch.setattr(pipeline_module, "classify", fake_classify)
    monkeypatch.setattr(pipeline_module, "generate_answer", lambda payload, history, cfg: "ANSWER")
    monkeypatch.setattr(
        pipeline_module,
        "execute",
        lambda cfg, sql: ExecutionResult(rows=[{"VALUE": 100, "BLOCK_GROUP_COUNT": 1}], row_count=1, exec_time_seconds=0.1),
    )
    run_turn(pipeline_b, "how about again", "account-alice")

    assert len(captured_histories) == 2
    assert captured_histories[0] == []  # first turn, first process: no prior history
    assert len(captured_histories[1]) == 1  # second turn, after "restart": sees turn 1
    assert captured_histories[1][0].question == "population of Texas"
