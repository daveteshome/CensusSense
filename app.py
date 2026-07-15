import logging
import sqlite3
import time
from pathlib import Path

import streamlit as st
from langgraph.checkpoint.sqlite import SqliteSaver

from agent.guardrails import RateLimiter
from agent.pipeline import build_checkpoint_serde, build_pipeline, run_turn
from config import load_config
from logging_config import configure_logging, log_turn
from metadata.metadata_store import load_store

configure_logging()
logger = logging.getLogger("censussense")

st.set_page_config(page_title="CensusSense", page_icon="\U0001F4CA")


@st.cache_resource
def get_pipeline():
    """Cached once per server process: loads the metadata store and
    compiles the LangGraph pipeline a single time, shared across all
    sessions. Conversation history is persisted per-account (not
    per-session) via a SQLite-backed checkpointer, keyed by a stable
    thread_id derived from the signed-in username -- see thread_id_for."""
    cfg = load_config()
    store = load_store()

    db_path = Path(cfg.conversations_db_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path), check_same_thread=False)
    conn.execute("PRAGMA busy_timeout = 5000")
    checkpointer = SqliteSaver(conn, serde=build_checkpoint_serde())
    checkpointer.setup()

    pipeline = build_pipeline(store, cfg, checkpointer=checkpointer)
    return pipeline, cfg


def thread_id_for(username: str) -> str:
    """Stable per-account thread id -- not a random per-session uuid --
    so signing in as the same account always resumes the same LangGraph
    checkpointed thread, regardless of browser/device/session."""
    return f"account-{username}"


def require_signin(cfg) -> bool:
    if st.session_state.get("authenticated"):
        return True

    st.title("CensusSense")
    st.caption("Sign in to chat with the US Census data agent.")
    with st.form("signin"):
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        submitted = st.form_submit_button("Sign in")
    if submitted:
        if cfg.demo_accounts.get(username) == password:
            st.session_state["authenticated"] = True
            st.session_state["username"] = username
            st.rerun()
        else:
            st.error("Invalid username or password.")
    return False


def sign_out():
    """Clears browser-session display state only -- the real
    conversation record lives server-side in SQLite keyed by thread_id,
    so signing back in with the same account restores it in full."""
    for key in ("authenticated", "username", "thread_id", "messages", "rate_limiter"):
        st.session_state.pop(key, None)
    st.rerun()


def render_sidebar():
    with st.sidebar:
        st.write(f"Signed in as **{st.session_state.get('username', '')}**")
        if st.button("Sign out"):
            sign_out()

        st.subheader("About this dataset")
        st.write(
            "Grounded in the US Open Census Data (Neighborhood Insights) on "
            "Snowflake Marketplace: ACS 5-year estimates (2019, 2020) and "
            "2020 decennial redistricting counts, at nationwide/state/county "
            "granularity."
        )
        st.write(
            "Known limits: no city/place-level geography, and metrics "
            "reported per Census block group (medians/rates) are approximated "
            "when aggregated to a state or county."
        )


def _restore_messages(pipeline, thread_id: str) -> list[tuple[str, str]]:
    """Rebuilds the chat display from the checkpointed conversation
    history on sign-in -- this is a read of existing state, not a new
    pipeline turn, so it doesn't invoke the LLM/Snowflake at all."""
    snapshot = pipeline.get_state({"configurable": {"thread_id": thread_id}})
    history = (snapshot.values or {}).get("history", []) if snapshot else []
    messages: list[tuple[str, str]] = []
    for turn in history:
        if turn is None:
            # A stale checkpoint from before a Turn schema change (e.g. a
            # renamed field) deserializes to None rather than raising --
            # see agent/state.py's format_history for the same guard.
            continue
        messages.append(("user", turn.question))
        messages.append(("assistant", turn.answer or ""))
    return messages


def main():
    pipeline, cfg = get_pipeline()

    if not require_signin(cfg):
        return

    if "thread_id" not in st.session_state:
        st.session_state["thread_id"] = thread_id_for(st.session_state["username"])
    if "rate_limiter" not in st.session_state:
        st.session_state["rate_limiter"] = RateLimiter(max_requests=20, window_seconds=60)
    if "messages" not in st.session_state:
        st.session_state["messages"] = _restore_messages(pipeline, st.session_state["thread_id"])

    render_sidebar()
    st.title("CensusSense")
    st.caption(
        "Ask about US population, income, age, housing, and more, grounded "
        "in Census Bureau data (2019/2020 ACS estimates, 2020 decennial counts)."
    )

    for role, content in st.session_state["messages"]:
        with st.chat_message(role):
            st.write(content)

    question = st.chat_input("Ask a question about US Census data...")
    if not question:
        return

    st.session_state["messages"].append(("user", question))
    with st.chat_message("user"):
        st.write(question)

    with st.chat_message("assistant"):
        if not st.session_state["rate_limiter"].allow():
            answer = "You're sending questions a bit fast, please wait a moment before asking another."
            st.write(answer)
        else:
            with st.spinner("Looking that up..."):
                start = time.perf_counter()
                try:
                    answer = run_turn(pipeline, question, st.session_state["thread_id"])
                except Exception:
                    logger.exception("unhandled pipeline error")
                    answer = (
                        "Something went wrong on my end while processing that. "
                        "Please try rephrasing your question, or try again in a moment."
                    )
                exec_time = time.perf_counter() - start
                log_turn(
                    logger,
                    question=question,
                    answer=answer,
                    exec_time_seconds=round(exec_time, 2),
                    thread_id=st.session_state["thread_id"],
                )
            st.write(answer)

    st.session_state["messages"].append(("assistant", answer))


if __name__ == "__main__":
    main()
