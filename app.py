import logging
import time
import uuid

import streamlit as st

from agent.guardrails import RateLimiter
from agent.pipeline import build_pipeline, run_turn
from config import load_config
from logging_config import configure_logging, log_turn
from metadata.metadata_store import load_store

configure_logging()
logger = logging.getLogger("censussense")

st.set_page_config(page_title="CensusSense", page_icon="\U0001F4CA")


@st.cache_resource
def get_pipeline():
    """Cached once per server process: loads the metadata store and
    compiles the LangGraph pipeline (with its in-memory checkpointer)
    a single time, shared across all sessions. Per-session isolation
    comes from each session's own thread_id, not from separate
    pipeline instances."""
    cfg = load_config()
    store = load_store()
    pipeline = build_pipeline(store, cfg)
    return pipeline, cfg


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
        if username == cfg.demo_username and password == cfg.demo_password:
            st.session_state["authenticated"] = True
            st.rerun()
        else:
            st.error("Invalid username or password.")
    return False


def render_sidebar():
    with st.sidebar:
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


def main():
    pipeline, cfg = get_pipeline()

    if not require_signin(cfg):
        return

    if "thread_id" not in st.session_state:
        st.session_state["thread_id"] = str(uuid.uuid4())
    if "rate_limiter" not in st.session_state:
        st.session_state["rate_limiter"] = RateLimiter(max_requests=20, window_seconds=60)
    if "messages" not in st.session_state:
        st.session_state["messages"] = []

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
