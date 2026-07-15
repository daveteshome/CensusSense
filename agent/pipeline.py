"""LangGraph wiring for the full request flow:

Gatekeeper -> Resolver -> SQL Builder+Validator+Executor -> Responder

with a shared `guardrail` terminal node for every non-happy-path case
(out of scope, inappropriate, ambiguous, no match, unsupported year,
empty result, or an unhandled exception in any stage). Conversation
memory is LangGraph's own checkpointer, keyed by a per-session thread_id
-- see the Conversation memory section of the plan for the deliberate
scope (in-session only, not cross-restart).
"""

import operator
from typing import Annotated, Optional, TypedDict

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
from langgraph.graph import END, StateGraph

from agent.executor import ExecutorError, execute
from agent.gatekeeper import classify
from agent.guardrails import build_message
from agent.llm_client import LLMError
from agent.resolver import resolve_geography, resolve_metric, resolve_year
from agent.responder import RankingRow, ResponderInput, generate_answer
from agent.sql_builder import MAX_RANKING_LIMIT, SqlBuildError, build_query, build_ranking_query
from agent.sql_validator import SqlValidationError, validate
from agent.state import Turn
from config import Config
from metadata.metadata_store import MetadataStore


class PipelineState(TypedDict, total=False):
    question: str
    history: Annotated[list[Turn], operator.add]
    gatekeeper: Optional[object]
    year_resolution: Optional[object]
    geography_resolution: Optional[object]
    metric_resolution: Optional[object]
    built_query: Optional[object]
    execution_result: Optional[object]
    ranking_unsupported_metric: Optional[bool]
    answer: Optional[str]
    error: Optional[str]


def _turn_from_state(state: PipelineState, answer: str) -> Turn:
    gk = state.get("gatekeeper")
    year_res = state.get("year_resolution")
    geo_res = state.get("geography_resolution")
    return Turn(
        question=state["question"],
        concept=(gk.concept if gk else None) or None,
        state=(geo_res.state_name if geo_res else None) or (gk.state if gk else None) or None,
        county=(geo_res.county_name if geo_res else None) or None,
        year=(year_res.year if year_res else None) or None,
        answer=answer,
    )


def build_checkpoint_serde() -> JsonPlusSerializer:
    """Shared serde registration for whichever checkpointer backend is in
    use (InMemorySaver for tests/default, SqliteSaver for the app's
    persisted per-account history) -- both need the same custom-type
    allow-list, see the comment on graph.compile below."""
    return JsonPlusSerializer(
        allowed_msgpack_modules=[
            ("agent.gatekeeper", "GatekeeperResult"),
            ("agent.resolver", "YearResolution"),
            ("agent.resolver", "GeographyResolution"),
            ("agent.resolver", "MetricResolution"),
            ("agent.resolver", "MetricCandidate"),
            ("metadata.metadata_store", "MetricEntry"),
            ("agent.sql_builder", "BuiltQuery"),
            ("agent.sql_builder", "BuiltRankingQuery"),
            ("agent.executor", "ExecutionResult"),
            ("agent.state", "Turn"),
        ]
    )


def build_pipeline(store: MetadataStore, cfg: Config, checkpointer=None):
    def gatekeeper_node(state: PipelineState) -> dict:
        try:
            result = classify(state["question"], state.get("history", []), cfg)
            return {"gatekeeper": result}
        except LLMError as exc:
            return {"error": str(exc)}

    def route_after_gatekeeper(state: PipelineState) -> str:
        if state.get("error"):
            return "guardrail"
        return "resolve" if state["gatekeeper"].status == "IN_SCOPE" else "guardrail"

    def resolve_node(state: PipelineState) -> dict:
        try:
            gk = state["gatekeeper"]
            year_res = resolve_year(gk.year or None, store)
            updates: dict = {"year_resolution": year_res}
            if year_res.status != "RESOLVED":
                return updates

            geo_res = resolve_geography(gk.state or None, gk.county or None, store)
            updates["geography_resolution"] = geo_res
            if geo_res.status != "RESOLVED":
                return updates

            metric_res = resolve_metric(gk.concept, store, year_res.year)
            updates["metric_resolution"] = metric_res
            return updates
        except Exception as exc:  # noqa: BLE001 - graceful degradation, see module docstring
            return {"error": str(exc)}

    def route_after_resolve(state: PipelineState) -> str:
        if state.get("error"):
            return "guardrail"
        if state.get("year_resolution") is None or state["year_resolution"].status != "RESOLVED":
            return "guardrail"
        if state.get("geography_resolution") is None or state["geography_resolution"].status != "RESOLVED":
            return "guardrail"
        if state.get("metric_resolution") is None or state["metric_resolution"].status != "RESOLVED":
            return "guardrail"
        if state["gatekeeper"].operation == "RANKING":
            return "execute_ranking"
        return "execute"

    def execute_node(state: PipelineState) -> dict:
        try:
            entry = state["metric_resolution"].entry
            geo = state["geography_resolution"]
            built = build_query(entry, state_fips=geo.state_fips, county_fips=geo.county_fips)
            validate(built.sql, store)
            result = execute(cfg, built.sql)
            return {"built_query": built, "execution_result": result}
        except (SqlBuildError, SqlValidationError, ExecutorError) as exc:
            return {"error": str(exc)}

    def route_after_execute(state: PipelineState) -> str:
        if state.get("error"):
            return "guardrail"
        result = state["execution_result"]
        if not result.rows or result.rows[0].get("VALUE") is None:
            return "guardrail"
        return "respond"

    def execute_ranking_node(state: PipelineState) -> dict:
        try:
            gk = state["gatekeeper"]
            entry = state["metric_resolution"].entry
            if entry.aggregation_type != "SUM":
                # A controlled, specific decline -- not a generic error.
                # Ranking a median/rate metric would compound one
                # approximation (AVG-of-block-group-medians) on top of
                # another (comparing 50+ of those noisy numbers), which
                # could produce a genuinely misleading "winner".
                return {"ranking_unsupported_metric": True}
            geo = state["geography_resolution"]
            scope = gk.ranking_scope or "STATE"
            built = build_ranking_query(
                entry,
                sort=gk.sort,
                scope=scope,
                limit=gk.limit,
                state_fips=geo.state_fips if scope == "COUNTY" else None,
            )
            validate(built.sql, store)
            result = execute(cfg, built.sql)
            return {"built_query": built, "execution_result": result}
        except (SqlBuildError, SqlValidationError, ExecutorError) as exc:
            return {"error": str(exc)}

    def route_after_ranking_execute(state: PipelineState) -> str:
        if state.get("error") or state.get("ranking_unsupported_metric"):
            return "guardrail"
        result = state["execution_result"]
        if not result.rows:
            return "guardrail"
        # Null-valued rows (e.g. a geography with zero matching block
        # groups) are filtered per-row in respond_ranking_node, not here
        # -- with N rows now possible, a null can appear anywhere, not
        # just at position 0.
        return "respond_ranking"

    def respond_node(state: PipelineState) -> dict:
        try:
            metric = state["metric_resolution"].entry
            geo = state["geography_resolution"]
            year_res = state["year_resolution"]
            built = state["built_query"]
            row = state["execution_result"].rows[0]
            payload = ResponderInput(
                question=state["question"],
                metric_description=metric.description,
                value=row.get("VALUE"),
                block_group_count=row.get("BLOCK_GROUP_COUNT", 0),
                year=year_res.year,
                state_name=geo.state_name,
                county_name=geo.county_name,
                is_approximate=built.is_approximate,
                caveat=built.caveat,
                year_was_defaulted=year_res.was_defaulted,
            )
            answer = generate_answer(payload, state.get("history", []), cfg)
            return {"answer": answer, "history": [_turn_from_state(state, answer)]}
        except LLMError as exc:
            return {"error": str(exc)}

    def _ranking_row_label(gk, geo, fips: str) -> str:
        if gk.ranking_scope != "COUNTY":
            return store.state_name_by_fips(fips) or f"FIPS {fips}"
        if geo.state_fips is not None:
            # GROUP_FIPS here is the combined 5-digit state+county FIPS
            # (per the COUNTY-scope GROUP BY in build_ranking_query) --
            # strip the 2-digit state prefix to get the bare county FIPS
            # county_name_by_fips expects. Every row shares the same
            # named state, so a bare county name is unambiguous.
            county_fips = fips[len(geo.state_fips):]
            county_name = store.county_name_by_fips(geo.state_fips, county_fips)
            return county_name or f"FIPS {fips}"
        # Nationwide COUNTY ranking: each row can come from a DIFFERENT
        # state, so resolve that row's own leading 2 digits too, and
        # disambiguate same-named counties across states (e.g. several
        # states have a "Washington County").
        row_state_fips, row_county_fips = fips[:2], fips[2:]
        county_name = store.county_name_by_fips(row_state_fips, row_county_fips)
        state_name = store.state_name_by_fips(row_state_fips)
        if county_name and state_name:
            return f"{county_name}, {state_name}"
        return f"FIPS {fips}"

    def respond_ranking_node(state: PipelineState) -> dict:
        try:
            gk = state["gatekeeper"]
            metric = state["metric_resolution"].entry
            year_res = state["year_resolution"]
            geo = state["geography_resolution"]
            rows = [r for r in state["execution_result"].rows if r.get("VALUE") is not None]
            if not rows:
                return {"error": "ranking query returned no non-null rows"}

            ranking_rows = [
                RankingRow(label=_ranking_row_label(gk, geo, row["GROUP_FIPS"]), value=row["VALUE"])
                for row in rows
            ]

            payload = ResponderInput(
                question=state["question"],
                metric_description=metric.description,
                value=None,
                block_group_count=0,
                year=year_res.year,
                state_name=geo.state_name,
                county_name=None,
                is_approximate=False,
                caveat=None,
                year_was_defaulted=year_res.was_defaulted,
                ranking_rows=ranking_rows,
                ranking_sort=gk.sort,
                ranking_truncated=gk.limit_was_all and len(ranking_rows) >= MAX_RANKING_LIMIT,
            )
            answer = generate_answer(payload, state.get("history", []), cfg)
            return {"answer": answer, "history": [_turn_from_state(state, answer)]}
        except LLMError as exc:
            return {"error": str(exc)}

    def guardrail_node(state: PipelineState) -> dict:
        message = build_message(state)
        if state.get("error"):
            # A transient/technical failure (LLM provider error, SQL
            # build/validation/execution error) isn't meaningful
            # conversational context -- persisting it into history would
            # feed the generic failure text back into every future
            # turn's prompt as dead weight, for no benefit. Worse, since
            # the Responder/Gatekeeper prompts include recent history,
            # this measurably shrinks the token budget actually available
            # for genuine content and could contribute to *another* rate
            # limit collision on the very next turn -- verified this
            # compounding effect live: an account's history filled with
            # repeated "something went wrong" turns for the same retried
            # question. Deliberate guardrail declines (out of scope,
            # ambiguous, clarification, "did you mean X?", etc.) are
            # still recorded below -- those genuinely are meaningful
            # context for a natural follow-up.
            return {"answer": message}
        return {"answer": message, "history": [_turn_from_state(state, message)]}

    graph = StateGraph(PipelineState)
    graph.add_node("gatekeeper", gatekeeper_node)
    graph.add_node("resolve", resolve_node)
    graph.add_node("execute", execute_node)
    graph.add_node("execute_ranking", execute_ranking_node)
    graph.add_node("respond", respond_node)
    graph.add_node("respond_ranking", respond_ranking_node)
    graph.add_node("guardrail", guardrail_node)

    graph.set_entry_point("gatekeeper")
    graph.add_conditional_edges("gatekeeper", route_after_gatekeeper, {"resolve": "resolve", "guardrail": "guardrail"})
    graph.add_conditional_edges(
        "resolve", route_after_resolve, {"execute": "execute", "execute_ranking": "execute_ranking", "guardrail": "guardrail"}
    )
    graph.add_conditional_edges("execute", route_after_execute, {"respond": "respond", "guardrail": "guardrail"})
    graph.add_conditional_edges(
        "execute_ranking", route_after_ranking_execute, {"respond_ranking": "respond_ranking", "guardrail": "guardrail"}
    )
    graph.add_edge("respond", END)
    graph.add_edge("respond_ranking", END)
    graph.add_edge("guardrail", END)

    # The checkpointer snapshots the full PipelineState after every node,
    # including the dataclasses that only need to live within a single
    # invocation (gatekeeper/resolution/execution results) -- LangGraph's
    # default serializer warns on unregistered custom types and will
    # reject them outright in a future version. Registering them
    # explicitly here is the supported way to opt in, rather than
    # silently depending on deprecated fallback behavior.
    #
    # `checkpointer` defaults to an in-memory saver (tests, and any caller
    # that doesn't need persistence across process restarts); app.py
    # passes in a SqliteSaver built with the same serde via
    # build_checkpoint_serde() for real per-account persisted history.
    if checkpointer is None:
        checkpointer = InMemorySaver(serde=build_checkpoint_serde())
    return graph.compile(checkpointer=checkpointer)


def run_turn(pipeline, question: str, thread_id: str) -> str:
    config = {"configurable": {"thread_id": thread_id}}
    result = pipeline.invoke({"question": question}, config=config)
    return result["answer"]
