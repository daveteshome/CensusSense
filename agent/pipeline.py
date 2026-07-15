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
from agent.responder import ResponderInput, generate_answer
from agent.sql_builder import SqlBuildError, build_query
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
    answer: Optional[str]
    error: Optional[str]


def _turn_from_state(state: PipelineState, answer: str) -> Turn:
    gk = state.get("gatekeeper")
    year_res = state.get("year_resolution")
    geo_res = state.get("geography_resolution")
    return Turn(
        question=state["question"],
        metric_phrase=(gk.metric_phrase if gk else None) or None,
        state=(geo_res.state_name if geo_res else None) or (gk.state if gk else None) or None,
        county=(geo_res.county_name if geo_res else None) or None,
        year=(year_res.year if year_res else None) or None,
        answer=answer,
    )


def build_pipeline(store: MetadataStore, cfg: Config):
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

            metric_res = resolve_metric(gk.metric_phrase, store, year_res.year)
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

    def guardrail_node(state: PipelineState) -> dict:
        message = build_message(state)
        return {"answer": message, "history": [_turn_from_state(state, message)]}

    graph = StateGraph(PipelineState)
    graph.add_node("gatekeeper", gatekeeper_node)
    graph.add_node("resolve", resolve_node)
    graph.add_node("execute", execute_node)
    graph.add_node("respond", respond_node)
    graph.add_node("guardrail", guardrail_node)

    graph.set_entry_point("gatekeeper")
    graph.add_conditional_edges("gatekeeper", route_after_gatekeeper, {"resolve": "resolve", "guardrail": "guardrail"})
    graph.add_conditional_edges("resolve", route_after_resolve, {"execute": "execute", "guardrail": "guardrail"})
    graph.add_conditional_edges("execute", route_after_execute, {"respond": "respond", "guardrail": "guardrail"})
    graph.add_edge("respond", END)
    graph.add_edge("guardrail", END)

    # The checkpointer snapshots the full PipelineState after every node,
    # including the dataclasses that only need to live within a single
    # invocation (gatekeeper/resolution/execution results) -- LangGraph's
    # default serializer warns on unregistered custom types and will
    # reject them outright in a future version. Registering them
    # explicitly here is the supported way to opt in, rather than
    # silently depending on deprecated fallback behavior.
    serde = JsonPlusSerializer(
        allowed_msgpack_modules=[
            ("agent.gatekeeper", "GatekeeperResult"),
            ("agent.resolver", "YearResolution"),
            ("agent.resolver", "GeographyResolution"),
            ("agent.resolver", "MetricResolution"),
            ("agent.resolver", "MetricCandidate"),
            ("metadata.metadata_store", "MetricEntry"),
            ("agent.sql_builder", "BuiltQuery"),
            ("agent.executor", "ExecutionResult"),
            ("agent.state", "Turn"),
        ]
    )
    return graph.compile(checkpointer=InMemorySaver(serde=serde))


def run_turn(pipeline, question: str, thread_id: str) -> str:
    config = {"configurable": {"thread_id": thread_id}}
    result = pipeline.invoke({"question": question}, config=config)
    return result["answer"]
