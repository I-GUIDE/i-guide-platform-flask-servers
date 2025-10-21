from __future__ import annotations

from typing import Any, List, MutableMapping

from .search_keyword import retrieve_keyword
from .search_neo4j import retrieve_neo4j
from .semantic_search import retrieve_semantic
from .spatial_search import retrieve_spatial
from .state import (
    AgentState,
    RoutingDecision,
    ensure_state_shapes,
    get_query_text,
    merge_retrieval,
    summarize_evidence,
)


def _record_decision(decisions: List[RoutingDecision], source: str, reason: str) -> None:
    decisions.append(RoutingDecision(source=source, reason=reason))


def _limit_for(state: MutableMapping[str, Any]) -> int:
    params = state.get("params") or {}
    try:
        return max(1, int(params.get("top_k", 8)))
    except (TypeError, ValueError):
        return 8


def run_retrieval(state: MutableMapping[str, Any]) -> AgentState:
    state = ensure_state_shapes(state)
    limit = _limit_for(state)
    query = get_query_text(state)
    decisions: List[RoutingDecision] = []

    keyword_hits = retrieve_keyword(state)
    appended = merge_retrieval(
        state,
        source="keyword",
        hits=keyword_hits,
        limit=limit,
    )
    _record_decision(decisions, "keyword", f"hits:{len(keyword_hits)} appended:{len(appended)}")

    semantic_hits = retrieve_semantic(state)
    appended = merge_retrieval(
        state,
        source="semantic",
        hits=semantic_hits,
        limit=limit,
    )
    _record_decision(decisions, "semantic", f"hits:{len(semantic_hits)} appended:{len(appended)}")

    session_ctx = state.get("session_context") or {}
    if "graph" in query.lower() or session_ctx.get("use_neo4j"):
        neo_hits = retrieve_neo4j(state)
        appended = merge_retrieval(
            state,
            source="neo4j",
            hits=neo_hits,
            limit=limit,
        )
        _record_decision(decisions, "neo4j", f"hits:{len(neo_hits)} appended:{len(appended)}")

    if session_ctx.get("use_spatial"):
        spatial_hits = retrieve_spatial(state)
        appended = merge_retrieval(
            state,
            source="spatial",
            hits=spatial_hits,
            limit=limit,
        )
        _record_decision(decisions, "spatial", f"hits:{len(spatial_hits)} appended:{len(appended)}")

    trace = state.setdefault("trace_observability", {})
    trace["retrieval_summary"] = summarize_evidence(state["evidence"])
    trace["retrieval_routing_decisions"] = [
        {"source": decision.source, "reason": decision.reason} for decision in decisions
    ]
    return state  # type: ignore[return-value]


__all__ = ["run_retrieval"]
