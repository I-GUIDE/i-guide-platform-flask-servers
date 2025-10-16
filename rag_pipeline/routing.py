from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, List, MutableMapping, Optional, Sequence, Tuple

from .search_agents import run_agent_search
from .search_keyword import run_keyword_search
from .semantic_search import run_semantic_search
from .spatial_module import run_spatial_search
from .search_neo4j import run_neo4j_search
from .state import AgentState, EvidenceEntry, RoutingDecision, ensure_state_shapes, get_query_text


Predicate = Callable[[str, AgentState], bool]
Runner = Callable[..., List[EvidenceEntry]]


@dataclass(frozen=True)
class SearchStrategy:
    name: str
    predicate: Predicate
    runner: Runner
    default_limit: int = 12
    optional: bool = True


GEO_KEYWORDS = {
    "geospatial",
    "map",
    "spatial",
    "location",
    "polygon",
    "bounding box",
    "latitude",
    "longitude",
}

GRAPH_KEYWORDS = {
    "graph",
    "network",
    "relationship",
    "connected",
    "node",
    "edge",
}


def _query_text(state: AgentState) -> str:
    return get_query_text(state)


def _context_filters(state: AgentState) -> Sequence[Any]:
    info = state.get("query_information") or {}
    hints = info.get("context_hints") or {}
    filters = hints.get("filters")
    if isinstance(filters, (list, tuple)):
        return filters
    return ()


def _has_geo_signal(query: str, state: AgentState) -> bool:
    query_lower = query.lower()
    if any(key in query_lower for key in GEO_KEYWORDS):
        return True
    filters = _context_filters(state)
    return any(isinstance(item, dict) and "geo" in "".join(item.keys()).lower() for item in filters)


def _should_use_semantic(query: str, state: AgentState) -> bool:
    return bool(query)


def _should_use_agent(query: str, state: AgentState) -> bool:
    if not query:
        return False
    if _has_geo_signal(query, state):
        return True
    filters = _context_filters(state)
    return bool(filters) or len(query.split()) >= 6


def _always(query: str, state: AgentState) -> bool:
    return bool(query)


def _has_graph_signal(query: str) -> bool:
    query_lower = query.lower()
    return any(keyword in query_lower for keyword in GRAPH_KEYWORDS)


def _should_use_neo4j(query: str, state: AgentState) -> bool:
    if not query:
        return False
    if _has_graph_signal(query):
        return True
    filters = _context_filters(state)
    return any(isinstance(item, dict) and "graph" in "".join(item.keys()).lower() for item in filters)


def _should_use_spatial(query: str, state: AgentState) -> bool:
    return _has_geo_signal(query, state)


def default_strategies() -> List[SearchStrategy]:
    return [
        SearchStrategy(
            name="semantic",
            predicate=_should_use_semantic,
            runner=run_semantic_search,
            default_limit=12,
            optional=True,
        ),
        SearchStrategy(
            name="agent",
            predicate=_should_use_agent,
            runner=run_agent_search,
            default_limit=12,
            optional=True,
        ),
        SearchStrategy(
            name="spatial",
            predicate=_should_use_spatial,
            runner=run_spatial_search,
            default_limit=12,
            optional=True,
        ),
        SearchStrategy(
            name="neo4j",
            predicate=_should_use_neo4j,
            runner=run_neo4j_search,
            default_limit=12,
            optional=True,
        ),
        SearchStrategy(
            name="keyword",
            predicate=_always,
            runner=run_keyword_search,
            default_limit=12,
            optional=False,
        ),
    ]


def select_strategies(
    state: AgentState,
    *,
    strategies: Optional[Sequence[SearchStrategy]] = None,
) -> List[SearchStrategy]:
    strategies = list(strategies or default_strategies())
    query = _query_text(state)
    selected: List[SearchStrategy] = []
    for strategy in strategies:
        if strategy.predicate(query, state):
            selected.append(strategy)
        elif not strategy.optional:
            selected.append(strategy)
    return selected


def route_search(
    state: MutableMapping[str, Any],
    *,
    limit_per_source: Optional[int] = None,
    max_total: Optional[int] = None,
    strategies: Optional[Sequence[SearchStrategy]] = None,
) -> Tuple[AgentState, List[RoutingDecision]]:
    """
    Decide which search strategies to run, merge their evidence into the shared state,
    and return the updated state alongside routing decisions.
    """
    ensure_state_shapes(state)
    agent_state: AgentState = state  # type: ignore[assignment]
    selected = select_strategies(agent_state, strategies=strategies)
    query = _query_text(agent_state)
    decisions: List[RoutingDecision] = []

    if not query:
        decisions.append(RoutingDecision(source="routing", reason="no_query"))
        return agent_state, decisions

    for strategy in selected:
        limit = limit_per_source or strategy.default_limit
        reason = "selected"
        try:
            appended = strategy.runner(
                agent_state,
                query=query,
                limit=limit,
                max_total=max_total,
            )
        except Exception as exc:
            decisions.append(RoutingDecision(strategy.name, f"error:{exc}"))
            continue

        reason += f":{len(appended)}"
        decisions.append(RoutingDecision(strategy.name, reason))

        if max_total is not None:
            current = len(agent_state["evidence"]["retrieved_documents"])
            if current >= max_total:
                decisions.append(RoutingDecision("routing", "max_total_reached"))
                break

    return agent_state, decisions
