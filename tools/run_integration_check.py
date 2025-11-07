from __future__ import annotations

import json
import os
from typing import Any, Dict, List, MutableMapping

from rag_pipeline.routing import rag_pipeline
from rag_pipeline.state import (
    AgentState,
    ensure_state_shapes,
    get_query_text,
    build_evidence_items,
    merge_retrieval,
    summarize_evidence,
)


def make_state(query: str, **overrides: Any) -> AgentState:
    base: MutableMapping[str, Any] = {
        "query_information": {"raw_text": query},
        "session_context": {},
        "params": {"top_k": 5, "max_context_tokens": 3000},
        "evidence": {"retrieved_documents": [], "sources": {}},
        "answer": {"final_composed_answer": None, "citations": [], "confidence_score": None},
        "planner_reasoning": {},
        "safety_checks": {},
        "trace_observability": {},
    }
    base.update(overrides)
    return ensure_state_shapes(base)


def assert_agent_state_shape(state: AgentState) -> None:
    assert isinstance(state.get("query_information"), dict), "query_information must be dict"
    assert isinstance(state.get("session_context"), dict), "session_context must be dict"
    evidence = state.get("evidence")
    assert isinstance(evidence, dict), "evidence must be dict"
    assert isinstance(evidence.get("retrieved_documents"), list), "retrieved_documents must be list"
    assert isinstance(evidence.get("sources"), dict), "evidence.sources must be dict"
    answer = state.get("answer")
    assert isinstance(answer, dict), "answer must be dict"
    assert "citations" in answer and isinstance(answer["citations"], list), "citations must be list"
    for key in ("planner_reasoning", "safety_checks", "trace_observability"):
        assert isinstance(state.get(key), dict), f"{key} must be dict"


def _retrieved_as_hits(state: AgentState) -> List[Dict[str, Any]]:
    hits = []
    for entry in state["evidence"]["retrieved_documents"]:
        doc = entry["document"]
        hits.append(
            {
                "_id": doc["doc_id"],
                "_score": entry.get("score", 0.0),
                "_source": doc,
            }
        )
    return hits


def run(query: str = "integration smoke test") -> AgentState:
    state = make_state(query)
    print(f"Running RAG pipeline for query: {get_query_text(state)}")

    result = rag_pipeline(state)
    result = ensure_state_shapes(result)
    assert_agent_state_shape(result)

    summary = summarize_evidence(result["evidence"])
    print("\n=== RETRIEVAL SUMMARY ===")
    print(json.dumps(summary, indent=2))

    print("\n=== FINAL ANSWER ===")
    print(result["answer"]["final_composed_answer"])

    print("\n=== CITATIONS ===")
    print(json.dumps(result["answer"]["citations"], indent=2))

    hits = _retrieved_as_hits(result)
    if hits:
        diagnostic_state = make_state(f"{query} (diagnostic)", params={"top_k": len(hits)})
        diagnostic_state["evidence"]["retrieved_documents"].clear()
        diagnostic_state["evidence"]["sources"].clear()
        merge_retrieval(diagnostic_state, source="diagnostic", hits=hits, limit=len(hits), dedupe=True)

        diagnostic_items = build_evidence_items(hits, source="diagnostic")
        print("\n=== DIAGNOSTIC MERGE ===")
        print(json.dumps([item["document"]["doc_id"] for item in diagnostic_items], indent=2))

    confidence = result["answer"]["confidence_score"]
    print("\n=== CONFIDENCE SCORE ===")
    print(confidence)

    return result


if __name__ == "__main__":
    run()
