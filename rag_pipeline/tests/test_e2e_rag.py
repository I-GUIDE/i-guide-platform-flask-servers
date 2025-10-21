from __future__ import annotations

from typing import Any, Dict, Iterable, MutableMapping

from ..routing import rag_pipeline
from ..state import ensure_state_shapes


def make_state(query: str) -> MutableMapping[str, Any]:
    state: Dict[str, Any] = {
        "query_information": {"raw_text": query},
        "session_context": {},
        "params": {"top_k": 5, "max_context_tokens": 3000},
        "evidence": {"retrieved_documents": [], "sources": {}},
        "answer": {"final_composed_answer": None, "citations": [], "confidence_score": None},
        "planner_reasoning": {},
        "safety_checks": {},
        "trace_observability": {},
    }
    return ensure_state_shapes(state)


def fake_hit(doc_id: str, title: str, contents: str, score: float = 10.0) -> Dict[str, Any]:
    return {
        "_id": doc_id,
        "_score": score,
        "_source": {
            "doc_id": doc_id,
            "title": title,
            "contents": contents,
            "element_type": "text",
        },
    }


def _patch_retriever(monkeypatch, module, name: str, hits: Iterable[Dict[str, Any]]) -> None:
    monkeypatch.setattr(module, name, lambda state: list(hits))


def test_e2e_answers_with_citations(monkeypatch):
    from .. import llm_utils, search_core

    _patch_retriever(
        monkeypatch,
        search_core,
        "retrieve_keyword",
        [
            fake_hit("A1", "River Flow Dataset", "USGS daily discharge 1980-2020 shows increase post-2000."),
        ],
    )
    _patch_retriever(
        monkeypatch,
        search_core,
        "retrieve_semantic",
        [
            fake_hit("B2", "Watershed Report", "Mean runoff increased by ~4% after 2000 in XYZ basin."),
        ],
    )
    _patch_retriever(monkeypatch, search_core, "retrieve_neo4j", [])
    _patch_retriever(monkeypatch, search_core, "retrieve_spatial", [])

    monkeypatch.setattr(llm_utils, "call_llm", lambda prompt: "Mean runoff rose after 2000 in XYZ basin [A1][B2].")

    state = make_state("How did mean runoff change after 2000 in XYZ basin?")
    out = rag_pipeline(state)

    answer = out["answer"]["final_composed_answer"]
    citations = out["answer"]["citations"]
    assert isinstance(answer, str) and answer.strip()
    assert citations and {c["doc_id"] for c in citations} >= {"A1", "B2"}


def test_empty_evidence_fallback(monkeypatch):
    from .. import llm_utils, search_core

    _patch_retriever(monkeypatch, search_core, "retrieve_keyword", [])
    _patch_retriever(monkeypatch, search_core, "retrieve_semantic", [])
    _patch_retriever(monkeypatch, search_core, "retrieve_neo4j", [])
    _patch_retriever(monkeypatch, search_core, "retrieve_spatial", [])
    monkeypatch.setattr(llm_utils, "call_llm", lambda prompt: "SHOULD NOT BE CALLED")

    state = make_state("What is the average discharge?")
    out = rag_pipeline(state)

    answer = out["answer"]["final_composed_answer"]
    assert "narrow" in answer.lower() or "try" in answer.lower()
    assert out["answer"]["citations"] == []
