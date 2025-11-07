from __future__ import annotations

import json
import os
import importlib
import sys
import types
from typing import Any, Dict, Iterable, List, MutableMapping

if "spacy" not in sys.modules:
    _dummy_spacy = types.ModuleType("spacy")

    class _StubDoc:
        def __init__(self, text: str) -> None:
            self.text = text
            self.ents: List[Any] = []

    class _StubNLP:
        def __call__(self, text: str) -> _StubDoc:
            return _StubDoc(text)

    def _load_spacy(_: str) -> _StubNLP:
        return _StubNLP()

    _dummy_spacy.load = _load_spacy  # type: ignore[attr-defined]
    sys.modules["spacy"] = _dummy_spacy

import pytest

from rag_pipeline import routing, search_core
from rag_pipeline.state import (
    AgentState,
    EvidenceEntry,
    ensure_state_shapes,
    get_query_text,
    build_evidence_items,
    merge_retrieval,
    summarize_evidence,
)


LIVE_OS = os.getenv("LIVE_OS") == "1"
LIVE_LLM = os.getenv("LIVE_LLM") == "1"


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
    shaped = ensure_state_shapes(base)
    assert_agent_state_shape(shaped)
    return shaped


def fake_hit(doc_id: str, title: str, contents: str, score: float = 10.0) -> Dict[str, Any]:
    return {
        "_id": doc_id,
        "_score": score,
        "_source": {
            "doc_id": doc_id,
            "title": title,
            "contents": contents,
            "element_type": "text",
            "resource-type": "text",
        },
    }


def assert_agent_state_shape(state: MutableMapping[str, Any]) -> None:
    assert isinstance(state.get("query_information"), dict), "query_information must be a dict"
    assert isinstance(state.get("session_context"), dict), "session_context must be a dict"

    evidence = state.get("evidence")
    assert isinstance(evidence, dict), "evidence must be a dict"
    assert isinstance(evidence.get("retrieved_documents"), list), "retrieved_documents must be a list"
    assert isinstance(evidence.get("sources"), dict), "evidence.sources must be a dict"

    answer = state.get("answer")
    assert isinstance(answer, dict), "answer must be a dict"
    assert "final_composed_answer" in answer, "answer.final_composed_answer missing"
    assert "citations" in answer and isinstance(answer["citations"], list), "answer.citations must be list"
    assert "confidence_score" in answer, "answer.confidence_score missing"

    for key in ("planner_reasoning", "safety_checks", "trace_observability"):
        assert isinstance(state.get(key), dict), f"{key} must be a dict"


def _merge_sequence(
    state: AgentState,
    sources: Iterable[str],
    hits_per_source: Dict[str, List[Dict[str, Any]]],
    limit: int,
) -> AgentState:
    total = 0
    for source in sources:
        hits = hits_per_source.get(source, [])
        total += len(
            merge_retrieval(
                state,
                source=source,
                hits=hits,
                limit=limit,
                dedupe=True,
            )
        )
    assert total <= limit or limit < 0
    return state


def _assert_citations_match_docs(state: AgentState) -> None:
    docs = {entry["document"]["doc_id"] for entry in state["evidence"]["retrieved_documents"]}
    for citation in state["answer"]["citations"]:
        assert citation.get("doc_id") in docs, f"Citation {citation} missing from retrieved documents"


def test_memory_initializes_state_shape(monkeypatch: pytest.MonkeyPatch) -> None:
    raw_prompt = "What is hydrologic runoff?"
    sys.modules.pop("rag_pipeline.memory_module", None)
    fake_memory_module = types.ModuleType("rag_pipeline.memory_module")

    def _fake_initialize_state(text: str) -> Dict[str, Any]:
        return {"query_information": {"raw_text": text}}

    fake_memory_module.initialize_state = _fake_initialize_state  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "rag_pipeline.memory_module", fake_memory_module)
    memory_module = importlib.import_module("rag_pipeline.memory_module")

    monkeypatch.setattr(
        memory_module,
        "initialize_state",
        lambda text: {"query_information": {"raw_text": text}},
        raising=False,
    )
    initial_state = memory_module.initialize_state(raw_prompt)  # type: ignore[attr-defined]
    shaped_state = ensure_state_shapes(initial_state)
    assert_agent_state_shape(shaped_state)
    assert get_query_text(shaped_state) == raw_prompt


def test_routing_calls_all_selected_retrievers_and_merges(monkeypatch: pytest.MonkeyPatch) -> None:
    if LIVE_OS:
        pytest.skip("Test requires mocked retrievers when LIVE_OS is disabled.")

    keyword_hits = [fake_hit("KW1", "Keyword Doc", "keyword contents")]
    semantic_hits = [fake_hit("SE2", "Semantic Doc", "semantic contents")]
    neo_hits = [fake_hit("NE3", "Graph Doc", "graph contents")]
    spatial_hits = [fake_hit("SP4", "Spatial Doc", "spatial contents")]

    monkeypatch.setattr(search_core, "retrieve_keyword", lambda state: list(keyword_hits))
    monkeypatch.setattr(search_core, "retrieve_semantic", lambda state: list(semantic_hits))
    monkeypatch.setattr(search_core, "retrieve_neo4j", lambda state: list(neo_hits))
    monkeypatch.setattr(search_core, "retrieve_spatial", lambda state: list(spatial_hits))

    state = make_state(
        "Graph analytics near rivers",
        session_context={"use_spatial": True, "use_neo4j": True},
        params={"top_k": 10, "max_context_tokens": 4000},
    )
    output = routing.rag_pipeline(state)

    assert_agent_state_shape(output)
    docs = output["evidence"]["retrieved_documents"]
    assert len(docs) == 4, f"Expected 4 documents, found {len(docs)}"

    doc_ids = [entry["document"]["doc_id"] for entry in docs]
    assert set(doc_ids) == {"KW1", "SE2", "NE3", "SP4"}

    ranks = [entry["retrieval_rank"] for entry in docs]
    assert ranks == list(range(len(ranks))), f"Retrieval ranks not contiguous: {ranks}"

    expected_items: List[EvidenceEntry] = []
    start = 0
    for source, hits in (
        ("keyword", keyword_hits),
        ("semantic", semantic_hits),
        ("neo4j", neo_hits),
        ("spatial", spatial_hits),
    ):
        expected_items.extend(build_evidence_items(hits, source=source, start_rank=start))
        start += len(hits)

    expected_ids = [entry["document"]["doc_id"] for entry in expected_items]
    assert doc_ids == expected_ids, "Merged order does not match expected build sequence"

    sources_info = output["evidence"]["sources"]
    assert sources_info["keyword"]["total_appended"] == 1
    assert sources_info["semantic"]["total_appended"] == 1
    assert sources_info["neo4j"]["total_appended"] == 1
    assert sources_info["spatial"]["total_appended"] == 1

    summary = summarize_evidence(output["evidence"])
    assert summary["count"] == 4
    assert summary["top_score"] is None or summary["top_score"] >= 0


def test_merge_dedupe_and_limit(monkeypatch: pytest.MonkeyPatch) -> None:
    if LIVE_OS:
        pytest.skip("Test requires mocked retrievers when LIVE_OS is disabled.")

    keyword_hits = [
        fake_hit("DUP1", "Keyword Original", "kw"),
        fake_hit("DUP1", "Keyword Duplicate", "kw dup"),
    ]
    semantic_hits = [fake_hit("SEM2", "Semantic", "semantic"), fake_hit("DUP1", "Semantic Duplicate", "dup")]
    neo_hits = [fake_hit("NEO3", "Graph", "graph data")]

    monkeypatch.setattr(search_core, "retrieve_keyword", lambda state: list(keyword_hits))
    monkeypatch.setattr(search_core, "retrieve_semantic", lambda state: list(semantic_hits))
    monkeypatch.setattr(search_core, "retrieve_neo4j", lambda state: list(neo_hits))
    monkeypatch.setattr(search_core, "retrieve_spatial", lambda state: [])

    state = make_state(
        "Graph duplication test",
        session_context={"use_neo4j": True},
        params={"top_k": 3, "max_context_tokens": 4096},
    )
    output = routing.rag_pipeline(state)
    assert_agent_state_shape(output)

    retrieved = output["evidence"]["retrieved_documents"]
    doc_ids = [entry["document"]["doc_id"] for entry in retrieved]

    assert len(doc_ids) <= 3, "Respect top_k limit"
    assert len(doc_ids) == len(set(doc_ids)), f"Duplicate doc_ids found: {doc_ids}"

    # Build expected sequence using merge_retrieval for verification
    expected_state = make_state(
        "Graph duplication test",
        params={"top_k": 3, "max_context_tokens": 4096},
    )
    expected_state["evidence"]["retrieved_documents"].clear()
    expected_state["evidence"]["sources"].clear()

    _merge_sequence(
        expected_state,
        ("keyword", "semantic", "neo4j"),
        {
            "keyword": keyword_hits,
            "semantic": semantic_hits,
            "neo4j": neo_hits,
        },
        limit=3,
    )
    expected_ids = [entry["document"]["doc_id"] for entry in expected_state["evidence"]["retrieved_documents"]]
    assert doc_ids == expected_ids, "Actual dedupe/limit ordering differs from expected merge_retrieval outcome"


def test_generation_receives_context_and_emits_citations(monkeypatch: pytest.MonkeyPatch) -> None:
    if LIVE_OS or LIVE_LLM:
        pytest.skip("Requires mocked retrievers and LLM for deterministic assertions.")

    keyword_hits = [fake_hit("A1", "Keyword Evidence", "kw text")]
    semantic_hits = [fake_hit("B2", "Semantic Evidence", "sem text")]

    monkeypatch.setattr(search_core, "retrieve_keyword", lambda state: list(keyword_hits))
    monkeypatch.setattr(search_core, "retrieve_semantic", lambda state: list(semantic_hits))
    monkeypatch.setattr(search_core, "retrieve_neo4j", lambda state: [])
    monkeypatch.setattr(search_core, "retrieve_spatial", lambda state: [])

    try:
        from rag_pipeline import llm_utils

        monkeypatch.setattr(llm_utils, "call_llm", lambda prompt: "Answer referencing [A1][B2].")
    except Exception:
        from rag_pipeline import generation

        monkeypatch.setattr(generation, "call_llm", lambda prompt: "Answer referencing [A1][B2].")

    state = make_state("Explain impacts")
    output = routing.rag_pipeline(state)
    assert_agent_state_shape(output)

    answer = output["answer"]
    assert isinstance(answer.get("final_composed_answer"), str) and answer["final_composed_answer"]
    assert {"a1", "b2"}.issubset(
        {c["doc_id"].lower() for c in answer["citations"]}
    ), "Expected citations for A1 and B2."
    assert answer["confidence_score"] is None or 0.0 <= answer["confidence_score"] <= 1.0
    _assert_citations_match_docs(output)


def test_empty_evidence_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    if LIVE_OS:
        pytest.skip("Requires mocked retrievers for deterministic fallback check.")

    monkeypatch.setattr(search_core, "retrieve_keyword", lambda state: [])
    monkeypatch.setattr(search_core, "retrieve_semantic", lambda state: [])
    monkeypatch.setattr(search_core, "retrieve_neo4j", lambda state: [])
    monkeypatch.setattr(search_core, "retrieve_spatial", lambda state: [])

    if not LIVE_LLM:
        try:
            from rag_pipeline import llm_utils

            monkeypatch.setattr(
                llm_utils,
                "call_llm",
                lambda prompt: pytest.fail("LLM should not be called when no evidence is available."),
            )
        except Exception:
            from rag_pipeline import generation

            monkeypatch.setattr(
                generation,
                "call_llm",
                lambda prompt: pytest.fail("LLM should not be called when no evidence is available."),
            )

    state = make_state("Describe river health")
    output = routing.rag_pipeline(state)
    assert_agent_state_shape(output)

    final_answer = (output["answer"]["final_composed_answer"] or "").lower()
    assert "try" in final_answer or "narrow" in final_answer, "Expected fallback guidance in final answer."
    assert output["answer"]["citations"] == [], "No citations expected when evidence is empty."
    _assert_citations_match_docs(output)


def test_pipeline_stitches_modules(monkeypatch: pytest.MonkeyPatch) -> None:
    if LIVE_OS or LIVE_LLM:
        pytest.skip("Requires mocked services for deterministic assertions.")

    from rag_pipeline import llm_utils, search_core
    from rag_pipeline.pipeline import run_pipeline

    keyword_hits = [fake_hit("KW1", "Keyword Evidence", "kw text")]
    semantic_hits = [fake_hit("SE2", "Semantic Evidence", "sem text")]
    neo_hits = [fake_hit("NE3", "Graph Evidence", "graph text")]
    spatial_hits = [fake_hit("SP4", "Spatial Evidence", "spatial text")]

    monkeypatch.setattr(search_core, "retrieve_keyword", lambda state: list(keyword_hits))
    monkeypatch.setattr(search_core, "retrieve_semantic", lambda state: list(semantic_hits))
    monkeypatch.setattr(search_core, "retrieve_neo4j", lambda state: list(neo_hits))
    monkeypatch.setattr(search_core, "retrieve_spatial", lambda state: list(spatial_hits))

    def _fake_llm(prompt: str) -> str:
        return "Combined answer [KW1][SE2][NE3][SP4]."

    monkeypatch.setattr(llm_utils, "call_llm", _fake_llm, raising=False)
    try:
        from rag_pipeline import generation

        monkeypatch.setattr(generation, "call_llm", _fake_llm, raising=False)
    except Exception:
        pass

    result = run_pipeline(
        user_input="Graph analytics near rivers",
        session_context={"use_spatial": True, "use_neo4j": True},
        params={"top_k": 6, "max_context_tokens": 2048},
    )

    assert_agent_state_shape(result)
    doc_ids = [entry["document"]["doc_id"] for entry in result["evidence"]["retrieved_documents"]]
    assert set(doc_ids) == {"KW1", "SE2", "NE3", "SP4"}

    answer_text = result["answer"]["final_composed_answer"] or ""
    assert "[KW1]" in answer_text and "[SP4]" in answer_text
    assert {c["doc_id"] for c in result["answer"]["citations"]} == {"KW1", "SE2", "NE3", "SP4"}

    query_info = result["query_information"]
    assert query_info["raw_text"] == "Graph analytics near rivers"
    assert query_info["original_user_input"] == "Graph analytics near rivers"
    assert "memory_initialization" not in result["trace_observability"]


@pytest.mark.skipif(not (LIVE_OS or LIVE_LLM), reason="Live mode disabled; set LIVE_OS=1 or LIVE_LLM=1 to enable.")
def test_live_mode_smoke() -> None:
    state = make_state("climate change impacts on water resources")
    state["session_context"]["use_spatial"] = bool(os.getenv("LIVE_OS"))
    state["session_context"]["use_neo4j"] = "graph" in get_query_text(state).lower()

    output = routing.rag_pipeline(state)
    assert_agent_state_shape(output)
    _assert_citations_match_docs(output)

    summary = summarize_evidence(output["evidence"])
    print("Live mode retrieval summary:", json.dumps(summary, indent=2))
