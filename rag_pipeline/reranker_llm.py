from __future__ import annotations

import json
from typing import Any, Dict, List, MutableMapping, Tuple

from .llm_utils import call_llm
from .state import AgentState, ensure_state_shapes, get_query_text


def _summarize_document(entry: Dict[str, Any]) -> Tuple[str, str]:
    document = entry.get("document") or {}
    doc_id = str(document.get("doc_id") or entry.get("metadata", {}).get("hit_id") or "")
    title = (document.get("title") or document.get("element_type") or "Untitled").strip()
    contents = (document.get("contents") or document.get("snippet") or "")[:400].strip()
    summary = f"title={title}\nexcerpt={contents or 'No excerpt available.'}"
    return doc_id, summary


def _build_prompt(query: str, doc_summaries: List[Tuple[str, str]]) -> str:
    items = "\n\n".join(
        f"{idx+1}. doc_id={doc_id}\n{summary}"
        for idx, (doc_id, summary) in enumerate(doc_summaries)
    )
    return (
        "You are an expert relevance judge for a retrieval-augmented QA system.\n"
        "Your task is to re-rank candidate documents by *true semantic relevance* to the user query.\n\n"
        "Follow these rules strictly:\n"
        "1. For each document, assign a relevance score between 0 and 1 (floats allowed).\n"
        "   - Higher = more directly useful for answering the query.\n"
        "   - Lower = unrelated, tangential, vague, or redundant.\n\n"
        "2. Scores MUST show meaningful variance.\n"
        "   - Do NOT give identical scores unless two documents are genuinely equally relevant.\n"
        "   - Avoid ties unless absolutely necessary.\n\n"
        "3. Perform **pairwise comparative judging**:\n"
        "   - Explicitly compare each document’s usefulness to the others.\n"
        "   - If Doc A is more helpful than Doc B, its score MUST be higher.\n\n"
        "4. Penalize documents that:\n"
        "   - Do not contain the specific information the query needs\n"
        "   - Only mention related background but no answer\n"
        "   - Are redundant with higher-ranked documents (lower their score)\n"
        "   - Are off-topic or vague\n\n"
        "5. Reward documents that:\n"
        "   - Directly answer the query\n"
        "   - Contain concrete facts, steps, or definitions relevant to the question\n"
        "   - Are unique and not repeats\n\n"
        "6. Ranking must be strictly sorted from highest score → lowest score.\n\n"
        "7. Return JSON ONLY. Follow this exact schema:\n\n"
        "{\n"
        '  "ranking": [\n'
        '    {\n'
        '      "doc_id": "<id>",\n'
        '      "score": <float 0-1>,\n'
        '      "reason": "<brief justification referencing the snippet>"\n'
        "    },\n"
        "    ... include EVERY document exactly once ...\n"
        "  ]\n"
        "}\n\n"
        "Do NOT invent ids. Reuse the provided doc_id values.\n\n"
        f"User query:\n{query}\n\n"
        f"Candidate documents:\n{items}\n"
    )


def _parse_ranking(response: str) -> List[Dict[str, Any]]:
    try:
        start = response.find("{")
        end = response.rfind("}")
        parsed = json.loads(response[start : end + 1])
        ranking = parsed.get("ranking")
        if isinstance(ranking, list):
            normalized = []
            for item in ranking:
                if not isinstance(item, dict):
                    continue
                doc_id = str(item.get("doc_id") or "").strip()
                if not doc_id:
                    continue
                normalized.append(
                    {
                        "doc_id": doc_id,
                        "score": float(item.get("score", 0.0)),
                        "reason": str(item.get("reason") or "").strip(),
                    }
                )
            return normalized
    except Exception:
        pass
    return []


def rerank_evidence_with_llm(
    state: MutableMapping[str, Any],
    *,
    top_k: int | None = None,
) -> AgentState:
    """
    Use the configured LLM (via llm_utils.call_llm) to reorder evidence.
    """
    state = ensure_state_shapes(state)
    docs: List[Dict[str, Any]] = state["evidence"]["retrieved_documents"]
    if len(docs) <= 1:
        return state

    query = get_query_text(state)
    if not query:
        return state

    doc_summaries: List[Tuple[str, str]] = []
    doc_by_id: Dict[str, Dict[str, Any]] = {}
    for entry in docs:
        doc_id, summary = _summarize_document(entry)
        if not doc_id:
            doc_id = f"doc-{len(doc_summaries)}"
        doc_by_id[doc_id] = entry
        doc_summaries.append((doc_id, summary))
    prompt = _build_prompt(query, doc_summaries)
    response = call_llm(prompt)
    ranking = _parse_ranking(response)

    new_order: List[Dict[str, Any]] = []
    seen = set()
    for item in ranking:
        entry = doc_by_id.get(item["doc_id"])
        if not entry or item["doc_id"] in seen:
            continue
        entry.setdefault("metadata", {})["llm_rerank_reason"] = item.get("reason", "")
        entry["metadata"]["llm_rerank_score"] = item.get("score", 0.0)
        new_order.append(entry)
        seen.add(item["doc_id"])

    for doc_id, entry in doc_by_id.items():
        if doc_id in seen:
            continue
        new_order.append(entry)

    if top_k is not None:
        new_order = new_order[:top_k]

    for idx, entry in enumerate(new_order):
        entry["retrieval_rank"] = idx

    state["evidence"]["retrieved_documents"] = new_order
    return state


__all__ = ["rerank_evidence_with_llm"]
