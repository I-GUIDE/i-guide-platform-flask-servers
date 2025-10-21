from __future__ import annotations

from typing import Any, Dict, List, MutableMapping

from .llm_utils import call_llm
from .state import AgentState, ensure_state_shapes, get_query_text


def _context_budget(state: MutableMapping[str, Any]) -> int:
    params = state.get("params") or {}
    try:
        max_tokens = int(params.get("max_context_tokens", 6000))
    except (TypeError, ValueError):
        max_tokens = 6000
    return max_tokens * 4  # coarse chars-per-token estimate


def _build_context_and_citations(docs: List[Dict[str, Any]], max_chars: int) -> Dict[str, Any]:
    ctx_lines: List[str] = []
    citations: List[Dict[str, Any]] = []
    used = 0

    for entry in docs:
        document = entry.get("document") or {}
        doc_id = str(document.get("doc_id", "") or "")
        source = str(entry.get("source", "") or "")
        title = document.get("title") or document.get("element_type") or ""
        contents = (document.get("contents") or "")[:800]
        line = f"[{doc_id}] {title}: {contents}"

        projected = used + len(line) + 1
        if projected > max_chars and ctx_lines:
            break

        ctx_lines.append(line)
        citations.append({"doc_id": doc_id, "source": source})
        used = projected

    if not ctx_lines and docs:
        document = docs[0].get("document") or {}
        doc_id = str(document.get("doc_id", "") or "")
        title = document.get("title") or document.get("element_type") or ""
        contents = (document.get("contents") or "")[:800]
        ctx_lines.append(f"[{doc_id}] {title}: {contents}")
        citations.append({"doc_id": doc_id, "source": docs[0].get("source", "")})

    return {"context": "\n".join(ctx_lines), "citations": citations}


def run_generation(state: MutableMapping[str, Any]) -> AgentState:
    state = ensure_state_shapes(state)
    docs: List[Dict[str, Any]] = state["evidence"]["retrieved_documents"]
    answer: Dict[str, Any] = state["answer"]

    query = get_query_text(state)
    if not docs:
        answer["final_composed_answer"] = (
            f'I could not find evidence for "{query}". '
            "Try narrowing with a location, time range, or data source keyword."
        )
        answer["citations"] = []
        answer["confidence_score"] = 0.1
        return state  # type: ignore[return-value]

    context_info = _build_context_and_citations(docs, _context_budget(state))
    context_block = context_info["context"]
    citations = context_info["citations"]

    system = (
        "You are a factual assistant. Use ONLY the provided evidence. "
        "Cite by [doc_id] after each relevant claim. If something is not in the evidence, say so briefly."
    )
    prompt = (
        f"{system}\n\n"
        f"User question:\n{query}\n\n"
        f"Evidence:\n{context_block}\n\n"
        "Answer with inline [doc_id] citations."
    )

    text = call_llm(prompt).strip()
    if citations and "[" not in text:
        text = f"{text} [{citations[0]['doc_id']}]".strip()

    answer["final_composed_answer"] = text
    answer["citations"] = citations
    answer["confidence_score"] = 0.7
    return state  # type: ignore[return-value]


__all__ = ["run_generation"]
