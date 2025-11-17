from __future__ import annotations

import json
from typing import Any, Dict, List, MutableMapping, Optional

from .llm_utils import call_llm
from .state import AgentState, ensure_state_shapes, get_query_text

PROMPT_TEMPLATE = (
    "You are auditing a retrieval-augmented answer.\n"
    "Given the user question, the generated answer, and the supporting evidence snippets, "
    "decide whether the answer contains hallucinations (claims not grounded in the evidence).\n\n"
    "Respond ONLY with JSON of the form:\n"
    "{{\n"
    '  "hallucination_detected": true|false,\n'
    '  "severity": "none"|"low"|"medium"|"high",\n'
    '  "issues": [{{"claim": "...", "reason": "..."}}],\n'
    '  "summary": "one sentence verdict"\n'
    "}}\n"
    "Make sure every issue refers to a specific unsupported sentence.\n\n"
    "Question:\n{question}\n\n"
    "Answer:\n{answer}\n\n"
    "Evidence:\n{evidence}\n"
)


def _format_evidence(docs: List[Dict[str, Any]], *, limit: int = 5, max_chars: int = 600) -> str:
    lines: List[str] = []
    for entry in docs[:limit]:
        document = entry.get("document") or {}
        doc_id = str(document.get("doc_id") or entry.get("metadata", {}).get("hit_id") or "")
        title = document.get("title") or document.get("element_type") or "Untitled"
        contents = (document.get("contents") or document.get("snippet") or "").strip()
        snippet = contents[:max_chars]
        lines.append(f"[{doc_id}] {title}\n{snippet}")
    if not lines:
        lines.append("(no evidence supplied)")
    return "\n\n".join(lines)


def _extract_json_block(text: str) -> str:
    if "```" in text:
        start = text.find("```")
        end = text.find("```", start + 3)
        if end != -1:
            candidate = text[start + 3 : end].strip()
            if candidate.lower().startswith("json"):
                candidate = candidate[4:].strip()
            return candidate
    return text


def _parse_response(payload: str) -> Dict[str, Any]:
    try:
        clean = _extract_json_block(payload)
        parsed = json.loads(clean)
        if isinstance(parsed, dict):
            parsed.setdefault("hallucination_detected", False)
            parsed.setdefault("severity", "none")
            parsed.setdefault("issues", [])
            parsed.setdefault("summary", "")
            return parsed
    except Exception:
        pass
    return {
        "hallucination_detected": False,
        "severity": "unknown",
        "issues": [],
        "summary": "LLM response could not be parsed.",
    }


def evaluate_hallucination(
    state: MutableMapping[str, Any],
    *,
    evidence_limit: int = 5,
    snippet_chars: int = 600,
) -> Dict[str, Any]:
    """
    Use the LLM to audit whether the generated answer is grounded in evidence.
    """
    shaped: AgentState = ensure_state_shapes(state)
    question = get_query_text(shaped)
    answer = shaped.get("answer", {}).get("final_composed_answer") or ""
    docs = shaped["evidence"]["retrieved_documents"]

    if not question or not answer or not docs:
        return {
            "hallucination_detected": False,
            "severity": "none",
            "issues": [],
            "summary": "Insufficient data to evaluate hallucinations.",
        }

    evidence_block = _format_evidence(docs, limit=evidence_limit, max_chars=snippet_chars)
    prompt = PROMPT_TEMPLATE.format(question=question, answer=answer, evidence=evidence_block)
    response = call_llm(prompt)
    return _parse_response(response)


__all__ = ["evaluate_hallucination"]
