from __future__ import annotations

import logging
import os
import re
import uuid
from math import sqrt
from typing import Any, Dict, List, Mapping, MutableMapping, Optional

from dotenv import load_dotenv
from opensearchpy import NotFoundError, OpenSearch

try:  # pragma: no cover - optional dependency
    from sentence_transformers import SentenceTransformer
except Exception:  # pragma: no cover - optional dependency
    SentenceTransformer = None  # type: ignore[assignment]

from .state import AgentState, ensure_state_shapes

load_dotenv()

logger = logging.getLogger(__name__)

MEMORY_INDEX = os.getenv("OPENSEARCH_MEMORY_INDEX", "chat_memory")
EMBEDDING_MODEL = os.getenv("MEMORY_EMBEDDER_MODEL", "all-MiniLM-L6-v2")
DEFAULT_STATE_PARAMS: Dict[str, Any] = {"top_k": 8, "max_context_tokens": 6000}

_OPENSEARCH_CLIENT: Optional[OpenSearch] = None
_EMBEDDER: Optional[Any] = None

_PRONOUN_PATTERN = re.compile(r"\b(it|they|them|those|these|that|this|ones?|he|she|we|you)\b", re.IGNORECASE)
REFERENCE_PREFIXES = (
    "and ",
    "also ",
    "what about ",
    "what else ",
    "any other ",
    "how about ",
    "more about ",
    "any ",
    "another ",
    "similarly ",
    "in addition ",
    "like before ",
    "as well ",
)
TRIGGER_WORDS = {
    "and",
    "also",
    "another",
    "more",
    "others",
    "any",
    "else",
    "those",
    "these",
    "them",
    "it",
    "they",
    "that",
    "this",
}
TRIGGER_WORDS.update(prefix.strip() for prefix in REFERENCE_PREFIXES)


def configure_opensearch_client(client: OpenSearch) -> None:
    """
    Override the default OpenSearch client (useful for testing).
    """
    global _OPENSEARCH_CLIENT
    _OPENSEARCH_CLIENT = client


def _get_opensearch_client() -> OpenSearch:
    global _OPENSEARCH_CLIENT
    if _OPENSEARCH_CLIENT is not None:
        return _OPENSEARCH_CLIENT

    node = os.getenv("OPENSEARCH_NODE")
    if not node:
        raise RuntimeError("OPENSEARCH_NODE must be set before using the memory module.")

    user = os.getenv("OPENSEARCH_USERNAME", "")
    pwd = os.getenv("OPENSEARCH_PASSWORD", "")
    use_ssl = node.lower().startswith("https")

    _OPENSEARCH_CLIENT = OpenSearch(
        hosts=[node],
        http_auth=(user, pwd) if (user or pwd) else None,
        use_ssl=use_ssl,
        verify_certs=False,
        ssl_assert_hostname=False,
        ssl_show_warn=False,
        timeout=30,
        max_retries=2,
        retry_on_timeout=True,
    )
    return _OPENSEARCH_CLIENT


def configure_embedder(embedder: Any) -> None:
    """
    Allow tests or callers to inject a custom embedder compatible with SentenceTransformer.encode.
    """
    global _EMBEDDER
    _EMBEDDER = embedder


def _get_embedder() -> Any:
    global _EMBEDDER
    if _EMBEDDER is not None:
        return _EMBEDDER
    if SentenceTransformer is None:
        raise RuntimeError(
            "SentenceTransformer is unavailable. Install sentence-transformers or configure a custom embedder."
        )

    model_name = EMBEDDING_MODEL or "all-MiniLM-L6-v2"
    try:
        _EMBEDDER = SentenceTransformer(model_name)
    except Exception as exc:  # pragma: no cover - network/model download
        logger.error("Failed to load SentenceTransformer model '%s': %s", model_name, exc)
        raise
    return _EMBEDDER


def _coerce_mapping(value: Optional[Mapping[str, Any]]) -> Dict[str, Any]:
    if isinstance(value, Mapping):
        return dict(value)
    return {}


def create_memory(conversation_name: str) -> str:
    memory_id = str(uuid.uuid4())
    new_memory = {"conversationName": conversation_name, "chat_history": []}
    _get_opensearch_client().index(index=MEMORY_INDEX, id=memory_id, body=new_memory)
    return memory_id


def get_or_create_memory(memory_id: str) -> Dict:
    client = _get_opensearch_client()
    try:
        response = client.get(index=MEMORY_INDEX, id=memory_id)
        return response["_source"]
    except NotFoundError:
        new_memory = {"conversationName": f"conversation-{memory_id}", "chat_history": []}
        client.index(index=MEMORY_INDEX, id=memory_id, body=new_memory)
        return new_memory


def get_memory(memory_id: str) -> Optional[Dict]:
    try:
        response = _get_opensearch_client().get(index=MEMORY_INDEX, id=memory_id)
        return response["_source"]
    except NotFoundError:
        logger.info("Memory not found for ID %s", memory_id)
        return None
    except Exception as err:
        logger.error("Error fetching memory %s: %s", memory_id, err)
        raise


def update_memory(
    memory_id: str,
    user_query: str,
    message_id: str,
    answer: str,
    elements: List[Dict],
    ratings: Optional[Dict] = None,
) -> None:
    try:
        client = _get_opensearch_client()
        doc = client.get(index=MEMORY_INDEX, id=memory_id)
        chat_history = doc["_source"].get("chat_history", [])

        entry = {
            "userQuery": user_query,
            "messageId": message_id,
            "answer": answer,
            "elements": elements,
        }
        if ratings:
            entry["ratings"] = ratings

        chat_history.append(entry)
        client.update(index=MEMORY_INDEX, id=memory_id, body={"doc": {"chat_history": chat_history}})
    except Exception as err:
        logger.error("Error updating memory %s: %s", memory_id, err)
        raise


def delete_memory(memory_id: str) -> None:
    try:
        _get_opensearch_client().delete(index=MEMORY_INDEX, id=memory_id)
        logger.info("Memory deleted for ID %s", memory_id)
    except NotFoundError:
        logger.warning("Memory ID %s not found for deletion.", memory_id)
    except Exception as err:
        logger.error("Error deleting memory %s: %s", memory_id, err)
        raise


def _needs_context(new_query: str) -> bool:
    lowered = new_query.strip().lower()
    if not lowered:
        return False
    if any(lowered.startswith(prefix) for prefix in REFERENCE_PREFIXES):
        return True
    tokens = set(lowered.split())
    if tokens & TRIGGER_WORDS:
        return True
    if _PRONOUN_PATTERN.search(lowered):
        return True
    return False


def _cosine_similarity(vec_a, vec_b) -> float:
    vec_a = [float(x) for x in vec_a]
    vec_b = [float(x) for x in vec_b]
    dot_val = sum(a * b for a, b in zip(vec_a, vec_b))
    norm_a = sqrt(sum(a * a for a in vec_a)) or 1e-12
    norm_b = sqrt(sum(b * b for b in vec_b)) or 1e-12
    return dot_val / (norm_a * norm_b)


def _select_relevant_context(chat_history: List[Dict], new_query: str, top_n: int = 3) -> List[str]:
    candidates = [entry.get("userQuery", "") for entry in chat_history if entry.get("userQuery")]
    if not candidates or top_n <= 0:
        return []

    embedder = _get_embedder()
    new_vector = embedder.encode(new_query, convert_to_numpy=True)
    candidate_vectors = embedder.encode(candidates, convert_to_numpy=True)

    scores = [_cosine_similarity(new_vector, candidate_vec) for candidate_vec in candidate_vectors]
    ranked = sorted(zip(scores, candidates), key=lambda item: item[0], reverse=True)

    selected: List[str] = []
    threshold = float(os.getenv("MEMORY_CONTEXT_THRESHOLD", "0.35") or 0.35)
    for score, query in ranked:
        if score < threshold:
            continue
        selected.append(query)
        if len(selected) >= top_n:
            break
    return selected


def _keyword_summary(text: str, max_words: int = 6) -> str:
    tokens = re.findall(r"[a-zA-Z0-9']+", text.lower())
    stopwords = {
        "the",
        "a",
        "an",
        "is",
        "are",
        "about",
        "any",
        "what",
        "who",
        "where",
        "when",
        "how",
        "why",
        "and",
        "or",
        "to",
        "for",
        "on",
        "in",
        "of",
        "show",
        "tell",
        "give",
        "find",
        "does",
        "do",
    }
    keywords: List[str] = []
    for token in tokens:
        if token in stopwords:
            continue
        if token not in keywords:
            keywords.append(token)
        if len(keywords) >= max_words:
            break
    return " ".join(keywords)


def _rewrite_with_context(context_queries: List[str], followup: str) -> str:
    if not context_queries:
        return followup.strip()

    context_focus_parts = []
    for query in context_queries:
        summary = _keyword_summary(query) or query.strip()
        if summary:
            context_focus_parts.append(summary)

    if not context_focus_parts:
        context_focus_parts = [query.strip() for query in context_queries if query.strip()]

    context_focus = " ; ".join(context_focus_parts)
    followup_clean = followup.strip()

    if _PRONOUN_PATTERN.search(followup_clean.lower()):
        merged = _PRONOUN_PATTERN.sub(context_focus, followup_clean)
    else:
        merged = f"{context_focus} {followup_clean}"

    normalized = " ".join(merged.split())
    words = normalized.split()
    if len(words) > 12:
        normalized = " ".join(words[:12])
    return normalized


def form_comprehensive_user_query(memory_id: str, new_user_query: str, recent_k: Optional[int] = None) -> Optional[str]:
    try:
        memory = get_memory(memory_id)
        if not memory:
            return None

        chat_history = memory.get("chat_history", [])
        if recent_k is not None and recent_k > 0:
            chat_history = chat_history[-recent_k:]

        clean_query = new_user_query.strip()
        if not chat_history or not _needs_context(clean_query):
            return clean_query

        context_queries = _select_relevant_context(chat_history, clean_query)
        if not context_queries:
            return clean_query

        return _rewrite_with_context(context_queries, clean_query)
    except Exception as err:
        logger.error("Error forming comprehensive user query: %s", err)
        raise


def initialize_state(
    user_input: str,
    *,
    memory_id: Optional[str] = None,
    session_context: Optional[Mapping[str, Any]] = None,
    params: Optional[Mapping[str, Any]] = None,
    recent_k: Optional[int] = None,
    extra_state: Optional[Mapping[str, Any]] = None,
) -> AgentState:
    """
    Construct an AgentState from raw user input, optionally enriching the query with stored chat memory.
    """
    raw_user_input = str(user_input or "")
    trimmed_input = raw_user_input.strip()
    effective_input = trimmed_input or raw_user_input
    resolved_query = effective_input

    memory_meta: Dict[str, Any] = {}
    if memory_id:
        memory_meta["memory_id"] = memory_id
        if recent_k is not None:
            memory_meta["recent_k"] = recent_k
        try:
            augmented = form_comprehensive_user_query(memory_id, effective_input, recent_k=recent_k)
        except Exception as exc:
            memory_meta["error"] = str(exc)
            logger.warning("Memory augmentation failed for %s: %s", memory_id, exc)
        else:
            if augmented:
                resolved_candidate = str(augmented).strip()
                resolved_query = resolved_candidate or str(augmented)
                memory_meta["augmented_query"] = resolved_query
                if resolved_query != effective_input:
                    memory_meta["original_query"] = effective_input
            else:
                memory_meta["augmented_query"] = None

    query_information: Dict[str, Any] = {"raw_text": resolved_query, "original_user_input": raw_user_input}
    if effective_input and effective_input != resolved_query:
        query_information["initial_query"] = effective_input
    if memory_meta:
        query_information["memory"] = memory_meta

    state: MutableMapping[str, Any] = {
        "query_information": query_information,
        "session_context": _coerce_mapping(session_context),
        "params": {**DEFAULT_STATE_PARAMS, **_coerce_mapping(params)},
        "evidence": {"retrieved_documents": [], "sources": {}},
        "answer": {"final_composed_answer": None, "citations": [], "confidence_score": None},
        "planner_reasoning": {},
        "safety_checks": {},
        "trace_observability": {},
    }

    if memory_id:
        state["session_context"].setdefault("memory_id", memory_id)

    if extra_state:
        for key, value in extra_state.items():
            if key in ("query_information", "session_context", "params") and isinstance(value, Mapping):
                state[key].update(dict(value))
            else:
                state[key] = value

    shaped = ensure_state_shapes(state)
    shaped["query_information"].setdefault("query", shaped["query_information"]["raw_text"])

    trace = shaped.setdefault("trace_observability", {})
    if memory_meta:
        trace["memory_initialization"] = memory_meta
    elif memory_id:
        trace["memory_initialization"] = {"memory_id": memory_id, "augmented_query": None}

    return shaped  # type: ignore[return-value]


def update_rating(memory_id: str, message_id: str, ratings: Dict) -> None:
    try:
        script = {
            "script": {
                "lang": "painless",
                "source": """
                    boolean found = false;
                    for (item in ctx._source.chat_history) {
                      if (item.messageId == params.mid) {
                        item.ratings = params.ratings;
                        found = true;
                        break;
                      }
                    }
                    if (!found) ctx.op = 'none';
                """,
                "params": {"mid": message_id, "ratings": ratings},
            }
        }
        _get_opensearch_client().update(index=MEMORY_INDEX, id=memory_id, body=script, refresh=False)
    except Exception as err:
        logger.error("update_rating error for memory %s: %s", memory_id, err)
        raise


__all__ = [
    "configure_embedder",
    "configure_opensearch_client",
    "create_memory",
    "delete_memory",
    "form_comprehensive_user_query",
    "get_memory",
    "get_or_create_memory",
    "initialize_state",
    "update_memory",
    "update_rating",
]
