import os
import re
import uuid
from math import sqrt
from typing import Dict, List, Optional
import numpy as np
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv
from opensearchpy import NotFoundError, OpenSearch

load_dotenv()

client = OpenSearch(
    hosts=[{'host': os.getenv('OPENSEARCH_NODE'), 'port': 9200}],
    http_auth=(os.getenv('OPENSEARCH_USERNAME'), os.getenv('OPENSEARCH_PASSWORD')),
    use_ssl=False,
    verify_certs=False,
)

MEMORY_INDEX = os.getenv("OPENSEARCH_MEMORY_INDEX", "chat_memory")

EMBEDDING_MODEL = os.getenv("MEMORY_EMBEDDER_MODEL", "all-MiniLM-L6-v2")
_embedder = SentenceTransformer(EMBEDDING_MODEL) if SentenceTransformer else None
CONTEXT_SIMILARITY_THRESHOLD = float(os.getenv("MEMORY_CONTEXT_THRESHOLD", "0.35"))

_PRONOUN_PATTERN = re.compile(r"\b(it|they|them|those|these|that|this|ones?|he|she|we|you)\b", re.IGNORECASE)
REFERENCE_PREFIXES = (
    "and ", "also ", "what about ", "what else ", "any other ", "how about ", "more about ", "any ", "another ",
    "similarly ", "in addition ", "like before ", "as well "
)
TRIGGER_WORDS = {
    "and", "also", "another", "more", "others", "any", "else", "those", "these", "them", "it", "they", "that", "this",
}
TRIGGER_WORDS.update(prefix.strip() for prefix in REFERENCE_PREFIXES)


def create_memory(conversation_name: str) -> str:
    memory_id = str(uuid.uuid4())
    new_memory = {"conversationName": conversation_name, "chat_history": []}

    client.index(index=MEMORY_INDEX, id=memory_id, body=new_memory)
    return memory_id


def get_or_create_memory(memory_id: str) -> Dict:
    try:
        response = client.get(index=MEMORY_INDEX, id=memory_id)
        return response["_source"]
    except NotFoundError:
        new_memory = {"conversationName": f"conversation-{memory_id}", "chat_history": []}
        client.index(index=MEMORY_INDEX, id=memory_id, body=new_memory)
        return new_memory


def get_memory(memory_id: str) -> Optional[Dict]:
    try:
        response = client.get(index=MEMORY_INDEX, id=memory_id)
        return response["_source"]
    except NotFoundError:
        print(f"Memory not found for ID {memory_id}")
        return None


def update_memory(
    memory_id: str,
    user_query: str,
    message_id: str,
    answer: str,
    elements: List[Dict],
    ratings: Optional[Dict] = None,
) -> None:
    try:
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
        print(f"Error updating memory: {err}")
        raise


def delete_memory(memory_id: str) -> None:
    try:
        client.delete(index=MEMORY_INDEX, id=memory_id)
        print(f"Memory deleted for ID {memory_id}")
    except NotFoundError:
        print(f"Memory ID {memory_id} not found for deletion.")
    except Exception as err:
        print(f"Error deleting memory: {err}")
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


def _select_relevant_context(
    chat_history: List[Dict], new_query: str, top_n: int = 3
) -> List[str]:
    candidates = [entry.get("userQuery", "") for entry in chat_history if entry.get("userQuery")]
    if not candidates or top_n <= 0:
        return []
    if _embedder is None:
        raise RuntimeError("SentenceTransformer model is unavailable. Provide an embedder before calling this function.")

    new_vector = _embedder.encode(new_query, convert_to_numpy=True)
    candidate_vectors = _embedder.encode(candidates, convert_to_numpy=True)

    scores = [
        _cosine_similarity(new_vector, candidate_vec) for candidate_vec in candidate_vectors
    ]
    ranked = sorted(zip(scores, candidates), key=lambda item: item[0], reverse=True)

    selected: List[str] = []
    for score, query in ranked:
        if score < CONTEXT_SIMILARITY_THRESHOLD:
            continue
        selected.append(query)
        if len(selected) >= top_n:
            break
    return selected


def _keyword_summary(text: str, max_words: int = 6) -> str:
    tokens = re.findall(r"[a-zA-Z0-9']+", text.lower())
    stopwords = {
        "the", "a", "an", "is", "are", "about", "any", "what", "who", "where", "when", "how", "why", "and", "or", "to", "for", "on",
        "in", "of", "show", "tell", "give", "find", "does", "do",
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
        print(f"Error forming comprehensive user query: {err}")
        raise


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
        client.update(index=MEMORY_INDEX, id=memory_id, body=script, refresh=False)
    except Exception as err:
        print(f"updateRating error: {err}")
        raise
