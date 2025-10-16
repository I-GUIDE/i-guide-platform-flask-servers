from __future__ import annotations

import json
import logging
import math
import os
import re
import time
from functools import lru_cache, wraps
from typing import Any, Dict, List, MutableMapping, Optional, Tuple

import requests
from neo4j import Driver, GraphDatabase
from neo4j.graph import Node as _Neo4jNode
from opensearchpy import OpenSearch

import rag_pipeline.search_core as search_core
from .state import EvidenceEntry, ensure_state_shapes, get_query_text, merge_retrieval

# ---------- logging ----------
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
logging.basicConfig(level=LOG_LEVEL, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
log = logging.getLogger("search_agents")

# ---------- helpers ----------
_SCHEMA_CACHE: Dict[str, Any] = {"ts": 0.0, "val": ""}
_SCHEMA_TTL_SEC = float(os.getenv("SCHEMA_CACHE_TTL_SEC", "300"))
_OS_SCHEMA_CACHE: Dict[str, Any] = {"ts": 0.0, "val": ""}


def _getenv(name: str, required: bool = True, default: Optional[str] = None) -> str:
    value = os.getenv(name, default)
    if required and (value is None or value == ""):
        raise RuntimeError(f"Missing required environment variable: {name}")
    if value and len(value) >= 2 and ((value[0] == value[-1] == '"') or (value[0] == value[-1] == "'")):
        value = value[1:-1]
    return value or ""


def _retry(times: int = 3, base_delay: float = 0.25, exc: Tuple = (Exception,)):
    def decorator(fn):
        @wraps(fn)
        def _inner(*args, **kwargs):
            last = None
            for attempt in range(times):
                try:
                    return fn(*args, **kwargs)
                except exc as err:
                    last = err
                    if attempt == times - 1:
                        raise
                    time.sleep(base_delay * (2 ** attempt))
            raise last

        return _inner

    return decorator


def _safe_score(val: Any, default: float = 1.0) -> float:
    try:
        score = float(val)
        return score if math.isfinite(score) else default
    except Exception:
        return default


def _normalize_source_fields(src: Dict[str, Any], hit_id: str) -> Dict[str, Any]:
    if not isinstance(src, dict):
        src = {}
    src = dict(src)

    if "element_type" not in src and "resource-type" in src:
        src["element_type"] = src["resource-type"]
    src.setdefault("doc_id", hit_id)
    src.setdefault("title", src.get("name") or "No Title")
    src.setdefault("contents", src.get("abstract") or src.get("description") or "No Content")
    return src


def _extract_node_from_record(rec: Dict[str, Any]) -> Optional[_Neo4jNode]:
    node = rec.get("node")
    if isinstance(node, _Neo4jNode):
        return node
    for value in rec.values():
        if isinstance(value, _Neo4jNode):
            return value
    return None


def _transform_thumbnail(value: Any) -> Any:
    """
    Placeholder for utils.generateMultipleResolutionImagesFor from the Node code.
    """
    return value


# ---------- OpenSearch utilities ----------
@lru_cache(maxsize=1)
def _os_client() -> OpenSearch:
    node = _getenv("OPENSEARCH_NODE")
    user = os.getenv("OPENSEARCH_USERNAME", "")
    pwd = os.getenv("OPENSEARCH_PASSWORD", "")
    return OpenSearch(
        hosts=[node],
        http_auth=(user, pwd) if (user or pwd) else None,
        use_ssl=node.lower().startswith("https"),
        verify_certs=False,
        ssl_assert_hostname=False,
        ssl_show_warn=False,
        timeout=30,
        max_retries=2,
        retry_on_timeout=True,
    )


def _os_index() -> str:
    index = _getenv("OPENSEARCH_INDEX")
    if not index:
        raise RuntimeError("OPENSEARCH_INDEX must not be empty")
    return index


@_retry()
def _os_search(body: Dict[str, Any]) -> Dict[str, Any]:
    response = _os_client().search(index=_os_index(), body=body)
    return response if isinstance(response, dict) else response.body


def _os_search_with_params(params: Dict[str, Any]) -> Dict[str, Any]:
    response = _os_client().search(**params)
    return response if isinstance(response, dict) else response.body


def _describe_properties(props: Dict[str, Any], prefix: str = "") -> List[str]:
    lines: List[str] = []
    for name, spec in sorted(props.items()):
        full_name = f"{prefix}{name}"
        dtype = spec.get("type", "object")
        lines.append(f"{full_name}: {dtype}")
        nested = spec.get("properties")
        if isinstance(nested, dict):
            lines.extend(_describe_properties(nested, f"{full_name}."))
    return lines


def get_opensearch_schema() -> str:
    now = time.time()
    cached = _OS_SCHEMA_CACHE.get("val")
    if cached and (now - _OS_SCHEMA_CACHE.get("ts", 0.0)) < _SCHEMA_TTL_SEC:
        return cached

    try:
        mapping = _os_client().indices.get_mapping(index=_os_index())
    except Exception as exc:
        log.error("Failed to fetch OpenSearch mapping: %s", exc)
        return ""

    props = (
        mapping.get(_os_index(), {})
        .get("mappings", {})
        .get("properties", {})
    )
    if not isinstance(props, dict) or not props:
        return ""

    snapshot = "\n".join(_describe_properties(props)[:200])
    _OS_SCHEMA_CACHE.update({"ts": now, "val": snapshot})
    return snapshot


_OS_FORBIDDEN_KEYS = {"delete", "update", "script", "bulk", "reindex", "indices"}


def _sanitize_opensearch_body(body: Dict[str, Any]) -> Dict[str, Any]:
    if not isinstance(body, dict):
        raise ValueError("Agent body must be a JSON object.")

    def _check(obj: Any) -> None:
        if isinstance(obj, dict):
            for key, value in obj.items():
                if key.lower() in _OS_FORBIDDEN_KEYS:
                    raise ValueError(f"Forbidden key in generated body: {key}")
                _check(value)
        elif isinstance(obj, list):
            for item in obj:
                _check(item)

    _check(body)
    size = body.get("size")
    if size is None:
        body["size"] = 12
    else:
        body["size"] = max(1, min(int(size), 100))
    return body


def _openai_endpoint(path: str) -> str:
    base = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1").rstrip("/")
    return f"{base}/{path.lstrip('/')}"


def _llm_chat(messages: List[Dict[str, str]], max_tokens: int = 512, temperature: float = 0.0) -> str:
    url = _openai_endpoint("chat/completions")
    api_key = _getenv("OPENAI_KEY")
    model = (
        os.getenv("OPENAI_CHAT_MODEL")
        or os.getenv("OPENAI_MODEL")
        or "gpt-4o-mini"
    ).strip()

    payload = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    response = requests.post(url, headers=headers, json=payload, timeout=45)
    response.raise_for_status()
    data = response.json()
    try:
        return data["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError) as err:
        raise ValueError(f"Unexpected OpenAI response payload: {data}") from err


def _agent_generate_opensearch_body(user_query: str, schema: str, limit: int) -> Dict[str, Any]:
    system = (
        "You translate natural language search questions into OpenSearch DSL JSON. "
        "Return ONLY JSON for a search body. Never perform destructive operations."
    )
    user_prompt = f"""OpenSearch field inventory:
{schema or '(unknown)'}

Task: Create an OpenSearch search body (JSON) that answers:
"{user_query}"

Constraints:
- Use only read-only APIs (query, sort, aggs).
- Limit results with "size": {limit}.
- Prefer geo filters for spatial hints.
- Prefer date range filters for temporal hints.
- Output JSON only."""

    content = _llm_chat(
        [{"role": "system", "content": system}, {"role": "user", "content": user_prompt}],
        max_tokens=400,
        temperature=0.0,
    ).strip()

    start, end = content.find("{"), content.rfind("}") + 1
    if start == -1 or end <= start:
        raise ValueError(f"Agent returned non-JSON content: {content[:200]}...")
    body = json.loads(content[start:end])
    body.setdefault("size", limit)
    return _sanitize_opensearch_body(body)


def agent_spatial_temporal_search_with_llm(user_query: str, schema: str, limit: int = 12) -> Dict[str, Any]:
    body = _agent_generate_opensearch_body(user_query, schema, limit)
    response = _os_search_with_params({"index": _os_index(), "body": body})
    hits = response.get("hits", {}).get("hits", []) or []
    results: List[Dict[str, Any]] = []
    for hit in hits:
        hit_id = hit.get("_id")
        source = hit.get("_source", {}) or {}
        doc = {
            "_id": hit_id,
            "_score": _safe_score(hit.get("_score", 1.0)),
            "contributor": source.get("contributor"),
            "contents": source.get("contents"),
            "resource-type": source.get("resource-type"),
            "title": source.get("title"),
            "authors": source.get("authors") or [],
            "tags": source.get("tags") or [],
            "thumbnail-image": _transform_thumbnail(source.get("thumbnail-image")),
            "click_count": source.get("click_count", 0),
        }
        results.append(doc)
    return {"results": results}


def get_opensearch_agent_results(user_query: str, limit: int = 12) -> List[Dict[str, Any]]:
    query = (user_query or "").strip()
    if not query:
        return []
    try:
        schema = get_opensearch_schema()
        result = agent_spatial_temporal_search_with_llm(query, schema, limit)
    except Exception as exc:
        log.error("OpenSearch agent query failed: %s", exc)
        # fallback to keyword search for resilience
        fallback_hits = search_core.get_keyword_search_results(query, size=limit)
        return fallback_hits

    docs = result.get("results", [])
    if not isinstance(docs, list):
        return []

    hits: List[Dict[str, Any]] = []
    for doc in docs:
        doc_id = str(doc.get("_id") or "")
        if not doc_id:
            continue
        score = _safe_score(doc.get("_score", 1.0))
        source = {
            "contributor": doc.get("contributor"),
            "contents": doc.get("contents"),
            "resource-type": doc.get("resource-type"),
            "title": doc.get("title"),
            "authors": doc.get("authors") or [],
            "tags": doc.get("tags") or [],
            "thumbnail-image": doc.get("thumbnail-image"),
            "click_count": doc.get("click_count", 0),
        }
        hits.append({"_id": doc_id, "_score": score, "_source": source})
    return hits


def get_embedding_from_flask(user_query: str) -> Optional[List[float]]:
    flask_url = os.getenv("FLASK_EMBEDDING_URL", "").rstrip("/")
    if not flask_url:
        log.error("FLASK_EMBEDDING_URL is not configured.")
        return None
    try:
        response = requests.post(
            f"{flask_url}/get_embedding",
            headers={"Content-Type": "application/json"},
            data=json.dumps({"text": user_query}),
            timeout=15,
        )
        response.raise_for_status()
        payload = response.json()
        embedding = payload.get("embedding")
        if isinstance(embedding, list):
            return embedding
        log.error("Embedding response missing 'embedding': %s", payload)
    except Exception as exc:
        log.error("Error getting embedding from Flask server: %s", exc)
    return None


def get_semantic_search_results(user_query: str, size: int = 12) -> List[Dict[str, Any]]:
    query = (user_query or "").strip()
    if not query:
        return []
    embedding = get_embedding_from_flask(query)
    if not embedding:
        return []

    body = {
        "size": max(1, min(int(size), 50)),
        "query": {
            "bool": {
                "should": [
                    {"knn": {"contents-embedding": {"vector": embedding, "k": 3}}},
                ]
            }
        },
    }

    try:
        response = _os_search(body)
    except Exception as exc:
        log.error("OpenSearch semantic query failed: %s", exc)
        return []

    hits = response.get("hits", {}).get("hits", []) or []
    results: List[Dict[str, Any]] = []
    for hit in hits:
        hit_id = hit.get("_id")
        score = _safe_score(hit.get("_score", 1.0))
        source = hit.get("_source", {}) or {}
        source.pop("contents-embedding", None)
        source.pop("pdf_chunks", None)
        if "thumbnail-image" in source:
            source["thumbnail-image"] = _transform_thumbnail(source["thumbnail-image"])
        doc = {"_id": hit_id, "_score": score}
        doc.update(source)
        results.append(doc)
    return results


# ---------- Neo4j agent search ----------
@lru_cache(maxsize=1)
def _neo4j_driver() -> Driver:
    uri = os.getenv("NEO4J_CONNECTION_STRING") or os.getenv("NEO4J_URI") or _getenv("NEO4J_CONNECTION_STRING")
    user = os.getenv("NEO4J_USER") or os.getenv("NEO4J_USERNAME") or _getenv("NEO4J_USER")
    password = _getenv("NEO4J_PASSWORD")
    return GraphDatabase.driver(uri, auth=(user, password), max_connection_lifetime=300)


def _neo4j_db() -> Optional[str]:
    value = os.getenv("NEO4J_DB", "").strip()
    return value or None


def _neo4j_run(cypher: str, params: Dict[str, Any]) -> List[Dict[str, Any]]:
    database = _neo4j_db()
    with (_neo4j_driver().session(database=database) if database else _neo4j_driver().session()) as session:
        return list(session.run(cypher, **params))


def get_comprehensive_schema() -> str:
    now = time.time()
    cached = _SCHEMA_CACHE.get("val")
    if cached and (now - _SCHEMA_CACHE.get("ts", 0.0)) < _SCHEMA_TTL_SEC:
        return cached

    parts: List[str] = []
    rows = _neo4j_run("CALL db.labels() YIELD label RETURN label", {})
    labels = [r["label"] for r in rows]
    parts.append(f"Labels: {', '.join(labels) if labels else '(none)'}")
    rows = _neo4j_run("CALL db.relationshipTypes() YIELD relationshipType RETURN relationshipType", {})
    rels = [r["relationshipType"] for r in rows]
    parts.append(f"Relationships: {', '.join(rels) if rels else '(none)'}")
    for label in labels[:6]:
        keys_rows = _neo4j_run(f"MATCH (n:`{label}`) WITH n LIMIT 3 RETURN keys(n) AS k", {})
        keys = sorted({key for row in keys_rows for key in row["k"]})
        parts.append(f"Properties[{label}]: {', '.join(keys) if keys else '(none)'}")

    snapshot = "\n".join(parts)
    _SCHEMA_CACHE.update({"ts": now, "val": snapshot})
    return snapshot


_FORBIDDEN = re.compile(r"\b(merge|create|delete|detach|set|load\s+csv|call\s+dbms|apoc\.periodic\.|apoc\.load)\b", re.I)


def _sanitize_cypher(cypher: str) -> str:
    if _FORBIDDEN.search(cypher):
        raise ValueError("Unsafe Cypher detected")
    if not re.search(r"\b(match|call\s+db\.index\.fulltext\.queryNodes|call\s+db\.index\.queryNodes)\b", cypher, re.I):
        raise ValueError("Cypher must contain MATCH or index query")
    return cypher


def _agent_generate_cypher(user_query: str, schema: str, limit: int) -> Tuple[str, Dict[str, Any]]:
    system = (
        "You translate natural language into READ-ONLY Neo4j Cypher. "
        "Return ONLY JSON with keys 'cypher' and 'params'. "
        "Use parameters (e.g., $q, $limit). Never write or modify the graph. "
        "Incorporate node popularity (e.g., log10(click_count + 1)) and graph connectivity (degree or related entities) "
        "into the relevance score along with text matching, and order results by the combined score descending."
    )
    user_prompt = f"""Schema:
{schema}

Task: Write a single read-only Cypher query to answer:
"{user_query}"

Constraints:
- Prefer MATCH patterns on labels/properties present in the schema.
- Capture thematic attributes such as click counts and use OPTIONAL MATCH clauses to bring in related tags/authors when available.
- Combine textual relevance with popularity (e.g., log10(coalesce(r.click_count, 0) + 1)) and connectivity (e.g., size((r)--())) into a single numeric score.
- Return nodes as 'node' plus a numeric 'score'.
- Limit results with $limit
- Use parameters: $q (string) and $limit (int)
- Output JSON: {{"cypher": "...", "params": {{"q": "...", "limit": 12}}}}"""

    content = _llm_chat(
        [{"role": "system", "content": system}, {"role": "user", "content": user_prompt}],
        max_tokens=400,
        temperature=0.0,
    ).strip()

    start, end = content.find("{"), content.rfind("}") + 1
    if start == -1 or end <= start:
        raise ValueError(f"Agent returned non-JSON content: {content[:200]}...")
    obj = json.loads(content[start:end])
    cypher = _sanitize_cypher(obj.get("cypher", ""))
    params = obj.get("params", {}) or {}
    params["q"] = params.get("q", user_query)
    params["limit"] = max(1, min(int(params.get("limit", limit)), 100))
    return cypher, params


@_retry()
def _run_agent_query(user_query: str, schema: str, limit: int) -> List[Dict[str, Any]]:
    cypher, params = _agent_generate_cypher(user_query, schema, limit)
    return _neo4j_run(cypher, params)


def _rows_to_docs(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    docs: List[Dict[str, Any]] = []
    for idx, record in enumerate(rows):
        node = _extract_node_from_record(record)
        score = _safe_score(record.get("score", 1.0))

        if node is not None:
            props = dict(node)
            ref_id = props.get("_id", getattr(node, "element_id", f"node:{idx}"))
        else:
            props = {k: v for k, v in record.items() if isinstance(v, (str, int, float, list, dict))}
            ref_id = props.get("doc_id", f"row:{idx}")

        src = _normalize_source_fields(props, str(ref_id))
        doc_id = str(src.get("doc_id", ref_id))

        authors = src.get("authors")
        if authors is None:
            authors_list: List[Any] = []
        elif isinstance(authors, list):
            authors_list = authors
        else:
            authors_list = [authors]

        tags = src.get("tags")
        if tags is None:
            tags_list: List[Any] = []
        elif isinstance(tags, list):
            tags_list = tags
        else:
            tags_list = [tags]

        doc = {
            "_id": doc_id,
            "_score": score,
            "contributor": src.get("contributor"),
            "contents": src.get("contents"),
            "resource-type": src.get("resource-type") or src.get("element_type"),
            "title": src.get("title"),
            "authors": authors_list,
            "tags": tags_list,
            "thumbnail-image": src.get("thumbnail-image", src.get("thumbnail_image")),
            "click_count": src.get("click_count", 0),
        }
        docs.append(doc)
    return docs


def agent_search_with_llm(user_query: str, schema: str, limit: int = 12) -> Dict[str, Any]:
    try:
        rows = _run_agent_query(user_query, schema, limit)
    except Exception as exc:
        log.warning("Agent query failed (%s); falling back to keyword Cypher.", exc)
        fallback_hits = search_core.get_neo4j_search_results(user_query, limit=limit)
        docs: List[Dict[str, Any]] = []
        for hit in fallback_hits:
            source = hit.get("_source", {})
            docs.append(
                {
                    "_id": hit.get("_id"),
                    "_score": hit.get("_score", 1.0),
                    "contributor": source.get("contributor"),
                    "contents": source.get("contents"),
                    "resource-type": source.get("resource-type"),
                    "title": source.get("title"),
                    "authors": source.get("authors", []),
                    "tags": source.get("tags", []),
                    "thumbnail-image": source.get("thumbnail-image"),
                    "click_count": source.get("click_count", 0),
                }
            )
        return {"results": docs}

    return {"results": _rows_to_docs(rows)}


def get_neo4j_agent_results(user_query: str, limit: int = 12) -> List[Dict[str, Any]]:
    query = (user_query or "").strip()
    if not query:
        return []
    try:
        schema = get_comprehensive_schema()
        result = agent_search_with_llm(query, schema, limit)
    except Exception as exc:
        log.error("Neo4j agent search failed: %s", exc)
        return search_core.get_neo4j_search_results(query, limit=limit)

    docs = result.get("results", [])
    if not isinstance(docs, list):
        log.warning("Neo4j agent returned no usable results.")
        return []

    hits: List[Dict[str, Any]] = []
    for doc in docs:
        doc_id = str(doc.get("_id") or "")
        if not doc_id:
            continue
        score = _safe_score(doc.get("_score", 1.0))
        authors = doc.get("authors")
        if authors is None:
            authors_list: List[Any] = []
        elif isinstance(authors, list):
            authors_list = authors
        else:
            authors_list = [authors]
        tags = doc.get("tags")
        if tags is None:
            tags_list: List[Any] = []
        elif isinstance(tags, list):
            tags_list = tags
        else:
            tags_list = [tags]

        source = {
            "contributor": doc.get("contributor"),
            "contents": doc.get("contents"),
            "resource-type": doc.get("resource-type"),
            "title": doc.get("title"),
            "authors": authors_list,
            "tags": tags_list,
            "thumbnail-image": doc.get("thumbnail-image"),
            "click_count": doc.get("click_count", 0),
        }
        hits.append({"_id": doc_id, "_score": score, "_source": source})
    return hits


# Convenience exports to access basic keyword searches from search_core
get_keyword_search_results = search_core.get_keyword_search_results
get_basic_neo4j_search_results = search_core.get_neo4j_search_results


def run_agent_search(
    state: MutableMapping[str, Any],
    *,
    query: Optional[str] = None,
    limit: int = 12,
    max_total: Optional[int] = None,
    dedupe: bool = True,
    source: str = "agent",
) -> List[EvidenceEntry]:
    ensure_state_shapes(state)
    actual_query = (query or get_query_text(state)).strip()
    if not actual_query:
        log.debug("Agent search skipped: empty query.")
        return []

    hits = get_opensearch_agent_results(actual_query, limit=limit)
    if not hits:
        return []

    return merge_retrieval(
        state,
        source=source,
        hits=hits,
        limit=max_total,
        dedupe=dedupe,
    )


__all__ = [
    "get_keyword_search_results",
    "get_basic_neo4j_search_results",
    "get_semantic_search_results",
    "get_opensearch_agent_results",
    "get_neo4j_agent_results",
    "get_comprehensive_schema",
    "agent_search_with_llm",
    "agent_spatial_temporal_search_with_llm",
    "run_agent_search",
]
