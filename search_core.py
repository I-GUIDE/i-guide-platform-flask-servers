from __future__ import annotations

import logging
import math
import os
from functools import lru_cache
from typing import Any, Dict, List, Optional

from neo4j import Driver, GraphDatabase
from neo4j.graph import Node as _Neo4jNode
from opensearchpy import OpenSearch

# ---------- logging ----------
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
logging.basicConfig(level=LOG_LEVEL, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
log = logging.getLogger("search_core")


# ---------- helpers ----------
def _getenv(name: str, required: bool = True, default: Optional[str] = None) -> str:
    value = os.getenv(name, default)
    if required and (value is None or value == ""):
        raise RuntimeError(f"Missing required environment variable: {name}")
    if value and len(value) >= 2 and ((value[0] == value[-1] == '"') or (value[0] == value[-1] == "'")):
        value = value[1:-1]
    return value or ""


def _safe_score(val: Any, default: float = 1.0) -> float:
    try:
        score = float(val)
        return score if math.isfinite(score) else default
    except Exception:
        return default


def _normalize_source_fields(source: Dict[str, Any], fallback_id: str) -> Dict[str, Any]:
    if not isinstance(source, dict):
        source = {}
    source = dict(source)

    source.setdefault("doc_id", fallback_id)
    source.setdefault("title", source.get("name") or "No Title")
    source.setdefault("contents", source.get("abstract") or source.get("description") or "No Content")
    if "element_type" not in source and "resource-type" in source:
        source["element_type"] = source["resource-type"]

    return source


def _extract_node_from_record(record: Dict[str, Any]) -> Optional[_Neo4jNode]:
    node = record.get("node")
    if isinstance(node, _Neo4jNode):
        return node
    for value in record.values():
        if isinstance(value, _Neo4jNode):
            return value
    return None


# ---------- OpenSearch keyword search ----------
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


def _os_index(default: Optional[str] = None) -> str:
    index = default or _getenv("OPENSEARCH_INDEX")
    if not index:
        raise RuntimeError("OPENSEARCH_INDEX must not be empty")
    return index


def get_keyword_search_results(
    user_query: str,
    size: int = 12,
    *,
    client: Optional[OpenSearch] = None,
    index: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """
    Perform a keyword search on the OpenSearch index.

    Environment variables used when `client`/`index` are not supplied:
    - OPENSEARCH_NODE
    - OPENSEARCH_USERNAME / OPENSEARCH_PASSWORD (optional)
    - OPENSEARCH_INDEX

    The optional `client` parameter allows dependency injection for testing.
    """
    query = (user_query or "").strip()
    if not query:
        return []

    size = max(1, min(int(size or 0), 100))
    body = {"size": size, "query": {"match": {"contents": query}}}

    os_client = client or _os_client()
    os_index = _os_index(index)

    try:
        response = os_client.search(index=os_index, body=body)
        payload = response if isinstance(response, dict) else response.body
        hits = payload.get("hits", {}).get("hits", []) or []
    except Exception as exc:
        log.error("OpenSearch keyword query failed: %s", exc)
        return []

    results: List[Dict[str, Any]] = []
    for hit in hits:
        hit_id = str(hit.get("_id", ""))
        source = _normalize_source_fields(hit.get("_source", {}) or {}, hit_id)
        results.append(
            {
                "_id": hit_id,
                "_score": _safe_score(hit.get("_score", 1.0)),
                "_source": source,
            }
        )
    return results


# ---------- Neo4j keyword search ----------
@lru_cache(maxsize=1)
def _neo4j_driver() -> Driver:
    uri = os.getenv("NEO4J_CONNECTION_STRING") or os.getenv("NEO4J_URI") or _getenv("NEO4J_CONNECTION_STRING")
    user = os.getenv("NEO4J_USER") or os.getenv("NEO4J_USERNAME") or _getenv("NEO4J_USER")
    password = _getenv("NEO4J_PASSWORD")
    return GraphDatabase.driver(uri, auth=(user, password), max_connection_lifetime=300)


def _neo4j_db() -> Optional[str]:
    value = os.getenv("NEO4J_DB", "").strip()
    return value or None


def _neo4j_run(cypher: str, params: Dict[str, Any], driver: Optional[Driver] = None) -> List[Dict[str, Any]]:
    database = _neo4j_db()
    session_factory = (driver or _neo4j_driver()).session
    with (session_factory(database=database) if database else session_factory()) as session:
        return list(session.run(cypher, **params))


def _build_neo4j_keyword_cypher() -> str:
    return """
    MATCH (r)
    WHERE
      (r.title IS NOT NULL    AND toLower(r.title)    CONTAINS toLower($q)) OR
      (r.contents IS NOT NULL AND toLower(r.contents) CONTAINS toLower($q)) OR
      (r.tags IS NOT NULL     AND any(tag IN r.tags WHERE toLower(tag) CONTAINS toLower($q)))
    WITH r,
         CASE
           WHEN r.title IS NOT NULL    AND toLower(r.title)    CONTAINS toLower($q) THEN 2.0
           WHEN r.contents IS NOT NULL AND toLower(r.contents) CONTAINS toLower($q) THEN 1.5
           ELSE 1.0
         END AS relevance,
         coalesce(log10(toFloat(coalesce(r.click_count, 0)) + 1), 0) AS popularity,
         coalesce(toFloat(count { (r)--() }), 0) AS connectivity
    WITH r,
         relevance + (popularity * 0.2) + (connectivity * 0.05) AS score
    RETURN r AS node, score
    ORDER BY score DESC
    LIMIT $limit
    """


def _records_to_hits(records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    hits: List[Dict[str, Any]] = []
    for idx, record in enumerate(records):
        node = _extract_node_from_record(record)
        score = _safe_score(record.get("score", 1.0))

        if node is not None:
            properties = dict(node)
            doc_id = properties.get("_id", getattr(node, "element_id", f"node:{idx}"))
        else:
            properties = {k: v for k, v in record.items() if isinstance(v, (str, int, float, list, dict))}
            doc_id = properties.get("doc_id", f"row:{idx}")

        source = _normalize_source_fields(properties, str(doc_id))
        hits.append(
            {
                "_id": str(doc_id),
                "_score": score,
                "_source": source,
            }
        )
    return hits


def get_neo4j_search_results(
    user_query: str,
    limit: int = 12,
    *,
    driver: Optional[Driver] = None,
    cypher: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """
    Perform a keyword-style search against Neo4j without any LLM dependency.

    Environment variables used when `driver` is not supplied:
    - NEO4J_CONNECTION_STRING / NEO4J_URI
    - NEO4J_USER (or NEO4J_USERNAME)
    - NEO4J_PASSWORD
    - NEO4J_DB (optional)

    For testing, pass a mocked `driver` or override the `cypher`.
    Returns hits in the same shape as `get_keyword_search_results`.
    """
    query = (user_query or "").strip()
    if not query:
        return []

    params = {"q": query, "limit": max(1, min(int(limit or 0), 100))}
    cypher_stmt = cypher or _build_neo4j_keyword_cypher()

    try:
        records = _neo4j_run(cypher_stmt, params, driver=driver)
    except Exception as exc:
        log.error("Neo4j keyword query failed: %s", exc)
        return []

    return _records_to_hits(records)


__all__ = [
    "get_keyword_search_results",
    "get_neo4j_search_results",
]
