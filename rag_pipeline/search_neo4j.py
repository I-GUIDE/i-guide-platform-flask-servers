from __future__ import annotations

from functools import lru_cache
from typing import Any, Dict, List, Optional

from neo4j import Driver, GraphDatabase
from neo4j.graph import Node as _Neo4jNode

from .search_utils import get_logger, getenv, normalize_source_fields, safe_score

log = get_logger("search_neo4j")


@lru_cache(maxsize=1)
def _neo4j_driver() -> Driver:
    uri = (
        getenv("NEO4J_CONNECTION_STRING", required=False)
        or getenv("NEO4J_URI", required=False)
        or getenv("NEO4J_CONNECTION_STRING")
    )
    user = (
        getenv("NEO4J_USER", required=False)
        or getenv("NEO4J_USERNAME", required=False)
        or getenv("NEO4J_USER")
    )
    password = getenv("NEO4J_PASSWORD")
    return GraphDatabase.driver(uri, auth=(user, password), max_connection_lifetime=300)


def _neo4j_db() -> Optional[str]:
    value = getenv("NEO4J_DB", required=False, default="").strip()
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


def _extract_node_from_record(record: Dict[str, Any]) -> Optional[_Neo4jNode]:
    node = record.get("node")
    if isinstance(node, _Neo4jNode):
        return node
    for value in record.values():
        if isinstance(value, _Neo4jNode):
            return value
    return None


def _records_to_hits(records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    hits: List[Dict[str, Any]] = []
    for idx, record in enumerate(records):
        node = _extract_node_from_record(record)
        score = safe_score(record.get("score", 1.0))

        if node is not None:
            properties = dict(node)
            doc_id = properties.get("_id", getattr(node, "element_id", f"node:{idx}"))
        else:
            properties = {k: v for k, v in record.items() if isinstance(v, (str, int, float, list, dict))}
            doc_id = properties.get("doc_id", f"row:{idx}")

        source = normalize_source_fields(properties, str(doc_id))
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


__all__ = ["get_neo4j_search_results"]
