from __future__ import annotations

from functools import lru_cache
from typing import Any, Dict, List, Optional

from opensearchpy import OpenSearch

from .search_utils import get_logger, getenv, normalize_source_fields, safe_score

log = get_logger("search_keyword")


@lru_cache(maxsize=1)
def _os_client() -> OpenSearch:
    node = getenv("OPENSEARCH_NODE")
    user = getenv("OPENSEARCH_USERNAME", required=False, default="")
    pwd = getenv("OPENSEARCH_PASSWORD", required=False, default="")
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
    index = default or getenv("OPENSEARCH_INDEX")
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
        source = normalize_source_fields(hit.get("_source", {}) or {}, hit_id)
        results.append(
            {
                "_id": hit_id,
                "_score": safe_score(hit.get("_score", 1.0)),
                "_source": source,
            }
        )
    return results


__all__ = ["get_keyword_search_results"]
