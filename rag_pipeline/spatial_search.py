from __future__ import annotations

import json
from functools import lru_cache
from typing import Any, Dict, List, MutableMapping, Optional

import requests
from flask import Flask, jsonify, request
try:
    from flask_cors import CORS
except Exception:  # pragma: no cover - optional dependency
    def CORS(app):  # type: ignore
        return app
from opensearchpy import OpenSearch
from dotenv import load_dotenv

from .search_utils import get_logger, getenv, normalize_source_fields, safe_score
from .state import EvidenceEntry, ensure_state_shapes, get_query_text, merge_retrieval

load_dotenv()

logger = get_logger("spatial_search")
app = Flask(__name__)
CORS(app)

try:
    import spacy

    nlp = spacy.load("en_core_web_sm")
except Exception as exc:  # pragma: no cover - optional dependency
    logger.warning("Failed to load Spacy model 'en_core_web_sm': %s", exc)
    nlp = None


@lru_cache(maxsize=1)
def _os_client() -> OpenSearch:
    node = getenv("OPENSEARCH_NODE")
    user = getenv("OPENSEARCH_USERNAME", required=False, default="")
    pwd = getenv("OPENSEARCH_PASSWORD", required=False, default="")
    use_ssl = node.lower().startswith("https")
    return OpenSearch(
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


def _os_index(default: Optional[str] = None) -> str:
    index = default or getenv("OPENSEARCH_INDEX")
    if not index:
        raise RuntimeError("OPENSEARCH_INDEX must not be empty")
    return index


def infer_geo_shape(coords_array: List[List[float]]) -> Dict[str, Any]:
    """
    Infer a GeoJSON shape from coordinate pairs:
      - 1 pair  => point
      - 2 pairs => envelope (normalized to top-left & bottom-right)
      - ≥3 pairs => polygon (auto-closed)
    """
    if not isinstance(coords_array, list) or not coords_array:
        raise ValueError("Coordinates must be a non-empty array of [lon, lat] pairs.")

    if len(coords_array) == 1:
        return {"type": "point", "coordinates": coords_array[0]}

    if len(coords_array) == 2:
        lon1, lat1 = coords_array[0]
        lon2, lat2 = coords_array[1]
        top_left = [min(lon1, lon2), max(lat1, lat2)]
        bottom_right = [max(lon1, lon2), min(lat1, lat2)]
        return {"type": "envelope", "coordinates": [top_left, bottom_right]}

    if coords_array[0] != coords_array[-1]:
        coords_array = [*coords_array, coords_array[0]]
    return {"type": "polygon", "coordinates": [coords_array]}


def _format_search_hits(hits: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    results: List[Dict[str, Any]] = []
    for hit in hits:
        doc_id = str(hit.get("_id", "") or "")
        source = normalize_source_fields(hit.get("_source", {}) or {}, doc_id)
        results.append(
            {
                "_id": doc_id,
                "_score": safe_score(hit.get("_score", 0.0)),
                "_source": source,
            }
        )
    return results


def _scroll_all_documents(search_body: Dict[str, Any], scroll_duration: str = "30s") -> List[Dict[str, Any]]:
    client = _os_client()
    response = client.search(index=_os_index(), body=search_body, scroll=scroll_duration)
    scroll_id = response.get("_scroll_id")

    collected: List[Dict[str, Any]] = []
    while True:
        hits = response.get("hits", {}).get("hits", [])
        if not hits:
            break
        collected.extend(hits)
        response = client.scroll(scroll_id=scroll_id, scroll=scroll_duration)
        scroll_id = response.get("_scroll_id")

    if scroll_id:
        try:
            client.clear_scroll(scroll_id=scroll_id)
        except Exception as exc:  # pragma: no cover - best effort cleanup
            logger.warning("Failed to clear OpenSearch scroll: %s", exc)

    return collected


def spatial_search(
    coords: Any,
    keyword: Optional[str] = None,
    relation: str = "INTERSECTS",
    limit: Any = "unlimited",
    element_type: Optional[str] = None,
) -> List[Dict[str, Any]] | Dict[str, Any]:
    try:
        coords_array = json.loads(coords) if isinstance(coords, str) else coords
        shape = infer_geo_shape(coords_array)
        logger.debug("Inferred geo shape: %s", shape)

        filters: List[Dict[str, Any]] = [
            {
                "geo_shape": {
                    "spatial-bounding-box-geojson": {
                        "shape": shape,
                        "relation": str(relation or "INTERSECTS").upper(),
                    }
                }
            }
        ]
        if element_type:
            filters.append({"term": {"resource-type": element_type}})

        bool_query: Dict[str, Any] = {"bool": {"filter": filters}}
        if keyword:
            bool_query["bool"]["must"] = [
                {
                    "multi_match": {
                        "query": keyword,
                        "fields": ["title^3", "authors^3", "tags^2", "contents", "contributor^3"],
                        "type": "best_fields",
                    }
                }
            ]

        search_body = {"query": bool_query, "track_total_hits": True}
        client = _os_client()
        if str(limit).isdigit():
            size = int(limit)
            response = client.search(index=_os_index(), body={**search_body, "size": size})
            hits = response.get("hits", {}).get("hits", [])
        else:
            hits = _scroll_all_documents(search_body)

        return _format_search_hits(hits)
    except Exception as exc:
        logger.error("Spatial search failed: %s", exc)
        return {"error": str(exc)}


@app.route("/search/spatial", methods=["GET", "OPTIONS"])
def spatial_search_endpoint():
    if request.method == "OPTIONS":
        return "", 200

    try:
        coords = request.args.get("coords")
        keyword = request.args.get("keyword")
        relation = request.args.get("relation", "INTERSECTS")
        limit = request.args.get("limit", "unlimited")
        element_type = request.args.get("element-type")

        if not coords:
            return jsonify({"error": "Missing required query parameter: coords"}), 400

        result = spatial_search(coords, keyword, relation, limit, element_type)
        if isinstance(result, dict) and "error" in result:
            status = 400 if "Missing" in result["error"] else 500
            return jsonify(result), status
        return jsonify(result)
    except Exception as exc:
        logger.error("Spatial search endpoint error: %s", exc)
        return jsonify({"error": str(exc)}), 500


def extract_locations_from_query(user_query: str) -> List[str]:
    if nlp is None:
        logger.debug("Spacy model unavailable; skipping spatial entity extraction.")
        return []
    doc = nlp(user_query)
    return list({ent.text for ent in doc.ents if ent.label_ in ("GPE", "LOC")})


def get_bounding_box(location: str) -> Optional[Dict[str, Any]]:
    api_key = getenv("GOOGLE_MAPS_API_KEY", required=False, default="")
    if not api_key:
        logger.error("Google Maps API key not set.")
        return None

    geocode_url = "https://maps.googleapis.com/maps/api/geocode/json"
    params = {"address": location, "key": api_key}

    try:
        response = requests.get(geocode_url, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()
    except requests.RequestException as exc:
        logger.error("Error fetching bounding box for %s: %s", location, exc)
        return None

    if not data.get("results"):
        logger.warning("No geocode results for location: %s", location)
        return None

    geometry = data["results"][0]["geometry"]
    bounds = geometry.get("bounds") or geometry.get("viewport")
    if not bounds:
        lat = geometry["location"]["lat"]
        lng = geometry["location"]["lng"]
        buffer = 0.1
        bounds = {
            "southwest": {"lat": lat - buffer, "lng": lng - buffer},
            "northeast": {"lat": lat + buffer, "lng": lng + buffer},
        }

    return {
        "type": "polygon",
        "coordinates": [
            [
                [bounds["southwest"]["lng"], bounds["southwest"]["lat"]],
                [bounds["northeast"]["lng"], bounds["southwest"]["lat"]],
                [bounds["northeast"]["lng"], bounds["northeast"]["lat"]],
                [bounds["southwest"]["lng"], bounds["northeast"]["lat"]],
                [bounds["southwest"]["lng"], bounds["southwest"]["lat"]],
            ]
        ],
    }


def get_spatial_search_results(user_query: str, size: int = 10) -> List[Dict[str, Any]]:
    locations = extract_locations_from_query(user_query)
    if not locations:
        return []

    bounding_box = get_bounding_box(locations[0])
    if not bounding_box:
        return []

    search_body = {
        "query": {
            "bool": {
                "should": [{"match": {"title": user_query}}],
                "filter": [
                    {
                        "geo_shape": {
                            "spatial-bounding-box-geojson": {"shape": bounding_box, "relation": "INTERSECTS"}
                        }
                    }
                ],
            }
        },
        "size": size,
    }

    try:
        response = _os_client().search(index=_os_index(), body=search_body)
        hits = response.get("hits", {}).get("hits", [])
        return _format_search_hits(hits)
    except Exception as exc:
        logger.error("Error performing NLP-based spatial search: %s", exc)
        return []


def _extract_spatial_config(state: MutableMapping[str, Any]) -> Dict[str, Any]:
    session_ctx = state.get("session_context") or {}
    config = session_ctx.get("spatial_search") or {}
    if not isinstance(config, dict):
        config = {}
    return config


def retrieve_spatial(state: MutableMapping[str, Any]) -> List[Dict[str, Any]]:
    """
    Retrieve spatially filtered documents using configuration from the shared state.
    
    If coordinates are not provided in the state, falls back to extracting locations
    from the query text using NLP and geocoding via Google Maps API.
    """
    ensure_state_shapes(state)
    config = _extract_spatial_config(state)

    coords = (
        config.get("coords")
        or state.get("session_context", {}).get("spatial_coords")
        or state.get("params", {}).get("spatial_coords")
    )
    
    # === FALLBACK: NLP + Geocoding if coords not in state ===
    if not coords:
        query_text = get_query_text(state) or ""
        if query_text:
            logger.debug("No coordinates in state; attempting NLP extraction from query")
            limit_value: Any = config.get("limit")
            if limit_value is None:
                limit_value = state.get("params", {}).get("top_k", 8)
            
            # Use the NLP + Maps API flow
            result = get_spatial_search_results(query_text, size=limit_value)
            if result:
                logger.debug(f"Spatial retriever found {len(result)} docs via NLP+geocoding")
                return result
        
        logger.debug("Spatial retriever skipped: missing coordinates and no extractable location")
        return []

    # === DIRECT COORDS PATH ===
    keyword = config.get("keyword")
    if keyword is None:
        keyword = get_query_text(state) or None

    relation = config.get("relation", "INTERSECTS")
    if not isinstance(relation, str):
        relation = "INTERSECTS"

    limit_value: Any = config.get("limit")
    if limit_value is None:
        limit_value = state.get("params", {}).get("top_k", 8)

    element_type = config.get("element_type")

    result = spatial_search(
        coords,
        keyword=keyword,
        relation=relation,
        limit=limit_value,
        element_type=element_type,
    )

    if isinstance(result, list):
        return result

    if isinstance(result, dict) and result.get("error"):
        logger.debug("Spatial retriever error suppressed: %s", result["error"])
    return []


def run_spatial_search(
    state: MutableMapping[str, Any],
    *,
    query: Optional[str] = None,
    limit: int = 10,
    max_total: Optional[int] = None,
    dedupe: bool = True,
    source: str = "spatial",
) -> List[EvidenceEntry]:
    ensure_state_shapes(state)
    actual_query = (query or get_query_text(state)).strip()
    if not actual_query:
        logger.debug("Spatial search skipped: empty query.")
        return []

    hits = get_spatial_search_results(actual_query, size=limit)
    if not hits:
        return []

    return merge_retrieval(
        state,
        source=source,
        hits=hits,
        limit=max_total,
        dedupe=dedupe,
    )


if __name__ == "__main__":  # pragma: no cover - CLI usage
    app.run(debug=True)


__all__ = ["spatial_search", "retrieve_spatial", "run_spatial_search", "get_spatial_search_results"]
