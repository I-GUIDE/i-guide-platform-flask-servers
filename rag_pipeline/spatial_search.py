import json
from functools import lru_cache
from typing import Any, Dict, List, MutableMapping, Optional

from flask import Flask, jsonify, request
try:
    from flask_cors import CORS
except Exception:  # pragma: no cover - optional dependency
    def CORS(app):  # type: ignore
        return app
from opensearchpy import OpenSearch
from dotenv import load_dotenv

from .search_utils import get_logger, getenv
from .state import ensure_state_shapes, get_query_text

load_dotenv()

logger = get_logger(__name__)
app = Flask(__name__)
CORS(app)


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


def _os_index() -> str:
    return getenv("OPENSEARCH_INDEX")


def infer_geo_shape(coords_array):
    """
    Infers a GeoJSON shape from coordinate pairs:
      - 1 pair  => Point
      - 2 pairs => Envelope (auto-normalized to top-left & bottom-right)
      - ≥3 pairs => Polygon (auto-closed)
    """
    if not isinstance(coords_array, list) or len(coords_array) == 0:
        raise ValueError('Coordinates must be a non-empty array of [lon, lat] pairs.')

    if len(coords_array) == 1:
        return {
            'type': 'point',
            'coordinates': coords_array[0]
        }

    elif len(coords_array) == 2:
        lon1, lat1 = coords_array[0]
        lon2, lat2 = coords_array[1]

        top_left = [min(lon1, lon2), max(lat1, lat2)]
        bottom_right = [max(lon1, lon2), min(lat1, lat2)]

        return {
            'type': 'envelope',
            'coordinates': [top_left, bottom_right]
        }

    else:
        if coords_array[0] != coords_array[-1]:
            coords_array.append(coords_array[0])

        return {
            'type': 'polygon',
            'coordinates': [coords_array]
        }


def scroll_all_documents(search_body, scroll_duration='30s'):
    all_hits = []
    client = _os_client()
    index = _os_index()
    response = client.search(index=index, body=search_body, scroll=scroll_duration)
    scroll_id = response.get('_scroll_id')

    while True:
        hits = response['hits']['hits']
        if not hits:
            break

        all_hits.extend(hits)

        response = client.scroll(scroll_id=scroll_id, scroll=scroll_duration)
        scroll_id = response.get('_scroll_id')

    if scroll_id:
        try:
            client.clear_scroll(scroll_id=scroll_id)
        except Exception as e:
            logger.warning("Failed to clear OpenSearch scroll: %s", e)

    return all_hits


def spatial_search(coords, keyword=None, relation='INTERSECTS', limit='unlimited', element_type=None):
    try:
        if isinstance(coords, str):
            coords_array = json.loads(coords)
        else:
            coords_array = coords

        if not coords_array:
            return {'error': 'Missing required parameter: coords'}

        shape = infer_geo_shape(coords_array)
        logger.debug("Inferred geo shape: %s", shape)

        filters = [{
            'geo_shape': {
                'spatial-bounding-box-geojson': {
                    'shape': shape,
                    'relation': relation.upper()
                }
            }
        }]

        if element_type:
            filters.append({'term': {'resource-type': element_type}})

        bool_query = {'bool': {'filter': filters}}

        if keyword:
            bool_query['bool']['must'] = [{
                'multi_match': {
                    'query': keyword,
                    'fields': ['title^3', 'authors^3', 'tags^2', 'contents', 'contributor^3'],
                    'type': 'best_fields'
                }
            }]

        search_body = {
            'query': bool_query,
            'track_total_hits': True
        }

        if str(limit).isdigit():
            size = int(limit)
            response = _os_client().search(index=_os_index(), body={**search_body, 'size': size})
            hits = response['hits']['hits']
        else:
            hits = scroll_all_documents(search_body)

        normalized_hits: List[Dict[str, Any]] = []
        for hit in hits:
            source = hit.get('_source', {}) or {}
            doc_id = hit.get('_id')
            normalized_source = {
                "doc_id": doc_id,
                "element_type": source.get('resource-type') or source.get('element_type'),
                "title": source.get('title', 'Untitled'),
                "contents": source.get('contents', 'No description available'),
                "contributor": source.get('contributor'),
                "authors": source.get('authors'),
                "tags": source.get('tags'),
                "thumbnail-image": source.get('thumbnail-image'),
                "spatial-bounding-box-geojson": source.get('spatial-bounding-box-geojson'),
                "spatial-centroid": source.get('spatial-centroid'),
            }
            normalized_hits.append(
                {
                    "_id": doc_id,
                    "_score": hit.get('_score', 0.0),
                    "_source": normalized_source,
                }
            )

        return normalized_hits

    except Exception as e:
        logger.error("Spatial search failed: %s", e)
        return {'error': str(e)}


@app.route('/search/spatial', methods=['GET', 'OPTIONS'])
def spatial_search_endpoint():
    if request.method == 'OPTIONS':
        return '', 200

    try:
        coords = request.args.get('coords')
        keyword = request.args.get('keyword')
        relation = request.args.get('relation', 'INTERSECTS')
        limit = request.args.get('limit', 'unlimited')
        element_type = request.args.get('element-type')

        if not coords:
            return jsonify({'error': 'Missing required query parameter: coords'}), 400

        result = spatial_search(coords, keyword, relation, limit, element_type)

        if 'error' in result:
            return jsonify(result), 400 if 'Missing' in result['error'] else 500

        return jsonify(result)

    except Exception as e:
        logger.error("Spatial search endpoint error: %s", e)
        return jsonify({'error': str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)


def _extract_spatial_config(state: MutableMapping[str, Any]) -> Dict[str, Any]:
    session_ctx = state.get("session_context") or {}
    config = session_ctx.get("spatial_search") or {}
    if not isinstance(config, dict):
        config = {}
    return config


def retrieve_spatial(state: MutableMapping[str, Any]) -> List[Dict[str, Any]]:
    """
    Retrieve spatially filtered documents using configuration from the shared state.
    """
    ensure_state_shapes(state)
    config = _extract_spatial_config(state)

    coords = (
        config.get("coords")
        or state.get("session_context", {}).get("spatial_coords")
        or state.get("params", {}).get("spatial_coords")
    )
    if not coords:
        logger.debug("Spatial retriever skipped: missing coordinates.")
        return []

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


__all__ = ["spatial_search", "retrieve_spatial"]
