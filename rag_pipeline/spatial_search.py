import json
import os
from flask import Flask, request, jsonify
from flask_cors import CORS
from opensearchpy import OpenSearch
from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__)
CORS(app)

client = OpenSearch(
    hosts=[{'host': os.getenv('OPENSEARCH_NODE'), 'port': 9200}],
    http_auth=(os.getenv('OPENSEARCH_USERNAME'), os.getenv('OPENSEARCH_PASSWORD')),
    use_ssl=False,
    verify_certs=False
)

INDEX_NAME = os.getenv('OPENSEARCH_INDEX')


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
    response = client.search(index=INDEX_NAME, body=search_body, scroll=scroll_duration)
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
            print(f"[Warning] Failed to clear scroll: {e}")

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
        print(f"[DEBUG] Inferred shape: {shape}")

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
            response = client.search(index=INDEX_NAME, body={**search_body, 'size': size})
            hits = response['hits']['hits']
        else:
            hits = scroll_all_documents(search_body)

        normalized_hits = []
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
        print(f"[ERROR] Spatial search failed: {e}")
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
        print(f"[ERROR] Endpoint exception: {e}")
        return jsonify({'error': str(e)}), 500

app.run(debug=True)
