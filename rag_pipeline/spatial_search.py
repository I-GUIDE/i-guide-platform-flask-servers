import os
import json
import logging
import requests
from flask import Flask, request, jsonify
from flask_cors import CORS
from opensearchpy import OpenSearch
from dotenv import load_dotenv
import spacy

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

nlp = spacy.load("en_core_web_sm")

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
            logger.warning(f"Failed to clear scroll: {e}")

    return all_hits


def format_search_hits(hits):
    return [{
        "score": hit.get('_score', 0.0),
        "document": {
            "_source": {
                "doc_id": hit.get('_id'),
                "element_type": hit['_source'].get('resource-type'),
                "title": hit['_source'].get('title', 'Untitled'),
                "contents": hit['_source'].get('contents', 'No description available')
            }
        }
    } for hit in hits]

def spatial_search(coords, keyword=None, relation='INTERSECTS', limit='unlimited', element_type=None):
    try:
        coords_array = json.loads(coords) if isinstance(coords, str) else coords
        shape = infer_geo_shape(coords_array)

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

        hits = (client.search(index=INDEX_NAME, body={**search_body, 'size': int(limit)})
                ['hits']['hits'] if str(limit).isdigit()
                else scroll_all_documents(search_body))

        return format_search_hits(hits)

    except Exception as e:
        logger.error(f"Spatial search failed: {e}")
        return {'error': str(e)}

def extract_locations_from_query(user_query: str):
    doc = nlp(user_query)
    return list({ent.text for ent in doc.ents if ent.label_ in ("GPE", "LOC")})


def get_bounding_box(location: str):
    api_key = os.getenv('GOOGLE_MAPS_API_KEY')
    if not api_key:
        logger.error("Google Maps API key not set.")
        return None

    geocode_url = "https://maps.googleapis.com/maps/api/geocode/json"
    params = {'address': location, 'key': api_key}

    try:
        response = requests.get(geocode_url, params=params)
        response.raise_for_status()
        data = response.json()

        if not data['results']:
            logger.warning(f"No geocode results for location: {location}")
            return None

        geometry = data['results'][0]['geometry']
        bounds = geometry.get('bounds') or geometry.get('viewport')
        if not bounds:
            lat = geometry['location']['lat']
            lng = geometry['location']['lng']
            buffer = 0.1
            bounds = {
                "southwest": {"lat": lat - buffer, "lng": lng - buffer},
                "northeast": {"lat": lat + buffer, "lng": lng + buffer}
            }

        return {
            "type": "polygon",
            "coordinates": [[
                [bounds["southwest"]["lng"], bounds["southwest"]["lat"]],
                [bounds["northeast"]["lng"], bounds["southwest"]["lat"]],
                [bounds["northeast"]["lng"], bounds["northeast"]["lat"]],
                [bounds["southwest"]["lng"], bounds["northeast"]["lat"]],
                [bounds["southwest"]["lng"], bounds["southwest"]["lat"]]
            ]]
        }

    except requests.RequestException as e:
        logger.error(f"Error fetching bounding box: {e}")
        return None

def get_spatial_search_results(user_query: str, size: int = 10):
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
                "filter": [{
                    "geo_shape": {
                        "spatial-bounding-box-geojson": {
                            "shape": bounding_box,
                            "relation": "INTERSECTS"
                        }
                    }
                }]
            }
        },
        "size": size
    }

    try:
        response = client.search(index=INDEX_NAME, body=search_body)
        hits = response.get('hits', {}).get('hits', [])
        return format_search_hits(hits)
    except Exception as e:
        logger.error(f"Error performing NLP-based spatial search: {e}")
        return []