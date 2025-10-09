import os
import logging
import requests
from opensearchpy import OpenSearch
import spacy  
#import geograpy

from dotenv import load_dotenv
load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

nlp = spacy.load("en_core_web_sm")

client = OpenSearch(
    hosts=[{'host': os.getenv('OPENSEARCH_NODE'), 'port': 9200}],
    http_auth=(os.getenv('OPENSEARCH_USERNAME'), os.getenv('OPENSEARCH_PASSWORD')),
    use_ssl=False,
    verify_certs=False, 
)

def extract_locations_from_query(user_query: str):
    doc = nlp(user_query)
    locations = set()
    for ent in doc.ents:
        if ent.label_ in ("GPE", "LOC"):
            locations.add(ent.text)

    locations = list(locations)
    logger.info(f"Extracted locations from query: {locations}")
    return locations
    
def get_bounding_box(location: str):
    """
    Use Google Maps Geocoding API to get bounding box coordinates for a location.
    """
    api_key = os.getenv('GOOGLE_MAPS_API_KEY')
    if not api_key:
        logger.error("Google Maps API key not set.")
        return None

    geocode_url = f"https://maps.googleapis.com/maps/api/geocode/json"
    params = {
        'address': location,
        'key': api_key
    }

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
            # If bounds not available, create a buffer around location
            lat = geometry['location']['lat']
            lng = geometry['location']['lng']
            buffer = 0.1
            bounds = {
                "southwest": {"lat": lat - buffer, "lng": lng - buffer},
                "northeast": {"lat": lat + buffer, "lng": lng + buffer}
            }

        polygon = {
            "type": "polygon",
            "coordinates": [[
                [bounds["southwest"]["lng"], bounds["southwest"]["lat"]],
                [bounds["northeast"]["lng"], bounds["southwest"]["lat"]],
                [bounds["northeast"]["lng"], bounds["northeast"]["lat"]],
                [bounds["southwest"]["lng"], bounds["northeast"]["lat"]],
                [bounds["southwest"]["lng"], bounds["southwest"]["lat"]]  # Close polygon
            ]]
        }

        logger.info(f"Bounding box polygon for {location}: {polygon}")
        return polygon

    except requests.RequestException as e:
        logger.error(f"Error fetching bounding box for {location}: {e}")
        return None

def get_spatial_search_results(user_query: str, size: int = 10):
    """
    Perform spatial search on OpenSearch based on location extracted from query.
    """
    locations = extract_locations_from_query(user_query)

    if not locations:
        logger.warning("No locations extracted from query. Returning empty result.")
        return []

    bounding_box = get_bounding_box(locations[0])
    if not bounding_box:
        logger.warning("Bounding box not found. Returning empty result.")
        return []

    search_body = {
        "query": {
            "bool": {
                "should": [
                    {
                    "match": {
                        "title": user_query
                    }
                }
                ], 
                "filter": [
                    {
                        "geo_shape": {
                            "spatial-bounding-box-geojson": {
                                "shape": bounding_box,
                                "relation": "intersects"
                            }
                        }
                    }
                ]
            }
        },
        "size": size
    }

    try:
        response = client.search(
            index=os.getenv('OPENSEARCH_INDEX'),
            body=search_body
        )
        hits = response.get('hits', {}).get('hits', [])
        logger.info(f"Found {len(hits)} spatial search results")
        return hits

    except Exception as e:
        logger.error(f"Error performing spatial search: {e}")
        return []
