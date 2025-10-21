import os
from functools import lru_cache
from typing import Any, Dict, List, MutableMapping, Optional

import requests
from opensearchpy import OpenSearch

from dotenv import load_dotenv

from .search_utils import get_logger, getenv

load_dotenv()

logger = get_logger(__name__)

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


def _os_index() -> str:
    return getenv("OPENSEARCH_INDEX")

def extract_locations_from_query(user_query: str):
    if nlp is None:
        logger.debug("Spacy model unavailable; skipping spatial entity extraction.")
        return []

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
        response = _os_client().search(
            index=_os_index(),
            body=search_body
        )
        hits = response.get('hits', {}).get('hits', [])
        logger.info(f"Found {len(hits)} spatial search results")
        return hits

    except Exception as e:
        logger.error(f"Error performing spatial search: {e}")
        return []


from .state import EvidenceEntry, ensure_state_shapes, get_query_text, merge_retrieval


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
    if not isinstance(hits, list) or not hits:
        return []

    return merge_retrieval(
        state,
        source=source,
        hits=hits,
        limit=max_total,
        dedupe=dedupe,
    )
