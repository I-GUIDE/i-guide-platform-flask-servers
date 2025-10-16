import os
import logging
import time
import asyncio
from typing import Any, Dict, List, MutableMapping, Optional

import aiohttp
from flask import Flask, request, jsonify
from opensearchpy import OpenSearch
from sentence_transformers import SentenceTransformer

from dotenv import load_dotenv
load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
app = Flask(__name__)


#model = SentenceTransformer('all-MiniLM-L6-v2')

client = OpenSearch(
    hosts=[{'host': os.getenv('OPENSEARCH_NODE'), 'port': 9200}],
    http_auth=(os.getenv('OPENSEARCH_USERNAME'), os.getenv('OPENSEARCH_PASSWORD')),
    use_ssl=False,
    verify_certs=False,  
)

async def get_embedding_from_flask(user_query):
    flask_url = os.getenv("FLASK_EMBEDDING_URL")
    if not flask_url:
        logger.error("FLASK_EMBEDDING_URL environment variable not set.")
        return None

    url = f"{flask_url}/get_embedding"
    payload = {"text": user_query}

    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(url, json=payload) as response:
                if response.status != 200:
                    raise Exception(f"Error from Flask server: {response.status}")
                data = await response.json()
                return data.get("embedding")
    except Exception as e:
        logger.error("Error getting embedding from Flask server: %s", e)
        return None


@app.route('/get_embedding', methods=['POST'])
def get_embedding(text: str) -> List[float]:
    logger.info(f"Fetching embedding from Flask server for input text")
    return asyncio.run(get_embedding_from_flask(text))

def semantic_search(query: str, size: int = 12) -> List[Dict[str, Any]]:
    start_time = time.time()
    logger.info(f"Starting semantic search for query: {query}")

    embedding = get_embedding(query)
    if not embedding:
        logger.error("Failed to get embedding for the query")
        return []

    try:
        response = client.search(
            index=os.getenv('OPENSEARCH_INDEX'),
            body={
                "size": size,
                "query": {
                    "knn": {
                        "contents-embedding": {
                            "vector": embedding,
                            "k": size
                        }
                    }
                }
            }
        )
    except Exception as e:
        logger.error(f"OpenSearch query failed: {e}")
        return []

    results = []
    hits = response.get('hits', {}).get('hits', [])
    logger.info(f"Retrieved {len(hits)} hits from OpenSearch")

    for hit in hits:
        source = hit.get('_source', {})
        score = hit.get('_score', 0)
        doc_id = hit.get('_id', '')

        if 'inner_hits' in hit and 'pdf_chunk_hits' in hit['inner_hits']:
            chunk_hits = hit['inner_hits']['pdf_chunk_hits']['hits']['hits']
            if chunk_hits:
                chunk_source = chunk_hits[0].get('_source', {})
                source['pdf_chunk'] = {
                    'chunk_id': chunk_source.get('chunk_id'),
                    'text': chunk_source.get('text')
                }

        results.append({
            '_id': doc_id,
            '_score': score,
            '_source': source
        })

    end_time = time.time()
    logger.info(f"Semantic search completed in {end_time - start_time:.2f} seconds")
    return results


# --- State-aligned helper ---
from .state import EvidenceEntry, ensure_state_shapes, get_query_text, merge_retrieval


def run_semantic_search(
    state: MutableMapping[str, Any],
    *,
    query: Optional[str] = None,
    limit: int = 12,
    max_total: Optional[int] = None,
    dedupe: bool = True,
    source: str = "semantic",
) -> List[EvidenceEntry]:
    ensure_state_shapes(state)
    actual_query = (query or get_query_text(state)).strip()
    if not actual_query:
        logger.debug("Semantic search skipped: empty query.")
        return []

    hits = semantic_search(actual_query, size=limit)
    if not hits:
        return []

    return merge_retrieval(
        state,
        source=source,
        hits=hits,
        limit=max_total,
        dedupe=dedupe,
    )
