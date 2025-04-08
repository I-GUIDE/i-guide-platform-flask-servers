from opensearchpy import OpenSearch
import os
import requests
import json
from dotenv import load_dotenv

# Configuration
# opensearch_host = os.getenv('OPENSEARCH_NODE')   # Replace with your OpenSearch host
# index_name = os.getenv('OPENSEARCH_INDEX') 
opensearch_host = 'https://149.165.169.165:9200'
flask_url = 'http://127.0.0.1:5000/get_embedding'  # Flask endpoint URL for embeddings
index_name = 'neo4j-elements'
# Use environment variables for credentials if available
username = os.getenv('OPENSEARCH_USERNAME') 
password = os.getenv('OPENSEARCH_PASSWORD') 

# OpenSearch client with authentication
opensearch_client = OpenSearch(
    ['https://149.165.169.165:9200'],
    http_auth=(username, password),
    use_ssl=False,
    verify_certs=False
)

# Step 1: Fetch Documents Without Embeddings
def fetch_documents_without_embeddings():
    query = {
        "query": {
            "bool": {
                "must_not": {
                    "exists": {
                        "field": "contents-embedding"
                    }
                }
            }
        }
    }
    response = opensearch_client.search(index=index_name, body=query, scroll='2m', size=1000)
    return response

# Step 2: Generate Embedding Using Flask API
def get_embedding(text):
    try:
        response = requests.post(flask_url, json={"text": text})
        response.raise_for_status()
        return response.json().get("embedding")
    except requests.exceptions.RequestException as e:
        print(f"Error generating embedding: {e}")
        return None

# Step 3: Update Document with Embedding
def update_document_with_embedding(doc_id, embedding):
    if embedding is None:
        return
    update_body = {
        "doc": {
            "contents-embedding": embedding
        }
    }
    opensearch_client.update(index=index_name, id=doc_id, body=update_body)

# Iterate over all documents and generate embeddings
scroll_id = None
scroll_size = 1

# Initial scroll search to get documents without embeddings
response = fetch_documents_without_embeddings()
scroll_id = response['_scroll_id']
scroll_size = len(response['hits']['hits'])

while scroll_size > 0:
    # Process each document
    for doc in response['hits']['hits']:
        doc_id = doc['_id']
        contents = doc['_source'].get('contents', '')

        # Generate embedding for the document's contents
        embedding = get_embedding(contents)
        
        # Update the document with the new embedding
        update_document_with_embedding(doc_id, embedding)

    # Fetch the next batch of documents using the scroll ID
    response = opensearch_client.scroll(scroll_id=scroll_id, scroll='2m')
    scroll_id = response['_scroll_id']
    scroll_size = len(response['hits']['hits'])

print("Completed updating all documents with embeddings.")

