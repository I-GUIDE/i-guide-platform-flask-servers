from opensearchpy import OpenSearch
import requests
import os
from dotenv import load_dotenv

# Load the .env file
load_dotenv()


# Configuration
opensearch_host = os.getenv('OPENSEARCH_NODE')   # Replace with your OpenSearch host
index_name = os.getenv('OPENSEARCH_INDEX') 
flask_url = 'http://127.0.0.1:5000/get_embedding'  # Flask endpoint URL for embeddings

# Use environment variables for credentials if available
username = os.getenv('OPENSEARCH_USERNAME') 
password = os.getenv('OPENSEARCH_PASSWORD') 



# OpenSearch client configuration
opensearch_client = OpenSearch(
    hosts=[opensearch_host],
    http_auth=(username, password),
    use_ssl=False,
    verify_certs=False
)

# Function to generate embedding for the query using Flask API
def get_query_embedding(question):
    try:
        response = requests.post(flask_url, json={"text": question})
        response.raise_for_status()
        return response.json().get("embedding")
    except requests.exceptions.RequestException as e:
        print(f"Error generating embedding for query: {e}")
        return None

# Function to perform a k-NN search in OpenSearch
def knn_search(index, query_embedding, k=5):
    if query_embedding is None:
        print("Error: Query embedding is None.")
        return

    # k-NN search query
    knn_query = {
        "size": k,  # Number of nearest neighbors to return
        "query": {
            "knn": {
                "contents-embedding": {
                    "vector": query_embedding,
                    "k": k  # Number of neighbors to search
                }
            }
        }
    }

    # Execute the search query
    response = opensearch_client.search(index=index, body=knn_query)
    return response

# Main function to take a sentence as input, generate embedding, and perform search
def main():
    # Input sentence
    question = input("Enter your question or search query: ")

    # Step 1: Generate the embedding for the input question
    query_embedding = get_query_embedding(question)

    # Step 2: Perform k-NN search in OpenSearch using the query embedding
    if query_embedding:
        response = knn_search(index_name, query_embedding, k=5)

        # Step 3: Print out the matching documents
        if response:
            print("\nMatching Documents:")
            for hit in response['hits']['hits']:
                print(f"Document ID: {hit['_id']}, Score: {hit['_score']}")
                print(f"Title: {hit['_source'].get('title', 'No title')}")
                print(f"Contents: {hit['_source'].get('contents', 'No contents')}\n")
        else:
            print("No results found.")

if __name__ == "__main__":
    main()

