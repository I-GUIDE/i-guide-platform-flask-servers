# I-GUIDE Platform Flask Servers

This repository contains servers and scripts for embedding generation, spatial reindexing, and metadata extraction for the I-GUIDE Platform.

## File Structure

```
.gitignore
README.md
embedding-server/
    create_embedding_for_existing.py
    dense_embedding_server.py
    dense_embedding.py
    package.json
    query_embeddint.py
    reindex_wkt_spatial.mjs
    reindex_wkt_spatial.py
    wkt_errors.log
metadata-extraction-server/
    extract_metadata_code_notebooks.py
    extract_metadata.py
    minio_webhook.py
```

## Embedding Server

To run the embedding server on port 5000:

```sh
cd embedding-server
python3 dense_embedding_server.py
```

## Metadata Extraction server(Under Prototyping)

```sh
cd metadata-extraction-server
python3 minio_webhook.py
```

### Main Files

- **[embedding-server/dense_embedding_server.py](embedding-server/dense_embedding_server.py)**  
  Flask server that generates dense vector embeddings using a transformer model.

- **[embedding-server/create_embedding_for_existing.py](embedding-server/create_embedding_for_existing.py)**  
  Script to generate and update embeddings for existing documents in OpenSearch.

- **[embedding-server/query_embeddint.py](embedding-server/query_embeddint.py)**  
  Script to query OpenSearch using vector embeddings.

- **[embedding-server/reindex_wkt_spatial.py](embedding-server/reindex_wkt_spatial.py)**  
  Python script to reindex spatial data from WKT to GeoJSON in OpenSearch.

- **[embedding-server/reindex_wkt_spatial.mjs](embedding-server/reindex_wkt_spatial.mjs)**  
  Node.js script for advanced WKT to GeoJSON reindexing.

## Metadata Extraction Server

- **[metadata-extraction-server/minio_webhook.py](metadata-extraction-server/minio_webhook.py)**  
  Flask webhook to trigger metadata extraction when new files are uploaded to Minio.

- **[metadata-extraction-server/extract_metadata.py](metadata-extraction-server/extract_metadata.py)**  
  Extracts spatial metadata from datasets and attaches it as tags in Minio.

- **[metadata-extraction-server/extract_metadata_code_notebooks.py](metadata-extraction-server/extract_metadata_code_notebooks.py)**  
  Extracts code and notebook metadata for tagging in Minio.

---

For more details, see the code
