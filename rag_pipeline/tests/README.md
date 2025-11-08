# RAG Pipeline Tests

This directory contains tests for validating the complete RAG (Retrieval-Augmented Generation) pipeline.

## 🧪 Test Files

### Core Pipeline Tests (Run These First)

#### `test_full_pipeline.py` ⭐
**Complete end-to-end pipeline test with detailed step-by-step output**

Shows the entire flow from memory → routing → search → generation with detailed logging.

```bash
PYTHONPATH=. python rag_pipeline/tests/test_full_pipeline.py
PYTHONPATH=. python rag_pipeline/tests/test_full_pipeline.py "custom query"
```

**Output includes:**
- Step 1: Memory module (chat history augmentation)
- Step 2: Routing & search decisions
- Step 3: Context construction
- Step 4: Full LLM prompt
- Step 5: AnvilGPT generation
- Step 6: Final answer with citations

---

#### `test_state_uniformity.py` ⭐
**Validates that all search modules maintain uniform state structure**

Ensures keyword, semantic, and spatial search all produce compatible `EvidenceEntry` structures for generation.

```bash
PYTHONPATH=. python rag_pipeline/tests/test_state_uniformity.py
```

**Verifies:**
- All evidence entries follow `EvidenceEntry` schema
- Document payloads are uniform across sources
- Generation module processes mixed sources seamlessly
- State structure is preserved throughout pipeline

---

#### `local_rag_test.py` ⭐
**Full pipeline test with connectivity checks**

Tests the complete RAG pipeline with pre-flight connectivity validation.

```bash
PYTHONPATH=. python rag_pipeline/tests/local_rag_test.py
PYTHONPATH=. python rag_pipeline/tests/local_rag_test.py "custom query"
PYTHONPATH=. python rag_pipeline/tests/local_rag_test.py --skip-checks  # Skip connectivity
```

**Features:**
- OpenSearch connectivity validation
- Flask embedding service health check
- AnvilGPT API validation
- Retrieval summary with source breakdown
- Generation output with citations
- Confidence scoring

---

### Service-Specific Debug Tests

#### `test_neo4j_direct.py`
**Direct Neo4j connectivity test (for debugging)**

Tests Neo4j graph database connection independently.

```bash
PYTHONPATH=. python rag_pipeline/tests/test_neo4j_direct.py
```

**Tests:**
- Neo4j driver connection
- Database node count
- Sample node queries
- Configuration validation

**Note:** Use this to debug Neo4j connectivity issues separately from the full pipeline.

---

#### `test_spatial_direct.py`
**Direct spatial search connectivity test**

Tests Google Maps API and OpenSearch spatial fields independently.

```bash
PYTHONPATH=. python rag_pipeline/tests/test_spatial_direct.py
```

**Tests:**
- Google Maps geocoding API
- OpenSearch spatial metadata fields
- Document count with spatial bounding boxes
- Sample spatial document retrieval

---

### Unit/Integration Tests (Pytest)

#### `test_e2e_rag.py`
**End-to-end unit tests with mocked data**

Pytest-based unit tests that use mocked retrievers and LLM responses.

```bash
pytest rag_pipeline/tests/test_e2e_rag.py -v
```

---

#### `test_integration_rag.py`
**Integration tests with mocked services**

Pytest-based integration tests with stubbed external services.

```bash
pytest rag_pipeline/tests/test_integration_rag.py -v
```

---

## 🚀 Quick Start

### Test the Complete Pipeline
```bash
# Full pipeline with all steps
PYTHONPATH=. python rag_pipeline/tests/test_full_pipeline.py

# Verify state uniformity across search modules
PYTHONPATH=. python rag_pipeline/tests/test_state_uniformity.py
```

### Debug Specific Services
```bash
# Debug Neo4j connectivity
PYTHONPATH=. python rag_pipeline/tests/test_neo4j_direct.py

# Debug spatial search
PYTHONPATH=. python rag_pipeline/tests/test_spatial_direct.py
```

### Run Unit Tests
```bash
# Run all pytest tests
pytest rag_pipeline/tests/ -v
```

---

## 📋 Environment Variables Required

Tests read from `rag_pipeline/.env.local`:

```bash
# OpenSearch
OPENSEARCH_NODE=http://149.165.159.254:9200
OPENSEARCH_INDEX=iguide-platform-embeddings-dev
OPENSEARCH_USERNAME=admin
OPENSEARCH_PASSWORD=***

# Flask Embedding Service
FLASK_EMBEDDING_URL=http://149.165.153.129:5000

# AnvilGPT LLM
ANVILGPT_URL=https://anvilgpt.rcac.purdue.edu/api/chat/completions
ANVILGPT_KEY=sk-***
ANVILGPT_MODEL=gpt-oss:120b

# Neo4j (optional)
NEO4J_URI=neo4j://149.165.155.135:7687
NEO4J_USER=***
NEO4J_PASSWORD=***

# Google Maps (optional, for spatial search)
GOOGLE_MAPS_API_KEY=***
```

---

## ✅ Test Coverage

| Component | Test Coverage |
|-----------|---------------|
| Memory Module | ✅ test_full_pipeline.py |
| Keyword Search | ✅ All tests |
| Semantic Search | ✅ All tests |
| Spatial Search | ✅ test_spatial_direct.py, test_state_uniformity.py |
| Neo4j Search | ⚠️ test_neo4j_direct.py (driver issues) |
| LLM Generation | ✅ test_full_pipeline.py, local_rag_test.py |
| State Uniformity | ✅ test_state_uniformity.py |
| Citations | ✅ All pipeline tests |

---

## 🐛 Troubleshooting

### Neo4j Connection Issues
If `test_neo4j_direct.py` fails with segmentation fault (exit code 139), this indicates a Python driver compatibility issue. The Neo4j configuration is correct, but the Python driver may need reinstallation or version update:

```bash
pip uninstall neo4j
pip install neo4j==5.14.0
```

### Spatial Search No Results
If spatial search returns no results:
1. Check `GOOGLE_MAPS_API_KEY` is set
2. Verify OpenSearch documents have `spatial-bounding-box-geojson` field
3. Run `test_spatial_direct.py` to see spatial document count

### LLM Generation Timeouts
If generation times out:
1. Verify `ANVILGPT_URL` and `ANVILGPT_KEY` are correct
2. Check `ANVILGPT_MODEL` matches available models
3. Test connectivity separately with curl

---

## 📝 Test Results Example

```
================================================================================
  COMPLETE RAG PIPELINE TEST
================================================================================

✅ STEP 1: Memory Module → Query initialized
✅ STEP 2: Routing & Search → 5 documents retrieved (keyword: 5)
✅ STEP 3: Context Construction → 4,097 chars
✅ STEP 4: Prompt Construction → 4,381 chars
✅ STEP 5: LLM Generation → gpt-oss:120b (1,935 chars)
✅ STEP 6: Final Answer → 5 citations, confidence: 0.7

🎉 RAG PIPELINE TEST COMPLETED
```





