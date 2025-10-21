# test_rag_pipeline.py
import os, json, importlib

# --- Load environment and ensure HTTP ---
try:
    from dotenv import load_dotenv
    load_dotenv(override=True)
except Exception:
    pass

node = os.getenv("OPENSEARCH_NODE", "http://149.165.159.254:9200").replace("https://", "http://")
os.environ["OPENSEARCH_NODE"] = node
os.environ["ES_HOST"] = node
os.environ["OPENSEARCH_INDEX"] = os.getenv("OPENSEARCH_INDEX", "iguide-platform-embeddings-dev")
os.environ["ES_INDEX"] = os.environ["OPENSEARCH_INDEX"]

# --- Import the pipeline modules ---
from rag_pipeline.state import ensure_state_shapes, summarize_evidence
from rag_pipeline.routing import rag_pipeline
import rag_pipeline.search_core as search_core

# Optional: disable retrievers if some services are down
# (Uncomment only if you see connection errors)
# search_core.retrieve_semantic = lambda s: []
# search_core.retrieve_neo4j    = lambda s: []
# search_core.retrieve_spatial  = lambda s: []

# --- Build the query ---
QUERY = "climate change"
state = ensure_state_shapes({
    "query_information": {"raw_text": QUERY},
    "session_context": {},
    "params": {"top_k": 8, "max_context_tokens": 6000},
    "evidence": {"retrieved_documents": [], "sources": {}},
    "answer": {"final_composed_answer": None, "citations": [], "confidence_score": None},
    "planner_reasoning": {}, "safety_checks": {}, "trace_observability": {},
})

# --- Run the RAG pipeline ---
out = rag_pipeline(state)

# --- Display results ---
print("\n=== RAG TEST: 'climate change' ===")
print("Effective OpenSearch endpoint:", os.environ["OPENSEARCH_NODE"])
print("\n--- Retrieval Summary ---")
print(json.dumps(summarize_evidence(out["evidence"]), indent=2))

print("\n--- Top Evidence ---")
for e in out["evidence"]["retrieved_documents"][:5]:
    doc = e["document"]
    print(f"- {doc.get('doc_id')} | {doc.get('title')} | src={e.get('source')} | score={e.get('score')}")

print("\n--- Final Answer ---")
print(out["answer"]["final_composed_answer"] or "<empty>")

print("\n--- Citations ---")
print(json.dumps(out["answer"]["citations"], indent=2))

print("\n--- Confidence ---")
print(out["answer"]["confidence_score"])
