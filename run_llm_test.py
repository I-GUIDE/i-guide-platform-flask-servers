import os
import json
from getpass import getpass

# --- Step 1: Configure LLM backend interactively ---
print("=== LLM BACKEND CONFIGURATION ===")
api_key = getpass("Enter your OpenAI API key (sk-...): ").strip()

if not api_key.startswith("sk-"):
    print("❌ Invalid key format. Exiting.")
    exit(1)

os.environ["LLM_BACKEND"] = "openai"
os.environ["OPENAI_API_KEY"] = api_key
os.environ["OPENAI_MODEL"] = "gpt-4o-mini"
os.environ["OPENAI_API_BASE"] = "https://api.openai.com/v1"
os.environ["OPENSEARCH_NODE"] = "http://149.165.159.254:9200"
os.environ["OPENSEARCH_INDEX"] = "iguide-platform-embeddings-dev"

print("\n✅ Configured backend:")
print(f"  LLM_BACKEND = {os.getenv('LLM_BACKEND')}")
print(f"  OPENAI_MODEL = {os.getenv('OPENAI_MODEL')}")
print(f"  OPENSEARCH_NODE = {os.getenv('OPENSEARCH_NODE')}\n")

# --- Step 2: Import your pipeline (after env is set) ---
try:
    from rag_pipeline.state import ensure_state_shapes, summarize_evidence
    from rag_pipeline.routing import rag_pipeline
except ImportError:
    print("❌ Could not import RAG pipeline modules. Run this from project root.")
    exit(1)

# --- Step 3: Run a test query ---
QUERY = input("Enter a test query (default: 'climate change'): ").strip() or "climate change"
print(f"\n=== Running RAG test for: '{QUERY}' ===\n")

state = ensure_state_shapes({
    "query_information": {"raw_text": QUERY},
    "session_context": {},
    "params": {"top_k": 8, "max_context_tokens": 6000},
    "evidence": {"retrieved_documents": [], "sources": {}},
    "answer": {"final_composed_answer": None, "citations": [], "confidence_score": None},
    "planner_reasoning": {}, "safety_checks": {}, "trace_observability": {},
})

out = rag_pipeline(state)

print("\n=== RETRIEVAL SUMMARY ===")
print(json.dumps(summarize_evidence(out["evidence"]), indent=2))

print("\n=== FINAL ANSWER ===")
print(out["answer"]["final_composed_answer"])

print("\n=== CITATIONS ===")
print(json.dumps(out["answer"]["citations"], indent=2))

print("\n=== CONFIDENCE ===")
print(out["answer"]["confidence_score"])
