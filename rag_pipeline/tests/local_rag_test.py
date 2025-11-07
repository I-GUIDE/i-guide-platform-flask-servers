#!/usr/bin/env python3
"""
Local RAG Pipeline End-to-End Integration Test

This script validates the complete RAG pipeline:
1. Environment configuration from .env.local
2. Connectivity to OpenSearch, Flask embedding service, and AnvilGPT
3. Full retrieval pipeline (keyword, semantic, neo4j, spatial)
4. Generation using AnvilGPT
5. Real data outputs with detailed reporting

Usage:
    python rag_pipeline/tests/local_rag_test.py
    python rag_pipeline/tests/local_rag_test.py "custom query"
    python rag_pipeline/tests/local_rag_test.py --skip-checks "query"
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List

import requests
from dotenv import load_dotenv


# === Configuration ===

def load_environment():
    """Load environment variables from rag_pipeline/.env.local"""
    env_path = Path(__file__).parent.parent / ".env.local"
    if not env_path.exists():
        print(f"❌ ERROR: .env.local not found at {env_path}")
        sys.exit(1)
    
    load_dotenv(env_path)
    print(f"✅ Loaded environment from {env_path}")


def get_config() -> Dict[str, str]:
    """Extract and validate required configuration"""
    config = {
        "opensearch_node": os.getenv("OPENSEARCH_NODE", ""),
        "opensearch_index": os.getenv("OPENSEARCH_INDEX", ""),
        "opensearch_username": os.getenv("OPENSEARCH_USERNAME", ""),
        "opensearch_password": os.getenv("OPENSEARCH_PASSWORD", ""),
        "flask_embedding_url": os.getenv("FLASK_EMBEDDING_URL", ""),
        "anvilgpt_url": os.getenv("ANVILGPT_URL", ""),
        "anvilgpt_key": os.getenv("ANVILGPT_KEY", ""),
        "anvilgpt_model": os.getenv("ANVILGPT_MODEL", "gpt-4"),  # Default to gpt-4
    }
    
    print("\n=== Configuration ===")
    for key, value in config.items():
        if "password" in key.lower() or "key" in key.lower():
            display = f"{value[:8]}..." if value else "[NOT SET]"
        else:
            display = value or "[NOT SET]"
        
        # Add helpful note for model
        if key == "anvilgpt_model":
            if value == "gpt-4":
                display = f"{display} (default - update ANVILGPT_MODEL in .env.local if needed)"
        
        print(f"  {key}: {display}")
    
    # Validate required fields (API key is optional as some LLM endpoints don't need it)
    required = ["opensearch_node", "opensearch_index", "flask_embedding_url", "anvilgpt_url"]
    missing = [k for k in required if not config[k]]
    if missing:
        print(f"\n❌ ERROR: Missing required configuration: {', '.join(missing)}")
        sys.exit(1)
    
    if not config["anvilgpt_key"]:
        print("\n⚠️  WARNING: ANVILGPT_KEY not set. Will attempt connection without authentication.")
    
    return config


# === Connectivity Checks ===

def check_opensearch(config: Dict[str, str]) -> bool:
    """Test OpenSearch connectivity"""
    print("\n=== OpenSearch Connectivity ===")
    node = config["opensearch_node"]
    auth = None
    if config["opensearch_username"] and config["opensearch_password"]:
        auth = (config["opensearch_username"], config["opensearch_password"])
    
    try:
        url = f"{node}/_count"
        response = requests.get(url, auth=auth, verify=False, timeout=10)
        response.raise_for_status()
        data = response.json()
        count = data.get("count", 0)
        print(f"  ✅ Connected to {node}")
        print(f"  ✅ Total documents: {count:,}")
        
        # Test index
        index_url = f"{node}/{config['opensearch_index']}/_count"
        response = requests.get(index_url, auth=auth, verify=False, timeout=10)
        response.raise_for_status()
        data = response.json()
        index_count = data.get("count", 0)
        print(f"  ✅ Index '{config['opensearch_index']}' has {index_count:,} documents")
        return True
    except Exception as exc:
        print(f"  ❌ FAILED: {exc}")
        return False


def check_embedding_service(config: Dict[str, str]) -> bool:
    """Test Flask embedding service connectivity"""
    print("\n=== Flask Embedding Service ===")
    url = config["flask_embedding_url"].rstrip("/")
    
    # If no port specified, try port 5000
    if "://" in url and ":" not in url.split("://")[1]:
        url = f"{url}:5000"
        print(f"  ℹ️  No port specified, trying {url}")
    
    try:
        # Check health endpoint
        health_url = f"{url}/health"
        try:
            response = requests.get(health_url, timeout=5)
            if response.status_code == 200:
                print(f"  ✅ Health check passed: {health_url}")
        except:
            print(f"  ⚠️  No /health endpoint (trying /get_embedding instead)")
        
        # Test actual embedding
        embed_url = f"{url}/get_embedding"
        response = requests.post(
            embed_url,
            json={"text": "test query"},
            timeout=15
        )
        response.raise_for_status()
        data = response.json()
        
        if "embedding" in data and isinstance(data["embedding"], list):
            dim = len(data["embedding"])
            print(f"  ✅ Embedding service working: {embed_url}")
            print(f"  ✅ Embedding dimension: {dim}")
            return True
        else:
            print(f"  ❌ FAILED: Invalid response format")
            return False
    except Exception as exc:
        print(f"  ❌ FAILED: {exc}")
        return False


def check_anvilgpt(config: Dict[str, str]) -> bool:
    """Test AnvilGPT API connectivity"""
    print("\n=== AnvilGPT API ===")
    url = config["anvilgpt_url"].rstrip("/")
    api_key = config["anvilgpt_key"]
    
    # Detect API format from URL
    is_openai_format = "completions" in url
    
    try:
        # Prepare headers
        headers = {"Content-Type": "application/json"}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        
        # OpenAI-compatible format
        if is_openai_format:
            model = config.get("anvilgpt_model", "gpt-4")
            test_payload = {
                "model": model,
                "messages": [
                    {"role": "user", "content": "Say 'test successful' and nothing else."}
                ],
                "max_tokens": 50,
                "temperature": 0.7
            }
        else:
            # Ollama format
            test_payload = {
                "model": "llama2",
                "messages": [
                    {"role": "user", "content": "Say 'test successful' and nothing else."}
                ],
                "stream": False
            }
        
        response = requests.post(
            url,
            headers=headers,
            json=test_payload,
            timeout=30
        )
        response.raise_for_status()
        data = response.json()
        
        # Extract response based on format
        result = ""
        if "choices" in data and len(data["choices"]) > 0:
            # OpenAI format
            result = data["choices"][0].get("message", {}).get("content", "")
        elif "message" in data and "content" in data["message"]:
            # Ollama format
            result = data["message"]["content"]
        else:
            result = data.get("response") or data.get("text") or str(data)
        
        print(f"  ✅ AnvilGPT responding: {url}")
        print(f"  ✅ Test response: {result[:100]}")
        return True
    except Exception as exc:
        print(f"  ❌ FAILED: {exc}")
        print(f"  ℹ️  Note: This is expected if AnvilGPT is not accessible")
        return False


# === RAG Pipeline Setup ===

def setup_llm_backend(config: Dict[str, str]):
    """Configure the LLM backend for generation"""
    from rag_pipeline import llm_utils
    
    def anvilgpt_callable(prompt: str) -> str:
        """Call AnvilGPT API (supports both OpenAI and Ollama formats)"""
        url = config["anvilgpt_url"].rstrip("/")
        api_key = config["anvilgpt_key"]
        
        # Detect API format from URL
        is_openai_format = "completions" in url
        
        try:
            # Prepare headers
            headers = {"Content-Type": "application/json"}
            if api_key:
                headers["Authorization"] = f"Bearer {api_key}"
            
            # Build payload based on format
            if is_openai_format:
                # OpenAI-compatible format
                model = config.get("anvilgpt_model", "gpt-4")
                payload = {
                    "model": model,
                    "messages": [
                        {"role": "user", "content": prompt}
                    ],
                    "max_tokens": 1000,
                    "temperature": 0.7
                }
            else:
                # Ollama format
                payload = {
                    "model": "llama2",
                    "messages": [
                        {"role": "user", "content": prompt}
                    ],
                    "stream": False
                }
            
            response = requests.post(
                url,
                headers=headers,
                json=payload,
                timeout=60
            )
            
            # More detailed error handling
            if response.status_code == 403:
                return "AnvilGPT authentication failed (403 Forbidden). Please verify ANVILGPT_KEY."
            elif response.status_code == 404:
                return "AnvilGPT endpoint not found (404). Please verify ANVILGPT_URL."
            elif response.status_code == 400:
                error_detail = response.text[:200]
                return f"AnvilGPT bad request (400). Error: {error_detail}"
            
            response.raise_for_status()
            data = response.json()
            
            # Extract response based on format
            if "choices" in data and len(data["choices"]) > 0:
                # OpenAI format
                return str(data["choices"][0].get("message", {}).get("content", "")).strip()
            elif "message" in data and "content" in data["message"]:
                # Ollama format
                return str(data["message"]["content"]).strip()
            
            # Try other response keys as fallback
            result = (
                data.get("response") or 
                data.get("text") or 
                data.get("answer") or 
                data.get("completion") or
                str(data)
            )
            return str(result).strip()
        except Exception as exc:
            return f"Error calling AnvilGPT: {exc}"
    
    llm_utils.register_llm_callable(anvilgpt_callable)
    api_format = "OpenAI-compatible" if "completions" in config["anvilgpt_url"] else "Ollama"
    print(f"\n✅ LLM backend registered (AnvilGPT with {api_format} API)")


def create_test_state(query: str) -> Dict[str, Any]:
    """Create a valid AgentState for testing"""
    from rag_pipeline.state import ensure_state_shapes
    
    state = {
        "query_information": {
            "raw_text": query,
            "query": query
        },
        "session_context": {},
        "params": {
            "top_k": 5,
            "max_context_tokens": 3000
        },
        "evidence": {
            "retrieved_documents": [],
            "sources": {}
        },
        "answer": {
            "final_composed_answer": None,
            "citations": [],
            "confidence_score": None
        },
        "planner_reasoning": {},
        "safety_checks": {},
        "trace_observability": {}
    }
    
    return ensure_state_shapes(state)


# === Output Formatting ===

def print_section(title: str):
    """Print a formatted section header"""
    print(f"\n{'='*70}")
    print(f"  {title}")
    print('='*70)


def print_retrieval_summary(state: Dict[str, Any]):
    """Print retrieval results summary"""
    from rag_pipeline.state import summarize_evidence
    
    print_section("RETRIEVAL SUMMARY")
    
    evidence = state.get("evidence", {})
    summary = summarize_evidence(evidence)
    
    print(f"\nTotal Documents Retrieved: {summary['count']}")
    print(f"\nBreakdown by Source:")
    for source, count in summary.get("sources", {}).items():
        print(f"  • {source}: {count} documents")
    
    if summary.get("top_score"):
        print(f"\nTop Score: {summary['top_score']:.4f}")
    
    # Routing decisions
    trace = state.get("trace_observability", {})
    decisions = trace.get("retrieval_routing_decisions", [])
    if decisions:
        print(f"\nRetrieval Routing Decisions:")
        for decision in decisions:
            print(f"  • {decision['source']}: {decision['reason']}")


def print_evidence_details(state: Dict[str, Any]):
    """Print detailed evidence documents"""
    print_section("EVIDENCE DETAILS")
    
    evidence = state.get("evidence", {})
    docs = evidence.get("retrieved_documents", [])
    
    if not docs:
        print("\n⚠️  No documents retrieved")
        return
    
    for i, entry in enumerate(docs[:10], 1):  # Show top 10
        document = entry.get("document", {})
        source = entry.get("source", "unknown")
        score = entry.get("score", 0.0)
        rank = entry.get("retrieval_rank", i-1)
        
        doc_id = document.get("doc_id", "N/A")
        title = document.get("title", "No Title")
        contents = document.get("contents", "")[:200]
        element_type = document.get("element_type", "unknown")
        
        print(f"\n[{i}] Rank {rank} | Score: {score:.4f} | Source: {source}")
        print(f"    Doc ID: {doc_id}")
        print(f"    Type: {element_type}")
        print(f"    Title: {title}")
        print(f"    Content: {contents}...")


def print_generation_output(state: Dict[str, Any]):
    """Print generation results"""
    print_section("GENERATION OUTPUT")
    
    answer = state.get("answer", {})
    final_answer = answer.get("final_composed_answer")
    citations = answer.get("citations", [])
    confidence = answer.get("confidence_score")
    
    print(f"\n📝 Final Answer:\n")
    print(f"{final_answer}\n")
    
    print(f"\n📚 Citations ({len(citations)}):")
    for i, citation in enumerate(citations, 1):
        doc_id = citation.get("doc_id", "N/A")
        source = citation.get("source", "unknown")
        print(f"  [{i}] {doc_id} (from {source})")
    
    print(f"\n📊 Confidence Score: {confidence}")


def print_full_state_debug(state: Dict[str, Any]):
    """Print complete state for debugging"""
    print_section("COMPLETE STATE (DEBUG)")
    print(json.dumps(state, indent=2, default=str))


# === Main Test Execution ===

def run_rag_test(query: str, debug: bool = False):
    """Execute the complete RAG pipeline test"""
    print_section(f"RAG PIPELINE TEST: '{query}'")
    
    try:
        # Import after environment is loaded
        from rag_pipeline.routing import rag_pipeline
        
        # Create test state
        print("\n⚙️  Creating test state...")
        state = create_test_state(query)
        
        # Run pipeline
        print("🚀 Running RAG pipeline...")
        result_state = rag_pipeline(state)
        
        # Print results
        print_retrieval_summary(result_state)
        print_evidence_details(result_state)
        print_generation_output(result_state)
        
        if debug:
            print_full_state_debug(result_state)
        
        # Validate results
        evidence = result_state.get("evidence", {})
        docs = evidence.get("retrieved_documents", [])
        answer = result_state.get("answer", {})
        final_answer = answer.get("final_composed_answer")
        
        print_section("TEST RESULTS")
        
        if docs:
            print(f"✅ Retrieved {len(docs)} documents")
        else:
            print("⚠️  No documents retrieved (query may be too specific)")
        
        if final_answer:
            print(f"✅ Generated answer ({len(final_answer)} characters)")
        else:
            print("❌ No answer generated")
        
        citations = answer.get("citations", [])
        if citations:
            print(f"✅ Citations present ({len(citations)} cited)")
        else:
            print("⚠️  No citations")
        
        print("\n🎉 RAG PIPELINE TEST COMPLETED")
        return True
        
    except Exception as exc:
        print(f"\n❌ PIPELINE ERROR: {exc}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Main test entry point"""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Test RAG pipeline end-to-end")
    parser.add_argument("query", nargs="*", help="Custom test query")
    parser.add_argument("--skip-checks", action="store_true", help="Skip connectivity checks")
    parser.add_argument("--debug", action="store_true", help="Print full state debug output")
    args = parser.parse_args()
    
    print("=" * 70)
    print("   LOCAL RAG PIPELINE END-TO-END TEST")
    print("=" * 70)
    
    # Load environment
    load_environment()
    config = get_config()
    
    # Connectivity checks
    if not args.skip_checks:
        print("\n" + "=" * 70)
        print("   CONNECTIVITY CHECKS")
        print("=" * 70)
        
        os_ok = check_opensearch(config)
        embed_ok = check_embedding_service(config)
        llm_ok = check_anvilgpt(config)
    else:
        print("\n⚠️  Skipping connectivity checks (--skip-checks flag)")
        os_ok = embed_ok = llm_ok = True
    
    if not all([os_ok, embed_ok, llm_ok]):
        print("\n⚠️  Some connectivity checks failed:")
        if not os_ok:
            print("   • OpenSearch - Required for retrieval")
        if not embed_ok:
            print("   • Flask Embedding Service - Required for semantic search")
        if not llm_ok:
            print("   • AnvilGPT - Required for generation")
        
        # Check if running in non-interactive mode
        if not sys.stdin.isatty():
            print("\n⚠️  Running in non-interactive mode. Proceeding with available services...")
        else:
            print("\n   (You can proceed anyway, but some features may fail)")
            response = input("\nContinue anyway? (y/N): ")
            if response.lower() != 'y':
                sys.exit(1)
    
    # Setup LLM backend - skip registration to use production AnvilGPT path
    # setup_llm_backend(config)  # Commented out to test real AnvilGPT integration
    print("\n✅ Using production AnvilGPT integration (no test override)")
    
    # Run test queries
    if args.query:
        # Use custom query from command line
        test_queries = [" ".join(args.query)]
    else:
        # Use default test queries
        test_queries = [
            "climate change impacts on water resources",
        ]
    
    # Run each test query
    for query in test_queries:
        success = run_rag_test(query, debug=args.debug)
        if not success:
            print(f"\n⚠️  Test failed for query: {query}")
    
    print("\n" + "=" * 70)
    print("   ALL TESTS COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()

