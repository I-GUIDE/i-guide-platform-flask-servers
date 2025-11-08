#!/usr/bin/env python
"""
End-to-End test of the LLM Router with real search backends and generation.

This test runs the complete pipeline:
1. Query → LLM Router decision
2. Execute selected search modules (real backends)
3. Generate final answer using real AnvilGPT
4. Display complete output

Prerequisites:
    # Required for all searches
    export OPENSEARCH_NODE=your_opensearch_url
    export OPENSEARCH_INDEX=your_index_name
    export OPENSEARCH_USERNAME=your_username  # optional
    export OPENSEARCH_PASSWORD=your_password  # optional
    
    # Required for LLM routing and generation
    export ANVILGPT_URL=https://anvilgpt.rcac.purdue.edu/api/chat/completions
    export ANVILGPT_KEY=your_api_key
    export ANVILGPT_MODEL=gpt-oss:120b  # optional
    
    # Optional: Enable additional search types
    export FLASK_EMBEDDING_URL=http://localhost:5002  # for semantic search
    export GOOGLE_MAPS_API_KEY=your_maps_key  # for spatial search
    export NEO4J_URI=bolt://localhost:7687  # for graph search
    export NEO4J_USER=neo4j
    export NEO4J_PASSWORD=your_password

Usage:
    # Basic test with keyword + semantic
    python test_e2e_llm_router.py

    # Test with spatial search enabled
    export SPATIAL_BACKEND_ENABLED=1
    python test_e2e_llm_router.py

    # Custom query
    python test_e2e_llm_router.py "Find datasets about climate change in Illinois"
"""

import sys
import os
import json
from typing import Dict, Any, List
from datetime import datetime

# Add parent directories to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

# Import router from rag_pipeline
from rag_pipeline.router_llm import (
    build_default_registry,
    LLMRouter,
    ModuleRegistry,
)

# Import real search backends
from rag_pipeline.search_keyword import retrieve_keyword
from rag_pipeline.semantic_search import retrieve_semantic
from rag_pipeline.spatial_search import retrieve_spatial
from rag_pipeline.search_neo4j import retrieve_neo4j

# Import generation
from rag_pipeline.generation import run_generation

# Import state management
from rag_pipeline.state import ensure_state_shapes, merge_retrieval


def print_section(title: str, char: str = "="):
    """Print a formatted section header."""
    width = 100
    print(f"\n{char * width}")
    print(f"{title:^{width}}")
    print(f"{char * width}\n")


def print_subsection(title: str):
    """Print a formatted subsection header."""
    print(f"\n{'─' * 100}")
    print(f"  {title}")
    print(f"{'─' * 100}")


def format_json(data: Any, indent: int = 2) -> str:
    """Format data as pretty JSON."""
    return json.dumps(data, indent=indent, default=str)


# ============================================================================
# ADAPTER IMPLEMENTATIONS - Wire real backends to router
# ============================================================================

class RealKeywordAdapter:
    """Adapter for real OpenSearch keyword search."""
    name = "keyword"
    
    def is_available(self) -> bool:
        return bool(os.getenv("OPENSEARCH_NODE"))
    
    def supports(self, query: str) -> bool:
        return True
    
    def execute(self, query: str, ctx: Dict[str, Any]) -> Dict[str, Any]:
        state = ctx.get("state", {})
        try:
            hits = retrieve_keyword(state)
            return {
                "module": self.name,
                "items": hits,
                "meta": {"engine": "opensearch", "hit_count": len(hits)}
            }
        except Exception as e:
            return {
                "module": self.name,
                "items": [],
                "meta": {"error": str(e)}
            }


class RealSemanticAdapter:
    """Adapter for real vector embedding search."""
    name = "semantic"
    
    def is_available(self) -> bool:
        # Check if embedding service is configured
        flask_url = os.getenv("FLASK_EMBEDDING_URL")
        opensearch_node = os.getenv("OPENSEARCH_NODE")
        return bool(flask_url and opensearch_node)
    
    def supports(self, query: str) -> bool:
        return True
    
    def execute(self, query: str, ctx: Dict[str, Any]) -> Dict[str, Any]:
        state = ctx.get("state", {})
        try:
            hits = retrieve_semantic(state)
            return {
                "module": self.name,
                "items": hits,
                "meta": {"engine": "vector_knn", "hit_count": len(hits)}
            }
        except Exception as e:
            return {
                "module": self.name,
                "items": [],
                "meta": {"error": str(e)}
            }


class RealSpatialAdapter:
    """Adapter for real spatial/geographic search."""
    name = "spatial"
    
    def is_available(self) -> bool:
        # Check if spatial tools are available
        maps_key = os.getenv("GOOGLE_MAPS_API_KEY")
        opensearch = os.getenv("OPENSEARCH_NODE")
        enabled = os.getenv("SPATIAL_BACKEND_ENABLED", "0") == "1"
        return bool(enabled and opensearch and maps_key)
    
    def supports(self, query: str) -> bool:
        return True
    
    def execute(self, query: str, ctx: Dict[str, Any]) -> Dict[str, Any]:
        state = ctx.get("state", {})
        try:
            hits = retrieve_spatial(state)
            return {
                "module": self.name,
                "items": hits,
                "meta": {"engine": "geo_shape", "hit_count": len(hits)}
            }
        except Exception as e:
            return {
                "module": self.name,
                "items": [],
                "meta": {"error": str(e)}
            }


class RealGraphAdapter:
    """Adapter for real Neo4j graph search."""
    name = "graph"
    
    def is_available(self) -> bool:
        return bool(os.getenv("NEO4J_URI"))
    
    def supports(self, query: str) -> bool:
        return True
    
    def execute(self, query: str, ctx: Dict[str, Any]) -> Dict[str, Any]:
        state = ctx.get("state", {})
        try:
            hits = retrieve_neo4j(state)
            return {
                "module": self.name,
                "items": hits,
                "meta": {"engine": "neo4j_cypher", "hit_count": len(hits)}
            }
        except Exception as e:
            return {
                "module": self.name,
                "items": [],
                "meta": {"error": str(e)}
            }


def build_real_registry() -> ModuleRegistry:
    """Build registry with real search backend adapters."""
    reg = ModuleRegistry()
    reg.register(RealKeywordAdapter())
    reg.register(RealSemanticAdapter())
    reg.register(RealSpatialAdapter())
    reg.register(RealGraphAdapter())
    return reg


# ============================================================================
# PIPELINE ORCHESTRATION
# ============================================================================

def merge_router_results_into_state(state: Dict[str, Any], router_output: Dict[str, Any], limit: int = 10):
    """Merge router search results into the state's evidence structure."""
    for result in router_output["results"]:
        module_name = result["module"]
        hits = result["items"]
        meta = result.get("meta", {})
        
        if "error" in meta:
            print(f"  ⚠️  {module_name}: {meta['error']}")
            continue
        
        # Merge hits into state using the existing merge_retrieval function
        appended = merge_retrieval(
            state,
            source=module_name,
            hits=hits,
            limit=limit,
        )
        print(f"  ✓ {module_name}: {len(hits)} hits retrieved, {len(appended)} new docs added")


def run_full_pipeline(query: str, verbose: bool = True) -> Dict[str, Any]:
    """
    Run the complete RAG pipeline with LLM router.
    
    Steps:
    1. Initialize state
    2. LLM router decides which search modules to use
    3. Execute selected searches (real backends)
    4. Merge all results into state
    5. Generate final answer with AnvilGPT
    6. Return complete results
    """
    if verbose:
        print_section(f"RAG Pipeline with LLM Router", "=")
        print(f"Query: {query}")
        print(f"Timestamp: {datetime.now().isoformat()}")
    
    # ========== STEP 1: Initialize State ==========
    if verbose:
        print_subsection("Step 1: Initialize State")
    
    state = ensure_state_shapes({
        "user_query": {"query": query},
        "params": {"top_k": 10, "max_context_tokens": 6000},
        "session_context": {},
        "evidence": {"retrieved_documents": []},
        "answer": {},
        "trace_observability": {},
    })
    
    if verbose:
        print(f"  State initialized with query: '{query}'")
        print(f"  Top-k: {state['params']['top_k']}")
    
    # ========== STEP 2: LLM Router Decision ==========
    if verbose:
        print_subsection("Step 2: LLM Router Decision")
    
    registry = build_real_registry()
    router = LLMRouter(registry)
    
    # Check what modules are available
    available_modules = [m.name for m in registry.all() if m.is_available()]
    if verbose:
        print(f"  Available modules: {', '.join(available_modules)}")
    
    # Get routing plan from LLM
    plan = router.plan(query)
    
    if verbose:
        print(f"\n  🤖 LLM Routing Decision:")
        print(f"     Selected modules: {', '.join(plan.chosen_modules)}")
        print(f"\n  📝 Rationale:")
        for module, reason in plan.rationale.items():
            symbol = "✓" if module in plan.chosen_modules or module == "keyword" else "✗"
            print(f"     {symbol} {module}: {reason}")
    
    # ========== STEP 3: Execute Search Modules ==========
    if verbose:
        print_subsection("Step 3: Execute Search Modules")
    
    router_output = router.execute(plan, ctx={"state": state})
    
    if verbose:
        print(f"  Executing {len(plan.chosen_modules)} search modules...")
        merge_router_results_into_state(state, router_output, limit=state["params"]["top_k"])
    else:
        merge_router_results_into_state(state, router_output, limit=state["params"]["top_k"])
    
    total_docs = len(state["evidence"]["retrieved_documents"])
    if verbose:
        print(f"\n  Total unique documents retrieved: {total_docs}")
    
    # ========== STEP 4: Generate Answer ==========
    if verbose:
        print_subsection("Step 4: Generate Answer with AnvilGPT")
    
    if total_docs == 0:
        if verbose:
            print("  ⚠️  No documents retrieved. Skipping generation.")
        state["answer"] = {
            "final_composed_answer": f'No evidence found for "{query}".',
            "citations": [],
            "confidence_score": 0.0,
        }
    else:
        if verbose:
            print(f"  Generating answer from {total_docs} documents...")
        try:
            state = run_generation(state)
            if verbose:
                print(f"  ✓ Answer generated successfully")
        except Exception as e:
            if verbose:
                print(f"  ✗ Generation failed: {e}")
            state["answer"] = {
                "final_composed_answer": f"Error generating answer: {e}",
                "citations": [],
                "confidence_score": 0.0,
            }
    
    # ========== STEP 5: Compile Results ==========
    results = {
        "query": query,
        "timestamp": datetime.now().isoformat(),
        "routing": {
            "selected_modules": plan.chosen_modules,
            "rationale": plan.rationale,
        },
        "retrieval": {
            "total_documents": total_docs,
            "by_module": {},
        },
        "answer": state["answer"],
        "trace": state.get("trace_observability", {}),
    }
    
    # Count documents by source
    for doc in state["evidence"]["retrieved_documents"]:
        source = doc.get("source", "unknown")
        results["retrieval"]["by_module"][source] = results["retrieval"]["by_module"].get(source, 0) + 1
    
    return results


def display_results(results: Dict[str, Any]):
    """Display the complete pipeline results in a readable format."""
    print_section("FINAL RESULTS", "=")
    
    # Query
    print_subsection("Query")
    print(f"  {results['query']}")
    
    # Routing Decision
    print_subsection("Routing Decision")
    print(f"  Modules used: {', '.join(results['routing']['selected_modules'])}")
    print(f"\n  Rationale:")
    for module, reason in results['routing']['rationale'].items():
        symbol = "✓" if module in results['routing']['selected_modules'] or module == "keyword" else "✗"
        print(f"    {symbol} {module}: {reason}")
    
    # Retrieval Stats
    print_subsection("Retrieval Statistics")
    print(f"  Total documents: {results['retrieval']['total_documents']}")
    if results['retrieval']['by_module']:
        print(f"\n  By module:")
        for module, count in results['retrieval']['by_module'].items():
            print(f"    • {module}: {count} documents")
    
    # Final Answer
    print_subsection("Final Answer")
    answer_text = results['answer'].get('final_composed_answer', 'No answer generated.')
    print(f"  {answer_text}")
    
    # Citations
    citations = results['answer'].get('citations', [])
    if citations:
        print(f"\n  Citations ({len(citations)} documents):")
        for i, citation in enumerate(citations[:5], 1):  # Show first 5
            doc_id = citation.get('doc_id', 'unknown')
            source = citation.get('source', 'unknown')
            print(f"    [{i}] {doc_id} (from {source})")
        if len(citations) > 5:
            print(f"    ... and {len(citations) - 5} more")
    
    # Confidence
    confidence = results['answer'].get('confidence_score', 0.0)
    print(f"\n  Confidence: {confidence:.2f}")
    
    print("\n" + "=" * 100 + "\n")


def check_environment():
    """Check required environment variables and print status."""
    print_section("Environment Check", "=")
    
    required = {
        "OPENSEARCH_NODE": "OpenSearch endpoint",
        "OPENSEARCH_INDEX": "OpenSearch index name",
        "ANVILGPT_URL": "AnvilGPT API endpoint",
        "ANVILGPT_KEY": "AnvilGPT API key",
    }
    
    optional = {
        "FLASK_EMBEDDING_URL": "Embedding service for semantic search",
        "GOOGLE_MAPS_API_KEY": "Google Maps for spatial search",
        "NEO4J_URI": "Neo4j database for graph search",
        "SPATIAL_BACKEND_ENABLED": "Enable spatial search",
    }
    
    print("Required Configuration:")
    all_required_set = True
    for var, description in required.items():
        value = os.getenv(var)
        status = "✓" if value else "✗"
        display_value = value[:50] + "..." if value and len(value) > 50 else value or "NOT SET"
        print(f"  {status} {var}: {display_value}")
        if not value:
            all_required_set = False
    
    print("\nOptional Configuration:")
    for var, description in optional.items():
        value = os.getenv(var)
        status = "✓" if value else "○"
        display_value = value[:50] + "..." if value and len(value) > 50 else value or "not set"
        print(f"  {status} {var}: {display_value} ({description})")
    
    if not all_required_set:
        print("\n⚠️  WARNING: Missing required environment variables.")
        print("   The test will attempt to run but may fail.")
    
    print("\n" + "=" * 100 + "\n")
    return all_required_set


def main():
    """Main test entry point."""
    # Check if custom query provided
    if len(sys.argv) > 1:
        query = " ".join(sys.argv[1:])
    else:
        # Default test query
        query = "What are the main sources of greenhouse gas emissions?"
    
    # Check environment
    env_ok = check_environment()
    
    if not env_ok:
        print("Proceed anyway? (y/n): ", end="")
        response = input().strip().lower()
        if response != 'y':
            print("Exiting. Please set required environment variables.")
            sys.exit(1)
    
    try:
        # Run full pipeline
        results = run_full_pipeline(query, verbose=True)
        
        # Display results
        display_results(results)
        
        # Optionally save to file
        output_file = f"e2e_test_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        print(f"📄 Full results saved to: {output_file}\n")
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()

