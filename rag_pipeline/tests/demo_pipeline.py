#!/usr/bin/env python3
"""
Demonstration of the full RAG pipeline with LLM router.
Shows each step: Query -> Memory -> LLM Router -> Search -> Generation

Loads credentials from rag_pipeline/.env.local
"""
import sys
import os
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

# Load environment variables from .env.local
from dotenv import load_dotenv
env_path = Path(__file__).parent.parent / ".env.local"
if env_path.exists():
    load_dotenv(env_path)
    print(f"✓ Loaded credentials from {env_path}")
else:
    print(f"⚠️  Warning: {env_path} not found")

def main():
    print("=" * 100)
    print(" FULL RAG PIPELINE WITH LLM ROUTER - STEP BY STEP ")
    print("=" * 100)
    
    # ============================================================================
    # STEP 1: INPUT QUERY
    # ============================================================================
    query = sys.argv[1] if len(sys.argv) > 1 else "What are the main sources of greenhouse gas emissions?"
    
    print(f"\n📝 INPUT QUERY:")
    print(f"   {query}")
    
    # ============================================================================
    # STEP 2: CHECK ENVIRONMENT
    # ============================================================================
    print(f"\n{'─' * 100}")
    print(" STEP 1: Environment Check")
    print(f"{'─' * 100}")
    
    required_env = {
        "OPENSEARCH_NODE": os.getenv("OPENSEARCH_NODE"),
        "OPENSEARCH_INDEX": os.getenv("OPENSEARCH_INDEX"),
        "ANVILGPT_URL": os.getenv("ANVILGPT_URL"),
        "ANVILGPT_KEY": os.getenv("ANVILGPT_KEY"),
    }
    
    optional_env = {
        "FLASK_EMBEDDING_URL": os.getenv("FLASK_EMBEDDING_URL"),
        "SPATIAL_BACKEND_ENABLED": os.getenv("SPATIAL_BACKEND_ENABLED"),
        "NEO4J_CONNECTION_STRING": os.getenv("NEO4J_CONNECTION_STRING"),
    }
    
    print("\nRequired:")
    missing_required = []
    for key, value in required_env.items():
        if value:
            display = value[:50] + "..." if len(value) > 50 else value
            print(f"  ✓ {key}: {display}")
        else:
            print(f"  ✗ {key}: NOT SET")
            missing_required.append(key)
    
    print("\nOptional:")
    for key, value in optional_env.items():
        if value:
            print(f"  ✓ {key}: {value}")
        else:
            print(f"  ○ {key}: not set")
    
    if missing_required:
        print(f"\n⚠️  ERROR: Missing required environment variables: {', '.join(missing_required)}")
        print("\nTo run this test with real data, set:")
        print("  export OPENSEARCH_NODE=your_opensearch_url")
        print("  export OPENSEARCH_INDEX=your_index_name")
        print("  export ANVILGPT_URL=https://anvilgpt.rcac.purdue.edu/api/chat/completions")
        print("  export ANVILGPT_KEY=your_api_key")
        return
    
    # ============================================================================
    # STEP 3: INITIALIZE STATE (Memory Module)
    # ============================================================================
    print(f"\n{'─' * 100}")
    print(" STEP 2: Initialize State (Memory Module)")
    print(f"{'─' * 100}")
    
    try:
        from rag_pipeline.state import ensure_state_shapes, get_query_text
        
        state = ensure_state_shapes({
            "query_information": {"query": query, "raw_text": query},
            "params": {"top_k": 10, "max_context_tokens": 3000},
            "session_context": {},
            "evidence": {"retrieved_documents": []},
            "answer": {},
            "trace_observability": {},
        })
        
        final_query = get_query_text(state)
        print(f"\n✓ State initialized")
        print(f"  Original query: {query}")
        print(f"  Final query for retrieval: {final_query}")
        print(f"  Top-k: {state['params']['top_k']}")
        
    except Exception as e:
        print(f"\n✗ State initialization failed: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # ============================================================================
    # STEP 4: LLM ROUTER DECISION
    # ============================================================================
    print(f"\n{'─' * 100}")
    print(" STEP 3: LLM Router Decision (AnvilGPT)")
    print(f"{'─' * 100}")
    
    try:
        from rag_pipeline.router_llm import build_default_registry, LLMRouter
        from rag_pipeline.search_keyword import retrieve_keyword
        from rag_pipeline.semantic_search import retrieve_semantic
        from rag_pipeline.spatial_search import retrieve_spatial
        from rag_pipeline.search_neo4j import retrieve_neo4j
        
        # Build adapters for real backends
        class RealKeywordAdapter:
            name = "keyword"
            def is_available(self): return bool(os.getenv("OPENSEARCH_NODE"))
            def supports(self, query): return True
            def execute(self, query, ctx):
                try:
                    hits = retrieve_keyword(ctx.get("state", {}))
                    return {"module": self.name, "items": hits, "meta": {"count": len(hits)}}
                except Exception as e:
                    return {"module": self.name, "items": [], "meta": {"error": str(e)}}
        
        class RealSemanticAdapter:
            name = "semantic"
            def is_available(self): 
                return bool(os.getenv("FLASK_EMBEDDING_URL") and os.getenv("OPENSEARCH_NODE"))
            def supports(self, query): return True
            def execute(self, query, ctx):
                try:
                    hits = retrieve_semantic(ctx.get("state", {}))
                    return {"module": self.name, "items": hits, "meta": {"count": len(hits)}}
                except Exception as e:
                    return {"module": self.name, "items": [], "meta": {"error": str(e)}}
        
        class RealSpatialAdapter:
            name = "spatial"
            def is_available(self): 
                return os.getenv("SPATIAL_BACKEND_ENABLED") == "1"
            def supports(self, query): return True
            def execute(self, query, ctx):
                try:
                    hits = retrieve_spatial(ctx.get("state", {}))
                    return {"module": self.name, "items": hits, "meta": {"count": len(hits)}}
                except Exception as e:
                    return {"module": self.name, "items": [], "meta": {"error": str(e)}}
        
        class RealGraphAdapter:
            name = "graph"
            def is_available(self): 
                return bool(os.getenv("NEO4J_URI") or os.getenv("NEO4J_CONNECTION_STRING"))
            def supports(self, query): return True
            def execute(self, query, ctx):
                try:
                    hits = retrieve_neo4j(ctx.get("state", {}))
                    return {"module": self.name, "items": hits, "meta": {"count": len(hits)}}
                except Exception as e:
                    return {"module": self.name, "items": [], "meta": {"error": str(e)}}
        
        from rag_pipeline.router_llm import ModuleRegistry
        registry = ModuleRegistry()
        registry.register(RealKeywordAdapter())
        registry.register(RealSemanticAdapter())
        registry.register(RealSpatialAdapter())
        registry.register(RealGraphAdapter())
        
        router = LLMRouter(registry)
        
        available = [m.name for m in registry.all() if m.is_available()]
        print(f"\nAvailable search modules: {', '.join(available)}")
        
        print(f"\nCalling AnvilGPT for routing decision...")
        plan = router.plan(query)
        
        print(f"\n🤖 LLM ROUTING DECISION:")
        print(f"   Selected modules: {', '.join(plan.chosen_modules)}")
        print(f"\n📝 Rationale:")
        for module, reason in plan.rationale.items():
            symbol = "✓" if module in plan.chosen_modules else "✗"
            print(f"   {symbol} {module}: {reason}")
        
    except Exception as e:
        print(f"\n✗ Router decision failed: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # ============================================================================
    # STEP 5: EXECUTE SEARCHES
    # ============================================================================
    print(f"\n{'─' * 100}")
    print(" STEP 4: Execute Selected Searches")
    print(f"{'─' * 100}")
    
    try:
        from rag_pipeline.state import merge_retrieval
        
        print(f"\nExecuting {len(plan.chosen_modules)} search modules...")
        router_output = router.execute(plan, ctx={"state": state})
        
        for result in router_output["results"]:
            module_name = result["module"]
            items = result["items"]
            meta = result.get("meta", {})
            
            if "error" in meta:
                print(f"  ✗ {module_name}: ERROR - {meta['error']}")
            else:
                print(f"  ✓ {module_name}: {len(items)} hits retrieved")
                # Merge into state
                appended = merge_retrieval(state, source=module_name, hits=items, limit=state["params"]["top_k"])
                print(f"    → {len(appended)} new documents added to evidence")
        
        total_docs = len(state["evidence"]["retrieved_documents"])
        print(f"\n📊 Total unique documents: {total_docs}")
        
        if total_docs > 0:
            print(f"\nAll retrieved documents:")
            for i, doc in enumerate(state["evidence"]["retrieved_documents"], 1):
                doc_data = doc.get("document", {})
                doc_id = doc_data.get("doc_id", "unknown")
                title = doc_data.get("title", "No title")[:60]
                source = doc.get("source", "unknown")
                print(f"  [{i}] {doc_id} (from {source})")
                print(f"      {title}...")
        
    except Exception as e:
        print(f"\n✗ Search execution failed: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # ============================================================================
    # STEP 6: GENERATE ANSWER
    # ============================================================================
    print(f"\n{'─' * 100}")
    print(" STEP 5: Generate Answer (AnvilGPT)")
    print(f"{'─' * 100}")
    
    if total_docs == 0:
        print(f"\n⚠️  No documents retrieved. Skipping generation.")
        state["answer"] = {
            "final_composed_answer": f'No evidence found for "{query}".',
            "citations": [],
            "confidence_score": 0.0,
        }
    else:
        try:
            from rag_pipeline.generation import run_generation
            
            print(f"\nGenerating answer from {total_docs} documents...")
            state = run_generation(state)
            print(f"✓ Answer generated successfully")
            
        except Exception as e:
            print(f"\n✗ Generation failed: {e}")
            import traceback
            traceback.print_exc()
            state["answer"] = {
                "final_composed_answer": f"Error generating answer: {e}",
                "citations": [],
                "confidence_score": 0.0,
            }
    
    # ============================================================================
    # STEP 7: DISPLAY FINAL ANSWER
    # ============================================================================
    print(f"\n{'=' * 100}")
    print(" FINAL OUTPUT ")
    print(f"{'=' * 100}")
    
    answer = state["answer"]
    final_answer = answer.get("final_composed_answer", "No answer generated")
    citations = answer.get("citations", [])
    confidence = answer.get("confidence_score", 0.0)
    
    print(f"\n📖 FINAL ANSWER:")
    print(f"   {final_answer}")
    
    if citations:
        print(f"\n📚 CITATIONS ({len(citations)} documents):")
        for i, cit in enumerate(citations, 1):
            doc_id = cit.get("doc_id", "unknown")
            source = cit.get("source", "unknown")
            print(f"   [{i}] {doc_id} (from {source})")
    
    print(f"\n📊 CONFIDENCE: {confidence:.2f}")
    
    print(f"\n{'=' * 100}\n")

if __name__ == "__main__":
    main()

