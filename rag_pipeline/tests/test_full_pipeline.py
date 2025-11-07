#!/usr/bin/env python3
"""
Complete RAG Pipeline Test - Shows the entire flow:
1. Memory Module (initialize_state with optional chat history context)
2. Routing (rag_pipeline)
3. Search (keyword, semantic, neo4j, spatial retrieval)
4. Generation (LLM response with citations)

This uses pipeline.py::run_pipeline() which orchestrates all modules.
"""

import os
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from dotenv import load_dotenv

# Load environment
env_path = Path(__file__).parent.parent / ".env.local"
load_dotenv(env_path)

from rag_pipeline.pipeline import run_pipeline


def print_section(title: str, char: str = "="):
    """Print a formatted section"""
    print(f"\n{char * 80}")
    print(f"  {title}")
    print(f"{char * 80}\n")


def test_full_pipeline(query: str, memory_id: str = None, session_context: dict = None):
    """
    Test the complete RAG pipeline from memory to generation.
    """
    
    if session_context is None:
        session_context = {}
    
    print_section("COMPLETE RAG PIPELINE TEST", "=")
    print(f"Query: \"{query}\"")
    if memory_id:
        print(f"Memory ID: {memory_id} (will augment query with chat history)")
    else:
        print("Memory ID: None (no chat history context)")
    if session_context:
        print(f"Session Context: {session_context}")
    
    # ========================================
    # STEP 1: Memory Module
    # ========================================
    print_section("STEP 1: MEMORY MODULE (initialize_state)", "-")
    print("Initializing state with memory module...")
    print("• Checking for chat history")
    print("• Augmenting query if follow-up detected")
    print("• Creating AgentState structure")
    
    # Run the complete pipeline (includes memory initialization)
    result_state = run_pipeline(
        user_input=query,
        memory_id=memory_id,
        session_context=session_context,
        params={"top_k": 5, "max_context_tokens": 3000}
    )
    
    # Show memory augmentation results
    query_info = result_state.get("query_information", {})
    memory_info = query_info.get("memory", {})
    
    if memory_info:
        print("\n✅ Memory Module Results:")
        print(f"   Original query: {memory_info.get('original_query', query)}")
        augmented = memory_info.get('augmented_query')
        if augmented and augmented != query:
            print(f"   Augmented query: {augmented}")
            print("   ℹ️  Query was enhanced with chat history context")
        else:
            print("   ℹ️  No augmentation needed (standalone query)")
    else:
        print("\n✅ Memory Module: No chat history (standalone query)")
    
    final_query = query_info.get("raw_text", query)
    print(f"\nFinal query for retrieval: \"{final_query}\"")
    
    # ========================================
    # STEP 2: Routing & Retrieval
    # ========================================
    print_section("STEP 2: ROUTING & SEARCH (run_retrieval)", "-")
    print("Routing module determines which search methods to use...")
    
    trace = result_state.get("trace_observability", {})
    routing_decisions = trace.get("retrieval_routing_decisions", [])
    
    print("\nRetrieval Routing Decisions:")
    for decision in routing_decisions:
        print(f"  • {decision['source']}: {decision['reason']}")
    
    evidence = result_state.get("evidence", {})
    docs = evidence.get("retrieved_documents", [])
    
    print(f"\n✅ Total documents retrieved: {len(docs)}")
    
    # Show breakdown by source
    sources = {}
    for doc in docs:
        source = doc.get("source", "unknown")
        sources[source] = sources.get(source, 0) + 1
    
    print("\nDocuments by source:")
    for source, count in sources.items():
        print(f"  • {source}: {count} documents")
    
    # Show top documents
    print("\nTop Retrieved Documents:")
    for i, entry in enumerate(docs[:3], 1):
        doc = entry.get("document", {})
        score = entry.get("score", 0.0)
        source = entry.get("source", "unknown")
        
        print(f"\n[{i}] {doc.get('title', 'No Title')}")
        print(f"    Doc ID: {doc.get('doc_id', 'N/A')}")
        print(f"    Score: {score:.4f} | Source: {source}")
        print(f"    Type: {doc.get('element_type', 'unknown')}")
        print(f"    Content: {doc.get('contents', '')[:150]}...")
    
    # ========================================
    # STEP 3: Context Construction
    # ========================================
    print_section("STEP 3: CONTEXT CONSTRUCTION (for LLM)", "-")
    
    from rag_pipeline.generation import _build_context_and_citations, _context_budget
    
    max_chars = _context_budget(result_state)
    context_info = _build_context_and_citations(docs, max_chars)
    context_block = context_info["context"]
    citations = context_info["citations"]
    
    print(f"Max context budget: {max_chars} characters")
    print(f"Actual context size: {len(context_block)} characters")
    print(f"Documents included: {len(citations)}")
    
    print("\n--- Context Block (Evidence for LLM) ---")
    print(context_block[:500] + "..." if len(context_block) > 500 else context_block)
    print("--- End Context ---")
    
    # ========================================
    # STEP 4: Prompt Construction
    # ========================================
    print_section("STEP 4: PROMPT CONSTRUCTION", "-")
    
    system = (
        "You are a factual assistant. Use ONLY the provided evidence. "
        "Cite by [doc_id] after each relevant claim. If something is not in the evidence, say so briefly."
    )
    prompt = (
        f"{system}\n\n"
        f"User question:\n{final_query}\n\n"
        f"Evidence:\n{context_block}\n\n"
        "Answer with inline [doc_id] citations."
    )
    
    print(f"Prompt length: {len(prompt)} characters\n")
    print("--- Full Prompt to LLM ---")
    print(prompt)
    print("--- End Prompt ---")
    
    # ========================================
    # STEP 5: LLM Generation
    # ========================================
    print_section("STEP 5: GENERATION (call_llm via AnvilGPT)", "-")
    
    print(f"Calling AnvilGPT API...")
    print(f"  URL: {os.getenv('ANVILGPT_URL')}")
    print(f"  Model: {os.getenv('ANVILGPT_MODEL', 'gpt-oss:120b')}")
    
    answer = result_state.get("answer", {})
    final_answer = answer.get("final_composed_answer")
    
    print("\n✅ Response received from AnvilGPT!")
    
    # ========================================
    # STEP 6: Final Answer
    # ========================================
    print_section("STEP 6: FINAL ANSWER WITH CITATIONS", "-")
    
    print("Generated Answer:\n")
    print(final_answer)
    
    print(f"\n\nAnswer Statistics:")
    print(f"  • Length: {len(final_answer) if final_answer else 0} characters")
    print(f"  • Word count: {len(final_answer.split()) if final_answer else 0} words")
    
    answer_citations = answer.get("citations", [])
    confidence = answer.get("confidence_score", 0.0)
    
    print(f"  • Citations: {len(answer_citations)}")
    print(f"  • Confidence: {confidence}")
    
    print("\n--- Citation Details ---")
    for i, citation in enumerate(answer_citations, 1):
        doc_id = citation.get("doc_id", "N/A")
        source = citation.get("source", "unknown")
        
        # Find matching document
        matching_doc = next(
            (d for d in docs if d.get("document", {}).get("doc_id") == doc_id),
            None
        )
        
        if matching_doc:
            doc = matching_doc.get("document", {})
            title = doc.get("title", "Unknown")
            doc_type = doc.get("element_type", "unknown")
            print(f"\n[{i}] {doc_id}")
            print(f"    Title: {title}")
            print(f"    Type: {doc_type}")
            print(f"    Source: {source}")
        else:
            print(f"\n[{i}] {doc_id} (source: {source})")
    
    # ========================================
    # Summary
    # ========================================
    print_section("PIPELINE EXECUTION SUMMARY", "=")
    
    print("✅ STEP 1: Memory Module")
    print(f"   → Query: \"{final_query}\"")
    if memory_info.get('augmented_query') != query:
        print(f"   → Augmented from: \"{query}\"")
    
    print("\n✅ STEP 2: Routing & Search")
    print(f"   → Methods: {', '.join(sources.keys())}")
    print(f"   → Documents: {len(docs)} retrieved")
    
    print("\n✅ STEP 3: Context Construction")
    print(f"   → Context: {len(context_block)} chars")
    print(f"   → Documents used: {len(citations)}")
    
    print("\n✅ STEP 4: Prompt Construction")
    print(f"   → Prompt: {len(prompt)} chars")
    
    print("\n✅ STEP 5: LLM Generation")
    print(f"   → Model: {os.getenv('ANVILGPT_MODEL', 'gpt-oss:120b')}")
    print(f"   → Response: {len(final_answer) if final_answer else 0} chars")
    
    print("\n✅ STEP 6: Final Answer")
    print(f"   → Citations: {len(answer_citations)}")
    print(f"   → Confidence: {confidence}")
    
    print("\n" + "="*80)
    print("  COMPLETE RAG PIPELINE TEST SUCCESSFUL!")
    print("="*80 + "\n")


def main():
    """Main entry point"""
    
    # Use command line argument or default query
    if len(sys.argv) > 1:
        query = " ".join(sys.argv[1:])
    else:
        query = "What are the impacts of climate change on water resources?"
    
    # Optional: provide a memory_id to test chat history augmentation
    # memory_id = "test-conversation-123"  # Uncomment to test with memory
    memory_id = None
    
    # Optional: enable spatial search to test geospatial retrieval
    # With explicit spatial_coords for Great Salt Lake area (latitude, longitude, radius in meters)
    session_context = {
        "use_spatial": True,
        # Don't provide spatial_coords - this will trigger NLP extraction + Google Maps geocoding
    }
    
    test_full_pipeline(query, memory_id, session_context)


if __name__ == "__main__":
    main()


