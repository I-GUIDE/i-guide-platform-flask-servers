#!/usr/bin/env python3
from pprint import pprint

from rag_pipeline.memory_module import initialize_state
from rag_pipeline.search_core import run_retrieval
from reranker import rerank_evidence_with_colbert

def summarize(label, docs):
    print(f"\n{label}")
    for idx, entry in enumerate(docs):
        doc = entry["document"]
        print(f"{idx:02d} doc_id={doc['doc_id']} score={entry['score']:.3f} title={doc.get('title','')!r}")

def main():
    state = initialize_state(
        user_input="What mitigation plans exist for aging dams near Cincinnati?",
        params={"top_k": 6},
    )

    run_retrieval(state)
    before = list(state["evidence"]["retrieved_documents"])
    summarize("Before ColBERT:", before)

    rerank_evidence_with_colbert(state, top_k=state["params"]["top_k"])
    after = state["evidence"]["retrieved_documents"]
    summarize("After ColBERT:", after)

    # Optional: dump trace details
    from pprint import pprint
    pprint(state["trace_observability"])

if __name__ == "__main__":
    main()
