from __future__ import annotations
from typing import List, MutableMapping, Tuple, Any

import os
os.environ["COLBERT_SKIP_CPP"] = "1"

import torch
from colbert.infra import ColBERTConfig
from colbert.modeling.checkpoint import Checkpoint
from colbert.modeling.colbert import colbert_score

from .rag_pipeline.state import AgentState, ensure_state_shapes, get_query_text




def _entry_to_text(entry: dict) -> Tuple[str, str]:
    doc = entry.get("document") or {}
    title = doc.get("title") or ""
    body_candidates = [doc.get("contents"), doc.get("summary"), doc.get("snippet")]
    body = next((c for c in body_candidates if isinstance(c, str) and c.strip()), "")
    text = "\n\n".join(part for part in (title.strip(), body.strip()) if part)
    return doc.get("doc_id", ""), text


# Simple cache so we don't re-load the checkpoint on every call
_COLBERT_CHECKPOINTS: dict[str, Checkpoint] = {}


def _get_colbert_checkpoint(model_name: str) -> Checkpoint:
    if model_name not in _COLBERT_CHECKPOINTS:
        config = ColBERTConfig()
        _COLBERT_CHECKPOINTS[model_name] = Checkpoint(
            model_name,
            colbert_config=config,
        )
    return _COLBERT_CHECKPOINTS[model_name]


def rerank_evidence_with_colbert(
    state: MutableMapping[str, Any],
    *,
    top_k: int | None = None,
    model_name: str = "colbert-ir/colbertv2.0",
) -> AgentState:
    state = ensure_state_shapes(state)
    docs: List[dict] = state["evidence"]["retrieved_documents"]
    if len(docs) <= 1:
        return state

    query = get_query_text(state)
    formatted = [(_entry_to_text(entry), entry) for entry in docs]
    filtered = [
        (doc_id, text, entry)
        for (doc_id, text), entry in formatted
        if isinstance(text, str) and text.strip()
    ]
    if len(filtered) <= 1:
        return state

    checkpoint = _get_colbert_checkpoint(model_name)

    texts = [text for _, text, _ in filtered]

    Q = checkpoint.queryFromText([query])              
    D = checkpoint.docFromText(texts, bsize=32)[0]    

    D_mask = torch.ones(D.shape[:2], dtype=torch.long, device=D.device)

    scores_tensor = colbert_score(Q, D, D_mask).flatten()
    scores = scores_tensor.detach().cpu().tolist()

    ranked = sorted(
        zip(filtered, scores),
        key=lambda item: item[1],
        reverse=True,
    )
    new_order = [entry for ((_, _, entry), _score) in ranked]

    if top_k is not None:
        new_order = new_order[:top_k]

    for idx, entry in enumerate(new_order):
        entry["retrieval_rank"] = idx

    state["evidence"]["retrieved_documents"] = new_order
    return state
