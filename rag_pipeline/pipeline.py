from __future__ import annotations

from typing import Any, Mapping, MutableMapping, Optional

from .memory_module import initialize_state
from .routing import rag_pipeline
from .state import AgentState, ensure_state_shapes


def run_pipeline(
    *,
    user_input: Optional[str] = None,
    state: Optional[MutableMapping[str, Any]] = None,
    memory_id: Optional[str] = None,
    session_context: Optional[Mapping[str, Any]] = None,
    params: Optional[Mapping[str, Any]] = None,
    recent_k: Optional[int] = None,
    extra_state: Optional[Mapping[str, Any]] = None,
) -> AgentState:
    """
    Orchestrate the full RAG flow: initialize memory-backed state, run retrieval, then generation.

    Provide either an existing `state` or a `user_input`. When `state` is supplied, the remaining keyword
    arguments are ignored.
    """
    if state is not None and user_input is not None:
        raise ValueError("Provide either an existing state or a user_input, not both.")

    if state is None:
        if user_input is None:
            raise ValueError("user_input is required when state is not provided.")
        shaped_state = initialize_state(
            user_input,
            memory_id=memory_id,
            session_context=session_context,
            params=params,
            recent_k=recent_k,
            extra_state=extra_state,
        )
    else:
        shaped_state = ensure_state_shapes(state)

    return rag_pipeline(shaped_state)


__all__ = ["run_pipeline"]
