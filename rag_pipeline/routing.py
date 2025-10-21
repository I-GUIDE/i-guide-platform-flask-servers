from __future__ import annotations

from typing import Any, MutableMapping

from .generation import run_generation
from .search_core import run_retrieval
from .state import AgentState, ensure_state_shapes


def rag_pipeline(state: MutableMapping[str, Any]) -> AgentState:
    """
    Run the retrieval and generation stages of the pipeline using shared AgentState.
    """
    state = ensure_state_shapes(state)
    state = run_retrieval(state)
    state = run_generation(state)
    return state  # type: ignore[return-value]


__all__ = ["rag_pipeline"]
