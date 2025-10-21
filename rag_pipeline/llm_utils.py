from __future__ import annotations

import logging
from typing import Callable, Optional

logger = logging.getLogger(__name__)

_llm_callable: Optional[Callable[[str], str]] = None


def register_llm_callable(func: Callable[[str], str]) -> None:
    """
    Allow applications to register a synchronous LLM callable.
    """
    global _llm_callable
    _llm_callable = func


def call_llm(prompt: str) -> str:
    """
    Thin wrapper. In tests, monkeypatch to return deterministic text.
    In production, register an LLM callable via `register_llm_callable`.
    Must never raise.
    """
    try:
        if _llm_callable is None:
            return "LLM backend not configured."
        result = _llm_callable(prompt)
        if isinstance(result, str):
            return result
        logger.warning("LLM callable returned non-string result; coercing to string.")
        return str(result)
    except Exception as exc:  # pragma: no cover - defensive fail-safe
        logger.error("LLM call failed: %s", exc)
        return "I could not compose an answer due to a generation error."


__all__ = ["call_llm", "register_llm_callable"]
