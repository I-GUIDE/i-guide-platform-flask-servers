from __future__ import annotations

import logging
import math
import os
from typing import Any, Dict, Optional

# Configure module-wide logging once so search modules share formatting.
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
logging.basicConfig(level=LOG_LEVEL, format="%(asctime)s %(levelname)s %(name)s: %(message)s")


def get_logger(name: str) -> logging.Logger:
    """Return a namespaced logger configured with the shared log level."""
    return logging.getLogger(name)


def getenv(name: str, required: bool = True, default: Optional[str] = None) -> str:
    value = os.getenv(name, default)
    if required and (value is None or value == ""):
        raise RuntimeError(f"Missing required environment variable: {name}")
    if value and len(value) >= 2 and ((value[0] == value[-1] == '"') or (value[0] == value[-1] == "'")):
        value = value[1:-1]
    return value or ""


def safe_score(val: Any, default: float = 1.0) -> float:
    try:
        score = float(val)
        return score if math.isfinite(score) else default
    except Exception:
        return default


def normalize_source_fields(source: Dict[str, Any], fallback_id: str) -> Dict[str, Any]:
    if not isinstance(source, dict):
        source = {}
    source = dict(source)

    source.setdefault("doc_id", fallback_id)
    source.setdefault("title", source.get("name") or "No Title")
    source.setdefault("contents", source.get("abstract") or source.get("description") or "No Content")
    if "element_type" not in source and "resource-type" in source:
        source["element_type"] = source["resource-type"]

    return source


__all__ = [
    "get_logger",
    "getenv",
    "normalize_source_fields",
    "safe_score",
]
