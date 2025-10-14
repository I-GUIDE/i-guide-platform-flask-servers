from __future__ import annotations

"""
Backwards-compatible entry point for search helpers.

The OpenSearch keyword logic now lives in `search_keyword.py` and the Neo4j
implementation lives in `search_neo4j.py`. This module re-exports the primary
interfaces so existing imports continue to work while providing a single place
to document the split.
"""

from .search_keyword import get_keyword_search_results
from .search_neo4j import get_neo4j_search_results

__all__ = ["get_keyword_search_results", "get_neo4j_search_results"]
