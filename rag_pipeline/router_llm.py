# router_llm.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Protocol, Dict, Any, List, Optional, Callable, Tuple
import json
import hashlib
import os
import time
import logging

logger = logging.getLogger(__name__)

# ---------- Contracts & Types ----------

class SearchModule(Protocol):
    """Pluggable search module contract."""
    name: str

    def is_available(self) -> bool:
        """Return True if the backend/dependencies are configured."""
        ...

    def supports(self, query: str) -> bool:
        """Optional local heuristic gate. Default True."""
        return True

    def execute(self, query: str, ctx: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute search and return results in a normalized envelope:
        {
          "module": self.name,
          "items": [...],   # backend-native results
          "meta": {...}     # any diagnostics
        }
        """
        ...

@dataclass(frozen=True)
class LLMDecision:
    use_semantic: bool
    use_spatial: bool
    use_graph: bool
    rationale: Dict[str, str]  # reasons per flag

@dataclass(frozen=True)
class SearchPlan:
    query: str
    chosen_modules: List[str]
    rationale: Dict[str, str]

# ---------- LLM Client (DI-friendly) ----------

class LLMClient(Protocol):
    def complete_json(self, prompt: str, schema_hint: str, temperature: float = 0.0, max_tokens: int = 256) -> Dict[str, Any]:
        ...

class AnvilGPTClient:
    """
    Uses the same AnvilGPT API as generation.py via llm_utils.call_llm.
    Parses JSON responses for routing decisions.
    """
    def __init__(self, model: Optional[str] = None):
        self.model = model  # Not used directly, call_llm reads ANVILGPT_MODEL from env

    def complete_json(self, prompt: str, schema_hint: str, temperature: float = 0.0, max_tokens: int = 256) -> Dict[str, Any]:
        try:
            # Import here to avoid circular dependency if llm_utils imports from this module
            from rag_pipeline.llm_utils import call_llm
            
            # Build the full prompt requesting JSON output
            full_prompt = (
                f"{prompt}\n\n"
                f"IMPORTANT: Respond ONLY with valid JSON matching this schema:\n{schema_hint}\n"
                f"Do not include any explanation or markdown formatting, just the raw JSON object."
            )
            
            response_text = call_llm(full_prompt).strip()
            
            # Try to extract JSON from response (handle markdown code blocks)
            json_text = response_text
            if "```json" in response_text:
                # Extract from markdown code block
                start = response_text.find("```json") + 7
                end = response_text.find("```", start)
                json_text = response_text[start:end].strip()
            elif "```" in response_text:
                # Extract from generic code block
                start = response_text.find("```") + 3
                end = response_text.find("```", start)
                json_text = response_text[start:end].strip()
            
            # Parse JSON
            parsed = json.loads(json_text)
            logger.debug(f"AnvilGPT routing decision: {parsed}")
            return parsed
            
        except ImportError:
            # Fallback if rag_pipeline not available (e.g., in standalone mode)
            logger.warning("rag_pipeline.llm_utils not available, using conservative fallback")
            return self._conservative_fallback()
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse JSON from AnvilGPT response: {e}\nResponse: {response_text[:200]}")
            return self._conservative_fallback()
        except Exception as e:
            logger.error(f"AnvilGPT call failed: {e}")
            return self._conservative_fallback()
    
    def _conservative_fallback(self) -> Dict[str, Any]:
        """Conservative fallback when LLM call fails."""
        return {
            "use_semantic": False,
            "use_spatial": False,
            "use_graph": False,
            "rationale": {
                "semantic": "LLM call failed, using keyword-only mode",
                "spatial": "LLM call failed, using keyword-only mode",
                "graph": "LLM call failed, using keyword-only mode",
            },
        }

# ---------- Prompting ----------

LLM_SYSTEM = (
    "You are a routing planner for a retrieval pipeline. "
    "Given a user query, decide which additional search modalities improve recall/precision. "
    "Keyword search is ALWAYS used outside of your decision."
)

LLM_USER_TEMPLATE = """Decide which search methods to add for this query.

Query:

{query}

Guidelines:
- semantic: enable when synonyms/paraphrase/generalization likely help (ambiguous wording, 'similar to', 'about', 'compare', long-form questions).
- spatial: enable when locations/coordinates/regions/distance/routing/maps are involved (place names, lat/lon, 'near', 'within X miles', addresses).
- graph: enable when entity-relationship traversal, paths, hierarchies, or joining across entities helps (people-orgs, dependencies, 'connected to', 'path between').

Return STRICT JSON only:
{{
  "use_semantic": true|false,
  "use_spatial": true|false,
  "use_graph": true|false,
  "rationale": {{
    "semantic": "<short reason>",
    "spatial": "<short reason>",
    "graph": "<short reason>"
  }}
}}
"""

LLM_JSON_SCHEMA_HINT = """
{
  "type": "object",
  "properties": {
    "use_semantic": {"type": "boolean"},
    "use_spatial": {"type": "boolean"},
    "use_graph": {"type": "boolean"},
    "rationale": {
      "type": "object",
      "properties": {
        "semantic": {"type": "string"},
        "spatial": {"type": "string"},
        "graph": {"type": "string"}
      },
      "required": ["semantic", "spatial", "graph"]
    }
  },
  "required": ["use_semantic", "use_spatial", "use_graph", "rationale"]
}
"""

def build_llm_prompt(query: str) -> str:
    return LLM_USER_TEMPLATE.format(query=query)

# ---------- Decision Engine ----------

class LLMDecisionEngine:
    def __init__(self, llm: Optional[LLMClient] = None, cache_ttl_s: int = 300):
        self.llm = llm or AnvilGPTClient()
        self.cache_ttl_s = cache_ttl_s
        self._cache: Dict[str, Tuple[float, Dict[str, Any]]] = {}

    def _cache_key(self, query: str) -> str:
        return hashlib.sha256(query.encode("utf-8")).hexdigest()

    def _get_cached(self, key: str) -> Optional[Dict[str, Any]]:
        entry = self._cache.get(key)
        if not entry:
            return None
        ts, payload = entry
        if time.time() - ts > self.cache_ttl_s:
            self._cache.pop(key, None)
            return None
        return payload

    def decide(self, query: str) -> LLMDecision:
        k = self._cache_key(query)
        cached = self._get_cached(k)
        if cached:
            return self._parse_decision(cached)

        prompt = build_llm_prompt(query)
        raw = self.llm.complete_json(prompt=prompt, schema_hint=LLM_JSON_SCHEMA_HINT, temperature=0.0, max_tokens=256)
        # Defensive parsing
        try:
            decision = self._parse_decision(raw)
        except Exception:
            decision = LLMDecision(False, False, False, {
                "semantic": "Parse error; falling back to conservative defaults.",
                "spatial": "Parse error; falling back to conservative defaults.",
                "graph": "Parse error; falling back to conservative defaults.",
            })
        self._cache[k] = (time.time(), {
            "use_semantic": decision.use_semantic,
            "use_spatial": decision.use_spatial,
            "use_graph": decision.use_graph,
            "rationale": decision.rationale,
        })
        return decision

    def _parse_decision(self, raw: Dict[str, Any]) -> LLMDecision:
        return LLMDecision(
            use_semantic=bool(raw["use_semantic"]),
            use_spatial=bool(raw["use_spatial"]),
            use_graph=bool(raw["use_graph"]),
            rationale={
                "semantic": str(raw["rationale"]["semantic"]),
                "spatial": str(raw["rationale"]["spatial"]),
                "graph": str(raw["rationale"]["graph"]),
            }
        )

# ---------- Module Registry & Built-in Adapters ----------

class ModuleRegistry:
    def __init__(self):
        self._modules: Dict[str, SearchModule] = {}

    def register(self, module: SearchModule) -> None:
        self._modules[module.name] = module

    def get(self, name: str) -> Optional[SearchModule]:
        return self._modules.get(name)

    def all(self) -> List[SearchModule]:
        return list(self._modules.values())

# Example adapters — wire to your real implementations
class KeywordSearchAdapter:
    name = "keyword"
    def __init__(self, impl: Callable[[str, Dict[str, Any]], Dict[str, Any]]):
        self.impl = impl
    def is_available(self) -> bool: return True
    def supports(self, query: str) -> bool: return True
    def execute(self, query: str, ctx: Dict[str, Any]) -> Dict[str, Any]:
        return self.impl(query, ctx)

class SemanticSearchAdapter:
    name = "semantic"
    def __init__(self, impl: Callable[[str, Dict[str, Any]], Dict[str, Any]]):
        self.impl = impl
    def is_available(self) -> bool:
        return os.getenv("SEMANTIC_BACKEND_ENABLED", "1") == "1"
    def supports(self, query: str) -> bool: return True
    def execute(self, query: str, ctx: Dict[str, Any]) -> Dict[str, Any]:
        return self.impl(query, ctx)

class SpatialSearchAdapter:
    name = "spatial"
    def __init__(self, impl: Callable[[str, Dict[str, Any]], Dict[str, Any]]):
        self.impl = impl
    def is_available(self) -> bool:
        # e.g., check spacy model / maps API keys
        return os.getenv("SPATIAL_BACKEND_ENABLED", "0") == "1"
    def supports(self, query: str) -> bool:
        return True
    def execute(self, query: str, ctx: Dict[str, Any]) -> Dict[str, Any]:
        return self.impl(query, ctx)

class GraphSearchAdapter:
    name = "graph"
    def __init__(self, impl: Callable[[str, Dict[str, Any]], Dict[str, Any]]):
        self.impl = impl
    def is_available(self) -> bool:
        return os.getenv("NEO4J_URI") is not None
    def supports(self, query: str) -> bool:
        return True
    def execute(self, query: str, ctx: Dict[str, Any]) -> Dict[str, Any]:
        return self.impl(query, ctx)

# ---------- Planner & Orchestration ----------

class LLMRouter:
    """
    Builds a SearchPlan via LLM and executes selected modules.
    Always includes 'keyword'; LLM may add 'semantic', 'spatial', 'graph'.
    """
    def __init__(self, registry: ModuleRegistry, llm_engine: Optional[LLMDecisionEngine] = None):
        self.registry = registry
        self.llm_engine = llm_engine or LLMDecisionEngine()

    def plan(self, query: str) -> SearchPlan:
        decision = self.llm_engine.decide(query)
        chosen = ["keyword"]  # always include
        if decision.use_semantic: chosen.append("semantic")
        if decision.use_spatial:  chosen.append("spatial")
        if decision.use_graph:    chosen.append("graph")

        # Filter by availability & local supports
        filtered: List[str] = []
        for name in chosen:
            mod = self.registry.get(name)
            if mod and mod.is_available() and mod.supports(query):
                filtered.append(name)

        # Ensure keyword remains even if not registered (fail-safe)
        if "keyword" not in filtered:
            if self.registry.get("keyword") is None:
                # still include name to signal upstream to handle keyword elsewhere
                filtered.insert(0, "keyword")
            else:
                filtered.insert(0, "keyword")

        return SearchPlan(query=query, chosen_modules=filtered, rationale=decision.rationale)

    def execute(self, plan: SearchPlan, ctx: Dict[str, Any]) -> Dict[str, Any]:
        results: List[Dict[str, Any]] = []
        for name in plan.chosen_modules:
            mod = self.registry.get(name)
            if not mod:
                results.append({"module": name, "items": [], "meta": {"skipped": "module not registered"}})
                continue
            try:
                results.append(mod.execute(plan.query, ctx))
            except Exception as e:
                results.append({"module": name, "items": [], "meta": {"error": str(e)}})

        return {
            "query": plan.query,
            "modules": plan.chosen_modules,
            "rationale": plan.rationale,
            "results": results,
        }

# ---------- Helper: default registry wiring ----------

def build_default_registry(
    keyword_impl: Callable[[str, Dict[str, Any]], Dict[str, Any]],
    semantic_impl: Optional[Callable[[str, Dict[str, Any]], Dict[str, Any]]] = None,
    spatial_impl: Optional[Callable[[str, Dict[str, Any]], Dict[str, Any]]] = None,
    graph_impl: Optional[Callable[[str, Dict[str, Any]], Dict[str, Any]]] = None,
) -> ModuleRegistry:
    reg = ModuleRegistry()
    reg.register(KeywordSearchAdapter(keyword_impl))
    if semantic_impl:
        reg.register(SemanticSearchAdapter(semantic_impl))
    if spatial_impl:
        reg.register(SpatialSearchAdapter(spatial_impl))
    if graph_impl:
        reg.register(GraphSearchAdapter(graph_impl))
    return reg

