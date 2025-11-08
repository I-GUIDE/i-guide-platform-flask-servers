# tests/test_router_llm.py
import os
import sys

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

from rag_pipeline.router_llm import LLMRouter, build_default_registry, LLMDecisionEngine, LLMDecision

class FakeLLM:
    def __init__(self, payload):
        self.payload = payload
    def complete_json(self, *args, **kwargs):
        return self.payload

def keyword_impl(q, c): return {"module":"keyword","items":[1],"meta":{}}
def semantic_impl(q, c): return {"module":"semantic","items":[2],"meta":{}}
def spatial_impl(q, c): return {"module":"spatial","items":[3],"meta":{}}
def graph_impl(q, c): return {"module":"graph","items":[4],"meta":{}}

def test_always_includes_keyword():
    reg = build_default_registry(keyword_impl)
    engine = LLMDecisionEngine(llm=FakeLLM({
        "use_semantic": False, "use_spatial": False, "use_graph": False,
        "rationale": {"semantic":"", "spatial":"", "graph":""}
    }))
    router = LLMRouter(registry=reg, llm_engine=engine)
    plan = router.plan("hello")
    assert "keyword" in plan.chosen_modules

def test_adds_semantic_when_llm_true():
    reg = build_default_registry(keyword_impl, semantic_impl)
    engine = LLMDecisionEngine(llm=FakeLLM({
        "use_semantic": True, "use_spatial": False, "use_graph": False,
        "rationale": {"semantic":"ok", "spatial":"", "graph":""}
    }))
    router = LLMRouter(registry=reg, llm_engine=engine)
    plan = router.plan("ambiguous synonym question")
    assert "semantic" in plan.chosen_modules

def test_respects_availability_flags(monkeypatch):
    monkeypatch.setenv("SPATIAL_BACKEND_ENABLED", "0")
    reg = build_default_registry(keyword_impl, spatial_impl=spatial_impl)
    engine = LLMDecisionEngine(llm=FakeLLM({
        "use_semantic": False, "use_spatial": True, "use_graph": False,
        "rationale": {"semantic":"", "spatial":"ok", "graph":""}
    }))
    router = LLMRouter(registry=reg, llm_engine=engine)
    plan = router.plan("near me")
    assert "spatial" not in plan.chosen_modules  # unavailable

def test_execute_returns_envelopes():
    reg = build_default_registry(keyword_impl, semantic_impl, spatial_impl, graph_impl)
    engine = LLMDecisionEngine(llm=FakeLLM({
        "use_semantic": True, "use_spatial": True, "use_graph": True,
        "rationale": {"semantic":"", "spatial":"", "graph":""}
    }))
    router = LLMRouter(registry=reg, llm_engine=engine)
    out = router.execute(router.plan("complex"), ctx={})
    assert out["modules"][0] == "keyword"
    assert len(out["results"]) >= 1

