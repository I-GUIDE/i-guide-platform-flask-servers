"""
Functional test that exercises keyword search together with the generation
pipeline using a sample prompt. The test loads environment variables from the
local dotenv files so it can talk to the configured OpenSearch and LLM
services.
"""

import asyncio
from pathlib import Path
from typing import Any, Dict

from dotenv import load_dotenv

import os

from rag_pipeline.generation import generate_answer
from rag_pipeline.search_keyword import run_keyword_search
from rag_pipeline.state import ensure_state_shapes


def _load_env() -> None:
    """
    Load environment variables from the repository root and rag_pipeline
    directories so the test runs with the same configuration as the main app.
    """
    preserved = dict(os.environ)
    repo_root = Path(__file__).resolve().parents[2]
    rag_root = Path(__file__).resolve().parents[1]
    for env_file in (repo_root / ".env", rag_root / ".env", rag_root / ".env.local"):
        if env_file.exists():
            load_dotenv(env_file, override=True)
    os.environ.update(preserved)
    if "OPENAI_API_KEY" not in os.environ and os.environ.get("OPENAI_KEY"):
        os.environ["OPENAI_API_KEY"] = os.environ["OPENAI_KEY"]


def _build_initial_state(prompt: str) -> Dict[str, Any]:
    """
    Construct the minimal agent state required by the keyword search and
    generation modules.
    """
    state: Dict[str, Any] = {
        "query_information": {
            "raw_text": prompt,
            "language": "en-US",
            "metadata": {"source": "functionality_test"},
            "context_hints": {"filters": []},
        },
        "session_context": {"turn_number": 0},
        "evidence": {},
        "answer": {},
    }
    ensure_state_shapes(state)
    return state


async def run_functionality_test(prompt: str) -> Dict[str, Any]:
    """
    Execute keyword search followed by answer generation for the supplied
    prompt and return a concise report with document count and answer text.
    """
    state = _build_initial_state(prompt)
    retrieved = run_keyword_search(state, query=prompt, limit=5)
    if not retrieved:
        raise RuntimeError("Keyword search returned no documents.")

    updated_state = await generate_answer(state)
    answer = updated_state["answer"]["final_composed_answer"] or ""
    if not answer.strip():
        raise RuntimeError("Generation produced an empty answer.")

    return {
        "retrieved_documents": len(updated_state["evidence"]["retrieved_documents"]),
        "answer": answer,
    }


def main() -> None:
    _load_env()
    prompt = "How is social media data used in geospatial analysis?"

    try:
        result = asyncio.run(run_functionality_test(prompt))
    except Exception as exc:
        print(f"[FAIL] Functionality test failed: {exc}")
        raise SystemExit(1) from exc

    print("[PASS] Keyword + generation test succeeded.")
    print(f"Retrieved documents: {result['retrieved_documents']}")
    print("Generated answer:\n")
    print(result["answer"])


if __name__ == "__main__":
    main()
