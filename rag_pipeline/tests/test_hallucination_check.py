#!/usr/bin/env python3
"""
Run the full RAG pipeline for a query and audit the generated answer for hallucinations.
"""

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

from dotenv import load_dotenv

from rag_pipeline.hallucination_check import evaluate_hallucination
from rag_pipeline.pipeline import run_pipeline


def _load_env() -> None:
    env_path = Path(__file__).parent.parent / ".env.local"
    if env_path.exists():
        load_dotenv(env_path)
    else:
        load_dotenv()


def _print_answer(state):
    answer = state["answer"]
    print("\n=== FINAL ANSWER ===")
    print(answer.get("final_composed_answer", "No answer produced."))
    print("\nCitations:", answer.get("citations", []))
    print("Confidence:", answer.get("confidence_score"))


def _print_hallucination_report(report):
    print("\n=== HALLUCINATION CHECK ===")
    print(f"Detected: {report.get('hallucination_detected')}")
    print(f"Severity: {report.get('severity')}")
    print(f"Summary: {report.get('summary')}")
    issues = report.get("issues") or []
    if issues:
        print("\nIssues:")
        for idx, issue in enumerate(issues, 1):
            claim = issue.get("claim", "")
            reason = issue.get("reason", "")
            print(f"  {idx}. Claim: {claim}")
            print(f"     Reason: {reason}")
    else:
        print("No specific issues reported.")


def main():
    _load_env()
    parser = argparse.ArgumentParser(description="Run pipeline and check for hallucinations.")
    default_query = "Summarize the recent wildfire mitigation efforts in Texas."
    parser.add_argument(
        "--query",
        default=default_query,
        help=f"User query to run through the pipeline (default: {default_query!r}).",
    )
    parser.add_argument("--top-k", type=int, default=10, help="Retrieval top-k.")
    parser.add_argument("--evidence-limit", type=int, default=5, help="How many docs to show the hallucination checker.")
    args = parser.parse_args()

    state = run_pipeline(user_input=args.query, params={"top_k": args.top_k})
    _print_answer(state)

    report = evaluate_hallucination(state, evidence_limit=args.evidence_limit)
    _print_hallucination_report(report)


if __name__ == "__main__":
    main()
