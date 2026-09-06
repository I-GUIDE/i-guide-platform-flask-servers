"""Code presented as executed, when nothing ran.

Reproduced on the deployed agent: a turn answered "count Chicago crimes by community area, show
me the code you actually ran" using an MCP tool, then narrated requests/pandas code it never
ran. The peer-level guard watches the PEER's prose; the user reads the SYNTHESIZER's, so the
failure sits one layer above it.

The fix works with the auditor rather than adding a detector of its own: it can only preserve or
release the auditor's OWN findings, never invent one, so a false positive costs at worst the
verdict the auditor already reached.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from agent_runtime.supervisor import graph as g

_ISSUE = {"claim": "Here is the code run: import requests", "reason": "no execution record"}
_AUDIT = {"hallucination_detected": True, "severity": "high", "issues": [_ISSUE]}


# --- the detector: shown vs claimed-as-run ------------------------------------------------

def test_claiming_a_run_is_an_execution_claim():
    for c in ("Here is the code run:",
              "the code I actually ran",
              "the code that was run",
              "Actual stdout from the run:",
              "I ran the script and it printed 42",
              "output from the run"):
        assert g._is_execution_claim(c), c


def test_merely_showing_code_is_not():
    """The false-positive surface: quoting, illustrating, explaining. These must pass through
    to the ordinary amnesty rules untouched."""
    for c in ("Here is the code from the knowledge-base block",
              "You could compute it with geopandas.sjoin",
              "The notebook defines load_chicago_crime_data",
              "This is the approach I would take",
              "The tool's source is reproduced below"):
        assert not g._is_execution_claim(c), c


# --- the environment fact the auditor was missing ------------------------------------------

def test_the_auditor_is_told_when_nothing_ran():
    lines = g._execution_environment_lines({"analysis_results": {"executed": False}})
    assert lines and "NO code was executed" in lines[0]
    assert "some OTHER tool produced it" in lines[0], \
        "it must say a real figure does not license a false provenance claim"


def test_the_auditor_is_told_when_something_ran():
    lines = g._execution_environment_lines({"code_result": {"executed": True}})
    assert lines and "returned successfully" in lines[0]


def test_a_failed_run_is_distinguished_from_no_run():
    lines = g._execution_environment_lines(
        {"code_result": {"executed": False, "execution_error": "ModuleNotFoundError"}})
    assert lines and "RAN AND FAILED" in lines[0]


# --- reconciliation: preserve when nothing ran, release when something did -----------------

def test_the_finding_survives_when_nothing_ran():
    out = g._reconcile_audit_with_artifacts(
        _AUDIT, execution_context={"analysis_results": {"executed": False}},
        artifacts=[], prior_rows=[])
    assert out["issues"] == [_ISSUE]
    assert out.get("severity") == "high"


def test_the_finding_is_released_when_code_really_ran():
    out = g._reconcile_audit_with_artifacts(
        _AUDIT, execution_context={"code_result": {"executed": True}},
        artifacts=[], prior_rows=[])
    assert out["issues"] == []
    assert out["hallucination_detected"] is False


def test_real_numbers_from_another_tool_do_not_release_it():
    """The trap. The figures in such a claim are usually REAL — a tool produced them — so the
    numbers amnesty would drop the issue on the strength of the very numbers whose provenance
    is disputed. That is why this rule is decided first."""
    issue = {"claim": "the code I ran counted 490249 incidents in Austin",
             "reason": "no execute_code record"}
    ctx = {"analysis_results": {"executed": False},
           "tool_results": [{"content": '{"community_area": "Austin", "count": 490249}'}]}
    out = g._reconcile_audit_with_artifacts(
        {"hallucination_detected": True, "severity": "high", "issues": [issue]},
        execution_context=ctx, artifacts=[], prior_rows=[])
    assert out["issues"] == [issue], "a real number must not launder a false provenance claim"


def test_an_ordinary_claim_still_gets_its_amnesties():
    """The new rule must not shadow the existing ones for non-execution claims. The figure has
    to be one _claim_numbers actually extracts — it ignores small integers, so a two-digit
    fixture would pass this test without exercising the amnesty at all."""
    issue = {"claim": "the analysis found 490249 incidents", "reason": "unsupported"}
    ctx = {"analysis_results": {"executed": False},
           "tool_results": [{"content": '{"n": 490249}'}]}
    out = g._reconcile_audit_with_artifacts(
        {"hallucination_detected": True, "severity": "high", "issues": [issue]},
        execution_context=ctx, artifacts=[], prior_rows=[])
    assert out["issues"] == [], "the numbers amnesty should still release this one"


def test_it_never_invents_a_finding():
    """The safety property that makes this the low-risk option: with no audit issues there is
    nothing to preserve, whatever the execution status."""
    clean = {"hallucination_detected": False, "severity": "none", "issues": []}
    out = g._reconcile_audit_with_artifacts(
        clean, execution_context={"analysis_results": {"executed": False}},
        artifacts=[], prior_rows=[])
    assert not out.get("issues")


def test_the_line_actually_reaches_the_audit_prompt():
    """End-to-end wiring: the deterministic rule only fires on issues the auditor RAISED, so if
    the line never reaches the rendered prompt the auditor can mark the claim supported and the
    rule never sees it. This is the half that a unit test of _reconcile_ cannot cover."""
    from agent_runtime.evidence_quality import _format_execution_context

    ctx = {"analysis_results": {"executed": False, "answer": "x"},
           "environment": g._execution_environment_lines({"analysis_results": {"executed": False}})}
    rendered = _format_execution_context(ctx)
    assert "NO code was executed this turn" in rendered, rendered[:400]
    assert "CONTRADICTED" in rendered, "the heading must invite the contradicting direction"


def test_a_real_run_reaches_the_prompt_as_support():
    from agent_runtime.evidence_quality import _format_execution_context

    ctx = {"analysis_results": {"executed": True, "answer": "x"},
           "environment": g._execution_environment_lines({"analysis_results": {"executed": True}})}
    rendered = _format_execution_context(ctx)
    assert "returned successfully" in rendered, rendered[:400]
    assert "NO code was executed" not in rendered
