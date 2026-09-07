"""A run's own output must survive, even when it lands on a staged input's name.

`_execute` excludes staged input names from BOTH the persisted artifacts and the durable
workspace copy-back, so an upload is not re-persisted as though the run had produced it. That
exclusion is unconditional and keyed on the NAME, with no check of whether the run modified the
file — unlike CARRIED workspace files, which get a (size, mtime_ns) comparison via `unchanged`.

So a run that writes its result under a staged input's name loses that result from both places,
silently. Reproduced against the real code path before any fix:

    ARTIFACTS: ['...py', 'summary.txt']      # data.csv absent
    WORKSPACE: ['summary.txt']               # data.csv absent

These tests pin the behaviour in both directions: a MODIFIED staged input is an output and must
survive; a PRISTINE staged input is still just an upload and must not be re-persisted.
"""
from __future__ import annotations

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))


def _exec_env(monkeypatch, tmp_path):
    """Point the file store, work root and workspaces at tmp_path."""
    monkeypatch.setenv("AGENT_FILE_STORAGE_ROOT", str(tmp_path / "fs"))
    monkeypatch.setenv("AGENT_CODE_EXEC_WORK_ROOT", str(tmp_path / "work"))
    monkeypatch.setenv("AGENT_CODE_EXEC_WORKSPACES", "1")


def _upload(tmp_path, name, text):
    p = tmp_path / "uploads" / name
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(text, encoding="utf-8")
    return p


def test_an_output_written_over_a_staged_input_is_kept(monkeypatch, tmp_path):
    from agent_runtime.code_execution import LocalSubprocessExecutor, _session_workspace

    _exec_env(monkeypatch, tmp_path)
    src = _upload(tmp_path, "data.csv", "id,v\n1,10\n2,20\n")
    session = "keep::overwritten"
    ws = _session_workspace(session)

    r = LocalSubprocessExecutor().execute(
        "open('data.csv','w').write('id,v\\n1,100\\n2,200\\n')\n"
        "open('summary.txt','w').write('done')\n",
        session=session, input_files=[{"source": str(src), "dest": "data.csv"}])

    assert r.ok, r.stderr
    names = [a["filename"] for a in r.artifacts]
    assert "summary.txt" in names, "a freshly named output must survive (control)"
    assert "data.csv" in names, (
        "the run REWROTE data.csv; that content is its output, not the upload. "
        f"artifacts={names}")

    ws_files = sorted(p.name for p in ws.rglob("*") if p.is_file())
    assert "data.csv" in ws_files, (
        f"the derived file must also reach the durable workspace; workspace={ws_files}")
    assert "100" in (ws / "data.csv").read_text(), "the workspace copy must be the RUN's version"


def test_an_untouched_staged_input_is_not_re_persisted(monkeypatch, tmp_path):
    """The reason the exclusion exists. Narrowing it must not start re-persisting uploads."""
    from agent_runtime.code_execution import LocalSubprocessExecutor, _session_workspace

    _exec_env(monkeypatch, tmp_path)
    src = _upload(tmp_path, "data.csv", "id,v\n1,10\n")
    session = "keep::pristine"
    ws = _session_workspace(session)

    r = LocalSubprocessExecutor().execute(
        "rows = open('data.csv').read()\nopen('summary.txt','w').write(str(len(rows)))\n",
        session=session, input_files=[{"source": str(src), "dest": "data.csv"}])

    assert r.ok, r.stderr
    names = [a["filename"] for a in r.artifacts]
    assert "summary.txt" in names
    assert "data.csv" not in names, f"a read-only upload must not be re-persisted; got {names}"

    ws_files = sorted(p.name for p in ws.rglob("*") if p.is_file())
    assert "data.csv" not in ws_files, (
        f"a pristine upload must not litter the durable workspace; workspace={ws_files}")


def test_the_file_id_copy_of_an_untouched_upload_stays_out(monkeypatch, tmp_path):
    """Inputs are staged under BOTH file_id and filename. Rewriting one must not drag the
    other into the workspace under its opaque file_id name."""
    from agent_runtime.code_execution import LocalSubprocessExecutor, _session_workspace

    _exec_env(monkeypatch, tmp_path)
    src = _upload(tmp_path, "data.csv", "id,v\n1,10\n")
    session = "keep::twonames"
    ws = _session_workspace(session)

    r = LocalSubprocessExecutor().execute(
        "open('data.csv','w').write('rewritten')\n",
        session=session,
        input_files=[{"source": str(src), "dest": "data.csv"},
                     {"source": str(src), "dest": "fileid-abc123"}])

    assert r.ok, r.stderr
    ws_files = sorted(p.name for p in ws.rglob("*") if p.is_file())
    assert "data.csv" in ws_files, ws_files
    assert "fileid-abc123" not in ws_files, (
        f"the untouched file_id-named copy must stay out of the workspace; got {ws_files}")


def test_a_shadowed_workspace_file_rewritten_by_the_run_is_not_reverted(monkeypatch, tmp_path):
    """The nastier variant: the name is ALREADY a durable workspace file and an upload of the
    same name shadows it. Before the fix the run's rewrite was dropped and the workspace kept
    the PREVIOUS run's content, so the conversation silently lost the new result."""
    from agent_runtime.code_execution import LocalSubprocessExecutor, _session_workspace

    _exec_env(monkeypatch, tmp_path)
    session = "keep::shadowed"
    ws = _session_workspace(session)

    first = LocalSubprocessExecutor().execute(
        "open('data.csv','w').write('id,v\\n9,999\\n')", session=session)
    assert first.ok and (ws / "data.csv").is_file()

    src = _upload(tmp_path, "data.csv", "id,v\n1,1\n")
    second = LocalSubprocessExecutor().execute(
        "open('data.csv','w').write('id,v\\n7,777\\n')",
        session=session, input_files=[{"source": str(src), "dest": "data.csv"}])

    assert second.ok, second.stderr
    body = (ws / "data.csv").read_text()
    assert "777" in body, (
        "the second run rewrote data.csv; the workspace must hold that, not the first run's "
        f"content. got {body!r}")
    assert "data.csv" in [a["filename"] for a in second.artifacts]


# --- the CLI code peers -------------------------------------------------------------------
# The same unconditional by-name exclusion lives in both CLI peers, and it is WORSE there:
# editing a file in place is the entire job of a coding agent, and opencode's work dir is a
# mkdtemp deleted in its `finally`, so a rewritten input is not merely withheld -- it is gone.
# claude_peer is the sharpest case: its exclude expression already ORs in the signature-checked
# `already_persisted(work)` ("a file it edited SHOULD be sent again") right beside an
# unconditional `set(staging["staged"])`. Same expression, two policies.
#
# These assert on the EXCLUDE SET handed to _persist_artifacts, which is where the policy
# lives -- no file store, no container.

def _peer_harness(monkeypatch, tmp_path, mod, writes):
    """Stage one upload for real, stub the container, capture the exclude set."""
    import subprocess
    import time

    monkeypatch.setenv("AGENT_CODE_EXEC_WORK_ROOT", str(tmp_path / "work"))
    upload = tmp_path / "uploads" / "data.csv"
    upload.parent.mkdir(parents=True, exist_ok=True)
    upload.write_text("id,v\n1,10\n", encoding="utf-8")

    def stage(work, ids):
        from agent_runtime.code_execution import _stage_inputs
        staged, errs, _ = _stage_inputs(work, [{"source": str(upload), "dest": "data.csv"}])
        return {"staged": staged, "staged_info": [], "errors": errs, "skipped": []}

    monkeypatch.setattr(mod, "_stage_conversation_files", stage)

    def fake_run(argv, **kwargs):
        mount = [a for a in argv if a.endswith(":/work:rw")]
        work = pathlib.Path(mount[0].split(":")[0])
        time.sleep(0.01)                       # keep mtime strictly after the staging stamp
        for name, body in writes.items():
            (work / name).write_text(body, encoding="utf-8")
        return subprocess.CompletedProcess(argv, 0, "{}", "")

    monkeypatch.setattr(subprocess, "run", fake_run)
    captured = {}

    def capture(work, exclude, *, defer=None):
        captured["exclude"] = set(exclude)
        captured["defer"] = set(defer or ())
        captured["on_disk"] = {p.name for p in work.iterdir() if p.is_file()}
        return []

    monkeypatch.setattr(mod, "_persist_artifacts", capture)
    return captured


def test_opencode_peer_keeps_a_rewritten_input(monkeypatch, tmp_path):
    import agent_runtime.opencode_peer as ocp

    monkeypatch.setenv("AGENT_OPENCODE_API_KEY", "sk-test")
    cap = _peer_harness(monkeypatch, tmp_path, ocp,
                        {"data.csv": "id,v\n1,999\n", "notes.txt": "ok"})
    ocp.run_opencode("do it", input_file_ids=["x"])
    assert "data.csv" in cap["on_disk"]
    assert "data.csv" not in cap["exclude"], (
        "the peer REWROTE data.csv in place; that is its result, not the upload. "
        f"exclude={cap['exclude']}")


def test_opencode_peer_still_drops_an_untouched_input(monkeypatch, tmp_path):
    import agent_runtime.opencode_peer as ocp

    monkeypatch.setenv("AGENT_OPENCODE_API_KEY", "sk-test")
    cap = _peer_harness(monkeypatch, tmp_path, ocp, {"notes.txt": "ok"})
    ocp.run_opencode("do it", input_file_ids=["x"])
    assert "data.csv" in cap["exclude"], (
        f"a read-only upload must still be withheld; exclude={cap['exclude']}")


def test_claude_peer_keeps_a_rewritten_input(monkeypatch, tmp_path):
    import agent_runtime.claude_peer as ccp

    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test")
    cap = _peer_harness(monkeypatch, tmp_path, ccp,
                        {"data.csv": "id,v\n1,999\n", "notes.txt": "ok"})
    ccp.run_claude("do it", input_file_ids=["x"])
    assert "data.csv" in cap["on_disk"]
    assert "data.csv" not in cap["exclude"], (
        "claude_peer withheld a file it edited, right next to already_persisted which would "
        f"not have. exclude={cap['exclude']}")


def test_claude_peer_still_drops_an_untouched_input(monkeypatch, tmp_path):
    import agent_runtime.claude_peer as ccp

    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test")
    cap = _peer_harness(monkeypatch, tmp_path, ccp, {"notes.txt": "ok"})
    ccp.run_claude("do it", input_file_ids=["x"])
    assert "data.csv" in cap["exclude"], (
        f"a read-only upload must still be withheld; exclude={cap['exclude']}")


def test_a_deleted_staged_input_is_not_resurrected(monkeypatch, tmp_path):
    """Absent from the after-map means "not pristine", which must not turn into "copy it back".
    There is nothing left to copy, and the comparison must not raise on the missing stat."""
    from agent_runtime.code_execution import LocalSubprocessExecutor, _session_workspace

    _exec_env(monkeypatch, tmp_path)
    src = _upload(tmp_path, "data.csv", "id,v\n1,10\n")
    session = "keep::deleted"
    ws = _session_workspace(session)

    r = LocalSubprocessExecutor().execute(
        "import os\nos.remove('data.csv')\nopen('summary.txt','w').write('gone')\n",
        session=session, input_files=[{"source": str(src), "dest": "data.csv"}])

    assert r.ok, r.stderr
    names = [a["filename"] for a in r.artifacts]
    assert "summary.txt" in names
    assert "data.csv" not in names
    ws_files = sorted(p.name for p in ws.rglob("*") if p.is_file())
    assert "data.csv" not in ws_files, f"a deleted input must not reappear; workspace={ws_files}"


def test_an_output_in_a_subdirectory_is_unaffected(monkeypatch, tmp_path):
    """_stage_inputs rejects separators in dest, so every staged name is top-level. A run that
    writes out/data.csv was never at risk and must stay unaffected by the narrowing."""
    from agent_runtime.code_execution import LocalSubprocessExecutor

    _exec_env(monkeypatch, tmp_path)
    src = _upload(tmp_path, "data.csv", "id,v\n1,10\n")
    r = LocalSubprocessExecutor().execute(
        "import os\nos.makedirs('out', exist_ok=True)\n"
        "open('out/data.csv','w').write('derived')\n",
        session="keep::subdir", input_files=[{"source": str(src), "dest": "data.csv"}])

    assert r.ok, r.stderr
    names = [a["filename"] for a in r.artifacts]
    assert "data.csv" in names, f"out/data.csv persists under its basename; got {names}"


def test_rewritten_inputs_never_displace_the_runs_own_outputs(monkeypatch, tmp_path):
    """The regression an adversarial review caught in the first cut of this fix.

    `_persist_artifacts` applies `exclude` BEFORE the `len(artifacts) >= MAX_ARTIFACTS` break,
    so once rewritten inputs stopped being excluded they began consuming the cap. With names
    that sort early ('aaa_*') they pushed the run's own later outputs past the break — trading
    the loss this narrowing exists to fix for a different one. Rewritten inputs are now
    deferred to the end of the budget.
    """
    from agent_runtime import code_execution as ce

    work = tmp_path / "w"
    work.mkdir()
    inputs = [f"aaa_{i}.csv" for i in range(5)]      # sort before every output
    outputs = [f"zzz_{i}.txt" for i in range(ce.MAX_ARTIFACTS)]
    for n in inputs + outputs:
        (work / n).write_text("x", encoding="utf-8")

    stored = []
    monkeypatch.setattr(
        "agent_runtime.file_store.create_output_file_from_path",
        lambda path, filename=None: (stored.append(filename or path.name),
                                     {"file_id": f"id{len(stored)}", "filename": filename,
                                      "download_url": None, "size_bytes": 1})[1])

    got = [a["filename"] for a in ce._persist_artifacts(work, set(), defer=set(inputs))]
    assert len(got) == ce.MAX_ARTIFACTS
    for n in outputs:
        assert n in got, f"a genuine output was displaced by a rewritten input: {n} missing"

    # Without the deferral the early-sorting inputs take the first slots and outputs fall off.
    stored.clear()
    naive = [a["filename"] for a in ce._persist_artifacts(work, set())]
    assert any(n in naive for n in inputs), "sanity: undeferred, inputs do claim slots"
    assert not all(n in naive for n in outputs), "sanity: undeferred, outputs are displaced"
