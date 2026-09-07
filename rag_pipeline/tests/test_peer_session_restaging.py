"""Multi-turn behaviour of the claude peer's PERSISTENT session directory.

No existing peer test drives two turns against one thread_id, which is why this class of bug
was invisible to the suite: `_stage_conversation_files` re-staged the same conversation uploads
on EVERY turn and `_stage_inputs` overwrote, so a peer that edited an attached file in place had
its edit replaced by the pristine upload at the start of the next turn. Measured before the fix:

    turn 1: before='id,v | 1,ORIGINAL'   data.csv ids=['file_790b4a663e4b']
    turn 2: before='id,v | 1,ORIGINAL'   data.csv ids=['file_92ad3dc06532']
    turn 3: before='id,v | 1,ORIGINAL'   data.csv ids=['file_606b61ba1cfd']

— the peer never saw its own work (so a multi-turn task restarted from the original each turn),
and because the clobber reset the baseline every turn the same logical file was re-delivered as
a NEW artifact each time.

Policy chosen: the peer's edit wins. An existing dest is not overwritten, and the peer is told
which files are its own and that the pristine original is still reachable under its file id.
"""
from __future__ import annotations

import json
import pathlib
import subprocess
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))


def _thread_run(monkeypatch, tmp_path, thread, uploads, edits_by_turn, turns=2):
    """Drive N turns of run_claude on ONE thread id, with the container stubbed.

    `edits_by_turn` maps turn number -> {name: body} the fake peer writes that turn.
    Returns (per-turn artifact names, per-turn data.csv file_ids, prompts, pre-run bodies).
    """
    monkeypatch.setenv("AGENT_FILE_STORAGE_ROOT", str(tmp_path / "fs"))
    monkeypatch.setenv("AGENT_CODE_EXEC_WORK_ROOT", str(tmp_path / "work"))
    monkeypatch.setenv("AGENT_CLAUDE_PERSIST", "1")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test")

    import agent_runtime.claude_peer as ccp
    from agent_runtime.file_store import create_output_file

    ids = [create_output_file(n, b)["file_id"] for n, b in uploads]
    state = {"turn": 0}
    before, works, prompts = [], [], []

    def fake_run(argv, **kwargs):
        work = pathlib.Path([a for a in argv if a.endswith(":/work:rw")][0].split(":")[0])
        works.append(str(work))
        f = work / "data.csv"
        before.append(f.read_text(encoding="utf-8") if f.is_file() else "<absent>")
        for name, body in (edits_by_turn.get(state["turn"]) or {}).items():
            (work / name).write_text(body, encoding="utf-8")
        return subprocess.CompletedProcess(
            argv, 0, json.dumps({"result": "ok", "is_error": False}), "")

    monkeypatch.setattr(subprocess, "run", fake_run)
    real_argv = ccp.build_docker_argv

    def spy(work, name, model, prompt, *a, **k):
        prompts.append(prompt)
        return real_argv(work, name, model, prompt, *a, **k)

    monkeypatch.setattr(ccp, "build_docker_argv", spy)

    arts, dids = [], []
    for t in range(1, turns + 1):
        state["turn"] = t
        out = ccp.run_claude("do it", input_file_ids=ids, thread_id=thread)
        got = [(a.get("filename"), a.get("file_id")) for a in out.get("artifacts") or []]
        arts.append([n for n, _ in got])
        dids.append([i for n, i in got if n == "data.csv"])
    assert len(set(works)) == 1, f"the whole point is one persistent dir; got {set(works)}"
    return arts, dids, prompts, before


def test_the_peer_keeps_its_own_edit_across_turns(monkeypatch, tmp_path):
    arts, dids, prompts, before = _thread_run(
        monkeypatch, tmp_path, "t::keep",
        uploads=[("data.csv", "id,v\n1,ORIGINAL\n")],
        edits_by_turn={1: {"data.csv": "id,v\n1,CLEANED\n"}},
        turns=3)

    assert "ORIGINAL" in before[0], before
    assert all("CLEANED" in b for b in before[1:]), (
        f"the attachment must not be re-copied over the peer's edit; saw {before}")


def test_an_untouched_edit_is_delivered_once_not_once_per_turn(monkeypatch, tmp_path):
    """An INVARIANT, not a regression test: this held before the fix too (via `pristine`, since
    the clobbered file matched the freshly staged upload) and must keep holding now that it holds
    for the right reason -- the file is unchanged since the staging baseline. Kept because the
    fix moves WHICH rule excludes it, and getting that wrong would re-deliver the peer's work
    every turn."""
    arts, dids, _, _ = _thread_run(
        monkeypatch, tmp_path, "t::once",
        uploads=[("data.csv", "id,v\n1,ORIGINAL\n")],
        edits_by_turn={1: {"data.csv": "id,v\n1,CLEANED\n"},
                       2: {"note2.txt": "x"}, 3: {"note3.txt": "y"}},
        turns=3)

    flat = [i for turn in dids for i in turn]
    assert len(flat) == 1, f"data.csv should reach the store once, got {len(flat)}: {dids}"
    assert arts[1] == ["note2.txt"] and arts[2] == ["note3.txt"], arts


def test_the_peer_is_told_which_files_are_its_own(monkeypatch, tmp_path):
    """Otherwise it reads its own earlier edit believing it is the user's attachment, and
    'start over from the original' becomes unanswerable."""
    _, _, prompts, _ = _thread_run(
        monkeypatch, tmp_path, "t::told",
        uploads=[("data.csv", "id,v\n1,ORIGINAL\n")],
        edits_by_turn={1: {"data.csv": "id,v\n1,CLEANED\n"}},
        turns=2)

    assert "data.csv" in prompts[1] and "earlier turn" in prompts[1], prompts[1][-500:]
    assert "file id" in prompts[1], "the pristine original must stay findable"
    assert "earlier turn" not in prompts[0], "nothing was kept on turn 1 -- no note"


def test_a_corrected_re_upload_is_still_reachable(monkeypatch, tmp_path):
    """The cost of this policy: a re-uploaded corrected file is NOT applied under its own
    filename, because the peer's edit holds that name. It must still arrive under its new
    file_id, or the user's correction would be unreachable."""
    monkeypatch.setenv("AGENT_FILE_STORAGE_ROOT", str(tmp_path / "fs"))
    monkeypatch.setenv("AGENT_CODE_EXEC_WORK_ROOT", str(tmp_path / "work"))
    monkeypatch.setenv("AGENT_CLAUDE_PERSIST", "1")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test")
    import agent_runtime.claude_peer as ccp
    from agent_runtime.file_store import create_output_file

    first = create_output_file("data.csv", "id,v\n1,ORIGINAL\n")["file_id"]
    state = {"turn": 0, "seen": {}}

    def fake_run(argv, **kwargs):
        work = pathlib.Path([a for a in argv if a.endswith(":/work:rw")][0].split(":")[0])
        if state["turn"] == 1:
            (work / "data.csv").write_text("id,v\n1,CLEANED\n", encoding="utf-8")
        state["seen"] = {p.name: p.read_text(encoding="utf-8", errors="replace")
                         for p in work.iterdir() if p.is_file()}
        return subprocess.CompletedProcess(
            argv, 0, json.dumps({"result": "ok", "is_error": False}), "")

    monkeypatch.setattr(subprocess, "run", fake_run)

    state["turn"] = 1
    ccp.run_claude("clean it", input_file_ids=[first], thread_id="t::reupload")

    corrected = create_output_file("data.csv", "id,v\n1,CORRECTED-BY-USER\n")["file_id"]
    state["turn"] = 2
    ccp.run_claude("now use my fixed file", input_file_ids=[first, corrected],
                   thread_id="t::reupload")

    seen = state["seen"]
    assert "CLEANED" in seen["data.csv"], (
        "the peer's edit holds the human filename -- that is the chosen policy")
    assert corrected in seen, f"the corrected upload must be staged under its id; got {sorted(seen)}"
    assert "CORRECTED-BY-USER" in seen[corrected], seen[corrected]


def test_a_rewrite_every_turn_still_duplicates_and_that_is_not_this_fix(monkeypatch, tmp_path):
    """Pins a RESIDUAL, so nobody later believes this fix removed it.

    The task this came from claimed the duplicate-artifact-per-turn was caused by the clobber
    resetting the baseline. Measured both ways, that attribution is wrong -- a peer that
    rewrites the file every turn produced three distinct file ids before the fix AND after it:

        pre-fix : peer saw its own edit = False, distinct data.csv ids = 3
        post-fix: peer saw its own edit = True,  distinct data.csv ids = 3

    The cause is (size, mtime_ns) change detection: rewriting identical bytes bumps mtime, so
    the file genuinely reads as modified each turn. That is the documented, deliberate trade --
    "a duplicate is noise, a withheld one is a lost result". Removing it needs content-addressed
    delivery (hash the manifest instead of stat-ing it), which is a separate change with its own
    cost, not a consequence of where uploads get staged."""
    arts, dids, _, before = _thread_run(
        monkeypatch, tmp_path, "t::residual",
        uploads=[("data.csv", "id,v\n1,ORIGINAL\n")],
        edits_by_turn={t: {"data.csv": "id,v\n1,CLEANED\n"} for t in (1, 2, 3)},
        turns=3)

    assert all("CLEANED" in b for b in before[1:]), "problem 1 IS fixed"
    flat = [i for turn in dids for i in turn]
    assert len(set(flat)) == 3, (
        "documented residual: an identical rewrite still counts as a modification. If this "
        f"ever drops to 1, delivery became content-addressed -- update the note. got {dids}")


def test_execute_code_still_lets_the_upload_win(monkeypatch, tmp_path):
    """keep_modified is opt-in. execute_code deliberately lets an upload beat a workspace file
    of the same name and reports it in `shadowed`; that contract must not move."""
    from agent_runtime.code_execution import _stage_inputs

    src = tmp_path / "up" / "data.csv"
    src.parent.mkdir(parents=True, exist_ok=True)
    src.write_text("UPLOAD\n", encoding="utf-8")
    work = tmp_path / "w"
    work.mkdir()
    (work / "data.csv").write_text("WORKSPACE\n", encoding="utf-8")

    staged, errs, shadowed = _stage_inputs(work, [{"source": str(src), "dest": "data.csv"}])
    assert staged == ["data.csv"] and shadowed == ["data.csv"] and not errs
    assert (work / "data.csv").read_text() == "UPLOAD\n", "the upload still wins here"


def test_identical_bytes_are_not_recopied(monkeypatch, tmp_path):
    """A no-op copy would bump mtime, making an untouched upload read as modified and be
    re-delivered every turn -- the duplicate this fix exists to remove."""
    from agent_runtime.code_execution import _stage_inputs

    src = tmp_path / "up" / "data.csv"
    src.parent.mkdir(parents=True, exist_ok=True)
    src.write_text("SAME\n", encoding="utf-8")
    work = tmp_path / "w"
    work.mkdir()
    (work / "data.csv").write_text("SAME\n", encoding="utf-8")
    before = (work / "data.csv").stat().st_mtime_ns

    kept = []
    staged, _, shadowed = _stage_inputs(work, [{"source": str(src), "dest": "data.csv"}],
                                        keep_modified=True, kept=kept)
    assert staged == ["data.csv"]
    assert kept == [], "identical bytes are not the peer's edit, so nothing was kept back"
    assert shadowed == [], "and nothing was shadowed either -- no copy happened"
    assert (work / "data.csv").stat().st_mtime_ns == before, "mtime must not move"
