"""Sandboxed `opencode` backend for the code peer (``AGENT_CODE_PEER=opencode``).

Runs the open-source opencode coding agent (https://opencode.ai) headlessly
(``opencode run``) inside a **fresh, hardened container per run** — the same
flag family as ``code_execution.DockerCodeExecutor`` with two deliberate
differences: the container keeps **network access** (opencode must reach the
OpenAI-compatible LLM endpoint, and its bash tool may pip-install), and it gets
a larger memory/pids budget (node runtime + agentic loop).

Unlike the LangChain code peer (generate → ``execute_code`` tool), opencode
iterates internally: it writes code into the throwaway work dir, runs it, reads
errors, and retries until done. Files it leaves in the work dir are persisted
to the agent file store as downloadable artifacts, exactly like ``execute_code``
outputs.

LLM wiring reuses ``build_default_llm``'s env contract (``VLLM_*`` →
``OPENAI_*`` precedence): a per-run ``opencode.json`` in the work dir declares
an ``@ai-sdk/openai-compatible`` provider pointing at the same endpoint. The
API key is passed to the container as an env var and referenced from the config
via ``{env:...}`` — it is never written into the work dir (which is persisted
as artifacts) — and session sharing/autoupdate are disabled so nothing leaves
the deployment.

The sandbox image must have opencode installed — see ``Dockerfile.opencode``
at the repo root; override the name via ``AGENT_OPENCODE_IMAGE``. Under
Docker-out-of-Docker the work dir must live on the host-shared bind mount
(``AGENT_CODE_EXEC_WORK_ROOT``), same as the execute_code sandbox. NOTE: the
LLM base URL is resolved from *inside* the container — a ``localhost`` vLLM
endpoint is unreachable there; set ``AGENT_OPENCODE_BASE_URL`` (e.g. to
``http://host.docker.internal:8000/v1``) for local development.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import tempfile
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from agent_runtime.code_execution import (
    MAX_ARTIFACTS,
    _clip,
    _host_user,
    _resolve_staged_aliases,
    _sig_map,
    _stage_inputs,
    _staged_aliases,
    _work_root,
    unclaimed_name,
)

# Env var read by is_opencode_peer_enabled(); any other value keeps the default
# LangChain code peer.
CODE_PEER_ENV = "AGENT_CODE_PEER"

DEFAULT_OPENCODE_IMAGE = "agent-opencode:latest"
# Agentic write→run→debug loops are much slower than a single execute_code run.
DEFAULT_OPENCODE_TIMEOUT = 600
DEFAULT_OPENCODE_MEMORY = "2g"
DEFAULT_OPENCODE_CPUS = "2.0"
DEFAULT_OPENCODE_PIDS = "1024"

# Provider id inside the generated opencode.json; the model ref is "<id>/<model>".
_PROVIDER_ID = "vllm"
# The key travels via this container env var, referenced from opencode.json as
# {env:...} so it never lands on disk in the (artifact-persisted) work dir.
_API_KEY_ENV = "AGENT_OPENCODE_API_KEY"

_CONFIG_FILENAME = "opencode.json"

# Bound how much retrieved evidence / upstream analysis is inlined into the
# opencode prompt (mirrors the caps used by the LangChain code peer).
MAX_EVIDENCE_CHARS = 6000
MAX_ANALYSIS_CHARS = 1500

_ANSI_RE = re.compile(r"\x1b\[[0-9;?]*[ -/]*[@-~]")


def selects_opencode(value: Optional[str]) -> bool:
    """Does this AGENT_CODE_PEER value name this backend? (pure, so the env default
    and a per-request override share one definition)."""
    return (value or "").strip().lower() == "opencode"


def is_opencode_peer_enabled() -> bool:
    """Whether the deployment default selects opencode for the code peer."""
    return selects_opencode(os.getenv(CODE_PEER_ENV))


def _strip_ansi(text: str) -> str:
    return _ANSI_RE.sub("", text or "")


def resolve_llm_settings() -> Dict[str, Optional[str]]:
    """Resolve model / base URL / API key for opencode.

    Same precedence as ``executor_factory.build_default_llm`` (VLLM_* →
    OPENAI_* → defaults), with ``AGENT_OPENCODE_*`` overrides on top so the
    code peer can run a stronger model than the rest of the agents.
    """
    from agent_runtime.executor_factory import normalize_openai_base_url

    model = (
        os.getenv("AGENT_OPENCODE_MODEL")
        or os.getenv("VLLM_MODEL")
        or os.getenv("OPENAI_CHAT_MODEL")
        or os.getenv("OPENAI_MODEL")
        or "Qwen/Qwen3.5-9B"
    )
    base_url = normalize_openai_base_url(
        os.getenv("AGENT_OPENCODE_BASE_URL") or os.getenv("VLLM_PROXY") or os.getenv("OPENAI_BASE_URL")
    )
    api_key = (
        os.getenv(_API_KEY_ENV)
        or os.getenv("VLLM_API_KEY")
        or os.getenv("OPENAI_KEY")
    )
    return {"model": model, "base_url": base_url, "api_key": api_key}


def model_ref(model: str) -> str:
    """opencode model reference: ``<provider>/<model>`` (model may contain slashes)."""
    return f"{_PROVIDER_ID}/{model}"


def _int_env(name: str, default: int) -> int:
    try:
        return max(1, int(os.getenv(name, str(default))))
    except (TypeError, ValueError):
        return default


def build_opencode_config(model: str, base_url: Optional[str]) -> Dict[str, Any]:
    """The per-run opencode.json: one OpenAI-compatible provider, nothing external."""
    options: Dict[str, Any] = {"apiKey": f"{{env:{_API_KEY_ENV}}}"}
    if base_url:
        options["baseURL"] = base_url
    # Custom providers get no metadata from models.dev, so opencode falls back to
    # a 32000 max_tokens default that many endpoints reject (gpt-4o caps at
    # 16384). Declare explicit limits; `output` is what is sent as max_tokens.
    limit = {
        "context": _int_env("AGENT_OPENCODE_CONTEXT_LIMIT", 128_000),
        "output": _int_env("AGENT_OPENCODE_OUTPUT_LIMIT", 8_192),
    }
    return {
        "$schema": "https://opencode.ai/config.json",
        "provider": {
            _PROVIDER_ID: {
                "npm": "@ai-sdk/openai-compatible",
                "name": "Platform LLM",
                "options": options,
                "models": {model: {"name": model, "limit": limit}},
            }
        },
        "model": model_ref(model),
        # Headless run — never block on an approval prompt.
        "permission": {"edit": "allow", "bash": "allow"},
        # Nothing leaves the deployment: no session-share uploads, no self-update.
        "share": "disabled",
        "autoupdate": False,
    }


def build_docker_argv(work: Path, name: str, model: str, prompt: str) -> List[str]:
    """``docker run`` argv for one opencode run.

    Hardened like the execute_code sandbox (read-only rootfs, cap-drop, no
    privilege escalation, cpu/mem/pid limits, /work the only writable mount) but
    WITH network — opencode is useless without its LLM endpoint. HOME=/work so
    all opencode state lands in the throwaway work dir (dot-dirs are excluded
    from artifact persistence).
    """
    argv = [
        "docker", "run", "--rm", "--init", "--name", name,
        "--cap-drop", "ALL",
        "--security-opt", "no-new-privileges",
        "--read-only",
        "--memory", os.getenv("AGENT_OPENCODE_MEMORY", DEFAULT_OPENCODE_MEMORY),
        "--cpus", os.getenv("AGENT_OPENCODE_CPUS", DEFAULT_OPENCODE_CPUS),
        "--pids-limit", os.getenv("AGENT_OPENCODE_PIDS", DEFAULT_OPENCODE_PIDS),
        "--workdir", "/work",
        "--tmpfs", "/tmp:rw,size=256m,exec",
        "--env", "HOME=/work",
        "--env", f"OPENCODE_CONFIG=/work/{_CONFIG_FILENAME}",
        # THE primary control against an upload steering this CLI, because a filename denylist
        # has to enumerate every path a third-party binary reads while Dockerfile.opencode
        # builds with OPENCODE_VERSION=latest — a release that adds a discovery path would
        # silently reopen the hole with no change here. Both verified against the shipped CLI:
        #   PROJECT_CONFIG gates the cwd config merge AND the ambient-instruction walk. Without
        #     it, /work/opencode.jsonc merges AFTER $OPENCODE_CONFIG and wins, so an upload
        #     could redirect provider.*.options.baseURL while keeping the
        #     `{env:AGENT_OPENCODE_API_KEY}` placeholder the CLI resolves from process.env —
        #     the REAL key, plus every prompt, to an arbitrary endpoint.
        #   CLAUDE_CODE_PROMPT drops CLAUDE.md from the instruction list, which is otherwise
        #     read even here: opencode's instructionFiles are AGENTS.md, CLAUDE.md, CONTEXT.md.
        # The generated config is unaffected — it loads through OPENCODE_CONFIG, not discovery.
        "--env", "OPENCODE_DISABLE_PROJECT_CONFIG=true",
        "--env", "OPENCODE_DISABLE_CLAUDE_CODE_PROMPT=true",
        "--env", "OPENCODE_DISABLE_AUTOUPDATE=true",
        # Name-only form: docker copies the value from the client process env
        # (set by run_opencode), so the key never appears in the argv.
        "--env", _API_KEY_ENV,
        "-v", f"{work}:/work:rw",
    ]
    network = (os.getenv("AGENT_OPENCODE_NETWORK") or "").strip()
    if network:
        argv += ["--network", network]
    user = _host_user()
    if user:
        argv += ["--user", user]
    image = os.getenv("AGENT_OPENCODE_IMAGE", DEFAULT_OPENCODE_IMAGE)
    argv += [image, "opencode", "run", "--model", model_ref(model), prompt]
    return argv


def _timeout_seconds() -> int:
    try:
        return max(30, int(os.getenv("AGENT_OPENCODE_TIMEOUT", str(DEFAULT_OPENCODE_TIMEOUT))))
    except (TypeError, ValueError):
        return DEFAULT_OPENCODE_TIMEOUT


def _persist_artifacts(work: Path, exclude: set, *,
                       defer: Optional[set] = None) -> List[Dict[str, Any]]:
    """Persist files opencode left in *work* to the agent file store.

    Skips dot-prefixed top-level entries (opencode/HOME state: ``.local``,
    ``.config``, ``.cache``, …) and anything in *exclude* (the generated config,
    staged input files).

    `defer` names files that only became artifacts because the run REWROTE an input. They are
    real outputs and must be persisted, but they must not displace the run's own new files:
    `exclude` is applied before the MAX_ARTIFACTS break, so once rewritten inputs started
    counting toward the cap an alphabetically-later genuine output could be silently dropped —
    trading the loss this narrowing was meant to fix for a different one. The sort is stable,
    so ordering inside each group is unchanged.
    """
    try:
        from agent_runtime.file_store import create_output_file_from_path
    except Exception:
        return []

    deferred = {str(x) for x in (defer or ())}
    candidates = [p for p in sorted(work.rglob("*")) if p.is_file()]
    candidates.sort(key=lambda p: str(p.relative_to(work)) in deferred)
    artifacts: List[Dict[str, Any]] = []
    for path in candidates:
        rel = path.relative_to(work)
        if not rel.parts or rel.parts[0].startswith(".") or rel.parts[0] == "__pycache__":
            continue
        if str(rel) in exclude:
            continue
        if len(artifacts) >= MAX_ARTIFACTS:
            break
        try:
            rec = create_output_file_from_path(path, filename=path.name)
            artifacts.append(
                {
                    "file_id": rec["file_id"],
                    "filename": rec["filename"],
                    "download_url": rec.get("download_url"),
                    "size_bytes": rec.get("size_bytes"),
                    # The path RELATIVE to the work dir, which the basename cannot stand in
                    # for once directories nest. A caller that persists across turns needs
                    # to know exactly which files reached the store — the cap above and the
                    # except below both mean "walked, not delivered".
                    "path": str(rel),
                }
            )
        except Exception:
            continue
    return artifacts


def _stage_conversation_files(work: Path, input_file_ids: Optional[List[str]]) -> Dict[str, Any]:
    """Copy conversation-attached files into *work* (same policy/caps as execute_code)."""
    refs = [str(x).strip() for x in (input_file_ids or []) if str(x).strip()]
    if not refs:
        return {"staged": [], "staged_info": [], "errors": [], "skipped": [], "aliases": {},
                "kept": []}
    from agent_runtime.langchain_exec_tools import _build_staging

    staging, staged_info, errors, skipped = _build_staging(refs)
    # keep_modified: the claude peer's work dir persists between turns, so an existing file of
    # this name is the peer's own edit and must not be re-clobbered by the pristine upload.
    # Harmless for opencode, whose work dir is a fresh mkdtemp every run — nothing pre-exists.
    kept: List[str] = []
    staged, stage_errors, _shadowed = _stage_inputs(work, staging,
                                                    keep_modified=True, kept=kept)
    return {
        "staged": staged,
        "staged_info": staged_info,
        "errors": [*errors, *stage_errors],
        "skipped": skipped,
        # {file_id name -> the filename it stands for}; see _resolve_staged_aliases. Additive,
        # so a caller that does not know about it is unaffected.
        "aliases": _staged_aliases(staging),
        # Names whose on-disk copy was kept because it differs from the upload -- the peer's own
        # edit from an earlier turn. The caller must tell the model, or it silently believes it
        # is looking at the attachment.
        "kept": kept,
    }


# Names an upload must never occupy in /work, because the CLI reads them as CONFIGURATION or
# INSTRUCTIONS rather than as data. Verified against the installed opencode core, not assumed:
#
# * `opencode.json` is what OPENCODE_CONFIG points at (build_docker_argv).
# * `opencode.jsonc` is NOT covered by that env var. The loader merges global -> $OPENCODE_CONFIG
#   -> project files discovered as up({targets: ["opencode.jsonc", "opencode.json"]}).toReversed(),
#   so a cwd-level file merges LAST and wins. A staged /work/opencode.jsonc therefore overrides
#   provider.*.options.baseURL on top of the generated config while keeping the
#   `{env:AGENT_OPENCODE_API_KEY}` placeholder docker resolves from the client env — the same
#   credential leak as an uploaded opencode.json, through a different filename.
# * `AGENTS.md` is opencode's ambient-instruction file: InstructionContext walks
#   fs.up({targets: ["AGENTS.md"], start: cwd}), so /work/AGENTS.md matches on the first
#   iteration and is injected as "Instructions from: <path>" — a higher-privilege channel than
#   file content, in a container permitted to edit files, run bash, and reach the network.
#
# * The ambient-instruction list is ["AGENTS.md", "CLAUDE.md", "CONTEXT.md"], and systemPaths()
#   BREAKS on the first name that matches — so neutralising only AGENTS.md just promotes
#   CLAUDE.md to the one that is read. (claude_peer has renamed CLAUDE.md aside for its own CLI
#   since it shipped; opencode reads the same filename.)
# * `.opencode` is collected as a config DIRECTORY by an `fs.exists` test, not an isDir test, so
#   a plain FILE of that name is read as `.opencode/opencode.json` and kills the run before any
#   model call — and OPENCODE_DISABLE_PROJECT_CONFIG does NOT cover it, because the `.opencode`
#   walk from HOME is ungated. Listed for the local-path staging route only: a CONVERSATION
#   upload cannot produce the name at all, because the file store runs every upload through
#   werkzeug secure_filename, which strips the leading dot (".opencode" is stored as "opencode").
#   Kept as depth for the `_resolve_allowed_path` branch, where the dest is an unsanitised
#   `host_path.name`.
#
# Defence in depth, NOT the primary control — see the kill switches in build_docker_argv.
# Matched case-insensitively, like claude_peer._INSTRUCTION_FILENAMES.
_RESERVED_UPLOAD_NAMES = {"opencode.json", "opencode.jsonc",
                          "agents.md", "claude.md", "context.md", ".opencode"}


def reserved_rename_map(staged: Optional[List[str]] = None) -> Dict[str, str]:
    """``{staged name -> on-disk name}`` for uploads that claimed a reserved CLI path.

    PURE, and keyed only on the staged name set, so the two callers that need the answer —
    ``run_opencode``, which does the move, and ``run_opencode_code_peer``, which writes the peer
    brief in a separate pass over the same inputs — agree without sharing state. They agree only
    when both passes see the same names; if a staging copy fails, the brief may still name a file
    that is not there, which is the pre-existing behaviour for any failed upload.

    Each chosen name steps aside from names already staged AND from names already chosen here.
    """
    names = [str(n) for n in (staged or [])]
    taken = set(names)
    out: Dict[str, str] = {}
    for name in names:
        if name.lower() not in _RESERVED_UPLOAD_NAMES:
            continue
        target = unclaimed_name(f"uploaded_{name}", taken)
        taken.add(target)
        out[name] = target
    return out


def neutralize_reserved_uploads(work: Path,
                                staged: Optional[List[str]] = None) -> Tuple[Dict[str, str], set]:
    """Move uploads that claimed a reserved CLI path out of the way. See _RESERVED_UPLOAD_NAMES.

    Renamed rather than deleted, matching ``claude_peer.neutralize_instruction_files``: the user
    uploaded it, so it stays readable as data and downloadable as an artifact, under a name the
    CLI does not read as config or instructions.

    Returns ``(applied {from -> to}, names that could not be moved)``. A name that could not be
    moved is reported so the caller can stop treating it as a staged file at all: the generated
    config is written over it regardless, so what sits there is no longer the upload.
    """
    renames = reserved_rename_map(staged)
    applied: Dict[str, str] = {}
    failed: set = set()
    for name, target in renames.items():
        src = work / name
        if not src.is_file():
            continue
        try:
            src.rename(work / target)
        except OSError:
            failed.add(name)
            continue
        applied[name] = target
    return applied, failed


def run_opencode(
    prompt: str,
    *,
    input_file_ids: Optional[List[str]] = None,
    timeout: Optional[int] = None,
) -> Dict[str, Any]:
    """One sandboxed ``opencode run``; returns a JSON-serializable result dict."""
    settings = resolve_llm_settings()
    if not settings["api_key"]:
        return {
            "ok": False, "exit_code": None, "answer": "", "stderr": "", "timed_out": False,
            "error": f"{_API_KEY_ENV} (or VLLM_API_KEY / OPENAI_KEY) is required for the opencode code peer.",
            "artifacts": [], "backend": "opencode-docker", "model": None,
        }
    model = settings["model"]
    timeout = int(timeout or _timeout_seconds())
    try:
        work = Path(tempfile.mkdtemp(prefix="agentoc_", dir=_work_root()))
    except OSError as exc:
        return {
            "ok": False, "exit_code": None, "answer": "", "stderr": "", "timed_out": False,
            "error": (f"opencode work dir unavailable: {exc}. "
                      "Check AGENT_CODE_EXEC_WORK_ROOT and its bind mount in the deployment."),
            "artifacts": [], "backend": "opencode-docker", "model": model_ref(model),
        }
    try:
        # Staging runs BEFORE the config is written, and the config is written LAST, so an
        # upload can never end up at the path the CLI loads. Previously the config went first
        # and _stage_inputs overwrote it — an upload named opencode.json simply became the
        # provider config.
        staging = _stage_conversation_files(work, input_file_ids)
        renamed, unmovable = neutralize_reserved_uploads(work, staging["staged"])
        (work / _CONFIG_FILENAME).write_text(
            json.dumps(build_opencode_config(model, settings["base_url"]), indent=2),
            encoding="utf-8",
        )
        # Fingerprint the uploads as staged, so a file the peer REWROTE can be told from one it
        # only read. Excluding staged names unconditionally discarded the peer's own result
        # whenever it edited an input in place -- and editing a file in place is the entire job
        # of a coding agent. Worse here than in execute_code: this work dir is a mkdtemp deleted
        # in the finally below, so there is no durable workspace to fall back on. See the same
        # `pristine` reasoning in code_execution._execute.
        staged_sigs = _sig_map(work, [renamed.get(n, n) for n in staging["staged"]
                                      if n not in unmovable])
        try:
            os.chmod(work, 0o777)  # non-root container user must write here
        except OSError:
            pass
        name = f"agentoc_{uuid.uuid4().hex[:12]}"
        argv = build_docker_argv(work, name, model, prompt)
        env = {**os.environ, _API_KEY_ENV: settings["api_key"]}
        exit_code: Optional[int] = None
        stdout, stderr, timed_out, error = "", "", False, None
        try:
            proc = subprocess.run(argv, capture_output=True, text=True, env=env, timeout=timeout + 5)
            exit_code, stdout, stderr = proc.returncode, proc.stdout, proc.stderr
        except subprocess.TimeoutExpired as exc:
            subprocess.run(["docker", "kill", name], capture_output=True)
            stdout, stderr = str(exc.stdout or ""), str(exc.stderr or "")
            timed_out, error = True, f"opencode run timed out after {timeout}s"
        except FileNotFoundError:
            error = "docker executable not found"
        except Exception as exc:  # pragma: no cover - defensive
            error = f"{type(exc).__name__}: {exc}"

        answer = _clip(_strip_ansi(stdout).strip())
        _now = _sig_map(work, staged_sigs)
        pristine = {rel for rel, sig in staged_sigs.items()
                    if _now.get(rel) == sig}                    # read but not written
        # A peer that writes through the opaque file_id name must not deliver the result under
        # it: layers_for_artifacts below sniffs by extension, so a .geojson written that way
        # would never become a map layer.
        # Primaries go through the same renaming: an alias whose filename was opencode.json
        # would otherwise have its bytes copied onto the generated config after the run.
        alias_names = _resolve_staged_aliases(
            work, {a: renamed.get(prim, prim)
                   for a, prim in (staging.get("aliases") or {}).items()},
            staged_sigs, pristine)
        artifacts = _persist_artifacts(work, {_CONFIG_FILENAME, *pristine, *alias_names},
                                       defer={r for r in staged_sigs
                                              if r not in pristine and r not in alias_names})
        # No tools means no add_map_layer means nothing checked what this wrote. See
        # layer_qa.inspect_artifacts.
        from agent_runtime.layer_qa import inspect_artifacts

        qa = inspect_artifacts(str(work), [a.get("filename") for a in artifacts])
        # A peer with no tools still wrote geodata. Turn it into layer descriptors here,
        # while the work dir still exists — the wrapper emits them from the request context.
        from agent_runtime.map_layers import layers_for_artifacts

        map_layers = layers_for_artifacts(work, artifacts)
        result: Dict[str, Any] = {
            "ok": error is None and not timed_out and exit_code == 0,
            "exit_code": exit_code,
            "answer": answer,
            "stderr": _clip(_strip_ansi(stderr).strip()),
            "timed_out": timed_out,
            "error": error,
            "artifacts": artifacts,
            "backend": "opencode-docker",
            "model": model_ref(model),
        }
        if qa:
            result["output_warnings"] = qa
        if map_layers:
            result["map_layers"] = map_layers
            # The supervisor's delivery check reads this off the execution record; without
            # it a turn that DID put something on the map still counts as undelivered.
            result["on_map"] = True
        if staging["staged_info"]:
            # Through the same rename, or the turn record tells every later reader the upload is
            # at opencode.json -- which by then holds the generated config, base URL and all.
            result["input_files"] = [
                {**info,
                 "available_as": [renamed.get(n, n) for n in (info.get("available_as") or [])
                                  if n not in unmovable]}
                for info in staging["staged_info"]]
        if staging["errors"]:
            result["input_file_errors"] = staging["errors"]
        if staging["skipped"]:
            result["input_files_skipped"] = staging["skipped"]
        return result
    finally:
        shutil.rmtree(work, ignore_errors=True)


def _build_peer_prompt(
    query: str,
    evidence: Optional[List[Any]],
    analysis_results: Any,
    staged_names: Optional[List[str]] = None,
) -> str:
    """The headless task brief, mirroring the context the LangChain code peer gets."""
    parts = [
        "You are the code peer of a multi-agent geospatial analysis platform, running "
        "headless in a sandboxed container. The working directory is yours: write code, "
        "RUN it, and debug until it works (install Python packages with "
        "`pip install --user <pkg>` if needed). Save any output files (plots, tables, "
        "data) to the working directory. Finish with a concise summary of what you did, "
        "the key results, and the files you wrote.",
        f"Task:\n{query}",
    ]
    if evidence:
        try:
            from agent_runtime.supervisor.evidence_subgraph import _format_documents

            parts.append(f"Evidence:\n{_format_documents(evidence)[:MAX_EVIDENCE_CHARS]}")
        except Exception:
            pass
    if analysis_results:
        parts.append(
            "Analysis results:\n"
            + json.dumps(analysis_results, ensure_ascii=True, default=str)[:MAX_ANALYSIS_CHARS]
        )
    if staged_names:
        parts.append("Input files already in the working directory: " + ", ".join(staged_names))
    return "\n\n".join(parts)


def run_opencode_code_peer(
    query: str,
    evidence: Optional[List[Any]] = None,
    state: Optional[Dict[str, Any]] = None,
    input_file_ids: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Code-peer adapter: returns the same flat shape as ``default_code_fn``
    (``answer`` + compact ``tool_calls``/``tool_results``) so synthesis and the
    trace pipeline are agnostic to which backend produced the code result."""
    from agent_runtime.map_layers import build_map_layers
    from agent_runtime.streaming_trace import emit_trace_event

    # Resolve refs to the names the files will be staged under (file_id AND
    # original filename) so the prompt tells opencode what is actually on disk.
    staged_names: List[str] = []
    refs = [str(x).strip() for x in (input_file_ids or []) if str(x).strip()]
    if refs:
        try:
            from agent_runtime.langchain_exec_tools import _build_staging

            _, staged_info, _, _ = _build_staging(refs)
            for info in staged_info:
                staged_names.extend(info.get("available_as") or [])
            # Through the same rename run_opencode applies, or the brief would point the peer at
            # opencode.json — which by then holds the GENERATED provider config, not the upload.
            # Reading it back would put the base URL in the peer's answer. config_rename_map is
            # pure and sees the same name set there, so both passes agree.
            _renames = reserved_rename_map(staged_names)
            staged_names = [_renames.get(n, n) for n in staged_names]
        except Exception:
            staged_names = list(refs)
    prompt = _build_peer_prompt(
        query,
        evidence,
        (state or {}).get("analysis_results"),
        staged_names=staged_names or None,
    )
    call_args = {"model": resolve_llm_settings()["model"], "prompt_chars": len(prompt)}
    emit_trace_event("tool_call", {"name": "opencode_run", "args": call_args}, node="code")
    result = run_opencode(prompt, input_file_ids=input_file_ids)
    emit_trace_event(
        "tool_result",
        {
            "name": "opencode_run",
            "content": {k: result.get(k) for k in
                        ("ok", "exit_code", "timed_out", "error", "artifacts", "backend", "model")},
        },
        node="code",
    )
    # The peer has no tools, so nothing emitted a map_layer on its behalf. This wrapper
    # runs in the request's trace context — the same place a tool callback would — so the
    # descriptors go out here, through the same build_map_layers boundary every tool's
    # layer crosses, and get the same validation.
    for layer in build_map_layers("opencode_run", result):
        emit_trace_event("map_layer", layer, node="code")

    answer = result.get("answer") or ""
    warnings = result.get("output_warnings") or []
    if warnings:
        lines = [f"- {w['file']}: {'; '.join(w['problems'])}" for w in warnings]
        answer = "\n\n".join(x for x in (answer, "Checks on the files this run produced "
                                                   "found problems — say so rather than "
                                                   "presenting them as results:\n"
                                                   + "\n".join(lines)) if x)
    if not result.get("ok"):
        failure = result.get("error") or f"opencode exited with code {result.get('exit_code')}"
        detail = str(result.get("stderr") or "")[-2000:]
        answer = "\n\n".join(
            x for x in (f"opencode code peer failed: {failure}", detail, answer) if x
        )
    # See the same block in claude_peer.run_claude_code_peer: the CLI peers bypass
    # _apply_execution_honesty, so without this the execution-provenance audit rule treats a
    # real sandboxed run as "nothing ran".
    return {
        "answer": answer,
        "executed": bool(result.get("ok")),
        **({} if result.get("ok") else
           {"execution_error": str(result.get("error")
                                   or f"opencode exited with code {result.get('exit_code')}")}),
        "tool_calls": [{"name": "opencode_run", "args": call_args}],
        "tool_results": [{"name": "opencode_run", "content": result}],
    }


__all__ = [
    "CODE_PEER_ENV",
    "is_opencode_peer_enabled",
    "resolve_llm_settings",
    "build_opencode_config",
    "build_docker_argv",
    "model_ref",
    "run_opencode",
    "run_opencode_code_peer",
]
